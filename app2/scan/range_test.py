# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured mat sweeps. Saved frames use the same analysis as live offers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import numpy as np
from app2.scan.assist import RANGE_STEP_MM


def analyze(depth,ir,jpeg,bundles,model,mode):
    from scanner import bed
    from app2.scan.mode_detection import compare
    rows=compare(depth,ir,bundles,model,jpeg);selected=bundles[mode]
    hit=bed.find_bed(depth,ir,selected['intrinsics'],model,prior=None) if model is not None else None
    if hit is None:return dict(accepted=False,reason='Point the scanner at the empty bed',fits=rows)
    v,u=np.indices(depth.shape);z=depth.astype(float)*.1;fx,fy,cx,cy=selected['intrinsics']
    P=np.stack([(u-cx)*z/fx,(v-cy)*z/fy,z],axis=-1)
    n=np.asarray(hit.n);off=hit.off
    # Project the learned bed outline onto its own plane to measure missing depth,
    # rather than letting the currently valid-pixel mask define its denominator.
    rays=np.stack([(u-cx)/fx,(v-cy)/fy,np.ones_like(z)],axis=-1);den=rays@n
    plane_z=np.divide(-off,den,out=np.zeros_like(z),where=np.abs(den)>np.finfo(float).eps)
    plane_points=rays*plane_z[:,:,None]
    from app2.scan.bed_adapter import _shape
    from app2.scan.bedcut import BedCut
    inside=(plane_z>0)&(BedCut._sd(plane_points.reshape(-1,3),n,hit.centre,_shape(model,hit.x_axis)).reshape(z.shape)<=0)
    valid=inside&(depth>0)
    if valid.sum()<bed.HIT_MIN:return dict(accepted=False,reason='Slow down and show more of the bed',fits=rows)
    residual=P[valid]@n+off;median=np.median(residual);sigma=1.482602218505602*np.median(np.abs(residual-median))
    # The mat must be empty: supported depth away from the fitted plane is not
    # allowed to improve a sweep's quality estimate.
    flat=np.abs(residual-median)<=3*max(sigma,.1/np.sqrt(12))
    if flat.sum()<len(flat)/2:return dict(accepted=False,reason='Take the part off the bed first',fits=rows)
    row=rows[mode];color=row.get('color') or {}
    votes=[key for key,value in rows.items() if value.get('bed_fit') or (value.get('color') or {}).get('accepted')]
    return dict(accepted=True,distance_mm=float(np.median(z[valid])),fill=float(valid.sum()/max(inside.sum(),1)),
                plane_rms_mm=float(np.sqrt(np.mean(residual[flat]**2))),dots=int(row['inliers']),dot_rms_mm=row.get('rms_mm'),
                overlay_error_px=color.get('median_px'),camera_match=votes[0] if len(votes)==1 else None,fits=rows)


def result(samples,mode,calibration_sha256):
    accepted=[x for x in samples if x.get('accepted')]
    groups={}
    for row in accepted:groups.setdefault(int(round(row['distance_mm']/RANGE_STEP_MM)),[]).append(row)
    bins=[]
    for key,rows in sorted(groups.items()):
        item=dict(distance_mm=key*RANGE_STEP_MM,frames=len(rows))
        for field in ('fill','plane_rms_mm','dots','dot_rms_mm','overlay_error_px'):
            values=[x[field] for x in rows if x.get(field) is not None];item[field]=float(np.median(values)) if values else None
        item['camera_matches']={m:sum(x.get('camera_match')==m for x in rows) for m in ('near','far')};bins.append(item)
    if not bins:return None
    fills=np.array([x['fill'] for x in bins]);noise=np.array([x['plane_rms_mm'] for x in bins]);fcut=float(np.median(fills));ncut=float(np.median(noise))
    # Best bins have at least median fill and at most median flatness noise.
    # Use the contiguous best run with most supporting frames. No extrapolated
    # optical limit; OK covers only observed bins either side, with gaps omitted.
    good=np.array([x['fill']>=fcut and x['plane_rms_mm']<=ncut for x in bins]);runs=[];run=[]
    for i,item in enumerate(bins):
        if good[i] and (not run or item['distance_mm']-bins[run[-1]]['distance_mm']<=RANGE_STEP_MM):run.append(i)
        else:
            if run:runs.append(run);run=[]
            if good[i]:run=[i]
    if run:runs.append(run)
    best=max(runs,key=lambda ids:sum(bins[i]['frames'] for i in ids)) if runs else []
    from scanner.bed import ACCEPT_HITS
    supported=[i for i,b in enumerate(bins) if b['frames']>=ACCEPT_HITS]
    bands=[]
    for i,item in enumerate(bins):
        if supported and i not in supported:continue
        state='strong depth' if i in best else 'usable depth';a=item['distance_mm']-RANGE_STEP_MM/2;b=item['distance_mm']+RANGE_STEP_MM/2
        if bands and bands[-1]['state']==state and bands[-1]['hi']==a:bands[-1]['hi']=b
        else:bands.append(dict(lo=a,hi=b,state=state))
    # No sweep is complete from a single held distance. Keep observations visible,
    # but never replace a profile without at least the camera vote minimum bins.
    from scanner.bed import ACCEPT_HITS
    complete=len(supported)>=ACCEPT_HITS and covers_span(mode,[0,len(supported)*RANGE_STEP_MM])
    return dict(version=1,mode=mode,date=datetime.now(timezone.utc).isoformat(),calibration_sha256=calibration_sha256,
                complete=complete,source_kind='empty learned mat Range test',observed_distance_mm=[bands[0]['lo'],bands[-1]['hi']],
                distance_bands=bands,fill_quartiles=np.percentile(fills,[25,50,75]).tolist(),zones=bands,bins=bins,
                rule='Range step bins; median observations; best contiguous run has fill >= median and plane RMS <= median, ranked by supporting frames; gaps unmeasured; bed acceptance count of bins and frames per bin; half the shipped span; explicit Apply',
                evidence=samples,camera_pair_confirmed=any(b['camera_matches'][mode]>=ACCEPT_HITS and b['camera_matches']['far' if mode=='near' else 'near']==0 for b in bins))


class Sweep:
    def __init__(self,mode,bundles,model):
        self.mode=mode;self.bundles=bundles;self.model=model;self.samples=[];self.future=None;self.skipped=0
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='range-test');self.hint='Move the scanner slowly from close to far'
    def poll(self):
        if self.future is not None and self.future.done():
            try:row=self.future.result()
            except Exception:row=dict(accepted=False,reason='Could not measure this frame')
            self.samples.append(row);self.hint=progress(self.samples,self.mode)['action'] if row.get('accepted') else row['reason'];self.future=None
    def offer(self,depth,ir,jpeg):
        self.poll()
        if self.future is not None:self.skipped+=1;return   # the last hint stands: swapping it each frame flashed
        self.future=self.pool.submit(analyze,depth.copy(),ir.copy(),jpeg,self.bundles,self.model,self.mode)
    def close(self):
        self.pool.shutdown(wait=True);self.poll()
        return result(self.samples,self.mode,self.bundles[self.mode]['source']['calibration_sha256'])


def covers_span(mode,span):
    """A test replaces the bands only when it swept at least half the mode's shipped working span; a few held
    distances once made the bar call a good spot too far."""
    from app2.scan.range_guide import measured_profile
    lo,hi=measured_profile(mode)['observed_distance_mm']
    return (span[1]-span[0])>=.5*(hi-lo)


def supported_bins(profile):
    from scanner.bed import ACCEPT_HITS
    rows=[row for row in profile.get('bins',[]) if row.get('frames',0)>=ACCEPT_HITS]
    return len(rows)>=ACCEPT_HITS and covers_span(profile.get('mode','near'),[0,len(rows)*RANGE_STEP_MM])


def progress(samples,mode):
    from app2.scan.range_guide import measured_profile
    from scanner.bed import ACCEPT_HITS
    lo,hi=measured_profile(mode)['observed_distance_mm'];target=(hi-lo)/2
    rows=[r for r in samples if r.get('accepted')];counts={}
    for row in rows:
        key=round(row['distance_mm']/RANGE_STEP_MM);counts[key]=counts.get(key,0)+1
    covered=[key*RANGE_STEP_MM for key,n in counts.items() if n>=ACCEPT_HITS]
    span=len(covered)*RANGE_STEP_MM
    distance=rows[-1]['distance_mm'] if rows else None
    if distance is None:action='Point the scanner at the empty bed'
    elif counts[round(distance/RANGE_STEP_MM)]<ACCEPT_HITS:action='Hold still at this distance'
    elif span>=target:action='Done: covered %g to %g mm. Stop to review.' % (min(covered),max(covered))
    else:
        direction='farther' if distance<=(lo+hi)/2 else 'closer'
        action='Move %g cm %s, then hold still' % (RANGE_STEP_MM/10,direction)
    return dict(distance_mm=distance,covered_mm=span,target_mm=target,fraction=min(1,span/target),action=action,frames=len(rows),
                lo_mm=lo,hi_mm=hi,step_mm=RANGE_STEP_MM,bins={key*RANGE_STEP_MM:min(1,n/ACCEPT_HITS) for key,n in counts.items()})
