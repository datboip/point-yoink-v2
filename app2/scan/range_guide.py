# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frame distance statistics in calibrated depth units, before the user's range cut."""
import numpy as np
from functools import lru_cache


def foreground_guess(depth):
    """Without a measured bed: the depth connected to the middle of the picture and near its distance, as the "part" for
    the distance bars (2026-10-25: the bars showed the table and the room behind the part, "what is scanned instead of
    the item"). A bool mask, or None when the middle shows no depth."""
    import cv2
    h,w=depth.shape;z=depth.astype(float)*.1
    ry,rx=h/np.sqrt(12),w/np.sqrt(12);v,u=np.indices(depth.shape)
    window=(depth>0)&(np.abs(u-(w-1)/2)<=rx)&(np.abs(v-(h-1)/2)<=ry)
    zw=z[window]
    if len(zw)<50:return None
    # the nearest sizeable depth cluster in the middle: the object stands in front of whatever is behind it, so the
    # median of the window (often the wall or the table) is not it
    counts,edges=np.histogram(zw,bins=np.arange(np.floor(zw.min()/10)*10,zw.max()+10,10.))
    big=np.flatnonzero(counts>=max(50,.1*len(zw)))
    c=float(np.median(zw[(zw>=edges[big[0]])&(zw<edges[big[0]+1])])) if len(big) else float(np.median(zw))
    near=(depth>0)&(np.abs(z-c)<=max(40.,.25*c))
    count,labels=cv2.connectedComponents(near.astype(np.uint8),connectivity=8)
    middle=near&window
    ids=np.unique(labels[middle]);ids=ids[ids>0]
    return np.isin(labels,ids) if len(ids) else None


def center_distance(depth):
    h,w=depth.shape
    # The central one-standard-deviation region of the image's pixel coordinates.
    ry,rx=h/np.sqrt(12),w/np.sqrt(12)
    v,u=np.indices(depth.shape)
    keep=(depth>0)&(np.abs(u-(w-1)/2)<=rx)&(np.abs(v-(h-1)/2)<=ry)
    return float(np.median(depth[keep]))*.1 if keep.any() else None


def describe(depth,working_range):
    z=depth[depth>0].astype(float)*.1
    lo,hi=map(float,working_range)
    if not np.isfinite([lo,hi]).all() or hi<=lo:return None
    if len(z):
        low=min(lo,float(z.min()));high=max(hi,float(z.max()))
        # Bins in whole depth units (0.1 mm), finer than any bar column: each bin holds the same number of possible
        # depth values, so the bars never show a comb from bins catching two values here and three there.
        k=max(1,int(round((hi-lo)/(2*depth.shape[1])/.1)));a0=np.floor(low*10)-.5
        edges=(a0+k*np.arange(int(np.ceil((high*10-a0)/k))+1))/10
        counts,edges=np.histogram(z,bins=edges)
    else:edges=np.array([lo,hi]);counts=np.zeros(1,int)
    center=center_distance(depth)
    state='no center depth' if center is None else 'too close' if center<lo else 'too far' if center>hi else 'in range'
    return dict(edges=edges.tolist(),counts=counts.tolist(),working_range=[lo,hi],center_mm=center,state=state)


# Fixed distance bands per mode from its measured working span: the middle 40% is best, 30% either side OK,
# outside is too close / too far. Shared by the distance bar and the colored depth dots.
UNSEEN_RGB=(135,145,164)
BAND_RGB={'best':(79,196,111),'ok':(217,192,74),'off':(224,88,75)}
DEFAULT_PALETTE=dict(BAND_RGB)
SAFE_PALETTE={'best':(0,114,178),'ok':(230,159,0),'off':(204,121,167)}
def set_palette(safe):BAND_RGB.update(SAFE_PALETTE if safe else DEFAULT_PALETTE)
BAND_LEVELS={'strong depth':2,'good depth':2,'usable depth':1}
def band_edges(lo,hi):
    w=hi-lo;return lo,lo+.3*w,hi-.3*w,hi


# These are display colors for measured depth support, not a precision scale.
COLORS={'strong depth':'#55c27c','good depth':'#69b5e6','usable depth':'#c4b45d','sparse depth':'#e58c58',
        'unmeasured':'#8791a4','no part depth':'#8791a4','too close':'#e58c58','too far':'#e58c58'}
LABELS={'strong depth':'Strong','good depth':'Good','usable depth':'Usable','sparse depth':'Sparse','unmeasured':'Unmeasured','no part depth':'No part depth'}


@lru_cache(maxsize=None)
def measured_profile(mode):
    import json
    from pathlib import Path
    return json.loads(Path(__file__).with_name('camera_profiles.json').read_text())['miraco']['modes'][mode]


def from_scene(depth,mode,scene,profile=None):
    """Distance observations and current foreground fill, with no extrapolated optical limit."""
    profile=profile or measured_profile(mode)
    if scene['source']=='part':hist_depth=np.where(scene['mask'],depth,0)
    else:
        guess=foreground_guess(depth);hist_depth=np.where(guess,depth,0) if guess is not None else depth
    g=describe(hist_depth,profile['observed_distance_mm'])
    g['mode']=mode;g['zones']=distance_zones(profile);g['source']='saved depth fill and measured foreground'
    center=scene['depth_mm'] if scene['rect'] is not None else center_distance(depth)
    if scene['source']!='part':state='unmeasured'
    elif scene['rect'] is None:state='no part depth'
    else:
        a,b,c=profile['fill_quartiles'];fill=scene['fill']
        state='strong depth' if fill>=c else 'good depth' if fill>=b else 'usable depth' if fill>=a else 'sparse depth'
    measured=bool(center is not None and any(z['lo']<=center<=z['hi'] for z in profile['zones']))
    g['profile_kind']=profile.get('source_kind','saved depth observations')
    g['bands_date']=profile.get('date') if profile.get('complete') else None
    g.update(center_mm=center,state=state,current_state=state,reference_distance_observed=measured,observed_distance=bool(measured or scene['source']=='part' and scene['rect'] is not None),center_source='part' if scene['source']=='part' else 'scene')
    g['label']=LABELS.get(state,state.capitalize())+(' depth' if state.endswith('depth') and state!='no part depth' else '')
    if center is not None and not g['observed_distance']:g['label']+='; distance unmeasured'
    return g


class SteadyGuide:
    """Share one held quality state between the distance bar and the framing box."""
    def __init__(self,window):self.window=float(window);self.history=[];self.state=None;self.last_center=None
    def update(self,guide,t):
        from collections import Counter
        self.history.append((t,guide['state']));self.history=[x for x in self.history if t-x[0]<=self.window]
        counts=Counter(s for _,s in self.history);best=max(counts.values());choices=[s for s,n in counts.items() if n==best]
        self.state=self.state if self.state in choices else choices[0]
        g=dict(guide);g['state']=self.state
        g['zones']=[dict(z,state=self.state) if z.get('live') else dict(z) for z in guide.get('zones',[])]
        if guide['center_mm'] is not None:self.last_center=guide['center_mm']
        else:g['center_mm']=self.last_center;g['center_source']='last part'
        g['label']=LABELS.get(self.state,self.state.capitalize())+(' depth' if self.state.endswith('depth') and self.state!='no part depth' else '')
        if g['center_mm'] is not None and not g['observed_distance']:g['label']+='; distance unmeasured'
        return g


def setup_check(scene,guide,dots=None,steady=False):
    """One practical fix for each measured issue; missing inputs stay unknown."""
    problems=[]
    if scene['source']!='part':problems.append('Learn bed, or show the bed, so the app can find the part.')
    if scene['rect'] is None:problems.append('No part depth yet. Check the view and IR gain.')
    else:
        if not scene['centered']:problems.append('Move the part onto the center cross.')
        if scene.get('missing_depth'):problems.append('Some part surfaces show no depth. Check dark or shiny areas.')
        if scene['clipped']:problems.append('Part goes past what depth can see. Move back or change the angle.')
        if guide['current_state']=='sparse depth':problems.append('Depth on the part is thin. Try more IR gain or a different angle.')
        if scene['background_pixels']>scene['kept_part_pixels']:problems.append('Range takes in a lot besides the part. Press Use part Range.')
        e=scene['exposure']
        if e is not None:
            # Endpoints are actual sensor clipping. A majority cannot represent a
            # normally exposed part; no scanner-specific brightness guess is used.
            if e['dark_share']>.5:problems.append('The part looks too dark to the depth camera. Try more IR gain.')
            if e['clipped_share']>.5:problems.append('The part looks too bright to the depth camera. Try less IR gain.')
    if dots is not None:
        from scanner import bed
        if dots<bed.HIT_MIN:problems.append('Show more bed dots so the app can check the camera.')
    hint='Scene steady. For handheld scanning, move slowly after Start.' if steady else ''
    return dict(problems=problems,hint=hint,points=scene['points'],fill=scene['fill'],exposure=scene['exposure'],dots=dots,
                source=scene['source'],observed_distance=guide['observed_distance'])


def distance_zones(profile):
    """OK, Best, OK. A sweep's own bands win; then Best from the measured zones; the middle 40% of the observed span last.

    Observed span is not a certified accuracy or hardware limit, and gaps in a sweep stay unmeasured.
    """
    if profile.get('distance_bands'):return [dict(x) for x in profile['distance_bands']]
    lo,hi=profile['observed_distance_mm'];best=best_from_zones(profile)
    if best is None:a,b,c,e=band_edges(lo,hi)
    else:b,c=best;w=c-b;a,e=max(lo,b-w),min(hi,c+w)   # OK reaches one Best width past each edge, inside what was seen
    return [dict(lo=a,hi=b,state='usable depth'),dict(lo=b,hi=c,state='strong depth'),dict(lo=c,hi=e,state='usable depth')]


def best_from_zones(profile):
    """(lo, hi) of Best: the measured zone with the most part depth, grown over neighbors that still give usable depth.

    Zones with fewer frames than the median zone say little, so they neither stop nor grow Best; a gap of more than
    one empty zone ends it. None when nothing was measured. A few far frames no longer drag Best away from the peak."""
    zones=sorted((z for z in profile.get('zones') or [] if z.get('frames')),key=lambda z:z['lo'])
    if not zones:return None
    enough=float(np.median([z['frames'] for z in zones]));usable=profile['fill_quartiles'][0]
    width=float(np.median([z['hi']-z['lo'] for z in zones]))
    kept=lambda z:z['frames']>=enough
    peak=max((i for i,z in enumerate(zones) if kept(z)),key=lambda i:zones[i]['median_fill'])
    lo,hi=zones[peak]['lo'],zones[peak]['hi']
    for step in (1,-1):
        edge=hi if step>0 else lo;i=peak+step
        while 0<=i<len(zones):
            z=zones[i];gap=z['lo']-edge if step>0 else edge-z['hi']
            if gap>1.5*width:break                          # more than one empty zone: unmeasured, Best stops
            if kept(z):
                if z['median_fill']<usable:break
                if step>0:hi=z['hi']
                else:lo=z['lo']
            edge=z['hi'] if step>0 else z['lo'];i+=step
    return lo,hi


def distance_levels(z,profile):
    """0 off, 1 OK, 2 best; one shared policy for dots, histogram and model."""
    z=np.asarray(z);out=np.zeros(z.shape,np.uint8)
    for band in distance_zones(profile):
        value=BAND_LEVELS.get(band['state'],0)
        mask=(z>=band['lo'])&(z<=band['hi']);out[mask]=np.maximum(out[mask],value)
    return out


def distance_action(guide):
    z=guide.get('center_mm');zones=guide.get('zones',[])
    best=[x for x in zones if x.get('state') in ('strong depth','good depth') and not x.get('live')]
    if z is None:return 'Show the part'
    if not best:return 'No best band measured'
    if any(b['lo']<=z<=b['hi'] for b in best):return 'Good distance'
    target=min([edge for b in best for edge in (b['lo'],b['hi'])],key=lambda edge:abs(z-edge));delta=target-z
    return 'Move about %g cm %s' % (max(1,round(abs(delta)/10)),'back' if delta>0 else 'closer')
