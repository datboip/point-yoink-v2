# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured foreground framing. This guide never removes points from a scan."""
from collections import deque
import numpy as np
import cv2
from app2.scan.bedcut import LOW_MM, RIM_MM, BedCut
from track import REGISTRATION_MIN_POINTS


def depth_noise(depth, mask):
    z=depth.astype(float)*.1
    ok=mask[:,2:]&mask[:,1:-1]&mask[:,:-2]
    values=(z[:,2:]-2*z[:,1:-1]+z[:,:-2])[ok]
    if len(values)>40000:values=values[::max(1,len(values)//40000)]   # a spread from 40k samples is as good, at a tenth of the time
    sigma=float(np.median(np.abs(values-np.median(values)))*1.482602218505602/np.sqrt(6)) if len(values) else 0.
    return max(.1/np.sqrt(12),sigma)


def bounds(mask,trim=False):
    v,u=np.nonzero(mask)
    if len(u)<REGISTRATION_MIN_POINTS:return None
    if trim:return np.array([np.percentile(u,1),np.percentile(v,1),np.percentile(u,99),np.percentile(v,99)],float)
    return np.array([u.min(),v.min(),u.max(),v.max()],float)


def analyze(depth,cal,geo=None,cut=None,dots=None,ir=None):
    """An above-bed component anchored inside the rim, including connected overhangs.

    The learned outline anchors the component; it is not a crop of the finished part.
    Unconnected background and the bed surface cannot enlarge the part box. Without a
    measured bed we explicitly return depth coverage instead of claiming a part.
    """
    valid=depth>0;v,u=np.nonzero(valid);z=depth[v,u].astype(float)*.1
    mask=valid.copy();source='coverage';sigma=depth_noise(depth,valid)
    if geo is not None:
        n,off,center,rim=geo; n=np.asarray(n,float)
        P=np.column_stack([(u-cal[2])*z/cal[0],(v-cal[3])*z/cal[1],z])
        above=P@n+float(off)>LOW_MM+3*sigma
        seed=above.copy()
        if center is not None and rim is not None:
            seed &= BedCut._sd(P,n,np.asarray(center),rim)<=RIM_MM
        mask[:]=False;mask[v[above],u[above]]=True
        if dots is not None:
            for x,y,radius in np.asarray(dots).reshape(-1,3):
                cv2.circle(mask.view(np.uint8),(round(x),round(y)),int(np.ceil(radius)),0,-1)
        # All well-supported components touching the known bed are retained. The
        # registration minimum is reused as the minimum support for a useful box.
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        anchors=np.bincount(labels[v[seed],u[seed]],minlength=count);anchors[0]=0
        ids=np.flatnonzero((anchors>0)&(stats[:,cv2.CC_STAT_AREA]>=REGISTRATION_MIN_POINTS))
        mask=np.isin(labels,ids) if len(ids) else np.zeros(depth.shape,bool)
        source='part'
    rect=bounds(mask,trim=source!='part');points=int(mask.sum());span=None;center_mm=None;fill=0.;clipped=False;centred=False;uncertainty=None;center_px=None;observed_span=None;hull=None;diagonal=None
    if rect is not None:
        v,u=np.nonzero(mask);z=depth[mask].astype(float)*.1;sigma=depth_noise(depth,mask)
        span=[max(.1,float(z.min())-3*sigma),float(z.max())+3*sigma] if source=='part' else [max(.1,float(np.percentile(z,1))-3*sigma),float(np.percentile(z,99))+3*sigma]
        center_mm=float(np.median(z));diagonal=float(np.linalg.norm([np.ptp(u)*center_mm/cal[0],np.ptp(v)*center_mm/cal[1],np.ptp(z)]));center_px=[float(u.mean()),float(v.mean())];observed_span=[float(z.min()),float(z.max())];h,w=depth.shape
        hull=cv2.convexHull(np.column_stack([u,v]).astype(np.int32));roi=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(roi,hull,1)
        fill=points/max(int(roi.sum()),1)
        if source=='part':
            # Shiny or dark parts break into pieces the bed seed never reaches. Depth inside the part's outline and in
            # front of it is the part too (only air lies between it and the scanner), so the near end never cuts it off.
            front=depth[roi.astype(bool)&valid].astype(float)*.1
            if front.size:span[0]=max(.1,min(span[0],float(front.min())-3*sigma));observed_span[0]=min(observed_span[0],float(front.min()))
        uncertainty=np.maximum(max(1.,3*sigma/(center_mm/min(cal[:2]))),3*np.array([u.std(),v.std()])/np.sqrt(points))
        center=np.array([w//2,h//2]);centred=bool(np.all(np.abs(np.array([u.mean(),v.mean()])-center)<=np.array([u.std(),v.std()])))
        # Use the actual extremal pixels for visibility; the trimmed box alone can
        # conceal a clipped edge. One pixel is the camera's sampling boundary.
        clipped=bool(u.min()==0 or v.min()==0 or u.max()==w-1 or v.max()==h-1)
        exposure=None
        if ir is not None and ir.shape==depth.shape:
            values=ir[roi.astype(bool)];limits=np.iinfo(ir.dtype)
            exposure=dict(dark_share=float(np.mean(values==limits.min)),clipped_share=float(np.mean(values==limits.max)),median=float(np.median(values)))
    else:exposure=None
    bed_distance=None
    if geo is not None and center_px is not None:
        ray=np.array([(center_px[0]-cal[2])/cal[0],(center_px[1]-cal[3])/cal[1],1.])
        den=float(np.asarray(geo[0])@ray)
        if den:bed_distance=float(-geo[1]/den)
    lo,hi=sorted(cut) if cut is not None else (0,np.inf)
    in_cut=valid&(depth*.1>=lo)&(depth*.1<=hi)
    background=int((in_cut&~mask).sum());kept=int((in_cut&mask).sum())
    cut_off=bool(span is not None and (span[0]<lo or span[1]>hi))
    return dict(source=source,rect=rect,mask=mask,cal=tuple(cal),bed_plane=None if geo is None else dict(normal=np.asarray(geo[0]).tolist(),offset_mm=float(geo[1]),center_mm=None if geo[2] is None else np.asarray(geo[2]).tolist(),radius_mm=float(geo[3]) if geo[3] is not None and np.isscalar(geo[3]) else None),depth_size=[depth.shape[1],depth.shape[0]],points=points,valid_pixels=int(valid.sum()),
                diagonal_mm=diagonal,fill=float(fill),bed_distance_mm=bed_distance,depth_mm=center_mm,range_mm=span,noise_mm=sigma,centered=centred,clipped=clipped,cut_off=cut_off,
                support_hull=None if hull is None else hull.reshape(-1,2),center_px=center_px,observed_range_mm=observed_span,uncertainty_px=uncertainty,background_pixels=background,kept_part_pixels=kept,exposure=exposure)


def public_scene(scene):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in scene.items() if k not in ('mask','uncertainty_px')}


class Framing:
    """Follow current depth and retain the full span observed during this side.

    Outward changes take effect immediately so a thin tip is never hidden by
    smoothing. The center and status use the existing observation window.
    """
    def __init__(self, window):
        self.window = float(window)
        self.reset()

    def reset(self):
        self.samples = deque()
        self.last = None
        self.last_time = None
        self.state = None
        self.candidate = None
        self.candidate_since = None
        self.union = None
        self.steady = False

    def update(self, scene, guide, t, automatic=True):
        rect = scene['rect']
        if rect is not None and self.union is not None:
            old = np.asarray(self.union['rect'])
            delta = np.linalg.norm((rect[:2] + rect[2:] - old[:2] - old[2:]) / 2)
            diameter = max(np.linalg.norm(old[2:] - old[:2]), np.linalg.norm(rect[2:] - rect[:2]))
            span=scene.get('observed_range_mm');old_span=self.union['observed_range_mm']
            gap=0 if span is None else max(span[0]-old_span[1],old_span[0]-span[1],0)
            physical=max(scene.get('diagonal_mm') or np.inf,self.union.get('diagonal_mm') or np.inf)
            if scene['source'] != self.union['source'] or delta > diameter or gap>physical:
                self.reset()
        self.last = scene
        self.last_time = float(t)
        if rect is not None:
            if self.union is None:
                self.union = dict(source=scene['source'],diagonal_mm=scene.get('diagonal_mm'),rect=np.asarray(rect).copy(), range_mm=list(scene['range_mm']),
                                  observed_range_mm=list(scene.get('observed_range_mm') or scene['range_mm']))
            else:
                if scene.get('diagonal_mm') is not None:self.union['diagonal_mm']=max(self.union.get('diagonal_mm') or 0,scene['diagonal_mm'])
                self.union['rect'][:2] = np.minimum(self.union['rect'][:2], rect[:2])
                self.union['rect'][2:] = np.maximum(self.union['rect'][2:], rect[2:])
                for name in ('range_mm', 'observed_range_mm'):
                    span = scene.get(name) or scene['range_mm']
                    self.union[name] = [min(self.union[name][0], span[0]), max(self.union[name][1], span[1])]
        quality = (guide or {}).get('state', 'unmeasured')
        if rect is None: state = 'no part depth' if scene['source'] == 'part' else 'no depth'
        elif scene['clipped']: state = 'partly out of view'
        elif scene.get('missing_depth'): state = 'missing part depth'
        elif scene.get('missing_extension_possible'):state='possible missing part depth'
        elif quality in ('too close', 'too far', 'sparse depth'): state = quality
        elif not scene['centered']: state = 'off center'
        elif scene['cut_off']: state = 'outside Range'
        else: state = 'framed'
        if state != self.candidate:
            self.candidate = state
            self.candidate_since = t
        if self.state is None or state == self.state or t - self.candidate_since >= self.window:
            self.state = state
        self.samples.append((t, None if rect is None else np.asarray(rect).copy(), scene.get('center_px')))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()
        boxes = [r for _, r, _ in self.samples if r is not None]
        tolerance = scene.get('uncertainty_px')
        self.steady = bool(len(boxes) > 1 and tolerance is not None and t - self.samples[0][0] >= self.window / 2 and
                           np.all(np.ptp(boxes, axis=0) <= np.tile(tolerance, 2)))
        centers = [c for _, _, c in self.samples if c is not None]
        center = np.median(centers, axis=0).tolist() if centers else None
        hints = {'framed': 'Part framed', 'off center': 'Bring the part center to the aim mark',
                 'partly out of view': 'Part goes past what depth can see: center it in the view',
                 'missing part depth': 'Part of the part shows no depth (dark or shiny surface)',
                 'possible missing part depth':'Check dark or shiny areas for missing part depth',
                 'outside Range': 'Press Use part Range', 'too close': 'Move the part farther away',
                 'too far': 'Move the part closer', 'sparse depth': 'Try more IR gain or a different angle',
                 'no part depth': 'Show the part above the bed', 'no depth': 'Show the part'}
        shown = self.union if rect is not None else None
        hint=hints[self.state]
        if self.state=='partly out of view' and scene.get('missing_depth'):
            hint='Part goes past the depth view, and depth is missing inside it too'
        elif self.state=='partly out of view' and scene.get('missing_extension_possible'):
            hint='Part goes past the depth view; check dark areas inside it too'
        # The same full observed depth span is used for every color projection.
        return dict(source=scene['source'], rect=None if shown is None else shown['rect'].tolist(),
                    depth_size=scene['depth_size'], depth_mm=(guide or {}).get('center_mm', scene['depth_mm']),
                    state=self.state, fresh_state=state, hint=hint, quality=quality,
                    range_mm=None if shown is None else list(shown['range_mm']), center_px=center,
                    observed_range_mm=None if shown is None else list(shown['observed_range_mm']),
                    depth_outline=None if scene.get('depth_outline') is None else np.asarray(scene['depth_outline']).tolist(), scene=public_scene(scene))
