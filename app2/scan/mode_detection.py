# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare physical bed geometry and color correspondence without trusting a mode label."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import numpy as np


def color_evidence(depth, ir, calibration, jpeg):
    """Count mutual circle matches within their observed radius; never force assignments."""
    from scanner import bed
    import cv2
    from app2.scan.preview import decode_colour
    folder=Path(calibration['folder'])
    try:
        raw=(folder/'Prgb.bin').read_bytes(); w,h=struct.unpack_from('<HH',raw)
        K=np.array(struct.unpack_from('<9f',raw,4)).reshape(3,3)
        rt=np.fromfile(folder/'LC_RT.bin',np.float32)
        if rt.size!=12 or not jpeg:return None
        # Exported RGB intrinsics describe their header size, not the JPEG's size.
        rgb=decode_colour(jpeg); K[0]*=rgb.shape[1]/w;K[1]*=rgb.shape[0]/h
        circles=bed.detect_dots(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        dots=bed.detect_dots(ir); P=bed.dot_depth_points(depth,calibration['intrinsics'],dots)
        good=np.isfinite(P).all(1); P=P[good];dots=dots[good]
        if not len(P) or not len(circles):return None
        Q=P@rt[:9].reshape(3,3).T+rt[9:]
        q=Q@K.T;q=q[:,:2]/q[:,2:]
        inside=(Q[:,2]>0)&(q[:,0]>=0)&(q[:,0]<rgb.shape[1])&(q[:,1]>=0)&(q[:,1]<rgb.shape[0])
        q=q[inside]
        if not len(q):return None
        dist=np.linalg.norm(q[:,None]-circles[None,:,:2],axis=2)
        j=dist.argmin(1); mutual=dist.argmin(0)[j]==np.arange(len(q)); offsets=dist[np.arange(len(q)),j]
        close=mutual&(offsets<=circles[j,2])
        spread=np.linalg.svd(q[close]-q[close].mean(0),compute_uv=False) if close.sum()>=3 else np.zeros(2)
        resolved=close.sum()>=3 and spread[-1]>float(np.max(circles[j[close],2]))
        return dict(projected=len(q),circles=len(circles),matches=int(close.sum()),
                    median_px=float(np.median(offsets[close])) if close.any() else None,
                    median_nearest_px=float(np.median(offsets)),
                    accepted=bool(resolved and close.sum()*bed.HIT_RATIO>=len(q)),
                    projected_points=q[close].tolist(),circle_points=circles[j[close],:2].tolist(),radii=circles[j[close],2].tolist())
    except (OSError,ValueError,struct.error,cv2.error):return None


def compare(depth,ir,bundles,model=None,jpeg=None):
    from scanner import bed
    result={}
    for mode,selected in bundles.items():
        hit=bed.find_bed(depth,ir,selected['intrinsics'],model,prior=None) if model is not None and len(model)>=bed.HIT_MIN else None
        color=color_evidence(depth,ir,selected,jpeg)
        dots=bed.detect_dots(ir); z=depth[depth>0]*.1
        result[mode]=dict(bed_fit=hit is not None,inliers=int(hit.inliers) if hit else 0,
                          rms_mm=float(hit.rms) if hit else None,plane=dict(normal=hit.n.tolist(),offset_mm=float(hit.off),center=hit.centre.tolist(),radius_mm=float(model.radius_mm),outline=model.outline.to_json(),x_axis=hit.x_axis.tolist()) if hit else None,color=color,
                          valid_pixels=int(len(z)),median_depth_mm=float(np.median(z)) if len(z) else None,
                          dot_radius_px=float(np.median(dots[:,2])) if len(dots) else None,
                          calibration_sha256=selected['source']['calibration_sha256'])
    return result


class Detector:
    """One newest-frame job at a time. Only the session thread consumes the result."""
    def __init__(self,bundles,model=None):
        from scanner import bed
        self.bundles=bundles;self.model=model;self.history=deque(maxlen=bed.ACCEPT_WINDOW)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='camera-check');self.future=None
        self.evidence=dict(state='checking',mode=None,frames=[])
    def begin(self):
        self.history.clear();self.evidence=dict(state='checking',mode=None,frames=[])
    def offer(self,d,ir,jpeg=None,sample_signature=None):
        if self.future is None:
            self._offered_signature=sample_signature;self.future=self.pool.submit(compare,d.copy(),ir.copy(),self.bundles,self.model,jpeg)
    def poll(self):
        if self.future is None or not self.future.done():return None
        from scanner import bed
        try:row=self.future.result()
        except Exception:
            self.future=None;self.evidence=dict(state='unavailable',mode=None,frames=list(self.history));return self.evidence
        self.future=None
        if getattr(self,'_offered_signature',None) is not None:row['_signature']=self._offered_signature
        self.history.append(row)
        votes={m:sum(bool(x.get(m,{}).get('bed_fit') or (x.get(m,{}).get('color') or {}).get('accepted')) for x in self.history) for m in self.bundles}
        winners=[m for m,n in votes.items() if n>=bed.ACCEPT_HITS]
        mode=winners[0] if len(winners)==1 and all(votes[m]==0 for m in votes if m!=winners[0]) else None
        self.evidence=dict(state='matched' if mode else 'checking' if len(self.history)<bed.ACCEPT_HITS else 'uncertain',mode=mode,votes=votes,frames=list(self.history))
        return self.evidence
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def signature(depth,ir):
    from scanner import bed
    from app2.scan.range_guide import center_distance
    z=depth[depth>0].astype(float)*.1;dots=bed.detect_dots(ir)
    return dict(distance=center_distance(depth),span=float(np.subtract(*np.percentile(z,[75,25]))) if len(z) else None,
                radius=float(np.median(dots[:,2])) if len(dots) else None,radius_mad=float(np.median(np.abs(dots[:,2]-np.median(dots[:,2])))) if len(dots) else None)


def changed(previous,current):
    if not previous:return False
    if previous['distance'] is None and current['distance'] is not None:return True
    if previous['radius'] is None and current['radius'] is not None:return True
    if previous['distance'] is not None and current['distance'] is not None and abs(current['distance']-previous['distance'])>max(.1,previous['span'] or .1):return True
    return previous['radius'] is not None and current['radius'] is not None and abs(current['radius']-previous['radius'])>max(1.,3*1.482602218505602*(previous['radius_mad'] or 0))
