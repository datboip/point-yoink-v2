# SPDX-License-Identifier: AGPL-3.0-or-later
"""Color dot observations for guidance only. Never used as scan poses."""
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from threading import RLock


def solve(rgb, rgb_cal, plane, model):
    """An absolute angle needs a measured plane and an unambiguous map match.

    Reusing a plane assumes the camera has not moved. Callers must expire that
    anchor and must keep this estimate out of depth registration and geometry.
    """
    from scanner import bed
    from app2.scan.rim import basis
    if rgb_cal is None or plane is None or model is None or not bed.has_dots(model):return None
    K,R,T=rgb_cal
    n,off=plane[:2]
    dots=bed.detect_dots(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    if len(dots)<bed.HIT_MIN:return None
    rays=np.column_stack([dots[:,:2],np.ones(len(dots))])@np.linalg.inv(K).T@R
    origin=-np.asarray(T)@R
    den=rays@np.asarray(n)
    with np.errstate(divide='ignore',invalid='ignore'):distance=-(origin@np.asarray(n)+off)/den
    p=origin+distance[:,None]*rays
    ok=np.isfinite(p).all(1)&(distance>0)&(p[:,2]>0)
    if len(plane)>2 and plane[2] is not None:
        # The recent measured bed footprint excludes unrelated bright spots
        # on walls and furniture. It contributes no new surface depth.
        radius=getattr(model,'radius_mm',None)
        if radius is not None:
            diameter=getattr(model,'dot_diam_mm',None) or 0.
            ok &= np.linalg.norm(p-np.asarray(plane[2]),axis=1)<=radius+diameter/2
    x,y=bed.basis(n)
    points=np.column_stack([p[ok]@x,p[ok]@y])
    reg=bed.register_2d(points,model.map2d)
    if not bed.is_hit(reg):return None
    _,_,axis=bed.bed_frame(n,off,reg.R,reg.t)
    x,y=basis(n)
    angle=float(np.degrees(np.arctan2(axis@y,axis@x)))%360
    return dict(angle_deg=angle,inliers=int(reg.inliers),rms_mm=float(reg.rms),source='Estimated from color dots')


class ColorObserver:
    """One color job at a time; no frame queue and no scan-thread inference."""
    def __init__(self, window, measure=solve, interval=0.):
        self.window=window; self.measure=measure
        self.interval=max(0.,float(interval));self.submitted=float('-inf')
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='color-angle')
        self.future=None;self.context=None;self.generation=0
        self.deadline=0;self.last=None;self.dropped=0;self.closed=False
        self.lock=RLock()

    def anchor(self, model, rgb_cal, plane, valid_for, now=None):
        with self.lock:
            now=time.monotonic() if now is None else now
            if model is None or rgb_cal is None or plane is None:
                self.clear();return
            copied=(np.array(plane[0],copy=True),float(plane[1]))
            if len(plane)>2 and plane[2] is not None:copied+= (np.array(plane[2],copy=True),)
            calibration=tuple(np.array(x,copy=True) for x in rgb_cal)
            changed=self.context is None or self.context[0] is not model
            if not changed:
                from scanner.bed import HIT_RMS
                old_cal,old_plane=self.context[1:]
                changed=any(not np.array_equal(a,b) for a,b in zip(calibration,old_cal)) or len(copied)!=len(old_plane)
                changed |= abs(copied[1]-old_plane[1])>HIT_RMS or np.linalg.norm(copied[0]-old_plane[0])*abs(copied[1])>HIT_RMS
                if len(copied)>2 and len(old_plane)>2:changed |= np.linalg.norm(copied[2]-old_plane[2])>HIT_RMS
            if changed:
                self.generation+=1;self.last=None;self.submitted=float('-inf')
            self.context=(model,calibration,copied)
            self.deadline=now+min(self.window,max(0.,valid_for))

    def clear(self):
        with self.lock:
            self.context=None;self.last=None;self.deadline=0;self.generation+=1;self.submitted=float('-inf')

    def latest(self, now=None):
        with self.lock:
            now=time.monotonic() if now is None else now
            if self.context is None or now>self.deadline or self.last is None:return None
            stamp,result=self.last
            return (stamp,result) if 0<=now-stamp<=self.window else None

    def offer(self, rgb, now=None):
        with self.lock:
            now=time.monotonic() if now is None else now
            if self.closed:return None
            if self.future is not None and self.future.done():
                generation,stamp,future=self.generation_job,self.stamp_job,self.future
                self.future=None
                try:result=future.result()
                except Exception:result=None
                self.last=(stamp,result) if generation==self.generation else None
            if self.context is None or now>self.deadline:
                self.last=None;return None
            if self.future is None and now-self.submitted>=self.interval:
                model,cal,plane=self.context
                self.submitted=now
                self.generation_job=self.generation;self.stamp_job=now
                self.future=self.pool.submit(self.measure,np.array(rgb,copy=True),cal,plane,model)
            elif self.future is not None:self.dropped+=1
            return None if self.last is None or now-self.last[0]>self.window else self.last[1]

    def close(self):
        with self.lock:self.closed=True;self.clear()
        self.pool.shutdown(wait=True,cancel_futures=True)
