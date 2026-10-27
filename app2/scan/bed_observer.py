# SPDX-License-Identifier: AGPL-3.0-or-later
"""Newest-only bed angle observations, independent of whether a frame is kept."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app2.scan.rim import RimFit,RimMarkers,detect_markers,basis


class BedObserver:
    def __init__(self,cal,model,window):
        self.cal=cal;self.model=model;self.window=window;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bed-angle');self.future=None;self.last=None;self.rim=RimFit(window);self.markers=RimMarkers(getattr(model,'rim_markers',None));self.geometry=None;self.seen_at=None
    def measure(self,depth,ir,now):
        from scanner.bed import find_bed
        from app2.scan.bed_adapter import _shape
        from app2.scan.framing import depth_noise
        hit=find_bed(depth,ir,self.cal,self.model,prior=None)
        angle=None;source=None
        if hit is not None:
            self.geometry=(hit.n,hit.off,hit.centre,_shape(self.model,hit.x_axis));self.seen_at=now
            x,y=basis(hit.n);angle=float(np.degrees(np.arctan2(hit.x_axis@y,hit.x_axis@x)))%360;source='top dots'
        geo=self.geometry;noise=depth_noise(depth,depth>0)
        rim=self.rim.update(depth,self.cal,geo,noise,now) if geo is not None else None
        markers=detect_markers(depth,ir,self.cal,geo,noise,getattr(self.model,'dot_diam_mm',None),None if self.markers.learned is None else self.markers.learned['band_mm'])
        fallback=self.markers.observe(markers)
        if angle is None and fallback is not None:angle=fallback['angle_deg'];source='rim markers'
        # A needle without fresh absolute evidence stays unknown. The last
        # fitted geometry may be reused only by the optional rim measurement.
        return dict(t=now,angle_deg=angle,source=source,bed_seen=angle is not None,geo=geo,rim=rim,marker=fallback)
    def offer(self,depth,ir,now):
        if self.future is not None and self.future.done():
            try:self.last=self.future.result()
            except Exception:self.last=None
            self.future=None
        if self.future is None:self.future=self.pool.submit(self.measure,depth.copy(),ir.copy(),now)
        if self.last is not None and now-self.last['t']<=self.window:return self.last
        return None
    def close(self):self.pool.shutdown(wait=True)
