# SPDX-License-Identifier: AGPL-3.0-or-later
"""One background distance worker. No depth download, fusion or ICP added."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from functools import wraps
import time


def distance_frame(function):
    @wraps(function)
    def frame(self,fr):
        self._distance_in_frame=True;gate=getattr(self,'_distance_idle',None)
        if gate is not None:gate.clear()
        start=time.monotonic()
        try:return function(self,fr)
        finally:
            end=time.monotonic();self._distance_in_frame=False
            gate=getattr(self,'_distance_idle',None)
            if gate is not None:
                self._distance_clock[0]=end;self._distance_clock[1]=end-start;gate.set()
    return frame



class DistanceSession:
    def _ensure_distance_pool(self):
        if hasattr(self,'_distance_pool'):return
        from app2.scan.capture_distance import background_priority
        context=get_context('spawn');self._distance_idle=context.Event();self._distance_clock=context.Array('d',[time.monotonic(),0],lock=False)
        if not getattr(self,'_distance_in_frame',False):self._distance_idle.set()
        self._distance_pool=ProcessPoolExecutor(max_workers=1,mp_context=context,initializer=background_priority,initargs=(self._distance_idle,self._distance_clock));self._distance_jobs={};self._distance_pending={}
    def _prepare_distance(self):
        # Start the process during the existing side preparation stage, before
        # recording. Bootstrap imports must not compete with incoming frames.
        import os
        start=time.monotonic();self._ensure_distance_pool();self._distance_pool.submit(os.getpid).result()
        self._distance_ready_ms=1000*(time.monotonic()-start)
    def _publish_distance(self,sd,payload):
        if not getattr(sd,'entries',None) or getattr(self,'_distance_closing',False):return
        self._ensure_distance_pool()
        self._distance_pending[sd]=payload
        if sd in self._distance_jobs:return
        from app2.scan.capture_distance import color_snapshot
        entries=[(a,b,M.copy()) for a,b,M in sd.entries];stamps=list(sd.stamps);cal=tuple(self.cal);shape=(self.cfg.h,self.cfg.w);mode=getattr(sd,'scan_profile',{}).get('mode','near')
        future=self._distance_pool.submit(color_snapshot,sd.folder,entries,cal,shape,mode,stamps,payload['P'],payload.get('B'));self._distance_jobs[sd]=future
        future.add_done_callback(lambda f:self._post(self._distance_ready,sd,payload,f))
    def _distance_ready(self,sd,payload,future):
        self._distance_jobs.pop(sd,None);pending=self._distance_pending.get(sd)
        try:levels=future.result()
        except Exception as error:self._log('Capture distance evidence: '+str(error));return
        if self._stop or getattr(self,'_distance_closing',False):return
        if pending is payload and self.picked is sd:
            colored=dict(payload,B=levels)
            if payload.get('V') is not None:                      # the surface takes the measured colors too, not the carried ones
                from app2.scan.session import cell_lookup
                colored['VB']=cell_lookup(payload['P'],levels,payload['V'],255)
            self._side_clouds[sd]=colored
            if not hasattr(self,'_side_colored'):self._side_colored={}
            self._side_colored[sd]=dict(P=payload['P'],B=levels)   # the newest measured colors, carried onto clouds that follow
            tm=self._touch_model(sd)
            if tm is not None:tm.capture_distance=levels
            self._emit('cloud',colored)
        elif pending is not None:self._publish_distance(sd,pending)
