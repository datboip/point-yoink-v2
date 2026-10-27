# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hold evidence of regular depth dropouts without diagnosing hardware."""
from collections import deque
import numpy as np


class StreamHealth:
    def __init__(self,window):
        self.samples=deque(maxlen=window);self.warning='';self.evidence=None;self.until=-np.inf;self.gap_note=None
    def add(self,t,valid):
        self.samples.append((float(t),int(valid)))
        if t>self.until:self.warning='';self.evidence=None
        # Four events give three independent periods. Two periods can agree by
        # chance in a short, uneven PC stream.
        if len(self.samples)<4:return self.warning
        a=np.array(self.samples);dt=np.diff(a[:,0]);positive=dt[dt>0]
        if not len(positive):return self.warning
        cadence=float(np.median(positive));mad=float(np.median(np.abs(positive-cadence)))
        jitter=1.482602218505602*mad
        a=a[a[:,0]>=t-self.samples.maxlen*cadence]
        if len(a)<4:return self.warning
        dt=np.diff(a[:,0])
        gaps=np.flatnonzero(dt>cadence+max(cadence,3*jitter))+1
        level=a[:,1];q=np.median(level[level>0]) if np.any(level>0) else 0
        floor=np.sqrt(max(q,1));pulses=np.flatnonzero((level<floor)&np.r_[True,level[:-1]>=floor])
        for kind,events in (('delivery gaps',gaps),('near-empty depth',pulses)):
            if len(events)<4:continue
            times=a[events,0];periods=np.diff(times);period=float(np.median(periods))
            # Event timing is quantized by one frame. Half a frame is the phase
            # uncertainty. Use the smaller of that and observed 3-sigma jitter.
            tolerance=min(cadence/2,max(np.finfo(float).eps*max(abs(t),1)*len(a),3*jitter))
            residual=times-(times[0]+np.arange(len(times))*period)
            if period>cadence and np.max(np.abs(residual))<=tolerance and kind=='delivery gaps':
                # Late frames with full depth are not depth loss (seen live: one late frame every ~2.8 s). Log only.
                if self.gap_note is None:self.gap_note=dict(kind=kind,period_s=period,cadence_s=cadence,jitter_s=jitter,events=len(events),valid_pixels=int(valid),median_valid_pixels=float(q))
                continue
            if period>cadence and np.max(np.abs(residual))<=tolerance:
                self.warning='Depth is dropping in a repeating pattern. Pause and check the stream.'
                self.evidence=dict(kind=kind,period_s=period,cadence_s=cadence,jitter_s=jitter,phase_tolerance_s=tolerance,
                                   events=len(events),valid_pixels=int(valid),median_valid_pixels=float(q))
                self.until=float(times[-1])+self.samples.maxlen*cadence;break
        return self.warning
