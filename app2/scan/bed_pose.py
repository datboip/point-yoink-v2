# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured dots resolve the spin that depth ICP cannot see on a flat bed."""
import numpy as np
from scanner import bed


def quality(hit,cal):
    pixels=np.asarray(hit.dots_px)[np.asarray(hit.matched,bool),:2]
    span=0. if not len(pixels) else min(np.ptp(pixels[:,0])/cal[0],np.ptp(pixels[:,1])/cal[1])*abs(float(hit.centre[2]))
    return dict(inliers=int(hit.inliers),runner_up=int(hit.runner_up),rms_mm=float(hit.rms),span_mm=float(span))


def flat_scene(depth,cal,hit,model):
    """Require resolved bed support and no resolved raised part inside it."""
    from track import REGISTRATION_MIN_POINTS
    from app2.scan.framing import depth_noise
    from app2.scan.bedcut import BELOW_MM
    v,u=np.nonzero(depth)
    if len(u)<REGISTRATION_MIN_POINTS:return False
    z=depth[v,u].astype(float)*.1
    points=np.column_stack([(u-cal[2])*z/cal[0],(v-cal[3])*z/cal[1],z])
    delta=points-hit.centre;radial=np.sqrt(np.maximum((delta*delta).sum(1)-(delta@hit.n)**2,0))
    heights=points@hit.n+hit.off
    inside=radial<=model.radius_mm
    if np.count_nonzero(inside)<REGISTRATION_MIN_POINTS:return False
    noise=depth_noise(depth,depth>0)
    tolerance=max(BELOW_MM,3*noise)
    return np.count_nonzero(inside&(heights>tolerance))<REGISTRATION_MIN_POINTS


def pose(world_bed,camera_bed):
    T=world_bed@np.linalg.inv(camera_bed)
    T=T.copy();T[:3,3]/=1000
    return T


class BedPoseGuide:
    def __init__(self):
        self.reference=None;self.corrections=[];self.correction=None
        self.last_hit=None;self.used_ir=0;self.used_color=0
        from app2.scan.bed_resume import Reference
        self.accepted_reference=Reference();self.step_hit=None;self.reference_samples=0

    def accepted(self,result):
        if self.reference is not None:return
        if result.get('ok') and self.step_hit is not None:
            T=np.array(result['T'],copy=True);T[:3,3]*=1000
            self.accepted_reference.add(T@self.step_hit.B_cam)
            if self.reference is None:self.reference=self.accepted_reference.value

    def paired(self,hit,color):
        if color is None or self.correction is not None:return
        correction=hit.B_cam@np.linalg.inv(color['B_cam'])
        self.corrections.append(correction);self.corrections=self.corrections[-bed.ACCEPT_HITS:]
        if len(self.corrections)==bed.ACCEPT_HITS:
            center=np.mean(self.corrections,axis=0);u,_,v=np.linalg.svd(center[:3,:3]);center[:3,:3]=u@v
            if all(np.linalg.norm(c[:3,3]-center[:3,3])<=bed.HIT_RMS and bed.ang_deg(c[:3,0],center[:3,0])<=bed.HINT_GUARD_DEG for c in self.corrections):self.correction=center

    def step(self,adapter,side,d,ir,tracker,resume=False):
        import time
        from app2.scan.preview import RGB_CAL,decoded
        from app2.scan.color_bed import solve_pose,depth_pose
        self.step_hit=None
        sb=side.bed.sb;model=sb.bed
        if not bed.has_dots(model) or sb.hints_off(side.ps):return False
        hit=bed.find_bed(d,ir,adapter.cal,model,prior=sb.prior_for(side.ps))
        now=time.monotonic()
        if hit is not None and hit.height_from!='prior-only':
            self.last_hit=hit
            self.step_hit=hit
            adapter.color_observer.anchor(model,RGB_CAL,(hit.n,hit.off,hit.centre),adapter.color_observer.window,now=now)
            packet=getattr(adapter,'color_packet',None)
            if packet is not None and self.correction is None:
                color=solve_pose(decoded(packet['jpeg']),RGB_CAL,(hit.n,hit.off,hit.centre),model)
                self.paired(hit,color)
        if resume and self.reference is not None and self.step_hit is not None:
            tracker.resume_hint(pose(self.reference,hit.B_cam))
            self.used_ir+=1
            return True
        if resume and hit is None and self.reference is not None and self.correction is not None:
            packet=getattr(adapter,'color_packet',None)
            color=None if packet is None else depth_pose(decoded(packet['jpeg']),RGB_CAL,d,adapter.cal,model)
            if color is not None:
                tracker.resume_hint(pose(self.reference,self.correction@color['B_cam']))
                self.used_color+=1
                return True
        if hit is not None and flat_scene(d,adapter.cal,hit,model):
            if self.reference is None:
                self.reference=sb.B_scan()
                if self.reference is None:
                    if resume:return False
                    current=np.asarray(tracker.T).copy();current[:3,3]*=1000
                    self.reference=current@hit.B_cam
            tracker.marker_hint(pose(self.reference,hit.B_cam),quality(hit,adapter.cal))
            self.used_ir+=1
            return True
        # A color-only fallback expires with its measured plane. It also needs
        # several paired IR checks to remove the inter-camera angle bias.
        sample=adapter.color_observer.latest(now)
        if hit is None and sample is not None and self.correction is not None:
            stamp,color=sample
            if color is not None and now-stamp<=adapter.color_sync_s:
                reference=self.reference if self.reference is not None else sb.B_scan()
                if reference is not None:
                    T=pose(reference,self.correction@color['B_cam'])
                    if self.last_hit is not None and flat_scene(d,adapter.cal,self.last_hit,model):tracker.marker_hint(T,color)
                    else:tracker.hint(T,guard_mm=bed.HINT_GUARD_MM,guard_deg=bed.HINT_GUARD_DEG)
                    self.used_color+=1
                    return True
        return False
