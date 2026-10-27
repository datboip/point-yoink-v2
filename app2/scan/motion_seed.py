# SPDX-License-Identifier: AGPL-3.0-or-later
"""Timestamped pose hints. Uncalibrated device fields never become poses."""
from collections import deque
from dataclasses import asdict
import json
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from app2.scan.telemetry import Sample


def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)


def jacobian(w):
    theta=np.linalg.norm(w);W=skew(w)
    if theta<np.sqrt(np.finfo(float).eps):return np.eye(3)+W/2+W@W/6
    return np.eye(3)+(1-np.cos(theta))/theta**2*W+(theta-np.sin(theta))/theta**3*W@W


def rate(delta,dt):
    w=Rotation.from_matrix(delta[:3,:3]).as_rotvec()
    return np.r_[w,np.linalg.solve(jacobian(w),delta[:3,3])]/dt


def move(velocity,dt):
    w=velocity[:3]*dt;T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(w).as_matrix();T[:3,3]=jacobian(w)@(velocity[3:]*dt);return T


class MotionLog:
    """JSONL raw samples in the same host-monotonic clock as depth frames."""
    def __init__(self,path,calibration=None,provenance='recorded',context=None):
        if provenance not in ('recorded','synthetic'):raise ValueError('Unknown motion provenance')
        self.lock=threading.Lock();self.context=dict(context or {})
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('w');self.file.write(json.dumps(dict(version=1,clock='host_monotonic_seconds',provenance=provenance,calibration=calibration,context=self.context,protocol='locally observed 60-byte packets; raw units and axes unless calibrated'))+'\n');self.file.flush()
    def set_context(self,**values):
        with self.lock:self.context.update(values)
    def append(self,sample):
        with self.lock:
            if not self.file.closed:self.file.write(json.dumps(dict(asdict(sample),context=dict(self.context)))+'\n');self.file.flush()
    def close(self):
        with self.lock:self.file.close()


class MotionReplay:
    """Offline provider: never returns a future packet or opens a socket."""
    def __init__(self,path,stale_after=None):
        from app2.scan.telemetry import TelemetryClient
        self.stale_after=TelemetryClient().stale_after if stale_after is None else stale_after
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.header=rows[0]
        if self.header.get('version')!=1 or self.header.get('clock')!='host_monotonic_seconds':raise ValueError('Unsupported motion log clock')
        self.samples=[]
        for row in rows[1:]:
            sample=Sample(**{name:tuple(row[name]) if name!='received' else float(row[name]) for name in Sample.__dataclass_fields__})
            if not np.isfinite(np.r_[sample.quaternion,sample.gyro,sample.acceleration,sample.position,sample.received]).all():raise ValueError('Invalid motion log sample')
            if self.samples and sample.received<=self.samples[-1].received:raise ValueError('Motion timestamps must increase')
            self.samples.append(sample)
    def snapshot(self,now):
        samples=[s for s in self.samples if s.received<=now]
        if not samples:return 'Waiting for data',None,0,None
        sample=samples[-1];age=now-sample.received
        return ('Receiving' if age<=self.stale_after else 'Stale'),sample,len(samples),age


class PoseSeeder:
    """Separate fitted object motion from calibrated camera motion before ICP."""
    def __init__(self,calibration=None,history=None):
        from app2.scan.tracking import TrackerHost
        self.calibration=calibration if isinstance(calibration,dict) else {};self.history=deque(maxlen=history or TrackerHost.CKPT_KEEP);self.axis=None;self.last_sensor=None
    def sensor(self,snapshot,t):
        try:return self._calibrated_sensor(snapshot,t)
        except (ValueError,TypeError,AttributeError,IndexError):return None
    def _calibrated_sensor(self,snapshot,t):
        if not snapshot:return None
        state,sample,*_=snapshot
        c=self.calibration
        offset=c.get('received_to_depth_offset_s')
        if c.get('timestamp_basis')!='host_monotonic_seconds' or offset is None or not np.isfinite(offset):return None
        if state!='Receiving' or sample is None or not np.isfinite(sample.received) or sample.received+offset>t:return None
        from app2.scan.telemetry import TelemetryClient
        if t-(sample.received+offset)>TelemetryClient().stale_after:return None
        c=self.calibration;M=np.asarray(c.get('sensor_to_camera',[]),float)
        if M.shape!=(3,3) or not np.allclose(M@M.T,np.eye(3)) or not np.isclose(np.linalg.det(M),1):return None
        if c.get('quaternion_frame')!='sensor_to_world':return None
        q=np.asarray(sample.quaternion,float)
        if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)==0:return None
        order=c.get('quaternion_order')
        if order=='wxyz':q=q[[1,2,3,0]]
        elif order!='xyzw':return None
        R=Rotation.from_quat(q).as_matrix()@M.T
        position=None
        if c.get('position_scale_m') is not None:
            scale=float(c['position_scale_m'])
            if not np.isfinite(scale) or scale<=0:return None
            position=np.asarray(sample.position,float)*scale
        gravity=None
        norm=c.get('gravity_norm');sigma=c.get('gravity_sigma')
        accel=np.asarray(sample.acceleration,float)
        magnitude=np.linalg.norm(accel)
        if norm is not None and sigma is not None and np.isfinite([norm,sigma,magnitude]).all() and norm>0 and 0<=sigma<norm/3 and magnitude>0 and abs(magnitude-norm)<=3*sigma:
            gravity=M@accel;gravity/=np.linalg.norm(gravity)
        gyro=None
        if c.get('gyro_scale_rad_s') is not None:
            scale=float(c['gyro_scale_rad_s'])
            if not np.isfinite(scale) or scale<=0:return None
            gyro=M@np.asarray(sample.gyro,float)*scale
            if not np.isfinite(gyro).all():return None
        return dict(R=R,p=position,gravity=gravity,gyro=gyro,t=sample.received)
    @staticmethod
    def camera_delta(previous,current):
        T=np.eye(4)
        if previous is None or current is None:return T
        T[:3,:3]=previous['R'].T@current['R']
        if previous['p'] is not None and current['p'] is not None:T[:3,3]=previous['R'].T@(current['p']-previous['p'])
        return T
    def predict(self,t,snapshot=None,board=None):
        from app2.scan.tracking import AGREE_MM,AGREE_DEG
        if len(self.history)<3:return None,[]
        sensor=self.sensor(snapshot,t);last=self.history[-1]
        if sensor is not None and not all(h['sensor'] is not None for h in self.history):sensor=None
        dt=t-last['t'];intervals=np.diff([h['t'] for h in self.history])
        if dt<=0 or dt>sum(intervals):return None,[]
        velocities=[]
        for a,b in zip(self.history,list(self.history)[1:]):
            camera=self.camera_delta(a['sensor'],b['sensor']) if sensor is not None else np.eye(4)
            velocities.append(rate(b['T']@np.linalg.inv(a['T']@camera),b['t']-a['t']))
        velocity=np.median(velocities,axis=0);spread=np.max(np.abs(velocities-velocity),axis=0)*dt
        if np.degrees(np.linalg.norm(spread[:3]))>AGREE_DEG or np.linalg.norm(spread[3:])*1000>AGREE_MM:return None,[]
        sources=['steady fitted motion']
        if self.axis is not None and board is not None:
            velocity[:3]=self.axis*np.dot(velocity[:3],self.axis);sources.append('bed plane axis')
        camera=self.camera_delta(last['sensor'],sensor)
        if sensor is not None and last['sensor'] is not None:
            if sensor['gyro'] is not None:
                observed=Rotation.from_matrix(camera[:3,:3]).as_rotvec()
                # Gyro is a consistency check, with its scale supplied by calibration.
                if np.degrees(np.linalg.norm(observed-sensor['gyro']*dt))>AGREE_DEG:return None,[]
                sources.append('calibrated gyro')
            sources.append('calibrated camera pose')
        T=move(velocity,dt)@last['T']@camera
        if sensor is not None and sensor['gravity'] is not None and self.axis is not None:
            current=T[:3,:3]@sensor['gravity'];cross=np.cross(current,self.axis);length=np.linalg.norm(cross)
            if length>np.finfo(float).eps:
                angle=np.arctan2(length,np.dot(current,self.axis));T[:3,:3]=Rotation.from_rotvec(cross/length*angle).as_matrix()@T[:3,:3]
            sources.append('calibrated gravity')
        return T,sources
    def accepted(self,T,t,snapshot=None,board=None):
        if not np.isfinite(t) or (self.history and t<=self.history[-1]['t']):return
        sensor=self.sensor(snapshot,t)
        if self.axis is None:
            n=board[0] if board is not None else (sensor['gravity'] if sensor is not None else None)
            if n is not None:
                self.axis=np.asarray(T)[:3,:3]@np.asarray(n);self.axis/=np.linalg.norm(self.axis)
                if sensor is not None and sensor['gravity'] is not None and np.dot(self.axis,np.asarray(T)[:3,:3]@sensor['gravity'])<0:self.axis=-self.axis
        self.history.append(dict(T=np.asarray(T).copy(),t=float(t),sensor=sensor))


class TurntableOrbit:
    """Certify a fixed axis and center from accepted poses before constraining ICP.

    A freehand path, short arc, or inconsistent orbit keeps ordinary registration.
    This measures the camera-pose orbit, not motion from undocumented IMU fields.
    """
    # 2026-10-25, measured on a 1771-frame hood side turning 0.4 deg a frame: 90 poses / 60 deg never certified (40 deg
    # seen); 40 poses / 12 deg certified and held the fit on the orbit in 1063 frames, drift sd 0.20 -> 0.17 mm, range
    # 1.9 -> 1.5 mm, roughness unchanged, and it still drops itself in the hand-held stretches.
    WINDOW=40            # poses the orbit is fitted over (None: bed.POOL_TO, 90); a shorter window follows slow tracker drift
    MIN_ARC_DEG=12       # arc the window must span before certifying (None: SidePlane.disc's default, 60)
    def __init__(self, tolerance_mm):
        import bed
        self.poses=deque(maxlen=self.WINDOW or bed.POOL_TO);self.tolerance=float(tolerance_mm)/1000
        self.model=None;self.used=0
    def accepted(self,T):
        from app2.scan.tracking import AGREE_DEG
        self.poses.append(np.asarray(T).copy())
        # The existing table detector also needs at least eight observed frames.
        if len(self.poses)<8:return
        first=self.poses[0];deltas=[T@np.linalg.inv(first) for T in self.poses]
        vectors=np.array([Rotation.from_matrix(D[:3,:3]).as_rotvec() for D in deltas])
        _,_,axes=np.linalg.svd(vectors,full_matrices=False);axis=axes[0]
        angles=vectors@axis
        from table_plane import SidePlane
        import inspect
        min_arc=self.MIN_ARC_DEG or inspect.signature(SidePlane.disc).parameters['min_arc_deg'].default
        if np.ptp(angles)<np.radians(min_arc):return
        transverse=vectors-angles[:,None]*axis
        if np.max(np.linalg.norm(transverse,axis=1))>np.radians(AGREE_DEG):return
        rotations=Rotation.from_rotvec(angles[:,None]*axis).as_matrix()
        A=np.concatenate([np.eye(3)-R for R in rotations]);b=np.concatenate([D[:3,3] for D in deltas])
        center=np.linalg.lstsq(A,b,rcond=None)[0]
        errors=np.array([(np.eye(3)-R)@center-D[:3,3] for R,D in zip(rotations,deltas)])
        if np.max(np.linalg.norm(errors,axis=1))>self.tolerance:return
        if np.linalg.norm(np.cross(axis,center))<self.tolerance:return
        self.model=dict(first=first,axis=axis,center=center,residual_mm=float(np.max(np.linalg.norm(errors,axis=1))*1000),frames=len(self.poses))
    def project(self,T):
        if self.model is None:return None
        m=self.model;D=np.asarray(T)@np.linalg.inv(m['first'])
        angle=np.dot(Rotation.from_matrix(D[:3,:3]).as_rotvec(),m['axis'])
        R=Rotation.from_rotvec(m['axis']*angle).as_matrix();Q=np.eye(4)
        Q[:3,:3]=R;Q[:3,3]=(np.eye(3)-R)@m['center']
        projected=Q@m['first']
        D=np.linalg.inv(projected)@T
        from app2.scan.tracking import AGREE_DEG
        if np.linalg.norm(D[:3,3])>self.tolerance or np.linalg.norm(Rotation.from_matrix(D[:3,:3]).as_rotvec())>np.radians(AGREE_DEG):
            self.model=None;self.poses.clear();return None
        return projected
