# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan useful scanner views in millimeters, using the saved rectified stereo pair."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

RANK_COLORS = ('#50cfed', '#ffbe55', '#7da7ff')

@dataclass
class Lens:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 400
    height: int = 300
    near: float = 100
    far: float = 400
    second: tuple = (0., 0., 0.)
    right_cx: float | None = None

    @classmethod
    def from_param(cls, folder, size=(400, 300), distance=(100, 400)):
        p = Path(folder)
        v = np.frombuffer((p/'Pl.bin').read_bytes(), '<f4')
        q = np.frombuffer((p/'Q.bin').read_bytes(), '<f4').reshape(4, 4)
        # Q maps rectified disparity to depth: Q[3,2] = -1/Tx.
        # This is the RIGHT IR camera, NOT LC_RT (depth -> RGB).
        if not np.allclose(q[:2,:2], np.eye(2), atol=1e-5) or abs(q[3,2]) < 1e-8:
            raise ValueError('Stereo calibration is not a supported rectified pair')
        baseline = 1./float(q[3,2])
        if not 5 < abs(baseline) < 250:
            raise ValueError('Stereo baseline is outside the supported range')
        for name in ('mapparamL.bin', 'mapparamR.bin'):
            b = (p/name).read_bytes()
            if len(b) < 48 or not np.allclose(np.frombuffer(b[12:48], '<f4').reshape(3,3), np.eye(3), atol=.01):
                raise ValueError('Stereo pair needs a verified rectified rotation')
        scale=size[0]/800.
        right_cx=(float(v[3])+float(q[3,3]/q[3,2]))*scale
        return cls(float(v[1])*scale,float(v[5])*scale,float(v[3])*scale,float(v[6])*scale,*size,*distance,(baseline,0.,0.),right_cx)


def unit(x):
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def look_at(eye, target, up):
    z = unit(np.asarray(target)-eye); x = unit(np.cross(z,up)); y = np.cross(z,x)
    if np.linalg.norm(x) < .5: raise ValueError('Scanner view is parallel to its up direction')
    T = np.eye(4); T[:3,:3] = np.column_stack((x,y,z)); T[:3,3] = eye
    return T


def axis_from_poses(poses, bed_normal):
    """Fit the scanner's first-turn circle; use the bed normal only if the fit is underconstrained."""
    a = np.array([np.asarray(T)[:3,3] for T in poses], float)
    up = unit(bed_normal)
    if len(a) < 6: return up, None
    c = a.mean(0); _,s,R = np.linalg.svd(a-c,full_matrices=False)
    if s[1] < 3 or s[2] > .25*s[1]: return up, None
    n = R[-1]; n = n if n@up >= 0 else -n
    xy = (a-c) @ R[:2].T
    A = np.column_stack((2*xy,np.ones(len(xy))))
    sol = np.linalg.lstsq(A,(xy*xy).sum(1),rcond=None)[0]
    return n, c + sol[:2] @ R[:2]


class Reachability:
    def __init__(self, V, F, points, normals, lens, bed=None, axis=None, center=None):
        import open3d as o3d
        self.o3d = o3d; self.P = np.asarray(points,float); self.N = unit(normals); self.lens = lens
        self.axis = unit(axis if axis is not None else (bed[0] if bed is not None else [0,1,0]))
        self.center = np.asarray(center if center is not None else np.mean(V,axis=0),float)
        self.scene = o3d.t.geometry.RaycastingScene()
        def add(v,f):
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        add(np.asarray(V,float), np.asarray(F,np.int32))
        if bed is not None:
            n,d = bed; n=unit(n); c = self.center - (self.center@n+float(d))*n
            x = unit(np.cross(n,[1,0,0] if abs(n[0])<.9 else [0,1,0])); y=np.cross(n,x)
            r=max(2000,np.linalg.norm(np.ptp(V,axis=0))*5)
            add(np.array([c+r*(sx*x+sy*y) for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1))]),[[0,1,2],[0,2,3]])

    def clear(self, eye, P):
        if not len(P): return np.zeros(0,bool)
        v=P-eye; dist=np.linalg.norm(v,axis=1)
        rays=np.column_stack((np.broadcast_to(eye,v.shape),unit(v))).astype(np.float32)
        hit=self.scene.cast_rays(self.o3d.core.Tensor(rays))['t_hit'].numpy()
        # Stop at the first surface. Small endpoint tolerance covers interpolated spots.
        return hit >= dist-np.maximum(.001,dist*1e-6)

    def visibility(self, pose, turntable=True, step=5):
        T=np.asarray(pose,float); l=self.lens
        angles=np.arange(0,360,step) if turntable else np.array([0])
        masks=[]
        for deg in angles:
            reached=np.zeros(len(self.P),bool)
            a=np.deg2rad(deg); n=self.axis
            K=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
            R=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
            P=self.P; N=self.N
            eye=(T[:3,3]-self.center)@R+self.center; camera=R.T@T[:3,:3]
            Pc=(P-eye)@camera; z=Pc[:,2]
            with np.errstate(divide='ignore',invalid='ignore'):
                u=Pc[:,0]*l.fx/z+l.cx; v=Pc[:,1]*l.fy/z+l.cy
            valid=(z>=l.near)&(z<=l.far)&(u>=0)&(u<l.width)&(v>=0)&(v<l.height)
            # A stereo surface must also fall inside the rectified right camera image.
            right=Pc-np.asarray(l.second)
            with np.errstate(divide='ignore',invalid='ignore'):
                ur=right[:,0]*l.fx/right[:,2]+(l.cx if l.right_cx is None else l.right_cx)
                vr=right[:,1]*l.fy/right[:,2]+l.cy
            valid &= (right[:,2]>0)&(ur>=0)&(ur<l.width)&(vr>=0)&(vr<l.height)
            valid &= (N*unit(eye-P)).sum(1)>=.5
            i=np.flatnonzero(valid)
            if len(i):
                second=eye+camera@np.asarray(l.second)
                ok=self.clear(eye,P[i])&self.clear(second,P[i])
                reached[i[ok]]=True
            masks.append(reached)
        return angles,np.asarray(masks,bool)


    def reachable(self, pose, turntable=True, step=5):
        return self.visibility(pose,turntable,step)[1].any(0)

    def phase_pose(self, pose, degrees):
        T=np.asarray(pose,float);a=np.deg2rad(degrees);n=self.axis
        K=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        R=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
        out=T.copy();out[:3,:3]=R.T@T[:3,:3];out[:3,3]=(T[:3,3]-self.center)@R+self.center
        return out


def candidates(checker, mode='turntable'):
    up=checker.axis; c=checker.center; l=checker.lens
    x=unit(np.cross(up,[0,0,1] if abs(up[2])<.9 else [1,0,0])); y=np.cross(up,x)
    # Stand stays on one side; the table supplies yaw. Handheld allows every yaw.
    for yaw in ([0] if mode=='turntable' else np.arange(0,360,30)):
        radial=np.cos(np.deg2rad(yaw))*x+np.sin(np.deg2rad(yaw))*y
        for tilt in (0,15,30,45,60):
            for dist in np.linspace(l.near*1.25,l.far*.85,4):
                t=np.deg2rad(tilt); eye=c+dist*(np.cos(t)*radial+np.sin(t)*up)
                T=look_at(eye,c,up)
                # Aim the middle of the depth image, including an off-center rectified crop.
                ray=unit([(l.width/2-l.cx)/l.fx,(l.height/2-l.cy)/l.fy,1])
                cross=np.cross(ray,[0,0,1]); cosine=float(ray[2])
                K=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
                align=np.eye(3)+K+K@K/(1+cosine)
                T[:3,:3]=T[:3,:3]@align
                actual=round(float(np.rad2deg(np.arcsin(np.clip(-T[:3,2]@up,-1,1)))))
                yield dict(T=T,tilt=actual,distance=float(dist),height=float((eye-c)@up),mode=mode)


def plan(checker, weights, mode='turntable', limit=3, step=5):
    w=np.asarray(weights,float);total=float(w.sum());remaining=w>0
    pool=[];possible=np.zeros(len(w),bool)
    for candidate in candidates(checker,mode):
        angles,masks=checker.visibility(candidate['T'],mode=='turntable',step)
        possible|=masks.any(0);pool.append((candidate,angles,masks))

    def advice(entry):
        c,angles,masks=entry;union=masks.any(0)
        scores=masks@(w*remaining);phase=int(np.argmax(scores))
        full=float(w[union&remaining].sum());best=float(scores[phase])
        compact=best>=.7*full
        groups=getattr(checker,'groups',None)
        if groups is not None:
            ids=np.unique(groups[union&remaining]);small=bool(len(ids) and len(ids)<=2 and (ids>=0).all())
            for k in ids:
                mb=(groups==k)&union&remaining
                small &= float(w[mb].sum())<=500 and np.linalg.norm((checker.N[mb]*w[mb,None]).sum(0)/max(float(w[mb].sum()),1e-12))>=.75
            compact |= small
        held=mode!='turntable' or compact
        mask=masks[phase] if held else union;new=mask&remaining;area=float(w[new].sum())
        repeat=float(w[mask].sum())>2*area
        capture='Smart frames' if repeat else ('Single' if held and area<=150 else 'Burst' if held else 'Continuous')
        return area,c,angles[phase],mask,new,held,capture

    selected=[]
    for rank in range(limit):
        if not pool or not remaining.any():break
        area,c,angle,mask,new,held,capture=max((advice(e) for e in pool),key=lambda r:r[0])
        if area<=0:break
        remaining[new]=False;share=area/max(total,1e-12)
        pose=checker.phase_pose(c['T'],angle) if held and mode=='turntable' else np.asarray(c['T'])
        if held:
            action=(f'turn the table about {int(angle)} degrees, ' if mode=='turntable' and angle else '')+'hold still for '+capture
        else:action='one turn'+(' with Smart frames' if capture=='Smart frames' else ' (Continuous)')
        selected.append(dict(c,rank=rank+1,share=share,area_mm2=area,indices=np.flatnonzero(new).tolist(),capture=capture,color=RANK_COLORS[rank],T=pose.tolist(),table_angle_deg=float(angle) if held and mode=='turntable' else None,hold_still=bool(held),text=f'Tilt down about {c["tilt"]} degrees, '+action+f': fills about {round(100*share)} percent'))
    bed_side=(checker.N@checker.axis)<-.3
    return dict(poses=selected,needs_flip=np.flatnonzero((w>0)&~possible&bed_side).tolist(),unreachable=np.flatnonzero((w>0)&~possible&~bed_side).tolist(),remaining=np.flatnonzero(remaining).tolist(),mode=mode)


def guidance(current,target, gravity=None, turntable=True, axis=(0,1,0), center=(0,0,0), tracking_ok=True, hold_still=False):
    if not tracking_ok:return dict(text="Find the part again to check your position.",ready=False,color="#ffbe55")
    C=np.asarray(current,float); T=np.asarray(target,float); up=unit(axis)
    # A calibrated acceleration vector supplies world up, without starting a motion connection.
    if gravity is not None and np.linalg.norm(gravity)>1e-6: up=unit(-np.asarray(gravity))
    roll=0.
    if turntable:
        current_up=C[:3,:2].T@up;target_up=T[:3,:2].T@up
        if min(np.linalg.norm(current_up),np.linalg.norm(target_up))>1e-6:
            roll=float(np.rad2deg(np.arctan2(current_up[0]*target_up[1]-current_up[1]*target_up[0],current_up@target_up)))
        tilt=np.rad2deg(np.arcsin(np.clip(-C[:3,2]@up,-1,1)))
        want=np.rad2deg(np.arcsin(np.clip(-T[:3,2]@up,-1,1))); angle=want-tilt
    else:
        angle=np.rad2deg(np.arccos(np.clip((np.trace(C[:3,:3].T@T[:3,:3])-1)/2,-1,1)))
    delta=C[:3,3]-T[:3,3]; height=float(delta@up)
    distance=np.linalg.norm(C[:3,3]-center)-np.linalg.norm(T[:3,3]-center)
    placement=abs(height)<15 if turntable else np.linalg.norm(delta)<15
    phase_error=0.
    if turntable and hold_still:
        here=C[:3,3]-center;there=T[:3,3]-center
        here=unit(here-up*(here@up));there=unit(there-up*(there@up))
        phase_error=float(np.rad2deg(np.arctan2(up@np.cross(here,there),np.clip(here@there,-1,1))))
        placement &= abs(phase_error)<5
    ready=abs(angle)<5 and abs(roll)<5 and abs(distance)<15 and placement
    if ready: return dict(text='You are there. Hold this position.',ready=True,color='#7de38d')
    words=[]
    if abs(roll)>=5:words.append(f'Roll the scanner about {round(abs(roll))} degrees back toward the ghost scanner')
    if abs(phase_error)>=5:words.append(f'Turn the table about {round(abs(phase_error))} degrees '+('clockwise' if phase_error>0 else 'counterclockwise')+', then hold it still')
    if abs(angle)>=5: words.append(f'Tilt {round(abs(angle))} degrees '+('more' if angle>=0 else 'less'))
    if height>15: words.append('Lower the scanner a bit')
    elif height< -15: words.append('Raise the scanner a bit')
    if distance>15: words.append('A bit closer')
    elif distance< -15: words.append('A bit farther back')
    if not turntable and not placement and abs(distance)<15 and abs(height)<15: words.append('Move toward the ghost scanner')
    return dict(text='. '.join(words),ready=False,color='#ffbe55')


class GravityAlignment:
    """Fit device acceleration axes from already received samples and tracked camera tilt.

    Two distinct directions are required. Until this fit is consistent, tracker tilt remains authoritative.
    """
    def __init__(self):self.samples=[];self.rotation=None

    def add(self,camera,acceleration,up):
        a=np.asarray(acceleration,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-6:return self.rotation
        observed=unit(a);expected=np.asarray(camera,float)[:3,:3].T @ -unit(up)
        self.samples.append((observed,expected));self.samples=self.samples[-60:]
        if len(self.samples)<6:return self.rotation
        A=np.array([a for a,_ in self.samples]);B=np.array([b for _,b in self.samples])
        if np.linalg.svd(A,compute_uv=False)[1]<.3:return self.rotation
        U,_,VT=np.linalg.svd(A.T@B);D=np.eye(3);D[-1,-1]=np.linalg.det(U@VT)
        rotation=(U@D@VT).T
        if np.max(np.linalg.norm(A@rotation.T-B,axis=1))<.08:self.rotation=rotation
        return self.rotation
