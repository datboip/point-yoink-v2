# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quality-driven saving beside the tracker's live TSDF, in side coordinates."""
from collections import deque
from pathlib import Path
import ast,json
import numpy as np
from app2.scan.range_guide import distance_levels,measured_profile


def tracker_default(name):
    tree=ast.parse((Path(__file__).resolve().parents[2]/'track.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Tracker')
    for n in cls.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple):
            values=ast.literal_eval(n.value)
            for target,value in zip(n.targets[0].elts,values):
                if isinstance(target,ast.Name) and target.id==name:return value
    raise KeyError(name)


def detail_levels(mode):
    """The mode's saved robust sigma sets spatial resolution, not an accuracy claim."""
    profile=json.loads(Path(__file__).with_name('surface_profiles.json').read_text())[mode]
    sigma=profile['noise_sigma_mm'];floor=max(.1/np.sqrt(12),profile['pixel_pitch_mm']/2)
    return {name:float(multiplier*max(sigma,floor)) for name,multiplier in (('Fine',2),('Balanced',4),('Coarse',8))}


def codes(points,step):
    xyz=np.floor(np.asarray(points)/step).astype(np.int64)
    # Exact packed coordinates make the per-frame sort numeric. This is the
    # same domain as the capture-distance field. Refuse impossible drift or a
    # corrupted pose rather than silently allowing a key collision.
    bits=np.iinfo(np.uint64).bits//3;bias=1<<(bits-1)
    if np.any((xyz < -bias)|(xyz >= bias)):raise ValueError('Live build pose is outside the supported model space. Start a new side.')
    q=(xyz+bias).astype(np.uint64)
    return (q[:,0]<<np.uint64(2*bits))|(q[:,1]<<np.uint64(bits))|q[:,2]


class QualityField:
    def __init__(self,step,profile=None,support=None,min_share=None):
        self.step=float(step);self.profile=profile or measured_profile('near');self.target=int(support or tracker_default('COVER_GOOD'))
        # Same tunable starting threshold as Smart. Unlike Smart, it measures
        # saved voxel evidence and has no every-tenth fallback after completion.
        self.min_share=.03 if min_share is None else float(min_share)
        if self.step<=0 or self.target<1 or not 0<=self.min_share<=1:raise ValueError('Invalid Live build settings')
        self.keys=codes(np.empty((0,3)),self.step);self.best=np.empty(0,np.uint8);self.support=np.empty(0,np.uint16)
        self.seen=0;self.kept=0;self.refused=0;self.last={}
    def observe(self,points,depth_mm,ok=True,force=False):
        self.seen+=1
        if not ok:
            self.refused+=1;self.last=dict(keep=False,reason='Tracking lost: paused',new=0,upgrades=0,support=0);return self.last
        k=codes(points,self.step);levels=distance_levels(depth_mm,self.profile)
        key,inv=np.unique(k,return_inverse=True);band=np.zeros(len(key),np.uint8);np.maximum.at(band,inv,levels)
        if not len(key):self.last=dict(keep=False,reason='No part surface',new=0,upgrades=0,support=0);return self.last
        indices=np.searchsorted(self.keys,key);present=indices<len(self.keys)
        ids=np.flatnonzero(present);present[ids]=self.keys[indices[ids]]==key[ids]
        oldband=np.zeros(len(key),np.uint8);support=np.zeros(len(key),np.uint16)
        oldband[present]=self.best[indices[present]];support[present]=self.support[indices[present]]
        new=~present;upgrade=present&(band>oldband);add_support=(band>=1)&(support<self.target)
        helps=new|upgrade|add_support;share=float(helps.mean());keep=bool(force or (helps.any() and share>=self.min_share))
        self.last=dict(keep=keep,reason='Adds surface or quality' if keep else 'Averaging only',new=int(new.sum()),upgrades=int(upgrade.sum()),support=int(add_support.sum()),share=share)
        if keep:
            combined=np.concatenate([self.keys,key[new]]);order=np.argsort(combined)
            self.best=np.concatenate([self.best,band[new]])[order];self.support=np.concatenate([self.support,np.zeros(int(new.sum()),np.uint16)])[order];self.keys=combined[order]
            indices=np.searchsorted(self.keys,key);self.best[indices]=np.maximum(self.best[indices],band)
            self.support[indices]=np.minimum(self.target,self.support[indices]+(band>=1).astype(np.uint16));self.kept+=1
        return self.last
    def levels(self,points):
        k=codes(points,self.step);indices=np.searchsorted(self.keys,k);valid=indices<len(self.keys);ids=np.flatnonzero(valid);valid[ids]=self.keys[indices[ids]]==k[ids]
        band=np.full(len(k),255,np.uint8);quality=np.zeros(len(k),np.uint8)
        band[valid]=self.best[indices[valid]];quality[valid]=np.where(self.support[indices[valid]]>=self.target,2,np.where(self.support[indices[valid]]>0,1,0))
        return band,quality
    def state(self):
        done=(self.support>=self.target)&(self.best>=1);fraction=float(done.mean()) if len(done) else 0.
        return dict(seen=self.seen,kept=self.kept,refused=self.refused,voxels=len(self.keys),done_share=fraction,step_mm=self.step,support_target=self.target,min_share=self.min_share,pass_done=bool(len(done) and fraction>=1-self.min_share),rule='Keep when new surface, better band or unfinished good-distance support reaches the chosen share. Every accepted frame still fuses into TSDF.')
    def save(self,folder):
        p=Path(folder)/'live-build.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(self.state(),indent=2));tmp.replace(p)


def frame_points(depth,cal,T,mask=None):
    valid=depth>0
    if mask is not None:valid &= mask
    v,u=np.nonzero(valid);z=depth[v,u].astype(float)*.1
    p=np.column_stack([(u-cal[2])*z/cal[0],(v-cal[3])*z/cal[1],z]);return p@T[:3,:3].T+T[:3,3]*1000,z


class RecentFusion:
    """CPU snapshots before recent integrations, bounded by the host's checkpoint count.

    Restore subtracts all newer contributions, including frames never saved. No
    inverse Open3D integration or approximate negative weights are used.
    """
    def __init__(self,seconds,max_frames):self.seconds=seconds;self.frames=deque(maxlen=max_frames)
    def before(self,trk,t,entries,stamps):
        from app2.scan.tracking import grid_snapshot
        snap=grid_snapshot(trk)
        if snap is None:return
        self.frames.append((t,snap,dict(T=trk.T.copy(),fused=trk.fused,n=getattr(trk,'n',0)),len(entries),len(stamps)))
        while self.frames and t-self.frames[0][0]>self.seconds:self.frames.popleft()
    def restore(self,trk):
        if not self.frames:return None
        from app2.scan.tracking import grid_restore,settle_pose
        row=self.frames[0];_,snap,state,_,_=row;grid_restore(trk,snap);settle_pose(trk,state['T'],state['fused']);trk.n=state['n'];self.frames.clear();return row
