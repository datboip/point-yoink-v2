# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bed reference from accepted model poses, never from refused frames."""
from pathlib import Path
import numpy as np
from scanner import bed


def difference(a,b):
    delta=np.linalg.inv(a)@b
    angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
    return float(np.linalg.norm(delta[:3,3])),angle


class Reference:
    def __init__(self):self.samples=[];self.value=None

    def add(self,B):
        if not np.isfinite(B).all():return
        self.samples.append(np.array(B,copy=True));self.samples=self.samples[-bed.ACCEPT_WINDOW:]
        groups=[]
        for candidate in self.samples:
            group=[x for x in self.samples if difference(candidate,x)[0]<=bed.AGREE_CENTRE and difference(candidate,x)[1]<=bed.AGREE_DEG]
            groups.append(group)
        group=max(groups,key=len)
        self.value=None
        if len(group)<bed.ACCEPT_HITS:return
        # A second equally supported, incompatible placement is not a seed.
        if any(len(other)==len(group) and (difference(group[0],other[0])[0]>bed.AGREE_CENTRE or difference(group[0],other[0])[1]>bed.AGREE_DEG) for other in groups):return
        mean=np.mean(group,axis=0);u,_,v=np.linalg.svd(mean[:3,:3]);rotation=u@v
        if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
        mean[:3,:3]=rotation;mean[3]=[0,0,0,1];self.value=mean


def restore(side,cal,model,guide):
    """Bounded reads from accepted entries when reloading a saved model."""
    from app2.scan import fmt,preview
    from app2.scan.color_bed import solve_pose
    entries=side.entries
    if not entries or not bed.has_dots(model):return
    indices=np.unique(np.linspace(0,len(entries)-1,min(len(entries),bed.ACCEPT_WINDOW),dtype=int))
    reference=Reference()
    for k in indices:
        ps,idx,M=entries[k];p=Path(side.cache)/fmt.frame_name(ps,idx)
        try:
            d=np.fromfile(p.with_suffix('.dph'),np.uint16).reshape(fmt.H,fmt.W)
            ir=np.fromfile(p.with_suffix('.irl'),np.uint8).reshape(d.shape)
        except (OSError,ValueError):continue
        hit=bed.find_bed(d,ir,cal,model,prior=model.camera)
        if hit is None or hit.height_from=='prior-only':continue
        reference.add(M@bed.F@hit.B_cam)
        jpeg=p.with_suffix('.img')
        if jpeg.exists() and guide.correction is None:
            color=solve_pose(preview.decoded(jpeg.read_bytes()),preview.RGB_CAL,(hit.n,hit.off,hit.centre),model)
            guide.paired(hit,color)
    if reference.value is not None:
        guide.reference=reference.value;guide.reference_samples=len(reference.samples)
