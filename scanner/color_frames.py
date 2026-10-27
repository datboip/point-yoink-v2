# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibrated saved RGB frames and weighted surface color observations."""
from pathlib import Path
import numpy as np


def available(frames,has_images=None):
    frames=Path(frames);param=frames.parent/'param'
    return (param/'Prgb.bin').is_file() and (param/'LC_RT.bin').is_file() and (has_images if has_images is not None else next(frames.glob('*.img'),None) is not None)


def enabled(sets,mode='auto'):
    if mode=='off':return False
    ready=all(available(s[0]) for s in sets)
    if mode=='on' and not ready:raise ValueError('One or more active sides have no saved color frames or color calibration.')
    return ready


def linear(rgb):
    rgb=np.asarray(rgb,np.float32)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def srgb(rgb):
    rgb=np.clip(rgb,0,1)
    return np.where(rgb<=.0031308,rgb*12.92,1.055*rgb**(1/2.4)-.055)


def project(points,cal,image_size):
    K,R,T=cal;P=np.asarray(points)@R.T+T
    K=K.copy();K[0]*=image_size[0]/800.;K[1]*=image_size[1]/600.
    z=P[:,2];q=P@K.T
    uv=q[:,:2]/np.maximum(q[:,2:3],1e-9)
    return uv,z


def read_rgb(dph,half=True):
    """The frame's color photo. half: decoded at half size (1000x750 from 2000x1500, still finer than the 800x600
    depth); the JPEG decoder does that nearly for free and it was most of the color build time."""
    from PIL import Image
    path=Path(dph).with_suffix('.img')
    if not path.is_file():return None
    with Image.open(path) as im:
        if half:im.draft('RGB',(im.width//2,im.height//2))
        return np.asarray(im.convert('RGB'))


def calibration(dph):
    from fuse import read_rgb_calib
    return read_rgb_calib(str(Path(dph).parent.parent/'param'))


def sample(rgb,points,cal):
    import cv2
    uv,z=project(points,cal,(rgb.shape[1],rgb.shape[0]))
    valid=(z>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<rgb.shape[1]-1)&(uv[:,1]<rgb.shape[0]-1)
    out=np.empty((len(points),3),np.uint8)
    for start in range(0,len(points),30000):
        part=uv[start:start+30000].astype(np.float32)
        out[start:start+len(part)]=cv2.remap(rgb,part[:,0,None],part[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,3)
    return out,valid


def aligned(dph,depth,depth_cal,depth_scale=.1):
    rgb=read_rgb(dph);cal=calibration(dph)
    if rgb is None or cal is None:return np.zeros((*depth.shape,3),np.uint8)
    fyx=np.indices(depth.shape,dtype=np.float32);z=depth.astype(np.float32)*depth_scale
    fx,fy,cx,cy=depth_cal
    points=np.stack(((fyx[1]-cx)*z/fx,(fyx[0]-cy)*z/fy,z),axis=-1).reshape(-1,3)
    values,valid=sample(rgb,points,cal);valid&=depth.reshape(-1)>0
    values[~valid]=0
    return np.ascontiguousarray(values.reshape(*depth.shape,3))


def white_gains(colors,keep=None):
    """Gray-world white balance on linear RGB: per-channel gains that make the typical mid-tone surface neutral.
    Very dark and very bright points (shadows, glare) are left out; gains stay within 0.6..1.6 so a truly colored
    part keeps most of its color."""
    c=colors if keep is None else colors[keep]
    if len(c)<500:return np.ones(3)
    lum=c@np.array([.2126,.7152,.0722]);lo,hi=np.percentile(lum,[15,90]);mid=c[(lum>lo)&(lum<hi)]
    if len(mid)<200:return np.ones(3)
    med=np.median(mid,axis=0)
    if med.min()<=1e-4:return np.ones(3)
    return np.clip(med.mean()/med,.6,1.6)


PAINT_FRAMES = 120          # per side: each spot is still seen by dozens of frames; all 431 took 52 s on one side


def spread(jobs,per_side=PAINT_FRAMES):
    """At most per_side frames from each side folder, evenly spread over the turn, in their original order."""
    by={}
    for i,j in enumerate(jobs):by.setdefault(str(Path(j[0]).parent),[]).append(i)
    keep=set()
    for ids in by.values():
        keep.update(ids if len(ids)<=per_side else [ids[int(k)] for k in np.linspace(0,len(ids)-1,per_side)])
    return [j for i,j in enumerate(jobs) if i in keep]


def paint(mesh,jobs,depth_scale=.1,tolerance_mm=1.8,balance=True,emit=None,white=True):
    """Observe visible vertices in linear RGB; favor clear, close, face-on views."""
    import cv2,open3d as o3d
    jobs=spread(list(jobs))
    from fuse import read_pose,frame_key
    V=np.asarray(mesh.vertices);N=np.asarray(mesh.vertex_normals)
    if len(N)!=len(V):mesh.compute_vertex_normals();N=np.asarray(mesh.vertex_normals)
    sums=np.zeros((len(V),3),np.float64);weights=np.zeros(len(V),np.float64);used=0;gains=[]
    flip=np.diag([1.,-1.,-1.,1.])
    for dph,poses,T,depth_cal,plane,size,own,deletes in jobs:
        rgb=read_rgb(dph);cal=calibration(dph)
        if rgb is None or cal is None:continue
        pose=poses.get(frame_key(dph))
        if pose is None and own and poses:continue
        if pose is None:
            if not Path(dph).with_suffix('.inf').is_file():continue
            pose=read_pose(str(Path(dph).with_suffix('.inf')))
        M=(T@pose if T is not None else pose)@flip
        points=(V-M[:3,3])@M[:3,:3];z=points[:,2];fx,fy,cx,cy=depth_cal;w,h=size
        u=np.rint(points[:,0]/np.maximum(z,1e-6)*fx+cx).astype(np.int32)
        v=np.rint(points[:,1]/np.maximum(z,1e-6)*fy+cy).astype(np.int32)
        valid=(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        ids=np.flatnonzero(valid)
        depth=np.fromfile(dph,np.uint16)
        if depth.size!=w*h:continue
        dz=depth.reshape(h,w)[v[ids],u[ids]].astype(np.float32)*depth_scale
        ids=ids[(dz>0)&(np.abs(dz-z[ids])<=tolerance_mm)]
        if not len(ids):continue
        values,visible=sample(rgb,points[ids],cal);ids=ids[visible];values=values[visible]
        if not len(ids):continue
        normals=N[ids]@M[:3,:3]
        cosine=np.abs((normals*points[ids]).sum(1))/np.maximum(np.linalg.norm(points[ids],axis=1),1e-8)
        gray=cv2.cvtColor(cv2.resize(rgb,(320,240)),cv2.COLOR_RGB2GRAY)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_32F).var());quality=np.clip(sharpness/100.,.15,1.)
        weight=np.clip(cosine,0,1)**2*np.minimum(4.,(300./np.maximum(z[ids],1.))**2)*quality
        colors=linear(values/255.);gain=np.ones(3)
        overlap=weights[ids]>.5
        if balance and overlap.sum()>=128:
            previous=sums[ids[overlap]]/weights[ids[overlap],None];current=colors[overlap]
            ratios=np.where((current>.015)&(previous>.015),previous/np.maximum(current,1e-6),np.nan)
            for channel in range(3):
                r=ratios[:,channel];r=r[np.isfinite(r)]
                if len(r)>=128:
                    q=np.percentile(r,[25,50,75])
                    if q[2]-q[0]<.25 and abs(q[1]-1)>.08:gain[channel]=np.clip(q[1],.8,1.25)
        colors=np.clip(colors*gain,0,1);sums[ids]+=colors*weight[:,None];weights[ids]+=weight
        used+=1;gains.append(gain.tolist())
        if emit and used%20==0:emit('color',done=used,total=len(jobs))
    good=weights>1e-6;colors=np.full((len(V),3),.5)
    lin=sums[good]/weights[good,None];wb=white_gains(lin) if white else np.ones(3)
    colors[good]=srgb(np.clip(lin*wb,0,1))
    if good.any() and (~good).any():
        from scipy.spatial import cKDTree
        distance,index=cKDTree(V[good]).query(V[~good],workers=1)
        missing=np.flatnonzero(~good);near=distance<=2*tolerance_mm
        colors[missing[near]]=colors[good][index[near]]
    mesh.vertex_colors=o3d.utility.Vector3dVector(colors)
    return dict(frames=used,coverage=float(good.mean()) if len(good) else 0.,gain_min=np.min(gains,axis=0).tolist() if gains else [1]*3,gain_max=np.max(gains,axis=0).tolist() if gains else [1]*3,white_gains=[round(float(x),3) for x in wb] if good.any() else [1]*3,linear_rgb=True)
