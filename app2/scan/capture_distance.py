# SPDX-License-Identifier: AGPL-3.0-or-later
"""Asynchronous capture-distance evidence in side coordinates, separate from fusion."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from app2.scan.range_guide import distance_levels,measured_profile,distance_zones,UNSEEN_RGB

NAME='capture-distance.npz'
UNKNOWN=np.uint8(255)
_CAPTURE_IDLE=None
_CAPTURE_CLOCK=None

def wait_for_budget():
    """Yield until the scan has had an idle gap at least as long as its last frame.

    Replay without idle gaps catches up at Pause. A real stream with spare time
    can update the colors during capture. The scan thread never waits for this.
    """
    if _CAPTURE_IDLE is None:return
    import time
    while True:
        _CAPTURE_IDLE.wait()
        delay=_CAPTURE_CLOCK[1]-(time.monotonic()-_CAPTURE_CLOCK[0])
        if delay>0:time.sleep(delay)
        if _CAPTURE_IDLE.is_set() and time.monotonic()-_CAPTURE_CLOCK[0]>=_CAPTURE_CLOCK[1]:return



def voxel_size():
    # Read the existing tracker default without loading a second CUDA/Open3D
    # runtime in the background process. Fail if that default stops being literal.
    import ast
    module=ast.parse((Path(__file__).resolve().parents[2]/'track.py').read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Tracker')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    return float(ast.literal_eval(defaults['voxel_mm']))


def query_batch_size():
    """Use the tracker's existing point budget without loading its GPU runtime."""
    return _query_batch_size()


from functools import lru_cache
@lru_cache(maxsize=1)
def _query_batch_size():
    import ast
    module=ast.parse((Path(__file__).resolve().parents[2]/'track.py').read_text())
    for node in module.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REGISTRATION_MIN_POINTS' for t in node.targets):return int(ast.literal_eval(node.value))
    raise RuntimeError('The tracker point budget is not a literal.')


MERGE_POINTS=4_000_000      # points held before they are folded into the voxel set (about 100 MB of keys)


def merge(keys,levels,new_keys,new_levels):
    keys=np.concatenate([keys,new_keys]);levels=np.concatenate([levels,new_levels])
    # Pack three signed voxel coordinates into uint64 (21 bits per axis).
    # This speeds sorting without a lossy hash. Out-of-domain scenes fall back
    # to full coordinates, so no position can silently collide.
    bits=np.iinfo(np.uint64).bits//3;bias=1<<(bits-1);mask=(1<<bits)-1
    if np.all((keys>=-bias)&(keys<bias)):
        q=(keys+bias).astype(np.uint64);code=(q[:,0]<<np.uint64(2*bits))|(q[:,1]<<np.uint64(bits))|q[:,2]
        unique,inverse=np.unique(code,return_inverse=True);best=np.zeros(len(unique),np.uint8);np.maximum.at(best,inverse,levels)
        xyz=np.column_stack([(unique>>np.uint64(2*bits))&np.uint64(mask),(unique>>np.uint64(bits))&np.uint64(mask),unique&np.uint64(mask)]).astype(np.int64)-bias
        return xyz,best
    unique,inverse=np.unique(keys,axis=0,return_inverse=True);best=np.zeros(len(unique),np.uint8);np.maximum.at(best,inverse,levels)
    return unique,best


def query(field,points):
    """Only the observed voxel gets a color. Missing evidence remains gray."""
    keys,levels,step=field;out=np.full(len(points),UNKNOWN,np.uint8)
    if not len(keys) or not len(points):return out
    points_keys=np.floor(np.asarray(points)/step).astype(np.int64);bits=np.iinfo(np.uint64).bits//3;bias=1<<(bits-1)
    if np.all((keys>=-bias)&(keys<bias)) and np.all((points_keys>=-bias)&(points_keys<bias)):
        def packed(a):
            a=(a+bias).astype(np.uint64);return (a[:,0]<<np.uint64(2*bits))|(a[:,1]<<np.uint64(bits))|a[:,2]
        k=packed(keys);q=packed(points_keys)
    else:
        dt=np.dtype([('x','<i8'),('y','<i8'),('z','<i8')]);k=np.ascontiguousarray(keys,np.int64).view(dt).ravel();q=np.ascontiguousarray(points_keys,np.int64).view(dt).ravel()
    order=np.argsort(k) if np.any(k[1:]<k[:-1]) else None;sorted_keys=k if order is None else k[order]
    i=np.searchsorted(sorted_keys,q);valid=i<len(k);indices=np.flatnonzero(valid);valid[indices]=sorted_keys[i[indices]]==q[indices]
    out[valid]=levels[i[valid] if order is None else order[i[valid]]];return out


def surface_query(field,points):
    """Map reconstructed surfaces within one voxel diagonal of saved evidence.

    Two independently sampled surfaces can lie in neighboring voxels. Their
    combined quantization bound is one diagonal; farther points remain unknown.
    The best supported band wins, so existing bands never dilute.
    """
    out=query(field,points)
    if not len(points) or not len(field[0]):return out
    from scipy.spatial import cKDTree
    keys,levels,step=field;radius=np.sqrt(3)*step
    # Query one band at a time to bound memory. Better nearby evidence upgrades
    # an existing voxel too; it never replaces green with a poorer observation.
    for level in (2,1,0):
        missing=np.flatnonzero((out==UNKNOWN)|(out<level));support=keys[levels==level]
        if not len(missing) or not len(support):continue
        wait_for_budget();tree=cKDTree((support+.5)*step)
        for start in range(0,len(missing),query_batch_size()):
            wait_for_budget();part=missing[start:start+query_batch_size()]
            distance,_=tree.query(np.asarray(points)[part],distance_upper_bound=radius,workers=1)
            out[part[np.isfinite(distance)]]=level
    return out


def load(folder):
    with np.load(Path(folder)/NAME,allow_pickle=False) as data:return data['keys'],data['levels'],float(data['step'])


def build_side(folder,entries,cal,shape,mode,stamps=None):
    """Read saved kept frames on a background worker; upgrades use maximum band.

    A removed or changed pose rebuilds the field, so Undo cannot leave evidence
    from discarded frames. No extra GPU copies or tracker integration occurs.
    """
    from app2.scan.fmt import frame_name
    wait_for_budget()
    folder=Path(folder)
    from app2.scan.scan_mode import read as read_profile
    step=max(voxel_size(),float(read_profile(folder).get('voxel_mm',voxel_size())));stamp_map={x['frame']:x for x in stamps or []};records=[]
    for ps,idx,M in entries:
        name=frame_name(ps,idx);path=folder/'cache'/(name+'.dph')
        if not path.exists():continue
        bands=stamp_map.get(name,{}).get('capture_bands');profile=dict(distance_bands=bands) if bands else measured_profile(mode)
        identity=hashlib.sha256(b'camera-convention-v2'+np.asarray(M,float).tobytes()+str(path.stat().st_mtime_ns).encode()+json.dumps(dict(profile=profile,bands=distance_zones(profile)),sort_keys=True).encode()).hexdigest();records.append((name,M,profile,identity))
    identities=[x[3] for x in records];keys=np.empty((0,3),np.int64);levels=np.empty(0,np.uint8);done=0
    try:
        metadata=json.loads((folder/'capture-distance.json').read_text());previous=metadata['frames']
        if identities[:len(previous)]==previous and metadata['step_mm']==step:
            keys,levels,_=load(folder);done=len(previous)
    except (OSError,ValueError,KeyError):pass
    if done==len(records) and done:return keys,levels,step
    incoming_keys=[];incoming_levels=[]
    pending=0
    for name,M,profile,identity in records[done:]:
        wait_for_budget()
        d=np.fromfile(folder/'cache'/(name+'.dph'),np.uint16).reshape(shape);v,u=np.nonzero(d);z=d[v,u].astype(float)*.1
        P=np.column_stack([(u-cal[2])*z/cal[0],(v-cal[3])*z/cal[1],z])
        # Exported poses use the scanner's flipped camera axes. Match fuse.py.
        M=np.asarray(M)@np.diag([1.,-1.,-1.,1.]);P=P@M[:3,:3].T+M[:3,3]
        incoming_keys.append(np.floor(P/step).astype(np.int64));incoming_levels.append(distance_levels(z,profile));pending+=len(z)
        if pending>=MERGE_POINTS:                                 # fold into the voxel set as we go: every frame's points kept
            keys,levels=merge(keys,levels,np.concatenate(incoming_keys),np.concatenate(incoming_levels))   # until the end took
            incoming_keys=[];incoming_levels=[];pending=0        # 5 MB a frame, 20+ GB on a 1,800-frame side (2026-10-22)
    wait_for_budget()
    if incoming_keys:keys,levels=merge(keys,levels,np.concatenate(incoming_keys),np.concatenate(incoming_levels))
    wait_for_budget()
    tmp=folder/(NAME+'.tmp')
    with tmp.open('wb') as f:np.savez(f,keys=keys,levels=levels,step=step)
    os.replace(tmp,folder/NAME);metadata=dict(version=2,step_mm=step,frames=identities,tracked_frames=len(records),rule='max band per tracker-size voxel; 0 off, 1 OK, 2 best; model mapping limited to one voxel diagonal; saved frame bands preferred, shipped mode profile for older frames',mode=mode)
    tmp=folder/'capture-distance.json.tmp';tmp.write_text(json.dumps(metadata,indent=2));os.replace(tmp,folder/'capture-distance.json');return keys,levels,step


def project_levels(project,points,side_folder=None):
    """Combine placed sides without crossing an unobserved voxel."""
    if side_folder:return surface_query(load(side_folder),points)
    from model_frames import read
    out=np.full(len(points),UNKNOWN,np.uint8)
    for folder,M in read(str(project)).items():
        path=Path(project)/'data'/Path(folder).name
        try:field=load(path)
        except (OSError,ValueError):continue
        inv=np.linalg.inv(M);local=np.asarray(points)@inv[:3,:3].T+inv[:3,3];level=surface_query(field,local);valid=level!=UNKNOWN
        out[valid]=np.where(out[valid]==UNKNOWN,level[valid],np.maximum(out[valid],level[valid]))
    return out


def rgb(levels):
    from app2.scan.range_guide import BAND_RGB
    colors=np.tile(np.uint8(UNSEEN_RGB),(len(levels),1))
    for i,key in enumerate(('off','ok','best')):colors[np.asarray(levels)==i]=BAND_RGB[key]
    return colors


def color_snapshot(folder,entries,cal,shape,mode,stamps,points,prior):
    levels=surface_query(build_side(folder,entries,cal,shape,mode,stamps),points)
    if prior is not None:
        valid=prior!=UNKNOWN;levels[valid]=np.where(levels[valid]==UNKNOWN,prior[valid],np.maximum(levels[valid],prior[valid]))
    return levels


def background_priority(idle=None,clock=None):
    global _CAPTURE_IDLE,_CAPTURE_CLOCK
    _CAPTURE_IDLE=idle;_CAPTURE_CLOCK=clock
    import os
    try:os.nice(10)
    except OSError:pass
    # Load native query code during side preparation, before recording starts.
    from scipy.spatial import cKDTree


def ensure_side(folder):
    """Regenerate the disposable field from authoritative per-frame band stamps."""
    from app2.scan import fmt
    from app2.scan.scan_mode import read
    from fuse import read_calib
    folder=Path(folder);track=folder/'cache'/fmt.TRACK_JSON
    if not track.exists():return
    data=json.loads(track.read_text());stamps=data.get('stamps',[])
    saved=fmt.read_side(str(folder))
    if saved is None:return
    profile=read(str(folder))
    if not any(x.get('capture_bands') for x in stamps) and not profile.get('mode'):return
    cal=read_calib(str(folder/'param/Pl.bin'))
    build_side(folder,saved[0],cal,(fmt.H,fmt.W),(profile or {}).get('mode','near'),stamps)
