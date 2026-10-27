# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact local live TSDF cache. Placement changes never resample this volume."""
from pathlib import Path
import hashlib,json,os,zipfile
import numpy as np

VERSION=1
NAME='live-volume.npz'


def identity(side):
    from app2.scan import fmt
    folder=Path(side.folder);h=hashlib.sha256();h.update(str(VERSION).encode())
    root=Path(__file__).resolve().parents[2]
    for name in ('track.py','app2/scan/tracking.py','app2/scan/live_build.py','app2/scan/live_cache.py'):
        h.update(hashlib.sha256((root/name).read_bytes()).digest())
    h.update((folder/'param/Pl.bin').read_bytes());h.update(json.dumps(side.scan_profile.get('live_build',{}),sort_keys=True).encode())
    for ps,idx,M in side.entries:
        name=fmt.frame_name(ps,idx);h.update(name.encode());h.update(np.asarray(M,dtype='<f8').tobytes())
        p=folder/'cache'/(name+'.dph');h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def snapshot(side,trk,field):
    """The grid and field as they are now, copied to the CPU (fast, on the tracker's thread); write_snapshot puts it on
    disk, which can happen later on another thread (2026-10-24: Pause no longer waits for the compression)."""
    if not hasattr(trk,'vbg'):return None
    from app2.scan.tracking import grid_snapshot
    keys,values=grid_snapshot(trk)
    meta=dict(version=VERSION,key=identity(side),T=trk.T.tolist(),fused=trk.fused,n=trk.n,field=field.state(),profile=field.profile)
    return dict(path=Path(side.folder)/NAME,metadata=json.dumps(meta),keys=keys.numpy().copy(),tsdf=values[0].numpy().copy(),weight=values[1].numpy().copy(),
                field_keys=np.array(field.keys,copy=True),field_best=np.array(field.best,copy=True),field_support=np.array(field.support,copy=True))


def write_snapshot(snap):
    path=snap['path'];tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**{k:v for k,v in snap.items() if k!='path'})
    os.replace(tmp,path);return True


def write(side,trk,field):
    snap=snapshot(side,trk,field)
    return False if snap is None else write_snapshot(snap)


def read(side):
    path=Path(side.folder)/NAME
    try:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta.get('version')!=VERSION or meta.get('key')!=identity(side):return None
            out={key:np.array(data[key]) for key in ('keys','tsdf','weight','field_keys','field_best','field_support')}
        if out['keys'].ndim!=2 or out['keys'].shape[1]!=3 or len(out['tsdf'])!=len(out['keys']) or out['tsdf'].shape!=out['weight'].shape:return None
        if out['field_best'].shape!=out['field_support'].shape or len(out['field_keys'])!=len(out['field_best']):return None
        if not np.isfinite(out['tsdf']).all() or not np.isfinite(out['weight']).all():return None
        out['metadata']=meta;return out
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile,EOFError):return None


def restore_field(data):
    from app2.scan.live_build import QualityField
    meta=data['metadata'];state=meta['field'];field=QualityField(state['step_mm'],meta['profile'],state['support_target'],state['min_share'])
    field.keys=data['field_keys'];field.best=data['field_best'];field.support=data['field_support'];field.seen=state['seen'];field.kept=state['kept'];field.refused=state['refused'];return field


def restore_tracker(trk,data):
    import open3d.core as o3c
    from app2.scan.tracking import grid_restore,settle_pose
    grid_restore(trk,(o3c.Tensor(data['keys']),[o3c.Tensor(data['tsdf']),o3c.Tensor(data['weight'])]));meta=data['metadata'];settle_pose(trk,np.asarray(meta['T']),meta['fused']);trk.n=meta['n']
