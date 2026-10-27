# SPDX-License-Identifier: AGPL-3.0-or-later
"""Software range and sampling presets derived from the current depth camera."""
import json
from pathlib import Path
import numpy as np


def read(folder):
    try:
        value=json.loads((Path(folder)/'property.rvproj').read_text()).get('pointyoink_scan',{})
        if not isinstance(value,dict) or value.get('mode') not in ('near','far'):return {}
        return {k:v for k,v in value.items() if k not in ('framing_lock','framing_locks')}
    except (OSError,ValueError,TypeError):return {}


def write(folder,profile):
    path=Path(folder)/'property.rvproj'
    try:data=json.loads(path.read_text())
    except (OSError,ValueError):data={}
    data['pointyoink_scan']={k:v for k,v in profile.items() if k not in ('framing_lock','framing_locks')}
    temp=path.with_suffix('.mode.tmp');temp.write_text(json.dumps(data,indent=1));temp.replace(path)


def profile(mode,cal,depth=None,near=100,far=450):
    if mode not in ('near','far'):raise ValueError('Choose Small part or Large part.')
    result=dict(mode=mode,near=int(near),far=int(far),hardware_switch=False)
    if mode=='near':return result
    z=np.asarray(depth) if depth is not None else np.zeros(0)
    z=z[z>0].astype(float)*.1
    if not len(z):
        from app2.scan.tracking import LIVE_MAX_DEPTH_MM
        result['far']=int(LIVE_MAX_DEPTH_MM)
        result['derived_from']='existing live depth support; awaiting measured depth'
        return result
    # Reuse Auto Range's measured depth percentiles and its existing padding.
    from app2.scan.assist import AutoRange
    auto=AutoRange();rng=auto.update(np.asarray(depth))
    if rng:
        from app2.scan.tracking import LIVE_MAX_DEPTH_MM
        result.update(near=min(int(near),rng[0]),far=min(int(LIVE_MAX_DEPTH_MM),max(int(far),rng[1])))
    distance=float(np.median(z));pitch=distance/min(float(cal[0]),float(cal[1]))
    from scanner.stages import params as PR
    spec=next(p for p in PR.COMMON['side.fuse'] if p.name=='voxel')
    image=np.asarray(depth,float)*.1
    valid=(image[:,2:]>0)&(image[:,1:-1]>0)&(image[:,:-2]>0)
    residual=(image[:,2:]-2*image[:,1:-1]+image[:,:-2])[valid]
    sigma=float(np.median(np.abs(residual-np.median(residual)))*1.482602218505602/np.sqrt(6)) if len(residual) else 0.
    sigma=max(sigma,.1/np.sqrt(12))   # Uniform depth-count quantization, not an assumed sensor precision.
    spacing=min(spec.hi,max(spec.default,pitch+3*sigma))
    result.update(distance_mm=distance,pixel_pitch_mm=pitch,depth_noise_sigma_mm=sigma,voxel_mm=spacing,
                  derived_from='depth median, calibrated focal length, and robust depth second differences')
    return result
