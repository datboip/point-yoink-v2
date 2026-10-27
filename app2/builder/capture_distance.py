# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read capture-distance evidence for the selected Build artifact and placements."""
from pathlib import Path
import hashlib,json,numpy as np
from app2.scan.capture_distance import build_side,load,query,rgb,UNKNOWN,ensure_side,surface_query


def colors(project,src,placements=None,side=None):
    from app2.builder.view_data import load_file
    from fuse import read_calib,read_global_poses
    root=Path(project);data=load_file(str(src));P=data['V'][data['F']].mean(axis=1) if 'F' in data else data.get('V',data.get('P'));out=np.full(len(P),UNKNOWN,np.uint8)
    if side:placements={side:np.eye(4)}
    if placements is None:
        from model_frames import read
        placements=read(str(root))
    for folder,M in placements.items():
        path=root/'data'/Path(folder).name
        ensure_side(path)
        if not (path/'capture-distance.npz').exists():continue
        field=load(path);inv=np.linalg.inv(M);local=P@inv[:3,:3].T+inv[:3,3];level=surface_query(field,local);valid=level!=UNKNOWN
        out[valid]=np.where(out[valid]==UNKNOWN,level[valid],np.maximum(out[valid],level[valid]))
    # Older sides with an explicit saved mode can be reconstructed from their
    # saved distances using the shipped profile. No mode is guessed. Facets
    # query their centers so a guessed hole stays gray without saved support.
    return rgb(out),dict(best=int((out==2).sum()),ok=int((out==1).sum()),off=int((out==0).sum()),unmeasured=int((out==UNKNOWN).sum()))
