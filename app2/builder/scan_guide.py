# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the shared missing/rough map and its saved-file scan guide. No device access."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np


def quality_colors(kind):
    kind=np.asarray(kind)
    colors=np.tile([70,215,120],(len(kind),1)).astype(np.uint8)
    colors[kind==1]=[255,70,50];colors[kind==2]=[180,90,230]
    return colors


def selected_lineup(project,lineup_file=None,frames=None):
    """Read the selected graph artifact, or match the published model's stored frame mapping."""
    root=Path(project)
    if lineup_file is not None:
        try:return json.loads(Path(lineup_file).read_text()).get('sides',[]),str(lineup_file)
        except (OSError,ValueError,TypeError):return [],None
    if frames is None:
        import model_frames
        frames=model_frames.read(str(root))
    if not frames:return [],None
    for path in sorted((root/'stages'/'nodes'/'lineup.adjust').glob('*/sides.json')):
        try:rows=json.loads(path.read_text()).get('sides',[])
        except (OSError,ValueError,TypeError):continue
        by_id={row.get('id'):row for row in rows}
        if all(k in by_id and np.allclose(np.asarray(by_id[k]['T']),np.asarray(M),atol=1e-5) for k,M in frames.items()):
            return [by_id[k] for k in frames],str(path)
    return [],None


def placement_text(result):
    """State optical-center height and distance with their physical references."""
    if not result.get('bed'):return result
    n,d=result['bed'];n=np.asarray(n,float);norm=np.linalg.norm(n)
    from scanner.scan_guide import RANK_COLORS
    for i,p in enumerate(result.get('poses',[])):
        p['color']=RANK_COLORS[int(p.get('rank',i+1))-1]
        eye=np.asarray(p['T'])[:3,3]
        height=float((eye@n+d)/norm);distance=float(np.linalg.norm(eye-np.asarray(result['center'])))
        p['height_above_bed_mm']=height;p['distance_from_center_mm']=distance
        placement=f'Camera {height:.0f} mm above the bed, {distance:.0f} mm from the part center. '
        if result['mode']=='handheld':placement+='Model position (%g, %g, %g) mm. ' % tuple(np.round(eye,1))
        p['text']=placement+p['text']
    return result


def guide(project, src, mode='turntable', distance=None, lineup_file=None, frames=None, live=False):
    import open3d as o3d
    from scanner.quality_map import quality_map, vertex_normals, vertex_areas
    from scanner.scan_guide import Lens, Reachability, plan, axis_from_poses
    from fuse import read_global_poses
    root=Path(project); path=Path(src); st=path.stat(); F=np.diag([1.,-1.,-1.,1.])
    if distance is None:
        config=Path(os.environ.get('POINTYOINK_CONFIG') or Path.home()/'.config/pointyoink/config.json')
        try:
            saved=json.loads(config.read_text()).get('scan',{})
            distance=(float(saved['near']),float(saved['far']))
        except (OSError,ValueError,KeyError,TypeError): distance=(100.,450.)
    distance=tuple(sorted(distance))
    params=sorted(root.glob('data/*/param/Pl.bin'))
    lineup,lineup_path=selected_lineup(root,lineup_file,frames)
    metadata_files=[Path(lineup_path)] if lineup_path else []
    metadata_files+=list(root.glob('data/*/table_plane.json'))+list(root.glob('data/*/cache/*register_pose.pose'))
    metadata_files+=list(root.glob('data/*/param/*.bin'))
    metadata=b''.join(str(p.relative_to(root)).encode()+p.read_bytes() for p in sorted(metadata_files))
    key=hashlib.sha256(str((9 if live else 8,st.st_size,st.st_mtime_ns,mode,distance,lineup_path)).encode()+metadata).hexdigest()[:16]
    out=root/'build'/('scan-guide-'+key+'.json'); rgb=out.with_suffix('.npy')
    if out.exists() and rgb.exists(): return dict(json.loads(out.read_text()),rgb=str(rgb))
    m=o3d.io.read_triangle_mesh(str(path)); V=np.asarray(m.vertices); T=np.asarray(m.triangles)
    if not len(T): raise ValueError('Build a mesh first to find missing and rough spots')
    q=quality_map(V,T); N=vertex_normals(V,T); A=vertex_areas(V,T)
    colors=quality_colors(q['kind'])
    result=dict(poses=[],needs_flip=[],unreachable=[],remaining=[],mode=mode,spots=len(q['spots']),missing_vertices=int((q['kind']==1).sum()),rough_vertices=int((q['kind']==2).sum()),surface_mm2=float(A[q['kind']>0].sum()),model=str(path),axis=[0,1,0],center=V.mean(0).tolist(),lens={},reason='')
    def finish():
        out.parent.mkdir(parents=True,exist_ok=True); np.save(rgb,colors); out.write_text(json.dumps(result,indent=2))
        return dict(result,rgb=str(rgb))
    if not params:
        result['reason']='Saved scanner calibration is needed to suggest positions';return finish()
    saved={s.get('id'):s for s in lineup}; first=next(iter(saved.values()),{})
    param=params[0].parent
    if first.get('calib'):param=(root/first['calib']).parent
    size=tuple(first.get('size',(400,300)))
    if not first:
        try:
            sp=json.loads((param.parent/'property.rvproj').read_text())['scan_param'];size=(int(sp['depth_width']),int(sp['depth_height']))
        except (OSError,ValueError,KeyError): pass
    try:lens=Lens.from_param(param,size=size,distance=distance)
    except (OSError,ValueError) as e:
        result['reason']='Saved stereo calibration is needed to suggest positions: '+str(e);return finish()
    if not lineup:
        result['reason']='Saved model placement is needed to suggest positions';return finish()
    bed=None;poses=[]
    for side in sorted(root.glob('data/*')):
        ss=saved.get(side.name,{}); M=np.asarray(ss.get('T',np.eye(4)),float)
        if not ss and len(params)>1:continue
        try:
            plane=ss.get('plane') or json.loads((side/'table_plane.json').read_text()); n=np.asarray(plane['n'],float)
            wn=M[:3,:3]@n;d=-float(plane['d'])-wn@M[:3,3]
            if not plane.get('keep_above',True):wn,d=-wn,-d
            bed=(wn,d)
        except (OSError,ValueError,KeyError): pass
        for name in ('pointyoink_register_pose.pose','global_register_pose.pose'):
            try:
                table=read_global_poses(str(side/'cache'/name))
                if table:
                    poses.extend(M@p@F for _,p in sorted(table.items())[:180]);break
            except (OSError,ValueError):pass
        if bed is not None:break
    if bed is None:
        result['reason']='Saved bed plane and placement are needed to suggest positions';return finish()
    axis,circle_center=axis_from_poses(poses,bed[0]);center=V.mean(0)
    if circle_center is not None:center=circle_center+axis*((center-circle_center)@axis)
    bad=np.flatnonzero(q['kind']>0)
    # Preserve opposing faces, normals and missing/rough labels in each millimeter cell.
    cells=np.column_stack((np.floor(V[bad]).astype(np.int64),np.rint(N[bad]*4).astype(np.int64),q['kind'][bad]))
    _,keep,inv=np.unique(cells,axis=0,return_index=True,return_inverse=True)
    idx=bad[keep]; weights=np.bincount(inv,weights=A[bad],minlength=len(keep))
    checker=Reachability(V,T,V[idx],N[idx],lens,bed=bed,axis=axis,center=center)
    groups=np.full(len(V),-1,np.int32)
    for k,spot in enumerate(q['spots']):groups[spot['idx']]=k
    checker.groups=groups[idx]
    result.update(plan(checker,weights,mode=mode))
    if live:
        # Associate the guide's full-mesh samples with stable quality spots once.
        result['live_samples'] = dict(vertex_ids=idx.tolist(), areas=weights.tolist(),
                                      spot_ids=groups[idx].tolist(), counts=np.bincount(inv).tolist(), kinds=q['kind'][idx].tolist(),
                                      spot_centers=[np.asarray(s['centre']).tolist() for s in q['spots']],
                                      spot_kinds=[s['kind'] for s in q['spots']])
    for p in result['poses']:p['indices']=[int(idx[i]) for i in p['indices']]
    for k in ('needs_flip','unreachable','remaining'):result[k]=[int(idx[i]) for i in result[k]]
    result.update(axis=axis.tolist(),center=center.tolist(),lens=vars(lens),bed=[bed[0].tolist(),float(bed[1])],sampled_vertices=len(idx),calibration='Q rectified second IR camera; LC_RT is RGB and is not used')
    placement_text(result)
    return finish()
