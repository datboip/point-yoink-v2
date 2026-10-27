# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retrack and rebuild an explicitly chosen Far side in a new project copy."""
from pathlib import Path
import argparse
import json
import os
import shutil
import tempfile
import time
import numpy as np


def restore_saved_color(source_folder, target_folder, plan):
    """Keep existing JPEGs and their recorded synchronization when poses change."""
    from app2.scan import fmt
    source_folder,target_folder=Path(source_folder),Path(target_folder)
    saved=fmt.read_side(str(target_folder))
    if saved is None:raise ValueError('The retracked side was not saved.')
    entries,stamps,_rgb,_model=saved
    if len(stamps)!=len(plan):raise ValueError('The replay did not retain every recorded frame.')
    count=0
    for row,item in zip(stamps,plan):
        original=item[3] or {};image=source_folder/'cache'/(item[2]+'.img')
        row.update({key:original[key] for key in fmt.COLOR_STAMP_KEYS if key in original})
        if image.exists():
            shutil.copy2(image,target_folder/'cache'/(row['frame']+'.img'));count+=1
        elif original.get('color_saved'):row['color_saved']=False
        right=source_folder/'cache'/(item[2]+'.irr')
        if right.exists():
            shutil.copy2(right,target_folder/'cache'/(row['frame']+'.irr'));row['right_ir_saved']=True
    fmt.write_side_report(str(target_folder/'cache'),entries,stamps)
    thumb=source_folder/'thumb_rgb.jpg'
    if thumb.exists():shutil.copy2(thumb,target_folder/'thumb_rgb.jpg')
    return count


def rebuild_copy(project, side, calibration, output, build=True, mesh_output=None):
    from app2.scan.calibration import load
    from app2.scan.session import ScanSession, ScanConfig
    from app2.scan.sources import ReplaySource, Frame
    from app2.scan.bed_adapter import ScannerBed
    from app2.scan import fmt
    from app2.scan.tracking import LIVE_MAX_DEPTH_MM
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError('Choose a new folder outside the source project.')
    replay = ReplaySource(str(project), sides=[side], lockstep=True)
    plan = replay.plan
    if not plan: raise ValueError('This side has no saved depth frames.')
    folder = Path(plan[0][1]); properties = json.loads((folder / 'property.rvproj').read_text())
    scan = properties.get('scan_param', {})
    width, height = int(scan.get('depth_width', fmt.W)), int(scan.get('depth_height', fmt.H))
    selected = load(calibration, 'far', (width, height))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project, output)
    started = time.monotonic()
    rows = []
    with tempfile.TemporaryDirectory(prefix='pointyoink-far-copy-', dir=output.parent) as scratch:
        work = Path(scratch) / 'project'; shutil.copytree(project, work)
        cfg = ScanConfig(str(work.parent), str(folder / 'param'), project=str(work),
                         near=0, far=int(LIVE_MAX_DEPTH_MM), auto_range=False, auto_gain=False,
                         w=width, h=height, save_color=False, below_bed=False, remove_base=False)
        cfg.scan_mode='far'; cfg.far_param=selected['folder'];cfg.capture_mode='continuous'
        def emit(kind, value):
            if kind == 'error': raise RuntimeError(value)
        session = ScanSession(cfg, replay, emit, bed=ScannerBed())
        try:
            session._open(); session._new_side(); session._go()
            for index, (_number, scan_folder, name, stamp) in enumerate(plan):
                cache = Path(scan_folder) / 'cache'
                raw = np.fromfile(cache / (name + '.dph'), np.uint16)
                if raw.size != width * height: raise ValueError('Saved depth size changed within this side.')
                ir_file = cache / (name + '.irl')
                ir = np.fromfile(ir_file, np.uint8) if ir_file.exists() else np.zeros(width * height, np.uint8)
                if ir.size != width * height:
                    # Exported lower-resolution depth can retain full-resolution IR.
                    import cv2
                    iw, ih = int(scan.get('ir_width', width)), int(scan.get('ir_height', height))
                    if ir.size != iw * ih: raise ValueError('Saved IR size does not match its metadata.')
                    ir = cv2.resize(ir.reshape(ih, iw), (width, height), interpolation=cv2.INTER_NEAREST)
                t = (stamp or {}).get('t')
                if not isinstance(t, (int, float)) or not np.isfinite(t):
                    raise ValueError('This side needs recorded depth timestamps to retrack.')
                before = time.monotonic(); session._drain_bed()
                session._frame(Frame(float(t), raw.reshape(height, width), ir.reshape(height, width), seq=index, meta={}))
                result = session.last
                rows.append(dict(index=index, ok=bool(result['ok']), seconds=time.monotonic()-before,
                                 marker_anchored=bool(result.get('marker_anchored')),
                                 registration_points=result.get('registration_points'),registration_stride=result.get('registration_stride'),
                                 refusal_reason=result.get('refusal_reason'),fit=float(result.get('fit',float('nan'))),rmse_mm=float(result.get('rmse',float('nan')))))
            replacement = Path(session.picked.folder)
            if mesh_output:
                import open3d as o3d
                mesh = session.host.trk.vbg.extract_triangle_mesh(weight_threshold=3.).to_legacy()
                mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 1000)
                o3d.io.write_triangle_mesh(str(mesh_output), mesh)
        finally:
            session._close(None)
        restored_images=restore_saved_color(folder,replacement,plan)
        if not any(row['ok'] for row in rows): raise ValueError('Large part replay kept no frames. The source was not changed.')
        target = output / 'data' / folder.name
        shutil.rmtree(target)
        shutil.copytree(replacement, target)
        prop = json.loads((target / 'property.rvproj').read_text())
        prop['guid'] = folder.name
        (target / 'property.rvproj').write_text(json.dumps(prop, indent=2)+'\n')
        record = dict(color_frames_preserved=restored_images,source_side=folder.name, source_calibration_sha256=__import__('hashlib').sha256(
                      (folder/'param/Pl.bin').read_bytes()).hexdigest(),
                      selected_calibration=selected['source'], frames=len(rows),
                      accepted=sum(row['ok'] for row in rows),
                      marker_anchored=sum(row['marker_anchored'] for row in rows),
                      frame_time_percentiles_s={str(q):float(np.percentile([row['seconds'] for row in rows], q))
                                                for q in (50, 95, 99, 100)},
                      tracking_seconds=time.monotonic()-started, rows=rows)
        (output/'far-rebuild.json').write_text(json.dumps(record, indent=2)+'\n')
    if build:
        from scanner.stages.engine import Engine
        from scanner.stages.project import Project
        result = Engine(Project(str(output))).run(force=['side.fuse@'+folder.name])
        if not result.ok: raise ValueError('The new copy could not finish its build. The source was not changed.')
        record['build_seconds']=result.secs
    record['project']=str(output)
    (output/'far-rebuild.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description='Rebuild with Large part calibration in a new copy.')
    parser.add_argument('project');parser.add_argument('--side',type=int,required=True)
    parser.add_argument('--calibration',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--track-only',action='store_true');parser.add_argument('--mesh-output')
    args=parser.parse_args()
    try: result=rebuild_copy(args.project,args.side,args.calibration,args.output,not args.track_only,args.mesh_output)
    except Exception as error:
        print('FAILED '+str(error).splitlines()[0],flush=True);return 1
    print('DONE '+json.dumps({key:value for key,value in result.items() if key!='rows'}),flush=True);return 0


if __name__=='__main__':
    raise SystemExit(main())
