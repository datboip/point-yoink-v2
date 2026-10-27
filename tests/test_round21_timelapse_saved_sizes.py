"""Imported Standard frames must produce genuine growing geometry."""
import json
from pathlib import Path
import numpy as np
from app2.scan import timelapse as TL
from test_app2_timelapse import make_project, CAL


def test_standard_import_uses_its_depth_size_and_calibration(tmp_path):
    project=make_project(str(tmp_path))
    from app2.scan import fmt
    folder=Path(fmt.saved_sides(project)[0])
    for path in (folder/'cache').glob('*.dph'):
        a=np.fromfile(path,np.uint16).reshape(600,800)
        path.write_bytes(a[::2,::2].copy().tobytes())
        ir=path.with_suffix('.irl')
        a=np.fromfile(ir,np.uint8).reshape(600,800)
        ir.write_bytes(a[::2,::2].copy().tobytes())
    (folder/'property.rvproj').write_text(json.dumps({'scan_param':{'depth_width':400,'depth_height':300}}))
    side=TL.pick_sides(project,[1])[0]
    assert side.depth(0) is not None, 'Standard frames were silently skipped'
    assert side.depth(0).shape==(300,400)
    assert side.ir(0).shape==(300,400)
    assert np.allclose(side.cal,np.array(CAL)*.5)
    trk=TL.make_tracker(side.cal,'CPU:0',*side.size)
    counts=[]
    for k in range(len(side.entries)):
        TL.fuse(trk,side,k)
        if k in (0,len(side.entries)-1):counts.append(len(TL.model_cloud(trk,side)[0]))
    assert trk.fused==len(side.entries)
    assert counts[1]>counts[0]>100, counts
