"""Measured board evidence stays identical when runtime changes."""
import json
import numpy as np
import sys
from scanner import board_plane,bed,table_plane


def test_recovered_plane_excludes_runtime_but_audit_keeps_it(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'bed',bed);monkeypatch.setitem(sys.modules,'table_plane',table_plane)
    (tmp_path/'cache').mkdir();(tmp_path/'param').mkdir()
    (tmp_path/'cache/pointyoink_register_pose.pose').touch();(tmp_path/'param/Pl.bin').touch()
    (tmp_path/'table_plane.json').write_text(json.dumps(dict(n=[0,0,1],d=100.,tilt_deg=0.)))
    poses={(0,k):np.eye(4) for k in range(60)}
    x,y=np.meshgrid(np.arange(10),np.arange(10));points=np.column_stack((x.ravel(),y.ravel(),np.full(100,100.)))
    monkeypatch.setattr(table_plane,'F',np.eye(4))
    monkeypatch.setattr(bed,'_side_io',lambda _: (None,poses))
    monkeypatch.setattr(bed,'read_frame',lambda *a:(None,np.zeros((2,2))))
    monkeypatch.setattr(bed,'detect_marks',lambda *a,**kw:(np.zeros((4,2)),np.ones(4,bool)))
    monkeypatch.setattr(bed,'ring_depth_points',lambda *a,**kw:points)
    monkeypatch.setattr(bed,'dot_depth_points',lambda *a:points[:4]+[0,0,.5])
    results=[];audits=[]
    for times in [(1.,2.),(4.,99.)]:
        clock=iter(times);monkeypatch.setattr(board_plane.time,'perf_counter',lambda:next(clock))
        plane,audit=board_plane.recover(tmp_path)
        assert plane is not None and board_plane.supported(plane)
        results.append(json.dumps(plane,sort_keys=True));audits.append(audit)
    assert results[0]==results[1]
    assert audits[0]['seconds']==1. and audits[1]['seconds']==95.
    assert 'seconds' not in json.loads(results[0])['provenance']
