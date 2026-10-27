"""Closed slot subtraction on disposable geometry, with no project data writes."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink-gap-'))/'config.json')
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scanner'))
import numpy as np
import pytest
import trimesh
from stages.tools.slice_gap import cut_gap, slot_geometry


def params(**updates):
    p={'anchors':json.dumps([[[-5,0,5],[0,0,1]],[[5,0,5],[0,0,1]]]),'width':1.,'depth':11.}
    p.update(updates); return p


def test_through_gap_leaves_two_closed_pieces_and_open_space():
    body=trimesh.creation.box(extents=[10,10,10])
    V,F=cut_gap(body.vertices,body.faces,params())
    result=trimesh.Trimesh(V,F,process=True)
    pieces=result.split()
    assert result.is_volume and result.is_watertight
    assert len(pieces)==2 and all(p.is_watertight for p in pieces)
    assert result.volume==pytest.approx(900.,rel=1e-6)
    # Every triangle belongs to one side of the open slot, including new caps.
    ys=result.triangles[:,:,1]
    assert np.all(np.all(ys>=.49999,axis=1)|np.all(ys<=-.49999,axis=1))


def test_shallow_gap_keeps_body_connected_and_caps_cut():
    body=trimesh.creation.box(extents=[10,10,10])
    V,F=cut_gap(body.vertices,body.faces,params(depth=2))
    result=trimesh.Trimesh(V,F,process=True)
    assert result.is_watertight and result.is_volume
    assert len(result.split())==1
    assert result.volume==pytest.approx(980.,rel=1e-6)


def test_open_source_is_refused():
    body=trimesh.creation.box()
    with pytest.raises(ValueError,match='closed mesh'):
        cut_gap(body.vertices,body.faces[:-1],params())


@pytest.mark.parametrize('update',[{'width':0},{'depth':float('nan')},{'anchors':'bad'},
    {'anchors':json.dumps([[[0,0,0],[0,0,1]],[[0,0,0],[0,0,1]]])}])
def test_invalid_parameters(update):
    with pytest.raises(ValueError): slot_geometry(params(**update))


def test_stale_anchors_refused():
    body=trimesh.creation.box(extents=[10,10,10])
    p=params(anchors=json.dumps([[[-5,0,15],[0,0,1]],[[5,0,15],[0,0,1]]]))
    with pytest.raises(ValueError,match='current mesh'): cut_gap(body.vertices,body.faces,p)


def test_whole_body_removal_refused():
    body=trimesh.creation.box(extents=[10,10,10])
    with pytest.raises(ValueError): cut_gap(body.vertices,body.faces,params(width=20,depth=20))


def test_no_intersection_is_refused(monkeypatch):
    from stages.tools import slice_gap
    body=trimesh.creation.box(extents=[10,10,10])
    far=trimesh.creation.box(); far.apply_translation([100,0,0])
    monkeypatch.setattr(slice_gap,'slot_geometry',lambda p:(far.vertices,far.faces))
    with pytest.raises(ValueError,match='did not cut'):
        cut_gap(body.vertices,body.faces,params())


def test_execute_reports_intentional_split_without_changing_source(tmp_path):
    from types import SimpleNamespace
    from stages.tools.slice_gap import SliceGap
    from stages.plyio import write_mesh, read_mesh
    body=trimesh.creation.box(extents=[10,10,10]); source=tmp_path/'source.ply'
    write_mesh(str(source),body.vertices,body.faces); before=source.read_bytes()
    out=tmp_path/'output'; out.mkdir()
    ctx=SimpleNamespace(inputs={'parent':SimpleNamespace(files={'mesh':str(source)})},params=params(),out_dir=str(out))
    result=SliceGap().execute(ctx)
    assert result['stats']['pieces']==2
    assert source.read_bytes()==before
    V,F,_=read_mesh(str(out/'mesh.ply'))
    assert trimesh.Trimesh(V,F,process=True).is_watertight
