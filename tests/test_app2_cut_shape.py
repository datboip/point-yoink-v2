"""Negative primitives remove material and leave a closed body."""
import os
import json
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
import app2
import numpy as np
import pytest
import trimesh
from stages.tools.drill import cut_shape,cut_geometry


def params(**extra):
    return dict(anchors=json.dumps([[[0,0,5],[0,0,1]]]),**extra)


@pytest.mark.parametrize('shape',['cylinder','box','sphere'])
def test_closed_subtraction(shape):
    mesh=trimesh.creation.box(extents=(20,20,10))
    v,f=cut_shape(mesh.vertices,mesh.faces,params(shape=shape,depth=12,inset=-1))
    result=trimesh.Trimesh(v,f)
    assert result.is_watertight and result.is_volume
    assert 0<result.volume<mesh.volume
    assert len(result.split())==1


@pytest.mark.parametrize('values',[{'offset_u':50},{'anchors':'[]'},{'diameter':0},{'tilt':90}])
def test_invalid_or_missed_cut(values):
    mesh=trimesh.creation.box(extents=(20,20,10));p=params();p.update(values)
    with pytest.raises(ValueError):cut_shape(mesh.vertices,mesh.faces,p)


def test_open_mesh_refused():
    mesh=trimesh.creation.box(extents=(20,20,10))
    with pytest.raises(ValueError,match='closed mesh'):cut_shape(mesh.vertices,mesh.faces[:-1],params())


def test_offsets_and_angle_preserve_closed_negative():
    v,f=cut_geometry(params(shape='box',width=2,height=4,depth=6,tilt=30,rotation=90,offset_u=3,offset_v=2))
    mesh=trimesh.Trimesh(v,f)
    assert mesh.is_volume
    axis=np.array([0,.5,-np.sqrt(3)/2]);margin=.05
    expected=np.array([3,2,5])+axis*(6-margin)/2
    assert np.allclose(mesh.center_mass,expected)
