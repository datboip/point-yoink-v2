"""Real geometry edits followed by the engine's shared color transfer."""
import json,os,subprocess,sys
import numpy as np
import pytest
import trimesh
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scanner"))   # "from stages import ..." below
from stages import color,plyio
from stages.tools import mesh_tools,edits,plane_cut


def finish_pair(tmp_path,mesh,V,F):
    C=np.column_stack((np.clip((mesh.vertices[:,0]+25)*5,0,255),np.full(len(mesh.vertices),90),np.full(len(mesh.vertices),180))).astype(np.uint8)
    src=str(tmp_path/'source.ply');dst=str(tmp_path/'result.ply')
    plyio.write_mesh(src,mesh.vertices,mesh.faces,colors=C)
    plyio.write_mesh(dst,V,F)
    assert not color.has_colors(dst)
    color.carry_job({'source':src,'target':dst})
    assert color.has_colors(dst) and len(color.colors(dst))==len(V)
    np.testing.assert_array_equal(plyio.read_mesh(dst)[0],np.asarray(V,np.float32))
    np.testing.assert_array_equal(plyio.read_mesh(dst)[1],F)
    return color.colors(dst)


@pytest.mark.parametrize('name',['isolation','smooth','delete_box','delete_mask','cut_shape'])
def test_mesh_edits_keep_color_after_geometry_edit(tmp_path,name):
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=20)
    V,F=mesh.vertices,mesh.faces;data=(V,F,None)
    if name=='isolation':out,_,_=mesh_tools.apply('isolation',data,{'rate':1.})
    elif name=='smooth':out,_,_=mesh_tools.apply('smooth',data,{'strength':5.,'times':2})
    elif name=='cut_shape':out,_,_=plane_cut.apply('mesh',data,{'nx':0,'ny':0,'nz':1,'d':-10,'margin':0})
    else:
        selected=V[F].mean(1)[:,0]>10
        if name=='delete_mask':selected[:len(selected)//2]=False
        out=edits.apply_mask('mesh',data,selected)
    finish_pair(tmp_path,mesh,*out[:2])




def test_added_fin_and_cut_gap_receive_surface_color(tmp_path):
    from test_app2_fin_validation import l_part,params
    from stages.tools.fin import join_fin
    from stages.tools.slice_gap import cut_gap
    from test_app2_slice_gap import params as gap_params
    body=l_part();V,F=join_fin(body.vertices,body.faces,params((12,0,2),(2,0,12),((0,0,1),(1,0,0))))
    finish_pair(tmp_path,body,V,F)
    body=trimesh.creation.box(extents=[10,10,10]);V,F=cut_gap(body.vertices,body.faces,gap_params())
    finish_pair(tmp_path,body,V,F)


def test_finish_for_print_carries_color_on_new_surface(tmp_path):
    from stages.tools.finish import app_has_pymeshlab
    if not app_has_pymeshlab():pytest.skip('pymeshlab is not installed')
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=20)
    src=str(tmp_path/'input.ply');target=str(tmp_path/'finished.ply');plyio.write_mesh(src,mesh.vertices,mesh.faces)
    from pathlib import Path
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scanner/finish_mesh.py'),src,target,'--depth','5','--faces','1000','--smooth','1']
    subprocess.run(command,check=True,capture_output=True)
    V,F,_=plyio.read_mesh(target);finish_pair(tmp_path,mesh,V,F)


def test_poisson_new_vertices_receive_color(tmp_path):
    import open3d as o3d
    sphere=o3d.geometry.TriangleMesh.create_sphere(radius=20,resolution=15);sphere.compute_vertex_normals()
    pc=sphere.sample_points_uniformly(2500)
    result,_=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc,depth=5,n_threads=2)
    finish_pair(tmp_path,trimesh.Trimesh(np.asarray(sphere.vertices),np.asarray(sphere.triangles),process=False),np.asarray(result.vertices),np.asarray(result.triangles))


def test_actual_cut_shape_receives_colors_on_boolean_faces(tmp_path):
    from stages.tools.drill import cut_shape
    body=trimesh.creation.box(extents=[20,20,10])
    params={'anchors':json.dumps([[[0,0,5],[0,0,1]]]),'shape':'box','width':4.,'height':4.,'depth':6.}
    V,F=cut_shape(body.vertices,body.faces,params)
    result=finish_pair(tmp_path,body,V,F)
    assert len(result)==len(V) and trimesh.Trimesh(V,F).is_watertight


def test_tool_colors_are_dropped_when_the_source_has_none(tmp_path):
    """Finish for print wrote all-black vertex colors on a gray model; with no source color the result stays gray."""
    import numpy as np
    from stages import plyio, color
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float); F = np.array([[0, 1, 2]]); N = np.zeros((3, 3))
    src = str(tmp_path / "src.ply"); dst = str(tmp_path / "dst.ply")
    plyio.write_mesh(src, V, F, N); plyio.write_mesh(dst, V, F, N, colors=np.zeros((3, 3), np.uint8))
    assert color.carry_job(dict(source=src, target=dst)) == dict(color=False) and not color.has_colors(dst)
