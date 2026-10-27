"""Canonical mesh bytes and the same wider claim in Plus planning and writing."""
import types,numpy as np
import fuse
from scanner import plus_memory


def test_canonical_mesh_is_identical_after_vertex_and_face_permutations(tmp_path):
    import open3d as o3d
    V=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float);F=np.array([[0,1,2],[0,2,3]])
    order=np.array([2,0,3,1]);inverse=np.argsort(order)
    a=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    b=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[order]),o3d.utility.Vector3iVector(inverse[F[::-1]][:,[1,2,0]]))
    for name,mesh in [('a',a),('b',b)]:o3d.io.write_triangle_mesh(str(tmp_path/(name+'.ply')),fuse._canonical_mesh(mesh))
    assert (tmp_path/'a.ply').read_bytes()==(tmp_path/'b.ply').read_bytes()


def test_plus_planner_and_kernel_claim_the_same_wider_band(tmp_path,monkeypatch):
    import open3d as o3d
    original=o3d.t.geometry.VoxelBlockGrid;claims=[]
    class Grid:
        def __init__(self,*args,**kwargs):self.grid=original(*args,**kwargs)
        def compute_unique_block_coordinates(self,*args,**kwargs):
            claims.append(args[-1]);return self.grid.compute_unique_block_coordinates(*args,**kwargs)
        def __getattr__(self,key):return getattr(self.grid,key)
    monkeypatch.setattr(o3d.t.geometry,'VoxelBlockGrid',Grid)
    a=types.SimpleNamespace(voxel=.4,plus_trunc=1.2,depth_scale=1.,max_depth=2000.,plus_block_count=128,plus_workers=1,_plus_memory=None)
    path=str(tmp_path/'frame_0_0.dph');job=(path,None,None,None,None,(4,4))
    load=lambda job:(path,np.full((4,4),1000,np.uint16),np.eye(4),(100.,100.,2.,2.))
    plan=plus_memory.sample_blocks(a,[job],load,.4,8)
    acc=fuse._plus_integrate(a,[job],load,o3d.core.Device('CPU:0'),8)
    assert len(claims)==2 and claims[0]==claims[1] and np.isclose(claims[0],9.)
    assert plan['frame_blocks']>0 and len(acc['keys'])>0


def test_coincident_vertices_keep_their_own_sheet_and_stable_bytes(tmp_path):
    import open3d as o3d
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,1]],float);F=np.array([[0,1,2],[3,4,5]])
    order=np.array([3,2,5,0,4,1]);inverse=np.argsort(order)
    for name,v,f in [('a',V,F),('b',V[order],inverse[F[::-1]])]:
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f));mesh=fuse._canonical_mesh(mesh)
        assert len(mesh.vertices)==6 and len(mesh.triangles)==2
        o3d.io.write_triangle_mesh(str(tmp_path/(name+'.ply')),mesh)
    assert (tmp_path/'a.ply').read_bytes()==(tmp_path/'b.ply').read_bytes()
