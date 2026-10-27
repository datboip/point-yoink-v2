"""Repeated cuts must store identical planes and meshes, including sampled fits."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import open3d as o3d


def test_sampled_base_cut_is_identical_in_fresh_processes(tmp_path):
    scan=tmp_path/'scan';(scan/'cache').mkdir(parents=True)
    matrices=[]
    for angle in np.linspace(0,2*np.pi,12,endpoint=False):
        matrix=np.eye(4);matrix[:3,3]=[150*np.cos(angle),150*np.sin(angle),100]
        matrices.append(matrix)
    poses=struct.pack('<i',len(matrices))
    poses+=b''.join(struct.pack('<ii16d',0,i,*matrix.ravel()) for i,matrix in enumerate(matrices))
    (scan/'cache/pointyoink_register_pose.pose').write_bytes(poses)
    mesh=o3d.geometry.TriangleMesh.create_box(200,200,1).translate((-100,-100,-1))
    mesh+=o3d.geometry.TriangleMesh.create_box(40,30,50).translate((-20,-15,0))
    mesh.compute_vertex_normals();raw=tmp_path/'raw.ply';o3d.io.write_triangle_mesh(str(raw),mesh)
    root=Path(__file__).resolve().parents[1]
    results=[]
    for i in range(2):
        cut=tmp_path/f'cut{i}.ply';plane=tmp_path/f'plane{i}.json'
        result=subprocess.run([sys.executable,str(root/'scanner/strip_table.py'),str(scan),str(cut),'--mesh',str(raw),'--plane-out',str(plane)],env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        results.append((cut.read_bytes(),plane.read_bytes()))
        assert o3d.io.read_triangle_mesh(str(cut)).has_triangles()
    assert results[0]==results[1]
