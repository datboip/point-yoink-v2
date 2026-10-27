"""A freshly fitted registration pair must repeat exactly in a new process."""
import json,os
from pathlib import Path
import subprocess,sys
import numpy as np
import open3d as o3d


def test_registration_pair_repeats_in_fresh_workers(tmp_path):
    root=Path(__file__).resolve().parents[1]
    mesh=o3d.geometry.TriangleMesh.create_box(30,20,10)
    mesh+=o3d.geometry.TriangleMesh.create_sphere(5,resolution=8).translate((24,15,13))
    mesh.compute_vertex_normals();a=tmp_path/'a.ply';b=tmp_path/'b.ply'
    o3d.io.write_triangle_mesh(str(a),mesh)
    mesh.translate((.4,-.2,.1));o3d.io.write_triangle_mesh(str(b),mesh)
    outputs=[]
    for i in range(2):
        cache=tmp_path/f'cache{i}.json';spec=tmp_path/f'job{i}.json'
        job=dict(fn='methods.pairs:pairs_job',version='regression',params=dict(voxel=1.,samples=3000,max_dist=6.),cache=str(cache),total=1,todo=[[0,1,0]],sides=[dict(n=n,mesh=str(path),digest=str(n),plane=None,plane_digest='-') for n,path in [(1,a),(2,b)]])
        spec.write_text(json.dumps(job))
        result=subprocess.run([sys.executable,str(root/'scanner/stages/job.py'),str(spec)],env=dict(os.environ,POINTYOINK_MEM_CAP_GB='100000',POINTYOINK_STABLE_PAIRS='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        def remove_time(value):
            if isinstance(value,dict):return {k:remove_time(v) for k,v in value.items() if k!='t'}
            if isinstance(value,list):return list(map(remove_time,value))
            return value
        outputs.append(remove_time(json.loads(cache.read_text())))
    assert outputs[0]==outputs[1]
