# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preserve vertex colors by closest surface interpolation after geometry edits."""
import os
import numpy as np
from . import plyio


def has_colors(path):
    try:
        if path.endswith('.npz'):
            import zipfile
            with zipfile.ZipFile(path) as z: return 'C.npy' in z.namelist()
        with open(path,'rb') as f:
            names=set()
            for _ in range(100):
                line=f.readline().decode('ascii','replace').strip()
                if line.startswith('property '):names.add(line.split()[-1])
                if line=='end_header':break
        return {'red','green','blue'}<=names
    except OSError:return False


def colors(path):
    if path.endswith('.npz'):return plyio.load_points(path)['C']
    v=plyio.read(path,faces=False)['vertex']
    return np.stack([v[x] for x in ('red','green','blue')],axis=1)


def transfer(V,F,C,points):
    import open3d as o3d
    if not len(points):return np.empty((0,3),np.uint8)
    if not len(F):
        from scipy.spatial import cKDTree
        return C[cKDTree(V).query(points,workers=1)[1]]
    scene=o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(np.asarray(V,np.float32)),o3d.core.Tensor(np.asarray(F,np.uint32)))
    out=np.empty((len(points),3),np.uint8)
    for start in range(0,len(points),100000):
        result=scene.compute_closest_points(o3d.core.Tensor(np.asarray(points[start:start+100000],np.float32)))
        faces=result['primitive_ids'].numpy();uv=result['primitive_uvs'].numpy()
        weights=np.column_stack((1-uv.sum(1),uv));value=(C[F[faces]].astype(np.float64)*weights[:,:,None]).sum(1)
        out[start:start+len(value)]=np.rint(np.clip(value,0,255)).astype(np.uint8)
    return out


def arrays(path):
    if path.endswith('.npz'):
        data=plyio.load_points(path)
        return data['P'],np.empty((0,3),np.int64),data['N']
    return plyio.read_mesh(path)


def store(path, C):
    if path.endswith('.npz'):
        data=plyio.load_points(path);data['C']=C
        tmp=path+'.color.tmp.npz';np.savez(tmp,**data);os.replace(tmp,path)
    else:
        V,F,N=plyio.read_mesh(path);tmp=path+'.color.tmp'
        plyio.write_mesh(tmp,V,F,N,colors=C);os.replace(tmp,path)


def carry_job(job):
    source=job['source'];target=job['target']
    if not has_colors(source):
        # no scan color to carry: drop any color the tool's writer added (Finish for print wrote all-black vertices)
        if has_colors(target) and not target.endswith('.npz'):
            V,F,N=plyio.read_mesh(target);tmp=target+'.color.tmp';plyio.write_mesh(tmp,V,F,N);os.replace(tmp,target)
        return dict(color=False)
    V,F,_=arrays(source);C=colors(source);points,faces,_=arrays(target)
    same_indices=not source.endswith(".npz") and not target.endswith(".npz") and len(V)==len(points) and np.array_equal(F,faces)
    output=C.copy() if same_indices else transfer(V,F,C,points);store(target,output)
    return dict(color=True,vertices=len(points))


def frame_job(job):
    import open3d as o3d
    from scanner.color_frames import paint
    from .frames import iter_frames, F
    from fuse import frame_key
    path=job['target'];P,_,N=arrays(path)
    mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(P);mesh.vertex_normals=o3d.utility.Vector3dVector(N)
    jobs=[]
    for f in iter_frames(job['sides_json'],job['project'],max_per_side=job['frames_per_side']):
        h,w=f.depth.shape
        jobs.append((f.path,{frame_key(f.path):f.T@F},None,f.cal,None,(w,h),True,None))
    stats=paint(mesh,jobs,white=job.get("white",True))
    store(path,np.rint(np.asarray(mesh.vertex_colors)*255).astype(np.uint8))
    return stats
