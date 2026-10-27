# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reinforcing rib joined to a closed mesh, recorded as an undoable mesh tool."""
import json
import os
import numpy as np
from ..methods import Tool, register


def _parameters(p):
    try:
        anchors = np.asarray(json.loads(p['anchors']), dtype=float)
        thick, reach = float(p['thickness']), float(p['reach'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('Pick two attachment spots and valid fin sizes.') from e
    if anchors.ndim != 3 or anchors.shape[1:] != (2, 3) or len(anchors) < 2 or not np.isfinite(anchors).all():
        raise ValueError('Pick two attachment spots on the mesh.')
    if not np.isfinite([thick, reach]).all() or not 0.2 <= thick <= 30 or not 0.5 <= reach <= 100:
        raise ValueError('Fin thickness must be 0.2 to 30 mm and reach 0.5 to 100 mm.')
    if p.get('shape', 'triangle') not in ('triangle', 'rounded'):
        raise ValueError('Choose a triangular or rounded fin.')
    norms = np.linalg.norm(anchors[:, 1], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('Pick attachment spots with valid surface directions.')
    anchors[:, 1] /= norms[:, None]
    return anchors, thick, reach


def nearest_surface(vertices, faces, points):
    """Exact closest triangles, with bounded temporary storage and no spatial-index dependency."""
    import trimesh
    vertices, faces, points = np.asarray(vertices), np.asarray(faces), np.asarray(points)
    best = np.full(len(points), np.inf)
    nearest = np.empty((len(points), 3)); indices = np.full(len(points), -1, dtype=int)
    for start in range(0, len(faces), 16384):
        triangles = vertices[faces[start:start + 16384]]
        for j, point in enumerate(points):
            close = trimesh.triangles.closest_point(triangles, np.broadcast_to(point, (len(triangles), 3)))
            distance = np.linalg.norm(close - point, axis=1)
            k = int(np.argmin(distance))
            if distance[k] < best[j]:
                best[j], nearest[j], indices[j] = distance[k], close[k], start + k
    if np.any(indices < 0):
        raise ValueError('The mesh has no attachment surfaces.')
    return nearest, best, indices


def rib_geometry(p):
    """The first two ends define the base; later spots define its plane, in mm."""
    import trimesh
    anchors, thick, reach = _parameters(p)
    a, b = anchors[:2, 0]; normals = anchors[:2, 1]
    axis = b - a; length = np.linalg.norm(axis)
    if length < max(thick, 0.1): raise ValueError('Pick attachment spots farther apart.')
    axis /= length
    normal = (anchors[2:, 0].mean(axis=0) - (a+b)*0.5) if len(anchors) >= 3 else normals.sum(axis=0)
    normal -= axis * np.dot(axis, normal)
    if np.linalg.norm(normal) < 0.05:
        if len(anchors) == 2:
            if np.linalg.norm(normals.sum(axis=0)) < 0.05:
                raise ValueError('These surfaces face opposite ways. Pick different attachment spots.')
            raise ValueError('The attachment directions lie on the end line. Pick different spots.')
        raise ValueError('The plane spots lie on the end line. Pick a spot away from that line.')
    normal /= np.linalg.norm(normal); width = np.cross(axis, normal)
    overlap = min(thick * 0.5, length * 0.1)
    a = a - normals[0] * overlap - axis * overlap
    b = b - normals[1] * overlap + axis * overlap
    if p.get('shape') == 'rounded':
        ts = np.linspace(1, 0, 17)
        arc = [(1-t)*a + t*b + 4*t*(1-t)*normal*(reach+overlap) for t in ts]
        profile = np.asarray([a, b] + arc[1:-1])
    else: profile = np.asarray([a, b, (a+b)*0.5 + normal*(reach+overlap)])
    n = len(profile); vertices = np.concatenate((profile-width*thick/2, profile+width*thick/2))
    faces = []
    for i in range(1,n-1): faces += [(0,i+1,i),(n,n+i,n+i+1)]
    for i in range(n):
        j=(i+1)%n; faces += [(i,j,j+n),(i,j+n,i+n)]
    mesh=trimesh.Trimesh(vertices, np.asarray(faces), process=True); mesh.fix_normals()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def join_fin(V, F, p):
    import trimesh
    import pymeshlab as ml
    body=trimesh.Trimesh(V,F,process=True); body.fix_normals()
    if not body.is_watertight or not body.is_volume:
        raise ValueError('Add fin needs a closed mesh. Fill holes or use Finish for print first.')
    anchors, thick, reach = _parameters(p)
    anchors = anchors[:2]  # Extra spots set the plane; only the ends attach.
    _, distances, indices = nearest_surface(body.vertices, body.faces, anchors[:, 0])
    tolerance = max(1e-5, min(thick * 0.005, 0.01))
    if np.any(distances > tolerance):
        raise ValueError('Both attachment spots must be on the current mesh. Pick them again.')
    # An outward normal is required for each end. Accept adjacent-face picks at sharp edges.
    if np.any(np.einsum('ij,ij->i', anchors[:, 1], body.face_normals[indices]) < -0.05):
        raise ValueError('An attachment faces into the part. Pick the spots again.')
    R, RF = rib_geometry(p)
    rib = trimesh.Trimesh(R, RF, process=False)
    if not rib.is_volume:
        raise ValueError('These spots cannot form a solid fin. Pick other spots.')
    # Both selected surfaces must sit strictly inside the rib, providing overlap at each end.
    # Rib profiles are convex, so signed face planes give an exact interior check.
    signed = np.einsum('pfi,fi->pf', anchors[:, 0, None, :] - rib.triangles_center[None, :, :], rib.face_normals)
    if np.any(np.max(signed, axis=1) >= -tolerance):
        raise ValueError('The fin must overlap both attachment spots. Pick different spots or increase thickness.')
    ms=ml.MeshSet(); ms.add_mesh(ml.Mesh(np.asarray(body.vertices),np.asarray(body.faces)), 'Part')
    ms.add_mesh(ml.Mesh(R,RF), 'Fin')
    ms.generate_boolean_union(first_mesh=0,second_mesh=1)
    result=ms.current_mesh()
    out=trimesh.Trimesh(result.vertex_matrix(),result.face_matrix(),process=True); out.fix_normals()
    if not out.is_volume or len(out.split()) > len(body.split()):
        raise ValueError('The fin did not join the part. Pick closer attachment spots.')
    if out.volume <= body.volume + 1e-6:
        raise ValueError('The fin is inside the part. Increase its reach or pick other spots.')
    return np.asarray(out.vertices),np.asarray(out.faces)


@register
class Fin(Tool):
    name='fin'; label='Add fin'; kinds=('mesh',); version=3; mem_gb=6
    scripts=(); modules=('tools/fin.py',); libs=('pymeshlab','trimesh')

    def execute(self,ctx):
        from ..plyio import read_mesh, write_mesh
        V,F,_=read_mesh(ctx.inputs['parent'].files['mesh'])
        V,F=join_fin(V,F,ctx.params)
        write_mesh(os.path.join(ctx.out_dir,'mesh.ply'),V,F)
        return {'outputs':{'mesh':'mesh.ply'},'stats':{'faces':len(F)}}

    def summary(self,p): return '%g mm thick, %g mm reach' % (p['thickness'],p['reach'])
