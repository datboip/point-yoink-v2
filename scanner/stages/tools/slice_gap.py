# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subtract a closed rectangular slot and cap its cut faces without filling the gap."""
import json
import os
import numpy as np
from ..methods import Tool, register
from .fin import nearest_surface


def _parameters(p):
    try:
        anchors = np.asarray(json.loads(p['anchors']), dtype=float)
        width, depth = float(p['width']), float(p['depth'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Pick two cut ends and valid gap sizes.') from exc
    if anchors.shape != (2, 2, 3) or not np.isfinite(anchors).all():
        raise ValueError('Pick two cut ends on the current mesh.')
    if not np.isfinite([width, depth]).all() or not .1 <= width <= 100 or not .1 <= depth <= 500:
        raise ValueError('Gap width must be 0.1 to 100 mm and depth 0.1 to 500 mm.')
    norms = np.linalg.norm(anchors[:, 1], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('Pick cut ends with valid surface directions.')
    anchors[:, 1] /= norms[:, None]
    return anchors, width, depth


def slot_geometry(p):
    """Box length joins picked ends; depth cuts inward from their mean surface plane."""
    import trimesh
    anchors, width, depth = _parameters(p)
    a, b = anchors[:, 0]
    along = b-a; length = np.linalg.norm(along)
    if length < .1:
        raise ValueError('Pick cut ends farther apart.')
    along /= length
    outward = anchors[:, 1].sum(axis=0)
    outward -= along*np.dot(along, outward)
    if np.linalg.norm(outward) < .05:
        raise ValueError('These surfaces face opposite ways. Pick different cut ends.')
    outward /= np.linalg.norm(outward)
    across = np.cross(outward, along)
    # A small outward overlap prevents a coplanar Boolean at the picked surface.
    margin = max(.001, min(width*.01, .05))
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((along, across, outward))
    transform[:3, 3] = (a+b)*.5 + outward*(margin-depth)*.5
    box = trimesh.creation.box(extents=(length+2*margin, width, depth+margin), transform=transform)
    return np.asarray(box.vertices), np.asarray(box.faces)


def cut_gap(V, F, p):
    import trimesh
    import pymeshlab as ml
    body = trimesh.Trimesh(V, F, process=True); body.fix_normals()
    if not body.is_watertight or not body.is_volume:
        raise ValueError('Cut gap needs a closed mesh. Fill holes or use Finish for print first.')
    anchors, width, depth = _parameters(p)
    _, distances, indices = nearest_surface(body.vertices, body.faces, anchors[:, 0])
    tolerance = max(1e-5, min(width*.005, .01))
    if np.any(distances > tolerance):
        raise ValueError('Both cut ends must be on the current mesh. Pick them again.')
    if np.any(np.einsum('ij,ij->i', anchors[:, 1], body.face_normals[indices]) < -.05):
        raise ValueError('A cut end faces into the part. Pick the ends again.')
    slot_v, slot_f = slot_geometry(p)
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(np.asarray(body.vertices), np.asarray(body.faces)), 'Part')
    ms.add_mesh(ml.Mesh(slot_v, slot_f), 'Cut gap')
    try:
        ms.generate_boolean_difference(first_mesh=0, second_mesh=1)
    except Exception as exc:
        raise ValueError('Could not cut this gap. Try other ends or a smaller width.') from exc
    result = ms.current_mesh()
    if result.face_number() == 0:
        raise ValueError('This cut removes the whole part. Use a smaller gap.')
    out = trimesh.Trimesh(result.vertex_matrix(), result.face_matrix(), process=True); out.fix_normals()
    if not out.is_watertight or not out.is_volume:
        raise ValueError('The cut could not close its new faces. Try other cut ends.')
    threshold = max(1e-8, abs(body.volume)*1e-9)
    if out.volume >= body.volume-threshold:
        raise ValueError('The gap did not cut the part. Increase depth or pick other ends.')
    if out.volume <= threshold:
        raise ValueError('This cut removes the whole part. Use a smaller gap.')
    return np.asarray(out.vertices), np.asarray(out.faces)


@register
class SliceGap(Tool):
    name = 'slice_gap'; label = 'Cut gap'; kinds = ('mesh',); version = 1; mem_gb = 6
    scripts = (); modules = ('tools/slice_gap.py', 'tools/fin.py'); libs = ('pymeshlab', 'trimesh')

    def execute(self, ctx):
        import trimesh
        from ..plyio import read_mesh, write_mesh
        V, F, _ = read_mesh(ctx.inputs['parent'].files['mesh'])
        V, F = cut_gap(V, F, ctx.params)
        write_mesh(os.path.join(ctx.out_dir, 'mesh.ply'), V, F)
        pieces = len(trimesh.Trimesh(V, F, process=False).split())
        return {'outputs': {'mesh': 'mesh.ply'}, 'stats': {'faces': len(F), 'pieces': pieces}}

    def summary(self, p):
        return '%g mm gap, %g mm deep' % (p['width'], p['depth'])
