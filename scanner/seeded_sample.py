# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registration surface samples with an explicit, local random seed."""
import numpy as np
from align import sample_points_with_normals

def sample(path, n, seed=0):
    import trimesh
    loaded = trimesh.load(path, process=False)
    m = loaded
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values()]
        meshes = [g for g in geoms if hasattr(g, "faces") and len(getattr(g, "faces", []))]
        clouds = [g for g in geoms if hasattr(g, "vertices") and len(getattr(g, "vertices", []))]
        if meshes:
            m = trimesh.util.concatenate(meshes)
        elif clouds:
            pts = np.concatenate([np.asarray(g.vertices, np.float64) for g in clouds], axis=0)
            return sample_points_with_normals(pts, n)
    if hasattr(m, "faces") and len(getattr(m, "faces", [])):
        if len(m.faces) > 800000:
            import fast_simplification
            v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32), target_count=800000)
            m = trimesh.Trimesh(v, f, process=False)
        pts, fidx = trimesh.sample.sample_surface(m, n, seed=seed)
        return np.asarray(pts, np.float64), np.asarray(m.face_normals[fidx], np.float64)
    pts = np.asarray(getattr(m, "vertices", []), np.float64)
    return sample_points_with_normals(pts, n)

