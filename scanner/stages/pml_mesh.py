#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""meshed.poisson, pml backend (runs under the PyMeshLab python: numpy, scipy, pymeshlab only; no open3d import).
Port of the bake's revo_points/pml_mesh.py: screened Poisson on the fused oriented points, then trim every face with a
vertex further than --trim mm from the points (no invented surface), drop small pieces, light Taubin.

    pml_mesh.py points.ply out.ply [--depth 10] [--scale 1.1] [--pointweight 4] [--spn 1.5] [--confidence] [--trim 0.5]
                [--min-piece 0.002] [--taubin 3]
--scale: Poisson's octree cube over the bounding box (1.1 = PyMeshLab's default); the cell is extent x scale / 2^depth."""
import argparse, time

import numpy as np


def main():
    import pymeshlab
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser(); ap.add_argument("points"); ap.add_argument("out")
    ap.add_argument("--depth", type=int, default=10); ap.add_argument("--scale", type=float, default=1.1)
    ap.add_argument("--pointweight", type=float, default=4.0)
    ap.add_argument("--spn", type=float, default=1.5); ap.add_argument("--confidence", action="store_true")
    ap.add_argument("--trim", type=float, default=0.5); ap.add_argument("--min-piece", type=float, default=0.002)
    ap.add_argument("--taubin", type=int, default=3); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    ms = pymeshlab.MeshSet(); ms.load_new_mesh(a.points)
    pts = ms.current_mesh().vertex_matrix().copy()
    ms.generate_surface_reconstruction_screened_poisson(depth=a.depth, scale=a.scale, samplespernode=a.spn, pointweight=a.pointweight,
                                                        confidence=a.confidence, preclean=False, threads=a.threads)
    m = ms.current_mesh(); print("poisson: %d verts %d faces, %.0f s" % (m.vertex_number(), m.face_number(), time.time() - t0), flush=True)
    V = m.vertex_matrix(); F = m.face_matrix()
    d = cKDTree(pts).query(V, workers=8)[0]
    keep = (d[F] <= a.trim).all(1)
    used = np.unique(F[keep]); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V[used], face_matrix=remap[F[keep]]), "trimmed")
    print("trim %.2f mm: kept %d / %d faces" % (a.trim, keep.sum(), len(F)), flush=True)
    ms.meshing_remove_duplicate_vertices(); ms.meshing_remove_unreferenced_vertices()
    nf = ms.current_mesh().face_number()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=max(25, int(a.min_piece * nf)))
    if a.taubin > 0: ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=a.taubin)
    ms.compute_normal_per_vertex()
    print("final: %d verts %d faces, %.0f s" % (ms.current_mesh().vertex_number(), ms.current_mesh().face_number(), time.time() - t0), flush=True)
    ms.save_current_mesh(a.out, save_vertex_color=False, save_vertex_quality=False, save_face_quality=False, save_face_color=False)


if __name__ == "__main__":
    main()
