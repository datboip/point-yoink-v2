#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep the scanned part, drop the floating islands (turntable dots, specks): every connected piece at least
--min-share of the biggest one survives. usage: keep_part.py in.ply out.ply"""
import sys, argparse, numpy as np, open3d as o3d
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--min-share", type=float, default=0.03)
a = ap.parse_args()
m = o3d.io.read_triangle_mesh(a.inp)
tri, cnt, area = m.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
keep = np.nonzero(cnt >= a.min_share * cnt.max())[0]
m.remove_triangles_by_mask(~np.isin(tri, keep)); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, m)
print("kept %d of %d pieces, %d triangles (%.1f%%)" % (len(keep), len(cnt), len(m.triangles), 100 * cnt[keep].sum() / cnt.sum()))
