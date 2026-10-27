# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watertight rebuild test: Poisson over the built model's points, keeping everything within reach of real data."""
import sys, time, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
src, out, depth, reach = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
m = o3d.io.read_triangle_mesh(src); m.compute_vertex_normals()
pc = m.sample_points_poisson_disk(400000, use_triangle_normal=True) if False else m.sample_points_uniformly(600000, use_triangle_normal=True)
t = time.time()
rm, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, n_threads=6)
V = np.asarray(rm.vertices); d = cKDTree(np.asarray(pc.points)).query(V, workers=6)[0]
rm.remove_vertices_by_mask(d > reach)                      # drop only what Poisson invented far from any data
tri, cnt, _ = rm.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
rm.remove_triangles_by_mask(tri != cnt.argmax()); rm.remove_unreferenced_vertices()
rm = rm.filter_smooth_taubin(number_of_iterations=5); rm.compute_vertex_normals()
o3d.io.write_triangle_mesh(out, rm)
print("depth %d reach %.1f mm: %d triangles, %d pieces before keeping the biggest, watertight %s, edges manifold %s, %.0f s"
      % (depth, reach, len(rm.triangles), len(cnt), rm.is_watertight(), rm.is_edge_manifold(), time.time() - t))
