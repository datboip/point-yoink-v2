#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut the turntable out of a fused live scan.

The camera circled the turntable, so the circle's axis is the spin axis and the platter is the plane at right angles
to it. Only planes within a few degrees of that are considered (so a flat wall of the part is never taken for the
platter). Everything at or below the platter is dropped, then the part's main piece is kept with any piece that lies
mostly inside its bounds (the platter's raised rim ring lies outside them).
usage: strip_table.py <scan dir> out.ply [--margin mm]"""
import os, sys, argparse, json, numpy as np, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuse import read_global_poses

ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("out")
ap.add_argument("--mesh", default=None, help="mesh to cut (default <scan>/live_fused.ply); the spin axis still comes from the scan's poses")
ap.add_argument("--margin", type=float, default=1.0, help="mm above the platter that is also dropped")
ap.add_argument("--plane-out", default=None, help="also write the platter plane as fuse.py's plane.json (so the final merge cuts it too)")
ap.add_argument("--max-tilt", type=float, default=8.0, help="deg, how far the platter plane may lean from the spin axis")
ap.add_argument("--plane", default=None, help="cut at this plane.json (found while scanning, table_plane.py) instead of looking for the platter")
a = ap.parse_args()

m = o3d.io.read_triangle_mesh(a.mesh or os.path.join(a.scan, "live_fused.ply"))
V = np.asarray(m.vertices)
# spin axis: normal of the plane the camera centres move in
C = np.array([M[:3, 3] for M in read_global_poses(os.path.join(a.scan, "cache", "pointyoink_register_pose.pose")).values()])
axis = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
# the part is on the camera side of the platter (the scanner looks down on the table)
if (C.mean(0) - V.mean(0)) @ axis < 0: axis = -axis

# the platter is the densest 1 mm layer along the spin axis (its top and its dots all sit at one height, even when the
# range cut left only the dots); fit a plane to that layer to take out any small error in the axis
o3d.utility.random.seed(0)
P = np.asarray(m.sample_points_uniformly(200000).points)
h = P @ axis
hist, edges = np.histogram(h, bins=np.arange(h.min(), h.max() + 1.0, 1.0))
layer = edges[hist.argmax()] + 0.5
sel = P[np.abs(h - layer) < 2.0]
# full_matrices=False: the default also builds an N x N matrix, 80 GB once the platter is in range and N passes 100k
cen = sel.mean(0); n = np.linalg.svd(sel - cen, full_matrices=False)[2][2]
if n @ axis < 0: n = -n
if np.degrees(np.arccos(min(1, n @ axis))) > a.max_tilt: n = axis            # a sloppy fit: trust the axis
d = -n @ cen
best = (len(sel), None)
if a.plane:
    import json; pj = json.load(open(a.plane)); n = np.array(pj["n"], float); d = -float(pj["d"]); best = (int(pj.get("frames", 0)), None)
from scanner.base_geometry import saved_spacing, verify_top, find_top
spacing = saved_spacing(a.scan)
support = verify_top(V, C, n, -d, spacing, a.margin, a.max_tilt)
if not a.plane and not support['verified']:
    top = find_top(V, C, spacing, a.margin, a.max_tilt)
    if top is not None:
        n, d, count, support = top
        best = (count, None)
h_all = V @ n + d
from scanner.base_cut import removed, OUTLINE_FIELDS
cut_plane={'n':n.tolist(),'d':float(-d),'margin':a.margin,'keep_above':True}
if a.plane:cut_plane.update({k:pj[k] for k in OUTLINE_FIELDS if k in pj})
elif support.get('verified'):
    from scanner.base_geometry import _context
    context=_context(V,C,n)
    if context is not None:
        _,_,u,v,center,_,_=context
        cut_plane.update(centre=(center[0]*u+center[1]*v-d*n).tolist(),radius_mm=support['radius_mm'])
keep_v = ~removed(V,cut_plane)
removed_share = float(np.mean(~keep_v))
# a plane through the part leaves part of the side well below it; the right plane only takes a thin flat layer (the plate), however
# much of the view that layer fills. Measured (2026-10-06 scan): right planes 0 to 3% more than 5 mm below while removing 46 to 60%;
# the same planes pushed 25 mm into the part 65 to 76%. The old test (removed > 40%) refused right cuts on plate-heavy sides.
deep_share = float(np.mean((h_all < -5.0) & ~keep_v))
print("STAGE turntable_support " + json.dumps(support), flush=True)
if deep_share > 0.10 and not support['verified']:
    print("STAGE wrong_plane " + json.dumps({"removed_share": round(removed_share, 4), "deep_share": round(deep_share, 4),
                                               "margin": a.margin}), flush=True)
T = np.asarray(m.triangles)
m.remove_triangles_by_mask(~keep_v[T].all(1)); m.remove_unreferenced_vertices()
if not m.has_triangles():
    print("STAGE cut_rejected " + json.dumps({"reason": "empty_cut", "margin": a.margin}), flush=True)
    sys.exit(2)
if a.plane_out:     # fuse.py keeps points with (x . n) >= d + margin, x in the scan's own frame (mm): same frame as this mesh
    with open(a.plane_out, "w") as f:
        plane_out = dict(cut_plane)
        if a.plane:
            plane_out.update({k: pj[k] for k in ("source", "provenance", "points") if k in pj})
        json.dump(plane_out, f)
tri, cnt, _ = m.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
V2 = np.asarray(m.vertices); T2 = np.asarray(m.triangles)
big = int(cnt.argmax())
bv = V2[np.unique(T2[tri == big])]; lo, hi = bv.min(0) - 3, bv.max(0) + 3
keep_c = []
for c in range(len(cnt)):
    if c == big: keep_c.append(c); continue
    if cnt[c] < 200: continue
    cv = V2[np.unique(T2[tri == c])]
    if ((cv >= lo) & (cv <= hi)).all(1).mean() > 0.8: keep_c.append(c)
m.remove_triangles_by_mask(~np.isin(tri, keep_c)); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, m)
tilt = np.degrees(np.arccos(min(1, abs(n @ axis))))
print("platter: %d support points, %.1f deg off the spin axis; kept %d triangles in %d pieces, part height %.1f mm"
      % (best[0], tilt, len(m.triangles), len(keep_c), float((np.asarray(m.vertices) @ n + d).max())))
