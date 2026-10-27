#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a built model needs more scanning: every open edge of the surface is a place the scanner never saw past (a rim
seen only from its two faces, a hollow, a shadow behind a rib). The model is coloured by distance to the nearest open
edge that matters (red = at it, yellow = near it, blue = fine) and the open edges are listed longest first.
Pinholes shorter than --min-mm are left out: clean-up fills those.
usage: needs_scan.py model.ply out.ply [--png out.png] [--json out.json] [--min-mm 8]"""
import sys, json, argparse, time, numpy as np, open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--png", default=None); ap.add_argument("--json", default=None)
ap.add_argument("--min-mm", type=float, default=8.0, help="open edges shorter than this are pinholes, not places to scan")
ap.add_argument("--red-mm", type=float, default=2.0); ap.add_argument("--yellow-mm", type=float, default=6.0)
ap.add_argument("--timings", default=None)
ap.add_argument("--edge-sort", choices=("packed", "legacy"), default="packed")
a = ap.parse_args()
timings = {}; started = phase = time.perf_counter()
def mark(name):
    global phase
    now = time.perf_counter(); timings[name] = now - phase; phase = now

m = o3d.io.read_triangle_mesh(a.inp); V = np.asarray(m.vertices); T = np.asarray(m.triangles)
mark("mesh_read")
E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
if a.edge_sort == "packed" and (not E.size or int(E.max()) < 2**32):
    keys = (E[:, 0].astype(np.uint64) << np.uint64(32)) | E[:, 1].astype(np.uint64)
    del E
    keys, cnt = np.unique(keys, return_counts=True)
    E = np.column_stack((keys >> np.uint64(32), keys & np.uint64(0xffffffff))).astype(np.int32)
    del keys
else:
    E, cnt = np.unique(E, axis=0, return_counts=True)
B = E[cnt == 1]                                                   # open edges: only one triangle uses them
mark("edge_sort")
ln = np.linalg.norm(V[B[:, 0]] - V[B[:, 1]], axis=1)
n, lab = connected_components(coo_matrix((np.ones(len(B)), (B[:, 0], B[:, 1])), shape=(len(V), len(V))), directed=False)
el = lab[B[:, 0]]                                                 # each open edge's run
length = np.bincount(el, weights=ln, minlength=n)
big = np.nonzero(length >= a.min_mm)[0]
keep = np.isin(el, big); bv = np.unique(B[keep])
c = (V.min(0) + V.max(0)) / 2; size = V.max(0) - V.min(0)
# runs that lie close together are one place to the person scanning: group them on a 15 mm grid
runs = []
for r in big:
    p = V[np.unique(B[el == r])]; runs.append((float(length[r]), p.mean(0)))
cells = {}
for L, p in runs:
    k = tuple(np.floor(p / 15.0).astype(int)); o = cells.get(k, [0.0, np.zeros(3), 0])
    cells[k] = [o[0] + L, o[1] + p * L, o[2] + 1]
places = sorted(((L, s / L, k) for L, s, k in cells.values()), key=lambda x: -x[0])
mark("components")
col = np.tile(np.array([0.35, 0.62, 0.95]), (len(V), 1))
if len(bv):
    d = cKDTree(V[bv]).query(V, workers=6)[0]
    t = np.clip((d - a.red_mm) / (a.yellow_mm - a.red_mm), 0, 1)[:, None]
    near = d < a.yellow_mm
    warm = (1 - t) * np.array([1.0, 0.25, 0.2]) + t * np.array([1.0, 0.8, 0.2])
    col[near] = warm[near]
    share = float((d < a.red_mm).mean())
else: share = 0.0
m.vertex_colors = o3d.utility.Vector3dVector(col); m.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, m)
mark("colored_mesh")
summary = dict(open_edge_mm=round(float(length[big].sum())), runs=len(big), pinholes=int(n_pin := (length[(length > 0) & (length < a.min_mm)] > 0).sum()),
               red_share=round(share, 3), size_mm=[round(float(x), 1) for x in size],
               places=[dict(open_edge_mm=round(L), at_mm=[round(float(x), 1) for x in p]) for L, p, _ in places[:12]])
if a.json: json.dump(summary, open(a.json, "w"), indent=1)
print("open edge that matters: %d mm in %d runs (plus %d pinholes left for clean-up); %.0f%% of the surface is right at one"
      % (summary["open_edge_mm"], summary["runs"], summary["pinholes"], 100 * share))
for k, pl in enumerate(summary["places"][:6]): print("  place %d: %d mm of open edge around %s" % (k + 1, pl["open_edge_mm"], pl["at_mm"]))

mark("summary")
if a.png:
    from PIL import Image, ImageDraw
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    tc = col[T].mean(1); r = np.linalg.norm(size) / 2
    mark("render_scene")
    def render(eye_dir, W=560, H=420):
        eye = c + np.asarray(eye_dir, float) / np.linalg.norm(eye_dir) * r * 2.3
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=45, center=c.tolist(), eye=eye.tolist(), up=[0, -1, 0], width_px=W, height_px=H)
        ans = sc.cast_rays(rays); hit = np.isfinite(ans["t_hit"].numpy()); nn = ans["primitive_normals"].numpy(); ids = ans["primitive_ids"].numpy()
        dd = rays.numpy()[..., 3:]; dd /= np.linalg.norm(dd, axis=-1, keepdims=True); sh = np.clip(0.2 + 0.8 * np.abs((nn * dd).sum(-1)), 0, 1)
        img = np.full((H, W, 3), 24, np.uint8); img[hit] = (sh[hit][:, None] * tc[ids[hit]] * 255).astype(np.uint8)
        return Image.fromarray(img)
    out = Image.new("RGB", (560 * 3, 440 * 2)); dr = ImageDraw.Draw(out)
    for k, e in enumerate(([0, -0.4, -1], [0, 0.4, 1], [1, -0.5, 0], [-1, 0.3, 0.2], [0, -1, -0.1], [0, 1, 0.1])):
        out.paste(render(e), ((k % 3) * 560, (k // 3) * 440 + 20))
    dr.text((6, 2), "red = scan here (open edge), yellow = near one, blue = fine      %d mm of open edge in %d runs" % (summary["open_edge_mm"], summary["runs"]), fill=(255, 255, 255))
    mark("six_renders")
    out.save(a.png)
    mark("png_write")

timings["total"] = time.perf_counter() - started
if a.timings: json.dump(timings, open(a.timings, "w"), indent=2)
