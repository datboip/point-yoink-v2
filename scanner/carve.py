#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remove from a built model what the other sides looked straight through: props the part was resting on for one side
(a block, a clamp, a hand), leftover floaters and bits of platter. The part itself is never looked through: every
camera that points at it sees it. So a spot of the model goes when frames from at least --sides different sides saw
something further away along the ray to it (by more than --margin mm), each side with at least --frames such frames.
Rays with no depth say nothing (the range cut, dark surfaces), and neither does a side whose own frames built the spot.
usage: carve.py <project> model.ply out.ply [--every 5]
The project's build/ folder gives each side's final placement (placement.json, T_refined_*.json)."""
import os, sys, json, glob, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuse import read_calib, read_global_poses, frame_key

F = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame of each side")
    ap.add_argument("--margin", type=float, default=4.0, help="mm a ray must go past the spot to count as seeing through it")
    ap.add_argument("--frames", type=int, default=5, help="frames of one side that must see through a spot")
    ap.add_argument("--sides", type=int, default=2, help="different sides that must see through a spot")
    ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    import open3d as o3d
    work = os.path.join(a.project, "build")
    sides = sorted(d for d in glob.glob(os.path.join(a.project, "data", "*")) if os.path.exists(os.path.join(d, "cache", "pointyoink_track.json")))
    order = json.load(open(os.path.join(work, "placement.json")))["order"]
    m = o3d.io.read_triangle_mesh(a.model); V = np.asarray(m.vertices).astype(np.float32); T = np.asarray(m.triangles)
    W, H = a.width, a.height; t0 = time.time()
    through_sides = np.zeros(len(V), np.int16); per_side = {}
    for k, i in enumerate(order):
        tf = os.path.join(work, "T_refined_%d.json" % (k + 1))
        if not os.path.exists(tf): tf = os.path.join(work, "T_side%d_init.json" % (i + 1))
        S = np.array(json.load(open(tf))["matrix"], float)                 # this side's scan frame -> the model's frame, mm
        s = sides[i]; cal = read_calib(os.path.join(s, "param", "Pl.bin"))
        gp = read_global_poses(os.path.join(s, "cache", "pointyoink_register_pose.pose"))
        keys = sorted(gp)[::max(1, a.every)]
        through = np.zeros(len(V), np.int16); seen = np.zeros(len(V), np.int16)
        for key in keys:
            p = os.path.join(s, "cache", "frame_%03d_%04d.dph" % key)
            if not os.path.exists(p): continue
            d = np.frombuffer(open(p, "rb").read(), np.uint16)
            if d.size != W * H: continue
            d = d.reshape(H, W)
            Tc = np.linalg.inv(S @ gp[key] @ F).astype(np.float32)          # model frame -> this camera (OpenCV), mm
            Pc = V @ Tc[:3, :3].T + Tc[:3, 3]
            z = Pc[:, 2]; ok = z > 20
            u = np.rint(Pc[:, 0] / np.maximum(z, 1) * cal[0] + cal[2]).astype(np.int32)
            v = np.rint(Pc[:, 1] / np.maximum(z, 1) * cal[1] + cal[3]).astype(np.int32)
            ok &= (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
            j = np.nonzero(ok)[0]
            # the nearest depth in the 3 x 3 pixels around the ray: a spot right at a silhouette is not called seen through
            # because its neighbour pixel looks past the edge
            dz = np.full(len(j), 1e9, np.float32); any_d = np.zeros(len(j), bool)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    q = d[v[j] + dv, u[j] + du].astype(np.float32) * 0.1
                    dz = np.where(q > 0, np.minimum(dz, q), dz); any_d |= q > 0
            full = np.ones(len(j), bool)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1): full &= d[v[j] + dv, u[j] + du] > 0
            through[j[full & (dz > z[j] + a.margin)]] += 1
            seen[j[any_d & (np.abs(dz - z[j]) <= a.margin)]] += 1
        gone = (through >= a.frames) & (seen < a.frames)                    # this side looked through it and never saw it there
        through_sides += gone
        per_side[i + 1] = int(gone.sum())
    carve = through_sides >= a.sides
    n0 = len(T)
    m.remove_triangles_by_mask(carve[T].any(1)); m.remove_unreferenced_vertices()
    tri, cnt, _ = m.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
    keep = np.nonzero(cnt >= 0.03 * cnt.max())[0]
    m.remove_triangles_by_mask(~np.isin(tri, keep)); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, m)
    rep = dict(vertices=int(len(V)), carved_vertices=int(carve.sum()), triangles_before=int(n0), triangles_after=int(len(m.triangles)),
               looked_through_by_side=per_side, secs=round(time.time() - t0, 1))
    if a.report: json.dump(rep, open(a.report, "w"), indent=1)
    print("carved %d of %d points (%.1f%%): %d -> %d triangles, %.0f s; points each side looked through: %s"
          % (rep["carved_vertices"], rep["vertices"], 100.0 * rep["carved_vertices"] / max(1, rep["vertices"]), n0, rep["triangles_after"],
             rep["secs"], per_side))
    return 0


if __name__ == "__main__":
    sys.exit(main())
