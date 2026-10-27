#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""frame_quality.py: how well each spot of a model was scanned, from the raw frames that saw it.

  frame_quality.py <project> <sides.json> <model.ply> [--out q.npz] [--points N] [--per-side N]

For a sample of the model's vertices, every sampled frame that really saw the spot (its depth within 1.5 mm of the spot, in view,
facing the camera within 80 degrees) adds to: frames (how many), dist (mean camera distance, mm), angle (mean angle off head-on,
degrees). Also local roughness (RMS height over a 1.5 mm neighborhood, mm) so a bumpy spot can be matched to its cause.
Writes an npz: P (sample), frames, dist, angle, rough."""
import argparse, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def roughness(P, k=14, r=2.0):
    from scipy.spatial import cKDTree
    d, idx = cKDTree(P).query(P, k=k, distance_upper_bound=r, workers=-1)
    out = np.full(len(P), np.nan, np.float32)
    ok = np.isfinite(d).sum(1) >= 8
    for s in range(0, len(P), 200000):
        i = np.nonzero(ok[s:s + 200000])[0] + s
        if not len(i): continue
        nb = np.where(np.isfinite(d[i]), idx[i], i[:, None]); Q = P[nb]; m = np.isfinite(d[i])[..., None]
        c = (Q * m).sum(1) / np.maximum(m.sum(1), 1); X = (Q - c[:, None]) * m
        cov = np.einsum("nki,nkj->nij", X, X) / np.maximum(m.sum(1), 1)[..., None]
        w = np.linalg.eigvalsh(cov)[:, 0]; out[i] = np.sqrt(np.maximum(w, 0))
    return out


PALETTE = np.array([[255, 70, 50], [255, 200, 40], [90, 170, 255]], np.uint8)       # red: few frames, yellow: some, blue: many (as Coverage)
LOW, HIGH = 6, 20                                                               # frames at 150 sampled per side: under LOW rough, HIGH and up smooth


def levels(frames, per_side=150):
    k = per_side / 150.0
    return np.where(frames < LOW * k, 0, np.where(frames < HIGH * k, 1, 2))


def paint(V, sample, frames, per_side=150):
    """A color for every vertex: its nearest sampled vertex's level."""
    from scipy.spatial import cKDTree
    _, i = cKDTree(sample).query(V, workers=-1)
    return PALETTE[levels(frames, per_side)[i]]


def measure(project, sides, model, points=400000, per_side=150):
    """-> (V (all vertices or points, mm), npz dict P/frames/dist/angle/rough) for a model file (.ply mesh or points .npz)."""
    from stages import plyio
    from stages.frames import iter_frames
    if model.endswith(".npz"): d = plyio.load_points(model); V = np.asarray(d["P"], np.float64); N = np.asarray(d["N"], np.float64)
    else: V, F, N = plyio.read_mesh(model); V = np.asarray(V, np.float64)
    rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(V), min(points, len(V)), replace=False))
    P = V[sel]; Nv = np.asarray(N, np.float64)[sel]
    rough = roughness(P)
    cnt = np.zeros(len(P), np.int32); sd = np.zeros(len(P)); sa = np.zeros(len(P))
    for f in iter_frames(sides, project, max_per_side=per_side):
        R, t = f.T[:3, :3], f.T[:3, 3]; Q = (P - t) @ R                 # camera (OpenCV) coordinates, mm
        z = Q[:, 2]; fx, fy, cx, cy = f.cal
        ok = z > 20
        u = np.rint(Q[:, 0] * fx / np.where(ok, z, 1) + cx).astype(np.int64); v = np.rint(Q[:, 1] * fy / np.where(ok, z, 1) + cy).astype(np.int64)
        H, W = f.depth.shape; ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        i = np.nonzero(ok)[0]
        dz = f.depth[v[i], u[i]].astype(np.float64) * 0.1
        seen = (dz > 0) & (np.abs(dz - z[i]) < 1.5); i = i[seen]
        ray = Q[i] / np.linalg.norm(Q[i], axis=1, keepdims=True); nc = Nv[i] @ R
        cosang = np.abs((nc * ray).sum(1)); good = cosang > np.cos(np.radians(80)); i = i[good]
        cnt[i] += 1; sd[i] += z[i]; sa[i] += np.degrees(np.arccos(np.clip(cosang[good], 0, 1)))
    m = np.maximum(cnt, 1)
    return V, dict(P=P, frames=cnt, dist=np.where(cnt > 0, sd / m, np.nan), angle=np.where(cnt > 0, sa / m, np.nan), rough=rough)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("sides"); ap.add_argument("model")
    ap.add_argument("--out", default="quality.npz"); ap.add_argument("--rgb", help="also write a color per vertex/point (uint8 n x 3 .npy)")
    ap.add_argument("--points", type=int, default=400000); ap.add_argument("--per-side", type=int, default=150)
    a = ap.parse_args(argv); t0 = time.time()
    V, q = measure(a.project, a.sides, a.model, a.points, a.per_side)
    np.savez(a.out, **q)
    if a.rgb: np.save(a.rgb, paint(V, q["P"], q["frames"], a.per_side))
    print("frame quality: %d vertices sampled, %.0f s -> %s" % (len(q["P"]), time.time() - t0, a.out))


if __name__ == "__main__": main()
