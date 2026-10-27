# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frame access for fused methods (port of the bake's frames.py): every depth frame of the active sides, cleaned exactly
like fuse.py (depth jumps > 3 mm and surfaces seen more than 80 deg edge-on dropped, one-pixel skirt), platter cut with
the side's plane, and its camera -> MODEL pose. Reads the engine's sides.json (lineup.adjust output) instead of build/.

    from stages.frames import sides, iter_frames, backproject
    for f in iter_frames(sides_json, project_root, max_per_side=300):
        P, N, view_cos = backproject(f)
        f.side (id), f.bit (index into sides.json order), f.key, f.depth (H x W uint16, 0.1 mm), f.cal, f.T, f.cam

Conventions: pose table <side>/cache/pointyoink_register_pose.pose (scanner camera -> side scan, mm);
F = diag(1,-1,-1,1) OpenCV camera <-> scanner camera; camera(OpenCV) -> model = T_side @ pose @ F."""
import glob, json, os, sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fuse import read_calib, read_global_poses, frame_key          # noqa: E402
from .keys import jload, jdump

F = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Frame:
    side: str; bit: int; key: tuple; path: str; depth: np.ndarray; cal: tuple; T: np.ndarray; cam: np.ndarray


def load_sides(sides_json):
    return jload(sides_json) if isinstance(sides_json, str) else sides_json


def sides(sides_json, project_root):
    """[(id, folder (abs), T 4x4 side->model, plane dict or None, (W, H))] in sides.json order."""
    sj = load_sides(sides_json); out = []
    for s in sj["sides"]:
        out.append((s["id"], os.path.join(project_root, s["folder"]), np.array(s["T"], float), s.get("plane"), tuple(s.get("size", (800, 600)))))
    return out


_grid = {}


def clean(d, cal, jump=3.0, grazing=80.0):
    """fuse.py's per-frame clean-up, same defaults. d: H x W uint16 (0.1 mm). Returns a cleaned copy."""
    d = d.copy(); fx, fy, cx, cy = cal
    z = d.astype(np.float32) * 0.1; valid = z > 0
    j = np.zeros_like(valid)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        zs = np.roll(z, (dy, dx), axis=(0, 1)); vs = np.roll(valid, (dy, dx), axis=(0, 1)); j |= vs & (np.abs(zs - z) > jump)
    e = j | ~valid
    e = e | np.roll(e, 1, 0) | np.roll(e, -1, 0) | np.roll(e, 1, 1) | np.roll(e, -1, 1)
    if d.shape not in _grid:
        u, v = np.meshgrid(np.arange(d.shape[1], dtype=np.float32), np.arange(d.shape[0], dtype=np.float32)); _grid[d.shape] = (u, v)
    u, v = _grid[d.shape]; X = (u - cx) * z / fx; Y = (v - cy) * z / fy
    dXu = np.roll(X, -1, 1) - np.roll(X, 1, 1); dYu = np.roll(Y, -1, 1) - np.roll(Y, 1, 1); dZu = np.roll(z, -1, 1) - np.roll(z, 1, 1)
    dXv = np.roll(X, -1, 0) - np.roll(X, 1, 0); dYv = np.roll(Y, -1, 0) - np.roll(Y, 1, 0); dZv = np.roll(z, -1, 0) - np.roll(z, 1, 0)
    nx = dYu * dZv - dZu * dYv; ny = dZu * dXv - dXu * dZv; nz = dXu * dYv - dYu * dXv
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9; rl = np.sqrt(X * X + Y * Y + z * z) + 1e-9
    cosang = np.abs((nx * X + ny * Y + nz * z) / (nl * rl))
    d[e | (valid & (cosang < np.cos(np.radians(grazing))))] = 0
    return d


def cut_plane(d, pose, cal, P):
    """Zero pixels on the platter side of the side's plane (plane in the side-scan frame), like fuse.py."""
    fx, fy, cx, cy = cal; v, u = np.nonzero(d); z = d[v, u].astype(np.float64) * 0.1
    X = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
    M = pose @ F; Xw = X @ M[:3, :3].T + M[:3, 3]; h = Xw @ np.asarray(P["n"], float)
    from scanner.base_cut import removed
    bad = removed(Xw, P)
    d = d.copy(); d[v[bad], u[bad]] = 0; return d


def pick_frames(dphs, max_per_side):
    if max_per_side and len(dphs) > max_per_side:
        return [dphs[k] for k in np.unique(np.linspace(0, len(dphs) - 1, max_per_side).round().astype(int))]
    return dphs


def iter_frames(sides_json, project_root, max_per_side=300, only_sides=None, max_depth_mm=600.0,
                side_deletes=None, stages_dir=None):
    """Every usable frame, in sides.json order. Frames the tracker lost (absent from the pose table) are skipped."""
    for bit, (sid, folder, Tside, plane, size) in enumerate(sides(sides_json, project_root)):
        if only_sides and sid not in only_sides: continue
        W, H = size
        cache = os.path.join(folder, "cache"); cal = read_calib(os.path.join(folder, "param", "Pl.bin"))
        if W != 800:
            k = W / 800.0; cal = (cal[0] * k, cal[1] * k, cal[2] * k, cal[3] * k)
        gp = read_global_poses(os.path.join(cache, "pointyoink_register_pose.pose"))
        for p in pick_frames(sorted(glob.glob(os.path.join(cache, "*.dph"))), max_per_side):
            key = frame_key(p); pose = gp.get(key)
            if pose is None or not os.path.exists(p[:-4] + ".inf"): continue
            raw = np.fromfile(p, np.uint16)
            if raw.size != W * H: continue
            d = raw.reshape(H, W).copy(); d[d * 0.1 > max_depth_mm] = 0
            d = clean(d, cal)
            if plane is not None: d = cut_plane(d, pose, cal, plane)
            ops = (side_deletes or {}).get(sid)
            if ops:
                from .tools.edits import mask_side_depth
                d = mask_side_depth(d, pose, cal, ops, stages_dir)
            T = Tside @ pose @ F
            yield Frame(sid, bit, key, p, d, cal, T, T[:3, 3].copy())


def backproject(f, step=1):
    """Points (mm, model frame), unit normals (facing the camera) and |cos| of the view angle, every step-th pixel."""
    fx, fy, cx, cy = f.cal; H, W = f.depth.shape; d = f.depth[::step, ::step]
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = d.astype(np.float64) * 0.1
    X = (u - cx) * z / fx; Y = (v - cy) * z / fy
    C = np.stack([X, Y, z], -1)
    du = np.roll(C, -1, 1) - np.roll(C, 1, 1); dv = np.roll(C, -1, 0) - np.roll(C, 1, 0)
    n = np.cross(du, dv); nl = np.linalg.norm(n, axis=-1)
    ok = (z > 0) & (np.roll(z, -1, 1) > 0) & (np.roll(z, 1, 1) > 0) & (np.roll(z, -1, 0) > 0) & (np.roll(z, 1, 0) > 0) & (nl > 1e-9)
    Pc = C[ok]; Nc = n[ok] / nl[ok][:, None]
    ray = Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
    s = np.einsum("ij,ij->i", Nc, ray); Nc[s > 0] *= -1
    R, t = f.T[:3, :3], f.T[:3, 3]
    return Pc @ R.T + t, Nc @ R.T, np.abs(s)
