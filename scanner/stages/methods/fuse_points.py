# SPDX-License-Identifier: AGPL-3.0-or-later
"""fused.points (port of the bake's revo_points fuse_points.py + process_points.merge): back-project every step-th
pixel of up to frames_per_side frames per side (stages.frames), weight each point by cos^2 x (z0/z)^2 x edge ramp,
reduce per (cell = point_distance, normal bin) in a pool of per-side workers, then sum over the sides and keep entries
seen by at least min_frames frames and min_sides sides whose normals agree. Output points.npz (P, N, W, NF, S bitmask).
Default ops: isolation 1 %, overlap 0.21 mm, smooth 10 x2 (the bake's revo_points; placeholders until RESULTS.md)."""
import os, time

import numpy as np

from .. import params as PR
from . import Method, register

OFF = 1 << 18
CFG = {}


@register
class FusedPoints(Method):
    node = "fused"; name = "points"; label = "Points (Revo order)"; version = 1; produces = "points"; mem_gb = 10
    # frames.py reads the frames with fuse.py's read_calib / read_global_poses / frame_key: both are in the key
    scripts = ("base_cut.py", "fuse.py", "fusion_memory.py",); modules = ("methods/fuse_points.py", "frames.py", "side_fusion_cache.py", "local_points_cache.py", "tools/edits.py", "tools/merge_cells.py", "plyio.py"); libs = ("numpy", "scipy")
    default_ops = [{"tool": "isolation", "params": {"rate": 1.0}}, {"tool": "overlap", "params": {"distance": 0.21}},
                   {"tool": "smooth", "params": {"strength": 10.0, "times": 2}},
                   {"tool": "match_cleaned_sides", "params": {"distance_mm": 1.5}, "on": False}]

    def execute(self, ctx):
        ctx.legacy("fuse", msg="fusing all sides into one model")
        job = {"sides_json": ctx.inputs["lineup"].files["sides"], "project": ctx.project.root, "params": ctx.params,
               "side_deletes": ctx.step.leaf.get("side_deletes", {}), "stages": ctx.project.stages,
               "side_cache": os.environ.get("POINTYOINK_SIDE_FUSION_CACHE", "1") != "0"}

        def on_line(ln):
            if ln.startswith("side "):
                try:
                    k, n = ln.split("[", 1)[1].split("]", 1)[0].split("/")
                    ctx.progress(int(k), int(n), ln.strip())
                except (IndexError, ValueError): pass
        r = ctx.job("methods.fuse_points:job", job, mem_gb=self.mem_gb, on_line=on_line)
        return {"outputs": {"points": "points.npz"}, "stats": r}

    def summary(self, p):
        return "points %s mm, step %d, %s frames/side" % (PR.fmt(p["point_distance"]), p["step"], p["frames_per_side"] or "all")


def camera_points(f, step, edge_px, z0):
    from scipy import ndimage
    fx, fy, cx, cy = f.cal; H, W = f.depth.shape
    valid_full = f.depth > 0
    edist = ndimage.distance_transform_edt(valid_full)[::step, ::step]
    d = f.depth[::step, ::step]
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
    cosv = np.abs(s)
    w = cosv ** 2 * (z0 / Pc[:, 2]) ** 2
    if edge_px > 0: w *= np.clip(edist[ok] / edge_px, 0.15, 1.0)
    return Pc, Nc, w


def frame_points(f, step, edge_px, z0):
    Pc, Nc, w = camera_points(f, step, edge_px, z0)
    R, t = f.T[:3, :3], f.T[:3, 3]
    return Pc @ R.T + t, Nc @ R.T, w


def keys_of(P, N, cell):
    ijk = np.floor(P / cell).astype(np.int64) + OFF
    a = np.argmax(np.abs(N), 1); b = a * 2 + (N[np.arange(len(N)), a] < 0)
    return (((ijk[:, 0] << 19) | ijk[:, 1]) << 19 | ijk[:, 2]) << 3 | b


def reduce_rows(K, cols):
    u, inv = np.unique(K, return_inverse=True)
    return u, [np.bincount(inv, weights=c, minlength=len(u)) for c in cols]


def _worker_threads():
    # The process pool already supplies parallelism. Small matrix products need no nested BLAS pool.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def side_worker(sid):
    from ..frames import iter_frames, F
    from ..side_fusion_cache import load_table, save_table
    from ..local_points_cache import batches, Writer
    import zipfile
    c = CFG; t0 = time.time()
    table = c.get("tables", {}).get(sid)
    if table:
        hit = load_table(*table)
        if hit is not None:
            n, K, C = hit
            return sid, n, K, C, time.time() - t0, True, False
    bufK, bufC = [], []; accK = None; accC = None; nfr = 0

    def flush():
        nonlocal accK, accC, bufK, bufC
        if not bufK: return
        K = np.concatenate(bufK + ([accK] if accK is not None else []))
        cols = [np.concatenate([b[j] for b in bufC] + ([accC[j]] if accC is not None else [])) for j in range(9)]
        accK, accC = reduce_rows(K, cols); bufK, bufC = [], []
    def accumulate(P, N, w):
        nonlocal nfr
        K = keys_of(P, N, c["cell"])
        u, s = reduce_rows(K, [w, w * P[:, 0], w * P[:, 1], w * P[:, 2], w * N[:, 0], w * N[:, 1], w * N[:, 2], w * w])
        bufK.append(u); bufC.append(s + [np.ones(len(u))])
        nfr += 1
        if len(bufK) >= 40: flush()

    local = c.get("locals", {}).get(sid); local_hit = False
    side = next(s for s in c["sides_json"]["sides"] if s["id"] == sid)
    Tside = np.asarray(side["T"], float)
    if local and os.path.exists(os.path.join(local[0], "meta.json")):
        try:
            for Pc, Nc, w, pose in batches(*local):
                M = Tside @ pose @ F
                accumulate(Pc @ M[:3, :3].T + M[:3, 3], Nc @ M[:3, :3].T, w)
            local_hit = True
        except (OSError, ValueError, KeyError, EOFError, ImportError, RuntimeError, zipfile.BadZipFile) as error:
            print("local side cache %s is unusable (%s); rebuilding" % (sid, type(error).__name__), flush=True)
            bufK, bufC, accK, accC, nfr = [], [], None, None, 0
    if not local_hit:
        writer = Writer(*local) if local else None
        if writer:
            from fuse import read_global_poses
            poses = read_global_poses(os.path.join(c["project"], side["folder"], "cache", "pointyoink_register_pose.pose"))
        try:
            for f in iter_frames(c["sides_json"], c["project"], max_per_side=c["max_per_side"], only_sides=[sid],
                                 side_deletes=c["side_deletes"], stages_dir=c["stages"]):
                Pc, Nc, w = camera_points(f, c["step"], c["edge_px"], c["z0"])
                if len(Pc) == 0: continue
                if writer: writer.append(Pc, Nc, w, poses[f.key])
                accumulate(Pc @ f.T[:3, :3].T + f.T[:3, 3], Nc @ f.T[:3, :3].T, w)
            if writer: writer.finish()
        finally:
            if writer: writer.abort()
    flush()
    if accK is None: accK = np.zeros(0, np.int64); accC = [np.zeros(0)] * 9
    C = np.stack(accC, 1).astype(np.float64)
    if table: save_table(table[0], table[1], nfr, accK, C)
    return sid, nfr, accK, C, time.time() - t0, False, local_hit


def job(job):
    from multiprocessing import Pool
    from ..frames import load_sides
    from ..tools.merge_cells import merge
    from .. import plyio
    from ..side_fusion_cache import table_paths
    from ..local_points_cache import local_paths
    p = job["params"]; t0 = time.time()
    sj = load_sides(job["sides_json"]); ids = [s["id"] for s in sj["sides"]]
    if len(ids) > 64: raise RuntimeError("fused.points supports at most 64 sides (side bitmask)")
    if not ids: raise ValueError("fused.points needs at least one side")
    CFG.update(sides_json=sj, project=job["project"], cell=p["point_distance"], step=p["step"], max_per_side=p["frames_per_side"],
               edge_px=p["edge_px"], z0=p["z0"], side_deletes=job.get("side_deletes", {}), stages=job["stages"])
    CFG["tables"] = table_paths(sj, job["project"], job["stages"], p, CFG["side_deletes"]) if job.get("side_cache", True) else {}
    CFG["locals"] = local_paths(sj, job["project"], job["stages"], p, CFG["side_deletes"]) if job.get("side_cache", True) and os.environ.get("POINTYOINK_LOCAL_POINTS_CACHE", "1") != "0" else {}
    res, nfr, cached, local_cached = {}, {}, [], []
    with Pool(max(1, min(int(p["workers"]), len(ids))), initializer=_worker_threads) as pool:
        for k, (sid, n, K, C, secs, hit, local_hit) in enumerate(pool.imap_unordered(side_worker, ids)):
            print("side [%d/%d] %s: %d frames, %d entries, %.0f s%s" % (k + 1, len(ids), sid, n, len(K), secs, " (cached)" if hit else " (local cache)" if local_hit else ""), flush=True)
            if hit: cached.append(sid)
            if local_hit: local_cached.append(sid)
            res[sid] = (K, C); nfr[sid] = n
    # concatenate in sides.json order, whatever order the workers finished in (same sums, same bytes)
    K = np.concatenate([res[s][0] for s in ids]); C = np.concatenate([res[s][1] for s in ids])
    B = np.concatenate([np.full(len(res[s][0]), np.uint64(1) << np.uint64(k), np.uint64) for k, s in enumerate(ids)])
    min_sides = max(1, min(int(p["min_sides"]), len(ids)))     # "seen by 2 sides" on a 1-side scan kept 0 points (and crashed later)
    P, N, W, NF, S = merge(K, C, None, B, p["min_frames"], min_sides, p["normal_agree"], p.get("rescue_mm", 0.0))
    plyio.save_points(os.path.join(job["out_dir"], "points.npz"), P, N, W, NF, S, ids, p["point_distance"])
    one = float(np.mean([bin(int(x)).count("1") == 1 for x in S[: min(len(S), 200000)]])) if len(S) else 0.0
    per_side = {sid: int(((S >> np.uint64(k)) & np.uint64(1)).sum()) for k, sid in enumerate(ids)}
    print("fused %d points from %d sides in %.0f s" % (len(P), len(ids), time.time() - t0), flush=True)
    return {"points": int(len(P)), "entries": int(len(K)), "frames": nfr, "per_side": per_side, "one_side_share": round(one, 3),
            "bbox_mm": [round(float(x), 1) for x in (P.max(0) - P.min(0))] if len(P) else None,
            "side_tables_cached": [sid for sid in ids if sid in cached], "side_tables_built": [sid for sid in ids if sid not in cached],
            "side_local_points_cached": [sid for sid in ids if sid in local_cached]}
