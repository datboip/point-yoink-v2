# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reusable point fusion accumulators, before the order-dependent side bitmask and global merge."""
import glob
from importlib.metadata import version
import os
import tempfile
import zipfile

import numpy as np

from .frames import pick_frames
from .keys import DigestMemo, H

VERSION = 1


def table_paths(sj, project, stages, params, deletes):
    """Hash every selected depth file, pose table, calibration, cut, placement and accumulation setting.

    A shared digest memo avoids rereading unchanged depth bytes. Worker count, side order and global
    merge thresholds do not affect an individual table. Region masks are content dependencies too.
    """
    memo = DigestMemo(os.path.join(stages, "cache", "side-fusion-digests.json"))
    settings = {k: params[k] for k in ("point_distance", "step", "frames_per_side", "edge_px", "z0")}
    here = os.path.dirname(__file__)
    code = [memo.digest(os.path.join(here, p)) for p in
            ("side_fusion_cache.py", "local_points_cache.py", "methods/fuse_points.py", "frames.py", "tools/edits.py", "../../fuse.py")]
    paths = {}
    for s in sj["sides"]:
        sid = s["id"]; folder = os.path.join(project, s["folder"]); cache = os.path.join(folder, "cache")
        frames = pick_frames(sorted(glob.glob(os.path.join(cache, "*.dph"))), params["frames_per_side"])
        files = [(os.path.basename(p), memo.digest(p), os.path.exists(p[:-4] + ".inf")) for p in frames]
        ops = deletes.get(sid, [])
        regions = [(op["region"], memo.digest(os.path.join(stages, op["region"])))
                   for op in ops if op.get("on", True) and op.get("shape") == "mask"]
        key = H({"version": VERSION, "code": code, "libs": {name: version(name) for name in ("numpy", "scipy")}, "settings": settings, "side": sid,
                 "T": s["T"], "plane": s.get("plane"), "size": s.get("size", [800, 600]),
                 "frames": files, "poses": memo.digest(os.path.join(cache, "pointyoink_register_pose.pose")),
                 "calibration": memo.digest(os.path.join(folder, "param", "Pl.bin")), "deletes": ops, "regions": regions})
        paths[sid] = (os.path.join(stages, "cache", "side-fusion", sid, key + ".npz"), key)
    memo.save()
    return paths


def load_table(path, key):
    """A missing, truncated or incompatible table is a miss, never a partially used result."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"].item()) != key or int(z["version"].item()) != VERSION: return None
            K, C, n = z["K"], z["C"], int(z["frames"].item())
            if K.dtype != np.int64 or K.ndim != 1 or C.dtype != np.float64 or C.shape != (len(K), 9): return None
            if n < 0 or not np.isfinite(C).all() or (len(K) > 1 and not np.all(K[1:] > K[:-1])): return None
            return n, K, C
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_table(path, key, n, K, C):
    """Atomic replace also makes an interrupted build's completed sides reusable."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".table-", suffix=".npz", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, version=VERSION, key=key, frames=n, K=K, C=C)
        os.replace(tmp, path)
        # Placement changes can produce large tables. Retain two variants per side for undo/retry.
        versions = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.npz")), key=os.path.getmtime, reverse=True)
        for old in versions[2:]:
            if old != path:
                try: os.unlink(old)
                except OSError: pass
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
