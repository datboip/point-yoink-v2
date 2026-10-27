# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lossless point batches in a side's coordinate system, independent of line-up placement.

Camera coordinates plus the saved camera-to-side matrix represent local side points. Keeping
that factorization preserves the original (T_side @ pose) @ F multiplication order exactly.
"""
import glob
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .frames import pick_frames
from .keys import DigestMemo, H

VERSION = 1


def local_paths(sj, project, stages, params, deletes):
    memo = DigestMemo(os.path.join(stages, "cache", "side-fusion-digests.json"))
    settings = {k: params[k] for k in ("step", "frames_per_side", "edge_px", "z0")}
    here = os.path.dirname(__file__)
    code = [memo.digest(os.path.join(here, p)) for p in
            ("local_points_cache.py", "methods/fuse_points.py", "frames.py", "tools/edits.py", "../../fuse.py")]
    paths = {}
    for side in sj["sides"]:
        sid = side["id"]; folder = os.path.join(project, side["folder"]); cache = os.path.join(folder, "cache")
        frames = pick_frames(sorted(glob.glob(os.path.join(cache, "*.dph"))), params["frames_per_side"])
        files = [(os.path.basename(p), memo.digest(p), os.path.exists(p[:-4] + ".inf")) for p in frames]
        ops = deletes.get(sid, [])
        regions = [(op["region"], memo.digest(os.path.join(stages, op["region"])))
                   for op in ops if op.get("on", True) and op.get("shape") == "mask"]
        key = H({"version": VERSION, "code": code, "settings": settings, "side": sid,
                 "libs": {name: version(name) for name in ("numpy", "scipy")},
                 "plane": side.get("plane"), "size": side.get("size", [800, 600]),
                 "frames": files, "poses": memo.digest(os.path.join(cache, "pointyoink_register_pose.pose")),
                 "calibration": memo.digest(os.path.join(folder, "param", "Pl.bin")), "deletes": ops, "regions": regions})
        paths[sid] = (os.path.join(stages, "cache", "side-local-points", sid, key), key)
    memo.save()
    return paths


def batches(path, key):
    """Validate the manifest first, then stream ordered batches with one frame in memory."""
    folder = Path(path); meta = json.loads((folder / "meta.json").read_text())
    if meta["version"] != VERSION or meta["key"] != key: raise ValueError("local cache key mismatch")
    if meta["frames"] != len(meta["batches"]): raise ValueError("local cache frame count mismatch")
    for item in meta["batches"]:
        name = item["file"]
        if Path(name).name != name: raise ValueError("invalid local cache batch name")
        raw = (folder / name).read_bytes()
        if meta["codec"] == "lz4":
            import lz4.frame
            raw = lz4.frame.decompress(raw)
        elif meta["codec"] != "raw": raise ValueError("unsupported local cache codec")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            P, N, W, pose = (data[k] for k in ("P", "N", "W", "pose"))
        n = item["points"]
        if (P.shape != (n, 3) or N.shape != P.shape or W.shape != (n,) or pose.shape != (4, 4) or
                any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (P, N, W, pose))):
            raise ValueError("invalid local cache arrays")
        yield P, N, W, pose
    os.utime(folder / "meta.json", None)


class Writer:
    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix=".local-", dir=self.path.parent))
        self.items = []
        try:
            import lz4.frame
            self.codec = "lz4"
        except ImportError:
            self.codec = "raw"

    def append(self, P, N, W, pose):
        buffer = io.BytesIO()
        np.savez(buffer, P=P, N=N, W=W, pose=pose)
        raw = buffer.getvalue()
        if self.codec == "lz4":
            import lz4.frame
            raw = lz4.frame.compress(raw, compression_level=0, content_checksum=True)
        name = "%04d.batch" % len(self.items)
        (self.temp / name).write_bytes(raw)
        self.items.append(dict(file=name, points=len(P)))

    def finish(self):
        meta = dict(version=VERSION, key=self.key, codec=self.codec, frames=len(self.items),
                    points=sum(item["points"] for item in self.items), batches=self.items)
        (self.temp / "meta.json").write_text(json.dumps(meta))
        if self.path.exists(): shutil.rmtree(self.path)
        os.replace(self.temp, self.path)
        self.temp = None
        variants = sorted((p for p in self.path.parent.iterdir() if (p / "meta.json").is_file()),
                          key=lambda p: (p / "meta.json").stat().st_mtime, reverse=True)
        for old in variants[2:]:
            if old != self.path: shutil.rmtree(old)

    def abort(self):
        if self.temp is not None: shutil.rmtree(self.temp, ignore_errors=True)


def collect(stages, budget_bytes, node_bytes):
    """Share the stage GC budget; protect one most recently used local variant per side.

    Like current/pinned node artifacts, protected local inputs may exceed the soft budget.
    World tables and older local variants are disposable and are evicted oldest first.
    """
    root = Path(stages) / "cache"; candidates = []; protected = set(); total = 0
    local = root / "side-local-points"
    for side in local.iterdir() if local.exists() else ():
        variants = []
        for folder in side.iterdir():
            meta = folder / "meta.json"
            if not folder.is_dir(): continue
            size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            if not meta.exists():
                age = folder.stat().st_mtime
                total += size; candidates.append((age, str(folder), size))
                if time.time() - age < 3600: protected.add(str(folder))
                continue
            variants.append((meta.stat().st_mtime, str(folder), size))
            total += size
        if variants:
            newest = max(variants); protected.add(newest[1]); candidates.extend(variants)
    world = root / "side-fusion"
    for p in world.glob("*/*.npz") if world.exists() else ():
        size = p.stat().st_size; total += size; candidates.append((p.stat().st_mtime, str(p), size))
    removed = []
    for _age, name, size in sorted(candidates):
        if node_bytes + total <= budget_bytes: break
        if name in protected: continue
        p = Path(name)
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
        total -= size; removed.append(name)
    protected_bytes = sum(size for _age, name, size in candidates if name in protected)
    return dict(bytes=total, removed=removed, protected=len(protected), protected_bytes=protected_bytes)
