# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pair fits of lineup.pairs: the content-keyed pair cache (stages/cache/pairs.json) and the job that fits the
missing pairs with pair_all's own functions.

A fit is keyed by pair_token() (pair_all.VERSION plus a digest of the code that makes a fit: pair_all.py, support.py,
align.py and what it imports (registration_metrics.py), this module and the Open3D version), the fit params, both side
meshes' digests and both platter planes. So editing pair_all.py refits, while editing the place / refine code in
lineup.py does not throw away the fits. This module is kept apart from lineup.py for exactly that reason.

Fits adopted from build/ that an OLDER pair_all made are stored under old_token(<their VERSION>), never the current
token: lineup.pairs uses them only as a fallback (a VERSION bump does not by itself make a fit wrong), `run --force
lineup.pairs` skips the fallback and refits them, and `adopt --refit-old-pairs` removes them."""
import hashlib, json, os, re, sys, time

import numpy as np

from ..keys import H, jload
from . import script, scanner_dir, stage_file, lib_version

PAIR_CODE = ("seeded_sample.py", "pair_all.py", "support.py", "align.py", "registration_metrics.py")


def _dump(obj, path):
    with open(path, "w") as f: json.dump(obj, f)


class PairCache:
    def __init__(self, path):
        self.path = path
        try: self.d = jload(path)
        except Exception: self.d = {}

    @staticmethod
    def key(version, p, di, dj, pi, pj):
        return H([version, p["voxel"], p["samples"], p["max_dist"], di, dj, pi, pj])

    def get(self, version, p, di, dj, pi, pj):
        e = self.d.get(self.key(version, p, di, dj, pi, pj))
        if e: return dict(e, matrix=e["matrix"])
        e = self.d.get(self.key(version, p, dj, di, pj, pi))
        if e: return dict(e, matrix=np.linalg.inv(np.array(e["matrix"], float)).tolist(), reversed=True)
        return None

    def put(self, version, p, di, dj, pi, pj, entry):
        self.d[self.key(version, p, di, dj, pi, pj)] = entry

    def get_old(self, p, di, dj, pi, pj, below):
        """An adopted fit of an older pair_all (old_token(v), v < below, newest first), or None."""
        for v in range(int(below) - 1, -1, -1):
            e = self.get(old_token(v), p, di, dj, pi, pj)
            if e: return e
        return None

    def drop_old(self):
        """Remove every adopted older-version fit (adopt --refit-old-pairs). -> how many."""
        bad = [k for k, e in self.d.items() if isinstance(e, dict) and e.get("old_version")]
        for k in bad: del self.d[k]
        return len(bad)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp.%d" % os.getpid(); _dump(self.d, tmp); os.replace(tmp, self.path)


def pair_version():
    """pair_all.VERSION, read without importing Open3D in the engine process."""
    try:
        with open(script("pair_all.py")) as f: m = re.search(r"^VERSION\s*=\s*(\d+)", f.read(), re.M)
        return int(m.group(1))
    except Exception: return 0


def old_token(v):
    """The cache token of a fit adopted from build/ that pair_all VERSION v made (v older than the current one)."""
    return "%d:adopted-old" % int(v)


def pair_token():
    """What a cached fit was made by: "<VERSION>:<code digest>" (see the module doc)."""
    h = hashlib.blake2b(digest_size=8)
    for n in PAIR_CODE:
        try:
            with open(script(n), "rb") as f: h.update(n.encode() + b"\0" + hashlib.blake2b(f.read(), digest_size=16).digest())
        except OSError: h.update(n.encode() + b"\0-")
    with open(stage_file("methods/pairs.py"), "rb") as f: h.update(b"pairs.py\0" + hashlib.blake2b(f.read(), digest_size=16).digest())
    h.update(("open3d=" + lib_version("open3d")).encode())
    return "%d:%s" % (pair_version(), h.hexdigest())


def plane_of(out):
    """The platter plane (fuse.py plane.json dict) of a side.platter output, or None."""
    p = out.files.get("plane")
    if not p or not os.path.exists(p): return None
    return jload(p)


def plane_digest(out):
    p = out.files.get("plane")
    if not p or not os.path.exists(p): return "-"
    pl = jload(p); return H({"n": pl["n"], "d": pl["d"]})


def pairs_job(job):
    """Runs in its own process (cap 6): fit the missing pairs with pair_all's functions, save the cache after every
    fit (a cancel keeps the finished pairs)."""
    os.environ.setdefault("POINTYOINK_MEM_CAP_GB", "100000")
    sys.path.insert(0, scanner_dir())
    # Native RANSAC termination depends on thread arrival order. One thread repeats exactly but is about 3x slower,
    # so it is opt-in (tests and result-identity checks).
    if os.environ.get("POINTYOINK_STABLE_PAIRS") == "1": os.environ["OMP_NUM_THREADS"] = "1"
    import open3d as o3d
    import pair_all
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    p = job["params"]; V = job["version"]; S = job["sides"]; prepared = {}
    cache = PairCache(job["cache"])

    def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)
    for i, j, k in job["todo"]:
        a, b = S[i], S[j]
        emit("align", done=k, total=job["total"], msg="lining up side %d with side %d" % (b["n"], a["n"]))
        for s in (i, j):
            if s not in prepared: prepared[s] = pair_all.prepare(S[s]["mesh"], p["voxel"], p["samples"])
        beds_i = dict(plane=None if a["plane"] is None else {"n": a["plane"]["n"], "d": a["plane"]["d"]}, B=None, bed_id=None)
        beds_j = dict(plane=None if b["plane"] is None else {"n": b["plane"]["n"], "d": b["plane"]["d"]}, B=None, bed_id=None)
        o3d.utility.random.seed(0)
        cands = pair_all.candidates(prepared[i], prepared[j], max_dist=p["max_dist"])
        Pi, Pj = np.asarray(prepared[i]["B"].points), np.asarray(prepared[j]["B"].points)
        pn = "%d-%d" % (a["n"], b["n"])
        M, f, e, sup, bed, warns = pair_all.pick(cands, Pi, Pj, beds_i, beds_j, pn, prepared[i], prepared[j])
        for m in warns: emit("warn", msg=m)
        cache.put(V, p, a["digest"], b["digest"], a["plane_digest"], b["plane_digest"],
                  dict(matrix=np.asarray(M).tolist(), fitness=float(f), rmse=float(e), support=sup, warns=warns, t=time.time()))
        cache.save()
        emit("aligned", sides=pn, fit=round(f, 2), rmse_mm=round(e, 2), bed="-")
    return {"fitted": len(job["todo"])}
