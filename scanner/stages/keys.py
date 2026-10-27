# SPDX-License-Identifier: AGPL-3.0-or-later
"""Content keys: canonical JSON, blake2b-128, and a memo of file digests (so a 40 MB mesh is hashed once, not on every
plan). Values are coerced before hashing: floats rounded to 9 decimals (0.1 + 0.2 == 0.3), ints stay ints."""
import hashlib, json, os, threading

ENGINE_VERSION = 2


def coerce(x):
    """The value as it is hashed: floats rounded to 9 decimals (an integral float stays a float), containers walked."""
    if isinstance(x, bool) or x is None or isinstance(x, str): return x
    if isinstance(x, int): return x
    if isinstance(x, float): return round(x, 9) + 0.0
    if isinstance(x, dict): return {str(k): coerce(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [coerce(v) for v in x]
    try:                                    # numpy scalars and arrays
        import numpy as np
        if isinstance(x, np.ndarray): return coerce(x.tolist())
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): return coerce(float(x))
        if isinstance(x, np.bool_): return bool(x)
    except Exception: pass
    return str(x)


def canonical(x):
    return json.dumps(coerce(x), sort_keys=True, separators=(",", ":"))


def H(x):
    """blake2b-128 of the canonical JSON of x, as 32 hex characters."""
    return hashlib.blake2b(canonical(x).encode(), digest_size=16).hexdigest()


def key16(key): return key[:16]


def file_digest_raw(path, bufsize=1 << 22):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return "b2:" + h.hexdigest()


def md5_file(path):
    """md5 of a file, '-' when it does not exist (build_project.py's digest())."""
    if not os.path.exists(path): return "-"
    with open(path, "rb") as f: return hashlib.md5(f.read()).hexdigest()


class DigestMemo:
    """{realpath: [size, mtime_ns, inode, digest(, geo digest)]} kept in stages/cache/digests.json. The geo digest (an
    order-independent digest of a mesh, plyio.geo_digest) is what downstream keys use, so a re-run that writes the same
    surface in another vertex order still cuts off; the byte digest stays the storage / exact-replay identity."""
    GEO_IN_PROCESS = 1_500_000        # faces: bigger meshes are digested in a capped job (geo_fn)

    def __init__(self, path, read_only=False):
        self.path = path; self.lock = threading.Lock(); self.dirty = False; self.read_only = read_only
        try:
            with open(path) as f: self.memo = json.load(f)
        except Exception: self.memo = {}

    def _hit(self, rp, st):
        e = self.memo.get(rp)
        if e and e[0] == st.st_size and e[1] == st.st_mtime_ns and e[2] == st.st_ino: return e
        return None

    def digest(self, path, real=False, st=None):
        """real: the caller already made the path real (one realpath per directory beats one per frame); st: its stat, when a
        directory scan already has it."""
        rp = path if real else os.path.realpath(path); st = st or os.stat(rp)
        with self.lock:
            e = self._hit(rp, st)
            if e: return e[3]
        d = file_digest_raw(rp)
        with self.lock:
            self.memo[rp] = [st.st_size, st.st_mtime_ns, st.st_ino, d]; self.dirty = True
        return d

    def geo(self, path, geo_fn=None):
        """Order-independent digest of a mesh PLY, or None (not a triangle mesh, or too big and no geo_fn)."""
        rp = os.path.realpath(path); bd = self.digest(rp); st = os.stat(rp)
        with self.lock:
            e = self._hit(rp, st)
            if e and len(e) > 4: return e[4]
        from .plyio import geo_digest, counts
        try:
            nf = counts(rp).get("face", 0)
            if not nf: g = None                                  # not a triangle mesh: its bytes are its identity
            elif nf <= self.GEO_IN_PROCESS: g = geo_digest(rp)
            elif geo_fn is None: return None                     # too big here and no job to ask: not remembered
            else: g = "job"
        except Exception: return None
        if g == "job":
            # the capped job: a failure or a cancel RAISES (the caller then stores nothing), so an artifact is never stored
            # without the geometry digest its downstream keys use
            g = geo_fn(rp)
            if not g: raise RuntimeError("no geometry digest for %s" % rp)
        with self.lock:
            self.memo[rp] = [st.st_size, st.st_mtime_ns, st.st_ino, bd, g]; self.dirty = True
        return g

    def forget_missing(self):
        with self.lock:
            for k in [k for k in self.memo if not os.path.exists(k)]: del self.memo[k]; self.dirty = True

    def save(self):
        if not self.dirty or self.read_only: return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp.%d" % os.getpid()
        with self.lock:
            with open(tmp, "w") as f: json.dump(self.memo, f)
            self.dirty = False
        os.replace(tmp, self.path)


def output_digest(file_digests):
    """blake2b-128 over (sorted file name, file digest) of an artifact's declared outputs."""
    return "b2:" + H(sorted([n, d] for n, d in file_digests.items()))


def jload(path):
    with open(path) as f: return json.load(f)


def jdump(obj, path, **kw):
    with open(path, "w") as f: json.dump(obj, f, **kw)
