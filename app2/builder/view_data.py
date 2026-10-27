# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's view loader: the "stages-load" thread reads stage results (PLY meshes and clouds, points.npz) with
stages.plyio in numpy, makes normals, face centroids (the elements a delete tests), colours per mode, and hands plain
arrays to the page (emit("stage_view", payload)). Newest request wins: a request made while one loads replaces the
one waiting. The last two results are kept by path (artifacts are content-addressed, so a path is its content), about
1.5 GB at most. Before loading a big result it checks MemAvailable (about 190 bytes a face with everything the view
and the selection need); too big: it says so (too_big) and the page shows the decimated view.ply read-only.

  L = ViewLoader(emit); L.request(dict(id=7, items=[dict(key="main", path=..., model=None, colour=None)], edit="main",
                                       mode="shaded", readonly=False)); L.close()
Payload: {id, items: [dict(key, kind, V, N, F, face_rgb | P, N, rgb, model, colour, n, path)], edit, elements (X, Nrm)
or None, n, readonly, too_big, error, secs}."""
import collections, os, threading, time, traceback
import numpy as np

BYTES_PER_FACE = 190
BYTES_PER_POINT = 64
CACHE_BYTES = int(1.5 * 1024 ** 3)


def mem_available():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"): return int(ln.split()[1]) * 1024
    except Exception: pass
    return None


def estimate_bytes(path):
    """From the header only: faces x 190 or points x 64."""
    from stages import plyio
    try:
        if path.endswith(".npz"): return plyio.points_count(path) * BYTES_PER_POINT
        c = plyio.counts(path)
        return c.get("face", 0) * BYTES_PER_FACE if c.get("face") else c.get("vertex", 0) * BYTES_PER_POINT
    except Exception: return 0


def hex_rgb(h): return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def side_rgb(S, side_ids, palette):
    """points.npz S (uint64 bit per side) -> the colour of the lowest side that saw each point."""
    S = np.asarray(S, np.uint64); out = np.zeros((len(S), 3), np.uint8); pal = np.array([hex_rgb(c) for c in palette], np.uint8)
    left = np.ones(len(S), bool)
    for b in range(min(64, max(1, len(side_ids)))):
        hit = left & ((S >> np.uint64(b)) & np.uint64(1)).astype(bool)
        out[hit] = pal[b % len(pal)]; left &= ~hit
    out[left] = 160
    return out


def load_file(path):
    """-> dict(kind="mesh", V, N, F, vrgb) or dict(kind="points", P, N, S, side_ids) as float32 / uint32 arrays."""
    from stages import plyio
    if path.endswith(".npz"):
        d = plyio.load_points(path)
        N = d.get("N"); P = np.ascontiguousarray(d["P"], np.float32)
        return dict(kind="points", P=P, N=np.ascontiguousarray(N if N is not None else np.tile([0, 0, 1], (len(P), 1)), np.float32),
                    S=d.get("S"), side_ids=[str(x) for x in d.get("side_ids", [])], vrgb=d.get("C"))
    d = plyio.read(path)
    v = d["vertex"]; V = np.ascontiguousarray(plyio.xyz(v), np.float32)
    vrgb = None
    if v.dtype.names and "red" in v.dtype.names:
        vrgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    F = d["faces"]
    Nf = plyio.normals(v)
    if F is None or not len(F):
        N = Nf if Nf is not None else np.tile([0, 0, 1.0], (len(V), 1))
        return dict(kind="points", P=V, N=np.ascontiguousarray(N, np.float32), S=None, side_ids=[], vrgb=vrgb)
    F = np.ascontiguousarray(F, np.uint32)
    N = Nf if Nf is not None else plyio.vertex_normals(V.astype(np.float64), F.astype(np.int64))
    return dict(kind="mesh", V=V, N=np.ascontiguousarray(N, np.float32), F=F, vrgb=vrgb)


class ViewLoader:
    def __init__(self, emit, cache=32):          # the Side view reloads every ghosted side when they fall out: CACHE_BYTES is the real cap
        self.emit = emit; self.cv = threading.Condition(); self.reqs = {}; self._stop = False
        self.cache = collections.OrderedDict(); self.cache_n = cache; self.loads = []      # paths read from disk (tests)
        self.elem_cache = collections.OrderedDict(); self._want_clear = False
        self.thread = threading.Thread(target=self._loop, name="stages-load", daemon=True); self.thread.start()

    def request(self, req):
        """Newest wins per slot (req["slot"]: "main" or "before")."""
        with self.cv: self.reqs[req.get("slot", "main")] = req; self.cv.notify()

    def close(self, timeout=3.0):
        with self.cv: self._stop = True; self.cv.notify()
        if timeout: self.thread.join(timeout)

    def clear(self):
        """Forget every cached result (a project switch): done by the loader thread before its next load."""
        with self.cv: self._want_clear = True; self.reqs.clear(); self.cv.notify()

    def _loop(self):
        while True:
            with self.cv:
                while not self.reqs and not self._stop and not self._want_clear: self.cv.wait(0.5)
                if self._stop: return
                if self._want_clear:
                    self.cache.clear(); self.elem_cache.clear(); self._want_clear = False
                    if not self.reqs: continue
                slot = "main" if "main" in self.reqs else next(iter(self.reqs))
                req = self.reqs.pop(slot)
            t0 = time.time()
            try: out = self._load(req)
            except Exception as e:
                traceback.print_exc(); out = {"error": "%s: %s" % (type(e).__name__, e)}
            out.update(id=req.get("id"), slot=req.get("slot", "main"), secs=round(time.time() - t0, 3), req=req)
            self.emit("stage_view", out)

    def _get(self, path):
        if path in self.cache: self.cache.move_to_end(path); return self.cache[path]
        d = load_file(path); self.loads.append(path)
        self.cache[path] = d
        while len(self.cache) > self.cache_n or (len(self.cache) > 1 and self._bytes() > CACHE_BYTES): self.cache.popitem(last=False)
        return d

    def _bytes(self):
        """Loaded results and their selection elements (centroids + normals: ~240 MB at 10 M faces) together."""
        return sum(a.nbytes for d in self.cache.values() for a in d.values() if isinstance(a, np.ndarray)) + \
            sum(a.nbytes for e in self.elem_cache.values() for a in e if isinstance(a, np.ndarray))

    def _load(self, req):
        from app2.qt import build_words as W
        from app2.qt.theme import T
        items = []; edit = req.get("edit"); elements = None; readonly = bool(req.get("readonly")); mode = req.get("mode", "shaded")
        if req.get("check_mem", True):
            need = sum(estimate_bytes(it["path"]) for it in req["items"] if it.get("path") and it["path"] not in self.cache)
            have = mem_available()
            if have is not None and need > 0.6 * have:
                return {"too_big": True, "need": need, "have": have, "items": [], "edit": None, "elements": None, "n": 0, "readonly": True}
        for it in req["items"]:
            p = it.get("path")
            if not p or not os.path.exists(p): continue
            d = self._get(p)
            colour = it.get("colour")
            if d["kind"] == "mesh":
                face_rgb = None
                if mode == "scanhere" and d.get("vrgb") is not None: face_rgb = d["vrgb"][d["F"][:,0]]
                elif mode == "quality" and it.get("rgb") is not None and len(it["rgb"]) == len(d["V"]): face_rgb = np.asarray(it["rgb"])[d["F"][:, 0]]
                elif mode == 'capture_distance' and it.get('rgb') is not None and len(it['rgb'])==len(d['F']): face_rgb=np.asarray(it['rgb'])
                guessed = os.path.join(os.path.dirname(p), "guessed.npy")
                if mode != 'capture_distance' and os.path.isfile(guessed):
                    mask = np.unpackbits(np.load(guessed))[:len(d["F"])].astype(bool)
                    if len(mask) == len(d["F"]) and mask.any():
                        face_rgb = np.tile(np.uint8(hex_rgb(T["mesh"])), (len(mask), 1)) if face_rgb is None else face_rgb.copy()
                        face_rgb[mask] = np.uint8([184, 102, 242])
                o = dict(key=it["key"], kind="mesh", V=d["V"], N=d["N"], F=d["F"], face_rgb=face_rgb, vertex_rgb=d.get("vrgb") if mode=="scan_color" else None, model=it.get("model"),
                         colour=colour if colour is not None else np.float32(hex_rgb(T["mesh"])) / 255.0, n=len(d["F"]), path=p)
            else:
                rgb = None
                if mode == "sides" and d.get("S") is not None: rgb = side_rgb(d["S"], d["side_ids"], W.SIDE_COLOURS)
                elif d.get("vrgb") is not None and mode in ("scanhere","scan_color"): rgb = d["vrgb"]
                elif mode in ("quality","capture_distance") and it.get("rgb") is not None and len(it["rgb"]) == len(d["P"]): rgb = np.asarray(it["rgb"])
                o = dict(key=it["key"], kind="points", P=d["P"], N=d["N"], rgb=rgb, model=it.get("model"),
                         colour=colour if colour is not None else np.float32(hex_rgb(T["cloud"])) / 255.0, n=len(d["P"]), path=p)
            items.append(o)
            if it["key"] == edit:
                if p in self.elem_cache: elements = self.elem_cache[p]; self.elem_cache.move_to_end(p)
                else:
                    from app2.builder.select_core import centroids
                    elements = centroids(d["V"], d["F"]) if d["kind"] == "mesh" else (d["P"], d["N"])
                    self.elem_cache[p] = elements
                    while len(self.elem_cache) > self.cache_n: self.elem_cache.popitem(last=False)
                    while len(self.elem_cache) > 1 and self._bytes() > CACHE_BYTES: self.elem_cache.popitem(last=False)
                    while len(self.cache) > 1 and self._bytes() > CACHE_BYTES: self.cache.popitem(last=False)
        n = len(elements[0]) if elements is not None else 0
        return {"items": items, "edit": edit if elements is not None else None, "elements": elements, "n": n, "readonly": readonly,
                "too_big": False}
