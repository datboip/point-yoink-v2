# SPDX-License-Identifier: AGPL-3.0-or-later
"""Clone: copy what lies in a cylinder or box on the surface at a source spot, and put it in place of what lies in the same shape at a
destination spot. Each spot is a point and the surface direction there; the copy is lined up along the two directions, turned about the
destination's normal by `rotation`, and optionally mirrored. The copied geometry is made up from the scan itself: a repair for repeated
features (holes, bosses, ribs), not a measurement."""
import json

import numpy as np

from ..methods import Tool, register
from .common import run_tool_job
from .edits import apply_mask

PER_POINT = ("P", "N", "W", "NF", "S")


def _spot(text, what):
    try: c, n = json.loads(text); c = np.asarray(c, np.float64); n = np.asarray(n, np.float64)
    except (TypeError, ValueError): raise ValueError("Pick the %s spot on the model first." % what) from None
    if c.shape != (3,) or n.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
        raise ValueError("Pick the %s spot on the model first." % what)
    return c, n / np.linalg.norm(n)


def frame(n, along=None):
    """Orthonormal columns (e1, e2, n) for the surface direction n. e1 follows `along` (the way from the source to the destination,
    flattened onto the surface) when there is one: a mirror then flips left and right as seen along that line."""
    e1 = None
    if along is not None:
        t = np.asarray(along, np.float64) - n * (np.asarray(along, np.float64) @ n)
        if np.linalg.norm(t) > 1e-6: e1 = t / np.linalg.norm(t)
    if e1 is None:
        a = np.eye(3)[int(np.argmin(np.abs(n)))]; e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return np.column_stack((e1, np.cross(n, e1), n))


def inside(X, c, F, p):
    """bool per row of X: inside the shape lying on the surface at c (u, v across it, w along the normal: some depth below, a little above)."""
    u, v, w = ((np.asarray(X, np.float64) - c) @ F).T; depth = float(p["depth"])
    keep = (w >= -depth) & (w <= .5 * depth)
    if p.get("shape", "cylinder") == "box": return keep & (np.abs(u) <= float(p["width"]) / 2) & (np.abs(v) <= float(p["height"]) / 2)
    return keep & (u * u + v * v <= (float(p["diameter"]) / 2) ** 2)


def setup(p):
    cs, ns = _spot(p["src"], "source"); cd, nd = _spot(p["dst"], "destination")
    reach = max(float(p.get("diameter", 0)), float(p.get("width", 0)), float(p.get("height", 0)))
    if np.linalg.norm(cs - cd) < reach: raise ValueError("The source and destination spots are too close: pick spots farther apart than the shape.")
    a = np.radians(float(p.get("rotation", 0.0))); c, s = np.cos(a), np.sin(a)
    mirror = str(p.get("mirror", "no")).lower() in ("yes", "true", "1")
    M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([-1.0 if mirror else 1.0, 1.0, 1.0])
    Fs, Fd = frame(ns, cd - cs), frame(nd, cd - cs)
    return cs, Fs, cd, Fd, Fd @ M @ Fs.T, mirror


def apply(kind, data, p):
    cs, Fs, cd, Fd, R, mirrored = setup(p)
    if kind == "points":
        P = np.asarray(data["P"], np.float64); src = inside(P, cs, Fs, p); dst = inside(P, cd, Fd, p)
        if not src.any(): raise ValueError("There is nothing to copy at the source spot.")
        new = {"P": (P[src] - cs) @ R.T + cd}
        if "N" in data: new["N"] = np.asarray(data["N"], np.float64)[src] @ R.T
        out = {}
        for k, v in data.items():
            if k in PER_POINT and isinstance(v, np.ndarray) and len(v) == len(P):
                add = new[k] if k in new else v[src]
                out[k] = np.concatenate([v[~dst], np.asarray(add, v.dtype)])
            else: out[k] = v
        return out, dst, {"removed": int(dst.sum()), "added": int(src.sum())}
    V, F, N = data; V = np.asarray(V, np.float64); F = np.asarray(F)
    cen = V[F].mean(axis=1) if len(F) else np.empty((0, 3)); src = inside(cen, cs, Fs, p); dst = inside(cen, cd, Fd, p)
    if not src.any(): raise ValueError("There is nothing to copy at the source spot.")
    used = np.unique(F[src]); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    nv = (V[used] - cs) @ R.T + cd; nf = remap[F[src]]
    if mirrored: nf = nf[:, ::-1]                                   # a mirror turns the faces inside out: turn them back
    nn = None if N is None else np.asarray(N, np.float64)[used] @ R.T
    V2, F2, N2 = apply_mask("mesh", (V, F, N), dst)
    out = (np.concatenate([V2, nv]), np.concatenate([F2, nf + len(V2)]), None if N2 is None or nn is None else np.concatenate([N2, nn]))
    return out, dst, {"removed": int(dst.sum()), "added": int(src.sum())}


def job_apply(job, kind, data):
    return apply(kind, data, job["params"])


@register
class Clone(Tool):
    name = "clone"; label = "Clone"; version = 1; mem_gb = 3
    kinds = ("mesh", "points")
    modules = ("tools/clone.py", "tools/common.py", "tools/edits.py", "plyio.py")
    libs = ("numpy",)

    def execute(self, ctx):
        return run_tool_job(ctx, self)

    def summary(self, p):
        return "%s %g mm" % (p.get("shape", "cylinder"), p.get("width" if p.get("shape") == "box" else "diameter", 0))
