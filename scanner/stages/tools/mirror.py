# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mirror: a symmetric part scanned well on one side gets the other side from a mirrored copy. The plane is picked by hand (three or more
spots down the middle) or found (the model's own principal planes, the one whose mirror image matches best), then refined so the copy
lines up with what was measured. Measured points are all kept; a mirrored point is added only where the original has nothing within
`gap_mm` of it, so a rough or missing area is filled from the good side and the good side is not doubled. The additions are a guess
made from the scan itself, listed in `guessed.npy` (faces) or counted in the stats (points), never blended into measured data."""
import numpy as np
from scipy.spatial import cKDTree

from ..methods import Tool, register
from .common import run_tool_job

PER_POINT = ("P", "N", "W", "NF", "S")


def plane_of(p):
    n = np.array([p["nx"], p["ny"], p["nz"]], np.float64); nl = np.linalg.norm(n)
    if not np.isfinite(nl) or nl < 1e-9: raise ValueError("Pick the mirror plane first: three spots down the middle of the part.")
    return n / nl, float(p["d"])


def reflect(X, n, d):
    """Mirror images of the rows of X across the plane n . x = d."""
    X = np.asarray(X, np.float64)
    return X - 2.0 * (X @ n - d)[:, None] * n[None, :]


def fit_plane(spots):
    """The plane through three or more spots (least squares), as (n, d); None when they lie in a line."""
    P = np.asarray(spots, np.float64)
    if len(P) < 3: return None
    c = P.mean(0); _u, s, vt = np.linalg.svd(P - c)
    if s[1] < 1e-6 * max(s[0], 1e-12): return None
    n = vt[2]; return n, float(c @ n)


def agreement(X, n, d, tol=1.5, tree=None, limit=300000):
    """Share of the mirrored copy of X that lands within tol mm of X itself: 1.0 for a perfectly symmetric part about this plane."""
    X = np.asarray(X, np.float64)
    if not len(X): return 0.0
    tree = tree or cKDTree(X)
    S = X[:: max(1, len(X) // limit)]
    dist, _ = tree.query(reflect(S, n, d), k=1, distance_upper_bound=tol, workers=-1)
    return float(np.isfinite(dist).mean())


def auto_plane(X, tol=1.5):
    """The best of the model's three principal planes through its middle (the one the part mirrors into best), refined. -> (n, d, score)."""
    X = np.asarray(X, np.float64); S = X[:: max(1, len(X) // 200000)]
    c = S.mean(0); _u, _s, vt = np.linalg.svd(S - c, full_matrices=False)
    tree = cKDTree(S); best = None
    for n in vt:
        score = agreement(S, n, float(c @ n), tol, tree)
        if best is None or score > best[2]: best = (n, float(c @ n), score)
    n, d = refine_plane(S, best[0], best[1], tree=tree)
    return n, d, agreement(S, n, d, tol, tree)


def normals_of(X, k=12):
    """Unit normals from the local neighborhood (PCA), for data that has none. Signs are arbitrary: the fit only uses |residual|."""
    X = np.asarray(X, np.float64); d, idx = cKDTree(X).query(X, k=min(k, len(X)), workers=-1)
    Q = X[idx]; C = Q - Q.mean(1, keepdims=True); cov = np.einsum("nki,nkj->nij", C, C)
    return np.linalg.eigh(cov)[1][:, :, 0]


def refine_plane(X, n, d, iters=8, tol=3.0, tree=None, normals=None):
    """Move the plane so the mirrored copy lies on the original surface: for each mirrored point with a match within tol mm, the
    distance along the match's normal to the match is the residual; the plane's tilt (two ways) and offset that bring those residuals
    to zero are solved (least squares, linearized) a few times. Point-to-plane, so a sheet gives no false pull along itself."""
    X = np.asarray(X, np.float64); n = np.asarray(n, np.float64) / np.linalg.norm(n); d = float(d)
    S = X[:: max(1, len(X) // 150000)]; tree = tree or cKDTree(X)
    if normals is None: normals = normals_of(X[:: max(1, len(X) // 300000)])
    else: normals = np.asarray(normals, np.float64)
    nstep = max(1, len(X) // 300000) if len(normals) != len(X) else 1
    for _ in range(iters):
        R = reflect(S, n, d); dist, idx = tree.query(R, k=1, distance_upper_bound=tol, workers=-1)
        ok = np.isfinite(dist)
        if ok.sum() < 100: break
        A = S[ok]; B = X[idx[ok]]; NB = normals[idx[ok] // nstep] if len(normals) != len(X) else normals[idx[ok]]
        NB = NB / np.maximum(np.linalg.norm(NB, axis=1, keepdims=True), 1e-12)
        r = ((R[ok] - B) * NB).sum(1)                                  # signed distance of the mirror image from the surface it should lie on
        # a tilt of the plane by t1 about e1 and t2 about e2 (e1, e2 across the plane) and an offset dd move the image by
        # 2*dd*n  -  2*((A.e_i) n + (A.n - d) e_i) t_i
        e1, e2 = _basis(n); h = A @ n - d
        J = np.stack([2.0 * (NB @ n), -2.0 * ((A @ e1) * (NB @ n) + h * (NB @ e1)), -2.0 * ((A @ e2) * (NB @ n) + h * (NB @ e2))], 1)
        w = 1.0 / (1.0 + (np.abs(r) / 1.0) ** 2)
        JW = J * w[:, None]
        try: dd, t1, t2 = np.linalg.solve(JW.T @ J + 1e-9 * np.eye(3), -(JW.T @ r))
        except np.linalg.LinAlgError: break
        n2 = n + t1 * e1 + t2 * e2; n2 /= np.linalg.norm(n2); d2 = d + dd
        moved = abs(d2 - d) + 100.0 * np.linalg.norm(n2 - n)
        n, d = n2, d2
        if moved < 1e-4: break
    return n, d


def _basis(n):
    a = np.eye(3)[int(np.argmin(np.abs(n)))]; e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _plan(X, p, normals=None):
    """-> (n, d, mask of X whose mirror image falls where the original has nothing within gap_mm, tree)."""
    n, d = plane_of(p); X = np.asarray(X, np.float64); tree = cKDTree(X)
    if str(p.get("fit", "yes")).lower() in ("yes", "true", "1"): n, d = refine_plane(X, n, d, tree=tree, normals=normals)
    gap = max(float(p.get("gap_mm", 1.5)), 0.05)
    dist, _ = tree.query(reflect(X, n, d), k=1, distance_upper_bound=gap, workers=-1)
    return n, d, ~np.isfinite(dist), tree


def apply(kind, data, p):
    if kind == "points":
        P = np.asarray(data["P"], np.float64); n, d, add, tree = _plan(P, p, data.get("N"))
        if not add.any(): raise ValueError("The mirrored copy adds nothing: the other side is already there within the gap.")
        out = {}
        for k, v in data.items():
            if k in PER_POINT and isinstance(v, np.ndarray) and len(v) == len(P):
                extra = reflect(v[add], n, d) if k == "P" else (np.asarray(v, np.float64)[add] - 2.0 * (np.asarray(v, np.float64)[add] @ n)[:, None] * n[None, :]) if k == "N" else v[add]
                out[k] = np.concatenate([v, np.asarray(extra, v.dtype)])
            else: out[k] = v
        st = {"added": int(add.sum()), "guessed": int(add.sum()), "agree": round(agreement(P, n, d, tree=tree), 3), "plane": [round(float(x), 5) for x in n] + [round(d, 3)]}
        return out, np.zeros(len(P), bool), st
    V, F, N = data; V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    from .. import plyio
    n, d, add_v, tree = _plan(V, p, plyio.vertex_normals(V, F) if len(F) else None)
    cen = V[F].mean(1) if len(F) else np.empty((0, 3))
    gap = max(float(p.get("gap_mm", 1.5)), 0.05)
    dist, _ = tree.query(reflect(cen, n, d), k=1, distance_upper_bound=gap, workers=-1)
    add_f = ~np.isfinite(dist)                                   # mirrored faces that land where there is no surface
    if not add_f.any(): raise ValueError("The mirrored copy adds nothing: the other side is already there within the gap.")
    used = np.unique(F[add_f]); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    nv = reflect(V[used], n, d); nf = remap[F[add_f]][:, ::-1]      # a mirror turns faces inside out: turn them back
    nn = None if N is None else np.asarray(N, np.float64)[used] - 2.0 * (np.asarray(N, np.float64)[used] @ n)[:, None] * n[None, :]
    V2 = np.concatenate([V, nv]); F2 = np.concatenate([F, nf + len(V)])
    N2 = None if N is None or nn is None else np.concatenate([np.asarray(N, np.float64), nn])
    guessed = np.zeros(len(F2), bool); guessed[len(F):] = True
    st = {"added": int(add_f.sum()), "guessed": int(add_f.sum()), "agree": round(agreement(V, n, d, tree=tree), 3),
          "plane": [round(float(x), 5) for x in n] + [round(d, 3)], "_guessed": guessed}
    return (V2, F2, N2), np.zeros(len(F), bool), st


def job_apply(job, kind, data):
    new, sel, st = apply(kind, data, job["params"])
    g = st.pop("_guessed", None)
    if g is not None and job.get("out_dir"):
        import os
        np.save(os.path.join(job["out_dir"], "guessed.npy"), np.packbits(g))
    return new, sel, st


@register
class Mirror(Tool):
    name = "mirror"; label = "Mirror"; version = 1; mem_gb = 4
    kinds = ("mesh", "points")
    modules = ("tools/mirror.py", "tools/common.py", "plyio.py")
    libs = ("numpy", "scipy")

    def execute(self, ctx):
        return run_tool_job(ctx, self)

    def summary(self, p):
        return "gap %g mm%s" % (float(p.get("gap_mm", 1.5)), "" if str(p.get("fit", "yes")).lower() in ("yes", "true", "1") else ", plane as picked")
