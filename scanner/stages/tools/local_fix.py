# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smooth area: smooth only inside a cylinder or box lying on the surface at one spot, fading out over `feather` mm at its edge, so the
rest of the model is left as it is. A Taubin-style smooth (a shrinking step and a growing step) so the area does not shrink."""
import json

import numpy as np

from ..methods import Tool, register
from .common import run_tool_job
from .clone import _spot, frame

K = 12


def weights(X, c, F, p):
    """0..1 per row of X: 1 inside the shape, falling to 0 over `feather` mm at its edge; 0 outside the depth range."""
    u, v, w = ((np.asarray(X, np.float64) - c) @ F).T; depth = float(p["depth"]); feather = max(float(p.get("feather", 3.0)), 1e-6)
    if p.get("shape", "cylinder") == "box": d = np.minimum(float(p["width"]) / 2 - np.abs(u), float(p["height"]) / 2 - np.abs(v))
    else: d = float(p["diameter"]) / 2 - np.sqrt(u * u + v * v)
    return np.where((w >= -depth) & (w <= .5 * depth) & (d >= 0), np.clip(d / feather, 0.0, 1.0), 0.0)


def _steps(strength, passes):
    lam = .5 * float(strength)
    return [lam, -1.06 * lam] * int(passes)                      # a shrinking step, then a growing one (a bit stronger): no net shrink


def smooth_points(P, W, p, cell=0.4):
    """Each point in the area moves toward the average of the points within `reach` mm (not just the nearest few), so a bump a millimeter
    or two wide is flattened as well as fine noise."""
    from scipy.spatial import cKDTree
    idx = np.nonzero(W > 0)[0]; n = len(P); reach = max(float(p.get("reach", 2.5)), 0.2)
    k = int(min(256, max(8, 1.2 * np.pi * (reach / max(cell, 0.05)) ** 2)))
    d, nb = cKDTree(P).query(P[idx], k=min(k, n), distance_upper_bound=reach)
    ok = nb < n; nb = np.where(ok, nb, 0); cnt = np.maximum(ok.sum(axis=1), 1)[:, None]; Q = np.asarray(P, np.float64).copy()
    for s in _steps(p.get("strength", .8), p.get("passes", 8)):
        mean = (Q[nb] * ok[:, :, None]).sum(axis=1) / cnt
        Q[idx] += s * W[idx, None] * (mean - Q[idx])
    return Q


def smooth_mesh(V, F, W, p):
    """Laplacian passes over the mesh edges; a larger reach runs more of them (each spreads about one edge length)."""
    from scipy.sparse import coo_matrix
    n = len(V); e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0); Q = np.asarray(V, np.float64).copy()
    edge = float(np.mean(np.linalg.norm(Q[e[:, 0]] - Q[e[:, 1]], axis=1))) or 1.0
    more = max(1, int(round(float(p.get("reach", 2.5)) / edge)))
    for s in _steps(p.get("strength", .8), int(p.get("passes", 8)) * more):
        Q += s * W[:, None] * ((A @ Q) / deg[:, None] - Q)
    return Q


def apply(kind, data, p):
    c, n = _spot(p["spot"], "area"); F = frame(n)
    if kind == "points" and p.get("action") == "fill": raise ValueError("Fill gap works on the finished mesh. Build the mesh first, then fill the gap there.")
    if kind == "points":
        P = np.asarray(data["P"], np.float64); W = weights(P, c, F, p)
        if not (W > 0).any(): raise ValueError("There is nothing to smooth at that spot.")
        out = dict(data); out["P"] = smooth_points(P, W, p, float(data.get("cell", 0.4))).astype(np.asarray(data["P"]).dtype)
        return out, W > 0, {"moved": int((W > 0).sum())}
    if p.get("action") == "fill":
        if kind == "points": raise ValueError("Fill gap works on the finished mesh. Build the mesh first, then fill the gap there.")
        from .guess_fill import fill_area
        V2, F2, guessed = fill_area(data[0], data[1], c, F, p)
        area = float(np.linalg.norm(np.cross(V2[F2[guessed, 1]] - V2[F2[guessed, 0]], V2[F2[guessed, 2]] - V2[F2[guessed, 0]]), axis=1).sum() / 2)
        return (V2, F2, None), np.zeros(len(data[1]), bool), {"filled_mm2": area, "added_faces": int(guessed.sum()), "guessed": int(guessed.sum()), "_guessed": guessed}
    V, Fc, N = data; V = np.asarray(V, np.float64); W = weights(V, c, F, p)
    if not (W > 0).any(): raise ValueError("There is nothing to smooth at that spot.")
    return (smooth_mesh(V, np.asarray(Fc), W, p), Fc, N), W > 0, {"moved": int((W > 0).sum())}


def job_apply(job, kind, data):
    new, sel, st = apply(kind, data, job["params"])
    g = st.pop("_guessed", None)
    if g is not None and job.get("out_dir"):
        import os
        np.save(os.path.join(job["out_dir"], "guessed.npy"), np.packbits(g))             # the faces that are a guess, never measured data
    return new, sel, st


@register
class SmoothArea(Tool):
    name = "local_fix"; label = "Smooth area"; version = 1; mem_gb = 3
    kinds = ("mesh", "points")
    modules = ("tools/local_fix.py", "tools/guess_fill.py", "tools/clone.py", "tools/common.py", "plyio.py")
    libs = ("numpy", "scipy")

    def execute(self, ctx):
        return run_tool_job(ctx, self)

    def summary(self, p):
        return ("fill " if p.get("action") == "fill" else "") + "%s %g mm" % (p.get("shape", "cylinder"), p.get("width" if p.get("shape") == "box" else "diameter", 0))


def preview_job(job):
    """Use the same fill as Apply, write only the proposed faces, and change no recipe."""
    import os
    from ..plyio import read_mesh, write_mesh, vertex_normals
    params = dict(job["params"], action="fill")
    data = read_mesh(job["in"])
    new, _, stats = apply("mesh", data, params)
    mask = stats.pop("_guessed"); V, F, _ = new
    faces = F[mask]; path = os.path.join(job["out_dir"], "patch.ply")
    ids, inverse = np.unique(faces, return_inverse=True)
    vertices = V[ids]; faces = inverse.reshape(-1, 3)
    write_mesh(path, vertices, faces, vertex_normals(vertices, faces))
    return {"path": path, "stats": stats}
