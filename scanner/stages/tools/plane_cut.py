# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut plane: remove everything below a plane (model space) from a finished Fused or Meshed result. Each use is its own op, so cuts stack."""
import numpy as np

from ..methods import Tool, register
from .common import run_tool_job
from .edits import apply_mask


def basis(n):
    """Two unit vectors across the plane with normal n, the same every time (the spots' outline is stored in them)."""
    n = np.asarray(n, np.float64); n = n / max(np.linalg.norm(n), 1e-12)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]; e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def outline(spots_xyz, n):
    """The spots as 2D points across the plane, as a JSON list (the cut stays within their outline)."""
    import json
    e1, e2 = basis(n); P = np.asarray(spots_xyz, np.float64)
    return json.dumps([[round(float(x @ e1), 3), round(float(x @ e2), 3)] for x in P])


def below(X, p):
    """bool per row of X: below the plane (n . x < d + margin; the part side is the keep side) and, when the op has an outline,
    inside the spots' outline grown by pad mm (seen straight down onto the plane): the cut is not an infinite plane."""
    import json
    n = np.array([p["nx"], p["ny"], p["nz"]], np.float64); n /= max(np.linalg.norm(n), 1e-12)
    X = np.asarray(X, np.float64); sel = X @ n < float(p["d"]) + float(p.get("margin", 0.0))
    poly = p.get("poly")
    if poly is not None and not (isinstance(poly, str) and not poly.strip()):
        try:
            pts = np.asarray(json.loads(poly) if isinstance(poly, str) else poly, np.float64)
        except (ValueError, TypeError) as ex:
            raise ValueError("Cut base outline is invalid. Pick at least three finite spots.") from ex
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError("Cut base outline is invalid. Pick at least three finite spots.")
        from scipy.spatial import ConvexHull, QhullError
        try:
            eq = ConvexHull(pts).equations
        except QhullError as ex:
            raise ValueError("Cut base outline must enclose an area. Pick at least three noncollinear spots.") from ex
        e1, e2 = basis(n); idx = np.nonzero(sel)[0]
        uv = np.stack([X[idx] @ e1, X[idx] @ e2], 1)
        inside = ((uv @ eq[:, :2].T + eq[:, 2]) <= float(p.get("pad", 4.0))).all(1)
        sel[idx[~inside]] = False
    return sel


def apply(kind, data, params):
    if kind == "mesh":
        V, F, _ = data
        centers = V[F].mean(axis=1) if len(F) else np.empty((0, 3))
        sel = below(centers, params)
        out = apply_mask("mesh", data, sel)
    else:
        sel = below(data["P"], params)
        out = {k: (v[~sel] if k in ("P", "N", "W", "NF", "S") else v) for k, v in data.items()}
    n = max(1, len(sel))
    return out, sel, {"removed": int(sel.sum()), "removed_pct": round(100 * float(sel.sum()) / n, 2)}


def job_apply(job, kind, data):
    return apply(kind, data, job["params"])


@register
class CutPlane(Tool):
    name = "cut_plane"; label = "Cut base"; version = 1; mem_gb = 3
    kinds = ("mesh", "points")
    modules = ("tools/plane_cut.py", "tools/common.py", "tools/edits.py", "plyio.py")
    libs = ("numpy",)

    def execute(self, ctx):
        return run_tool_job(ctx, self)

    def summary(self, p):
        return "%.1f mm" % p["margin"] if p.get("margin") else "plane"
