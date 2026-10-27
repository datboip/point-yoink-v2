# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep fused geometry supported by a placed, cleaned side mesh."""
import os

import numpy as np
from scipy.spatial import cKDTree

from .. import plyio
from ..keys import jload
from ..methods import Tool, register
from .common import run_tool_job
from .edits import apply_mask


def support_mask(query, sides, project, distance):
    """Find a cleaned side vertex within distance, in model coordinates."""
    query = np.asarray(query, np.float64)
    supported = np.zeros(len(query), bool)
    if not sides:
        raise ValueError("Match the cleaned sides needs at least one placed side")
    for side in sides:
        V, _, _ = plyio.read_mesh(os.path.join(project, side["mesh"]))
        if not len(V):
            continue
        T = np.asarray(side["T"], np.float64)
        placed = V @ T[:3, :3].T + T[:3, 3]
        tree = cKDTree(placed)
        for start in range(0, len(query), 100000):
            end = min(start + 100000, len(query))
            missing = np.flatnonzero(~supported[start:end]) + start
            if len(missing):
                d, _ = tree.query(query[missing], k=1, distance_upper_bound=distance, workers=1)
                supported[missing] = np.isfinite(d)
        del tree, placed, V
        if supported.all():
            break
    return supported


def apply(kind, data, sides, project, distance):
    if kind == "mesh":
        V, F, _ = data
        centers = V[F].mean(axis=1) if len(F) else np.empty((0, 3))
        removed = ~support_mask(centers, sides, project, distance)
        out = apply_mask("mesh", data, removed)
    else:
        removed = ~support_mask(data["P"], sides, project, distance)
        out = {k: (v[~removed] if k in ("P", "N", "W", "NF", "S") else v)
               for k, v in data.items()}
    return out, removed, {"removed": int(removed.sum()), "removed_pct": round(100 * float(removed.mean()), 2) if len(removed) else 0,
                          "distance_mm": distance, "sides": len(sides)}


def job_apply(job, kind, data):
    sides = jload(job["sides_json"])["sides"]
    return apply(kind, data, sides, job["project"], float(job["params"]["distance_mm"]))


@register
class MatchCleanedSides(Tool):
    name = "match_cleaned_sides"; label = "Match the cleaned sides"; version = 1; mem_gb = 6
    kinds = ("mesh", "points")
    modules = ("tools/match_sides.py", "tools/common.py", "tools/edits.py", "plyio.py")
    libs = ("numpy", "scipy")

    def execute(self, ctx):
        if "lineup" not in ctx.inputs:
            raise ValueError("Match the cleaned sides needs the line-up")
        return run_tool_job(ctx, self)

    def summary(self, p):
        return "%s mm" % p["distance_mm"]
