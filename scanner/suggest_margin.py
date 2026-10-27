#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conservative, non-applied per-side platter margin suggestion.

The histogram alone cannot identify a plate top on every scan. In particular,
a flat part bottom can look like a second platter. This tool therefore returns
zero when the evidence is ambiguous and never changes the project recipe.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scanner.plate_top import turntable_plane


def suggest_margin_from_heights(heights, maximum=40.0):
    """Return a margin in mm plus evidence, without modifying the mesh.

    A near-zero, sharply ending layer supports a small margin above the current
    turntable plane. The 40 percent quantile is an upper guard, not evidence of
    where the plate top is. A distant competing layer or weak base makes the
    answer zero instead of an unverified destructive guess.
    """
    h = np.asarray(heights, dtype=float)
    h = h[np.isfinite(h)]
    if not len(h):
        return {"suggest_margin_mm": 0.0, "reason": "no finite mesh heights",
                "max_guarded_margin_mm": 0.0, "confidence": "none"}
    safe = float(max(0.0, min(float(maximum), np.floor((np.quantile(h, 0.4) - 0.05) * 10) / 10)))
    top = max(42.5, float(np.percentile(h, 99.9)) + 2.0)
    counts, edges = np.histogram(h, bins=np.arange(-2.0, top, 0.5))
    centers = (edges[:-1] + edges[1:]) / 2
    near = np.flatnonzero((centers >= -1.0) & (centers <= 1.0))
    peak_i = int(near[np.argmax(counts[near])])
    peak = int(counts[peak_i])
    base_share = float(np.mean(np.abs(h - centers[peak_i]) <= 0.75))
    far = counts[centers >= 3.0]
    far_peak = int(far.max()) if len(far) else 0
    result = {"suggest_margin_mm": 0.0,
              "max_guarded_margin_mm": round(safe, 1),
              "base_layer_share": round(base_share, 4),
              "competing_layer_ratio": round(far_peak / max(1, peak), 2),
              "confidence": "none"}
    if base_share < 0.035:
        result["reason"] = "no concentrated layer at the current turntable plane"
        return result
    if far_peak >= peak * 0.75:
        result["reason"] = "a competing high layer could be the plate or part"
        return result
    for i in range(peak_i + 1, len(counts) - 1):
        if edges[i] > min(safe, 12.0):
            break
        if counts[i] <= peak * 0.22 and counts[i + 1] <= peak * 0.22:
            candidate = min(safe, max(0.0, edges[i]))
            result["suggest_margin_mm"] = round(np.floor(candidate * 10) / 10, 1)
            result["confidence"] = "layer edge" if candidate > 0 else "none"
            result["reason"] = "first sparse band after the near-plane layer"
            return result
    result["reason"] = "no clear layer edge below the removal guard"
    return result


def suggest_for_side(scan, mesh_path, baseline_plane=None):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        return suggest_margin_from_heights([])
    if baseline_plane:
        with open(baseline_plane, encoding="utf-8") as stream:
            plane = json.load(stream)
        n = np.asarray(plane["n"], dtype=float)
        d = float(plane["d"])
    else:
        live = os.path.join(scan, "table_plane.json")
        o3d.utility.random.seed(0)
        n, d = turntable_plane(mesh, scan, live if os.path.exists(live) else None)
    vertices = np.asarray(mesh.vertices)
    heights = vertices @ n - d
    result = suggest_margin_from_heights(heights)
    removed = float(np.mean(heights <= result["suggest_margin_mm"]))
    result.update({"plane_n": n.tolist(), "plane_d": float(d),
                   "vertices": int(len(vertices)), "applied": False,
                   "estimated_removed_share": round(removed, 4),
                   "passes_removal_guard": bool(removed <= 0.40)})
    if removed > 0.40:
        result["reason"] = "current plane already removes over 40 percent at this margin"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", help="copied side folder with scanner poses")
    parser.add_argument("mesh", help="side's raw mesh")
    parser.add_argument("--baseline-plane", help="existing Auto plane JSON, if available")
    args = parser.parse_args()
    print(json.dumps(suggest_for_side(args.scan, args.mesh, args.baseline_plane), sort_keys=True))


if __name__ == "__main__":
    main()
