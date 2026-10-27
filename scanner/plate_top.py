#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find a broad plate top above the turntable without mistaking a part bottom for it.

This helper only proposes a plane. strip_table.py performs the actual side cut,
so the side mesh and the later per-frame fuse use the same plane and margin.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuse import read_global_poses


def turntable_plane(mesh, scan, live=None, max_tilt=8.0):
    """Use strip_table's axis and densest 1 mm layer, or its saved live plane."""
    V = np.asarray(mesh.vertices)
    C = np.array([M[:3, 3] for M in read_global_poses(
        os.path.join(scan, "cache", "pointyoink_register_pose.pose")).values()])
    axis = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
    if (C.mean(0) - V.mean(0)) @ axis < 0:
        axis = -axis
    if live:
        with open(live) as f:
            pl = json.load(f)
        return np.asarray(pl["n"], float), float(pl["d"])
    P = np.asarray(mesh.sample_points_uniformly(200000).points)
    h = P @ axis
    hist, edges = np.histogram(h, bins=np.arange(h.min(), h.max() + 1.0, 1.0))
    layer = edges[hist.argmax()] + 0.5
    sel = P[np.abs(h - layer) < 2.0]
    cen = sel.mean(0)
    n = np.linalg.svd(sel - cen, full_matrices=False)[2][2]
    if n @ axis < 0:
        n = -n
    if np.degrees(np.arccos(np.clip(n @ axis, -1, 1))) > max_tilt:
        n = axis
    return n, float(n @ cen)


def choose_plate_top(vertices, vertex_normals, normal, base_d, margin, min_width=210.0, backoff=2.0):
    """Find a distinct elevated flat band and cut below its top.

    Return the output plane's d and a report. strip_table adds ``margin`` to
    that plane, so subtract it here: the actual cut stays ``backoff`` below
    the top of the flat band even when the user configured a larger margin.
    """
    v = np.asarray(vertices)
    vn = np.asarray(vertex_normals)
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    if len(v) == 0 or vn.shape != v.shape:
        return None, {"reason": "no raw vertices with normals", "candidates": []}
    h = v @ n - base_d
    flat = np.abs(vn @ n) >= np.cos(np.deg2rad(10.0))
    bins = np.floor(h).astype(int)
    flat_counts = np.bincount(bins[(bins >= 0) & (bins <= 150) & flat], minlength=151)[:151]
    reference = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)
    w = np.cross(n, u)

    def width_at(height):
        layer = v[(bins == height) & flat]
        if not len(layer):
            return np.zeros(2)
        xy = np.column_stack((layer @ u, layer @ w))
        return np.percentile(xy, 98, axis=0) - np.percentile(xy, 2, axis=0)

    base_bin = int(np.argmax(flat_counts[:3]))
    base_width = width_at(base_bin) if flat_counts[base_bin] >= 1000 else np.zeros(2)
    if min(base_width) < min_width:
        return None, {"reason": "turntable reference is not a broad flat layer", "candidates": [],
                      "base_width_mm": np.round(base_width, 1).tolist()}
    support_floor = max(1000, int(len(v) * 0.01))
    qualifying = []
    for height in range(3, 151):
        if flat_counts[height] < support_floor:
            continue
        width = width_at(height)
        ratio = width / base_width
        if min(width) < min_width or min(ratio) < 0.75 or max(ratio) > 1.4:
            continue
        qualifying.append((height, width))
    groups = []
    for height, width in qualifying:
        if not groups or height > groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append((height, width))
    candidates = []
    for group in groups:
        peak = max(group, key=lambda item: flat_counts[item[0]])
        peak_height, width = peak
        edge = group[-1][0] + 1
        if edge + 1 > 150 or flat_counts[edge] > 0.10 * flat_counts[peak_height] or flat_counts[edge + 1] > 0.10 * flat_counts[peak_height]:
            continue
        candidates.append({"height_mm": round(peak_height + 0.5, 1),
                           "top_edge_mm": float(edge),
                           "width_mm": np.round(width, 1).tolist(),
                           "flat_support": int(sum(flat_counts[item[0]] for item in group)),
                           "bins": [item[0] for item in group]})
    if not candidates:
        return None, {"reason": "no distinct elevated broad flat layer from 0 to 150 mm",
                      "candidates": [], "base_width_mm": np.round(base_width, 1).tolist()}
    candidates.sort(key=lambda item: item["flat_support"], reverse=True)
    if len(candidates) > 1 and candidates[0]["flat_support"] < 2 * candidates[1]["flat_support"]:
        return None, {"reason": "multiple competing broad flat layers", "candidates": candidates,
                      "base_width_mm": np.round(base_width, 1).tolist()}
    best = candidates[0]
    cut_height = best["top_edge_mm"] - backoff
    removed = float(np.mean(h <= cut_height))
    plane_height = cut_height - margin
    report = dict(best, candidates=candidates, base_width_mm=np.round(base_width, 1).tolist(),
                  cut_height_mm=round(cut_height, 3), plane_height_mm=round(plane_height, 3),
                  backoff_mm=backoff, margin_mm=margin,
                  removed_share=round(removed, 4),
                  below_plane_share=round(float(np.mean(h <= plane_height)), 4))
    if removed > 0.40:
        report["reason"] = "actual cut would remove over 40% of side vertices"
        return None, report
    return float(base_d + plane_height), report


def main():
    o3d.utility.random.seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("mesh")
    ap.add_argument("plane_out")
    ap.add_argument("report_out")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--max-tilt", type=float, default=8.0)
    ap.add_argument("--live-plane")
    a = ap.parse_args()
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    V = np.asarray(mesh.vertices)
    n, d = turntable_plane(mesh, a.scan, a.live_plane, a.max_tilt)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    top, report = choose_plate_top(V, np.asarray(mesh.vertex_normals), n, d, a.margin)
    report = dict(report, base_d=d, auto_removed_share=round(float(np.mean(V @ n <= d + a.margin)), 4))
    with open(a.report_out, "w") as f:
        json.dump(report, f, indent=2)
    if top is not None:
        with open(a.plane_out, "w") as f:
            json.dump({"n": n.tolist(), "d": top, "margin": a.margin, "keep_above": True}, f)
    elif os.path.exists(a.plane_out):
        os.unlink(a.plane_out)
    print("STAGE plate_top " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
