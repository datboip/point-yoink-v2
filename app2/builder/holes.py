# SPDX-License-Identifier: AGPL-3.0-or-later
"""The holes in a mesh, for the Fill holes preview: each boundary loop as a closed polyline and how big a hole it is by
the filler's own measure.

Open3D's fill_holes (VTK's vtkFillHolesFilter underneath) fills a loop when the radius of the sphere around it is at
most hole_size. We take half the loop's bounding-box diagonal, which is that radius to within a few percent on the
blob-shaped holes a scan leaves. Auto (hole_mm 0) is 4% of the part's bounding-box diagonal, as in process.py."""
import numpy as np

AUTO_SHARE = 0.04


def boundary_loops(V, F):
    """[(points (k,3) float32, radius_mm)] for every closed run of open edges (edges used by one triangle). A run
    that does not close (a non-manifold tangle) is still returned as the open polyline it is."""
    F = np.asarray(F).reshape(-1, 3).astype(np.int64)
    if not len(F): return []
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.minimum(E[:, 0], E[:, 1]) * (int(F.max()) + 1) + np.maximum(E[:, 0], E[:, 1])
    _u, first, count = np.unique(key, return_index=True, return_counts=True)
    open_e = E[first[count == 1]]
    if not len(open_e): return []
    nxt = {}                                                           # the open edges are directed by their triangle
    for a, b in open_e.tolist(): nxt.setdefault(a, []).append(b)
    V = np.asarray(V, np.float32); loops = []; used = set()
    for a0 in list(nxt):
        if a0 in used: continue
        run = [a0]; a = a0
        while True:
            used.add(a); outs = [b for b in nxt.get(a, ()) if b not in used]
            if not outs: break
            a = outs[0]; run.append(a)
            if a == a0: break
        if len(run) < 2: continue
        if run[-1] != run[0] and run[0] in nxt.get(run[-1], ()): run.append(run[0])
        P = V[np.asarray(run)]
        radius = float(np.linalg.norm(P.max(0) - P.min(0))) / 2.0
        loops.append((P, radius))
    return loops


def auto_mm(V):
    V = np.asarray(V, np.float32)
    return float(np.linalg.norm(V.max(0) - V.min(0))) * AUTO_SHARE if len(V) else 0.0


def split(loops, hole_mm):
    """(filled, kept): the loops the filler closes at this size, and the ones it leaves open."""
    filled = [l for l in loops if l[1] <= hole_mm]; kept = [l for l in loops if l[1] > hole_mm]
    return filled, kept


def segments(loops):
    """(n,2,3) float32 line segments for a list of loops (each polyline's consecutive pairs)."""
    parts = [np.stack([P[:-1], P[1:]], 1) for P, _ in loops if len(P) > 1]
    return np.concatenate(parts) if parts else np.zeros((0, 2, 3), np.float32)
