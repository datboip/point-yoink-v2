#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Platter support check for side placements (numpy only).

Every side's part stood on its platter, and a rigid part lies wholly on one side of any plane it rested on. So for a fit
T (side j -> side i), side j's platter plane carried into side i's frame must have all of side i's part above it, and
the other way round. A fit that came out flipped on a near-symmetric part pushes part of a side into the other side's
table. Measured on a 14-side project (91 pair fits): 0 of 45 right fits and 21 of 25 flips above REJECT.
Planes are fuse.py's plane.json dicts: points kept where n . x >= d (+ margin)."""
import numpy as np

REJECT = 0.005          # share of a side's points more than TOL mm inside the other side's table
TOL = 3.0


def plane_in(plane, T):
    """plane (dict n, d, in frame A) seen in frame B, T = A -> B (4x4): n' = R n, d' = d + n' . t."""
    n = np.asarray(plane["n"], float); T = np.asarray(T, float); n2 = T[:3, :3] @ n
    return {"n": n2, "d": float(plane["d"]) + float(n2 @ T[:3, 3])}


def under(P, plane, tol=TOL):
    """Share of the points P (N x 3) more than tol mm under the plane."""
    if P is None or not len(P): return 0.0
    n = np.asarray(plane["n"], float)
    return float((np.asarray(P, float) @ n < float(plane["d"]) - tol).mean())


def pair_under(Pi, plane_i, Pj, plane_j, T_j_to_i, tol=TOL):
    """The worse of: side i's part under side j's table (carried into i), side j's part under side i's table."""
    T = np.asarray(T_j_to_i, float)
    return max(under(Pi, plane_in(plane_j, T), tol), under(Pj, plane_in(plane_i, np.linalg.inv(T)), tol))
