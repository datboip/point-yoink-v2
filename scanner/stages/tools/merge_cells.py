# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sum of the per-side cell tables into one point set (the Fused points method's last step). Its own module so that editing the
cleanup tools in point_tools.py does not change Fused's cache key and re-run a stage that would give the same bytes."""
import numpy as np
from scipy.spatial import cKDTree


def merge(K, C, side, side_bits, min_frames=3, min_sides=1, normal_agree=0.3, rescue_mm=0.0):
    """Sum the per-side cell tables (K keys, C columns: w, w*P(3), w*N(3), w*w, frames) over the sides; keep entries
    seen by at least min_frames frames and min_sides sides whose normals agree. side_bits: uint64 bit per row."""
    u, inv = np.unique(K, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=C[:, j], minlength=len(u)) for j in range(C.shape[1])], 1)
    ns = np.bincount(inv, minlength=len(u))
    order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.nonzero(np.diff(inv[order]))[0] + 1]
    mask = np.bitwise_or.reduceat(side_bits[order], starts) if len(order) else np.zeros(0, np.uint64)
    w = S[:, 0]; P = S[:, 1:4] / w[:, None]; N = S[:, 4:7]; nl = np.linalg.norm(N, axis=1)
    NF = S[:, 8]
    good = (NF >= min_frames) & (nl > normal_agree * w)
    ok = good & (ns >= min_sides)
    if rescue_mm > 0 and min_sides > 1 and ok.any():
        # a one-side point survives where no multi-side surface is near: a scan aimed at a hole is the only one that saw it, but a
        # rough second layer beside a surface the sides agree on is still dropped
        lone = good & ~ok
        if lone.any():
            d, _ = cKDTree(P[ok]).query(P[lone], k=1, distance_upper_bound=rescue_mm, workers=-1)
            ok[np.nonzero(lone)[0][~np.isfinite(d)]] = True
    N = N / np.maximum(nl, 1e-12)[:, None]
    return P[ok], N[ok], w[ok], NF[ok], mask[ok]
