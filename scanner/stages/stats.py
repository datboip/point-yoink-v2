# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mesh and point statistics (numpy + scipy; no Open3D): verts, faces, pieces, open edge mm, bbox, area."""
import numpy as np


def mesh_stats(V, F):
    out = {"verts": int(len(V)), "faces": int(len(F))}
    if not len(F):
        return out
    bb = V.max(0) - V.min(0); out["bbox_mm"] = [round(float(x), 1) for x in bb]
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out["area_cm2"] = round(float(a.sum()) / 100.0, 1)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    key = E[:, 0].astype(np.int64) * (len(V) + 1) + E[:, 1]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    open_e = cnt[inv] == 1
    B = E[open_e]
    out["open_edge_mm"] = round(float(np.linalg.norm(V[B[:, 0]] - V[B[:, 1]], axis=1).sum()), 1)
    out["nonmanifold_edges"] = int((cnt > 2).sum())
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    A = coo_matrix((np.ones(len(F) * 2), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    used = np.zeros(n, bool); used[F.ravel()] = True
    pieces = len(np.unique(lab[used]))
    out["pieces"] = int(pieces)
    fl = lab[F[:, 0]]; c = np.bincount(fl)
    out["biggest_share"] = round(float(c.max() / len(F)), 4)
    return out


def points_stats(d):
    P = d["P"]; out = {"points": int(len(P))}
    if not len(P): return out
    out["bbox_mm"] = [round(float(x), 1) for x in (P.max(0) - P.min(0))]
    S = d["S"].astype(np.uint64); ids = [str(x) for x in d["side_ids"]]
    cnt = np.zeros(len(S), np.int64)
    per = {}
    for k, sid in enumerate(ids):
        b = ((S >> np.uint64(k)) & np.uint64(1)).astype(bool); per[sid] = int(b.sum()); cnt += b
    out["per_side"] = per
    out["one_side_share"] = round(float((cnt == 1).mean()), 4)
    out["weak_share"] = round(float(((cnt <= 1) | (d["NF"] < 5)).mean()), 4)
    return out
