# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does each side line up with the others? A side whose surface mostly sits on its neighbors agrees; one that lies elsewhere (a bad
alignment, or a plate nobody else saw) does not. Measured on the cleaned side meshes placed by the line-up, so it needs no scan frames.

agreement(sides, root) -> {side n: {"alone": share with no other side within `near` mm, "agree": share within `close` mm of one, "points": n}}
flag_bad(result, base_n) -> [side n] that clearly do not agree: alone > 5 %, or agree < 85 % with alone > 1 %; never the base side, at
most a quarter of the sides. A side with almost nothing alone sits on the others everywhere (within 3 mm): a low agree there means it is
noisier or scanned at another gain, not misplaced, so it stays in. Measured on the 12-side project: side 11 had 72 % agree and 10.4 %
alone; every other side 91 to 100 % and 0 to 0.1 %. On the 10-side scan of 2026-10-06 YOLO left out sides at 68 to 83 % agree with
0 % alone: they were fine, and the model lost their coverage."""
import os

import numpy as np

AGREE_MIN = 0.85
ALONE_MAX = 0.05
ALONE_SOME = 0.01                                   # under this, a side is on the others everywhere: a low agree is noise, not a bad placement


def agreement(sides, root, stride=8, near=3.0, close=1.0):
    from stages import plyio
    pts = {}
    for s in sides:
        V, _, _ = plyio.read_mesh(os.path.join(root, s["mesh"])); T = np.asarray(s["T"], np.float64)
        pts[s["n"]] = (np.asarray(V, np.float64)[::stride] @ T[:3, :3].T + T[:3, 3])
    return agreement_of_points(pts, near, close)


def agreement_of_points(pts, near=3.0, close=1.0):
    from scipy.spatial import cKDTree
    trees = {n: cKDTree(Q) for n, Q in pts.items() if len(Q)}; out = {}
    for n, Q in pts.items():
        if not len(Q): out[n] = {"alone": 0.0, "agree": 1.0, "points": 0}; continue
        seen = np.zeros(len(Q), bool); on = np.zeros(len(Q), bool)
        for m, t in trees.items():
            if m == n: continue
            d, _ = t.query(Q, distance_upper_bound=near)
            seen |= np.isfinite(d); on |= d < close
        out[n] = {"alone": float(np.mean(~seen)), "agree": float(np.mean(on)), "points": int(len(Q))}
    return out


def flag_bad(result, base_n=None):
    if len(result) < 4: return []                                  # too few sides to say which one is the odd one
    bad = [(r["agree"], n) for n, r in result.items() if n != base_n and (r["alone"] > ALONE_MAX or (r["agree"] < AGREE_MIN and r["alone"] > ALONE_SOME))]
    bad.sort()
    return [n for _a, n in bad[:max(1, len(result) // 4)]]
