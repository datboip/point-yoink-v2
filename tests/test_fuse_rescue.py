"""Min. sides drops one-side points beside a surface several sides agree on, but keeps the ones that fill a hole."""
import numpy as np
from scanner.stages.tools.point_tools import merge


def _table():
    # keys are ids; columns: w, w*P(3), w*N(3), w*w, frames
    def row(x, nside):
        p = np.array([x, 0.0, 0.0]); n = np.array([0.0, 0.0, 1.0])
        return [1.0, *p, *n, 1.0, 5.0], nside
    pts = [(float(x), 2) for x in range(0, 10)] + [(5.0, 1)]           # a surface two sides saw; one side's point beside it
    pts += [(100.0, 1)]                                                # far from any surface: a hole filler
    K, C, B = [], [], []
    for i, (x, ns) in enumerate(pts):
        for j in range(ns):
            r, _ = row(x, ns); K.append(i); C.append(r); B.append(np.uint64(1) << np.uint64(j))
    return np.array(K), np.array(C, float), np.array(B, np.uint64)


def test_rescue_keeps_only_hole_fillers():
    K, C, B = _table()
    P, *_ = merge(K, C, None, B, 1, 2, 0.3, 0.0); assert 100.0 not in P[:, 0]
    P, *_ = merge(K, C, None, B, 1, 2, 0.3, 3.0)
    assert 100.0 in P[:, 0] and len(P) == 11
