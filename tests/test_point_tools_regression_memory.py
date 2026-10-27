"""Exact regression against the original local-fit arithmetic after lifetime changes."""
import os
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scanner"))
from stages.tools.point_tools import project_lr, tangent_frame


def reference_project_lr(P, N, W, radius=1.5, sig_t=0.5, sig_n=1.0, gate=0.7, k=64, chunk=120000, wpow=1.0, quad=False, ridge=0.02):
    """One pass of normal-gated local regression (revo_points' "overlap collapse"), global mode."""
    tree = cKDTree(P); Wp = W ** wpow
    P2 = P.copy(); N2 = N.copy(); nn = np.zeros(len(P), np.int32); spread = np.zeros(len(P))
    for s in range(0, len(P), chunk):
        e = min(len(P), s + chunk); Pi = P[s:e]; Ni = N[s:e]
        d, idx = tree.query(Pi, k=min(k, len(P)), distance_upper_bound=radius, workers=8)
        if idx.ndim == 1: idx = idx[:, None]
        valid = idx < len(P); idx = np.where(valid, idx, 0)
        D = (P[idx] - Pi[:, None, :]).astype(np.float32); Nj = N[idx].astype(np.float32)
        g = valid & (np.einsum("nkj,nj->nk", Nj, Ni.astype(np.float32)) > gate)
        along0 = np.einsum("nkj,nj->nk", D, Ni.astype(np.float32))
        t2 = np.maximum((D * D).sum(-1) - along0 * along0, 0)
        w = np.where(g, Wp[idx] * np.exp(-t2 / (2 * sig_t ** 2)) * np.exp(-along0 * along0 / (2 * sig_n ** 2)), 0).astype(np.float32)
        nbar = np.einsum("nk,nkj->nj", w, Nj).astype(np.float64); nl = np.linalg.norm(nbar, axis=1)
        nbar = np.where(nl[:, None] > 1e-9, nbar / np.maximum(nl, 1e-12)[:, None], Ni)
        u, v = tangent_frame(nbar)
        h = np.einsum("nkj,nj->nk", D, nbar.astype(np.float32)); x = np.einsum("nkj,nj->nk", D, u.astype(np.float32)); y = np.einsum("nkj,nj->nk", D, v.astype(np.float32))
        cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if quad else [])
        B = np.stack(cols, -1)
        sw = w.sum(1) + 1e-12
        A = np.einsum("nki,nkj->nij", B * w[:, :, None], B).astype(np.float64)
        r = np.einsum("nki,nk->ni", B * w[:, :, None], h).astype(np.float64)
        reg = np.zeros(B.shape[-1]); reg[1:3] = ridge; reg[3:] = ridge * 0.25
        A += (reg[None, :] * sw[:, None])[:, :, None] * np.eye(B.shape[-1])[None]
        A[:, 0, 0] += 1e-9
        c = np.linalg.solve(A, r[..., None])[..., 0]
        cnt = g.sum(1); good = cnt >= 4
        a0 = c[:, 0]; n_new = nbar - c[:, 1:2] * u - c[:, 2:3] * v; n_new /= np.linalg.norm(n_new, axis=1, keepdims=True)
        P2[s:e][good] = Pi[good] + a0[good, None] * nbar[good]
        N2[s:e][good] = n_new[good]
        res = h - (B @ c[:, :, None].astype(np.float32))[..., 0]
        nn[s:e] = cnt; spread[s:e] = np.sqrt((w * res * res).sum(1) / sw)
    return P2, N2, nn, spread


class LocalRegressionEquivalence(unittest.TestCase):
    def compare(self, P, N, W, **kwargs):
        before = tuple(a.copy() for a in (P, N, W))
        expected = reference_project_lr(P, N, W, **kwargs)
        actual = project_lr(P, N, W, gpu=False, **kwargs)
        for old, new in zip(expected, actual):
            np.testing.assert_array_equal(new, old)
        for value, snapshot in zip((P, N, W), before):
            np.testing.assert_array_equal(value, snapshot)
        return actual

    def test_curved_opposing_surfaces_chunk_boundaries(self):
        rng = np.random.default_rng(731)
        xy = rng.uniform(-1.5, 1.5, (197, 2))
        z = .12 * xy[:, 0] ** 2 + .04 * xy[:, 1] + rng.normal(0, .015, len(xy))
        P = np.column_stack((xy, z))
        N = np.column_stack((-.24 * xy[:, 0], np.full(len(xy), -.04), np.ones(len(xy))))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        N[::13] *= -1
        W = rng.lognormal(0, 1, len(xy))
        for quad in (False, True):
            for chunk in (17, 64, 120000):
                with self.subTest(quad=quad, chunk=chunk):
                    result = self.compare(P, N, W, quad=quad, chunk=chunk, k=24, radius=.9, wpow=.7)
                    self.assertTrue(np.all(np.isfinite(result[0])))
                    self.assertTrue(np.all(np.isfinite(result[1])))
                    self.assertGreater(np.count_nonzero(result[2] >= 4), 0)

    def test_sparse_cloud_and_missing_neighbors(self):
        P = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.], [30., 0., 0.], [100., 0., 0.]])
        N = np.tile([0., 0., 1.], (len(P), 1))
        W = np.array([1., 2., 0., .5, 3.])
        result = self.compare(P, N, W, k=64, chunk=2, radius=1.5)
        np.testing.assert_array_equal(result[0], P)
        np.testing.assert_array_equal(result[1], N)

    def test_single_point_and_empty_cloud(self):
        for n in (0, 1):
            with self.subTest(points=n):
                self.compare(np.zeros((n, 3)), np.tile([0., 0., 1.], (n, 1)), np.ones(n))


if __name__ == "__main__":
    unittest.main()
