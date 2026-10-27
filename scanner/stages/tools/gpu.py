# SPDX-License-Identifier: AGPL-3.0-or-later
"""The overlap-collapse regression (point_tools.project_lr) on the GPU with Open3D's CUDA tensors: the neighbor search and every
per-point contraction run on the card, only the points and the result cross the bus. Same formulas as the CPU version (three-term fit,
solved per point with Cramer's rule), in float32 around the cloud's center, so results match within a few micrometers, not bit for bit.
`available()` is False without a CUDA build of Open3D: callers fall back to the CPU path."""
import numpy as np

_state = {}


def available():
    if "ok" not in _state:
        try:
            import open3d as o3d
            _state["ok"] = bool(o3d.core.cuda.is_available())
        except Exception: _state["ok"] = False
    return _state["ok"]


def _cross(a, b):
    import open3d as o3d
    c = lambda t, i: t[:, i:i + 1]
    return o3d.core.concatenate([c(a, 1) * c(b, 2) - c(a, 2) * c(b, 1), c(a, 2) * c(b, 0) - c(a, 0) * c(b, 2), c(a, 0) * c(b, 1) - c(a, 1) * c(b, 0)], 1)


def project_lr(P, N, W, radius=1.5, sig_t=0.5, sig_n=1.0, gate=0.7, k=64, chunk=150000, wpow=1.0, ridge=0.02):
    """Linear fit only (quad=False). Returns P2, N2, nn, spread like point_tools.project_lr."""
    import open3d as o3d
    o3c = o3d.core; dev = o3c.Device("CUDA:0"); f32 = o3c.float32; f64 = o3c.float64
    c0 = P.mean(0); Pc = (P - c0).astype(np.float32)
    Pg = o3c.Tensor(Pc, f32, dev); Ng = o3c.Tensor(N.astype(np.float32), f32, dev); Wg = o3c.Tensor((W ** wpow).astype(np.float32), f32, dev)
    bias = o3c.Tensor(np.array([[0, 1e-6, 2e-6]], np.float32), f32, dev)
    nns = o3c.nns.NearestNeighborSearch(Pg); nns.hybrid_index(radius)
    P2 = P.copy(); N2 = N.copy(); nn = np.zeros(len(P), np.int32); spread = np.zeros(len(P))
    for s in range(0, len(P), chunk):
        e = min(len(P), s + chunk); Pi = Pg[s:e]; Ni = Ng[s:e]
        idx, _d2, _cnt = nns.hybrid_search(Pi, radius, min(k, len(P)))                       # (n, k) int64, -1 where there is no neighbor
        valid = idx.ge(0); idxc = (idx * valid.to(o3c.int64))
        D = Pg[idxc] - Pi.reshape((-1, 1, 3)); Nj = Ng[idxc]; Nr = Ni.reshape((-1, 1, 3))
        along0 = (D * Nr).sum(2)
        g = valid.logical_and((Nj * Nr).sum(2).gt(gate))
        t2 = ((D * D).sum(2) - along0 * along0).clip(0, 1e9)
        w = Wg[idxc] * (t2 * (-1.0 / (2 * sig_t ** 2))).exp() * (along0 * along0 * (-1.0 / (2 * sig_n ** 2))).exp() * g.to(f32)
        cnt = g.to(o3c.int32).sum(1)
        nbar = (w.reshape((*w.shape, 1)) * Nj).sum(1); nl = (nbar * nbar).sum(1).sqrt()
        ok = nl.gt(1e-9).to(f32).reshape((-1, 1))
        nbar = ok * nbar / nl.clip(1e-12, 1e9).reshape((-1, 1)) + (1 - ok) * Ni
        # tangent frame (tools.point_tools.tangent_frame): e = axis of the smallest |component|
        # tangent frame: e = the axis of the smallest |component| (ties favor x; the fit is rotation invariant in the tangent plane)
        a = nbar.abs() + bias; amin = a.min(1).reshape((-1, 1))
        e_ = a.eq(amin).to(f32)
        u = _cross(nbar, e_); u = u / (u * u).sum(1).sqrt().reshape((-1, 1)); v = _cross(nbar, u)
        nb = nbar.reshape((-1, 1, 3)); h = (D * nb).sum(2); x = (D * u.reshape((-1, 1, 3))).sum(2); y = (D * v.reshape((-1, 1, 3))).sum(2)
        hh = [t.to(f64) for t in (w, x, y, h)]; w64, x64, y64, h64 = hh
        s0 = w64.sum(1); sx = (w64 * x64).sum(1); sy = (w64 * y64).sum(1); sxx = (w64 * x64 * x64).sum(1); sxy = (w64 * x64 * y64).sum(1); syy = (w64 * y64 * y64).sum(1)
        r0 = (w64 * h64).sum(1); r1 = (w64 * x64 * h64).sum(1); r2 = (w64 * y64 * h64).sum(1)
        sw = s0 + 1e-12
        A00 = s0 + 1e-9; A11 = sxx + ridge * sw; A22 = syy + ridge * sw
        # symmetric 3x3 [[A00 sx sy],[sx A11 sxy],[sy sxy A22]] c = [r0 r1 r2]
        cof0 = A11 * A22 - sxy * sxy; cof1 = sy * sxy - sx * A22; cof2 = sx * sxy - A11 * sy
        det = A00 * cof0 + sx * cof1 + sy * cof2
        c_0 = (r0 * cof0 + r1 * (sy * sxy - sx * A22) + r2 * (sx * sxy - A11 * sy)) / det
        c_1 = (A00 * (r1 * A22 - sxy * r2) - r0 * (sx * A22 - sxy * sy) + sy * (sx * r2 - r1 * sy)) / det
        c_2 = (A00 * (A11 * r2 - sxy * r1) - sx * (sx * r2 - r1 * sy) + r0 * (sx * sxy - A11 * sy)) / det
        res = h64 - (c_0.reshape((-1, 1)) + c_1.reshape((-1, 1)) * x64 + c_2.reshape((-1, 1)) * y64)
        sp = ((w64 * res * res).sum(1) / sw).sqrt()
        nnew = nbar.to(f64) - c_1.reshape((-1, 1)) * u.to(f64) - c_2.reshape((-1, 1)) * v.to(f64)
        nnew = nnew / (nnew * nnew).sum(1).sqrt().reshape((-1, 1))
        a0 = c_0.reshape((-1, 1)) * nbar.to(f64)
        cn = cnt.cpu().numpy(); good = cn >= 4
        P2[s:e][good] = P[s:e][good] + a0.cpu().numpy()[good]; N2[s:e][good] = nnew.cpu().numpy()[good]
        nn[s:e] = cn; spread[s:e] = sp.cpu().numpy()
    return P2, N2, nn, spread
