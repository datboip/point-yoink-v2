# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep point regression arrays on CUDA through all passes of one cleanup operation.

The regression formulas match gpu.project_lr. Its implementation remains available as the
reference path. Only the final packed arrays and pass diagnostics are downloaded.
"""
import time

import numpy as np

from .gpu import _cross


class ResidentCloud:
    def __init__(self, P, N, W, wpow=1.0):
        import open3d as o3d
        c = o3d.core; device = c.Device("CUDA:0")
        self.P = c.Tensor(P.astype(np.float64), c.float64, device)
        self.N = c.Tensor(N.astype(np.float64), c.float64, device)
        self.Wg = c.Tensor((W ** wpow).astype(np.float32), c.float32, device)
        self.rows = c.Tensor(np.arange(len(P), dtype=np.int64), c.int64, device)
        self.first_center = c.Tensor(P.mean(0), c.float64, device)
        self.index_points = None; self.index_positions = None
        self.nns = None; self.radius = None; self.index_builds = 0

    def project(self, radius=1.5, sig_t=0.5, sig_n=1.0, gate=0.7, k=64, chunk=150000, ridge=0.02):
        import open3d as o3d
        o3c = o3d.core; dev = self.P.device; f32 = o3c.float32; f64 = o3c.float64
        same = (self.index_positions is not None and self.radius == radius and
                self.index_positions.shape == self.P.shape and bool(self.P.eq(self.index_positions).all().item()))
        if same:
            Pg = self.index_points
        else:
            c0 = self.first_center if self.first_center is not None else self.P.mean(0)
            Pg = (self.P - c0).to(f32)
        self.first_center = None
        Ng = self.N.to(f32); Wg = self.Wg
        bias = o3c.Tensor(np.array([[0, 1e-6, 2e-6]], np.float32), f32, dev)
        if not same:
            self.index_points = Pg; self.index_positions = self.P; self.radius = radius
            self.nns = o3c.nns.NearestNeighborSearch(Pg); self.nns.hybrid_index(radius); self.index_builds += 1
        nns = self.nns
        P2 = self.P.clone(); N2 = self.N.clone()
        nn = o3c.Tensor.zeros((len(self.P),), o3c.int32, dev)
        spread = o3c.Tensor.zeros((len(self.P),), f64, dev)
        for s in range(0, len(self.P), chunk):
            e = min(len(self.P), s + chunk); Pi = Pg[s:e]; Ni = Ng[s:e]
            idx, _d2, _cnt = nns.hybrid_search(Pi, radius, min(k, len(self.P)))                       # (n, k) int64, -1 where there is no neighbor
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
            good = cnt.ge(4); rows = self.rows[s:e][good]
            P2[rows] = self.P[rows] + a0[good]; N2[rows] = nnew[good]
            nn[s:e] = cnt; spread[s:e] = sp
        self.P, self.N = P2, N2
        return nn, spread


def project_many(P, N, W, passes, diagnostics=False, wpow=1.0):
    import open3d as o3d
    if not passes:
        return P.copy(), N.copy(), None, np.zeros(len(P)), [], {}
    started = time.perf_counter(); cloud = ResidentCloud(P, N, W, wpow)
    o3d.core.cuda.synchronize(); upload = time.perf_counter() - started
    seconds = []; history = []; nn = None
    for params in passes:
        started = time.perf_counter(); nn, spread = cloud.project(**params)
        o3d.core.cuda.synchronize(); seconds.append(time.perf_counter() - started)
        if diagnostics: history.append(spread)
    if not diagnostics: history = [spread]
    started = time.perf_counter()
    packed = o3d.core.concatenate([cloud.P, cloud.N, nn.to(o3d.core.float64).reshape((-1, 1))] +
                                 [v.reshape((-1, 1)) for v in history], 1).cpu().numpy()
    download = time.perf_counter() - started
    medians = [round(float(np.median(packed[:, 7+i])), 4) for i in range(len(history))] if diagnostics else []
    stats = dict(gpu_resident=True, gpu_upload_seconds=upload, gpu_pass_seconds=seconds,
                 gpu_download_seconds=download, gpu_index_builds=cloud.index_builds)
    return packed[:, :3].copy(), packed[:, 3:6].copy(), packed[:, 6].astype(np.int32), packed[:, -1].copy(), medians, stats
