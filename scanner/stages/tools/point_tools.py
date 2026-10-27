# SPDX-License-Identifier: AGPL-3.0-or-later
"""Point tools on fused points (the tabs of Revo's Fusion panel), ported from the bake's revo_points/process_points.py.
The port carries W, NF and the side bitmask S through dedupe (bitwise OR per cell; revo_points dropped the mask).

isolation  26-connected occupancy clusters at cluster_voxel; drop clusters under max(min_pts, rate % of the biggest)
overlap    s = distance/0.21; normal-gated weighted plane projection (project_lr) with sig_n = s x (1.0, 0.7, 0.5),
           then dedupe at s x 0.25 keeping one point per (cell, normal bin); opposite-facing points never merge
smooth     project_lr passes with sig_n = 0.03 x strength, then drop points with too few same-facing neighbours"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .. import params as PR
from ..methods import Tool, register


# ------------------------------------------------------------------------------------------------------ ports
from .merge_cells import merge  # noqa: F401  (its own module: Fused depends on it, not on the cleanup below)


def components(P, voxel=1.0, min_frac=0.01, min_pts=200):
    if not len(P): return np.zeros(0, bool), 0, 0
    ijk = np.floor((P - P.min(0)) / voxel).astype(np.int64)
    G = np.zeros(ijk.max(0) + 1, bool); G[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(G, structure=np.ones((3, 3, 3)))
    L = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]; cnt = np.bincount(L)
    keep_lab = cnt >= max(min_pts, min_frac * cnt[1:].max()); keep_lab[0] = False
    return keep_lab[L], n, int(keep_lab.sum())


def tangent_frame(n):
    e = np.zeros_like(n); e[np.arange(len(n)), np.argmin(np.abs(n), 1)] = 1.0
    u = np.cross(n, e); u /= np.linalg.norm(u, axis=1, keepdims=True); v = np.cross(n, u)
    return u, v


def thread_budget(want=None, per_thread_gb=0.6):
    """How many chunks may run at once: the CPU threads the job was given, but never more than the job's memory limit can hold (each chunk
    needs about `per_thread_gb`) and at most 8."""
    import os
    n = int(want or os.environ.get("POINTYOINK_THREADS") or os.environ.get("OMP_NUM_THREADS") or (os.cpu_count() or 1))
    try:
        cg = open("/proc/self/cgroup").read().strip().split(":")[-1]
        lim = open("/sys/fs/cgroup%s/memory.max" % cg).read().strip()
        if lim.isdigit() and int(lim) < (1 << 50): n = min(n, max(1, int(0.7 * int(lim) / 1e9 / per_thread_gb)))
    except (OSError, ValueError, IndexError): pass
    return max(1, min(n, 8))


def project_lr(P, N, W, radius=1.5, sig_t=0.5, sig_n=1.0, gate=0.7, k=64, chunk=120000, wpow=1.0, quad=False, ridge=0.02, threads=None, gpu=None):
    """One pass of normal-gated local regression (revo_points' "overlap collapse"), global mode. Chunks run in parallel (`threads`, by
    default thread_budget()): numpy releases the GIL in the heavy contractions, and each chunk writes only its own slice, so the result
    is the same as one chunk at a time."""
    import os
    if not quad and gpu is not False and os.environ.get("POINTYOINK_GPU") != "0":
        from . import gpu as G
        if G.available():
            try: return G.project_lr(P, N, W, radius, sig_t, sig_n, gate, k, wpow=wpow, ridge=ridge)
            except Exception as e: print("GPU overlap merge failed (%s: %s); using the CPU" % (type(e).__name__, str(e)[:120]), flush=True)
    threads = thread_budget(threads); tree = cKDTree(P); Wp = W ** wpow
    P2 = P.copy(); N2 = N.copy(); nn = np.zeros(len(P), np.int32); spread = np.zeros(len(P))
    qw = 2 if threads > 1 else 8                                       # the neighbor search has its own threads: fewer each when chunks run side by side

    def work(s):
        e = min(len(P), s + chunk); Pi = P[s:e]; Ni = N[s:e]
        d, idx = tree.query(Pi, k=min(k, len(P)), distance_upper_bound=radius, workers=qw)
        del d
        if idx.ndim == 1: idx = idx[:, None]
        valid = idx < len(P); idx = np.where(valid, idx, 0)
        D = (P[idx] - Pi[:, None, :]).astype(np.float32); Nj = N[idx].astype(np.float32)
        g = valid & (np.einsum("nkj,nj->nk", Nj, Ni.astype(np.float32)) > gate)
        along0 = np.einsum("nkj,nj->nk", D, Ni.astype(np.float32))
        t2 = np.maximum((D * D).sum(-1) - along0 * along0, 0)
        w = np.where(g, Wp[idx] * np.exp(-t2 / (2 * sig_t ** 2)) * np.exp(-along0 * along0 / (2 * sig_n ** 2)), 0).astype(np.float32)
        del valid, idx, along0, t2
        cnt = g.sum(1); del g
        nbar = np.einsum("nk,nkj->nj", w, Nj).astype(np.float64); nl = np.linalg.norm(nbar, axis=1)
        del Nj
        nbar = np.where(nl[:, None] > 1e-9, nbar / np.maximum(nl, 1e-12)[:, None], Ni)
        u, v = tangent_frame(nbar)
        h = np.einsum("nkj,nj->nk", D, nbar.astype(np.float32)); x = np.einsum("nkj,nj->nk", D, u.astype(np.float32)); y = np.einsum("nkj,nj->nk", D, v.astype(np.float32))
        del D
        cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if quad else [])
        B = np.stack(cols, -1)
        del cols, x, y
        sw = w.sum(1) + 1e-12
        # Both contractions use the same weighted basis. Release neighbor-sized
        # temporaries before the next chunk without changing arithmetic order.
        weighted_B = B * w[:, :, None]
        A = np.einsum("nki,nkj->nij", weighted_B, B).astype(np.float64)
        r = np.einsum("nki,nk->ni", weighted_B, h).astype(np.float64)
        del weighted_B
        reg = np.zeros(B.shape[-1]); reg[1:3] = ridge; reg[3:] = ridge * 0.25
        A += (reg[None, :] * sw[:, None])[:, :, None] * np.eye(B.shape[-1])[None]
        A[:, 0, 0] += 1e-9
        c = np.linalg.solve(A, r[..., None])[..., 0]
        good = cnt >= 4
        a0 = c[:, 0]; n_new = nbar - c[:, 1:2] * u - c[:, 2:3] * v; n_new /= np.linalg.norm(n_new, axis=1, keepdims=True)
        P2[s:e][good] = Pi[good] + a0[good, None] * nbar[good]
        N2[s:e][good] = n_new[good]
        res = h - (B @ c[:, :, None].astype(np.float32))[..., 0]
        nn[s:e] = cnt; spread[s:e] = np.sqrt((w * res * res).sum(1) / sw)
        del B, w, h, res

    starts = list(range(0, len(P), chunk))
    if threads > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(threads, len(starts))) as ex: list(ex.map(work, starts))     # chunks are independent and write their own slices
    else:
        for s in starts: work(s)
    return P2, N2, nn, spread


def dedupe_keys(P, N, cell):
    ijk = np.floor(P / cell).astype(np.int64) + (1 << 18)
    a = np.argmax(np.abs(N), 1); b = a * 2 + (N[np.arange(len(N)), a] < 0)
    return (((ijk[:, 0] << 19) | ijk[:, 1]) << 19 | ijk[:, 2]) << 3 | b


def project_passes(P, N, W, passes, diagnostics=False, gpu=None):
    """Run a cleanup schedule with one CUDA upload and one final download.

    A failed resident job restarts on CPU from the unchanged input arrays.
    """
    import os
    fallback = False
    if passes and gpu is not False and os.environ.get("POINTYOINK_GPU") != "0":
        from . import gpu as G
        if G.available():
            try:
                from .gpu_resident import project_many
                return project_many(P, N, W, passes, diagnostics)
            except Exception as exc:
                fallback = True
                print("Resident GPU cleanup failed (%s: %s); using the CPU" %
                      (type(exc).__name__, str(exc)[:120]), flush=True)
    nn = None; spread = np.zeros(len(P)); medians = []
    for params in passes:
        P, N, nn, spread = project_lr(P, N, W, gpu=False, **params)
        if diagnostics: medians.append(round(float(np.median(spread)), 4))
    return P, N, nn, spread, medians, {"gpu_resident": False, "gpu_fallback": fallback}


def dedupe(P, N, W, NF, S, cell):
    """Cell average at `cell` mm, one point per (cell, normal bin). Returns new arrays and, per input point, whether it
    was the group's representative (highest weight) - the others are 'merged away'."""
    K = dedupe_keys(P, N, cell)
    u, inv = np.unique(K, return_inverse=True)
    sw = np.bincount(inv, weights=W, minlength=len(u))
    P2 = np.stack([np.bincount(inv, weights=W * P[:, j], minlength=len(u)) for j in range(3)], 1) / sw[:, None]
    N2 = np.stack([np.bincount(inv, weights=W * N[:, j], minlength=len(u)) for j in range(3)], 1).astype(np.float64)   # empty input: bincount gives ints
    N2 /= np.maximum(np.linalg.norm(N2, axis=1), 1e-12)[:, None]
    NF2 = np.bincount(inv, weights=NF, minlength=len(u))
    order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.nonzero(np.diff(inv[order]))[0] + 1] if len(order) else np.zeros(0, int)
    S2 = np.bitwise_or.reduceat(S[order], starts) if len(order) else np.zeros(0, np.uint64)
    # representative: highest weight in its group
    o2 = np.lexsort((-W, inv)); first = np.r_[True, inv[o2][1:] != inv[o2][:-1]]
    rep = np.zeros(len(P), bool); rep[o2[first]] = True
    return P2, N2, sw, NF2, S2, rep


# ------------------------------------------------------------------------------------------------------ tools
def _arrays(d):
    return (d["P"].astype(np.float64), d["N"].astype(np.float64), d["W"].astype(np.float64), d["NF"].astype(np.float64),
            d["S"].astype(np.uint64))


def _pack(d, P, N, W, NF, S):
    out = dict(d); out.update(P=P, N=N, W=W, NF=NF, S=S); return out


def apply(name, d, p, detect=False):
    """-> (new data, removed mask over the input points, stats)."""
    P, N, W, NF, S = _arrays(d)
    if name == "isolation":
        keep, ncomp, nkeep = components(P, p["cluster_voxel"], p["rate"] / 100.0, p["min_pts"])
        new = _pack(d, P[keep], N[keep], W[keep], NF[keep], S[keep])
        return new, ~keep, {"clusters": int(ncomp), "kept_clusters": nkeep}
    if name == "overlap":
        sched, cell = PR.overlap_schedule(p); st = {"schedule": list(sched), "dedupe": cell}
        passes = [dict(radius=p["radius"], sig_t=p["sig_t"], sig_n=sn, gate=p["gate"], k=p["k"]) for sn in sched]
        P, N, nn, spread, medians, gpu_stats = project_passes(P, N, W, passes, diagnostics=True)
        st.update(gpu_stats); st["residual_med"] = medians
        P2, N2, W2, NF2, S2, rep = dedupe(P, N, W, NF, S, cell)
        return _pack(d, P2, N2, W2, NF2, S2), ~rep, st
    if name == "smooth":
        sig_n = 0.03 * float(p["strength"]); sig_t = float(p["sig_t"]); r = max(3 * sig_t, 0.8); nn = None; st = {"sig_n": round(sig_n, 4)}
        passes = [dict(radius=r, sig_t=sig_t, sig_n=sig_n, gate=0.7, k=32) for _ in range(int(p["times"]))]
        P, N, nn, spread, _, gpu_stats = project_passes(P, N, W, passes)
        st.update(gpu_stats)
        drop = np.zeros(len(P), bool)
        if nn is not None and p["min_support"] > 0: drop = nn < p["min_support"]
        keep = ~drop
        return _pack(d, P[keep], N[keep], W[keep], NF[keep], S[keep]), drop, st
    raise ValueError("no point tool %r" % name)


def backend_tag():
    """"backend=cuda" or "backend=cpu": the overlap and smooth tools' output differs a little between the two, so the choice is in their key."""
    import os
    from . import gpu as G
    return "backend=" + ("cuda" if os.environ.get("POINTYOINK_GPU") != "0" and G.available() else "cpu")


class _PointTool(Tool):
    kinds = ("points",)
    modules = ("tools/point_tools.py", "tools/gpu.py", "tools/gpu_resident.py", "tools/common.py", "plyio.py"); libs = ("numpy", "scipy")

    def code(self, params):
        files, libs = super().code(params)
        return files, libs + ([backend_tag()] if self.name in ("overlap", "smooth") else [])

    def execute(self, ctx):
        from .common import run_tool_job
        result = run_tool_job(ctx, self)
        stats = result.get("stats", {})
        expected_cuda = self.name in ("overlap", "smooth") and backend_tag() == "backend=cuda"
        actual_cuda = stats.get("gpu_resident")
        mismatch = self.name in ("overlap", "smooth") and actual_cuda is not None and bool(actual_cuda) != expected_cuda
        if stats.get("gpu_fallback") or mismatch:
            result["transient"] = True
            result.setdefault("warn", []).append("Cleanup backend changed during the job; this result will be rebuilt")
        return result


@register
class Isolation(_PointTool):
    name = "isolation"; label = "Isolation"; version = 1; mem_gb = 4; detect = True

    def summary(self, p): return "%s %%" % PR.fmt(p["rate"])


@register
class Overlap(_PointTool):
    name = "overlap"; label = "Overlap Detection"; version = 1; mem_gb = 8; detect = True
    tables = ("overlap",)                          # sig_n schedule and dedupe cell from params.overlap_schedule

    def summary(self, p): return "%s mm" % PR.fmt(p["distance"])


@register
class Smooth(_PointTool):
    name = "smooth"; label = "Smooth"; version = 1; mem_gb = 6

    def summary(self, p): return "%s x%d" % (PR.fmt(p["strength"]), p["times"])
