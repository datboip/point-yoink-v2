#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refine the placement of several scanned sides together, so every overlapping pair agrees (not just each side with
the base). Start: each side's side->base transform (align.py output). Every pair is fitted with point-to-plane ICP
where they overlap; a pose graph finds the placement that agrees best with all pairs; side 1 in the list stays put.

Thin-walled parts: a point is only paired with a point on the other side whose surface faces the same way. Without
that, the inside skin of one side got paired with the outside skin of another 1-2 mm away and every placement came out
about half a wall thick off; fusing then needed a wide truncation band to glue the sides, and that band erased thin
walls seen from both faces (the headrest lip, 12-side build). The fit also closes in step by step (3 mm, 1.5, 0.8, 0.5)
so the last steps only look at really matching surface.

Speed (same math, same point pairs as asking the tree every iteration, several times faster; see Nearest):
  - a pair fit only looks at the moving points that can pair at all (within the step's distance + MARGIN of the other
    side); the rest are provably out of reach until the pose has moved them MARGIN, and only then checked again;
  - each moving point remembers its KNOWN nearest points of the other side; after a small pose change the triangle
    inequality proves which of them is still the nearest, so only the few undecided points ask the tree again;
  - each side's points are stored in space-filling-curve order, so tree queries and lookups stay in cache;
  - the pair fits and agreement checks run in parallel processes (--jobs), one query thread each;
  - the last lookup of a fit is reused for the pose graph instead of being asked again.
usage: refine_sides.py --mesh s1.ply --init t1.json [--mesh s2.ply --init t2.json ...] --out-prefix refined_
       [--measure]   (only print how well the given placements agree)
       [--jobs N]    (parallel processes, default 8)   [--seed S]  (seed the point sampling: repeatable runs)
Writes <prefix><i>.json ({"matrix": 4x4 side->base, mm}) and prints pair agreement before and after."""
import argparse, json, os, time
import multiprocessing as mp
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree

L = o3d.pipelines.registration
STEPS = (3.0, 1.5, 0.8, 0.5)            # mm: how far apart a pair of points may be, one fitting round each
FACING = 0.8                            # cos of the largest angle between the two surfaces' normals in a pair
MARGIN = 2.0                            # mm beyond the pair distance a point stays a candidate (left-out points are
                                        # checked again once the pose may have moved them this far)
KNOWN = 4                               # nearest points each moving point remembers between lookups
REASK = 1.0                             # mm beyond the pair distance a repeated tree question looks
EPS = 1e-9                              # mm: float safety in the distance bounds

SIDES = []                              # per side: Side (below); shared with the worker processes by fork
WORKERS = -1                            # threads per big tree query (1 inside the worker processes)


class Side:
    """A side's points P and unit outward normals N (also side by side in PN, so one lookup fetches both), its tree,
    P with one extra point "at infinity" (the tree's index for "no neighbour"), and its centre and radius (to bound how
    far a transform moves any of its points)."""

    def __init__(self, P, N):
        self.P, self.N = P, N; self.PN = np.hstack([P, N]); self.tree = cKDTree(P)
        self.Px = np.vstack([P, np.full((1, 3), 1e100)])
        self.cen = P.mean(0); self.rho = float(np.sqrt(((P - self.cen) ** 2).sum(1).max()))


def curve_order(P, cell=2.0):
    """Order of the points along a Morton (Z-order) curve of `cell` mm cells: neighbours in space end up close in
    memory. Only the storage order changes."""
    g = np.floor((P - P.min(0)) / cell).astype(np.int64); key = np.zeros(len(P), np.int64)
    for bit in range(16):
        for ax in range(3): key |= ((g[:, ax] >> bit) & 1) << (3 * bit + ax)
    return np.argsort(key, kind="stable")


def load(path, voxel, count=250000):
    """Points and outward normals of a side's mesh (the fused meshes' triangles face the camera, so their normals do)."""
    m = o3d.io.read_triangle_mesh(path); m.compute_triangle_normals()
    pc = m.sample_points_uniformly(count, use_triangle_normal=True).voxel_down_sample(voxel)
    P = np.asarray(pc.points); N = np.asarray(pc.normals)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    o = curve_order(P)
    return np.ascontiguousarray(P[o]), np.ascontiguousarray(N[o])


def moved(P, T):
    """Points P moved by 4x4 T (same arithmetic as P @ R.T + t)."""
    Q = P @ T[:3, :3].T; Q += T[:3, 3]; return Q


def pairs(t, Q, Nq, r):
    """Point-to-plane pairs of moved points Q (normals Nq) with points of side t within r mm whose surface faces the
    same way. Returns (index into Q, index into t, signed distance along t's normal)."""
    d, k = t.tree.query(Q, distance_upper_bound=r, workers=WORKERS)
    i = np.flatnonzero(d < np.inf); k = k[i]
    return facing_pairs(t, Q, np.take(Nq, i, axis=0), i, k)[:3]


def facing_pairs(t, Q, Nq, i, k):
    """Of pairs (i into Q, k into t; Nq: the moved normals of Q[i]), keep those whose surfaces face the same way.
    Returns (i, k, signed distance along t's normal, t's normals there, Q[i])."""
    X = np.take(t.PN, k, axis=0); same = np.flatnonzero(np.einsum("ij,ij->i", Nq, X[:, 3:]) > FACING)
    i, k, X = i[same], k[same], np.take(X, same, axis=0); n = X[:, 3:]; q = np.take(Q, i, axis=0)
    return i, k, np.einsum("ij,ij->i", q - X[:, :3], n), n, q


class Nearest:
    """Exact nearest neighbours in side t of side m's points, for a pose that changes a little between calls: the same
    answers as asking the tree every time, with far fewer questions. All by the triangle inequality:
    - Points left out: at the last check (pose Tf) nothing of t was within lb mm of them, and they have moved at most
      shift(T) since, so nothing is within r of them while lb - shift(T) > r. Only then are they checked again.
    - Each candidate keeps, from where it last asked the tree (Qf), its KNOWN nearest points of t (kK) and a lower
      bound (dout) on the distance to every other point of t; and, from where it was last evaluated (Qe), its nearest
      point ke at d1 and a lower bound d2 on the distance to every other point. Having moved dl since Qe:
        nothing is within r              while d1 - dl >= r (it is not even looked at);
        ke is still the nearest          while d1 + 2 dl < d2 (and it is within r if d1 + dl < r);
        else the KNOWN points are measured again (the nearest of them is the nearest while it is closer than
        dout - its distance from Qf), and only if that does not settle it is the tree asked again."""

    FIELDS = ("sel", "Ps", "Ns", "Qf", "kK", "dout", "Qe", "ke", "d1", "d2")

    def __init__(self, t, m):
        self.t, self.m = t, m; self.sel = None; self.lb = -np.inf; self.Tf = None

    def shift(self, T, exact=False):
        """Upper bound on how far any point of m moved between Tf and T (mm); exact=True measures it."""
        A = T[:3, :3] - self.Tf[:3, :3]; dt = T[:3, 3] - self.Tf[:3, 3]
        if exact:
            D = self.m.P @ A.T; D += dt; return float(np.sqrt(np.einsum("ij,ij->i", D, D).max()))
        return float(np.linalg.norm(A @ self.m.cen + dt) + np.linalg.norm(A, 2) * self.m.rho)

    def _ask(self, Q, thr):
        """Ask the tree about points Q (bound thr): KNOWN nearest, the bound on all others, the nearest (index, distance;
        thr and len(t.P) where none is within thr) and the bound on all but the nearest."""
        d, k = self.t.tree.query(Q, k=KNOWN + 1, distance_upper_bound=thr, workers=WORKERS if len(Q) > 20000 else 1)
        d = np.minimum(d, thr)
        return k[:, :KNOWN], d[:, KNOWN], k[:, 0], d[:, 0], d[:, 1]

    def _new(self, idx, Q, thr):
        """Candidates among points idx of m (moved to Q): those with a point of t within thr, all fields filled."""
        kK, dout, k1, d1, d2 = self._ask(Q, thr); keep = np.flatnonzero(d1 < thr)
        idx = idx[keep]; Q = np.take(Q, keep, axis=0)
        return dict(sel=idx, Ps=np.take(self.m.P, idx, axis=0), Ns=np.take(self.m.N, idx, axis=0), Qf=Q, Qe=Q.copy(),
                    kK=np.take(kK, keep, axis=0), dout=dout[keep], ke=k1[keep], d1=d1[keep], d2=d2[keep])

    def check(self, T, r):
        """Look again at the points left out (all points the first time) at pose T for pair distance r: those now
        within r + MARGIN of t become candidates, the rest are at least that far."""
        thr = r + MARGIN
        if self.sel is None:
            f = self._new(np.arange(len(self.m.P)), moved(self.m.P, T), thr)
        else:
            out = np.ones(len(self.m.P), bool); out[self.sel] = False; out = np.flatnonzero(out)
            f = self._new(out, moved(np.take(self.m.P, out, axis=0), T), thr)
            f = {k: np.concatenate([getattr(self, k), f[k]]) for k in self.FIELDS}
        for k in self.FIELDS: setattr(self, k, f[k])
        self.lb = thr; self.Tf = T.copy()

    def _fresh(self, T, r):
        """Check the left-out points again if they may have come within r."""
        if self.sel is None or (self.lb - self.shift(T) <= r and self.lb - self.shift(T, True) <= r): self.check(T, r)

    def narrow(self, T, r):
        """A new, tighter pair distance r at pose T: drop the candidates now provably further than r + MARGIN."""
        thr = r + MARGIN
        if self.sel is None or not (self.lb - self.shift(T) >= thr or self.lb - self.shift(T, True) >= thr): self.check(T, r)
        D = moved(self.Ps, T); D -= self.Qe
        keep = np.flatnonzero(self.d1 - np.sqrt(np.einsum("ij,ij->i", D, D)) - EPS < thr)
        for k in self.FIELDS: setattr(self, k, np.take(getattr(self, k), keep, axis=0))
        self.lb = thr; self.Tf = T.copy()

    def _measure(self, j, Q, r):
        """Evaluate candidates j at their moved points Q[j]: nearest by the KNOWN points where that settles it, else by
        the tree."""
        Qj = np.take(Q, j, axis=0); kK = np.take(self.kK, j, axis=0)
        C = np.take(self.t.Px, kK, axis=0); C -= Qj[:, None, :]
        dd = np.einsum("ijk,ijk->ij", C, C); a = dd.argmin(1)[:, None]
        n1 = np.sqrt(np.take_along_axis(dd, a, 1)[:, 0]); np.put_along_axis(dd, a, np.inf, 1)
        n2 = np.sqrt(dd.min(1))
        D = Qj - np.take(self.Qf, j, axis=0); rest = self.dout[j] - np.sqrt(np.einsum("ij,ij->i", D, D))
        ok = n1 + EPS < rest                                                # the nearest known point is the nearest
        lo = np.minimum(n1, rest)
        self.Qe[j] = Qj; self.ke[j] = np.take_along_axis(kK, a, 1)[:, 0]
        self.d1[j] = np.where(ok, n1, lo); self.d2[j] = np.where(ok, np.minimum(n2, rest), lo)
        q = np.flatnonzero(~ok & (lo - EPS < r))                            # unsettled and maybe within r: the tree
        if len(q):
            jq = j[q]; kK, dout, k1, d1, d2 = self._ask(np.take(Qj, q, axis=0), r + REASK)
            self.Qf[jq], self.kK[jq], self.dout[jq] = Qj[q], kK, dout
            self.ke[jq], self.d1[jq], self.d2[jq] = k1, d1, d2

    def lookup(self, T, r):
        """For pose T: the moved candidates Q, and (index into Q, index into t) of every candidate whose nearest point
        of t is within r (that nearest point)."""
        self._fresh(T, r)
        Q = moved(self.Ps, T)
        D = Q - self.Qe; dl = np.sqrt(np.einsum("ij,ij->i", D, D))
        live = self.d1 - dl - EPS < r                                       # the rest: nothing within r
        redo = live & ~(self.d1 + 2 * dl + EPS < self.d2)                   # nearest may have changed
        j = np.flatnonzero(redo)
        if len(j): self._measure(j, Q, r); dl[j] = 0.0
        d1 = self.d1; sure = live & (d1 + dl < r - EPS)                     # certainly within r
        amb = np.flatnonzero(live & ~sure & (d1 - dl - EPS < r))            # maybe: measure
        if len(amb):
            E = np.take(Q, amb, axis=0) - np.take(self.t.Px, self.ke[amb], axis=0)
            sure[amb[np.einsum("ij,ij->i", E, E) < r * r]] = True
        i = np.flatnonzero(sure)
        return Q, i, self.ke[i]


def fit(t, m, T):
    """Move side m onto side t starting from 4x4 T (m -> t frame). Returns T, pairs used, residual rms (mm) and the
    indices of t's points in the last pairs."""
    T = T.copy(); nb = Nearest(t, m)
    for r in STEPS:
        nb.narrow(T, r)                                                     # a tighter step: fewer candidates
        for _ in range(12):
            Q, i, k = nb.lookup(T, r)
            i, k, e, n, q = facing_pairs(t, Q, np.take(nb.Ns, i, axis=0) @ T[:3, :3].T, i, k)
            if len(i) < 300: return T, 0, np.inf, None
            w = np.clip(1 - (e / r) ** 2, 0, None) ** 2                     # Tukey: far pairs count little
            J = np.empty((len(i), 6)); J[:, 3:] = n                          # [q x n, n]
            J[:, 0] = q[:, 1] * n[:, 2] - q[:, 2] * n[:, 1]; J[:, 1] = q[:, 2] * n[:, 0] - q[:, 0] * n[:, 2]
            J[:, 2] = q[:, 0] * n[:, 1] - q[:, 1] * n[:, 0]
            Jw = J * w[:, None]; A = Jw.T @ J; b = -(Jw.T @ e)
            x = np.linalg.solve(A + 1e-6 * np.eye(6), b)
            D = np.eye(4); D[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3]); D[:3, 3] = x[3:]
            T = D @ T
            if np.linalg.norm(x[:3]) < 1e-5 and np.linalg.norm(x[3:]) < 1e-3: break
    Q, i, k = nb.lookup(T, STEPS[-1])
    i, k, e, _, _ = facing_pairs(t, Q, np.take(nb.Ns, i, axis=0) @ T[:3, :3].T, i, k)
    return T, len(i), (float(np.sqrt(np.mean(e ** 2))) if len(i) else np.inf), k


def information(Pt):
    """Open3D's 6 x 6 information matrix of a pair from the target points it matched (rotation first)."""
    x, y, z = Pt[:, 0], Pt[:, 1], Pt[:, 2]; o, l = np.zeros_like(x), np.ones_like(x)
    G = np.stack([np.stack([o, z, -y, l, o, o], 1), np.stack([-z, o, x, o, l, o], 1), np.stack([y, -x, o, o, o, l], 1)], 1)
    return np.einsum("nki,nkj->ij", G, G)


def pair_fit(task):
    """Fit side j onto side i from relative pose R0: (i, j, R, used, rms, information matrix or None, cpu seconds)."""
    i, j, R0 = task; c = time.process_time()
    R, used, rms, k = fit(SIDES[i], SIDES[j], R0)
    info = information(np.take(SIDES[i].P, k, axis=0)) if used >= 500 else None
    return i, j, R, used, rms, info, ((i, j), time.process_time() - c)


def pair_agreement(task):
    """Median / 90th percentile distance (mm) between the matching surfaces of sides i and j (facing the same way,
    within 2 mm) and how many points matched; None under 500 points."""
    i, j, R = task; m = SIDES[j]
    e = pairs(SIDES[i], moved(m.P, R), m.N @ R[:3, :3].T, 2.0)[2]
    return i, j, ((float(np.median(np.abs(e))), float(np.percentile(np.abs(e), 90)), len(e)) if len(e) > 500 else None)


def load_one(task):
    return load(*task)


def _one_blas_thread():
    """The array maths here is small per call: BLAS threads only cost (and would multiply across the workers)."""
    try:
        from threadpoolctl import threadpool_limits; threadpool_limits(1, user_api="blas")
    except Exception:
        pass


def _worker_init():
    global WORKERS
    WORKERS = 1; _one_blas_thread()


def main():
    global SIDES
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="append", required=True); ap.add_argument("--init", action="append", required=True)
    ap.add_argument("--voxel", type=float, default=0.5, help="mm between the points used")
    ap.add_argument("--rounds", type=int, default=2, help="pair fits then pose graph, this many times")
    ap.add_argument("--out-prefix"); ap.add_argument("--measure", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="parallel processes (1 = none)")
    ap.add_argument("--times", action="store_true", help="print where the time went")
    ap.add_argument("--seed", type=int, default=0, help="seed Open3D's point sampling, for repeatable runs "
                    "(the sides are then loaded one after another, in list order)")
    a = ap.parse_args()
    t0 = time.time(); n = len(a.mesh); times = []; cpu = []
    def mark(what): times.append((what, time.time() - t0 - sum(t for _, t in times)))
    T0 = [np.array(json.load(open(p))["matrix"], float) for p in a.init]
    jobs = max(1, min(a.jobs, os.cpu_count() or 1))
    if jobs == 1: _one_blas_thread()
    ctx = mp.get_context("fork")
    if a.seed is not None or jobs == 1:
        if a.seed is not None: o3d.utility.random.seed(a.seed)
        loaded = [load(p, a.voxel) for p in a.mesh]
    else:
        with ctx.Pool(min(jobs, n), initializer=_worker_init) as p: loaded = p.map(load_one, [(m, a.voxel) for m in a.mesh], chunksize=1)
    SIDES = [Side(P, N) for P, N in loaded]; mark("load")
    pool = ctx.Pool(jobs, initializer=_worker_init) if jobs > 1 else None     # forked after loading: shares SIDES
    run = (lambda f, tasks: pool.map(f, tasks, chunksize=1)) if pool else (lambda f, tasks: list(map(f, tasks)))

    def agreement(T):
        """For each pair: median / 90th percentile distance (mm) between the surfaces that match (facing the same way,
        within 2 mm), and how many points matched."""
        tasks = [(i, j, np.linalg.inv(T[i]) @ T[j]) for i in range(n) for j in range(i + 1, n)]
        return {(i, j): v for i, j, v in run(pair_agreement, tasks) if v is not None}

    try:
        before = agreement(T0); mark("agreement before")
        if a.measure:
            for k, v in before.items(): print("  sides %d-%d: %.2f / %.2f mm (%d points)" % (k[0] + 1, k[1] + 1, *v))
            m = np.array([v[0] for v in before.values()]); print("median of pair medians %.3f mm" % np.median(m)); return 0
        T = [t.copy() for t in T0]
        for rnd in range(a.rounds):
            pg = L.PoseGraph()
            for i in range(n): pg.nodes.append(L.PoseGraphNode(T[i]))
            # biggest overlaps first keeps every worker busy to the end; the edges still go in in pair order
            tasks = sorted([(i, j, np.linalg.inv(T[i]) @ T[j]) for i in range(n) for j in range(i + 1, n) if (i, j) in before],
                           key=lambda t: -before[t[:2]][2])
            out = run(pair_fit, tasks); cpu += [c for *_, c in out]; mark("fits, round %d" % (rnd + 1))
            res = {(i, j): (R, used, rms, info) for i, j, R, used, rms, info, _ in out}
            for (i, j) in sorted(res):
                R, used, rms, info = res[(i, j)]
                if used < 500: continue
                pg.edges.append(L.PoseGraphEdge(j, i, R, info, uncertain=False))
                if rnd == a.rounds - 1: print("pair %d-%d: %d points match, fit error %.2f mm" % (i + 1, j + 1, used, rms))
            L.global_optimization(pg, L.GlobalOptimizationLevenbergMarquardt(), L.GlobalOptimizationConvergenceCriteria(),
                                  L.GlobalOptimizationOption(max_correspondence_distance=STEPS[-1], edge_prune_threshold=0.25,
                                                             reference_node=0))
            T1 = [np.asarray(nd.pose) for nd in pg.nodes]
            fix = T[0] @ np.linalg.inv(T1[0]); T = [fix @ t for t in T1]     # side 1 of the list is the base: it stays put
            mark("pose graph, round %d" % (rnd + 1))
        after = agreement(T); mark("agreement after")
    finally:
        if pool: pool.close(); pool.join()
    print("pair agreement (median / 90th pct mm)   before  ->  after")
    for k in before:
        b, f = before[k], after.get(k, (np.nan, np.nan, 0))
        print("  sides %d-%d:  %.2f / %.2f  ->  %.2f / %.2f" % (k[0] + 1, k[1] + 1, b[0], b[1], f[0], f[1]))
    mb = np.median([v[0] for v in before.values()]); ma = np.median([v[0] for v in after.values()]) if after else np.nan
    print("median of pair medians %.3f -> %.3f mm, %.0f s" % (mb, ma, time.time() - t0))
    if a.times:
        print("times: " + ", ".join("%s %.1f s" % w for w in times) + "; pair fits %.0f cpu-s (slowest: %s)" % (
            sum(c for _, c in cpu), ", ".join("%d-%d %.1f" % (p[0] + 1, p[1] + 1, c) for p, c in sorted(cpu, key=lambda x: -x[1])[:4])))
    for i, t in enumerate(T):
        json.dump({"matrix": t.tolist()}, open("%s%d.json" % (a.out_prefix, i + 1), "w"))
        D = np.linalg.inv(T0[i]) @ t
        print("side %d moved %.2f mm / %.2f deg" % (i + 1, np.linalg.norm(D[:3, 3]), np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
