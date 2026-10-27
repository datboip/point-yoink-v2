# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guess fill: close the big holes of a mesh with a smooth patch that continues the surrounding surface (a quadric fitted to the ring
of surface around the hole), stitched to the hole's own boundary vertices. Geometry only, no model: flat panels stay flat, curved
ones keep their curvature. The new faces are listed (`guessed.npy`, packed bool per output face) and counted, never blended into
measured data. Small holes (under min_mm across) are Fill holes' job; holes wider than max_mm are left open (an opening, not a gap)."""
import numpy as np
from scipy.spatial import cKDTree, Delaunay

from ..methods import Tool, register


def boundary_loops(F):
    """Closed loops of vertex ids along edges used by exactly one face (open ends and branches are dropped)."""
    F = np.asarray(F, np.int64)
    if not len(F):return []
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); key = np.sort(e, 1)
    k = key[:, 0] * (key.max() + 1) + key[:, 1]
    u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    b = e[cnt[inv] == 1]
    if not len(b): return []
    nxt = {}; multi = set()
    for a, c in b:
        if a in nxt: multi.add(int(a))
        nxt[int(a)] = int(c)
    loops = []; seen = set()
    for s in list(nxt):
        if s in seen or s in multi: continue
        loop = [s]; seen.add(s); cur = nxt.get(s); ok = True
        while cur is not None and cur != s:
            if cur in seen or cur in multi: ok = False; break
            loop.append(cur); seen.add(cur); cur = nxt.get(cur)
        if ok and cur == s and len(loop) >= 3: loops.append(np.array(loop))
    return loops


def _inside(poly, q):
    """Even-odd test of points q (n, 2) against polygon poly (m, 2)."""
    x, y = q[:, 0], q[:, 1]; res = np.zeros(len(q), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        res ^= cross; j = i
    return res


def patch(V, F, loop, tree, spacing, ring_mm):
    """-> (new vertices (k,3), new faces (m,3) over [existing ids ... , len(V)+i]) for one hole, or None."""
    P = V[loop]; c = P.mean(0)
    ring = np.unique(np.concatenate(tree.query_ball_point(P[:: max(1, len(P) // 200)], ring_mm)))
    ring = ring[~np.isin(ring, loop)] if len(ring) > 3 * len(loop) else ring
    R = np.vstack([V[ring], P]) if len(ring) else P
    if len(R) < 12: return None
    ctr = R.mean(0); u_, s_, vt = np.linalg.svd(R - ctr, full_matrices=False)
    a1, a2, nrm = vt[0], vt[1], vt[2]
    q = (R - ctr) @ np.stack([a1, a2, nrm], 1)                                  # local (x, y, h)
    x, y, h = q[:, 0], q[:, 1], q[:, 2]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    w = np.ones(len(x)); coef = np.zeros(6)
    for _ in range(3):                                                           # a few reweighted passes: a ragged edge must not drag the fit
        coef = np.linalg.lstsq(A * w[:, None], h * w, rcond=None)[0]
        r = h - A @ coef; s = max(np.median(np.abs(r)) * 1.4826, 1e-4); w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    Bq = (P - ctr) @ np.stack([a1, a2, nrm], 1); poly = Bq[:, :2]
    lo, hi = poly.min(0), poly.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0] + spacing / 2, hi[0], spacing), np.arange(lo[1] + spacing / 2, hi[1], spacing))
    g = np.c_[gx.ravel(), gy.ravel()]; g = g[_inside(poly, g)]
    # keep interior points off the boundary so the triangles stay sane
    if len(g):
        d, _ = cKDTree(poly).query(g); g = g[d > 0.6 * spacing]
    pts2 = np.vstack([poly, g]) if len(g) else poly
    try: tri = Delaunay(pts2).simplices
    except Exception: return None
    cen = pts2[tri].mean(1); tri = tri[_inside(poly, cen)]
    if not len(tri): return None
    gh = np.c_[np.ones(len(g)), g[:, 0], g[:, 1], g[:, 0] ** 2, g[:, 0] * g[:, 1], g[:, 1] ** 2] @ coef if len(g) else np.zeros(0)
    newv = ctr + np.c_[g, gh] @ np.stack([a1, a2, nrm], 0) if len(g) else np.zeros((0, 3))
    ids = np.concatenate([loop, len(V) + np.arange(len(g))])
    faces = ids[tri]
    # orient like the surrounding surface: the ring's face normals decide
    allv = np.vstack([V, newv]); fn = np.cross(allv[faces[:, 1]] - allv[faces[:, 0]], allv[faces[:, 2]] - allv[faces[:, 0]])
    ref = nrm if len(ring) < 3 else None
    if len(ring):
        rn = np.zeros(3)
        near = np.unique(np.concatenate(tree.query_ball_point(P[:: max(1, len(P) // 50)], 2 * spacing)))
        fi = np.nonzero(np.isin(F, near).any(1))[0][:5000]
        if len(fi): rn = np.cross(V[F[fi, 1]] - V[F[fi, 0]], V[F[fi, 2]] - V[F[fi, 0]]).sum(0)
        ref = rn if np.linalg.norm(rn) > 0 else nrm
    if (fn.sum(0) @ ref) < 0: faces = faces[:, ::-1]
    return newv, faces


def stitches_loop(faces, loop):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    wanted = np.sort(np.column_stack([loop, np.roll(loop, -1)]), axis=1)
    return not (counts > 2).any() and set(map(tuple, unique[counts == 1])) == set(map(tuple, wanted))


def apply(data, p):
    V, F, N = data; V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    loops = boundary_loops(F)
    if not loops: return (V, F, N), np.zeros(len(F), bool), {"holes": 0, "filled": 0, "guessed_faces": 0, "filled_mm2": 0.0}
    el = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1); med = float(np.median(el))
    tree = cKDTree(V); newV = [V]; newF = [F]; nv = len(V); filled = 0; skipped_small = skipped_big = 0; sizes = []
    for loop in loops:
        P = V[loop]; diam = float(np.linalg.norm(P.max(0) - P.min(0)))
        if diam < p["min_mm"]: skipped_small += 1; continue
        if diam > p["max_mm"]: skipped_big += 1; continue
        r = patch(V, F, loop, tree, max(med * 1.5, 0.2), max(4.0, 0.6 * diam))
        if r is None: continue
        v, f = r
        if not stitches_loop(f, loop): continue
        f = np.where(f >= len(V), f - len(V) + nv, f)
        newV.append(v); newF.append(f); nv += len(v); filled += 1; sizes.append(round(diam, 1))
    V2 = np.vstack(newV); F2 = np.vstack(newF)
    guessed = np.zeros(len(F2), bool); guessed[len(F):] = True
    st = {"holes": len(loops), "filled": filled, "too_small": skipped_small, "too_big": skipped_big, "guessed_faces": int(guessed.sum()),
          "sizes_mm": sizes[:20], "filled_mm2": float(np.linalg.norm(np.cross(V2[F2[guessed, 1]] - V2[F2[guessed, 0]], V2[F2[guessed, 2]] - V2[F2[guessed, 0]]), axis=1).sum() / 2)}
    return (V2, F2, None), guessed, st


def fill_area(V, F, c, Fr, p):
    """Fill gap at one spot (the Smooth area tool's Fill action): close the gap inside the chosen cylinder or box with a surface that continues
    the surrounding one. Frame Fr (columns u, v, normal) at spot c. Returns (V2, F2, guessed face mask) or raises ValueError.
    Where the mesh has no surface (no vertex within about two edge lengths) a grid is lifted onto a quadric fitted to the ring of surface
    around the area, then triangulated together with the nearby existing vertices so the patch meets the edge of the gap."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    q = (V - c) @ Fr; u, v, w = q.T
    box = p.get("shape", "cylinder") == "box"; half = (float(p["width"]) / 2, float(p["height"]) / 2) if box else (float(p["diameter"]) / 2,) * 2
    ring_r = 1.6 * max(half)
    depth = max(3 * float(p.get("depth", 6.0)), 0.5 * max(half))
    ring = np.nonzero((u * u + v * v < ring_r ** 2) & (np.abs(w) < depth))[0]
    if len(ring) < 30: raise ValueError("There is not enough surface around that spot to continue it. Pick a spot beside the gap.")
    X = np.stack([np.ones(len(ring)), u[ring], v[ring], u[ring] ** 2, u[ring] * v[ring], v[ring] ** 2], 1); h = w[ring]
    wt = np.ones(len(ring)); coef = np.zeros(6)
    for _ in range(3):
        coef = np.linalg.lstsq(X * wt[:, None], h * wt, rcond=None)[0]
        r = h - X @ coef; s = max(np.median(np.abs(r)) * 1.4826, 1e-4); wt = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    el = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1); sp = max(1.5 * float(np.median(el)), 0.2)
    gx, gy = np.meshgrid(np.arange(-half[0] + sp / 2, half[0], sp), np.arange(-half[1] + sp / 2, half[1], sp))
    g = np.c_[gx.ravel(), gy.ravel()]
    if not box: g = g[g[:, 0] ** 2 + g[:, 1] ** 2 <= half[0] ** 2]
    gh = np.c_[np.ones(len(g)), g[:, 0], g[:, 1], g[:, 0] ** 2, g[:, 0] * g[:, 1], g[:, 1] ** 2] @ coef
    L = np.c_[g, gh] @ Fr.T + c                                                    # grid points lifted onto the fitted surface (model frame)
    tree = cKDTree(V); d, _ = tree.query(L); new = L[d > 1.1 * sp]
    if len(new) >= 3:                                              # speckle: drop islands of a few points (pinholes), keep real gaps
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        pr = cKDTree(new).query_pairs(1.8 * sp, output_type="ndarray")
        _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(new), len(new))), directed=False)
        new = new[np.bincount(lab)[lab] >= 12]
    if len(new) < 3: raise ValueError("There is no gap at that spot: the surface is already there. Aim the area at the empty part.")
    ng = ((new - c) @ Fr)[:, :2]
    near = np.unique(np.concatenate([np.asarray(x, np.int64) for x in tree.query_ball_point(new[:: max(1, len(new) // 400)], 2.0 * sp)]))
    near = near[np.abs(q[near, 2]) < depth]
    pts2 = np.vstack([q[near, :2], ng]); allv = np.vstack([V[near], new])
    try: tri = Delaunay(pts2).simplices
    except Exception: raise ValueError("The area is too small or too flat to fill.")
    isnew = np.arange(len(pts2)) >= len(near)
    keep = isnew[tri].any(1)
    e = np.stack([np.linalg.norm(allv[tri[:, a]] - allv[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    keep &= e.max(1) < 3.5 * sp
    tri = tri[keep]
    if not len(tri): raise ValueError("The gap is too wide for one patch. Use a smaller area or fill it in pieces.")
    ids = np.concatenate([near, len(V) + np.arange(len(new))]); faces = ids[tri]
    all3 = np.vstack([V, new]); fn = np.cross(all3[faces[:, 1]] - all3[faces[:, 0]], all3[faces[:, 2]] - all3[faces[:, 0]])
    flip = (fn @ Fr[:, 2]) < 0; faces[flip] = faces[flip][:, ::-1]
    V2 = all3; F2 = np.vstack([F, faces]); guessed = np.zeros(len(F2), bool); guessed[len(F):] = True
    return V2, F2, guessed
