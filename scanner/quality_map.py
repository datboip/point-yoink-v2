#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a built model is not good yet, as a list of spots to touch up: holes (open edges, as needs_scan.py finds them)
and rough patches (striated, scarred or doubled surface). Smooth surface is left alone.

Rough: for every vertex, a small quadric (about 1.2 mm around it, same-facing neighbours only, so the far face of a thin
wall never counts) is fitted to the surface; the residual is how far the surface wanders from a smooth one. A clean
scan leaves a few hundredths of a mm; striations and scars leave a tenth or more. A second tell is a doubled layer
(another same-facing surface 0.25 mm or more straight above or below the point). A vertex counts as rough when a big
enough share of its surroundings (about 2.5 mm around) is bad: single speckles and the thin lines along sharp edges and
ribs never reach that share. Rough vertices are grouped into spots over the mesh; spots smaller than --min-area-mm2
are dropped, spots whose centres are close together are merged.

Library use (the Scan step's Touch up):  qm = quality_map(V, T)  ->  dict(kind per vertex 0 fine / 1 hole / 2 rough,
spots=[dict(kind, centre, normal, area_mm2, radius_mm, idx)], shares...)
usage: quality_map.py model.ply [--json out.json] [--ply out.ply] [--png out.png]"""
import argparse, json, os, sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

FINE, HOLE, ROUGH = 0, 1, 2
KIND_NAME = {HOLE: "hole", ROUGH: "rough"}


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, T[:, k], fn)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def vertex_areas(V, T):
    a = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    A = np.zeros(len(V))
    for k in range(3): np.add.at(A, T[:, k], a / 3.0)
    return A


def open_edges(V, T, min_mm=8.0):
    """Vertices on open-edge runs of at least min_mm (needs_scan.py's rule; shorter runs are pinholes for clean-up)."""
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    E, cnt = np.unique(E, axis=0, return_counts=True)
    B = E[cnt == 1]
    if not len(B): return np.zeros(0, np.int64)
    ln = np.linalg.norm(V[B[:, 0]] - V[B[:, 1]], axis=1)
    n, lab = connected_components(coo_matrix((np.ones(len(B)), (B[:, 0], B[:, 1])), shape=(len(V), len(V))), directed=False)
    el = lab[B[:, 0]]; length = np.bincount(el, weights=ln, minlength=n)
    return np.unique(B[np.isin(el, np.nonzero(length >= min_mm)[0])])


def roughness(V, N, tree, k=48, chunk=30000, workers=4):
    """Per vertex: the same-facing quadric residual (mm) and how many same-facing points lie 0.25 mm or more straight
    above or below it (a doubled layer)."""
    res = np.zeros(len(V), np.float32); dbl = np.zeros(len(V), np.int16)
    for s in range(0, len(V), chunk):
        e = min(len(V), s + chunk)
        idx = tree.query(V[s:e], k, workers=workers)[1]
        X = V[idx] - V[s:e, None]; cs = np.einsum('ckj,cj->ck', N[idx], N[s:e])
        along = np.einsum('ckj,cj->ck', X, N[s:e]); tang = np.sqrt(np.maximum((X * X).sum(2) - along ** 2, 0))
        dbl[s:e] = ((cs > 0.7) & (np.abs(along) > 0.25) & (tang < 0.4)).sum(1)
        w = (cs > 0.6).astype(np.float64); w[:, 0] = 1.0; sw = w.sum(1)      # [0] is the vertex itself (a stray one has no normal)
        mu = (w[..., None] * X).sum(1, keepdims=True) / sw[:, None, None]
        C = np.einsum('cki,ckj->cij', (X - mu) * w[..., None], X - mu) / sw[:, None, None]
        E = np.linalg.eigh(C)[1]
        L = np.einsum('cki,cij->ckj', X, E); z, x, y = L[..., 0], L[..., 1], L[..., 2]
        A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], -1)
        AtA = np.einsum('cki,ck,ckj->cij', A, w, A) + 1e-9 * np.eye(6)
        coef = np.linalg.solve(AtA, np.einsum('cki,ck,ck->ci', A, w, z)[..., None])[..., 0]
        r = (z - np.einsum('cki,ci->ck', A, coef)) ** 2
        res[s:e] = np.sqrt((r * w).sum(1) / sw)
    return res, dbl


def folded(V, T, rounds=10, deg=90.0):
    """Vertices of folded triangles: a triangle that faces more than deg away from the surface's smoothed normal there
    (the vertex normals averaged over about `rounds` rings of neighbours over the mesh, never across a thin wall). Pits
    and scars are full of them; a sharp edge or a rib's side is not (its faces sit at 45-90 deg to the smoothed normal)."""
    from scipy.sparse import diags
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); fu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    n = len(V)
    VF = coo_matrix((np.ones(T.size), (T.reshape(-1), np.repeat(np.arange(len(T)), 3))), shape=(n, len(T))).tocsr()
    S = VF @ fn; S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0; A = A + diags(np.ones(n)); A = diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(rounds): S = A @ S; S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    Sf = S[T].sum(1); Sf /= np.linalg.norm(Sf, axis=1, keepdims=True) + 1e-12
    bad_f = (fu * Sf).sum(1) < np.cos(np.radians(deg))
    return np.asarray(VF @ bad_f.astype(np.float64)).ravel() > 0


def neighbour_share(flag, tree, V, k, chunk=40000, workers=4):
    """For every vertex, the share of its k nearest vertices that are flagged."""
    out = np.zeros(len(V), np.float32)
    f = flag.astype(np.float32)
    for s in range(0, len(V), chunk):
        idx = tree.query(V[s:s + chunk], k, workers=workers)[1]; out[s:s + chunk] = f[idx].mean(1)
    return out


def edge_components(T, mask):
    """Connected groups of masked vertices over the mesh's edges: label per vertex (-1 = not masked)."""
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = E[mask[E[:, 0]] & mask[E[:, 1]]]
    nV = len(mask)
    n, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nV, nV)), directed=False)
    lab = lab.astype(np.int64); lab[~mask] = -1
    return lab


def quality_map(V, T, N=None, min_mm=8.0, hole_mm=3.0, bad_k=8.0, bad_floor=0.08, dbl_min=2, share_k=160,
                share_min=0.45, min_area_mm2=30.0, merge_mm=10.0, max_spot_mm2=300.0, workers=4):
    """V (n,3) mm, T (m,3). Returns dict(kind: uint8 per vertex, spots: [...], rough_share, hole_share, bad_mm)."""
    V = np.asarray(V, np.float64); T = np.asarray(T, np.int64)
    if N is None: N = vertex_normals(V, T)
    tree = cKDTree(V); area = vertex_areas(V, T); total = float(area.sum()) or 1.0
    kind = np.zeros(len(V), np.uint8)
    # holes: every vertex within hole_mm of an open-edge run that matters
    bv = open_edges(V, T, min_mm)
    if len(bv):
        d = cKDTree(V[bv]).query(V, distance_upper_bound=hole_mm, workers=workers)[0]
        kind[np.isfinite(d)] = HOLE
    # rough: bad vertices (residual well above this model's own smooth level, or a doubled layer), then the share of
    # bad around each vertex
    res, dbl = roughness(V, N, tree, workers=workers)
    bad_mm = max(bad_floor, bad_k * float(np.median(res)))
    fold = folded(V, T)
    bad = fold | (res > bad_mm) | (dbl >= dbl_min)
    share = neighbour_share(bad, tree, V, share_k, workers=workers)
    rough = (share >= share_min) & (kind == FINE)
    # grow back by the neighbourhood the share was taken over, so a spot covers its whole patch, not just its middle
    if rough.any():
        d = cKDTree(V[rough]).query(V, distance_upper_bound=1.5, workers=workers)[0]
        rough = np.isfinite(d) & (kind == FINE) & (share >= share_min * 0.5)
    kind[rough] = ROUGH
    # spots: connected groups over the mesh, small ones dropped, close ones merged
    spots = []
    for kd in (HOLE, ROUGH):
        lab = edge_components(T, kind == kd)
        ids = np.unique(lab[lab >= 0])
        groups = []
        for g in ids:
            idx = np.nonzero(lab == g)[0]; a = float(area[idx].sum())
            groups.append([idx, a])
        # merge groups whose centres are within merge_mm (one place to the person scanning)
        cen = [np.average(V[i], axis=0, weights=area[i] + 1e-9) for i, _ in groups]
        parent = list(range(len(groups)))
        def root(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        if len(groups) > 1:
            pairs = cKDTree(np.array(cen)).query_pairs(merge_mm)
            for i, j in pairs: parent[root(i)] = root(j)
        merged = {}
        for i, (idx, a) in enumerate(groups): merged.setdefault(root(i), []).append(idx)
        pieces = []
        for parts in merged.values():
            idx = np.concatenate(parts); a = float(area[idx].sum())
            if kd == ROUGH and a < min_area_mm2:
                kind[idx] = FINE; continue
            if kd == HOLE and len(idx) < 3: continue
            # a long scarred strip is several places to aim at: pieces of about max_spot_mm2 each
            k = int(np.ceil(a / max_spot_mm2))
            if k > 1 and len(idx) > 20 * k:
                from scipy.cluster.vq import kmeans2
                lab_k = kmeans2(V[idx], k, seed=0, minit="++")[1]
                pieces += [idx[lab_k == j] for j in range(k) if (lab_k == j).sum() >= 3]
            else: pieces.append(idx)
        for idx in pieces:
            a = float(area[idx].sum())
            w = area[idx] + 1e-9; c = np.average(V[idx], axis=0, weights=w)
            nrm = (N[idx] * w[:, None]).sum(0); nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
            spots.append(dict(kind=KIND_NAME[kd], centre=c, normal=nrm, area_mm2=a,
                              radius_mm=float(np.percentile(np.linalg.norm(V[idx] - c, axis=1), 90)), idx=idx))
    spots.sort(key=lambda s: (s["kind"] != "hole", -s["area_mm2"]))
    return dict(kind=kind, spots=spots, bad_mm=bad_mm, residual=res,
                rough_share=float(area[kind == ROUGH].sum() / total), hole_share=float(area[kind == HOLE].sum() / total))


def spots_json(qm):
    """The spots without their vertex lists, for a report."""
    return [dict(kind=s["kind"], centre_mm=[round(float(x), 1) for x in s["centre"]],
                 normal=[round(float(x), 3) for x in s["normal"]], area_mm2=round(s["area_mm2"], 1),
                 radius_mm=round(s["radius_mm"], 1), vertices=int(len(s["idx"]))) for s in qm["spots"]]


# the colors of the touch-up map: fine dimmed gray, holes red, rough purple (0-1 floats)
COLORS = {FINE: (0.42, 0.44, 0.47), HOLE: (1.0, 0.27, 0.2), ROUGH: (0.72, 0.33, 0.95)}


def colors(kind):
    c = np.empty((len(kind), 3));
    for k, v in COLORS.items(): c[kind == k] = v
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp")
    ap.add_argument("--json", default=None); ap.add_argument("--ply", default=None); ap.add_argument("--png", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import open3d as o3d
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(a.inp); V = np.asarray(m.vertices); T = np.asarray(m.triangles)
    qm = quality_map(V, T, workers=a.workers)
    sp = spots_json(qm)
    out = dict(version=1, model=os.path.basename(a.inp), rough_share=round(qm["rough_share"], 4),
               hole_share=round(qm["hole_share"], 4), bad_mm=round(qm["bad_mm"], 3), secs=round(time.time() - t0, 1), spots=sp)
    if a.json:
        tmp = a.json + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.json)
    print("%d spots to touch up (%d holes, %d rough); rough %.1f%% of the surface, near holes %.1f%% (%.1f s)" % (
        len(sp), sum(s["kind"] == "hole" for s in sp), sum(s["kind"] == "rough" for s in sp),
        100 * qm["rough_share"], 100 * qm["hole_share"], out["secs"]))
    for k, s in enumerate(sp[:12]):
        print("  spot %d: %s, %.0f mm2 around %s" % (k + 1, s["kind"], s["area_mm2"], s["centre_mm"]))
    if a.ply or a.png:
        m.vertex_colors = o3d.utility.Vector3dVector(colors(qm["kind"])); m.compute_vertex_normals()
        if a.ply: o3d.io.write_triangle_mesh(a.ply, m)
        if a.png:
            from PIL import Image, ImageDraw
            sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
            c = (V.min(0) + V.max(0)) / 2; r = np.linalg.norm(V.max(0) - V.min(0)) / 2; tc = colors(qm["kind"])[T].mean(1)
            img = Image.new("RGB", (560 * 3, 440 * 2)); dr = ImageDraw.Draw(img)
            for k, e in enumerate(([0, -0.4, -1], [0, 0.4, 1], [1, -0.5, 0], [-1, 0.3, 0.2], [0, -1, -0.1], [0, 1, 0.1])):
                eye = c + np.asarray(e, float) / np.linalg.norm(e) * r * 2.3
                rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=45, center=c.tolist(), eye=eye.tolist(), up=[0, -1, 0], width_px=560, height_px=420)
                ans = sc.cast_rays(rays); hit = np.isfinite(ans["t_hit"].numpy()); nn = ans["primitive_normals"].numpy(); ids = ans["primitive_ids"].numpy()
                dd = rays.numpy()[..., 3:]; dd /= np.linalg.norm(dd, axis=-1, keepdims=True); sh = np.clip(0.2 + 0.8 * np.abs((nn * dd).sum(-1)), 0, 1)
                im = np.full((420, 560, 3), 24, np.uint8); im[hit] = (sh[hit][:, None] * tc[ids[hit]] * 255).astype(np.uint8)
                img.paste(Image.fromarray(im), ((k % 3) * 560, (k // 3) * 440 + 20))
            dr.text((6, 2), "red = hole, purple = rough, gray = fine      %d spots, rough %.1f%%" % (len(sp), 100 * qm["rough_share"]), fill=(255, 255, 255))
            img.save(a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
