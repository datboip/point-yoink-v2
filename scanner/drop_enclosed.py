#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drop surface no scanner could ever see, then tidy ragged rims. Two clean-ups from the reconstruction bake-off
(hybrid_patch), for the fused model after keep_part.py:
  1. enclosed surface: a vertex is enclosed when none of --rays rays spread over the hemisphere on EITHER side of it
     (along its normal and against it) gets out of the mesh. Loose TSDF flakes sealed inside a rolled bead and the walls
     of sealed voids go; outer skin, slots and the inside of a ring keep (some rays get out). A triangle goes when all
     three of its vertices are enclosed. Testing both sides keeps a wall whose normals point the wrong way.
  2. ragged-rim tidy: drop triangles with 2 or more open edges (hanging on by one edge), up to --tidy-rounds rounds.
  Pieces under --min-share of the biggest are dropped (keep_part.py's rule) before step 1 (so it also takes a raw fuse
  straight) and again after step 2, as step 1 can cut a flake loose.
Invents nothing; only removes. On an open mesh rays can escape through a nearby hole, so an enclosed test is a lower
bound: flakes next to a hole stay.
usage: drop_enclosed.py in.ply out.ply [--rays 32] [--max-escape 0] [--tidy-rounds 4] [--min-share 0.03]"""
import argparse, json, os, sys, time
import numpy as np


def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)


def hemisphere(k):
    """k unit directions spread uniformly in cos over the +z hemisphere (golden-angle spiral)."""
    i = np.arange(k) + 0.5; z = 1 - i / k
    r = np.sqrt(1 - z * z); phi = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def escape_share(scene, P, N, k=32, eps=0.03, chunk=100000):
    """Share of k hemisphere rays around N from each point (offset eps mm along N) that hit nothing."""
    import open3d as o3d
    H = hemisphere(k).astype(np.float32)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    a = np.where(np.abs(N[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(N, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(N, t1)
    out = np.zeros(len(P), np.float32)
    for s in range(0, len(P), chunk):
        e = slice(s, s + chunk)
        D = (H[None, :, 0:1] * t1[e, None, :] + H[None, :, 1:2] * t2[e, None, :] + H[None, :, 2:3] * N[e, None, :]).astype(np.float32)
        O = np.broadcast_to((P[e] + eps * N[e])[:, None, :], D.shape).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(np.concatenate([O, D], 2).reshape(-1, 6)))["t_hit"].numpy().reshape(-1, k)
        out[e] = np.isinf(th).mean(1)
    return out


def tri_open_edges(T):
    """Number of open (boundary) edges of every triangle."""
    nv = int(T.max()) + 1
    E = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1), 2); key = E[..., 0].astype(np.int64) * nv + E[..., 1]
    _u, inv, c = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    return (c[inv] == 1).reshape(-1, 3).sum(1)


def tri_area(V, T):
    return 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)


def drop_enclosed(mesh, rays=32, max_escape=0.0, tidy_rounds=4, min_share=0.03):
    """In place on an Open3D TriangleMesh (mm). Returns stats."""
    import open3d as o3d
    st = {"triangles_in": len(mesh.triangles)}

    def keep_pieces():
        tri, cnt, _area = mesh.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
        keep = np.nonzero(cnt >= min_share * cnt.max())[0]
        mesh.remove_triangles_by_mask(~np.isin(tri, keep)); mesh.remove_unreferenced_vertices()
        return int(len(cnt) - len(keep))
    if min_share > 0 and len(mesh.triangles):
        st["pieces_dropped_first"] = keep_pieces()          # keep_part.py's rule first: floaters must not block rays
    if rays > 0 and len(mesh.triangles):
        mesh.compute_vertex_normals()
        V = np.asarray(mesh.vertices); VN = np.asarray(mesh.vertex_normals); T = np.asarray(mesh.triangles)
        scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        enc = escape_share(scene, V, VN, rays) <= max_escape
        enc[enc] = escape_share(scene, V[enc], -VN[enc], rays) <= max_escape          # the other side only where needed
        bad = enc[T].all(1)
        st["enclosed_vertices"] = int(enc.sum()); st["enclosed_triangles"] = int(bad.sum())
        st["enclosed_cm2"] = round(float(tri_area(V, T[bad]).sum()) / 100, 3)
        mesh.remove_triangles_by_mask(bad); mesh.remove_unreferenced_vertices()
    tidy = 0; tidy_mm2 = 0.0
    for _ in range(max(0, tidy_rounds)):
        T = np.asarray(mesh.triangles)
        if not len(T): break
        bad = tri_open_edges(T) >= 2
        if not bad.any(): break
        tidy += int(bad.sum()); tidy_mm2 += float(tri_area(np.asarray(mesh.vertices), T[bad]).sum())
        mesh.remove_triangles_by_mask(bad); mesh.remove_unreferenced_vertices()
    st["tidy_triangles"] = tidy; st["tidy_mm2"] = round(tidy_mm2, 1)
    if min_share > 0 and len(mesh.triangles):
        st["pieces_dropped"] = keep_pieces()
    st["triangles_out"] = len(mesh.triangles)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--rays", type=int, default=32, help="hemisphere rays per side for the enclosed test (0 = skip it)")
    ap.add_argument("--max-escape", type=float, default=0.0, help="enclosed when at most this share of the rays get out on both sides")
    ap.add_argument("--tidy-rounds", type=int, default=4, help="rounds of dropping triangles with 2+ open edges (0 = off)")
    ap.add_argument("--min-share", type=float, default=0.03, help="then keep pieces at least this share of the biggest (0 = keep all)")
    a = ap.parse_args()
    import open3d as o3d
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(a.inp)
    if not len(m.triangles): emit("error", msg="no triangles in " + a.inp); return 2
    st = drop_enclosed(m, a.rays, a.max_escape, a.tidy_rounds, a.min_share)
    m.compute_vertex_normals()
    tmp = a.out + ".tmp.%d.ply" % os.getpid()
    if not o3d.io.write_triangle_mesh(tmp, m, write_ascii=False) or os.path.getsize(tmp) < 64:
        try: os.remove(tmp)
        except OSError: pass
        emit("error", msg="could not write " + a.out); return 3
    os.replace(tmp, a.out)
    st["secs"] = round(time.time() - t0, 1)
    emit("drop_enclosed", **st)
    print("dropped %s cm2 of enclosed surface (%d triangles) and %d ragged-rim triangles; %d triangles left"
          % (st.get("enclosed_cm2", 0), st.get("enclosed_triangles", 0), st["tidy_triangles"], st["triangles_out"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
