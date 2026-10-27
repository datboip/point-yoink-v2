# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preview PNGs and viewer exports (run inside jobs, never in the engine process).
Meshes: Open3D RaycastingScene like render_views.py (560 x 420 per view, fov 45, the same 6 view directions; a 1-view
preview uses the first). Points: numpy z-buffer splats (2 px) shaded by |normal . view|, subsampled above 3M.
Colour modes: shaded, sides (one stable colour per side id), removed (an op's removed elements in red over grey),
scanhere (the report's needs.ply colours), support (points: red where seen by one side or fewer than 5 frames).
Every PNG carries a header line with the node, method and counts."""
import hashlib, json, os

import numpy as np

from . import plyio
from .stats import mesh_stats, points_stats
from .keys import jload, jdump

DIRS = ([0, -0.4, -1], [0, 0.4, 1], [1, -0.5, 0], [-1, 0.3, 0.2], [0, -1, -0.1], [0, 1, 0.1])
W, HH = 560, 420
BLUE = np.array([90, 170, 255.0]) / 255.0
GREY = np.array([0.62, 0.62, 0.62]); RED = np.array([1.0, 0.18, 0.15])
MAX_PTS = 3_000_000


PALETTE = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [213, 94, 0], [204, 121, 167],
                    [255, 255, 255], [148, 103, 189], [140, 86, 75], [127, 127, 127], [188, 189, 34], [23, 190, 207], [255, 152, 150],
                    [197, 176, 213], [196, 156, 148], [247, 182, 210], [219, 219, 141], [158, 218, 229], [44, 160, 44]]) / 255.0
_ORDER = {}


def side_colour(sid, all_ids=None):
    """One colour per side: its place among the project's side ids (sorted, so a side keeps its colour while sides are
    added after it), from a 20-colour palette; a lone id falls back to a hash."""
    if all_ids is not None: _ORDER.update({s: k for k, s in enumerate(sorted(all_ids))})
    k = _ORDER.get(sid)
    if k is None: k = int(hashlib.blake2b(str(sid).encode(), digest_size=4).hexdigest(), 16)
    return PALETTE[k % len(PALETTE)]


def _camera(V):
    c = (V.min(0) + V.max(0)) / 2; r = float(np.linalg.norm(V.max(0) - V.min(0)) / 2) or 1.0
    return c, r


def render_mesh(V, F, fcol=None, views=1, header=""):
    import open3d as o3d
    from PIL import Image, ImageDraw
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(V.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(m)
    c, r = _camera(V)
    tiles = []
    for e in DIRS[:views]:
        eye = c + np.asarray(e, float) / np.linalg.norm(e) * r * 2.3
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=45, center=c.tolist(), eye=eye.tolist(), up=[0, -1, 0], width_px=W, height_px=HH)
        ans = sc.cast_rays(rays); hit = np.isfinite(ans["t_hit"].numpy()); n = ans["primitive_normals"].numpy(); ids = ans["primitive_ids"].numpy()
        d = rays.numpy()[..., 3:]; d /= np.linalg.norm(d, axis=-1, keepdims=True); sh = np.clip(0.15 + 0.85 * np.abs((n * d).sum(-1)), 0, 1)
        img = np.full((HH, W, 3), 24, np.uint8)
        col = BLUE[None, :] if fcol is None else fcol[ids[hit]]
        img[hit] = np.clip(sh[hit][:, None] * col * 255, 0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(img))
    return _sheet(tiles, header)


def render_points(P, N, col=None, views=1, header="", seed=0):
    from PIL import Image
    if len(P) > MAX_PTS:
        k = np.random.default_rng(seed).choice(len(P), MAX_PTS, replace=False); P = P[k]; N = N[k]; col = col[k] if col is not None else None
    c, r = _camera(P); f = (HH / 2) / np.tan(np.radians(22.5))
    tiles = []
    for e in DIRS[:views]:
        fwd = -np.asarray(e, float) / np.linalg.norm(e); eye = c - fwd * r * 2.3
        up = np.array([0, -1, 0.0]); right = np.cross(fwd, up); right /= np.linalg.norm(right); up2 = np.cross(right, fwd)
        Q = P - eye; z = Q @ fwd; x = Q @ right; y = -(Q @ up2)
        ok = z > 1e-3
        u = (W / 2 + f * x[ok] / z[ok]).astype(np.int64); v = (HH / 2 + f * y[ok] / z[ok]).astype(np.int64); zz = z[ok]
        sh = np.clip(0.15 + 0.85 * np.abs(N[ok] @ fwd), 0, 1)
        cc = (BLUE[None, :] if col is None else col[ok]) * sh[:, None]
        img = np.full((HH, W, 3), 24, np.uint8); zb = np.full((HH, W), np.inf)
        order = np.argsort(-zz)                                  # far first: near points overwrite
        for du in (0, 1):
            for dv in (0, 1):
                uu = u[order] + du; vv = v[order] + dv
                m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < HH)
                uu, vv, z2, c2 = uu[m], vv[m], zz[order][m], cc[order][m]
                closer = z2 <= zb[vv, uu] + 0.5
                zb[vv[closer], uu[closer]] = np.minimum(zb[vv[closer], uu[closer]], z2[closer])
                img[vv[closer], uu[closer]] = np.clip(c2[closer] * 255, 0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(img))
    return _sheet(tiles, header)


def _sheet(tiles, header):
    from PIL import Image, ImageDraw
    cols = min(3, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (W * cols, (HH + 20) * rows)); dr = ImageDraw.Draw(out)
    for k, t in enumerate(tiles): out.paste(t, ((k % cols) * W, (k // cols) * (HH + 20) + 20))
    dr.text((6, 2), header[:200], fill=(255, 255, 255))
    return out


def _load_main(files):
    for k in ("points", "mesh"):
        if k in files and os.path.exists(files[k]): return k, files[k]
    return None, None


def placed_sides(sides_json, project, max_faces=1_500_000):
    """[(id, V model frame, F)] of the active sides as placed; each side decimated to its share of max_faces (a
    20-side line-up is ~8M faces: a preview needs far fewer, and the job stays well under 1 GB)."""
    sj = jload(sides_json); out = []
    try: side_colour(None, [d for d in os.listdir(os.path.join(project, "data")) if not d.startswith(".")])
    except OSError: side_colour(None, [s["id"] for s in sj["sides"]])
    for s in sj["sides"]:
        V, F, _ = plyio.read_mesh(os.path.join(project, s["mesh"])); T = np.array(s["T"], float)
        share = max(20000, max_faces // max(1, len(sj["sides"])))
        if len(F) > share:
            import fast_simplification
            V, F = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_count=share)
            V = V.astype(np.float64); F = F.astype(np.int64)
        out.append((s["id"], V @ T[:3, :3].T + T[:3, 3], F))
    return out


def side_labels(X, sides_json, project):
    from scipy.spatial import cKDTree
    from .tools.edits import sample_surface
    ps = placed_sides(sides_json, project); S = []; L = []
    for k, (sid, V, F) in enumerate(ps):
        q = sample_surface(V, F, 30000, k); S.append(q); L.append(np.full(len(q), k))
    tree = cKDTree(np.vstack(S)); _, i = tree.query(X, workers=4)
    lab = np.concatenate(L)[i]
    return np.array([side_colour(ps[k][0]) for k in range(len(ps))])[lab]


def colours(kind, data, color, job):
    """Per-element colours (points: per point; mesh: per face) or None for plain shading. Also a header note."""
    if color == "shaded": return None, ""
    if kind == "points":
        n = len(data["P"])
        if color == "sides":
            S = data["S"].astype(np.uint64); ids = [str(x) for x in data["side_ids"]]; side_colour(None, ids)
            low = np.zeros(n, np.int64); seen = np.zeros(n, bool)
            for k in range(len(ids)):
                b = ((S >> np.uint64(k)) & np.uint64(1)).astype(bool) & ~seen; low[b] = k; seen |= b
            return np.array([side_colour(s) for s in ids])[low], "colour = lowest side seeing it"
        if color == "support":
            S = data["S"].astype(np.uint64); cnt = np.zeros(n, np.int64)
            for k in range(len(data["side_ids"])): cnt += ((S >> np.uint64(k)) & np.uint64(1)).astype(np.int64)
            bad = (cnt <= 1) | (data["NF"] < 5); c = np.tile(GREY, (n, 1)); c[bad] = RED
            return c, "red = seen by one side or < 5 frames (%.1f%%)" % (100 * bad.mean())
    else:
        V, F, _ = data
        if color == "sides" and job.get("sides_json"):
            return side_labels(V[F].mean(1), job["sides_json"], job["project"]), "colour = nearest placed side"
        if color == "scanhere" and job.get("needs"):
            d = plyio.read(job["needs"])["vertex"]
            if d.dtype.names and "red" in d.dtype.names:
                vc = np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.float64) / 255.0
                return vc[F].mean(1), "red = scan here"
    if color == "removed":
        return None, "no removed mask here"
    return None, "(%s n/a here)" % color


def render_any(kind, data, col, views, header):
    if kind == "points": return render_points(data["P"].astype(np.float64), data["N"].astype(np.float64), col, views, header)
    V, F, _ = data
    return render_mesh(V, F, col, views, header)


def header_of(label, kind, data, note=""):
    if kind == "points": return "%s   %d points   %s" % (label, len(data["P"]), note)
    V, F, _ = data; bb = V.max(0) - V.min(0) if len(V) else np.zeros(3)
    return "%s   %d triangles   %.1f x %.1f x %.1f mm   %s" % (label, len(F), *bb, note)


def render_removed(kind, data, sel, png, title=""):
    """The input with the selected elements in red over grey (detect and the removed colour mode)."""
    n = len(data["P"]) if kind == "points" else len(data[1])
    c = np.tile(GREY, (n, 1)); c[sel[:n]] = RED
    render_any(kind, data, c, 6, title).save(png)


# ------------------------------------------------------------------------------------------------------ jobs
def post_job(job):
    """Stats + a 1-view preview (6 side-coloured views for lineup.adjust) for each target, reading each file once."""
    for t in job["targets"]:
        d = t["dir"]
        try:
            if t["node"] == "lineup.adjust":
                ps = placed_sides(t["sides_json"], job["project"])
                V = np.vstack([p[1] for p in ps]); off = np.cumsum([0] + [len(p[1]) for p in ps])[:-1]
                F = np.vstack([p[2] + o for p, o in zip(ps, off)])
                fc = np.vstack([np.tile(side_colour(p[0]), (len(p[2]), 1)) for p in ps])
                render_mesh(V, F, fc, 6, "line-up: %d sides placed (one colour per side)" % len(ps)).save(os.path.join(d, "preview.png"))
                st = {"sides": len(ps), "bbox_mm": [round(float(x), 1) for x in (V.max(0) - V.min(0))]}
            else:
                kind, path = _load_main(t["files"])
                if kind is None: continue
                data = plyio.load_points(path) if kind == "points" else plyio.read_mesh(path)
                st = points_stats(data) if kind == "points" else mesh_stats(data[0], data[1])
                render_any(kind, data, None, 1, header_of(t["label"], kind, data)).save(os.path.join(d, "preview.png"))
            jdump(st, os.path.join(d, "post.json"))
        except Exception as e:
            print("post %s failed: %r" % (t["node"], e), flush=True)
    return {"ok": True}


def preview_job(job):
    files = job["files"]; color = job["color"]
    if job["kind"] == "lineup" or "sides" in files and "mesh" not in files and "points" not in files:
        ps = placed_sides(files["sides"], job["project"])
        V = np.vstack([p[1] for p in ps]); off = np.cumsum([0] + [len(p[1]) for p in ps])[:-1]
        F = np.vstack([p[2] + o for p, o in zip(ps, off)])
        fc = np.vstack([np.tile(side_colour(p[0]), (len(p[2]), 1)) for p in ps])
        render_mesh(V, F, fc, job["views"], "%s: %d sides placed (one colour per side)" % (job["label"], len(ps))).save(job["png"])
        return {"png": job["png"]}
    kind, path = _load_main(files)
    if kind is None: raise ValueError("nothing to render in %s" % files)
    data = plyio.load_points(path) if kind == "points" else plyio.read_mesh(path)
    if color == "removed" and job.get("parent"):
        pk, pp = _load_main(job["parent"])
        pdata = plyio.load_points(pp) if pk == "points" else plyio.read_mesh(pp)
        sel = None
        if os.path.exists(os.path.join(job["dir"], "removed.npy")):
            rj = jload(os.path.join(job["dir"], "removed.json"))
            sel = np.unpackbits(np.load(os.path.join(job["dir"], "removed.npy")))[:rj["n"]].astype(bool)
        elif job.get("tool") == "isolation" and pk == "mesh":
            # script tools keep no mask: isolation is recomputed on request (same rule as keep_part.py)
            from .tools.mesh_tools import apply as mesh_apply
            _new, sel, _st = mesh_apply("isolation", pdata, job["params"])
        if sel is not None:
            render_removed(pk, pdata, sel, job["png"], title="%s   removed %d of %d (red)" % (job["label"], int(sel.sum()), len(sel)))
            return {"png": job["png"]}
    col, note = colours(kind, data, color, job)
    render_any(kind, data, col, job["views"], header_of(job["label"], kind, data, note)).save(job["png"])
    return {"png": job["png"]}


def view_job(job):
    """view.ply for viewer.py: meshes over 3M faces decimated, clouds over 3M points subsampled, colours by mode."""
    kind, path = _load_main(job["files"])
    data = plyio.load_points(path) if kind == "points" else plyio.read_mesh(path)
    col, _ = colours(kind, data, job["color"], job)
    if kind == "points":
        P, N = data["P"].astype(np.float64), data["N"].astype(np.float64)
        if len(P) > MAX_PTS:
            k = np.random.default_rng(0).choice(len(P), MAX_PTS, replace=False); P, N = P[k], N[k]; col = col[k] if col is not None else None
        plyio.write_points_ply(job["out"], P, N, None, col)
        return {"out": job["out"]}
    V, F, N = data
    vcol = None
    if col is not None:                      # face colours -> vertex colours (mean of the faces around each vertex)
        acc = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
        for k in range(3): np.add.at(acc, F[:, k], col); np.add.at(cnt, F[:, k], 1)
        vcol = acc / np.maximum(cnt, 1)[:, None]
    if len(F) > MAX_PTS:
        import fast_simplification
        V2, F2 = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_count=MAX_PTS)
        if vcol is not None:
            from scipy.spatial import cKDTree
            vcol = vcol[cKDTree(V).query(V2)[1]]
        V, F = V2.astype(np.float64), F2.astype(np.int64)
    plyio.write_mesh(job["out"], V, F, plyio.vertex_normals(V, F), vcol)
    return {"out": job["out"]}
