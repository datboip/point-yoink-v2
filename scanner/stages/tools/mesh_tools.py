# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mesh tools (fused tsdf meshes and meshed results; the tabs of Revo's Mesh panel).
isolation   keep_part.py in out --min-share rate/100 (today's step after the fuse: the same bytes)
fill_holes  process.py --clean --isolation-rate 0 --smooth-times 0 --fill-holes --hole-size X (Open3D fill, rounded)
simplify    process.py --clean --isolation-rate 0 --smooth-times 0 --simplify-pct N
smooth      in-engine: trimesh Taubin, lamb = nu = clip(0.5 x strength/30, 0.05, 0.9), runaway vertices pinned back
slivers     process.py --clean --isolation-rate 0 --smooth-times 0 --slivers F
process.py jobs get POINTYOINK_MEM_CAP_GB=100000: its own address-space cap fights the CUDA Open3D mappings, so the
systemd cap is the real limit (as for pair_all)."""
import os, re

import numpy as np

from .. import params as PR
from ..methods import Tool, register, script
from .common import run_tool_job

PROC_ENV = {"POINTYOINK_MEM_CAP_GB": "100000"}
KEPT_RE = re.compile(r"kept (\d+) of (\d+) pieces, (\d+) triangles \(([\d.]+)%\)")


def apply(name, data, p, detect=False):
    """Job side. -> (new data, removed mask over the input faces or None, stats)."""
    V, F, N = data
    if name == "isolation":
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
        tri, cnt, _ = m.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
        keep_c = cnt >= p["rate"] / 100.0 * cnt.max() if len(cnt) else np.zeros(0, bool)
        sel = ~keep_c[tri] if len(tri) else np.zeros(0, bool)
        from .edits import apply_mask
        return apply_mask("mesh", data, sel), sel, {"pieces": int(len(cnt)), "kept_pieces": int(keep_c.sum())}
    if name == "smooth":
        import trimesh
        lamb = PR.mesh_smooth_lambda(p["strength"])
        V0 = V.copy(); el = np.linalg.norm(V[F[:, [0, 1, 2]]] - V[F[:, [1, 2, 0]]], axis=2); med = float(np.median(el)) if el.size else 0.0
        m = trimesh.Trimesh(V.copy(), F, process=False)
        trimesh.smoothing.filter_taubin(m, lamb=lamb, nu=lamb, iterations=int(p["times"]))
        V1 = np.asarray(m.vertices, np.float64).copy()
        pinned = 0
        if med > 0:
            bad = np.linalg.norm(V1 - V0, axis=1) > 3.0 * med
            V1[bad] = V0[bad]; pinned = int(bad.sum())
        return (V1, F, None), None, {"lambda": round(lamb, 4), "pinned": pinned, "median_edge_mm": round(med, 4),
                                     "max_move_mm": round(float(np.linalg.norm(V1 - V0, axis=1).max()) if len(V0) else 0.0, 4)}
    raise ValueError("no in-engine mesh tool %r" % name)


class _MeshTool(Tool):
    kinds = ("mesh",)


@register
class Isolation(_MeshTool):
    name = "isolation"; label = "Isolation"; version = 1; mem_gb = 4; detect = True
    scripts = ("keep_part.py",); modules = ("tools/mesh_tools.py",); libs = ("open3d",)

    def execute(self, ctx):
        args = [script("keep_part.py"), ctx.inputs["parent"].files["mesh"], os.path.join(ctx.out_dir, "mesh.ply")]
        rate = float(ctx.params["rate"])
        if round(rate / 100.0, 9) != 0.03: args += ["--min-share", str(round(rate / 100.0, 9))]
        from ..plyio import counts
        text = ctx.call(args, mem_gb=self.mem_gb, light=counts(ctx.inputs["parent"].files["mesh"]).get("face", 0) < 1_500_000)
        st = {}
        for ln in text.strip().splitlines():
            m = KEPT_RE.search(ln)
            if m:
                ctx.line(ln.strip())
                st = {"pieces_kept": int(m.group(1)), "pieces": int(m.group(2)), "faces": int(m.group(3)), "kept_pct": float(m.group(4))}
        return {"outputs": {"mesh": "mesh.ply"}, "stats": st}

    def summary(self, p): return "%s %%" % PR.fmt(p["rate"])


class _ProcessTool(_MeshTool):
    scripts = ("process.py",); modules = ("tools/mesh_tools.py",); libs = ("open3d", "trimesh", "fast_simplification")

    def flags(self, p): return []

    def execute(self, ctx):
        args = [script("process.py"), ctx.inputs["parent"].files["mesh"], os.path.join(ctx.out_dir, "mesh.ply"),
                "--clean", "--isolation-rate", "0", "--smooth-times", "0"] + self.flags(ctx.params)
        text = ctx.call(args, mem_gb=self.mem_gb, env=PROC_ENV)
        st = {}
        for ln in text.splitlines():
            if ln.startswith("STAGE ") and ln.split(" ", 2)[1] in ("filled", "slivers", "cleaned", "done"):
                import json
                try: st[ln.split(" ", 2)[1]] = json.loads(ln.split(" ", 2)[2])
                except ValueError: pass
        return {"outputs": {"mesh": "mesh.ply"}, "stats": st}


@register
class FillHoles(_ProcessTool):
    name = "fill_holes"; label = "Hole Filling"; version = 1; mem_gb = 6

    def flags(self, p): return ["--fill-holes", "--hole-size", str(p["hole_mm"])]

    def summary(self, p): return "auto" if not p["hole_mm"] else "%s mm" % PR.fmt(p["hole_mm"])


@register
class Simplify(_ProcessTool):
    name = "simplify"; label = "Simplify"; version = 1; mem_gb = 6

    def flags(self, p): return ["--simplify-pct", str(p["keep_pct"])]

    def summary(self, p): return "%s %%" % PR.fmt(p["keep_pct"])


@register
class Slivers(_ProcessTool):
    name = "slivers"; label = "Slivers"; version = 1; mem_gb = 6

    def flags(self, p): return ["--slivers", str(p["factor"])]

    def summary(self, p): return "x%s" % PR.fmt(p["factor"])


@register
class Smooth(_MeshTool):
    name = "smooth"; label = "Smooth"; version = 1; mem_gb = 4
    modules = ("tools/mesh_tools.py", "tools/common.py", "plyio.py"); libs = ("trimesh",); tables = ("mesh_smooth",)

    def execute(self, ctx):
        return run_tool_job(ctx, self)

    def summary(self, p): return "%s%% x%d" % (PR.fmt(p["strength"]), p["times"])
