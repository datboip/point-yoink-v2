# SPDX-License-Identifier: AGPL-3.0-or-later
"""meshed (Revo "Meshed"): auto (as_fused for a mesh, poisson for points), as_fused (alias), poisson, bpa.

poisson: octree depth = clamp(round(log2(max bbox extent / grid)), 6, 11), grid from the Quality table; faces with a
vertex further than trim (or fill_trim with hole_fill) from the points are cut, small pieces dropped, light Taubin.
Backend pml = stages/pml_mesh.py under the PyMeshLab python (env POINTYOINK_PML_PY or
~/.config/pointyoink/stages.json {"pml_python": ...}); o3d = Open3D Poisson in the app venv (no confidence weights).
When the fused result is a mesh its vertices and vertex normals are the points (quality 1)."""
import json, os

import numpy as np

from .. import params as PR
from ..resources import config_dir
from . import Method, register, script, HERE
from ..keys import jload, jdump


def pml_python():
    p = os.environ.get("POINTYOINK_PML_PY")
    if p and os.path.exists(p): return p
    try:
        c = jload(str(config_dir() / "stages.json")).get("pml_python")
        if c and os.path.exists(c): return c
    except Exception: pass
    return None


@register
class AsFused(Method):
    node = "meshed"; name = "as_fused"; label = "As fused (the fused mesh)"; version = 1; produces = "mesh"

    def static_alias(self, step, graph):
        return "fused"


@register
class Auto(Method):
    node = "meshed"; name = "auto"; label = "Auto"; version = 1; produces = "mesh"


def _points_input(ctx):
    f = ctx.inputs["fused"]
    return f.files.get("points") or f.files.get("mesh")


@register
class Poisson(Method):
    node = "meshed"; name = "poisson"; label = "Poisson"; version = 1; produces = "mesh"; mem_gb = 10
    modules = ("methods/mesh.py", "plyio.py"); tables = ("poisson",)        # depth / scale from params.poisson_plan

    @staticmethod
    def backend_of(p):
        b = p.get("backend", "auto")
        if b == "auto": return "pml" if pml_python() else "o3d"
        if b == "pml" and not pml_python(): return "o3d"
        return b

    def code(self, params):
        files, libs = super().code(params)
        be = self.backend_of(params)
        if be == "pml": files.append(("stages/pml_mesh.py", os.path.join(HERE, "pml_mesh.py"))); libs += ["pymeshlab", "numpy", "scipy"]
        else: libs += ["open3d", "scipy"]
        return files, libs + ["backend=" + be]          # auto resolves by what is configured: part of the key

    def execute(self, ctx):
        p = ctx.params; o = ctx.out_dir
        info = ctx.job("methods.mesh:points_ply_job", {"in": _points_input(ctx)}, mem_gb=4)
        plan = PR.poisson_plan(info["extent"], p["quality"]); depth, scale = plan["depth"], plan["scale"]
        cap = 12 if depth >= 11 else self.mem_gb
        trim = p["fill_trim"] if p["hole_fill"] else p["trim"]
        if p["backend"] == "pml" and not pml_python():
            ctx.warn("no PyMeshLab python configured (POINTYOINK_PML_PY): using Open3D Poisson")
        backend = self.backend_of(p); mesh_stats = {}
        pts = os.path.join(o, "points.ply")
        if backend == "pml":
            args = [os.path.join(HERE, "pml_mesh.py"), pts, os.path.join(o, "mesh.ply"), "--depth", str(depth),
                    "--pointweight", str(p["pointweight"]), "--spn", str(p["samples_per_node"]), "--trim", str(trim),
                    "--min-piece", str(p["min_piece"]), "--taubin", str(p["taubin"])] + (["--confidence"] if p["confidence"] else [])
            if abs(scale - PR.POISSON_SCALE) > 1e-9: args += ["--scale", str(scale)]
            text = ctx.call(args, mem_gb=cap, python=pml_python())
        else:
            mesh_stats = ctx.job("methods.mesh:poisson_o3d_job", {"points": pts, "out": os.path.join(o, "mesh.ply"), "depth": depth, "trim": trim,
                                                     "scale": scale, "min_piece": p["min_piece"], "taubin": p["taubin"]}, mem_gb=cap)
        try: os.remove(pts)
        except OSError: pass
        ctx.line("poisson: " + PR.poisson_label(plan))
        return {"outputs": {"mesh": "mesh.ply"}, "stats": {"depth": depth, "scale": scale, "grid_mm": plan["grid_mm"],
                                                          "same_as": plan["same_as"], "extent_mm": round(info["extent"], 1),
                                                          "backend": backend, "trim_mm": trim, "points": info["n"],
                                                          **({"threads": mesh_stats["threads"]} if "threads" in mesh_stats else {})}}

    def summary(self, p):
        # the grid depends on the part: the run's stats carry the one really used (depth, grid_mm)
        return "Q%d%s" % (p["quality"], ", holes filled" if p["hole_fill"] else "")


@register
class Bpa(Method):
    node = "meshed"; name = "bpa"; label = "Ball pivoting (keeps openings)"; version = 1; produces = "mesh"; mem_gb = 8
    scripts = ("process.py",); modules = ("methods/mesh.py", "plyio.py"); libs = ("open3d", "trimesh")

    def execute(self, ctx):
        o = ctx.out_dir
        ctx.job("methods.mesh:points_ply_job", {"in": _points_input(ctx)}, mem_gb=4)
        ctx.call([script("process.py"), os.path.join(o, "points.ply"), os.path.join(o, "mesh.ply"), "--rebuild-points"],
                 mem_gb=self.mem_gb, env={"POINTYOINK_MEM_CAP_GB": "100000"})
        os.remove(os.path.join(o, "points.ply"))
        return {"outputs": {"mesh": "mesh.ply"}, "stats": {}}


# ------------------------------------------------------------------------------------------------------ jobs
def points_ply_job(job):
    """points.npz or a mesh PLY -> out_dir/points.ply (x y z nx ny nz quality); returns the count and bbox extent."""
    from .. import plyio
    src = job["in"]; out = os.path.join(job["out_dir"], "points.ply")
    if src.endswith(".npz"):
        d = plyio.load_points(src); P = d["P"].astype(np.float64); N = d["N"].astype(np.float64); Q = np.log1p(d["W"].astype(np.float64))
    else:
        V, F, N = plyio.read_mesh(src); P = V
        if N is None or not len(N): N = plyio.vertex_normals(V, F)
        Q = np.ones(len(P))
    plyio.write_points_ply(out, P, N, Q)
    ext = float((P.max(0) - P.min(0)).max()) if len(P) else 0.0
    return {"n": int(len(P)), "extent": ext}


def poisson_o3d_job(job):
    """Open3D Poisson on points + normals, then the same distance trim, min_piece and Taubin as pml_mesh.py."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    from .. import plyio
    d = plyio.read(job["points"], faces=False)["vertex"]
    P = plyio.xyz(d); N = plyio.normals(d)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals = o3d.utility.Vector3dVector(N)
    threads = max(1, int(job.get("threads", min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))))
    m, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=int(job["depth"]), scale=float(job.get("scale", 1.1)),
                                                                         n_threads=threads)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    print("poisson: %d verts %d faces" % (len(V), len(F)), flush=True)
    dist = cKDTree(P).query(V, workers=min(threads, 8))[0]
    keep = (dist[F] <= job["trim"]).all(1)
    m.remove_triangles_by_mask(~keep); m.remove_unreferenced_vertices()
    tri, cnt, _ = m.cluster_connected_triangles(); tri = np.asarray(tri); cnt = np.asarray(cnt)
    if len(cnt):
        nf = len(m.triangles); small = cnt < max(25, int(job["min_piece"] * nf))
        m.remove_triangles_by_mask(small[tri]); m.remove_unreferenced_vertices()
    if job["taubin"] > 0: m = m.filter_smooth_taubin(number_of_iterations=int(job["taubin"]), lambda_filter=0.5, mu=-0.53)
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(job["out"], m)
    print("final: %d verts %d faces" % (len(m.vertices), len(m.triangles)), flush=True)
    return {"faces": len(m.triangles), "vertices": len(m.vertices), "threads": threads}


def export_job(job):
    """export <out.stl|.obj>: the mesh in another format (trimesh)."""
    import trimesh
    trimesh.load(job["in"], force="mesh", process=False).export(job["out"])
    return {"out": job["out"]}
