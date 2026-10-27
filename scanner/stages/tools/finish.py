# SPDX-License-Identifier: AGPL-3.0-or-later
"""finish (Meshed tool "Finish for print"): scanner/finish_mesh.py on the chosen mesh, the print recipe the user picked
("E2", 2026-09-27): screened Poisson (depth) -> drop Poisson surface more than trim_mm from the scan -> drop loose
pieces -> close holes up to hole_edges rim edges -> Taubin (smooth passes) -> simplify to `faces` triangles.
A tool on top of the meshed list, so undo / redo, Turn off and the cache work as for any other step; the script's
digest and the app python's PyMeshLab version are in its key. Runs under the app python when that has PyMeshLab (the
venv does), else under the configured PyMeshLab python (POINTYOINK_PML_PY). Its own module, so adding it changed no
other tool's key (mesh_tools.py is hashed into Isolation, Hole Filling, ...)."""
import importlib.util, json, os, sys

from .. import params as PR
from ..methods import Tool, register, script

STAGES = ("poisson", "trim", "holes", "smooth", "simplify")
STAGE_TEXT = {"poisson": "Poisson surface", "trim": "trimming to the scan", "holes": "closing holes", "smooth": "smoothing",
              "simplify": "simplifying"}


def app_has_pymeshlab():
    try: return importlib.util.find_spec("pymeshlab") is not None
    except (ImportError, ValueError): return False


def python_for():
    """(python, key token) the script runs under: the app python when it has PyMeshLab, else the PyMeshLab python."""
    if app_has_pymeshlab():
        try:
            from importlib.metadata import version
            v = version("pymeshlab")
        except Exception: v = "?"
        return None, "pymeshlab-app=" + v
    from ..methods.mesh import pml_python
    from ..methods import lib_version
    return pml_python(), "pymeshlab-pml=" + lib_version("pymeshlab")


def cap_gb(p, faces_in=0):
    """Memory cap: measured 0.99 GB peak, 47 s on 6 cores, at depth 9 on the sprayed headrest (709k faces in, 2026-09-27);
    a finer depth can take up to 4x the octree a level, so it gets more room."""
    base = {6: 3, 7: 3, 8: 3, 9: 4, 10: 8, 11: 16}.get(int(p["depth"]), 4)
    return base + (2 if faces_in > 3_000_000 else 0)


def argv(p, src, out):
    return [script("finish_mesh.py"), src, out, "--faces", str(int(p["faces"])), "--depth", str(int(p["depth"])),
            "--trim", PR.fmt(float(p["trim_mm"])), "--hole-edges", str(int(p["hole_edges"])), "--smooth", str(int(p["smooth"]))]


@register
class Finish(Tool):
    name = "finish"; label = "Finish for print"; kinds = ("mesh",); version = 1; mem_gb = 6
    scripts = ("finish_mesh.py",); modules = ("tools/finish.py",); libs = ("open3d",)

    def code(self, params):
        files, libs = super().code(params)
        return files, libs + [python_for()[1]]

    def execute(self, ctx):
        src = ctx.inputs["parent"].files["mesh"]; out = os.path.join(ctx.out_dir, "mesh.ply")
        from ..plyio import counts
        try: n_in = counts(src).get("face", 0)
        except Exception: n_in = 0
        done = {}

        def on_line(ln):
            if not ln.startswith("STAGE "): return
            parts = ln.split(" ", 2)
            try: j = json.loads(parts[2]) if len(parts) > 2 else {}
            except ValueError: j = {}
            if parts[1] in STAGES:
                k = STAGES.index(parts[1]); ctx.progress(k, len(STAGES), STAGE_TEXT[parts[1]])
            elif parts[1] == "done":
                done.update(j); ctx.progress(len(STAGES), len(STAGES), "done")
        py, _tok = python_for()
        ctx.call(argv(ctx.params, src, out), mem_gb=cap_gb(ctx.params, n_in), python=py, on_line=on_line)
        st = {"faces_in": done.get("faces_in", n_in), "faces": done.get("faces"), "secs": done.get("secs"),
              "stl_mb": round((50 * int(done["faces"]) + 84) / 1e6, 1) if done.get("faces") else None}
        return {"outputs": {"mesh": "mesh.ply"}, "stats": st}

    def summary(self, p):
        return "%s triangles, smooth %d, holes %d edges" % (faces_text(p["faces"]), int(p["smooth"]), int(p["hole_edges"]))


def faces_text(n):
    n = int(n)
    if n >= 1_000_000 and n % 100_000 == 0: return "%gM" % (n / 1e6)
    if n >= 1000 and n % 1000 == 0: return "%dk" % (n // 1000)
    return str(n)


def stl_bytes(faces):
    """Binary STL size: an 84-byte header plus 50 bytes a triangle."""
    return 84 + 50 * int(faces)
