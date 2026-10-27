# SPDX-License-Identifier: AGPL-3.0-or-later
"""fused.plus: the reconstruction bake-off's tsdf_plus (fuse.py --mode plus) over every active side (--set per side, in
line-up order, at most frames_per_side frames each), then drop_enclosed.py (surface sealed inside the part and ragged
rims; keep_part's 3 % rule first). The same frames, clean-up, platter cut and placements as fused.tsdf.
point_distance is the voxel: 0.3 mm is the tuned recipe (preset "plus"); 0.4 mm had no tab tear on the headrest.
smooth_mm is the field blur in mm (fuse.py takes voxels: smooth_mm / point_distance), so both measured recipes come out
of one setting. A failed plus fuse (CUDA out of memory next to the live scanner, or anything else) falls back to the
classic fuse with fuse.py's own defaults, stored flagged for retry. Default op: isolation 3 % (as fused.tsdf)."""
import os

from .. import params as PR
from ..runner import StepError, Cancelled
from . import Method, register, script, legacy_flags
from .stage_events import _stage_json
from .fusion_events import _FusionBackend
from ..keys import jload, jdump


def set_specs(ctx):
    """--set list for fuse.py from the line-up's sides.json (transforms and planes written into out_dir)."""
    sj = jload(ctx.inputs["lineup"].files["sides"]); out = []
    for s in sj["sides"]:
        tp = os.path.join(ctx.out_dir, "T_%s.json" % s["id"]); jdump({"matrix": s["T"]}, tp)
        spec = "%s,%s,%s" % (os.path.join(ctx.project.root, s["frames"]), os.path.join(ctx.project.root, s["calib"]), tp)
        if s.get("plane"):
            pp = os.path.join(ctx.out_dir, "plane_%s.json" % s["id"]); jdump(s["plane"], pp); spec += "," + pp
        elif ctx.step.leaf.get("side_deletes", {}).get(s["id"]):
            spec += ","  # keep the optional plane field before the delete file
        deletes = ctx.step.leaf.get("side_deletes", {}).get(s["id"])
        if deletes:
            dp = os.path.join(ctx.out_dir, "deletes_%s.json" % s["id"])
            jdump({"ops": deletes, "stages": ctx.project.stages}, dp)
            spec += "," + dp
        out += ["--set", spec]
    return out, len(sj["sides"])


def plus_argv(p, raw, sets):
    """fuse.py argv of the plus fuse for resolved params p (the flag mapping is in the step key via argv_params)."""
    args = [script("fuse.py"), "--mode", "plus", "--out", raw, "--voxel", str(p["point_distance"]), "--gpu"]
    if p["frames_per_side"]: args += ["--set-max-frames", str(p["frames_per_side"])]
    args += ["--plus-field-smooth", "%.6g" % (p["smooth_mm"] / p["point_distance"])]
    args += legacy_flags(p, PR.METHOD_PARAMS[("fused", "plus")])
    return args + sets


def classic_argv(p, raw, sets):
    """The fallback: fused.tsdf's command with fuse.py's own defaults (trunc 3 voxels, weight 3)."""
    args = [script("fuse.py"), "--out", raw, "--voxel", str(p["point_distance"]), "--gpu"]
    if p["frames_per_side"]: args += ["--set-max-frames", str(p["frames_per_side"])]
    return args + sets


@register
class FusedPlus(Method):
    node = "fused"; name = "plus"; label = "TSDF plus"; version = 1; produces = "mesh"; gpu = True; argv_params = True
    argv_schema = "method"
    scripts = ("base_cut.py", "fuse.py", "fusion_memory.py", "drop_enclosed.py", "plus_memory.py"); modules = ("methods/fuse_plus.py", "methods/side.py", "methods/stage_events.py", "methods/fusion_events.py", "tools/edits.py"); libs = ("open3d", "scipy")
    default_ops = [{"tool": "isolation", "params": {"rate": 3.0}},
                   {"tool": "match_cleaned_sides", "params": {"distance_mm": 1.5}, "on": False}]
    enclosed_mem_gb = 4

    def execute(self, ctx):
        p = ctx.params
        raw = os.path.join(ctx.out_dir, "raw.ply"); out = os.path.join(ctx.out_dir, "mesh.ply")
        if p["point_distance"] < 0.25: ctx.warn("plus below 0.25 mm needs a lot of RAM and VRAM")
        if p["block_res"] * p["point_distance"] < 3.0:
            ctx.warn("blocks of %s mm: free-space votes reach less than 3 mm, the thin lip may close" % PR.fmt(p["block_res"] * p["point_distance"]))
        sets, n = set_specs(ctx)
        ctx.legacy("fuse", msg="fusing all sides into one model (plus)")
        done = {}; grid = {}; memory = {}; errors = {}; de = {}; backend = _FusionBackend()

        def on_line(ln):
            backend.on_line(ln)
            j = _stage_json(ln, "integrate")
            if j:
                ctx.progress(j.get("done", 0), j.get("total", 1), "frame %d of %d" % (j.get("done", 0), j.get("total", 1)))
                ctx.legacy("frames", step="all", **j)
            j = _stage_json(ln, "phase")                       # after the frames: the grid read-back, smoothing, marching cubes, sorting
            if j:
                # the frames were the first 90% of the stage; the phases after them take the rest, smoothing most of it
                base = {"grid": 90, "extract": 91, "smooth": 92, "mesh": 98, "order": 99}.get(j.get("what"), 95)
                pct = base + (6 * j.get("done", 0) / j["total"] if j.get("what") == "smooth" and j.get("total") else 0)
                ctx.progress(int(pct), 100, j.get("text", "")); ctx.line(j.get("text", ""))
            j = _stage_json(ln, "plus_memory")
            if j: memory.update(j)
            j = _stage_json(ln, "error")
            if j: errors.update(j)
            j = _stage_json(ln, "warn")
            if j and str(j.get("msg", "")).startswith("plus:"): ctx.warn(j["msg"])
            j = _stage_json(ln, "plus_grid")
            if j: grid.update(j)
            j = _stage_json(ln, "done")
            if j: done.update(j)
            j = _stage_json(ln, "drop_enclosed")
            if j: de.update(j)
        stats = {"sides": n, "mode": "plus"}
        try:
            ctx.call(plus_argv(p, raw, sets)+["--color",p.get("color","auto")]+["--white-balance","on" if p.get("white_balance",True) else "off"], mem_gb=ctx.fuse_cap, gpu=True, on_line=on_line)
        except Cancelled:
            raise
        except StepError as e:
            if not p["fallback"]: raise
            why = "%s, exit %s" % (e.kind, e.rc)
            warning = "Built the classic way: plus ran out of GPU memory" if errors.get("oom") or "memory" in str(errors.get("msg", "")).lower() else "Built the classic way: plus failed"
            ctx.legacy("warn", msg="plus fuse failed (%s); using the classic fuse" % why)
            done.clear(); backend = _FusionBackend()
            ctx.call(classic_argv(p, raw, sets)+["--color",p.get("color","auto")]+["--white-balance","on" if p.get("white_balance",True) else "off"], mem_gb=ctx.fuse_cap, gpu=True, on_line=on_line)
            os.replace(raw, out)
            stats.update(mode="classic (fallback)", fallback_reason=errors.get("msg", why), spacing_mm=p["point_distance"], requested_spacing_mm=p["point_distance"], faces=done.get("faces"), verts=done.get("verts"))
            stats.update(backend.stats(ctx, True))
            return {"outputs": {"mesh": "mesh.ply"}, "stats": stats, "transient": True,
                    "fail": {"kind": e.kind, "rc": e.rc, "sig": e.sig},
                    "warn": [warning + " (" + why + ")"]}
        stats.update(backend.stats(ctx, True))
        stats.update(faces_fused=done.get("faces"), blocks=grid.get("blocks"), grid_mb=grid.get("grid_mb"),
                     fuse_peak_rss_gb=done.get("peak_rss_gb"), requested_spacing_mm=p["point_distance"],
                     spacing_mm=done.get("voxel", p["point_distance"]), memory_plan=memory)
        if p["drop_enclosed"]:
            ctx.call([script("drop_enclosed.py"), raw, out, "--rays", str(p["enclosed_rays"]), "--tidy-rounds", str(p["tidy_rounds"])],
                     mem_gb=self.enclosed_mem_gb, on_line=on_line)
            os.remove(raw)
            stats.update(enclosed_cm2=de.get("enclosed_cm2"), tidy_triangles=de.get("tidy_triangles"), faces=de.get("triangles_out"))
        else:
            os.replace(raw, out); stats["faces"] = done.get("faces")
        return {"outputs": {"mesh": "mesh.ply"}, "stats": stats}

    def summary(self, p):
        return "plus %s mm, %s frames/side, blur %s mm on %d+ sides%s" % (
            PR.fmt(p["point_distance"]), p["frames_per_side"] or "all", PR.fmt(p["smooth_mm"]), p["smooth_min_sides"],
            ", enclosed dropped" if p["drop_enclosed"] else "")
