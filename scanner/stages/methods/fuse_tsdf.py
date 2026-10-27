# SPDX-License-Identifier: AGPL-3.0-or-later
"""fused.tsdf: one fuse.py TSDF over every active side (--set per side, in line-up order), at most frames_per_side
frames each: build_project.py's final fuse. Default op: isolation 3 % (today's keep_part after the fuse)."""
import os

from .. import params as PR
from . import Method, register, script, legacy_flags
from .stage_events import _stage_json
from .fusion_events import _FusionBackend
from ..keys import jload
from .fuse_plus import set_specs


@register
class FusedTsdf(Method):
    node = "fused"; name = "tsdf"; label = "TSDF"; version = 1; produces = "mesh"; gpu = True; argv_params = True
    argv_schema = "method"
    scripts = ("base_cut.py", "fuse.py", "fusion_memory.py",); modules = ("methods/fuse_tsdf.py", "methods/fuse_plus.py", "methods/side.py", "methods/stage_events.py", "methods/fusion_events.py", "tools/edits.py"); libs = ("open3d",)
    default_ops = [{"tool": "isolation", "params": {"rate": 3.0}},
                   {"tool": "match_cleaned_sides", "params": {"distance_mm": 1.5}, "on": False}]

    def execute(self, ctx):
        p = ctx.params; sj = jload(ctx.inputs["lineup"].files["sides"])
        out = os.path.join(ctx.out_dir, "mesh.ply")
        if p["point_distance"] < 0.25: ctx.warn("tsdf below 0.25 mm needs a lot of RAM")
        args = [script("fuse.py"), "--out", out, "--voxel", str(p["point_distance"]), "--gpu"]
        if p["frames_per_side"]: args += ["--set-max-frames", str(p["frames_per_side"])]
        args += legacy_flags(p, PR.METHOD_PARAMS[("fused", "tsdf")])
        args += ["--color",p.get("color","auto")]+["--white-balance","on" if p.get("white_balance",True) else "off"]
        args += set_specs(ctx)[0]
        ctx.legacy("fuse", msg="fusing all sides into one model")
        done = {}; backend = _FusionBackend()

        def on_line(ln):
            backend.on_line(ln)
            j = _stage_json(ln, "integrate")
            if j:
                ctx.progress(j.get("done", 0), j.get("total", 1), "frame %d of %d" % (j.get("done", 0), j.get("total", 1)))
                ctx.legacy("frames", step="all", **j)
            j = _stage_json(ln, "done")
            if j: done.update(j)
        ctx.call(args, mem_gb=ctx.fuse_cap, gpu=True, on_line=on_line)
        stats = {"faces": done.get("faces"), "verts": done.get("verts"), "sides": len(sj["sides"])}
        stats.update(backend.stats(ctx, True))
        return {"outputs": {"mesh": "mesh.ply"}, "stats": stats}

    def summary(self, p):
        return "tsdf %s mm, %s frames/side" % (PR.fmt(p["point_distance"]), p["frames_per_side"] or "all")
