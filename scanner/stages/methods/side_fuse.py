# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuse one saved side without bed-cut dependencies."""
import os
from .. import params as PR
from . import Method, register, script, legacy_flags
from .stage_events import _stage_json
from .fusion_events import _FusionBackend

@register
class SideFuseTsdf(Method):
    node = "side.fuse"; name = "tsdf"; label = "TSDF"; version = 1; produces = "mesh"; gpu = True; argv_params = True
    argv_skip = ("voxel", "gpu")
    scripts=("fuse.py","fusion_memory.py","base_cut.py"); modules = ("methods/side.py", "methods/side_fuse.py", "methods/stage_events.py", "methods/fusion_events.py"); libs = ("open3d",)

    def execute(self, ctx):
        p = ctx.params; S = ctx.project.folder(ctx.step.side); out = os.path.join(ctx.out_dir, "raw.ply")
        args = [script("fuse.py"), "--frames", os.path.join(S, "cache"), "--calib", os.path.join(S, "param", "Pl.bin"),
                "--out", out, "--voxel", str(p["voxel"])] + (["--gpu"] if p["gpu"] else [])
        args += ["--color",p.get("color","auto")]+["--white-balance","on" if p.get("white_balance",True) else "off"]
        args += legacy_flags(p, PR.COMMON["side.fuse"], skip=self.argv_skip)
        done = {}; backend = _FusionBackend()

        def on_line(ln):
            backend.on_line(ln)
            j = _stage_json(ln, "integrate")
            if j:
                ctx.progress(j.get("done", 0), j.get("total", 1), "side %d frame %d of %d" % (ctx.step.n, j.get("done", 0), j.get("total", 1)))
                ctx.legacy("frames", step="side", side=ctx.step.n, sides=ctx.n_sides, **j)
            j = _stage_json(ln, "done")
            if j: done.update(j)
        ctx.call(args, mem_gb=ctx.fuse_cap, gpu=p["gpu"], on_line=on_line)
        stats = {"faces": done.get("faces"), "verts": done.get("verts")}
        stats.update(backend.stats(ctx, p["gpu"]))
        return {"outputs": {"mesh": "raw.ply"}, "stats": stats}

    def summary(self, p):
        return "voxel %s mm" % PR.fmt(p["voxel"])


