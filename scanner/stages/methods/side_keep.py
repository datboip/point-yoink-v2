# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep components of one saved side."""
import os, re
from .. import params as PR
from . import Method, register, script, legacy_flags

KEPT_RE = re.compile(r"kept (\d+) of (\d+) pieces, (\d+) triangles \(([\d.]+)%\)")


@register
class SideKeep(Method):
    node = "side.keep"; name = "keep_part"; label = "Keep the part"; version = 1; produces = "mesh"; mem_gb = 4; argv_params = True
    scripts = ("keep_part.py",); modules = ("methods/side.py", "methods/side_keep.py"); libs = ("open3d",)

    def execute(self, ctx):
        args = [script("keep_part.py"), ctx.inputs["cut"].files["mesh"], os.path.join(ctx.out_dir, "side.ply")]
        args += legacy_flags(ctx.params, PR.COMMON["side.keep"])
        text = ctx.call(args, mem_gb=self.mem_gb, light=True)              # measured ~0.5 GB peak
        st = {}
        m = KEPT_RE.search(text)
        if m: st = {"pieces_kept": int(m.group(1)), "pieces": int(m.group(2)), "faces": int(m.group(3)), "kept_pct": float(m.group(4))}
        return {"outputs": {"mesh": "side.ply"}, "stats": st}

    def summary(self, p):
        return "isolation %s" % PR.fmt(p["min_share"])
