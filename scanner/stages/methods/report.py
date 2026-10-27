# SPDX-License-Identifier: AGPL-3.0-or-later
"""report (ours, "Scan here"): needs_scan.py (open edges, red = scan here; soft fail) and render_views.py (model.png)
on the meshed result. Publish follows it (engine)."""
import json, os

from .. import params as PR
from ..runner import StepError
from . import Method, register, script, legacy_flags
from ..keys import jload, jdump


@register
class Report(Method):
    node = "report"; name = "needs"; label = "Scan here (needs_scan)"; version = 1; produces = "report"; mem_gb = 6; argv_params = True
    scripts = ("needs_scan.py", "render_views.py"); modules = ("methods/report.py", "plyio.py"); libs = ("open3d", "Pillow", "numpy", "scipy")

    def execute(self, ctx):
        mesh = ctx.inputs["mesh"].files["mesh"]; o = ctx.out_dir
        from ..plyio import counts
        small = counts(mesh).get("face", 0) < 1_500_000
        ctx.call([script("render_views.py"), mesh, os.path.join(o, "model.png")], mem_gb=4, light=small)
        outs = {"preview": "model.png"}; warn = []; st = {}
        args = [script("needs_scan.py"), mesh, os.path.join(o, "needs.ply"), "--png", os.path.join(o, "needs.png"),
                "--json", os.path.join(o, "needs.json")] + legacy_flags(ctx.params, PR.COMMON["report"])
        try:
            from ..plyio import counts
            faces = counts(mesh).get("face", 0)
            # measured 0.56 GB peak on a 750k-face model: the full cap (6) only for big meshes (Poisson Q7 is ~9M faces)
            text = ctx.call(args, mem_gb=4 if faces < 1_500_000 else self.mem_gb, light=faces < 1_500_000)
            for ln in text.strip().splitlines():
                if ln.startswith(("open edge", "  place")): ctx.line(ln)
            outs.update(needs="needs.ply", needs_png="needs.png", needs_json="needs.json")
            j = jload(os.path.join(o, "needs.json"))
            st = {k: j.get(k) for k in ("open_edge_mm", "runs", "pinholes", "red_share", "size_mm")}
            st["places"] = j.get("places", [])[:12]
        except StepError as e:
            warn.append("scan-here map failed: " + str(e)[:120]); ctx.legacy("warn", msg="scan-here map failed: " + (str(e) + ": " + e.tail)[:120])
            if not e.deliberate:
                # killed / crashed / not launched: publish the model without a scan-here map, stored flagged for retry
                # (engine._transient): the next run makes the map again
                return {"outputs": outs, "stats": st, "warn": warn, "transient": True, "fail": {"kind": e.kind, "rc": e.rc, "sig": e.sig}}
        return {"outputs": outs, "stats": st, "warn": warn}

    def summary(self, p):
        return "min %s mm" % PR.fmt(p["min_mm"])


@register
class Textured(Method):
    node = "textured"; name = "none"; label = "colour comes later"
