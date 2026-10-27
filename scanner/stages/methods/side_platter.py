# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saved-side bed detection and cutting."""
import glob, json, os, re
from .. import params as PR
from . import Method, register, script, legacy_flags
from ..runner import StepError
from ..keys import jload, jdump
from .stage_events import _stage_json

PLATTER_RE = re.compile(r"platter: (\d+) support points, ([\d.]+) deg off the spin axis; kept (\d+) triangles in (\d+) pieces, part height ([-\d.]+) mm")


def observe_no_base(job):
    """Stage-job entry for the saved-frame visibility check."""
    from scanner.no_base import observe
    return observe(job)


class _Platter(Method):
    node = "side.platter"; version = 1; produces = "mesh"; mem_gb = 8; soft_fail = True; argv_params = True
    scripts = ("strip_table.py", "fuse.py", "board_plane.py", "bed.py", "table_plane.py", "no_base.py", "base_geometry.py", "base_cut.py"); modules = ("methods/side.py", "methods/side_platter.py", "methods/stage_events.py"); libs = ("open3d", "numpy", "scipy")

    def board_plane(self, ctx):
        # Keep provenance in the keyed artifact; never mutate the side input during a build.
        if getattr(self, "_board_context", None) == ctx.out_dir:
            return self._board_file
        self._board_context = ctx.out_dir; self._board_file = None
        from scanner.board_plane import generic, supported
        side = ctx.project.folder(ctx.step.side)
        live = os.path.join(side, "table_plane.json")
        saved = jload(live) if os.path.exists(live) else {}
        if supported(saved): self._board_file = live; return live
        if not generic(saved) or not os.path.isdir(os.path.join(side, "cache")): return None
        if not glob.glob(os.path.join(side, "cache", "*.irl")): return None
        candidate = os.path.join(ctx.out_dir, "board_plane.json")
        ctx.call([script("board_plane.py"), side, candidate], mem_gb=2, light=True)
        if os.path.exists(candidate) and supported(jload(candidate)):
            self._board_file = candidate
        return self._board_file

    def plane_arg(self, ctx):
        return None

    def execute(self, ctx):
        S = ctx.project.folder(ctx.step.side); raw = ctx.inputs["raw"].files["mesh"]
        args = [script("strip_table.py"), S, os.path.join(ctx.out_dir, "cut.ply"), "--mesh", raw,
                "--plane-out", os.path.join(ctx.out_dir, "plane.json")]
        if self.name in ("auto","plate"):
            touch=os.path.isfile(os.path.join(S,"touchup.json"))
            live=os.path.isfile(os.path.join(S,"table_plane.json"))
            if touch:evidence=dict(no_base=True,frames=0)
            elif not live and os.path.isdir(os.path.join(S,"cache")) and not self.board_plane(ctx):
                evidence=ctx.job("methods.side:observe_no_base",{"folder":S,"size":ctx.project.frame_size(ctx.step.side)},mem_gb=1)
            else:evidence=dict(no_base=False)
            if evidence.get("no_base"):
                ctx.legacy("table",side=ctx.step.n,msg="No base in view: nothing to cut")
                return {"outputs":{},"alias_of":"raw","stats":{"source":"no base in view","base_cut_skipped":True,"lineup":"shape","observed_frames":evidence.get("frames",0)}}
        pa=self.plane_arg(ctx)
        if pa: args += ["--plane",pa]
        args += legacy_flags(ctx.params, PR.COMMON["side.platter"])
        try:
            text = ctx.call(args, mem_gb=self.mem_gb, light=True)          # measured ~0.6 GB peak
        except StepError as e:
            rejected = next((j for line in e.tail.splitlines() if (j := _stage_json(line, "cut_rejected"))), None)
            if rejected and rejected.get("reason") == "empty_cut":
                raise ValueError("Side %d: base cut at %g mm removed every triangle. Lower the margin, adjust the plane, "
                                 "or choose Off to keep this side uncut." % (ctx.step.n, rejected.get("margin", ctx.params["margin"]))) from e
            if self.name == "manual":
                raise StepError("Side %d: manual base cut failed. Processing stopped to keep the removed base out. "
                                "Retry the cut or explicitly choose Off." % ctx.step.n,
                                e.tail, e.rc, e.kind) from e
            msg = "%s failed (%s): %s" % ("strip_table.py", e.rc, e.tail)
            ctx.legacy("warn", msg="side %d: table cut failed (%s); using it uncut" % (ctx.step.n, msg[:120]))
            if not e.deliberate:
                # killed (memory cap, OOM), crashed, not launched, or anything unrecognised: the side goes on uncut, the
                # result is stored flagged for retry (engine._transient), so the next run tries the cut again
                return {"outputs": {}, "alias_of": "raw", "transient": True, "stats": {"source": "failed", "why": e.kind},
                        "fail": {"kind": e.kind, "rc": e.rc, "sig": e.sig},
                        "warn": ["table cut failed (%s, exit %s); using the side uncut" % (e.kind, e.rc)]}
            return {"outputs": {}, "alias_of": "raw", "warn": ["table cut failed; using it uncut"], "stats": {"source": "failed"}}
        msg = text.strip()
        ctx.legacy("table", side=ctx.step.n, msg=msg)
        st = {"source": self.source(ctx)}
        support = next((j for line in text.splitlines() if (j := _stage_json(line, "turntable_support"))), None)
        if support:
            st["turntable_support"] = support
        wrong = next((j for line in text.splitlines() if (j := _stage_json(line, "wrong_plane"))), None)
        if wrong and self.name in ("auto", "plate") and not (pa and self.trusted_plane(pa)):
            # the plane runs through the part: that cut would throw away most of the side (sides 11 and 12 of the 2026-10-06 scan lost
            # 60 to 65 %). The side goes on uncut instead: its plate is removed later by Isolation or a Cut base by hand, which costs
            # nothing, while surface cut away is gone for good
            ctx.warn("Side %d: the base plane found runs through the part (it would remove %.0f%% of the side), so this side was left "
                     "uncut. Set its base by hand in Cut base, or leave it: Isolation and the other sides' cuts take the plate out." % (ctx.step.n, 100 * wrong["removed_share"]))
            return {"outputs": {}, "alias_of": "raw", "stats": {"source": "uncut (plane crossed the part)", "wrong_plane_removed_share": wrong["removed_share"]}}
        if wrong:
            st["removed_share"] = wrong["removed_share"]
        if pa and self.trusted_plane(pa):
            st["source"] = "board dots"; st["board_provenance"] = jload(pa)["provenance"]
        m = PLATTER_RE.search(text)
        if m:
            st.update(support=int(m.group(1)), tilt_deg=float(m.group(2)), faces_kept=int(m.group(3)), pieces=int(m.group(4)),
                      part_height_mm=float(m.group(5)))
        return {"outputs": {"mesh": "cut.ply", "plane": "plane.json"}, "stats": st}

    @staticmethod
    def trusted_plane(path):
        from scanner.board_plane import supported
        return supported(jload(path))

    def source(self, ctx):
        return self.name

    def summary(self, p):
        return self.name + ("" if p.get("margin") == 1.0 else ", margin %s mm" % PR.fmt(p["margin"]))


@register
class PlatterAuto(_Platter):
    name = "auto"; label = "Auto (live plane, else search)"

    def plane_arg(self, ctx):
        board = self.board_plane(ctx)
        if board: return board
        live = os.path.join(ctx.project.folder(ctx.step.side), "table_plane.json")
        return live if os.path.exists(live) else None

    def source(self, ctx):
        return "live plane" if self.plane_arg(ctx) else "search"


@register
class PlatterPlate(_Platter):
    name = "plate"; label = "Plate top (guarded Auto fallback)"
    version = 2
    scripts = ("plate_top.py", "strip_table.py", "fuse.py", "board_plane.py", "bed.py", "table_plane.py", "no_base.py", "base_geometry.py", "base_cut.py")

    def execute(self, ctx):
        board = self.board_plane(ctx)
        if board:
            self._chosen_plane = board
            return super().execute(ctx)
        S = ctx.project.folder(ctx.step.side)
        raw = ctx.inputs["raw"].files["mesh"]
        plane = os.path.join(ctx.out_dir, "plate_top.json")
        report_path = os.path.join(ctx.out_dir, "plate_report.json")
        args = [script("plate_top.py"), S, raw, plane, report_path, "--margin", str(ctx.params["margin"]),
                "--max-tilt", str(ctx.params["max_tilt"])]
        live = os.path.join(S, "table_plane.json")
        if os.path.exists(live): args += ["--live-plane", live]
        try:
            ctx.call(args, mem_gb=self.mem_gb, light=True)
        except StepError as e:
            result = PlatterAuto().execute(ctx)
            result.setdefault("warn", []).append("plate top search failed (%s); used Auto" % e.kind)
            return result
        report = jload(report_path)
        if not os.path.exists(plane):
            result = PlatterAuto().execute(ctx)
            reason = report.get("reason", "plate top could not be verified")
            result.setdefault("warn", []).append("plate top rejected (%s); used Auto" % reason)
            result.setdefault("stats", {})["plate_top"] = report
            return result
        self._chosen_plane = plane
        result = super().execute(ctx)
        result.setdefault("stats", {})["plate_top"] = report
        return result

    def plane_arg(self, ctx):
        return self._chosen_plane

    def source(self, ctx):
        return "plate top"


@register
class PlatterSearch(_Platter):
    name = "search"; label = "Search (ignore the live plane)"


@register
class PlatterManual(_Platter):
    name = "manual"; label = "Manual plane"

    def plane_arg(self, ctx):
        pl = ctx.params.get("plane")
        if not pl: raise StepError("manual platter without a plane")
        p = os.path.join(ctx.out_dir, "manual_plane.json"); jdump(pl, p)
        return p


@register
class PlatterNone(_Platter):
    name = "none"; label = "None (keep the platter)"

    def static_alias(self, step, graph):
        return "raw"


