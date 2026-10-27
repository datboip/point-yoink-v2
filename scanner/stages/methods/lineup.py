# SPDX-License-Identifier: AGPL-3.0-or-later
"""Line-up nodes (model frame = the base side's scan frame, as today):
lineup.pairs   every pair of active sides fitted with pair_all's own functions; fits cached by content in
               stages/cache/pairs.json (key: both side meshes' digests and platter planes), so adding or leaving out a
               side never refits old pairs
lineup.place   place_sides.py, run unchanged in a temp dir of symlinks named the way it expects
lineup.fix     flip, match and turn ops (tools/edits.py fix_job): absolute poses kept against a placed side, each ended by a
               local point-to-plane fit (also with 2 sides, where refine is an alias)
lineup.refine  refine_sides.py, unchanged (3+ active sides; otherwise an alias, as in build_project)
lineup.adjust  nudge ops after refine (so refine never undoes a hand adjustment); writes lineup.json and sides.json,
               the frame contract every fused method reads"""
import json, os, re, sys, time

import numpy as np

from .. import params as PR
from ..keys import H
from . import Method, register, script, legacy_flags, SCANNER
from ..keys import jload, jdump
from .pairs import PairCache, pair_version, pair_token, plane_of, plane_digest     # noqa: F401  (legacy.py imports them here)


def _load(path): return jload(path)


def _dump(obj, path):
    with open(path, "w") as f: json.dump(obj, f)


def active_ids(ctx):
    return [k[5:] for k in ctx.step.inputs if k.startswith("mesh@")]


@register
class Pairs(Method):
    node = "lineup.pairs"; name = "fpfh_icp"; label = "FPFH + ICP (pair_all)"; produces = "pairs"; mem_gb = 6; light = True
    scripts = ("seeded_sample.py", "pair_all.py", "support.py", "align.py", "registration_metrics.py"); modules = ("methods/lineup.py", "methods/pairs.py")
    libs = ("open3d",)

    @property
    def version(self): return pair_version()

    def execute(self, ctx):
        ids = active_ids(ctx); V = pair_token(); p = ctx.params; t0 = time.time()
        cache = PairCache(os.path.join(ctx.project.stages, "cache", "pairs.json"))
        info = {s: dict(digest=ctx.inputs["mesh@" + s].digest, plane=plane_digest(ctx.inputs["plane@" + s]),
                        mesh=ctx.inputs["mesh@" + s].files["mesh"], plane_obj=plane_of(ctx.inputs["plane@" + s]), n=ctx.side_n(s))
                for s in ids}
        todo = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        pv = pair_version()

        def lookup(c, a, b):
            """A fit of the current code, else (unless this run was forced) one adopted from an older pair_all."""
            e = c.get(V, p, info[a]["digest"], info[b]["digest"], info[a]["plane"], info[b]["plane"])
            if e is None and not ctx.forced:
                e = c.get_old(p, info[a]["digest"], info[b]["digest"], info[a]["plane"], info[b]["plane"], pv)
            return e
        missing = [(i, j) for i, j in todo if lookup(cache, ids[i], ids[j]) is None]
        fresh = 0
        if missing:
            job = {"version": V, "params": p, "cache": cache.path, "total": len(todo),
                   "sides": [dict(id=s, n=info[s]["n"], mesh=info[s]["mesh"], digest=info[s]["digest"], plane_digest=info[s]["plane"],
                                  plane=info[s]["plane_obj"]) for s in ids],
                   "todo": [[i, j, todo.index((i, j))] for i, j in missing]}

            def on_line(ln):
                if not ln.startswith("STAGE "): return
                parts = ln.split(" ", 2)
                try: j = json.loads(parts[2])
                except (ValueError, IndexError): return
                if parts[1] == "align":
                    ctx.progress(j.get("done", 0), j.get("total", 1), j.get("msg", "")); ctx.legacy("align", **j)
                elif parts[1] in ("aligned", "warn"): ctx.legacy(parts[1], **j)
            ctx.job("methods.pairs:pairs_job", job, mem_gb=self.mem_gb, on_line=on_line)
            cache = PairCache(cache.path); fresh = len(missing)
        pairs = []; weak = 0; mset = set(map(tuple, missing)); adopted = not missing; old = 0
        for k, (i, j) in enumerate(todo):
            a, b = ids[i], ids[j]
            e = lookup(cache, a, b)
            if e is None: raise RuntimeError("pair %s-%s missing after the fit" % (a, b))
            if e.get("old_version"): old += 1
            if (i, j) not in mset:
                for m in e.get("warns", []): ctx.legacy("warn", msg=m)
                ctx.legacy("aligned", sides="%d-%d" % (info[a]["n"], info[b]["n"]), fit=round(e["fitness"], 2), rmse_mm=round(e["rmse"], 2), bed="-")
            if e["fitness"] < 0.3: weak += 1
            if not e.get("adopted"): adopted = False
            pairs.append({"base": a, "moving": b, "matrix": e["matrix"], "fitness": e["fitness"], "rmse": e["rmse"],
                          "support": e.get("support")})
        _dump({"version": V, "sides": ids, "pairs": pairs}, os.path.join(ctx.out_dir, "pairs.json"))
        secs = time.time() - t0
        ctx.line("pairs: %d fitted, %d from the cache, %.0f s" % (fresh, len(todo) - fresh, secs))
        warn = ["%d pair fit%s from an older pair_all (adopted); `run --force lineup.pairs` refits them" % (old, "" if old == 1 else "s")] if old else []
        return {"outputs": {"pairs": "pairs.json"}, "adopted": adopted and bool(todo), "warn": warn,
                "stats": {"n": len(todo), "fitted": fresh, "cached": len(todo) - fresh, "weak": weak, "old_fits": old,
                          "secs": round(secs, 1)}}

    def summary(self, p):
        return "voxel %s mm" % PR.fmt(p["voxel"])


# ------------------------------------------------------------------------------------------------------ place
def auto_base(ids, pairs):
    """build_project's rule: the side with the highest summed pair fitness (first one on a tie)."""
    n = len(ids); fit = {}
    for e in pairs:
        i, j = ids.index(e["base"]), ids.index(e["moving"]); fit[(i, j)] = fit[(j, i)] = float(e["fitness"])
    return max(range(n), key=lambda i: sum(fit[(i, j)] for j in range(n) if j != i)) if n > 1 else 0


@register
class Place(Method):
    node = "lineup.place"; name = "consensus"; label = "Consensus (place_sides)"; version = 1; produces = "placement"; mem_gb = 4
    argv_params = True
    scripts = ("place_sides.py", "support.py"); modules = ("methods/lineup.py", "methods/pairs.py"); libs = ("open3d",)

    def execute(self, ctx):
        pj = _load(ctx.inputs["pairs"].files["pairs"]); ids = pj["sides"]; M = len(ids)
        base = ctx.params.get("base", "auto")
        b = ids.index(base) if base in ids else auto_base(ids, pj["pairs"])
        ctx.legacy("base", side=ctx.side_n(ids[b]), msg="side %d fits best with the others" % ctx.side_n(ids[b]))
        out = os.path.join(ctx.out_dir, "placement.json")
        if M == 1:
            _dump({"base": ids[0], "order": [ids[0]], "T": {ids[0]: np.eye(4).tolist()}, "score": {ids[0]: 1.0}, "flagged": []}, out)
            return {"outputs": {"placement": "placement.json"}, "stats": {"base": ids[0], "order": [ids[0]]}}
        work = os.path.join(ctx.out_dir, "legacy"); os.makedirs(work)
        beds = {"version": 1, "sides": {}}
        for m, s in enumerate(ids):
            os.symlink(ctx.inputs["mesh@" + s].files["mesh"], os.path.join(work, "side%d.ply" % (m + 1)))
            pl = plane_of(ctx.inputs["plane@" + s])
            beds["sides"][str(m + 1)] = {"folder": s, "plane": None if pl is None else {"n": pl["n"], "d": pl["d"]}, "B_scan": None, "bed_id": None}
        for e in pj["pairs"]:
            i, j = ids.index(e["base"]), ids.index(e["moving"])
            _dump({"matrix": e["matrix"], "fitness": e["fitness"], "rmse": e["rmse"]}, os.path.join(work, "T_%d_to_%d.json" % (j + 1, i + 1)))
        _dump(beds, os.path.join(work, "beds.json"))
        args = [script("place_sides.py"), "--work", work, "--n", str(M), "--base", str(b), "--beds", os.path.join(work, "beds.json")]
        args += legacy_flags(ctx.params, PR.COMMON["lineup.place"])
        hints = []
        for part in str(ctx.params.get("hints") or "").split(","):                  # "side id:flipped": how that side was scanned (the placement skips fits that contradict it)
            sid, _, how = part.strip().rpartition(":")
            if sid in ids and how in ("same", "flipped", "sideways"): hints.append("%d:%s" % (ids.index(sid), how))
        if hints: args += ["--hints", ",".join(hints)]
        text = ctx.call(args, mem_gb=self.mem_gb)
        same = [s for s in ids] == [ctx.side_ids_all[k] for k in range(len(ids))] if len(ids) == len(ctx.side_ids_all) else False
        for ln in text.strip().splitlines():
            if ln.startswith("STAGE warn "):
                try:
                    warning = json.loads(ln.split(" ", 2)[2]); msg = warning.get("msg", "Placement warning")
                    if not same:      # place_sides counts the active sides only: say the positions the user knows
                        msg = re.sub(r"(?i)\bside (\d+)", lambda mm: mm.group(0)[:4] + " %d" % ctx.side_n(ids[int(mm.group(1)) - 1]), msg)
                    warning["msg"] = msg; ctx.legacy("warn", **warning)
                    ctx.warn(msg)
                except ValueError: pass
                continue
            if not same:      # place_sides counts the active sides only: say the positions the user knows
                ln = re.sub(r"side (\d+)", lambda mm: "side %d" % ctx.side_n(ids[int(mm.group(1)) - 1]), ln)
            ctx.line(ln)
        pl = _load(os.path.join(work, "placement.json"))
        res = {"base": ids[b], "order": [ids[k] for k in pl["order"]], "T": {ids[int(k)]: v for k, v in pl["T"].items()},
               "score": {ids[int(k)]: v for k, v in pl["score"].items()}, "flagged": [ids[k] for k in pl.get("flagged", [])]}
        _dump(res, out)
        for f in os.listdir(work):
            if f.startswith("side") and f.endswith(".ply"): os.remove(os.path.join(work, f))
        return {"outputs": {"placement": "placement.json"},
                "stats": {"base": res["base"], "order": res["order"], "score": {k: round(v, 3) for k, v in res["score"].items()},
                          "flagged": res["flagged"]}}

    def summary(self, p):
        return "base %s" % p.get("base", "auto")


# ------------------------------------------------------------------------------------------------------ fix
@register
class Fix(Method):
    node = "lineup.fix"; name = "manual"; label = "Flips and point matches"; version = 2; produces = "placement"; mem_gb = 3
    scripts = ("registration_metrics.py",); modules = ("methods/lineup.py", "tools/edits.py", "plyio.py"); libs = ("numpy", "scipy")

    def static_alias(self, step, graph):        # "flips" holds the flip and match ops, in order
        return None if step.params.get("flips") else "placement"

    def execute(self, ctx):
        ids = active_ids(ctx)
        job = {"placement": ctx.inputs["placement"].files["placement"], "flips": ctx.params["flips"],
               "meshes": {s: ctx.inputs["mesh@" + s].files["mesh"] for s in ids},
               "digests": {s: ctx.inputs["mesh@" + s].digest for s in ids}, "cache": os.path.join(ctx.project.stages, "cache", "samples")}
        r = ctx.job("tools.edits:fix_job", job, mem_gb=self.mem_gb, light=True)
        return {"outputs": {"placement": "placement.json"}, "warn": r.get("warn") or [],
                "stats": {"flipped": r.get("flipped", {}), "matched": r.get("matched", {}), "turned": r.get("turned", {}), "fits": r.get("fits", {})}}

    def summary(self, p):
        n = len(p.get("flips", []))
        return "%d flip%s or match%s" % (n, "" if n == 1 else "s", "" if n == 1 else "es")


# ------------------------------------------------------------------------------------------------------ refine
MED_RE = re.compile(r"median of pair medians ([\d.naninf]+) -> ([\d.naninf]+) mm")
MOVED_RE = re.compile(r"side (\d+) moved ([\d.]+) mm / ([\d.]+) deg")


class _Refine(Method):
    node = "lineup.refine"; version = 1; produces = "transforms"; mem_gb = 5; argv_params = True
    scripts = ("refine_sides.py",); modules = ("methods/lineup.py",); libs = ("open3d", "scipy")


@register
class RefinePoseGraph(_Refine):
    name = "pose_graph"; label = "Pose graph (refine_sides)"

    def static_alias(self, step, graph):
        return "placement" if step.params.get("active", 0) < 3 else None

    def execute(self, ctx):
        pl = _load(ctx.inputs["placement"].files["placement"]); order = pl["order"]
        ctx.legacy("refine", msg="adjusting all sides together")
        args = [script("refine_sides.py"), "--out-prefix", os.path.join(ctx.out_dir, "T_")]
        args += legacy_flags(ctx.params, PR.COMMON["lineup.refine"])
        for k, s in enumerate(order):
            ip = os.path.join(ctx.out_dir, "init_%d.json" % (k + 1)); _dump({"matrix": pl["T"][s]}, ip)
            args += ["--mesh", ctx.inputs["mesh@" + s].files["mesh"], "--init", ip]
        text = ctx.call(args, mem_gb=self.mem_gb)
        for ln in text.rstrip("\n").splitlines(): ctx.line(ln)
        T = {s: _load(os.path.join(ctx.out_dir, "T_%d.json" % (k + 1)))["matrix"] for k, s in enumerate(order)}
        st = {}
        try:
            m = MED_RE.search(text)
            if m: st["agreement"] = {"before": float(m.group(1)), "after": float(m.group(2))}
            mv = {}
            for mm in MOVED_RE.finditer(text):
                k = int(mm.group(1)) - 1
                if 0 <= k < len(order): mv[order[k]] = [float(mm.group(2)), float(mm.group(3))]
            if mv: st["moved"] = mv
        except Exception: pass
        _dump({"frame": "model", "base": pl["base"], "order": order, "T": T, "agreement": st.get("agreement"), "moved": st.get("moved")},
              os.path.join(ctx.out_dir, "transforms.json"))
        return {"outputs": {"transforms": "transforms.json"}, "stats": st}

    def summary(self, p):
        return "%d rounds, voxel %s mm" % (p["rounds"], PR.fmt(p["voxel"]))


@register
class RefineNone(_Refine):
    name = "none"; label = "None"

    def static_alias(self, step, graph):
        return "placement"


# ------------------------------------------------------------------------------------------------------ adjust
@register
class Adjust(Method):
    node = "lineup.adjust"; name = "manual"; label = "Nudges"; version = 1; produces = "lineup"; light = True
    modules = ("methods/lineup.py", "methods/pairs.py", "tools/edits.py", "plyio.py", "project.py")

    def execute(self, ctx):
        from ..tools.edits import nudge_matrix
        from ..plyio import vertex_mean
        src = ctx.inputs["transforms"]
        tj = _load(src.files.get("transforms") or src.files["placement"])
        base, order = tj["base"], tj["order"]
        T = {s: np.array(tj["T"][s], float) for s in order}
        for nd in ctx.params.get("nudges", []):
            s = nd["side"]
            if s not in T: continue
            c = T[s][:3, :3] @ vertex_mean(ctx.inputs["mesh@" + s].files["mesh"]) + T[s][:3, 3]
            T[s] = nudge_matrix(c, nd.get("t", [0, 0, 0]), nd.get("r", [0, 0, 0])) @ T[s]
        Tl = {s: T[s].tolist() for s in order}
        _dump({"frame": "model", "base": base, "order": order, "T": Tl}, os.path.join(ctx.out_dir, "lineup.json"))
        P = ctx.project; sides = []
        for s in order:
            fold = P.folder(s)
            sides.append({"id": s, "n": ctx.side_n(s), "folder": P.rel(fold), "frames": P.rel(os.path.join(fold, "cache")),
                          "calib": P.rel(os.path.join(fold, "param", "Pl.bin")),
                          "poses": P.rel(os.path.join(fold, "cache", "pointyoink_register_pose.pose")),
                          "size": P.frame_size(s), "n_frames": P.dph_stats(s)[0], "T": Tl[s],
                          "plane": plane_of(ctx.inputs["plane@" + s]), "mesh": P.rel(ctx.inputs["mesh@" + s].files["mesh"])})
        _dump({"frame": "model", "base": base, "sides": sides}, os.path.join(ctx.out_dir, "sides.json"))
        return {"outputs": {"lineup": "lineup.json", "sides": "sides.json"},
                "stats": {"base": base, "order": order, "nudged": len(ctx.params.get("nudges", []))}}

    def summary(self, p):
        return "%d nudge%s" % (len(p.get("nudges", [])), "" if len(p.get("nudges", [])) == 1 else "s")
