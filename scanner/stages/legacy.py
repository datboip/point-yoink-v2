# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adopting build_project.py's caches (build/) into the store, and writing the legacy build/ files other tools read.

Everything adopted is COPIED (build_project and strip_table write their outputs in place, which would rewrite store
files). Anything that does not match exactly is simply not adopted (it stays stale); nothing is ever adopted wrongly.
  sides   build/sides_cache.json["side<n>"] = "1|<id>|<ndph>|<mtime>|<pose md5>|<plane md5>|<voxel>" (n = position at
          that build): adopted when the id is still at position n, the dph count/mtime and pose md5 match, voxel equals
          side.fuse.voxel, the other side params are at their defaults and the files exist
  pairs   build/pairs_cache.json["<j>_to_<i>"]: base/moving stamps (<pair_all.VERSION>:size:mtime_ns) equal the current
          stamps of build/side<i>.ply / side<j>.ply (whatever VERSION made the fit: the stamps tie it to exactly the side
          meshes that were adopted, and a VERSION bump does not by itself make an old fit wrong), the beds digest, when
          the fit recorded one, matches the adopted platter planes (no bed poses), and both sides were adopted. A fit of
          an older VERSION is stored under its own token (methods.pairs.old_token): used as a fallback, refit by
          `run --force lineup.pairs`, removed by `adopt --refit-old-pairs`
  platter build_project's soft fail (the side used uncut, no side<n>_plane.json) is adopted flagged for retry (engine
          _transient): the next run tries the cut again instead of keeping the uncut side for good
  line-up placement.json -> lineup.place, T_refined_<k>.json -> lineup.refine, all_raw.ply -> fused (tsdf base), only
          when every side and pair was adopted, nothing is left out / flipped / nudged, and the mtimes are in order
Scratch copies of a project must be made with cp -a (mtimes kept), or nothing matches."""
import glob, hashlib, json, os, shutil

import numpy as np

from . import params as PR
from .keys import md5_file
from .keys import jload, jdump

LEGACY_SIDE_VERSION = 1


def _load(p):
    try: return jload(p)
    except Exception: return None


class Adopter:
    def __init__(self, project, accept_versions=None, refit_old=False):
        """refit_old: only take pair fits of the current pair_all VERSION (plus accept_versions); the default takes a
        fit of any version whose side stamps match."""
        self.accept = set(int(v) for v in accept_versions or ()); self.refit_old = refit_old
        self.p = project; self.b = project.build; self.notes = []; self.ok = False; self.why = None
        self.sides_cache = _load(os.path.join(self.b, "sides_cache.json"))
        self.pairs_cache = _load(os.path.join(self.b, "pairs_cache.json")) or {}
        if not os.path.isdir(self.b): self.why = "no build/ folder"; return
        if not self.sides_cache: self.why = "no build/sides_cache.json"; return
        self.ok = True
        self.adopted_sides = set(); self.pairs_ok = False; self.pairs_done = False
        self.sides = project.sides()

    def f(self, name): return os.path.join(self.b, name)

    # ---------------------------------------------------------------------------------------------- per node
    def adopt(self, engine, st, key, outs, g):
        try:
            fn = {"side.fuse": self._side_fuse, "side.platter": self._platter, "side.keep": self._keep,
                  "lineup.place": self._place, "lineup.refine": self._refine, "fused": self._fused}.get(st.kind)
            if fn is None: return None
            res = fn(st, outs, g)
        except Exception as e:
            self.notes.append("%s: %s" % (st.id, str(e)[:160])); return None
        if res is None: return None
        from .runner import Cancelled
        try:
            return self._store(engine, st, key, outs, res)
        except Cancelled: raise
        except Exception as e:                       # e.g. its geometry digest failed: not adopted, never half-stored
            self.notes.append("%s: not adopted (%s)" % (st.id, str(e)[:160])); return None

    def _store(self, engine, st, key, outs, res):
        tmp = engine.store.tmp_dir(st.store, key); outputs = {}
        for name, src in res.get("files", {}).items():
            fn = res.get("names", {}).get(name, os.path.basename(src))
            shutil.copy2(src, os.path.join(tmp, fn)); outputs[name] = fn
        for name, obj in res.get("json", {}).items():
            fn = res["names"][name]; jdump(obj, os.path.join(tmp, fn)); outputs[name] = fn
        meta = {"node": st.id, "method": st.method.name, "version": st.method.version, "params": st.params, "kind": st.produces,
                "inputs": {n: {"node": i, "digest": outs[i].digest} for n, i in st.inputs.items()}, "stats": res.get("stats", {}),
                "secs": 0, "peak_mb": 0, "adopted": True, "status": res.get("status", "ok"), "warn": res.get("warn", [])}
        if res.get("alias_of"):
            src = outs[st.inputs[res["alias_of"]]]
            meta["alias_of"] = res["alias_of"]; meta["digest"] = src.bdigest; meta["geo_digest"] = src.digest; outputs = {}
        if res.get("retry"):                # an earlier soft fail: kept for this build, tried again by the next run
            meta["transient"] = True; meta["retry"] = res["retry"]
        try:
            final = engine.store.publish(tmp, st.store, key, meta, outputs, geo_fn=engine._geo_job)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True); raise            # nothing half-stored (a cancelled geometry digest)
        engine.store.note_state(st.id, key, "adopted")
        return engine._out_meta(st, final, outs)

    def _entry(self, st):
        e = self.sides_cache.get("side%d" % st.n)
        if not e: return None
        parts = e.split("|")
        if len(parts) != 7: return None
        ver, sid, ndph, mt, pose, plane, voxel = parts
        return dict(ver=int(ver), id=sid, ndph=int(ndph), mtime=int(mt), pose=pose, plane=plane, voxel=float(voxel))

    def _defaults_except(self, node, params, allowed=()):
        for p in PR.COMMON.get(node, []):
            if p.name in allowed: continue
            if params.get(p.name) != p.default: return False
        return True

    def _side_fuse(self, st, outs, g):
        e = self._entry(st)
        if not e or e["ver"] != LEGACY_SIDE_VERSION or e["id"] != st.side: return None
        n, mt = self.p.dph_stats(st.side)
        if (n, mt) != (e["ndph"], e["mtime"]): self.notes.append("side %d: frames changed since the build" % st.n); return None
        if md5_file(os.path.join(self.p.folder(st.side), "cache", "pointyoink_register_pose.pose")) != e["pose"]:
            self.notes.append("side %d: poses changed since the build" % st.n); return None
        if abs(st.params["voxel"] - e["voxel"]) > 1e-9 or not self._defaults_except("side.fuse", st.params, ("voxel",)): return None
        raw = self.f("side%d_raw.ply" % st.n)
        if not os.path.exists(raw) or not os.path.exists(self.f("side%d.ply" % st.n)): return None
        return {"files": {"mesh": raw}, "names": {"mesh": "raw.ply"}}

    def _platter(self, st, outs, g):
        e = self._entry(st)
        if not e or st.method.name != "auto" or not self._defaults_except("side.platter", st.params): return None
        if outs[st.inputs["raw"]].meta is None or not outs[st.inputs["raw"]].meta.get("adopted"): return None
        if self.p.live_plane_fp(st.side) != e["plane"]: return None
        raw = self.f("side%d_raw.ply" % st.n); cut = self.f("side%d_cut.ply" % st.n); pl = self.f("side%d_plane.json" % st.n)
        rt = os.path.getmtime(raw)
        if os.path.exists(cut) and os.path.exists(pl) and os.path.getmtime(cut) >= rt and os.path.getmtime(pl) >= rt - 1:
            return {"files": {"mesh": cut, "plane": pl}, "names": {"mesh": "cut.ply", "plane": "plane.json"}}
        if not os.path.exists(pl):          # build_project's soft fail: the side was used uncut; the next run retries the cut
            return {"alias_of": "raw", "status": "warn", "warn": ["table cut failed in the earlier build; using it uncut, will retry next run"],
                    "retry": {"kind": "adopted", "sig": "adopted|build_project could not cut the platter", "rc": None, "count": 0}}
        return None

    def _keep(self, st, outs, g):
        e = self._entry(st)
        if not e or not self._defaults_except("side.keep", st.params): return None
        up = outs[st.inputs["cut"]]
        if up.meta is None or not up.meta.get("adopted"): return None
        part = self.f("side%d.ply" % st.n)
        if not os.path.exists(part): return None
        self.adopted_sides.add(st.side)
        return {"files": {"mesh": part}, "names": {"mesh": "side.ply"}}

    # ---------------------------------------------------------------------------------------------- pairs
    def before_light(self, engine, st, outs, g):
        """lineup.pairs: put the build's cached fits into the engine pair cache (keyed by content)."""
        if st.kind != "lineup.pairs" or self.pairs_done: return
        self.pairs_done = True; self._sync_sides(outs, g)
        from .methods.pairs import PairCache, pair_version, pair_token, plane_digest, old_token
        V = pair_version(); token = pair_token(); cache = PairCache(os.path.join(self.p.stages, "cache", "pairs.json"))
        ids = [s.id for s in g.active]; pos = {s.id: s.n for s in g.sides}; added = 0; missing = 0; old = 0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = pos[ids[a]], pos[ids[b]]
                e = self.pairs_cache.get("%d_to_%d" % (j, i))
                ok = e is not None and ids[a] in self.adopted_sides and ids[b] in self.adopted_sides
                ev = None
                if ok:
                    try: ev = int(str(e.get("base", "")).split(":")[0])
                    except ValueError: ev = None
                    ok = ev is not None and (not self.refit_old or ev in ({V} | self.accept)) \
                        and e.get("base") == self._stamp(i, ev) and e.get("moving") == self._stamp(j, ev)
                if ok and (ev == V or "beds" in e):
                    # the fit was checked against platter planes: they must be the adopted ones (no bed poses)
                    ok = e.get("beds") == self._beds_digest(ids[a], i) + self._beds_digest(ids[b], j)
                if ok and ev != V: old += 1
                ka, kb = g.final("side.keep@" + ids[a]), g.final("side.keep@" + ids[b])
                if not ok or ka not in outs or kb not in outs: missing += 1; continue
                oa, ob = outs["side.platter@" + ids[a]], outs["side.platter@" + ids[b]]
                # an older pair_all's fit keeps its own token: a fallback only (run --force lineup.pairs refits it)
                cache.put(token if ev == V else old_token(ev), st.params, outs[ka].digest, outs[kb].digest,
                          plane_digest(oa), plane_digest(ob),
                          dict(matrix=e["matrix"], fitness=e["fitness"], rmse=e["rmse"], support=e.get("support"), warns=e.get("warns", []),
                               adopted=True, fitted_by_version=ev, old_version=ev != V))
                added += 1
        cache.save(); self.pairs_ok = missing == 0
        self.notes.append("pairs: %d adopted%s, %d not" % (added, (" (%d fitted by an older pair_all, whose side files match; "
                                                                  "run --force lineup.pairs refits)" % old) if old else "", missing))

    def _sync_sides(self, outs, g):
        """Sides whose kept mesh is adopted (now or by an earlier adopt)."""
        for s in g.sides:
            o = outs.get(g.final("side.keep@" + s.id))
            if o is not None and o.meta and o.meta.get("adopted"): self.adopted_sides.add(s.id)

    def _stamp(self, n, V):
        p = self.f("side%d.ply" % n)
        if not os.path.exists(p): return None
        s = os.stat(p); return "%d:%d:%d" % (V, s.st_size, s.st_mtime_ns)

    def _beds_digest(self, sid, n):
        """pair_all.load_beds' digest of the side's beds.json entry, for a build without a learned bed."""
        pl = _load(self.f("side%d_plane.json" % n))
        e = {"folder": sid, "plane": None if pl is None else {"n": pl["n"], "d": pl["d"]}, "B_scan": None, "bed_id": None}
        return hashlib.md5(json.dumps(e, sort_keys=True).encode()).hexdigest()[:12]

    # ---------------------------------------------------------------------------------------------- line-up / fused
    def _clean_recipe(self, g):
        return len(g.active) == len(g.sides) and not g.by_id["lineup.fix"].params.get("flips") and not g.by_id["lineup.adjust"].params.get("nudges")

    def _place(self, st, outs, g):
        self._sync_sides(outs, g)
        pm = outs["lineup.pairs"].meta or {}
        if not (pm.get("adopted") and self._clean_recipe(g) and len(self.adopted_sides) == len(g.sides)): return None
        if not self._defaults_except("lineup.place", st.params): return None
        pp = self.f("placement.json"); pl = _load(pp)
        if not pl or len(pl["order"]) != len(g.sides): return None
        if any(os.path.getmtime(pp) < os.path.getmtime(self.f("side%d.ply" % s.n)) for s in g.sides): return None
        ids = [s.id for s in g.sides]
        res = {"base": ids[pl["order"][0]], "order": [ids[k] for k in pl["order"]], "T": {ids[int(k)]: v for k, v in pl["T"].items()},
               "score": {ids[int(k)]: v for k, v in pl["score"].items()}, "flagged": [ids[k] for k in pl.get("flagged", [])]}
        return {"json": {"placement": res}, "names": {"placement": "placement.json"},
                "stats": {"base": res["base"], "order": res["order"], "score": {k: round(v, 3) for k, v in res["score"].items()}}}

    def _refine(self, st, outs, g):
        up = outs[st.inputs["placement"]]
        if up.meta is None or not up.meta.get("adopted") or st.method.name != "pose_graph": return None
        if not self._defaults_except("lineup.refine", st.params): return None
        pl = jload(up.files["placement"]); order = pl["order"]; pt = os.path.getmtime(self.f("placement.json"))
        T = {}
        for k, s in enumerate(order):
            p = self.f("T_refined_%d.json" % (k + 1))
            if not os.path.exists(p) or os.path.getmtime(p) < pt: return None
            T[s] = jload(p)["matrix"]
        if os.path.exists(self.f("T_refined_%d.json" % (len(order) + 1))) and \
                os.path.getmtime(self.f("T_refined_%d.json" % (len(order) + 1))) >= pt: return None
        res = {"frame": "model", "base": pl["base"], "order": order, "T": T, "agreement": None, "moved": None}
        return {"json": {"transforms": res}, "names": {"transforms": "transforms.json"}}

    def _fused(self, st, outs, g):
        if st.method.name != "tsdf" or not self._clean_recipe(g): return None
        lu = outs[st.inputs["lineup"]]
        chain = ["lineup.place", "lineup.refine"] if len(g.active) >= 3 else ["lineup.place"]
        for c in chain:
            m = outs.get(c).meta if outs.get(c) else None
            if m is None or not m.get("adopted"): return None
        raw = self.f("all_raw.ply")
        if not os.path.exists(raw): return None
        vox = {self._entry(g.by_id["side.fuse@" + s.id])["voxel"] for s in g.sides}
        if len(vox) != 1 or abs(st.params["point_distance"] - vox.pop()) > 1e-9 or st.params["frames_per_side"] != 300: return None
        for p in PR.METHOD_PARAMS[("fused", "tsdf")]:
            if st.params[p.name] != p.default: return None
        ts = [self.f("T_refined_%d.json" % (k + 1)) for k in range(len(g.sides))] if len(g.sides) >= 3 else [self.f("placement.json")]
        if any(os.path.getmtime(raw) < os.path.getmtime(t) for t in ts if os.path.exists(t)): return None
        return {"files": {"mesh": raw}, "names": {"mesh": "mesh.ply"}}


# ---------------------------------------------------------------------------------------------- legacy export
def write_legacy_build(project, lineup_json, placement_json, sides_json, full=False, meshes=None, fused_raw=None):
    """build/placement.json (positions), build/T_refined_<k>.json and build/side<n>_plane.json, the three files
    bake/frames.py and other tools read. full: also build/side<n>.ply and build/all_raw.ply (copies)."""
    b = project.build
    project.guard_write(b)                  # build/ (or a file in it) must not resolve into a protected folder
    os.makedirs(b, exist_ok=True)
    pos = {s.id: s.n - 1 for s in project.sides()}
    lu = jload(lineup_json); pl = jload(placement_json); sj = jload(sides_json)
    names = ["placement.json"] + ["T_refined_%d.json" % (k + 1) for k in range(len(lu["order"]))] + \
        ["side%d_plane.json" % s.n for s in project.sides()]
    if full: names += ["side%d.ply" % (pos[sid] + 1) for sid in (meshes or {})] + (["all_raw.ply"] if fused_raw else [])
    for n in names: project.guard_write(os.path.join(b, n))

    def put(obj, name):
        p = os.path.join(b, name); tmp = p + ".tmp"; jdump(obj, tmp); os.replace(tmp, p)
    put({"order": [pos[s] for s in pl["order"]], "T": {str(pos[s]): pl["T"][s] for s in pl["order"]},
         "score": {str(pos[s]): v for s, v in pl.get("score", {}).items()}, "flagged": [pos[s] for s in pl.get("flagged", [])]},
        "placement.json")
    for f in glob.glob(os.path.join(b, "T_refined_*.json")):
        try:
            if int(os.path.basename(f)[10:-5]) > len(lu["order"]): os.remove(f)
        except ValueError: pass
    for k, s in enumerate(lu["order"]): put({"matrix": lu["T"][s]}, "T_refined_%d.json" % (k + 1))
    planes = {e["id"]: e.get("plane") for e in sj["sides"]}
    for s in project.sides():
        p = os.path.join(b, "side%d_plane.json" % s.n)
        if planes.get(s.id): put(planes[s.id], "side%d_plane.json" % s.n)
        elif os.path.exists(p): os.remove(p)
    if full:
        for sid, src in (meshes or {}).items():
            dst = os.path.join(b, "side%d.ply" % (pos[sid] + 1)); shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        if fused_raw:
            dst = os.path.join(b, "all_raw.ply"); shutil.copyfile(fused_raw, dst + ".tmp"); os.replace(dst + ".tmp", dst)
