"""Staged build engine (scanner/stages): keys, recipe history, graph, early cutoff, op prefixes, gc, lock and deny guard.
Fake methods echo their params into small files: no scans, no subprocesses, no GPU (T12 starts one CLI process that
exits at the lock)."""
from unittest.mock import patch
import io, json, os, shutil, struct, subprocess, sys, tempfile, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner"))

from stages import params as PR                      # noqa: E402
from stages import graph as G                        # noqa: E402
from stages.engine import Engine, Locked             # noqa: E402
from stages.keys import H, coerce                    # noqa: E402
from stages.methods import REGISTRY, Method, Tool, _load   # noqa: E402
from stages.project import Project, DeniedError      # noqa: E402

CALLS = []
IGNORE = {"lineup.place": ("tol",)}     # T2: this param changes the key but not the output bytes


def wb(path, data):
    with open(path, "wb" if isinstance(data, bytes) else "w") as f: f.write(data)


def make_project(root, n=4, frames=3):
    ids = []
    for k in range(n):
        sid = "09272026%06d" % (2100 + 300 * k); ids.append(sid)
        d = os.path.join(root, "data", sid); os.makedirs(os.path.join(d, "cache")); os.makedirs(os.path.join(d, "param"))
        wb(os.path.join(d, "cache", "pointyoink_track.json"), "{}")
        pose = struct.pack("<i", frames)
        for f in range(frames):
            wb(os.path.join(d, "cache", "frame_0_%d.dph" % f), bytes([k, f]) * 4)
            wb(os.path.join(d, "cache", "frame_0_%d.inf" % f), b"\0" * 16)
            pose += struct.pack("<ii", 0, f) + struct.pack("<16d", *[1.0 if i % 5 == 0 else 0.0 for i in range(16)])
        wb(os.path.join(d, "cache", "pointyoink_register_pose.pose"), pose)
        wb(os.path.join(d, "param", "Pl.bin"), struct.pack("<10f", *range(10)))
    return ids


def ident(): return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


class FakeMixin:
    """Echo params and input digests into every declared output; realistic JSON for the line-up files."""
    OUTS = {"side.fuse": {"mesh": "raw.ply"}, "side.platter": {"mesh": "cut.ply", "plane": "plane.json"}, "side.keep": {"mesh": "side.ply"},
            "lineup.pairs": {"pairs": "pairs.json"}, "lineup.place": {"placement": "placement.json"},
            "lineup.refine": {"transforms": "transforms.json"}, "fused": {"mesh": "mesh.ply"},
            "meshed": {"mesh": "mesh.ply"}, "report": {"preview": "model.png", "needs_json": "needs.json"}}

    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        st = ctx.step; ign = IGNORE.get(st.kind, ())
        body = {"node": st.id, "params": {k: v for k, v in ctx.params.items() if k not in ign},
                "inputs": {n: o.digest for n, o in sorted(ctx.inputs.items())}}
        ids = [k[5:] for k in st.inputs if k.startswith("mesh@")]
        outs = {}
        for name, fn in self.OUTS[st.kind].items():
            obj = body
            if name == "plane": obj = {"n": [0, 0, 1], "d": -10.0 - (st.n or 0), "margin": 1.0, "keep_above": True}
            if name == "placement": obj = {"base": ids[0], "order": ids, "T": {s: ident() for s in ids}, "score": {s: 1.0 for s in ids}, "flagged": [], "b": body}
            if name == "transforms":
                with open(ctx.inputs["placement"].files["placement"]) as f: pl = json.load(f)
                obj = {"frame": "model", "base": pl["base"], "order": pl["order"], "T": pl["T"], "b": body}
            with open(os.path.join(ctx.out_dir, fn), "w") as f: json.dump(obj, f, sort_keys=True)
            outs[name] = fn
        if st.kind == "side.fuse": ctx.legacy("frames", step="side", side=st.n, sides=ctx.n_sides, done=3, total=3)
        if st.kind == "side.platter": ctx.legacy("table", side=st.n, msg="platter: fake")
        if st.kind == "lineup.pairs":
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)): ctx.legacy("aligned", sides="%d-%d" % (ctx.side_n(ids[i]), ctx.side_n(ids[j])), fit=0.8, rmse_mm=0.5, bed="-")
            ctx.line("pairs: 0 fitted, %d from the cache, 0 s" % (len(ids) * (len(ids) - 1) // 2))
        if st.kind == "lineup.place": ctx.legacy("base", side=ctx.side_n(ids[0]), msg="side %d fits best with the others" % ctx.side_n(ids[0]))
        if st.kind == "lineup.refine": ctx.legacy("refine", msg="adjusting all sides together"); ctx.line("median of pair medians 0.300 -> 0.200 mm, 1 s")
        if st.kind == "fused":
            ctx.legacy("fuse", msg="fusing all sides into one model")
            for k in (5, 10): ctx.legacy("frames", step="all", done=k, total=10)
        return {"outputs": outs, "stats": {"fake": True}}


class FakeTool(Tool):
    kinds = ("mesh",); version = 1

    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        body = {"tool": self.name, "params": ctx.params, "parent": ctx.inputs["parent"].digest}
        with open(os.path.join(ctx.out_dir, "mesh.ply"), "w") as f: json.dump(body, f, sort_keys=True)
        if self.name == "isolation": ctx.line("kept 1 of 1 pieces, 10 triangles (100.0%)")
        return {"outputs": {"mesh": "mesh.ply"}, "stats": {}}


SAVED = {}


def install_fakes():
    _load()
    if SAVED: return
    SAVED.update({k: dict(v) for k, v in REGISTRY.items()})
    for (node, name), cls in list(REGISTRY["methods"].items()):
        if node in FakeMixin.OUTS and not (node == "meshed" and name in ("as_fused", "auto")):
            REGISTRY["methods"][(node, name)] = type("Fake_" + cls.__name__, (FakeMixin, cls), {})
    for tname in ("isolation", "smooth", "simplify", "slivers", "fill_holes"):
        REGISTRY["tools"][(tname, "mesh")] = type("FakeTool_" + tname, (FakeTool,), {"name": tname, "label": tname})


def remove_fakes():
    if SAVED:
        for k, v in SAVED.items(): REGISTRY[k].clear(); REGISTRY[k].update(v)
        SAVED.clear()


class Base(unittest.TestCase):
    def setUp(self):
        install_fakes()
        self.tmp = tempfile.mkdtemp(prefix="stages_test_"); self.ids = make_project(self.tmp)
        self.p = Project.open(self.tmp); self.out = io.StringIO()
        self.e = Engine(self.p, out=self.out); self.e._post = lambda *a, **k: None
        CALLS.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        remove_fakes()

    def run_(self, **kw):
        CALLS.clear(); r = self.e.run(quiet=True, **kw); self.assertTrue(r.ok, r.error); return list(CALLS)

    def keys(self, recipe=None, wp=None):
        g = self.e.graph(recipe, wp); _o, info, _ = self.e._walk(g, "plan", None)
        return {k: v.get("key") for k, v in info.items()}


class T1Keys(Base):
    def test_keys(self):
        self.assertEqual(H({"a": 0.1 + 0.2}), H({"a": 0.3}))
        self.assertEqual(coerce(3), 3); self.assertIsInstance(coerce(3), int); self.assertEqual(coerce(2.0), 2.0)
        self.run_()
        k0 = self.keys()
        self.e.set({"fused.point_distance": 0.4})             # equal to the default: recipe unchanged, same key
        self.assertEqual(self.e.recipe()["params"], {}); self.assertEqual(self.keys(), k0)
        self.e.set({"fused.point_distance": 0.1 + 0.2})
        self.assertEqual(self.e.recipe()["params"]["fused"]["point_distance"], 0.3)
        g = self.e.graph(); _o, info, _ = self.e._walk(g, "plan", None)
        self.assertEqual(info["fused"]["status"], "stale")
        for sid, v in info.items():                          # nothing above fused changed
            if sid.startswith(("side.", "lineup.")): self.assertEqual(v.get("key"), k0[sid], sid)
        self.e.set({"fused.point_distance": 0.3})             # same value again: no history entry
        self.assertEqual(len(self.e.history()[0]), 2)


class PlateChoice(Base):
    def test_plate_is_a_platter_method_the_engine_accepts(self):
        self.e.platter(self.ids[0], "plate")
        self.assertEqual(self.e.recipe()["sides"]["platter"][self.ids[0]], {"method": "plate"})
        g = self.e.graph(); self.assertEqual(g.by_id["side.platter@" + self.ids[0]].method.name, "plate")
        self.e.platter(self.ids[0], "auto")
        self.assertNotIn(self.ids[0], self.e.recipe()["sides"].get("platter", {}))


class LeftOutSurvivesPresets(Base):
    def test_a_preset_keeps_the_sides_left_out(self):
        self.e.leave_out(self.ids[1])
        for name in ("revo", "today"):
            if name in PR.PRESETS:
                self.e.preset(name)
                self.assertIn(self.ids[1], self.e.recipe()["sides"]["leave_out"], name)


class PresetsUsePlate(Base):
    def test_yolo_presets_use_the_plate_cut_and_keep_hand_cuts(self):
        import json as _j
        pj = os.path.join(self.tmp, "plane.json"); _j.dump({"n": [0, 0, 1], "d": 1.0, "margin": 1.0, "keep_above": True}, open(pj, "w"))
        self.e.platter(self.ids[0], pj)                                        # a by-hand cut on side 1
        for name in ("today", "revo", "revo-device", "plus", "print"):
            self.e.preset(name)
            r = self.e.recipe()
            self.assertEqual(r["params"]["side.platter"]["method"], "plate", name)
            self.assertEqual(r["sides"]["platter"][self.ids[0]]["method"], "manual", name)   # the hand cut stays


class PointPresetsNeedTwoSides(Base):
    def test_point_presets_keep_only_points_two_sides_agree_on(self):
        for name in ("revo", "revo-device"):
            self.e.preset(name)
            self.assertEqual(self.e.recipe()["params"]["fused"]["points"]["min_sides"], 2, name)


class HintCacheKey(Base):
    def test_how_scanned_hint_invalidates_placement_but_not_pair_fits(self):
        self.run_()
        before = self.keys()
        self.e.set({"lineup.place.hints": "%s:flipped" % self.ids[1]})
        after = self.keys()
        self.assertEqual(after["lineup.pairs"], before["lineup.pairs"])
        self.assertNotEqual(after["lineup.place"], before["lineup.place"])
        self.assertNotEqual(after["lineup.adjust"], before["lineup.adjust"])


class T2EarlyCutoff(Base):
    def test_cutoff(self):
        self.run_()
        self.e.set({"lineup.place.tol": 2.0})
        ran = self.run_()
        self.assertEqual(ran, ["lineup.place"])              # new key, identical bytes: everything below is a cache hit


class T3History(Base):
    def test_history(self):
        h = self.e.hist
        self.e.set({"fused.point_distance": 0.5}); after_a = open(h.rp).read()
        self.e.set({"meshed.method": "poisson"})
        self.assertEqual(self.e.undo(), "Meshed: meshed.method auto -> poisson")
        self.assertEqual(open(h.rp).read(), after_a)
        self.e.redo(); self.assertEqual(self.e.recipe()["params"]["meshed"]["method"], "poisson")
        self.e.undo(); self.e.set({"report.min_mm": 5})
        ents, cur = self.e.history(); self.assertEqual(cur, len(ents) - 1); self.assertIsNone(self.e.redo())
        self.assertNotIn("meshed", self.e.recipe()["params"])
        for k in range(205): self.e.set({"report.min_mm": 10 + k % 50 + (k // 50) * 0.5})
        ents, cur = self.e.history(); self.assertEqual(len(ents), 200); self.assertEqual(cur, 199)
        self.assertIsNotNone(self.e.undo()); self.assertIsNotNone(self.e.redo())
        self.assertEqual(json.load(open(h.rp)), self.e.history()[0][199]["recipe"])


class T4OpPrefixes(Base):
    def test_prefixes(self):
        self.run_()
        initial = self.e.graph(); initial_out, _, _ = self.e._walk(initial, "plan", None)
        first, inactive = initial.lists["fused.tsdf"][:2]
        self.assertEqual(initial.by_id[inactive].alias, "parent")
        self.assertEqual(initial_out[inactive].digest, initial_out[first].digest)
        for t in ("smooth", "simplify", "slivers"): self.e.add_op("fused", {"tool": t})
        ran = self.run_(to="fused")
        self.assertEqual(len(ran), 3); self.assertTrue(all(r.startswith("fused.tsdf#op-") for r in ran), ran)
        _lk, ops, _parked = self.e.ops_of("fused"); self.assertEqual([o["tool"] for _l, _j, o in ops],
                                                                      ["isolation", "match_cleaned_sides", "smooth", "simplify", "slivers"])
        self.assertFalse(ops[1][2]["on"])
        self.e.drop_op("fused", 4)
        self.assertEqual(self.run_(to="fused"), [])           # the shorter prefix is already stored
        self.e.undo()
        self.assertEqual(self.run_(to="fused"), [])
        self.e.toggle_op("fused", 3)                          # simplify off: an alias, slivers re-runs on smooth's output
        g = self.e.graph(); L = g.lists["fused.tsdf"]
        self.assertEqual(g.by_id[L[3]].alias, "parent")
        ran = self.run_(to="fused")
        self.assertEqual(ran, [L[4]])
        _o, info, _ = self.e._walk(g, "plan", None)
        self.assertEqual(info[L[3]]["status"], "off")
        self.assertEqual(_o[L[3]].digest, _o[L[2]].digest)
        self.e.undo(); self.e.undo()
        self.assertEqual(self.run_(to="fused"), [])


class TargetsForce(Base):
    def test_targets_and_force(self):
        self.run_()
        self.assertEqual(self.run_(), [])
        ran = self.run_(force=["lineup.place"])
        self.assertEqual(ran, ["lineup.place"])               # same bytes again: early cutoff below it
        ran = self.run_(from_="fused")
        self.assertEqual(ran[0], "fused")
        self.assertTrue(all(r.startswith(("fused", "report")) for r in ran), ran)


class T9Gc(Base):
    def test_gc(self):
        self.run_()
        for v in (0.5, 0.6, 0.7, 0.8, 0.9):
            self.e.set({"fused.point_distance": v}); self.run_()
        cur = {k for k, v in self.keys().items() if v}
        before = self.e.store.artifacts()
        vers = self.e.versions("fused"); self.assertGreaterEqual(len(vers), 6)
        oldest = sorted(vers, key=lambda v: v["last_used"])[0]
        self.e.pin("fused", oldest["key16"])
        for _s, _k, d, _m in before: os.utime(os.path.join(d, "meta.json"), (time.time() - 1000, time.time() - 1000))
        # Model large stored artifacts without allocating gigabytes in the test.
        with patch.object(self.e.store, "dir_bytes", return_value=4 * 1024 ** 3):
            r = self.e.gc(max_gb=0.0, keep=1)
        self.assertGreaterEqual(r["budget_gb"], r["protected_gb"] + 2)
        after = {d for _s, _k, d, _m in self.e.store.artifacts()}
        self.assertIn(oldest["dir"], after)                   # pinned
        g = self.e.graph(); _o, info, _ = self.e._walk(g, "plan", None)
        for k, v in info.items():
            if v.get("key"): self.assertIn(v["status"], ("fresh", "warn"), k)   # current plan kept
        # all 5 fused versions are within 10 history steps: nothing of them may go
        self.assertEqual(len(self.e.versions("fused")), len(vers))
        # far outside the history window and unpinned: gone
        for k in range(12): self.e.set({"report.min_mm": 20 + k})
        # Model large stored artifacts without allocating gigabytes in the test.
        with patch.object(self.e.store, "dir_bytes", return_value=4 * 1024 ** 3):
            r = self.e.gc(max_gb=0.0, keep=1)
        self.assertGreaterEqual(r["budget_gb"], r["protected_gb"] + 2)
        left = self.e.versions("fused")
        self.assertLess(len(left), len(vers)); self.assertIn(oldest["key16"], [v["key16"] for v in left])


    def test_viewer_exports_do_not_raise_protected_budget(self):
        self.run_()
        original = self.e.store.dir_bytes
        with patch.object(self.e.store, "dir_bytes", side_effect=lambda d: 2 * 1024 ** 3 + original(d)):
            before = self.e.gc(keep=1)
            exports = []
            for _store, _key, folder, meta in self.e.store.artifacts():
                if meta is None: continue
                path = os.path.join(folder, "view_mesh.ply")
                wb(path, b"x" * 1024 ** 2); exports.append(path)
            after = self.e.gc(keep=1)
        self.assertTrue(exports)
        self.assertTrue(all(not os.path.exists(p) for p in exports))
        self.assertEqual(before["protected_gb"], after["protected_gb"])
        self.assertEqual(before["budget_gb"], after["budget_gb"])


class T11Mappings(unittest.TestCase):
    def test_numbers(self):
        sch, cell = PR.overlap_schedule({"distance": 0.21, "schedule": ""})
        self.assertEqual(sch, (1.0, 0.7, 0.5)); self.assertEqual(cell, 0.25)
        # Q6 on a 180 mm part = depth 10 (the bake), Q7 = 11; the grid is the one Poisson really uses: extent x 1.1 / 2^depth
        self.assertEqual(PR.poisson_depth(180.0, 6), (10, round(180 * 1.1 / 1024, 4))); self.assertEqual(PR.poisson_depth(180.0, 7)[0], 11)
        self.assertEqual(PR.mesh_smooth_lambda(30), 0.5)
        self.assertEqual(PR.parse_path("side[0927].keep.min_share")["side"], "0927")
        self.assertEqual(PR.parse_path("lineup.refine.rounds")["node"], "lineup.refine")
        r = {}; PR.set_path(r, "meshed.poisson.quality", "revo"); self.assertEqual(r["params"]["meshed"]["poisson"]["quality"], 7)


class T12LockDeny(Base):
    def test_lock(self):
        self.e._lock()
        try:
            env = dict(os.environ); env.pop("POINTYOINK_STAGES_DENY", None)
            p = subprocess.run([sys.executable, os.path.join(ROOT, "scanner", "stages", "cli.py"), self.tmp, "run"],
                               capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
            self.assertIn("STAGE error", p.stdout); self.assertIn("a build is running (pid %d)" % os.getpid(), p.stdout)
            with self.assertRaises(Locked): Engine(self.p).run()
        finally:
            self.e._unlock()

    def test_deny(self):
        root = tempfile.mkdtemp(prefix="stages_deny_")
        try:
            make_project(os.path.join(root, "proj"), n=1)
            old = os.environ.get("POINTYOINK_STAGES_DENY"); os.environ["POINTYOINK_STAGES_DENY"] = "/nonexistent:" + root
            try:
                with self.assertRaises(DeniedError): Project.open(os.path.join(root, "proj"))
            finally:
                if old is None: os.environ.pop("POINTYOINK_STAGES_DENY", None)
                else: os.environ["POINTYOINK_STAGES_DENY"] = old
            self.assertFalse(os.path.exists(os.path.join(root, "proj", "stages")))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class GraphShape(Base):
    def test_leave_out(self):
        self.run_()
        self.e.leave_out(self.ids[2])
        ran = self.run_()
        self.assertFalse(any(r.startswith("side.") for r in ran))
        self.assertIn("lineup.pairs", ran)
        self.e.include(self.ids[2])
        self.assertEqual(self.run_(), [])
        with self.assertRaises(ValueError):
            for s in self.ids: self.e.leave_out(s)


if __name__ == "__main__":
    unittest.main()
