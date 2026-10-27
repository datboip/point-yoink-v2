"""Round-3 fixes of the staged build engine: one test class per item (R1..R9), each written to fail on the engine as it
was before round 3. Fake methods as in test_stages_core (no scans, no GPU); where a real method's own code is under test
its child processes are simulated, or the engine's jobs (stages/job.py) run in this process (test_stages_fixes.inproc).

R1 soft fail stored for retry   R2 launch failures are transient   R3 cancel scope   R4 old pair fits refit on request
R5 code and tables in the keys  R6 stage / side / param tokens checked   R7 export deny   R8 points delete area
R9 old fused deletes survive presets; an adopted platter soft fail is retried"""
import contextlib, dataclasses, io, json, os, shutil, sys, tempfile, threading, time, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_stages_core import Base, CALLS, SAVED, make_project, install_fakes, remove_fakes, wb, ident   # noqa: E402
from test_stages_fixes import inproc, use, status_of, PlyFusedMixin, plates_mesh                       # noqa: E402
from test_stages_legacy import fake_build                                                              # noqa: E402
from stages import params as PR                                                                       # noqa: E402
from stages import plyio                                                                              # noqa: E402
from stages import runner as RN                                                                       # noqa: E402
from stages import engine as EN                                                                       # noqa: E402
from stages import keys as KY                                                                         # noqa: E402
from stages import cli                                                                                # noqa: E402
from stages.engine import Engine                                                                      # noqa: E402
from stages.project import Project                                                                    # noqa: E402
from stages.runner import Runner, StepError, Cancelled                                                # noqa: E402
from stages.methods import REGISTRY, get_method                                                       # noqa: E402

PLATTER_OK = "platter: 100 support points, 1.0 deg off the spin axis; kept 10 triangles in 1 pieces, part height 50.0 mm\n"
EMPTY_CUT = "Traceback (most recent call last):\n  File \"strip_table.py\", line 90\nValueError: attempt to get argmax of an empty sequence\n"


def info_of(e):
    e._begin(); g = e.graph(); _o, info, _ = e._walk(g, "plan", None)
    return info


def cut_ok(cmd):
    wb(cmd[3], "cut " + cmd[2]); wb(cmd[cmd.index("--plane-out") + 1], json.dumps({"n": [0, 0, 1], "d": -10.0, "margin": 1.0, "keep_above": True}))
    return PLATTER_OK


def cli_run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err): rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------------------------------------- R1
class R1SoftFailStoredForRetry(Base):
    """strip_table crashing the same way every run: the run finishes, the planner shows everything below the platter
    built, the next run re-runs ONLY the platter (same output: cut off), and after 3 identical failures it is kept."""

    def setUp(self):
        super().setUp(); self.calls = []; self.err = EMPTY_CUT
        use("side.platter", "auto"); inproc(self.e, self.fake_cmd)

    def fake_cmd(self, cmd):
        name = os.path.basename(cmd[1]); self.calls.append((name, cmd[2]))
        if name == "strip_table.py":
            if cmd[2].endswith(self.ids[0]): raise StepError("strip_table.py failed (exit 1)", self.err, 1)
            return cut_ok(cmd)
        raise AssertionError(cmd)

    def test_retry_then_give_up(self):
        p1 = "side.platter@" + self.ids[0]
        self.run_()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model.ply")))
        info = info_of(self.e)
        self.assertEqual(info[p1]["status"], "warn"); self.assertTrue(info[p1].get("retry"))
        for i, v in info.items():                     # everything below the platter is built, nothing is waiting
            self.assertNotIn(v["status"], ("stale", "waiting"), (i, v))
        groups, _g, est = self.e.status()
        node = {n.id: n for gr in groups for n in gr.nodes}[p1]
        self.assertTrue(node.retry); self.assertIn("will retry next run", " ".join(node.warn))
        self.assertLess(est, 10)                      # only the 4 s platter would run
        self.assertIsNotNone(self.e.output("meshed"))
        rc, out, _err = cli_run([self.tmp, "status"])
        self.assertEqual(rc, 0); self.assertIn("platter warn (will retry)", out)
        # run 2: only the platter re-runs (strip_table once, for side 1), same alias out: cut off below it
        self.calls.clear(); ran = self.run_()
        self.assertEqual(ran, []); self.assertEqual(self.calls, [("strip_table.py", self.e.p.folder(self.ids[0]))])
        self.assertEqual(info_of(self.e)[p1]["meta"]["retry"]["count"], 2)
        # a different failure is not "the same": the count starts again
        self.err = EMPTY_CUT.replace("argmax", "argmin"); self.run_()
        self.assertEqual(info_of(self.e)[p1]["meta"]["retry"]["count"], 1)
        self.err = EMPTY_CUT; self.run_(); self.run_(); self.run_()             # 1, 2, 3 identical in a row
        info = info_of(self.e)
        self.assertEqual(info[p1]["status"], "warn"); self.assertFalse(info[p1].get("retry"))
        self.assertIn("not retried any more", " ".join(info[p1]["meta"]["warn"]))
        self.calls.clear(); self.run_(); self.assertEqual(self.calls, [])       # kept: no more retries
        self.calls.clear(); self.run_(force=[p1]); self.assertEqual(len(self.calls), 1)   # --force tries again
        self.assertTrue(info_of(self.e)[p1].get("retry"))

    def test_fixed_script_then_fresh(self):
        self.run_(); self.err = None
        self.fake_cmd = lambda cmd: (self.calls.append((os.path.basename(cmd[1]), cmd[2])), cut_ok(cmd))[1]
        inproc(self.e, self.fake_cmd)
        self.calls.clear(); ran = self.run_()
        self.assertEqual(len(self.calls), 1); self.assertIn("side.keep@" + self.ids[0], ran)   # the cut changed: below re-runs
        info = info_of(self.e)
        self.assertEqual(info["side.platter@" + self.ids[0]]["status"], "fresh")


# ---------------------------------------------------------------------------------------------------------- R2
class R2LaunchIsTransient(Base):
    def test_classify(self):
        c = RN.classify
        self.assertEqual(c(1, "Failed to connect to bus: No medium found"), "launch")
        self.assertEqual(c(1, "Failed to start transient scope unit: Unit run-u12.scope already exists."), "launch")
        self.assertEqual(c(1, "", started=False), "launch")                          # the child never started
        self.assertEqual(c(127, "/bin/sh: 1: exec: /nope/python: not found\n", started=True, pre=""), "launch")
        self.assertEqual(c(1, "flock: 9: Bad file descriptor", started=False), "launch")
        # the script's own output after it started is not the wrapper's
        self.assertEqual(c(1, "Failed to connect to bus", started=True, pre=""), "unknown")
        self.assertEqual(c(3, "something odd"), "unknown")
        self.assertFalse(StepError("x", "no platter found", 2).deliberate)                # unrecognised: not deliberate
        self.assertTrue(StepError("x", 'STAGE no_result {"why": "no platter"}\n', 2).deliberate)
        self.assertFalse(StepError("x", "Failed to connect to bus: No medium found", 1).deliberate)

    def test_real_runner(self):
        rn = Runner(wrap=None)
        rn.argv = lambda cmd, mem_gb, gpu, light=False: ["/bin/sh", "-c", "echo 'Failed to connect to bus: No medium found' >&2; exit 1"]
        with self.assertRaises(StepError) as cm: rn.run([sys.executable, "-c", "print(1)"])
        self.assertEqual(cm.exception.kind, "launch")
        rn.argv = lambda cmd, mem_gb, gpu, light=False: list(cmd)
        with self.assertRaises(StepError) as cm: rn.run(["/nonexistent/python3", "-c", "print(1)"])
        self.assertEqual(cm.exception.kind, "launch")                                # the shell could not exec it
        with self.assertRaises(StepError) as cm: rn.run([sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertEqual(cm.exception.kind, "unknown"); self.assertFalse(cm.exception.deliberate)
        with self.assertRaises(StepError) as cm:
            rn.run([sys.executable, "-c", "import sys; print('STAGE no_result {}'); sys.exit(2)"])
        self.assertEqual(cm.exception.kind, "exit")
        with self.assertRaises(StepError) as cm:
            rn.run([sys.executable, "-c", "import sys; print('Failed to connect to bus'); sys.exit(1)"])
        self.assertEqual(cm.exception.kind, "unknown")                               # printed by the script itself

    def test_bus_failure_retried(self):
        use("side.platter", "auto"); calls = []; mode = {"bus": True}

        def fake(cmd):
            calls.append(os.path.basename(cmd[1]))
            if mode["bus"]: raise StepError("strip_table.py failed (exit 1)", "Failed to connect to bus: No medium found\n", 1)
            return cut_ok(cmd)
        inproc(self.e, fake)
        for _ in range(4): self.run_()                                  # never kept, however often the bus is gone
        st = info_of(self.e)
        for s in self.ids:
            self.assertEqual(st["side.platter@" + s]["meta"]["retry"]["kind"], "launch"); self.assertTrue(st["side.platter@" + s].get("retry"))
        mode["bus"] = False; calls.clear(); self.run_()
        self.assertEqual(calls.count("strip_table.py"), 4)
        for s in self.ids: self.assertEqual(status_of(self.e)["side.platter@" + s], "fresh")


# ---------------------------------------------------------------------------------------------------------- R3
class R3CancelScope(Base):
    def test_cancel_does_not_stick_to_other_calls(self):
        inproc(self.e); self.run_()
        self.e.cancel()                                                 # the UI's Cancel after the run ended
        ply = os.path.join(self.tmp, "t.ply"); V, F = plates_mesh(); plyio.write_mesh(ply, V, F)
        self.assertTrue(self.e.job("plyio:geo_digest_job", {"in": ply}, mem_gb=1).get("geo"))
        self.assertTrue(self.e._geo_job(ply))
        self.assertTrue(self.e.adopt_legacy() is not None)
        self.assertTrue(self.e.run(quiet=True).ok)

    def test_cancel_stops_what_runs_now(self):
        rn = self.e.runner; rn.argv = lambda cmd, mem_gb, gpu, light=False: list(cmd)
        box = {}

        def go():
            try:
                with rn.scope(): rn.run([sys.executable, "-c", "import time; time.sleep(20)"]); box["r"] = "ran"
            except Cancelled: box["r"] = "cancelled"
        t = threading.Thread(target=go, daemon=True); t.start(); time.sleep(0.8)
        t0 = time.time(); self.e.cancel(); t.join(10)
        self.assertEqual(box.get("r"), "cancelled"); self.assertLess(time.time() - t0, 8)
        with rn.scope(): self.assertIn("1", rn.run([sys.executable, "-c", "print(1)"])[0])      # a new scope runs

    def test_cancelled_geo_digest_stores_nothing(self):
        use("fused", "tsdf", PlyFusedMixin); inproc(self.e); self.e.clear_ops("fused")
        old = KY.DigestMemo.GEO_IN_PROCESS; KY.DigestMemo.GEO_IN_PROCESS = 10          # the plates go to the capped job
        real_job = self.e.job

        def job(fn, payload, **kw):
            if fn == "plyio:geo_digest_job": raise Cancelled()
            return real_job(fn, payload, **kw)
        self.e.job = job
        try:
            r = self.e.run(quiet=True)
        finally:
            KY.DigestMemo.GEO_IN_PROCESS = old; self.e.job = real_job
        self.assertFalse(r.ok); self.assertEqual(r.failed, "fused")
        fd = os.path.join(self.tmp, "stages", "nodes", "fused")
        self.assertEqual(os.listdir(fd) if os.path.isdir(fd) else [], [])          # nothing stored without its geo digest
        self.assertEqual(info_of(self.e)["fused"]["status"], "stale")
        self.run_()
        meta = info_of(self.e)["fused"]["meta"]; self.assertTrue(meta.get("geo_digest"))


# ---------------------------------------------------------------------------------------------------------- R4
class R4OldPairFitsRefit(unittest.TestCase):
    """adopt takes an older pair_all's fits (B5); `run --force lineup.pairs` and `adopt --refit-old-pairs` refit them."""

    def setUp(self):
        install_fakes(); REGISTRY["methods"][("lineup.pairs", "fpfh_icp")] = SAVED["methods"][("lineup.pairs", "fpfh_icp")]
        self.tmp = tempfile.mkdtemp(prefix="stages_r4_"); self.ids = make_project(self.tmp); fake_build(self.tmp, self.ids)
        pp = os.path.join(self.tmp, "build", "pairs_cache.json"); pc = json.load(open(pp))
        from stages.methods.pairs import pair_version
        V = pair_version()
        for e in pc.values():
            for k in ("base", "moving"): e[k] = "%d:%s" % (V - 1, e[k].split(":", 1)[1])
            e.pop("beds", None)
        json.dump(pc, open(pp, "w"))
        self.jobs = []; self.real_job = EN.Ctx.job; jobs = self.jobs; real = self.real_job

        def fit(ctx, fn, payload, **kw):                 # the pair fit job, simulated: every missing pair fitted
            if fn != "methods.pairs:pairs_job": return real(ctx, fn, payload, **kw)
            if ctx.light_only: raise EN.NotLight()          # as ctx.call does: adopt never fits
            jobs.append(fn)
            from stages.methods.pairs import PairCache
            c = PairCache(payload["cache"]); S = payload["sides"]
            for i, j, _k in payload["todo"]:
                c.put(payload["version"], payload["params"], S[i]["digest"], S[j]["digest"], S[i]["plane_digest"], S[j]["plane_digest"],
                      dict(matrix=ident(), fitness=0.9, rmse=0.4, support=None, warns=[]))
            c.save(); return {"fitted": len(payload["todo"])}
        EN.Ctx.job = fit

    def tearDown(self):
        EN.Ctx.job = self.real_job; remove_fakes(); shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self):
        e = Engine(Project.open(self.tmp), out=io.StringIO()); e._post = lambda *a, **k: None; return e

    def pairs_meta(self, e): return info_of(e)["lineup.pairs"]["meta"]

    def test_force_refits_old_fits(self):
        e = self.engine(); r = e.adopt_legacy()
        self.assertIn("lineup.pairs", r["adopted"]); self.assertTrue(any("older pair_all" in n for n in r["notes"]))
        self.assertEqual(self.pairs_meta(e)["stats"]["old_fits"], 6)
        rr = e.run(force=["lineup.pairs"], to="lineup.pairs", quiet=True); self.assertTrue(rr.ok, rr.error)
        self.assertEqual(self.jobs, ["methods.pairs:pairs_job"])
        st = self.pairs_meta(e)["stats"]; self.assertEqual((st["fitted"], st["old_fits"]), (6, 0))
        self.jobs.clear(); e.run(to="lineup.pairs", quiet=True); self.assertEqual(self.jobs, [])

    def test_adopt_refit_old_pairs_after_a_default_adopt(self):
        e = self.engine(); e.adopt_legacy()
        self.assertEqual(status_of(e)["lineup.pairs"], "adopted")
        r = e.adopt_legacy(refit_old_pairs=True)
        self.assertNotIn("lineup.pairs", r["adopted"]); self.assertIn(status_of(e)["lineup.pairs"], ("stale", "waiting"))
        cache = json.load(open(os.path.join(self.tmp, "stages", "cache", "pairs.json")))
        self.assertFalse([k for k, v in cache.items() if v.get("old_version")])
        e.run(to="lineup.pairs", quiet=True)
        self.assertEqual(self.jobs, ["methods.pairs:pairs_job"]); self.assertEqual(self.pairs_meta(e)["stats"]["fitted"], 6)

    def test_cli_note_command_works(self):
        e = self.engine(); r = e.adopt_legacy()
        note = next(n for n in r["notes"] if "older pair_all" in n)
        self.assertIn("run --force lineup.pairs", note)
        rc, _out, err = cli_run([self.tmp, "run", "--force", "lineup.pairs", "--to", "lineup.pairs"])
        self.assertEqual(rc, 0, err); self.assertEqual(self.jobs, ["methods.pairs:pairs_job"])


# ---------------------------------------------------------------------------------------------------------- R5
class R5CodeAndTablesInKeys(Base):
    def setUp(self):
        super().setUp(); self.ov = tempfile.mkdtemp(prefix="stages_repo_"); self.old = os.environ.get("POINTYOINK_STAGES_REPO")

    def tearDown(self):
        if self.old is None: os.environ.pop("POINTYOINK_STAGES_REPO", None)
        else: os.environ["POINTYOINK_STAGES_REPO"] = self.old
        shutil.rmtree(self.ov, ignore_errors=True); super().tearDown()

    def keys_(self, wp=None):
        _g, info, _r, _e = self.e.plan(with_params=wp); return {k: v.get("key") for k, v in info.items()}

    def codes(self):
        """step id -> the code digest its key carries (Engine.step_key hashes it in), for every step, built or not."""
        self.e._begin(); g = self.e.graph(); return {st.id: self.e.code_digest(st) for st in g.steps if st.method is not None}

    def test_fuse_py_in_points_key(self):
        files = [n for n, _p in get_method("fused", "points").code({})[0]]
        self.assertIn("fuse.py", files); self.assertIn("stages/frames.py", files)
        self.assertIn("registration_metrics.py", [n for n, _p in get_method("lineup.pairs", "fpfh_icp").code({})[0]])
        self.e.set({"fused.method": "points"}); k0 = self.codes()
        shutil.copy2(os.path.join(ROOT, "fuse.py"), os.path.join(self.ov, "fuse.py"))
        os.environ["POINTYOINK_STAGES_REPO"] = self.ov; self.assertEqual(self.codes(), k0)
        with open(os.path.join(self.ov, "fuse.py"), "a") as f: f.write("\n# edited\n")
        k1 = self.codes(); self.assertNotEqual(k1["fused"], k0["fused"]); self.assertEqual(k1["lineup.place"], k0["lineup.place"])

    def test_tables_in_keys(self):
        self.e.set({"fused.method": "points"})
        g = self.e.graph(); ov = [i for i in g.lists["fused.points"] if g.by_id[i].op["tool"] == "overlap"][0]
        k0 = self.codes(); saved = (dict(PR.QUALITY_STEPS), PR.overlap_schedule)
        try:
            PR.QUALITY_STEPS[5] = (0, 1.3)
            k1 = self.codes(); self.assertNotEqual(k1["meshed"], k0["meshed"]); self.assertEqual(k1[ov], k0[ov])
            PR.overlap_schedule = lambda p: ((1.0,), 0.3)
            k2 = self.codes(); self.assertNotEqual(k2[ov], k1[ov]); self.assertEqual(k2["fused"], k0["fused"])
        finally:
            PR.QUALITY_STEPS.clear(); PR.QUALITY_STEPS.update(saved[0]); PR.overlap_schedule = saved[1]
        self.assertEqual(self.codes(), k0)
        self.assertTrue(any(l.startswith("params.mesh_smooth=") for l in SAVED["tools"][("smooth", "mesh")]().code({})[1]))

    def test_script_flag_mapping_in_keys(self):
        self.run_(); k0 = self.keys_(); lst = PR.COMMON["side.keep"]; old = lst[0]
        try:
            lst[0] = dataclasses.replace(old, script_default=0.05)          # keep_part.py's own default moved
            k1 = self.keys_()
        finally:
            lst[0] = old
        for s in self.ids:
            self.assertNotEqual(k1["side.keep@" + s], k0["side.keep@" + s]); self.assertEqual(k1["side.fuse@" + s], k0["side.fuse@" + s])


# ---------------------------------------------------------------------------------------------------------- R6
class R6TokensChecked(Base):
    def test_with_params(self):
        g = self.e.graph(None, {"side[2].keep.min_share": "0.2"})
        self.assertEqual(g.by_id["side.keep@" + self.ids[1]].params["min_share"], 0.2)
        self.assertEqual(g.by_id["side.keep@" + self.ids[0]].params["min_share"], 0.03)
        for bad in ({"fused.pointdistance": "0.3"}, {"side[9].keep.min_share": "0.2"}, {"fused.tsdf.nope": "1"}, {"nosuch.x": "1"}):
            with self.assertRaises(ValueError): self.e.graph(None, bad)
        with self.assertRaises(ValueError): self.e.run(with_params={"fused.pointdistance": "0.3"}, quiet=True)
        self.run_(); ran = self.run_(with_params={"side[2].keep.min_share": "0.2"})
        self.assertIn("side.keep@" + self.ids[1], ran); self.assertNotIn("side.keep@" + self.ids[0], ran)

    def test_to_force_from_by_position(self):
        self.run_()
        self.e.set({"side[2].keep.min_share": "0.2"})
        self.assertEqual(self.run_(to="side@2"), ["side.keep@" + self.ids[1]])
        self.assertEqual(self.run_(force=["side.platter@2"], to="side@2"), ["side.platter@" + self.ids[1]])
        self.assertEqual(self.run_(force=["side@3"], to="side.keep@3")[:1], ["side.fuse@" + self.ids[2]])
        for kw in ({"to": "side@9"}, {"force": ["side.keep@nosuch"]}, {"force": ["nosuch"]}, {"to": "lineup.nope"}, {"from_": "side@0"}):
            with self.assertRaises(ValueError, msg=str(kw)): self.e.run(quiet=True, **kw)
        _g, info, to_run, _est = self.e.plan(to="side.keep@2")
        self.assertIn("side.keep@" + self.ids[1], info)
        self.assertEqual(self.e.versions("side.keep@2")[0]["node"], "side.keep@" + self.ids[1])

    def test_show(self):
        self.run_()
        rc, out, err = cli_run([self.tmp, "show", "side.keep@2"])
        self.assertEqual(rc, 0, err); self.assertIn("side.keep@" + self.ids[1], out)
        rc, out, _err = cli_run([self.tmp, "show", "side@2"])
        self.assertEqual(rc, 0); self.assertIn("side.platter@" + self.ids[1], out)
        for bad in ("side@9", "side.keep@nosuch", "nosuch"):
            rc, _out, err = cli_run([self.tmp, "show", bad]); self.assertEqual(rc, 2, bad); self.assertIn("no s", err)


# ---------------------------------------------------------------------------------------------------------- R7
class R7ExportDeny(Base):
    def test_export_into_denied(self):
        self.run_()
        denied = tempfile.mkdtemp(prefix="stages_denied_"); old = os.environ.get("POINTYOINK_STAGES_DENY")
        os.environ["POINTYOINK_STAGES_DENY"] = denied
        try:
            wb(os.path.join(denied, "model.ply"), "PROTECTED")
            link = os.path.join(self.tmp, "out.ply"); os.symlink(os.path.join(denied, "model.ply"), link)
            rc, _out, err = cli_run([self.tmp, "export", link])
            self.assertEqual(rc, 2); self.assertIn("POINTYOINK_STAGES_DENY", err)
            self.assertEqual(open(os.path.join(denied, "model.ply")).read(), "PROTECTED")
            rc, _out, _err = cli_run([self.tmp, "export", os.path.join(denied, "new.ply")]); self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(os.path.join(denied, "new.ply")))
            ok = os.path.join(self.tmp, "fine.ply"); rc, _out, err = cli_run([self.tmp, "export", ok])
            self.assertEqual(rc, 0, err); self.assertTrue(os.path.exists(ok))
        finally:
            if old is None: os.environ.pop("POINTYOINK_STAGES_DENY", None)
            else: os.environ["POINTYOINK_STAGES_DENY"] = old
            shutil.rmtree(denied, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------- R8
class DensePointsMixin:
    """fused points of the same two plates as PlyFusedMixin, dense (0.1 mm spacing: several points per 0.4 mm cell)."""
    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        g = np.arange(-6.0, 6.0 + 1e-9, 0.1); X, Y = np.meshgrid(g, g); n = X.size
        P = np.r_[np.c_[X.ravel(), Y.ravel(), np.full(n, 1.5)], np.c_[X.ravel(), Y.ravel(), np.zeros(n)]]
        N = np.r_[np.tile([0, 0, 1.0], (n, 1)), np.tile([0, 0, -1.0], (n, 1))]
        plyio.save_points(os.path.join(ctx.out_dir, "points.npz"), P, N, np.ones(len(P)), np.full(len(P), 5.0),
                          np.ones(len(P), np.uint64), ["s"], ctx.params["point_distance"])
        return {"outputs": {"points": "points.npz"}, "stats": {"bbox_mm": [12.0, 12.0, 1.5]}}


class R8PointsDeleteArea(Base):
    def test_area_comparable_after_switch(self):
        use("fused", "tsdf", PlyFusedMixin); use("fused", "points", DensePointsMixin); inproc(self.e)
        from stages import preview
        real = preview.render_removed; preview.render_removed = lambda kind, data, sel, png, title="": wb(png, "png")
        try:
            self.e.clear_ops("fused"); self.run_()
            r = self.e.delete("fused", {"shape": "sphere", "center": [0.0, 0.0, 1.5], "radius": 3.0})
        finally:
            preview.render_removed = real
        rec = self.e.recipe()["ops"]["fused"][0]["recorded"]
        true = np.pi * 9 + np.pi * (9 - 1.5 ** 2)          # the top plate's disc and the bottom plate's (1.5 mm below)
        self.assertAlmostEqual(rec["area_mm2"], true, delta=0.1 * true)                    # faces of the tsdf plates
        self.e.set({"fused.method": "points"})
        self.e._edit(lambda r: (r.setdefault("ops", {}).__setitem__("fused.points", []) or ("no point tools", None)))
        self.run_()
        info = info_of(self.e); oid = "fused#" + r["id"]; meta = info[oid]["meta"]
        self.assertEqual(meta["stats"]["replay"], "shape")
        self.assertLess(meta["stats"]["drift"], 0.10, meta["stats"]); self.assertFalse(meta.get("check"))
        self.assertEqual(info[oid]["status"], "fresh")

    def test_points_area_unit(self):
        from stages.tools.edits import points_area
        rng = np.random.default_rng(1)
        for step in (0.1, 0.25):
            g = np.arange(-8, 8 + 1e-9, step); X, Y = np.meshgrid(g, g)
            P = np.c_[X.ravel(), Y.ravel(), rng.normal(0, 0.02, X.size)]; N = np.tile([0, 0, 1.0], (len(P), 1))
            t = np.radians(40); R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
            P, N = P @ R.T, N @ R.T; sel = np.linalg.norm(P, axis=1) <= 3.0
            self.assertAlmostEqual(points_area({"P": P, "N": N, "cell": 0.4}, sel), np.pi * 9, delta=0.08 * np.pi * 9)


# ---------------------------------------------------------------------------------------------------------- R9
class R9OldDeletesAndAdoptedPlatter(Base):
    def test_preset_moves_old_fused_deletes(self):
        oid = self.e._add_op(lambda r: "fused.tsdf", {"tool": "delete", "shape": "sphere", "center": [0, 0, 0], "radius": 1.0,
                                                      "frame": "model"})                          # made before B1
        self.e.preset("today")
        ops = self.e.recipe()["ops"]
        self.assertEqual([o["id"] for o in ops.get("fused", [])], [oid]); self.assertNotIn("fused.tsdf", ops)
        self.assertEqual(self.e.graph().lists["fused"], ["fused#" + oid])
        self.e.preset("revo"); self.assertEqual([o["id"] for o in self.e.recipe()["ops"]["fused"]], [oid])

    def test_deletes_go_to_the_fused_list(self):
        use("fused", "tsdf", PlyFusedMixin); inproc(self.e)
        from stages import preview
        real = preview.render_removed; preview.render_removed = lambda kind, data, sel, png, title="": wb(png, "png")
        try:
            self.e.clear_ops("fused"); self.run_()
            r = self.e.delete("fused.tsdf", {"shape": "sphere", "center": [0, 0, 1.5], "radius": 2.0})
        finally:
            preview.render_removed = real
        self.assertEqual([o["id"] for o in self.e.recipe()["ops"]["fused"]], [r["id"]])
        with self.assertRaises(ValueError): self.e.add_op("fused", {"tool": "delete", "params": {}})   # no shape

    def test_adopted_platter_soft_fail_is_retried(self):
        fake_build(self.tmp, self.ids)
        for f in ("side2_plane.json", "side2_cut.ply"): os.remove(os.path.join(self.tmp, "build", f))   # build_project's soft fail
        use("side.platter", "auto"); calls = []
        inproc(self.e, lambda cmd: (calls.append((os.path.basename(cmd[1]), cmd[2])), cut_ok(cmd))[1])
        self.e.adopt_legacy()
        info = info_of(self.e); p2 = "side.platter@" + self.ids[1]
        self.assertEqual(info[p2]["status"], "warn"); self.assertTrue(info[p2].get("retry"))
        self.assertEqual(info["side.platter@" + self.ids[0]]["status"], "adopted")
        self.run_()
        self.assertEqual(calls, [("strip_table.py", self.e.p.folder(self.ids[1]))])            # only side 2 tried again
        self.assertEqual(status_of(self.e)[p2], "fresh")


if __name__ == "__main__":
    unittest.main()
