"""Fixes of the staged build engine after review (H1, H2, M1-M8, L1, L2, B1-B5): one test class per item, each written
to fail on the engine as it was before the fix. Fake methods as in test_stages_core (no scans, no GPU). Where a real
method's own code is under test (the platter and report soft fail, delete detect) its child processes are simulated, or
the engine's jobs (stages/job.py) run in this process."""
import fcntl, importlib, io, json, os, shutil, subprocess, sys, tempfile, threading, time, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_stages_core import Base, FakeMixin, CALLS, SAVED, make_project, install_fakes, remove_fakes, wb   # noqa: E402
from test_stages_legacy import fake_build                                                            # noqa: E402
from stages import params as PR                                                                     # noqa: E402
from stages import graph as G                                                                       # noqa: E402
from stages import plyio                                                                            # noqa: E402
from stages import runner as RN                                                                     # noqa: E402
from stages import engine as EN                                                                     # noqa: E402
from stages.engine import Engine                                                                    # noqa: E402
from stages.project import Project                                                                  # noqa: E402
from stages.runner import Runner, StepError, Cancelled                                              # noqa: E402
from stages.methods import REGISTRY, JOB                                                            # noqa: E402
from stages import compat                                                                           # noqa: E402

CLI = os.path.join(ROOT, "scanner", "stages", "cli.py")


# ---------------------------------------------------------------------------------------------------------- helpers
def inproc(engine, fake_cmd=None):
    """Run the engine's jobs (stages/job.py) in this process; any other command goes to fake_cmd(cmd) -> stdout text
    (it may raise StepError)."""
    rn = engine.runner

    def run(cmd, mem_gb=4, gpu=False, log=None, on_line=None, env=None, cwd=None, light=False):
        rn.last_wait = 0.0
        if rn.cancelled: raise Cancelled()
        if len(cmd) >= 3 and cmd[1] == JOB:
            with open(cmd[2]) as f: job = json.load(f)
            mod, fn = job["fn"].split(":")
            res = getattr(importlib.import_module("stages." + mod), fn)(job)
            if job.get("result"):
                with open(job["result"], "w") as f: json.dump(res if res is not None else {}, f)
            return "", 0, 0.0
        if fake_cmd is None: raise AssertionError("unexpected command %s" % cmd)
        text = fake_cmd(cmd)
        for ln in text.splitlines():
            if on_line: on_line(ln)
        return text, 0, 0.0
    rn.run = run


def plates_mesh(step=0.4, half=6.0, gap=1.5):
    """Two parallel square plates (triangulated grids) gap mm apart: the top one faces up, the bottom one down."""
    g = np.arange(-half, half + 1e-9, step); n = len(g)
    X, Y = np.meshgrid(g, g); P = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    Ft = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    top = np.c_[P, np.full(len(P), gap)]; bot = np.c_[P, np.zeros(len(P))]
    V = np.vstack([top, bot]); F = np.vstack([Ft, Ft[:, ::-1] + len(P)])
    return V, F


def write_permuted(path, V, F, seed):
    """The same surface with its vertices, faces and each face's starting corner shuffled (a GPU fuse's order)."""
    rng = np.random.default_rng(seed)
    pv = rng.permutation(len(V)); inv = np.argsort(pv)
    F2 = inv[F][rng.permutation(len(F))]
    r = rng.integers(0, 3, len(F2)); k = np.arange(len(F2))
    F2 = np.stack([F2[k, (r + j) % 3] for j in range(3)], 1)
    plyio.write_mesh(path, V[pv], F2)


class PlyFusedMixin:
    """fused tsdf that writes a real PLY: the same plates in an order that depends on frames_per_side."""
    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        V, F = plates_mesh()
        write_permuted(os.path.join(ctx.out_dir, "mesh.ply"), V, F, int(ctx.params["frames_per_side"]))
        return {"outputs": {"mesh": "mesh.ply"}, "stats": {"bbox_mm": [12.0, 12.0, 1.5]}}


class PlyKeepMixin:
    """side.keep that writes a real PLY (fine plates, 0.1 mm grid, in a fixed order)."""
    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        V, F = plates_mesh(step=0.1, half=3.0); plyio.write_mesh(os.path.join(ctx.out_dir, "side.ply"), V, F)
        return {"outputs": {"mesh": "side.ply"}, "stats": {}}


def use(node, name, *mixins):
    """Put a fake (mixins over the real class) in the registry for this test (Base.tearDown restores the registry)."""
    real = SAVED["methods"][(node, name)]
    REGISTRY["methods"][(node, name)] = type("T_" + real.__name__, tuple(mixins) + (real,), {})


def status_of(e):
    getattr(e, "_begin", lambda: None)(); g = e.graph(); _o, info, _ = e._walk(g, "plan", None)
    return {k: v["status"] for k, v in info.items()}


def retry_of(e):
    """step id -> True when its stored result is a soft fail flagged for retry (round 3)."""
    e._begin(); g = e.graph(); _o, info, _ = e._walk(g, "plan", None)
    return {k: bool(v.get("retry")) for k, v in info.items()}


# ---------------------------------------------------------------------------------------------------------- H1
class H1CodeInKeys(Base):
    """Editing a script (or a method module, or a package version) re-runs exactly the steps that use it."""

    def setUp(self):
        super().setUp(); self.ov = tempfile.mkdtemp(prefix="stages_repo_"); self.old = os.environ.get("POINTYOINK_STAGES_REPO")

    def tearDown(self):
        if self.old is None: os.environ.pop("POINTYOINK_STAGES_REPO", None)
        else: os.environ["POINTYOINK_STAGES_REPO"] = self.old
        shutil.rmtree(self.ov, ignore_errors=True); super().tearDown()

    def plan_keys(self):
        _g, info, _r, _e = self.e.plan(); return {k: v.get("key") for k, v in info.items()}

    def test_script_edit(self):
        self.run_(); k0 = self.plan_keys()
        os.makedirs(os.path.join(self.ov, "scanner"))
        src = os.path.join(ROOT, "scanner", "keep_part.py"); dst = os.path.join(self.ov, "scanner", "keep_part.py")
        shutil.copy2(src, dst)
        os.environ["POINTYOINK_STAGES_REPO"] = self.ov
        self.assertEqual(self.plan_keys(), k0)                       # an identical copy: same keys, nothing to do
        with open(dst, "a") as f: f.write("\n# a comment\n")
        k1 = self.plan_keys()
        for s in self.ids:
            self.assertNotEqual(k1["side.keep@" + s], k0["side.keep@" + s])
            self.assertEqual(k1["side.fuse@" + s], k0["side.fuse@" + s]); self.assertEqual(k1["side.platter@" + s], k0["side.platter@" + s])
        ran = self.run_()
        self.assertEqual(sorted(ran), sorted("side.keep@" + s for s in self.ids))   # same bytes out: cut off below them
        # the real mesh isolation tool runs keep_part.py too (the unit tests' fake tool does not)
        files, libs = SAVED["tools"][("isolation", "mesh")]().code({})
        self.assertIn(dst, [pth for _n, pth in files]); self.assertIn("open3d", libs)

    def test_module_and_lib(self):
        self.run_(); k0 = self.plan_keys()
        mod = os.path.join(self.ov, "place_code.py"); wb(mod, "x = 1\n")
        real = SAVED["methods"][("lineup.place", "consensus")]
        REGISTRY["methods"][("lineup.place", "consensus")] = type("T_Place", (FakeMixin, real), {"modules": (mod,)})
        k1 = self.plan_keys(); self.run_(); wb(mod, "x = 2\n"); k2 = self.plan_keys()
        self.assertNotEqual(k1["lineup.place"], k2["lineup.place"]); self.assertEqual(k1["lineup.pairs"], k2["lineup.pairs"])
        real_lv = EN.lib_version
        EN.lib_version = lambda n: "0.0-test" if n == "open3d" else real_lv(n)
        try: k3 = self.plan_keys()
        finally: EN.lib_version = real_lv
        self.assertNotEqual(k3["side.fuse@" + self.ids[0]], k2["side.fuse@" + self.ids[0]])      # fuse.py runs on Open3D


# ---------------------------------------------------------------------------------------------------------- H2
class H2KilledNotStored(Base):
    """A killed / crashed soft-fail step is stored flagged for retry (round 3: shown "warn", retry=True, everything below
    it built) and tried again next run; a deliberate clean exit (STAGE no_result) is kept."""

    def setUp(self):
        super().setUp(); self.mode = "kill"; self.rmode = "ok"; self.calls = []
        use("side.platter", "auto")                                  # the real platter code (its child is simulated)
        inproc(self.e, self.fake_cmd)

    def fake_cmd(self, cmd):
        name = os.path.basename(cmd[1]); self.calls.append(name)
        if name == "strip_table.py":
            if self.mode == "kill": raise StepError("strip_table.py failed (exit 137)", "Killed\n", 137)
            if self.mode == "crash": raise StepError("strip_table.py failed (exit 1)", "Traceback (most recent call last):\nValueError: x\n", 1)
            if self.mode == "exit": raise StepError("strip_table.py failed (exit 2)", 'STAGE no_result {"why": "no platter found"}\n', 2)
            wb(cmd[3], "cut"); wb(cmd[cmd.index("--plane-out") + 1], json.dumps({"n": [0, 0, 1], "d": -10.0, "margin": 1.0, "keep_above": True}))
            return "platter: 100 support points, 1.0 deg off the spin axis; kept 10 triangles in 1 pieces, part height 50.0 mm\n"
        if name == "render_views.py": wb(cmd[3], "png"); return ""
        if name == "needs_scan.py":
            if self.rmode == "kill": raise StepError("needs_scan.py failed (exit 137)", "Killed\n", 137)
            wb(cmd[3], "needs ply"); wb(cmd[cmd.index("--png") + 1], "needs png")
            wb(cmd[cmd.index("--json") + 1], json.dumps({"open_edge_mm": 12.0, "places": []}))
            return "open edge 12 mm\n"
        raise AssertionError(cmd)

    def test_platter_kill_retried(self):
        for mode in ("kill", "crash"):
            self.mode = mode; self.calls.clear(); self.run_()
            self.assertEqual(self.calls.count("strip_table.py"), 4)
            st = status_of(self.e); rt = retry_of(self.e)
            for s in self.ids:
                self.assertEqual(st["side.platter@" + s], "warn", (mode, st)); self.assertTrue(rt["side.platter@" + s])
        self.mode = "ok"; self.calls.clear(); self.run_()
        self.assertEqual(self.calls.count("strip_table.py"), 4)               # retried now that it works
        self.calls.clear(); self.run_(); self.assertEqual(self.calls, [])

    def test_deliberate_exit_kept(self):
        self.mode = "exit"; self.run_(); self.calls.clear(); self.run_()
        self.assertEqual(self.calls.count("strip_table.py"), 0)
        st = status_of(self.e)
        for s in self.ids: self.assertEqual(st["side.platter@" + s], "warn")

    def test_report_kill_retried(self):
        use("fused", "tsdf", PlyFusedMixin); use("report", "needs")        # the real report on a real PLY
        self.e.clear_ops("fused")
        self.mode = "ok"; self.rmode = "kill"; self.run_()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "model_needs.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model.png")))
        self.assertEqual(status_of(self.e)["report"], "warn"); self.assertTrue(retry_of(self.e)["report"])
        self.rmode = "ok"; self.calls.clear(); self.run_()
        self.assertIn("needs_scan.py", self.calls)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model_needs.json")))

    def test_classify(self):
        self.assertEqual(RN.classify(-9, ""), "killed"); self.assertEqual(RN.classify(137, ""), "killed")
        self.assertEqual(RN.classify(1, "Killed\n"), "killed"); self.assertEqual(RN.classify(1, "MemoryError"), "killed")
        self.assertEqual(RN.classify(1, "Traceback (most recent call last):\n  x"), "crash")
        self.assertEqual(RN.classify(2, 'STAGE no_result {"why": "no platter found"}'), "exit"); self.assertEqual(RN.classify(None, ""), "crash")
        self.assertEqual(RN.classify(2, "no platter found"), "unknown")          # round 3: unrecognised = transient


# ---------------------------------------------------------------------------------------------------------- M1
class CallingMixin:
    def execute(self, ctx):
        ctx.call(["-c", "print('child ran')"], mem_gb=1)
        return super().execute(ctx)


class M1Cancel(Base):
    def test_cancel_is_for_one_run(self):
        REGISTRY["methods"][("report", "needs")] = type("T_Report", (CallingMixin, FakeMixin, SAVED["methods"][("report", "needs")]), {})
        self.e.runner.argv = lambda cmd, mem_gb, gpu, light=False: list(cmd)          # no systemd in unit tests
        self.e.cancel()                                                               # nothing running: must not stick
        r = self.e.run(quiet=True)
        self.assertTrue(r.ok, r.error)

    def _runner(self):
        self.config_dir = tempfile.mkdtemp(prefix="stages_config_")
        self.old_config = os.environ.get("POINTYOINK_CONFIG")
        os.environ["POINTYOINK_CONFIG"] = os.path.join(self.config_dir, "config.json")
        rn = Runner(wrap=None); rn.argv = lambda cmd, mem_gb, gpu, light=False: list(cmd)
        return rn

    def _restore(self):
        if self.old_config is None: os.environ.pop("POINTYOINK_CONFIG", None)
        else: os.environ["POINTYOINK_CONFIG"] = self.old_config
        shutil.rmtree(self.config_dir, ignore_errors=True)

    def test_cancel_while_waiting_for_gpu(self):
        rn = self._runner()
        try:
            lp = os.path.dirname(rn.gpu_lock_path); os.makedirs(lp, exist_ok=True)
            held = open(rn.gpu_lock_path, "a+"); fcntl.flock(held, fcntl.LOCK_EX)
            out = {}

            def go():
                try: rn.run([sys.executable, "-c", "print(1)"], gpu=True); out["r"] = "ran"
                except Cancelled: out["r"] = "cancelled"
            t = threading.Thread(target=go, daemon=True); t.start(); time.sleep(0.4)
            rn.cancel(); t.join(3.0)
            alive = t.is_alive()
            fcntl.flock(held, fcntl.LOCK_UN); held.close(); t.join(10)
            self.assertFalse(alive, "still waiting for the GPU lock after cancel")
            self.assertEqual(out.get("r"), "cancelled")
        finally:
            self._restore()

    def test_cancel_between_popen_and_register(self):
        rn = self._runner(); real = RN.subprocess.Popen

        def popen(*a, **k):
            p = real(*a, **k); rn.cancel(); return p                   # cancel lands before the child is registered
        RN.subprocess.Popen = popen
        try:
            t0 = time.time()
            with self.assertRaises(Cancelled): rn.run([sys.executable, "-c", "import time; time.sleep(20)"])
            self.assertLess(time.time() - t0, 10)
        finally:
            RN.subprocess.Popen = real; self._restore()


# ---------------------------------------------------------------------------------------------------------- M2
class M2NewFrames(Base):
    def test_long_lived_engine_sees_new_frames(self):
        self.run_()
        d = os.path.join(self.tmp, "data", self.ids[0], "cache"); wb(os.path.join(d, "frame_0_99.dph"), b"\1\2" * 4)
        groups, _g, _est = self.e.status()
        st = {n.id: n.status for gr in groups for n in gr.nodes}
        self.assertEqual(st["side.fuse@" + self.ids[0]], "stale")
        self.assertIn("side.fuse@" + self.ids[0], self.run_())


# ---------------------------------------------------------------------------------------------------------- M3
class M3PresetOps(Base):
    def test_preset_resets_op_lists(self):
        self.e.preset("revo-device")
        self.assertEqual(len(self.e.recipe()["ops"]["fused.points"]), 3)
        self.e.preset("revo")
        self.assertNotIn("fused.points", self.e.recipe().get("ops", {}))
        g = self.e.graph()
        self.assertEqual([g.by_id[i].op["tool"] for i in g.lists["fused.points"]],
                         ["isolation", "overlap", "smooth", "match_cleaned_sides"])
        self.assertFalse(g.by_id[g.lists["fused.points"][-1]].op["on"])
        self.assertEqual(g.by_id[g.lists["fused.points"][0]].params["rate"], 1.0)        # revo's own, not revo-device's 15
        self.e.preset("today"); self.e.add_op("meshed", {"tool": "simplify", "params": {"keep_pct": 40}})
        self.e._add_op(lambda r: "meshed", {"tool": "delete", "shape": "sphere", "center": [0, 0, 0], "radius": 1.0, "frame": "model"})
        self.e.preset("today")
        self.assertEqual([o["tool"] for o in self.e.recipe()["ops"].get("meshed", [])], ["delete"])   # manual deletes stay


# ---------------------------------------------------------------------------------------------------------- M4
class M4SideTokens(Base):
    def test_positions_become_ids(self):
        self.e.set({"side[2].keep.min_share": "0.2"})
        self.assertEqual(self.e.graph().by_id["side.keep@" + self.ids[1]].params["min_share"], 0.2)
        self.assertEqual(list(self.e.recipe()["sides"]["params"]), [self.ids[1]])
        self.e.reset(["side[2].keep.min_share"]); self.assertEqual(self.e.recipe()["sides"].get("params", {}), {})
        with self.assertRaises(ValueError): self.e.set({"side[nosuchside].keep.min_share": "0.2"})
        with self.assertRaises(ValueError): self.e.set({"side[9].keep.min_share": "0.2"})
        self.e.set({"lineup.place.base": "3"})
        self.assertEqual(self.e.recipe()["params"]["lineup.place"]["base"], self.ids[2])
        with self.assertRaises(ValueError): self.e.set({"lineup.place.base": "nosuchside"})
        with self.assertRaises(ValueError): self.e.graph(None, {"lineup.place.base": "nosuchside"})


# ---------------------------------------------------------------------------------------------------------- M5
class M5ReadOnlyAndDeny(Base):
    def test_read_only_commands_write_nothing(self):
        root = tempfile.mkdtemp(prefix="stages_ro_")
        try:
            ids = make_project(root, n=3); fake_build(root, ids)
            for cmd in (["status"], ["history"], ["versions", "fused"], ["show", "fused"], ["list", "--params"]):
                p = subprocess.run([sys.executable, CLI, root] + cmd, capture_output=True, text=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertFalse(os.path.exists(os.path.join(root, "stages")), cmd)
            e = Engine(Project.open(root, read_only=True))
            self.assertTrue(e.status()[0])
            with self.assertRaises(ValueError): e.set({"report.min_mm": 5})
            self.assertFalse(os.path.exists(os.path.join(root, "stages")))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _deny(self):
        self.denied = tempfile.mkdtemp(prefix="stages_denied_"); self.old = os.environ.get("POINTYOINK_STAGES_DENY")
        os.environ["POINTYOINK_STAGES_DENY"] = self.denied

    def _undeny(self):
        if self.old is None: os.environ.pop("POINTYOINK_STAGES_DENY", None)
        else: os.environ["POINTYOINK_STAGES_DENY"] = self.old
        shutil.rmtree(self.denied, ignore_errors=True)

    def test_publish_target_resolving_into_denied(self):
        self._deny()
        try:
            wb(os.path.join(self.denied, "model.ply"), "PROTECTED")
            os.symlink(os.path.join(self.denied, "model.ply"), os.path.join(self.tmp, "model.ply"))
            r = self.e.run(quiet=True)
            self.assertFalse(r.ok); self.assertEqual(r.failed, "publish")
            self.assertTrue(os.path.islink(os.path.join(self.tmp, "model.ply")))
            self.assertEqual(open(os.path.join(self.denied, "model.ply")).read(), "PROTECTED")
        finally:
            self._undeny()

    def test_build_dir_resolving_into_denied(self):
        self._deny()
        try:
            os.symlink(self.denied, os.path.join(self.tmp, "build"))
            self.assertEqual(compat.main([self.tmp], out=io.StringIO()), 0)
            self.assertEqual(os.listdir(self.denied), [])
        finally:
            self._undeny()


# ---------------------------------------------------------------------------------------------------------- M6
class M6StalePublish(Base):
    def test_absent_report_files_removed(self):
        for f in ("model_needs.ply", "model_needs.png", "model_needs.json"): wb(os.path.join(self.tmp, f), "OLD")
        self.run_()                              # the fake report makes a preview and needs.json only
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "model_needs.ply")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "model_needs.png")))
        self.assertNotEqual(open(os.path.join(self.tmp, "model_needs.json")).read(), "OLD")


# ---------------------------------------------------------------------------------------------------------- M7
class M7ConcurrentEdits(Base):
    def test_no_lost_update(self):
        e2 = Engine(Project.open(self.tmp), out=io.StringIO())
        real = G.build; fired = []; threads = []

        def hooked(project, recipe, with_params=None):
            if not fired and recipe.get("params", {}).get("report", {}).get("min_mm") == 5.0:
                fired.append(1)          # the other editor changes the recipe while this one is between read and write
                t = threading.Thread(target=lambda: e2.set({"report.red_mm": 3.0})); t.start(); threads.append(t)
                time.sleep(0.5)
            return real(project, recipe, with_params)
        G.build = hooked
        try:
            self.e.set({"report.min_mm": 5.0})
            for t in threads: t.join(10)
        finally:
            G.build = real
        rp = self.e.recipe()["params"]["report"]
        self.assertEqual(rp.get("min_mm"), 5.0); self.assertEqual(rp.get("red_mm"), 3.0)
        ents, cur = self.e.history(); self.assertEqual(cur, 2)


# ---------------------------------------------------------------------------------------------------------- M8
class M8DetectIsWhatRuns(Base):
    def test_mask_detect_equals_run(self):
        use("side.keep", "keep_part", PlyKeepMixin); inproc(self.e)
        from stages import preview
        real = preview.render_removed
        preview.render_removed = lambda kind, data, sel, png, title="": wb(png, "png")
        try:
            self.run_(to="sides")
            M = np.array([[0.05, 0.05, 1.5]])                  # a 1 mm brush dab on the top plate (on a 0.1 mm mesh the
            mp = os.path.join(self.tmp, "mask.ply")             # geometric replay differs from the exact dab at its rim)
            plyio.write_points_ply(mp, M, np.array([[0, 0, 1.0]]))
            det = self.e.delete("side@1", {"shape": "mask"}, mask_pts=mp, radius=1.0)
            op = self.e.recipe()["ops"]["side.keep@" + self.ids[0]][0]
            reg = np.load(os.path.join(self.tmp, "stages", op["region"]))
            shown = np.unpackbits(np.load(det["mask"]))[:det["n"]].astype(bool)
            exact = np.zeros(det["n"], bool); exact[reg["idx"]] = True
            self.assertEqual(int((shown != exact).sum()), 0, "the preview differs from what the run deletes")
            self.assertEqual(det["count"], len(reg["idx"]))
            self.run_(to="sides")
            g = self.e.graph(); oid = g.lists["side.keep@" + self.ids[0]][0]
            _o, info, _ = self.e._walk(g, "plan", None)
            st = info[oid]["meta"]["stats"]
            self.assertEqual(st["replay"], "exact"); self.assertEqual(st["removed"], det["count"])
        finally:
            preview.render_removed = real


# ---------------------------------------------------------------------------------------------------------- L1 / L2
class L1AsFusedNeedsMesh(Base):
    def test_rejected(self):
        with self.assertRaises(ValueError): self.e.set({"fused.method": "points", "meshed.method": "as_fused"})
        self.e.set({"meshed.method": "as_fused"})
        with self.assertRaises(ValueError): self.e.set({"fused.method": "points"})
        self.assertEqual(PR.method_of(self.e.recipe(), "fused"), "tsdf")


class L2UnknownToolParam(Base):
    def test_rejected(self):
        n = len(self.e.history()[0])
        with self.assertRaises(ValueError): self.e.add_op("meshed", {"tool": "smooth", "params": {"strenght": "80"}})
        self.assertEqual(len(self.e.history()[0]), n)
        self.e.add_op("meshed", {"tool": "smooth", "params": {"strength": "80"}})
        with self.assertRaises(ValueError): self.e.set_op("meshed", 0, {"strenght": "20"})


# ---------------------------------------------------------------------------------------------------------- B1
class B1FusedDeletesAcrossMethods(Base):
    def test_delete_survives_method_switch(self):
        use("fused", "tsdf", PlyFusedMixin); inproc(self.e)
        from stages import preview
        real = preview.render_removed
        preview.render_removed = lambda kind, data, sel, png, title="": wb(png, "png")
        try:
            self.e.clear_ops("fused"); self.run_()
            r = self.e.delete("fused", {"shape": "box", "box": {"min": [-2, -2, 1.0], "max": [2, 2, 2.0], "T": None}})
            self.assertGreater(r["count"], 0)
        finally:
            preview.render_removed = real
        self.e.add_op("fused", {"tool": "smooth"})                        # a tsdf tool: parked when the method changes
        self.e.set({"fused.method": "points"})
        g = self.e.graph()
        self.assertEqual(g.lists["fused"], ["fused#" + r["id"]])
        self.assertEqual(g.by_id["fused#" + r["id"]].inputs["parent"], g.lists["fused.points"][-1])
        self.assertEqual(g.final("fused"), "fused#" + r["id"])
        _lk, chain, parked = self.e.ops_of("fused")
        self.assertEqual([o["tool"] for _l, _j, o in chain],
                         ["isolation", "overlap", "smooth", "match_cleaned_sides", "delete"])
        self.assertEqual([(lk, o["tool"]) for lk, _j, o in parked], [("fused.tsdf", "smooth")])
        groups, _g, _e = self.e.status()
        self.assertIn("parked", [n.status for gr in groups if gr.name == "fused" for n in gr.nodes])
        self.e.set({"fused.method": "tsdf"})
        self.assertEqual(self.e.graph().lists["fused"], ["fused#" + r["id"]])
        self.e.drop_op("fused", r["id"])                                   # by op id through the chain view
        self.assertEqual(self.e.recipe()["ops"]["fused"], [])


# ---------------------------------------------------------------------------------------------------------- B2
class B2EditOpInPlace(Base):
    def test_one_entry_and_undo(self):
        self.e.add_op("meshed", {"tool": "smooth"}); self.run_()
        d0 = json.load(open(os.path.join(self.tmp, "stages", "published.json")))["digest"]
        n = len(self.e.history()[0])
        op = self.e.set_op("meshed", 0, {"strength": "60"})
        self.assertEqual(op["params"]["strength"], 60.0); self.assertEqual(len(self.e.history()[0]), n + 1)
        self.assertIn("Smooth 30 -> 60", self.e.history()[0][-1]["note"])
        ran = self.run_(); self.assertEqual(ran, [self.e.graph().lists["meshed"][0], "report"])
        self.assertEqual(self.e.undo(), self.e.history()[0][-1]["note"])
        self.assertEqual(self.run_(), [])
        self.assertEqual(json.load(open(os.path.join(self.tmp, "stages", "published.json")))["digest"], d0)


# ---------------------------------------------------------------------------------------------------------- B3
class B3PoissonQuality(unittest.TestCase):
    def test_levels_differ_and_label_is_real(self):
        plans = {q: PR.poisson_plan(190.0, q) for q in range(1, 8)}
        grids = [plans[q]["grid_mm"] for q in range(1, 8)]
        self.assertTrue(all(a > b for a, b in zip(grids, grids[1:])), grids)          # every level finer than the one below
        self.assertEqual((plans[6]["depth"], plans[7]["depth"]), (10, 11))            # Q6 / Q7 as calibrated
        self.assertNotEqual((plans[5]["depth"], plans[5]["scale"]), (plans[6]["depth"], plans[6]["scale"]))
        for q, pl in plans.items():
            self.assertAlmostEqual(pl["grid_mm"], 190.0 * pl["scale"] / 2 ** pl["depth"], places=4)
            self.assertIsNone(pl["same_as"])
        big = PR.poisson_plan(400.0, 6)                     # depth 11 already: Q7 cannot be finer
        self.assertEqual(big["same_as"], 7); self.assertIn("same as Q7", PR.poisson_label(big))
        self.assertIn("grid %s mm" % PR.fmt(plans[6]["grid_mm"]), PR.poisson_label(plans[6]))


# ---------------------------------------------------------------------------------------------------------- B4
class B4OrderIndependentDigest(Base):
    def test_digest(self):
        V, F = plates_mesh(); d = tempfile.mkdtemp(prefix="stages_geo_")
        try:
            a, b = os.path.join(d, "a.ply"), os.path.join(d, "b.ply")
            write_permuted(a, V, F, 1); write_permuted(b, V, F, 2)
            self.assertNotEqual(open(a, "rb").read(), open(b, "rb").read())
            self.assertEqual(plyio.geo_digest(a), plyio.geo_digest(b))
            V2 = V.copy(); V2[5, 2] += 0.01; self.assertNotEqual(plyio.geo_digest_arrays(V2, F), plyio.geo_digest_arrays(V, F))
            F2 = F.copy(); F2[3] = F2[3, ::-1]; self.assertNotEqual(plyio.geo_digest_arrays(V, F2), plyio.geo_digest_arrays(V, F))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_cutoff_on_reordered_mesh(self):
        use("fused", "tsdf", PlyFusedMixin)
        self.run_()
        self.e.set({"fused.frames_per_side": 200})              # fused re-runs, writes the same surface in another order
        self.assertEqual(self.run_(), ["fused"])


# ---------------------------------------------------------------------------------------------------------- B5
class B5AdoptOlderPairFits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stages_b5_"); self.ids = make_project(self.tmp); fake_build(self.tmp, self.ids)
        pp = os.path.join(self.tmp, "build", "pairs_cache.json"); pc = json.load(open(pp))
        from stages.methods.lineup import pair_version
        V = pair_version()
        for e in pc.values():                   # fitted by the previous pair_all (no beds digest back then)
            for k in ("base", "moving"): e[k] = "%d:%s" % (V - 1, e[k].split(":", 1)[1])
            e.pop("beds", None)
        json.dump(pc, open(pp, "w"))

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def test_adopted_by_default(self):
        e = Engine(Project.open(self.tmp)); r = e.adopt_legacy()
        self.assertIn("lineup.pairs", r["adopted"], r)
        self.assertEqual(status_of(e)["fused"], "adopted")

    def test_refit_on_request(self):
        e = Engine(Project.open(self.tmp)); r = e.adopt_legacy(refit_old_pairs=True)
        self.assertNotIn("lineup.pairs", r["adopted"])


if __name__ == "__main__":
    unittest.main()
