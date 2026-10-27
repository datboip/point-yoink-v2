"""Build page, Side stage, Isolation: Detect finds every piece under the rate % of the biggest on the side's platter-cut
mesh, the page says how many pieces / polygons would go, Delete sets the side's rate (undoable), and the info line
counts the side's faces. Offscreen (QT_QPA_PLATFORM=offscreen), no GL, a temp project with fake side methods writing a
real mesh: one big plate, one 19% plate and three small floaters. Nothing outside the temp dirs is written.

Before the first fix the Side's Detect subtracted what the side's current rate (3%) already drops, so any rate at or
over it said "Nothing would go at this setting." while the floaters were there; the info line said Faces: 0 for a
side whose keep step has no "faces" stat (every adopted side). Now the host reports both: every piece under the rate
on the cut mesh, and what is left of them on the finished side; the page offers a Delete only for the latter."""
import os, shutil, sys, tempfile, time, types, unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XMODIFIERS"] = "@im=none"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))
_CFG = tempfile.mkdtemp(prefix="app2_iso_cfg_")
os.environ["POINTYOINK_CONFIG"] = os.path.join(_CFG, "config.json")      # the isolated config (never ~/.config)

import numpy as np                                                                           # noqa: E402
from test_stages_core import make_project, install_fakes, remove_fakes, SAVED, CALLS        # noqa: E402
from test_stages_fixes import inproc                                                         # noqa: E402
from stages import plyio                                                                     # noqa: E402
from stages.engine import Engine                                                             # noqa: E402
from stages.project import Project                                                           # noqa: E402
from stages.methods import REGISTRY                                                          # noqa: E402


def plate(n, x0=0.0, y0=0.0, step=1.0, z=0.0):
    """An n x n vertex grid, 2 (n-1)^2 faces."""
    g = np.arange(n) * step; X, Y = np.meshgrid(g + x0, g + y0)
    V = np.c_[X.ravel(), Y.ravel(), np.full(n * n, z)]
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    return V, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def scene():
    """(V, F, piece id per face): 0 = big (3042 faces), 1 = 19% (578), 2..4 = floaters (8, 18, 32 faces)."""
    parts = [plate(40), plate(18, 60, 0), plate(3, 0, 60), plate(4, 10, 60), plate(5, 20, 60)]
    V, F, P, off = [], [], [], 0
    for k, (v, f) in enumerate(parts):
        V.append(v); F.append(f + off); P.append(np.full(len(f), k)); off += len(v)
    return np.vstack(V).astype(np.float64), np.vstack(F), np.concatenate(P)


V0, F0, PIECE = scene()
FLOATERS = PIECE >= 2                          # under 10% of the biggest: 58 faces in 3 pieces


class PlatterMixin:
    """side.platter writes the scene as its platter-cut mesh."""
    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        plyio.write_mesh(os.path.join(ctx.out_dir, "cut.ply"), V0, F0)
        with open(os.path.join(ctx.out_dir, "plane.json"), "w") as f: f.write('{"n": [0, 0, 1], "d": -5.0, "margin": 1.0, "keep_above": true}')
        return {"outputs": {"mesh": "cut.ply", "plane": "plane.json"}, "stats": {}}


class KeepMixin:
    """side.keep keeps pieces at least min_share of the biggest (keep_part.py's rule) and, like every adopted side,
    records no "faces" stat."""
    def execute(self, ctx):
        CALLS.append(ctx.step.id)
        V, F, _N = plyio.read_mesh(ctx.inputs["cut"].files["mesh"])
        from stages.tools import mesh_tools
        (V2, F2, _n), _sel, _st = mesh_tools.apply("isolation", (V, F, None), {"rate": 100.0 * float(ctx.params["min_share"])})
        plyio.write_mesh(os.path.join(ctx.out_dir, "side.ply"), V2, F2)
        return {"outputs": {"mesh": "side.ply"}, "stats": {}}


def tiny_png(path):
    from PIL import Image
    Image.new("RGB", (8, 8)).save(path)


def use(node, name, mixin):
    real = SAVED["methods"][(node, name)]
    REGISTRY["methods"][(node, name)] = type("T_" + real.__name__, (mixin, real), {})


class Base(unittest.TestCase):
    def setUp(self):
        install_fakes()
        use("side.platter", "auto", PlatterMixin); use("side.keep", "keep_part", KeepMixin)
        self.tmp = tempfile.mkdtemp(prefix="app2_iso_"); self.ids = make_project(self.tmp, n=2)
        import io
        self.e = Engine(Project.open(self.tmp), out=io.StringIO()); self.e._post = lambda *a, **k: None
        inproc(self.e)
        from stages import preview
        self._rr = preview.render_removed; preview.render_removed = lambda kind, data, sel, png, title="": tiny_png(png)
        r = self.e.run(to="sides", quiet=True); self.assertTrue(r.ok, r.error)

    def tearDown(self):
        from stages import preview
        preview.render_removed = self._rr
        shutil.rmtree(self.tmp, ignore_errors=True); remove_fakes()

    def host(self):
        from app2.builder.stagehost import StageHost
        h = StageHost.__new__(StageHost); h.e = self.e; h.p = self.e.p; return h


class HostDetect(Base):
    def test_detect_picks_exactly_the_floaters(self):
        r = self.host().h_side_detect(self.ids[0], 10.0)
        self.assertEqual(r["n"], len(F0))
        self.assertEqual(r["count"], int(FLOATERS.sum()), "the side's 3% drops them too: they must still show as going")
        self.assertTrue(np.array_equal(r["mask"], FLOATERS))
        self.assertEqual(r["pieces"], 3); self.assertEqual(r["back"], 0)
        self.assertTrue(r["input"].endswith("cut.ply"))

    def test_higher_rate_takes_the_medium_piece(self):
        r = self.host().h_side_detect(self.ids[0], 25.0)
        self.assertTrue(np.array_equal(r["mask"], PIECE >= 1)); self.assertEqual(r["pieces"], 4)

    def test_lower_than_now_says_what_comes_back(self):
        self.e.set({"side.keep.min_share": 0.25})
        r = self.host().h_side_detect(self.ids[0], 10.0)
        self.assertTrue(np.array_equal(r["mask"], FLOATERS)); self.assertEqual(r["back"], int((PIECE == 1).sum()))


def app():
    from PySide6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    from app2.qt import theme
    theme.install(a); return a


def pump(cond, secs=20.0):
    from PySide6.QtWidgets import QApplication
    t0 = time.time()
    while time.time() - t0 < secs:
        QApplication.processEvents(); time.sleep(0.01)
        if cond(): return True
    return False


class FakeSession:
    def __init__(self): self.runs = []
    def build(self, backend): self.runs.append(backend)


def open_page(test, size=(1688, 1044)):
    """The Build page on the test engine inside the recovered left-rail window."""
    app()
    from app2.qt.shell import MainWindow
    def offline(*args, **kwargs):
        raise RuntimeError("Isolated test: no scanner")
    win = MainWindow(offline, gl=False, projects_dir=test.tmp)
    page = win.build; page.engine_factory = lambda root: test.e
    win.show_step(1); win.resize(*size)
    page.session = FakeSession()
    page.state = types.SimpleNamespace(project=test.tmp, building=False, recording=False, any_frames=True, build_engine=None, build_msg=None)
    win.show(); page._ensure_host(); page.host.status()
    test.assertTrue(pump(lambda: page.snap is not None), "no status from the engine host")
    page.show_stage("side"); pump(lambda: False, 0.2)
    return win, page


class PageDetect(Base):
    def setUp(self):
        super().setUp(); self.win, self.page = open_page(self)

    def tearDown(self):
        self.page.shutdown(0.5); self.win.close(); self.win.deleteLater()
        super().tearDown()

    def test_detect_delete_undo(self):
        """Isolation on a side looks at the finished side (side.ply, the view): at 10% every piece it finds is one the
        side's 3% already removed, so it says so and offers no Delete; at 25% the 19% plate would go, Delete sets the
        side's rate (a run to its keep), and it undoes."""
        p = self.page; sp = p.panel["side"]; sp.show_tab("Isolation"); b = sp.body("Isolation")
        label = " ".join(str(part[0]) for part in p.vp.main.tile.parts)
        self.assertIn("Side 1 of 2", label)
        b.controls[0].set_value(10, emit=True)                           # 10%: only the floaters, which 3% already took
        p.on_button("Isolation")
        self.assertTrue(pump(lambda: b.mode != "idle"), "Detect said nothing: %r" % b.msg.text())
        self.assertEqual(b.mode, "none"); self.assertTrue(b.msg.text().startswith("Already removed at 3%"), b.msg.text())
        self.assertIn("3 pieces under 10% are gone", b.msg.text())
        self.assertEqual(b.main.text(), "Delete"); self.assertFalse(b.main.isEnabled()); self.assertTrue(b.second.isEnabled())
        self.assertEqual(p.shown.get("path"), p._view_path(), "the view left the finished side")
        p.on_second("Isolation"); self.assertEqual(b.mode, "idle"); self.assertEqual(b.main.text(), "Detect")
        b.controls[0].set_value(25, emit=True)                           # 25%: the 19% plate is still there
        p.on_button("Isolation")
        self.assertTrue(pump(lambda: b.mode == "detected"), "Detect found nothing: %r" % b.msg.text())
        self.assertEqual(b.msg.text(), "1 piece marked \u00b7 578 polygons (%.1f%%)\n3 more already removed at 3%%." % (100.0 * 578 / 3620))
        self.assertEqual(b.main.text(), "Delete"); self.assertTrue(b.main.isEnabled()); self.assertTrue(b.second.isEnabled())
        self.assertTrue(np.array_equal(p.detected["mask"], (PIECE == 1)[PIECE <= 1]))
        p.on_button("Isolation")                                         # Delete: the side's rate, then a run to its keep
        self.assertTrue(pump(lambda: p.session.runs), "Delete started no run")
        self.assertEqual(p.session.runs[-1].to, "side.keep@" + self.ids[0])
        from stages import params as PR
        self.assertAlmostEqual(float(PR.get_path(self.e.recipe(), "side[%s].keep.min_share" % self.ids[0])), 0.25)
        self.assertEqual(b.mode, "idle"); self.assertFalse(b.second.isEnabled())
        self.assertIsNotNone(self.e.undo())                              # undoable
        self.assertAlmostEqual(float(PR.get_path(self.e.recipe(), "side[%s].keep.min_share" % self.ids[0])), 0.03)


if __name__ == "__main__":
    unittest.main()
