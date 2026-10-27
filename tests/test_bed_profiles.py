"""Bed profiles (named beds in the config dir, the migration of the single learned bed, the project's pick), bed outlines
(circle or rectangle) and everything that used a fixed disc, ringed board dots, plane beds, and the Bed card's list.
Synthetic data only, POINTYOINK_CONFIG in temp dirs, Qt offscreen."""
import json, os, sys, tempfile, unittest
from pathlib import Path
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="bed-profiles-")) / "config.json")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen"); os.environ.setdefault("XMODIFIERS", "@im=none")
os.environ.setdefault("POINTYOINK_NO_DEVICE", "1")
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import bed as B                                                   # noqa: E402
from test_bed import CAL, random_map, rot2, rot3, camera_bed      # noqa: E402


def new_config():
    """A fresh config dir for one test (POINTYOINK_CONFIG points at a file in it)."""
    d = tempfile.mkdtemp(prefix="bed-cfg-"); os.environ["POINTYOINK_CONFIG"] = os.path.join(d, "config.json"); return d


def render_rect(Bcam, M2, rng, outline, dot_r=3.75, lift=0.8, surround=None):
    """IR and depth of a flat dark mat of any outline (bed frame) with dots at M2, like test_bed.render for a disc.
    surround: (height mm, IR) of a surface around the mat (None: nothing seen there)."""
    fx, fy, cx, cy = CAL
    u, v = np.meshgrid(np.arange(800, dtype=float), np.arange(600, dtype=float))
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    n = Bcam[:3, 2]; c = Bcam[:3, 3]; off = -n @ c
    t = -off / (ray @ n); P = ray * t[..., None]
    q = P - c; bx, by = q @ Bcam[:3, 0], q @ Bcam[:3, 1]
    on = (t > 0) & (outline.sdf2(np.stack([bx.ravel(), by.ravel()], 1)).reshape(u.shape) < 0)
    dmin = np.full(u.shape, 1e9)
    for m in M2: dmin = np.minimum(dmin, np.hypot(bx - m[0], by - m[1]))
    dot = on & (dmin < dot_r)
    ir = np.where(on, 25, 8).astype(np.float64) + rng.normal(0, 2, u.shape); ir[dot] = 235
    d = np.zeros(u.shape)
    keep = on & ~dot & (rng.random(u.shape) < 0.5); d[keep] = t[keep]
    dd = dot & (rng.random(u.shape) < 0.4); d[dd] = (-(off - lift) / (ray @ n))[dd]
    if surround is not None:
        h, irs = surround; t2 = -(off + h) / (ray @ n); m = (t2 > 0) & ~on & (rng.random(u.shape) < 0.5)
        d[m] = t2[m]; ir[m & ~on] = irs
    return np.clip(ir, 0, 255).astype(np.uint8), np.round(d * 10).astype(np.uint16)


def rect_map(rng, w=180.0, h=120.0, n=30, gap=18.0, centre=(20.0, -10.0)):
    """Dots spread (unevenly) over a w x h rectangle, off the spin centre."""
    out = []
    while len(out) < n:
        p = np.array([rng.uniform(-w / 2 + 8, w / 2 - 8), rng.uniform(-h / 2 + 8, h / 2 - 8)]) + centre
        if all(np.linalg.norm(p - q) >= gap for q in out): out.append(p)
    return np.array(out)


class OutlineTests(unittest.TestCase):
    def test_circle_and_rect_distances_and_json(self):
        c = B.Outline("circle", (0, 0), 97.0)
        self.assertTrue(c.centred_circle); self.assertAlmostEqual(c.reach, 97.0)
        np.testing.assert_allclose(c.sdf2([[0, 0], [97, 0], [0, 100]]), [-97, 0, 3])
        r = B.Outline("rect", (10, 0), width=100, depth=40, angle=90)
        np.testing.assert_allclose(r.sdf2([[10, 0], [10, 50], [10, 55], [30, 0], [25, 0]]), [-20, 0, 5, 0, -5], atol=1e-9)
        self.assertAlmostEqual(r.reach, np.hypot(30, 50), places=6)
        r2 = B.Outline.from_json(json.loads(json.dumps(r.to_json())))
        self.assertEqual(r2.key(), r.key()); self.assertEqual(B.Outline.from_json(None, 97.0).key(), c.key())
        b = B.BedModel(radius_mm=97.0); self.assertEqual(b.outline.key(), c.key())
        j = b.to_json(); self.assertEqual(j["radius_mm"], 97.0); self.assertEqual(B.BedModel.from_json(j).outline.key(), c.key())
        old = {k: v for k, v in j.items() if k not in ("outline", "kind")}                  # a bed.json from before
        self.assertEqual(B.BedModel.from_json(old).outline.key(), c.key())

    def test_centred_circle_uses_the_old_disc_maths(self):
        rng = np.random.default_rng(1); P = rng.uniform(-150, 150, (5000, 3)) + [0, 0, 300]
        hit = B.Hit(n=B._unit([0.1, -0.8, -0.5]), centre=np.array([5.0, 3.0, 300.0]), x_axis=None)
        q = P - hit.centre; rin = np.linalg.norm(q - np.outer(q @ hit.n, hit.n), axis=1)
        bed = B.BedModel(radius_mm=97.0)
        np.testing.assert_array_equal(B._in_outline(P, hit, bed, -5.0), rin < 92.0)
        np.testing.assert_array_equal(B._in_outline(P, hit, 97.0, 5.0), rin < 102.0)

    def test_fit_outline_round_map_stays_a_circle_rect_map_is_a_rectangle(self):
        rng = np.random.default_rng(2)
        o = B.fit_outline(random_map(rng), 7.5, reach=104.0)
        self.assertEqual(o.shape, "circle"); self.assertEqual(o.radius, 104.0)
        M = rect_map(rng, 240, 150, 40) @ rot2(0.4).T
        o = B.fit_outline(M, 7.5)
        self.assertEqual(o.shape, "rect")
        self.assertLess(abs(max(o.width, o.depth) - 240), 16); self.assertLess(abs(min(o.width, o.depth) - 150), 16)
        self.assertTrue((o.sdf2(M) < 0).all())
        grid = np.stack(np.meshgrid(np.arange(5) * 50.0, np.arange(8) * 50.0), -1).reshape(-1, 2) - [100, 175]
        o = B.fit_outline(grid, 12.0)                                  # the calibration board's 5 x 8 grid
        self.assertEqual(o.shape, "rect"); self.assertAlmostEqual(o.width, 350 + 18, delta=1); self.assertAlmostEqual(o.depth, 200 + 18, delta=1)


class RectBedTests(unittest.TestCase):
    def _bed(self, rng):
        M = rect_map(rng); o = B.Outline("rect", (20.0, -10.0), width=200.0, depth=140.0, angle=0.0)
        return M, o, B.BedModel(dots=np.column_stack([M, np.full(len(M), 10), np.full(len(M), 0.5)]), outline=o, lift_mm=0.8)

    def test_find_bed_on_a_rectangular_mat(self):
        """A rect bed is found like a round one; its in-bed checks use the rectangle, not a disc about the centre."""
        rng = np.random.default_rng(5); M, o, bed = self._bed(rng)
        for trial in range(3):
            Bc = camera_bed(rng, dist=330.0); ir, d = render_rect(Bc, M, rng, o, surround=(40.0, 8))
            h = B.find_bed(d, ir, CAL, bed)
            self.assertIsNotNone(h)
            self.assertLess(B.ang_deg(h.n, Bc[:3, 2]), 0.6); self.assertLess(abs(h.n @ Bc[:3, 3] + h.off), 0.6)
            self.assertLess(np.linalg.norm(h.centre - Bc[:3, 3]), 2.0)
            self.assertEqual(h.height_from, "surface"); self.assertLess(h.beneath, B.V1_MAX)
            # the floor 40 mm down, seen past the mat's edges: a disc as far out as the rectangle reaches takes it in
            disc = B.v1_nothing_beneath(d, CAL, h, o.reach)
            self.assertGreater(disc, B.V1_MAX); self.assertLess(h.beneath, disc / 3)

    def test_learner_learns_a_rectangle(self):
        rng = np.random.default_rng(8); M, o, _ = self._bed(rng)
        Bc0 = camera_bed(rng, tilt_deg=40.0, dist=380.0); n = Bc0[:3, 2]; lr = B.BedLearner(CAL, t0=0.0); st = None
        for k in range(90):
            Bc = Bc0.copy(); Bc[:3, :3] = rot3(n, np.radians(6.0 * k)) @ Bc0[:3, :3]
            ir, d = render_rect(Bc, M, rng, o)
            st = lr.add(d, ir, t=k / 3.0)
            if st["state"] in ("done", "failed"): break
        self.assertEqual(st["state"], "done", st)
        got = lr.result(); self.assertEqual(got.outline.shape, "rect", got.outline.to_json())
        self.assertLess(abs(max(got.outline.width, got.outline.depth) - 200), 12, got.outline.to_json())
        self.assertLess(abs(min(got.outline.width, got.outline.depth) - 140), 12, got.outline.to_json())
        self.assertEqual(got.quality["dots"], len(got)); self.assertIn("not_round", got.quality)
        r = B.register_maps(np.column_stack([M, np.ones(len(M))]), got.dots); self.assertGreaterEqual(r.inliers, 20)


def ringed_picture(rng, spots, r=7, ringed=()):
    """Bright round dots (radius r px) on a dark board; the ones in `ringed` sit in 3 arc segments (a broken ring
    1.6-2.2 r out, gaps in different places)."""
    ir = np.full((600, 800), 12.0) + rng.normal(0, 2, (600, 800))
    yy, xx = np.mgrid[0:600, 0:800]
    for k, (x, y) in enumerate(spots):
        rr = np.hypot(xx - x, yy - y); ir[rr < r] = 235
        if k in ringed:
            a = (np.degrees(np.arctan2(yy - y, xx - x)) + 360 * rng.random()) % 360
            arcs = (rr > 1.6 * r) & (rr < 2.2 * r) & (((a > 0) & (a < 100)) | ((a > 125) & (a < 225)) | ((a > 250) & (a < 340)))
            ir[arcs] = 225
    return np.clip(ir, 0, 255).astype(np.uint8)


class RingedBoardTests(unittest.TestCase):
    def test_learning_rejects_a_pose_with_incompatible_ring_labels(self):
        # A geometric match alone must not turn plain dots into a previously ringed pattern.
        lr = B.BedLearner(CAL, t0=0.0)
        lr.plane = (np.array([0.0, 0.0, -1.0]), 300.0); lr.e = B.basis(lr.plane[0])
        lr.height_from = "surface"
        dl = np.array([[x, y, 7.0, 150.0, 1.0, 1.0] for x, y in
                       [(120, 140), (230, 170), (370, 120), (510, 200), (620, 130), (180, 340), (410, 370), (650, 320)]])
        d = np.zeros((600, 800), np.uint16); ir = np.zeros((600, 800), np.uint8)
        P3 = np.full((len(dl), 3), np.nan); rings = np.ones(len(dl), bool)
        lr._fno = 1
        self.assertIsNotNone(lr._register(dl, P3, d, ir, rings))
        lr._fno = 2
        self.assertIsNotNone(lr._register(dl, P3, d, ir, rings))
        count = sum(lr.cn); accepted = len(lr.Rs)
        lr._fno = 3
        self.assertIsNone(lr._register(dl, P3, d, ir, ~rings))
        self.assertEqual(sum(lr.cn), count); self.assertEqual(len(lr.Rs), accepted)
        self.assertEqual(lr.fails, 1)
        # Callers without ring observations keep the existing geometric matching path.
        lr._fno = 4
        self.assertIsNotNone(lr._register(dl, P3, d, ir))

    def test_dots_in_broken_rings_are_kept_and_arcs_left_out(self):
        rng = np.random.default_rng(3)
        spots = [(100 + 90 * i, 120 + 110 * j) for i in range(7) for j in range(4)]
        ringed = set(range(0, len(spots), 2))
        ir = ringed_picture(rng, spots, ringed=ringed)
        plain = B.detect_dots(ir)                                          # the round-mat detection, as it was
        stats = {}; dots, fl = B.detect_marks(ir, True, stats)
        self.assertEqual(len(dots), len(spots), "every dot, ringed or not, and no arc")
        for (x, y), f in zip(dots[:, :2], fl):
            k = int(np.argmin([np.hypot(x - a, y - b) for a, b in spots]))
            self.assertLess(np.hypot(x - spots[k][0], y - spots[k][1]), 0.5); self.assertEqual(bool(f), k in ringed)
        self.assertGreater(stats["not_round"], 0); self.assertGreater(stats["blobs"], len(spots))
        self.assertLessEqual(len(plain), len(spots))
        for x, y in plain[:, :2]: self.assertLess(min(np.hypot(x - a, y - b) for a, b in spots), 0.5)   # never an arc

    def test_ring_flags_rule_out_the_half_turn_on_a_checkered_grid(self):
        g = np.stack(np.meshgrid(np.arange(5), np.arange(8)), -1).reshape(-1, 2)
        M = g * 30.0 - [60, 105]; fl = (g.sum(1) % 2) == 0                 # every other dot in a ring
        a, t = 0.5, np.array([7.0, -4.0]); F = (M - t) @ rot2(a)            # the whole board in view: map = R f + t
        r0 = B.register_2d(F, M, exhaustive=True)
        self.assertEqual(r0.runner_up, r0.inliers)                          # a plain grid matches turned 180 deg just as well
        r1 = B.register_2d(F, M, exhaustive=True, flags=(fl, fl))
        self.assertEqual(r1.inliers, len(M)); self.assertLess(r1.runner_up, r1.inliers)
        self.assertLess(abs(np.arctan2(r1.R[1, 0], r1.R[0, 0]) - a), 1e-6)
        self.assertLess(B.compute_self_ambiguity(M, fl), B.compute_self_ambiguity(M))


class PlaneBedTests(unittest.TestCase):
    def test_a_plane_bed_never_looks_for_dots(self):
        rng = np.random.default_rng(4); Bc = camera_bed(rng); M = random_map(rng)
        from test_bed import render
        ir, d = render(Bc, M, rng)                                          # dot-like letters in view
        pl = B.plane_profile("Bare turntable")
        called = []; orig = B.detect_marks
        B.detect_marks = lambda *a, **k: called.append(1) or orig(*a, **k)
        try:
            self.assertIsNone(B.find_bed(d, ir, CAL, pl))
            self.assertIsNone(B.solve_pool([({}, np.eye(4))] * 3, pl, CAL))
        finally: B.detect_marks = orig
        self.assertEqual(called, []); self.assertFalse(B.has_dots(pl)); self.assertEqual(pl.kind, "plane")

    def test_plane_fit_sits_on_the_lowest_layer_under_ridges(self):
        """The bare turntable's raised radial ridges must not lift or tilt its plane."""
        rng = np.random.default_rng(9); Bc = camera_bed(rng)
        fx, fy, cx, cy = CAL
        u, v = np.meshgrid(np.arange(800, dtype=float), np.arange(600, dtype=float))
        ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        n = Bc[:3, 2]; c = Bc[:3, 3]; off = -n @ c
        t = -off / (ray @ n); q = ray * t[..., None] - c; bx, by = q @ Bc[:3, 0], q @ Bc[:3, 1]
        ang = np.degrees(np.arctan2(by, bx)) % 360; r = np.hypot(bx, by)
        ridge = np.zeros(u.shape, bool)
        for a0 in (0, 37, 101, 160, 215, 290, 331): ridge |= (np.abs(ang - a0) < 4) & (r > 15)
        ridge &= r < 100
        h = np.where(ridge, 3.0, 0.0)                                       # ridges 3 mm high
        t2 = -(off - h) / (ray @ n); on = (r < 100) & (rng.random(u.shape) < 0.6)
        d = np.where(on, t2 + rng.normal(0, 0.3, u.shape), 0); d = np.round(d * 10).astype(np.uint16)
        got = B.find_plane_bed(d, CAL)
        self.assertIsNotNone(got)
        self.assertLess(B.ang_deg(got[0], n), 0.5); self.assertLess(abs(got[0] @ c + got[1]), 0.4)
        self.assertIsNone(B.match_relief(d, CAL, B.plane_profile("x"), got))


class ProfileTests(unittest.TestCase):
    def _learned(self, radius=109.04):
        rng = np.random.default_rng(10); M = random_map(rng)
        return B.BedModel(dots=np.column_stack([M, np.full(len(M), 100), np.full(len(M), 0.3)]), radius_mm=radius,
                          id="bed-20260927-181545", learned="2026-09-27T18:15:45", lift_mm=0.8)

    def test_migration_keeps_the_learned_bed_and_writes_the_presets(self):
        d = new_config(); b = self._learned(); b.save(os.path.join(d, "bed.json"))
        raw = open(os.path.join(d, "bed.json"), "rb").read()
        paths = B.migrate_profiles()
        self.assertEqual(sorted(os.path.basename(p) for p in paths),
                         ["bed-bare-turntable.json", "bed-calibration-board.json", "bed-round-dot-mat.json"])
        self.assertEqual(open(os.path.join(d, "bed.json"), "rb").read(), raw)          # bed.json untouched
        self.assertEqual([n for n, _ in B.list_profiles()], ["Bare turntable", "Calibration board", "Round dot mat"])
        self.assertEqual(B.picked_profile(), "Round dot mat")
        rm = B.load_profile("Round dot mat")
        self.assertEqual((rm.id, rm.learned, rm.kind, len(rm)), (b.id, b.learned, "dots", len(b)))
        np.testing.assert_allclose(rm.dots, np.round(b.dots, 3))
        self.assertEqual(rm.outline.key(), ("circle", 0.0, 0.0, 99.5))               # 109 mm ran long: the mat is 199 across
        self.assertEqual(rm.quality["learned_radius_mm"], 109.04)
        cb = B.load_profile("Calibration board")
        self.assertEqual((cb.kind, cb.board, len(cb), cb.outline.shape), ("dots", "ringed", 0, "rect"))
        self.assertEqual(sorted((cb.outline.width, cb.outline.depth)), [259.0, 410.0]); self.assertFalse(B.has_dots(cb))
        bt = B.load_profile("Bare turntable"); self.assertEqual((bt.kind, bt.outline.radius), ("plane", 100.0))
        self.assertEqual(B.migrate_profiles(), [])                                    # once only

    def test_migration_keeps_a_radius_near_the_measured_one(self):
        d = new_config(); self._learned(radius=101.0).save(os.path.join(d, "bed.json")); B.migrate_profiles()
        self.assertEqual(B.load_profile("Round dot mat").outline.radius, 101.0)

    def test_no_learned_bed_gives_presets_and_no_pick(self):
        new_config(); B.migrate_profiles()
        self.assertEqual(len(B.list_profiles()), 3); self.assertIsNone(B.picked_profile())
        self.assertEqual(len(B.load_profile("Round dot mat")), 0)

    def test_outline_check_warns_past_5_mm(self):
        b = self._learned(); b.nominal = B.Outline("circle", (0, 0), 99.5).to_json()
        self.assertIn("218 mm across", B.outline_check(b))
        b.radius_mm = 102.0; self.assertIsNone(B.outline_check(b))
        b.outline = B.Outline("rect", (0, 0), width=262, depth=409, angle=3); b.nominal = B.Outline("rect", width=410, depth=259).to_json()
        self.assertIsNone(B.outline_check(b))
        b.outline.width = 250; self.assertIn("409 x 250", B.outline_check(b))


class AdapterTests(unittest.TestCase):
    def _open(self, project, cfg_bed=None):
        from app2.scan.bed_adapter import ScannerBed
        a = ScannerBed(); a.open(project, tempfile.mkdtemp(), CAL, print); return a

    def _side(self, folder):
        from types import SimpleNamespace
        return SimpleNamespace(folder=folder, frames=0, bed=None, ps=0, idx=0)

    def tearDown(self):
        a = getattr(self, "a", None)
        if a is not None: a.close()

    def test_pick_is_kept_by_the_project_and_reopening_uses_it(self):
        d = new_config(); ProfileTests._learned(self).save(os.path.join(d, "bed.json"))
        proj = tempfile.mkdtemp(prefix="ProjectLive")
        self.a = a = self._open(proj)
        self.assertEqual(a.profile_name, "Round dot mat"); self.assertEqual(a.src, "config"); self.assertIsNotNone(a.bed)
        self.assertEqual(a.profiles(), ("Bare turntable", "Calibration board", "Round dot mat", "Freehand (no bed)"))
        info = a.summary(); self.assertEqual((info.name, info.kind, info.outline), ("Round dot mat", "dots", ("circle", 0.0, 0.0, 99.5)))
        sd = self._side(os.path.join(proj, "side")); a.side_opened(sd, False)
        msg = a.pick_profile("Bare turntable", [sd])
        self.assertIn("flat, no dots", msg); self.assertIsNone(a.bed); self.assertIsNotNone(a.plane)
        self.assertEqual(B.read_project_profile(proj)["name"], "Bare turntable"); self.assertEqual(B.picked_profile(), "Bare turntable")
        self.assertIsNone(sd.bed.sb.bed); self.assertEqual(a.base_text(sd, True), "base: Auto (flat bed, no dots)")
        self.assertIsNone(a.platter(sd)); self.assertEqual(a.summary().kind, "plane")
        # no provisional bed from a side's own dots while a plane bed is picked: its letters are not dots
        sd.bed.seed_tried = False; a.after_save(sd, (np.array([0, 0, 1.0]), -10.0), 1, [sd])
        self.assertTrue(a.seeds.empty())
        a.close(); self.a = None
        B.pick_profile("Round dot mat")                                            # another project picked something else since
        self.a = a = self._open(proj)
        self.assertEqual(a.profile_name, "Bare turntable"); self.assertIsNone(a.bed)  # the project's own pick
        msg = a.pick_profile("Round dot mat", [])
        self.assertIn("Round dot mat", msg); self.assertIsNotNone(a.bed)
        self.assertEqual(B.BedModel.load(os.path.join(proj, "bed.json")).id, "bed-20260927-181545")
        a.close(); self.a = None
        self.a = a = self._open(proj)
        self.assertEqual((a.profile_name, a.src, a.bed.id), ("Round dot mat", "project", "bed-20260927-181545"))

    def test_an_unlearned_profile_is_picked_but_not_looked_for(self):
        new_config(); proj = tempfile.mkdtemp(prefix="ProjectLive")
        self.a = a = self._open(proj)
        self.assertIsNone(a.bed); self.assertIsNone(a.profile_name)                 # no learned bed: as before
        msg = a.pick_profile("Calibration board", [])
        self.assertIn("not learned yet", msg); self.assertIsNone(a.bed); self.assertIsNone(a.summary())
        self.assertEqual(a.profile_name, "Calibration board")
        a.learn_toggle(3, False); self.assertEqual(a.learn_name, "Calibration board"); a.learn_toggle(3, False)

    def test_geometry_for_a_rect_bed_is_a_placed_outline(self):
        from app2.scan.bed_adapter import _shape, learned_text
        o = B.Outline("rect", (10, 0), width=200, depth=140)
        b = B.BedModel(dots=np.zeros((10, 4)), outline=o); pl = _shape(b, [1.0, 0, 0])
        self.assertTrue(hasattr(pl, "sdf")); self.assertAlmostEqual(float(pl), o.reach)
        self.assertEqual(_shape(B.BedModel(dots=np.zeros((10, 4)), radius_mm=97.0), [1.0, 0, 0]), 97.0)
        b.quality = {"not_round": 4}; self.assertEqual(learned_text(b), "10 dots kept (4 shapes that were not round dots left out), bed 200 x 140 mm")


class CutTests(unittest.TestCase):
    """Cut below bed only takes what lies over the bed's outline (grown by RIM_MM): a part hanging down past the bed's
    edge is kept; under the bed it goes as before."""
    def _frame(self, pts, n, off):
        from app2.scan.bedcut import BedCut
        fx, fy, cx, cy = CAL; d = np.zeros((600, 800), np.uint16); px = []
        for P in pts:
            u, v = int(round(P[0] / P[2] * fx + cx)), int(round(P[1] / P[2] * fy + cy)); d[v, u] = int(round(P[2] * 10)); px.append(v * 800 + u)
        return BedCut(CAL, 800, 600), d, px

    def test_hanging_part_outside_the_outline_is_kept(self):
        n = rot3([1, 0, 0], np.radians(-35.0)) @ np.array([0, 0, -1.0]); c = np.array([0.0, 0.0, 300.0]); off = float(-n @ c)
        x = np.array([1.0, 0, 0]) - n[0] * n; x /= np.linalg.norm(x); y = np.cross(n, x)     # x: to the right in the picture
        under_in = c + 20 * x - 10 * n                           # 10 mm under the plane, over the bed: cut
        under_out = c + 130 * x - 10 * n                         # 10 mm under, 33 mm past the 97 mm rim: a hanging part, kept
        floor_out = c + 130 * x - 60 * n                         # far under, beside the bed: the table, cut
        above_out = c + 130 * x + 5 * n                          # beside the bed, over its plane: kept
        for size in (97.0, B.Outline("circle", (0, 0), 97.0).placed(x)):
            cut, d, px = self._frame([under_in, under_out, floor_out, above_out], n, off)
            got = set(cut.cut(d, (n, off, c, size)).tolist())
            self.assertEqual([p in got for p in px], [True, False, True, False], size)
        rect = B.Outline("rect", (60, 0), width=200, depth=80).placed(x)    # the rectangle reaches past 130 mm along x
        cut, d, px = self._frame([under_in, under_out, c + 20 * y * 3 - 10 * n], n, off)
        got = set(cut.cut(d, (n, off, c, rect)).tolist())
        self.assertEqual([p in got for p in px], [True, True, False])
        cut, d, px = self._frame([under_in, under_out], n, off)                 # no outline known: every point under goes
        got = set(cut.cut(d, (n, off, None, None)).tolist()); self.assertEqual([p in got for p in px], [True, True])


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_bed_card_list_and_rect_drawing(self):
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtCore import QRectF
        from app2.qt.bed_card import BedCard, draw_bed, LEARN_NEW
        from app2.scan.bedhook import BedInfo
        card = BedCard(); picked = []; learn = []
        card.profile_picked.connect(picked.append); card.learn_new.connect(lambda: learn.append(1))
        card.set_profiles(("Bare turntable", "Round dot mat"), "Round dot mat")
        items = [card.pick.itemText(i) for i in range(card.pick.count())]
        self.assertEqual(items, ["Bare turntable", "Round dot mat", LEARN_NEW]); self.assertEqual(card.pick.currentText(), "Round dot mat")
        card.pick.setCurrentIndex(0); card._picked(0); self.assertEqual(picked, ["Bare turntable"])
        card._picked(2); self.assertEqual(learn, [1]); self.assertEqual(card.pick.currentText(), "Round dot mat")
        card.set_profiles(("Round dot mat",), None, enabled=False)
        self.assertEqual(card.pick.itemText(0), "Pick a bed"); self.assertFalse(card.pick.isEnabled())
        img = QImage(200, 200, QImage.Format.Format_ARGB32); img.fill(0)
        for o in (None, ("rect", 20.0, -10.0, 200.0, 140.0, 15.0), ("circle", 5.0, 0.0, 90.0)):
            info = BedInfo("b", ((10.0, 5.0),), 130.0, 7.5, "", False, None, "Mat", "dots", o)
            p = QPainter(img); draw_bed(p, QRectF(0, 0, 200, 200), info, big=True); p.end()
        from app2.scan.ui_model import ScanState, bed_card
        s = ScanState(bed_available=True, bed=BedInfo("b", (), 100.0, None, "", False, None, "Bare turntable", "plane", None))
        self.assertEqual(bed_card(s)["kind"], "plane")
        s = ScanState(bed_available=True, bed=None, bed_profile="Calibration board")
        self.assertIn("Calibration board", bed_card(s)["sub"])

    def test_ir_view_draws_a_rect_rim(self):
        from app2.scan.preview import ir_view
        rng = np.random.default_rng(13); Bc = camera_bed(rng); n = Bc[:3, 2]; c = Bc[:3, 3]
        d = np.full((600, 800), 3000, np.uint16); irl = np.full((600, 800), 20, np.uint8)
        rect = B.Outline("rect", (0, 0), width=150, depth=100).placed(Bc[:3, 0])
        a = ir_view(d, irl, CAL, False, (n, float(-n @ c), c, rect), None)
        b = ir_view(d, irl, CAL, False, (n, float(-n @ c), c, 97.0), None)
        self.assertEqual(a.shape, (600, 800, 3)); self.assertFalse(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
