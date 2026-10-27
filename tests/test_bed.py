"""Learn bed (scanner/bed.py) and the platter support check (scanner/support.py) on synthetic data: no scans, no GPU."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
import bed as B          # noqa: E402
import support as S      # noqa: E402

CAL = (1143.05, 1143.05, 186.69, 351.22)       # the MIRACO's Pl.bin


def random_map(rng, n=35, rmin=20, rmax=93, gap=18.0):
    out = []
    while len(out) < n:
        r = np.sqrt(rng.uniform(rmin ** 2, rmax ** 2)); a = rng.uniform(0, 2 * np.pi); p = np.array([r * np.cos(a), r * np.sin(a)])
        if all(np.linalg.norm(p - q) >= gap for q in out): out.append(p)
    return np.array(out)


def rot2(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def rot3(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def render(Bcam, M2, rng, radius=97.0, dot_r=3.75, lift=0.8, ring_depth=True, glints=0):
    """IR and depth of a platter (bed -> camera = Bcam) with dots at M2 (bed frame, mm), OpenCV camera, 800 x 600."""
    fx, fy, cx, cy = CAL
    u, v = np.meshgrid(np.arange(800, dtype=float), np.arange(600, dtype=float))
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    n = Bcam[:3, 2]; c = Bcam[:3, 3]; off = -n @ c
    t = -off / (ray @ n); P = ray * t[..., None]
    q = P - c; bx, by = q @ Bcam[:3, 0], q @ Bcam[:3, 1]
    on_disc = (t > 0) & (np.hypot(bx, by) < radius)
    dmin = np.full(u.shape, 1e9)
    for m in M2: dmin = np.minimum(dmin, np.hypot(bx - m[0], by - m[1]))
    dot = on_disc & (dmin < dot_r)
    ir = np.where(on_disc, 25, 8).astype(np.float64) + rng.normal(0, 2, u.shape)
    ir[dot] = 235
    d = np.zeros(u.shape)
    keep = on_disc & ~dot & (rng.random(u.shape) < (0.5 if ring_depth else 0.0))
    d[keep] = t[keep]                                                 # platter surface (z = t on these rays)
    dd = dot & (rng.random(u.shape) < 0.4)
    t_high = -(off - lift) / (ray @ n)                                # the dots read lift mm high (toward the camera)
    d[dd] = t_high[dd]
    for _ in range(glints):                 # small bright specks off the pattern (on the platter, no ring contrast issue)
        gu, gv = rng.integers(40, 760), rng.integers(40, 560); ir[gv - 3:gv + 4, gu - 3:gu + 4] = 240
    return np.clip(ir, 0, 255).astype(np.uint8), np.round(d * 10).astype(np.uint16)


def paste_face(Bcam, ir, d, rng, h=2.5, R=60.0, centre=(0.0, 0.0)):
    """A dark flat face of the part (IR 35, dense depth) h mm over the platter, a disc of radius R (bed frame): it hides
    what lies under it, like a part standing on the table with a flat dark face close to the platter."""
    fx, fy, cx, cy = CAL
    u, v = np.meshgrid(np.arange(800, dtype=float), np.arange(600, dtype=float))
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    n = Bcam[:3, 2]; c = Bcam[:3, 3] + h * n; off = -n @ c
    t = -off / (ray @ n); P = ray * t[..., None]; q = P - c
    bx, by = q @ Bcam[:3, 0] - centre[0], q @ Bcam[:3, 1] - centre[1]
    face = (t > 0) & (np.hypot(bx, by) < R)
    ir = ir.copy(); d = d.copy(); ir[face] = 35
    dense = face & (rng.random(u.shape) < 0.9); d[face] = 0; d[dense] = np.round(t[dense] * 10).astype(np.uint16)
    return ir, d


def camera_bed(rng, tilt_deg=35.0, dist=260.0):
    """A bed placement a MIRACO on a tripod would see: platter normal tilted toward the camera, centre in view."""
    n = rot3([1, 0, 0], np.radians(-tilt_deg)) @ np.array([0, 0, -1.0])
    c = np.array([40.0, 60.0, dist])
    x = np.cross(n, [0, 0, 1.0]); x /= np.linalg.norm(x); x = rot3(n, rng.uniform(0, 2 * np.pi)) @ x
    Bc = np.eye(4); Bc[:3, 0] = x; Bc[:3, 1] = np.cross(n, x); Bc[:3, 2] = n; Bc[:3, 3] = c
    return Bc


class RegisterTests(unittest.TestCase):
    def test_true_match_is_hit_and_accurate(self):
        rng = np.random.default_rng(1); errs = []
        for trial in range(20):
            M = random_map(rng)
            a = rng.uniform(-np.pi, np.pi); t = rng.uniform(-60, 60, 2)
            keep = rng.permutation(len(M))[:len(M) // 2]
            Fm = M[keep]                                           # frame dots in map coordinates
            Fr = (Fm - t) @ rot2(a)                                # frame coordinates: map = R f + t
            Fr = Fr + rng.normal(0, 0.7 / np.sqrt(2), Fr.shape)    # 0.7 mm rms noise
            Fr = np.vstack([Fr, rng.uniform(-90, 90, (5, 2))])     # glints
            r = B.register_2d(Fr, M)
            self.assertTrue(B.is_hit(r), "trial %d: %s" % (trial, None if r is None else (r.inliers, r.runner_up, r.rms)))
            da = np.degrees(abs((np.arctan2(r.R[1, 0], r.R[0, 0]) - a + np.pi) % (2 * np.pi) - np.pi))
            qc = Fr[:len(keep)].mean(0); dp = np.linalg.norm((r.R @ qc + r.t) - (rot2(a) @ qc + t))
            errs.append((da, dp))
        errs = np.array(errs)
        self.assertLess(np.median(errs[:, 0]), 0.2); self.assertLess(np.median(errs[:, 1]), 0.5)
        self.assertLess(errs[:, 0].max(), 0.5); self.assertLess(errs[:, 1].max(), 1.0)

    def test_mirrored_random_and_ring_maps_never_hit(self):
        rng = np.random.default_rng(2); hits = 0
        for trial in range(100):
            M = random_map(rng); keep = rng.permutation(len(M))[:18]
            Fr = M[keep] @ rot2(rng.uniform(-np.pi, np.pi)).T + rng.normal(0, 0.5, (18, 2))
            hits += B.is_hit(B.register_2d(Fr * np.array([1, -1]), M))      # mirrored
            hits += B.is_hit(B.register_2d(Fr, random_map(rng)))              # another map of the same density
        ring = np.array([[80 * np.cos(a), 80 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 13)[:-1]] +
                        [[40 * np.cos(a), 40 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]])
        Fr = ring[:14] @ rot2(0.3).T + [5.0, -3.0]
        hits += B.is_hit(B.register_2d(Fr, ring))                             # a regular layout matches many ways
        self.assertEqual(hits, 0)

    def test_self_ambiguity(self):
        rng = np.random.default_rng(3)
        self.assertLess(B.compute_self_ambiguity(random_map(rng)), 0.4)
        ring = np.array([[80 * np.cos(a), 80 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 13)[:-1]])
        self.assertGreater(B.compute_self_ambiguity(ring), 0.6)


class GeometryTests(unittest.TestCase):
    def test_lift_to_scan_roundtrip_and_fuse_convention(self):
        rng = np.random.default_rng(4); Bc = camera_bed(rng); n = Bc[:3, 2]; off = -n @ Bc[:3, 3]
        pts = Bc[:3, 3] + np.array([[10, 20, 0], [-30, 5, 0], [50, -40, 0]], float) @ Bc[:3, :3].T
        px = np.stack([pts[:, 0] / pts[:, 2] * CAL[0] + CAL[2], pts[:, 1] / pts[:, 2] * CAL[1] + CAL[3]], 1)
        dots = np.column_stack([px, np.full((3, 4), 5.0)])
        L = B.lift(dots, CAL, n, off)
        self.assertLess(np.abs(L - pts).max(), 1e-6)
        T = np.eye(4); T[:3, :3] = rot3([0.3, 1, 0.2], 0.7); T[:3, 3] = [5, -20, 30]
        hit = B.Hit(n=n, off=off, centre=Bc[:3, 3], x_axis=Bc[:3, 0], B_cam=Bc)
        s = B.to_scan(hit, T)
        self.assertLess(np.abs(s["B_scan"] - T @ Bc).max(), 1e-9)
        self.assertLess(abs(s["n"] @ s["centre"] + s["off"]), 1e-9)
        # table_plane.json from a bed keeps fuse.py's rule (keep n . x >= d + margin): 2 mm above kept, 0.5 mm cut
        sb = B.SideBed(B.BedModel(dots=np.zeros((10, 4)), radius_mm=97.0)); sb.state = "found"; sb.fixed = s; sb.dots_matched = 12
        j = sb.table_plane_json(margin=1.0)
        above = lambda h: np.asarray(j["n"]) @ (s["centre"] + h * s["n"]) >= j["d"] + j["margin"]
        self.assertTrue(above(2.0)); self.assertFalse(above(0.5))
        self.assertEqual(j["source"], "bed")

    def test_support_check(self):
        rng = np.random.default_rng(5)
        box = rng.uniform([-30, -20, 0.5], [30, 20, 40], (4000, 3))          # a box standing on z = 0 (n = +z, d = 0)
        plane = {"n": [0, 0, 1.0], "d": 0.0}
        T = np.eye(4); T[:3, :3] = rot3([0, 0, 1], 0.4); T[:3, 3] = [10, 5, 0]
        Pj = box; Pi = box @ T[:3, :3].T + T[:3, 3]; plane_i = S.plane_in(plane, T)
        self.assertEqual(S.pair_under(Pi, plane_i, Pj, plane, T), 0.0)
        Fl = np.eye(4); Fl[:3, :3] = rot3([1, 0, 0], np.pi)                  # turned over about its bottom edge: into the table
        self.assertGreater(S.pair_under(Pi, plane_i, Pj, plane, T @ Fl), 0.3)
        # a pyramid turned over about its middle: the other table cuts through its top (a box would fit either way up)
        pyr = rng.uniform([-30, -30, 0.5], [30, 30, 40], (20000, 3)); pyr = pyr[np.abs(pyr[:, :2]).max(1) < 30 * (1 - pyr[:, 2] / 40)]
        Fm = np.eye(4); Fm[:3, :3] = rot3([1, 0, 0], np.pi); Fm[:3, 3] = [0, 0, 20]
        self.assertGreater(S.pair_under(pyr, plane, pyr, plane, Fm), 0.05)
        self.assertEqual(S.pair_under(pyr, plane, pyr, plane, np.eye(4)), 0.0)


class PairPickTests(unittest.TestCase):
    def test_flipped_best_fit_is_thrown_out_and_same_pose_seen(self):
        import pair_all
        rng = np.random.default_rng(8)
        pyr = rng.uniform([-30, -30, 0.5], [30, 30, 40], (20000, 3)); pyr = pyr[np.abs(pyr[:, :2]).max(1) < 30 * (1 - pyr[:, 2] / 40)]
        plane = {"n": [0, 0, 1.0], "d": 0.0}
        T = np.eye(4); T[:3, :3] = rot3([0, 0, 1], 0.5); T[:3, 3] = [4, -3, 0]            # side j -> side i, part not moved
        Pj = pyr; Pi = pyr @ T[:3, :3].T + T[:3, 3]; pl_i = S.plane_in(plane, T)
        Fl = np.eye(4); Fl[:3, :3] = rot3([1, 0, 0], np.pi); Fl[:3, 3] = [0, 0, 20]
        beds_i = dict(plane=pl_i); beds_j = dict(plane=plane)
        cands = [("fgr", T @ Fl, 0.9, 0.5), ("ransac", T, 0.8, 0.4), ("bed", T, 0.8, 0.4)]
        M, f, e, sup, bed, warns = pair_all.pick(cands, Pi, Pj, beds_i, beds_j, "1-2")
        self.assertLess(np.abs(M - T).max(), 1e-9); self.assertFalse(sup["flag"]); self.assertEqual(sup["rejected"], 1)
        self.assertTrue(any("flipped" in w for w in warns)); self.assertTrue(bed["same_pose"])
        M, f, e, sup, bed, warns = pair_all.pick([("fgr", T @ Fl, 0.9, 0.5)], Pi, Pj, beds_i, beds_j, "1-2")
        self.assertTrue(sup["flag"]); self.assertTrue(any("flagged" in w for w in warns)); self.assertIsNone(bed)


class FindBedTests(unittest.TestCase):
    def _bed(self, rng):
        M = random_map(rng); return M, B.BedModel(dots=np.column_stack([M, np.full(len(M), 10), np.full(len(M), 0.5)]),
                                                  radius_mm=97.0, lift_mm=0.8)

    def test_find_bed_places_the_platter(self):
        rng = np.random.default_rng(6); M, bed = self._bed(rng)
        for ring in (True, False):
            Bc = camera_bed(rng); ir, d = render(Bc, M, rng, ring_depth=ring, glints=3)
            h = B.find_bed(d, ir, CAL, bed)
            self.assertIsNotNone(h)
            self.assertLess(B.ang_deg(h.n, Bc[:3, 2]), 0.6)
            self.assertLess(abs(h.n @ Bc[:3, 3] + h.off), 0.6, "height from %s" % h.height_from)
            self.assertLess(np.linalg.norm(h.centre - Bc[:3, 3]), 2.0)
            self.assertLess(B.ang_deg(h.x_axis, Bc[:3, 0]), 1.0)
            self.assertEqual(h.height_from, "surface" if ring else "lift")
            self.assertLess(h.beneath, 0.01)

    def test_sidebed_accepts_after_three_hits_and_learner_learns(self):
        rng = np.random.default_rng(7); M, bed = self._bed(rng); Bc0 = camera_bed(rng)
        sb = B.SideBed(bed); lr = B.BedLearner(CAL, t0=0.0); n = Bc0[:3, 2]
        for k in range(90):
            Bc = Bc0.copy(); Bc[:3, :3] = rot3(n, np.radians(6.0 * k)) @ Bc0[:3, :3]      # the table turns 6 deg a frame
            ir, d = render(Bc, M, rng, glints=2)
            if k < 6:
                ev = sb.add_hit(0, k, B.find_bed(d, ir, CAL, bed, prior=sb.prior_for(0)), np.eye(4))
                if k == 2: self.assertTrue(sb.found, ev)
            st = lr.add(d, ir, t=k / 3.0)
            if st["state"] in ("done", "failed"): break
        self.assertEqual(st["state"], "done", st)
        got = lr.result()
        r = B.register_maps(bed.dots, got.dots)
        self.assertGreaterEqual(r.inliers, 20); self.assertLess(r.rms, 1.0)
        self.assertLess(np.linalg.norm(r.t), 2.0)                             # spin centre found within 2 mm
        self.assertLess(abs(got.lift_mm - 0.8), 0.4)

    def test_dark_face_near_the_platter_does_not_lift_it(self):
        """A dark face of the part 1.5-3.5 mm over the platter pulled the surface refit up to it (fix 6): the refit must
        stay with the dots' own plane."""
        rng = np.random.default_rng(11); M, bed = self._bed(rng); worst = 0.0
        for h in (1.8, 2.5, 3.2):
            for trial in range(3):
                Bc = camera_bed(rng); ir, d = render(Bc, M, rng, glints=0)
                d[(ir < 200) & (rng.random(d.shape) < 0.85)] = 0           # the dark platter gives little depth, the face a lot
                ir, d = paste_face(Bc, ir, d, rng, h=h, R=55.0)
                for prior in (None, (Bc[:3, 2], -Bc[:3, 2] @ Bc[:3, 3])):
                    hit = B.find_bed(d, ir, CAL, bed, prior=prior)
                    if hit is None: continue
                    worst = max(worst, abs(hit.n @ Bc[:3, 3] + hit.off))
        self.assertLess(worst, 1.0, "the plane followed the face (%.2f mm)" % worst)

    def test_noisy_platter_depth_does_not_sink_the_plane(self):
        """Depth noise spreads the platter's heights; the lowest dense layer then sat under the platter (round 1: 1.2 mm
        at 1 mm noise, 2.4 mm at 1.5; real platters show up to 1.2 mm). The refit must stay on it (fix F)."""
        rng = np.random.default_rng(3); M = random_map(rng)
        bed = B.BedModel(dots=np.column_stack([M, np.full(len(M), 10), np.full(len(M), 0.5)]), radius_mm=97.0, lift_mm=0.8)
        for sig in (0.6, 1.0, 1.5):
            r2 = np.random.default_rng(100); errs = []
            for trial in range(4):
                Bc = camera_bed(r2); ir, d = render(Bc, M, r2)
                plat = (d > 0) & (ir < 200); d = d.astype(np.float64); d[plat] += r2.normal(0, sig * 10, plat.sum())
                h = B.find_bed(np.clip(np.round(d), 0, 65535).astype(np.uint16), ir, CAL, bed)
                if h is not None: errs.append(-(h.n @ Bc[:3, 3] + h.off))          # plane over the platter, mm
            self.assertEqual(len(errs), 4)
            self.assertLess(abs(np.median(errs)), 0.3, "noise %.1f mm: plane %+.2f mm off" % (sig, np.median(errs)))

    def test_prior_only_hits_do_not_accept_a_plane(self):
        """A hit whose height is only the prior's (no dot depth) must not count toward the 3 agreeing hits (fix 7)."""
        rng = np.random.default_rng(12); M, bed = self._bed(rng); Bc = camera_bed(rng)
        ir, d = render(Bc, M, rng, ring_depth=False)
        d[ir > 200] = 0                                                   # no depth on the dots, none around them
        prior = dict(n=Bc[:3, 2] + [0, 0.004, 0], off=-Bc[:3, 2] @ Bc[:3, 3] - 2.0, centre=Bc[:3, 3])
        hit = B.find_bed(d, ir, CAL, bed, prior=prior)
        self.assertIsNotNone(hit); self.assertEqual(hit.height_from, "prior-only")
        sb = B.SideBed(bed)
        for k in range(6): sb.add_hit(0, k, B.find_bed(d, ir, CAL, bed, prior=prior), np.eye(4))
        self.assertEqual(sb.state, "looking"); self.assertIsNone(sb.camera_bed(0))

    def test_scanner_moved_mid_pass_drops_the_camera_bed(self):
        """The camera bed holds only while the scanner stays put: moved mid-pass, it is dropped and found again at the
        new station, with no V4 veto (fix 10). Scan frame = bed frame (the part sits still on the table)."""
        rng = np.random.default_rng(13); M, bed = self._bed(rng); Bc0 = camera_bed(rng); n = Bc0[:3, 2]
        G = np.eye(4); G[:3, :3] = rot3([1, 0.3, 0], np.radians(-4.0)); G[:3, 3] = [-10.0, 25.0, 20.0]  # the scanner moved
        sb = B.SideBed(bed); drops_at = []; refound = None
        for k in range(40):
            Bc = Bc0.copy(); Bc[:3, :3] = rot3(n, np.radians(4.0 * k)) @ Bc0[:3, :3]; Bc[:3, 3] = Bc0[:3, 3]
            if k >= 15: Bc = np.linalg.inv(G) @ Bc                              # camera -> bed changes by G
            T = np.linalg.inv(Bc)                                                # tracked pose, camera -> scan (= bed)
            if sb.check_camera(0, k, T): drops_at.append(k)
            if sb.camera_bed(0) is None or k % 10 == 0:
                ir, d = render(Bc, M, rng)
                had = sb.camera_bed(0) is not None
                sb.add_hit(0, k, B.find_bed(d, ir, CAL, bed, prior=sb.prior_for(0)), T)
                if not had and sb.camera_bed(0) is not None and k > 15: refound = k
        self.assertEqual(drops_at, [17], drops_at)                                # 3 disagreeing tracked frames: 15, 16, 17
        self.assertIsNotNone(refound); self.assertLessEqual(refound, 22)
        cam = sb.camera_bed(0); Bc = np.linalg.inv(G) @ Bc0
        self.assertTrue(sb.found); self.assertEqual(sb.vetoes, [])
        self.assertLess(B.ang_deg(cam["n"], Bc[:3, 2]), 0.6); self.assertLess(abs(Bc[:3, 2] @ cam["centre"] - Bc[:3, 2] @ Bc[:3, 3]), 1.0)

    def test_learner_drops_specks_and_keeps_its_radius(self):
        """result() keeps only dots seen about as often as a typical dot (a speck seen 4 times is not one), and a dark
        surface around the platter at its height does not stretch the radius (fix 8)."""
        rng = np.random.default_rng(14); M, bed = self._bed(rng); Bc0 = camera_bed(rng); n = Bc0[:3, 2]
        lr = B.BedLearner(CAL, t0=0.0)
        g = np.stack(np.meshgrid(np.arange(-85, 86, 2.0), np.arange(-85, 86, 2.0)), -1).reshape(-1, 2)
        g = g[(np.hypot(*g.T) > 50) & (np.hypot(*g.T) < 85)]
        speck = g[[np.argmax(np.linalg.norm(g[:, None] - M[None], axis=2).min(1))]]     # as far from every dot as it gets
        seen = []
        for k in range(90):
            Bc = Bc0.copy(); Bc[:3, :3] = rot3(n, np.radians(6.0 * k)) @ Bc0[:3, :3]
            X = Bc[:3, :3] @ np.array([speck[0, 0], speck[0, 1], 0.0]) + Bc[:3, 3]
            u, v = X[0] / X[2] * CAL[0] + CAL[2], X[1] / X[2] * CAL[1] + CAL[3]
            if len(seen) < 4 and k >= 8 and 20 < u < 780 and 20 < v < 580: seen.append(k)
            Mk = np.vstack([M, speck]) if k in seen else M                             # a glint seen in 4 frames
            ir, d = render(Bc, Mk, rng, radius=140.0 if k % 2 else 97.0)             # a dark table top out to 140 mm
            st = lr.add(d, ir, t=k / 3.0)
            if st["state"] in ("done", "failed"): break
        got = lr.result(); self.assertIsNotNone(got, st); self.assertEqual(len(seen), 4)
        r = B.register_maps(bed.dots, got.dots)
        self.assertEqual(len(got), r.inliers, "a speck was kept")
        self.assertLess(got.radius_mm, np.linalg.norm(M, axis=1).max() + 3.75 + 3 + 15.5)

    def test_learner_says_the_dots_were_lost(self):
        """Frames whose dots never match: the message says tracking of the dots failed, not that the table stopped."""
        rng = np.random.default_rng(15); Bc0 = camera_bed(rng); lr = B.BedLearner(CAL, t0=0.0); st = None
        for k in range(40):
            ir, d = render(Bc0, random_map(rng), rng)
            st = lr.add(d, ir, t=k * (B.LEARN_TIMEOUT + 5) / 39)
            if st["state"] in ("done", "failed"): break
        self.assertEqual(st["state"], "failed", st); self.assertIn("Lost track of the table dots", st["msg"])


class LearnerCheckTests(unittest.TestCase):
    def _learn(self, rng, M, Bc0, frames=90, step_deg=6.0, dt=1 / 3.0, cover=()):
        lr = B.BedLearner(CAL, t0=0.0); n = Bc0[:3, 2]; st = None
        for k in range(frames):
            Bc = Bc0.copy(); Bc[:3, :3] = rot3(n, np.radians(step_deg * k)) @ Bc0[:3, :3]
            ir, d = render(Bc, M, rng)
            if k in cover: ir = np.zeros_like(ir)                                   # the dots not seen in this frame
            st = lr.add(d, ir, t=k * dt)
            if st["state"] in ("done", "failed"): break
        return lr, st

    def test_a_smeared_map_is_refused(self):
        """Frames placed at the wrong angle smear the dots into many clusters seen a few times each (side 210644 with the
        part taken off: 74 dots for 29, and round 1 installed it). The map check refuses such a map, and passes a good one."""
        rng = np.random.default_rng(23); M = random_map(rng); Bc0 = camera_bed(rng)
        lr, st = self._learn(rng, M, Bc0)
        self.assertEqual(st["state"], "done", st); self.assertIsNone(lr._check_map(), lr.map_check)
        sm = B.BedLearner(CAL, t0=0.0); sm.plane, sm.e, sm.view = lr.plane, lr.e, lr.view; sm.frames = lr.frames
        c = lr.c_est; drift = 0.0; r = np.random.default_rng(24)
        for fno, F2, R, t in lr.obs:                  # the same frames, each placed a little further off (a random walk)
            drift += np.radians(r.normal(0, 5.0)); G = rot2(drift)
            R2 = G @ R; t2 = G @ (t - c) + c
            sm.obs.append((fno, F2, R2, t2)); sm._add_dots(F2 @ R2.T + t2)
        msg = sm._check_map()
        self.assertIsNotNone(msg, sm.map_check); self.assertIn("did not line up", msg)

    def test_learner_says_why_the_turn_was_not_finished(self):
        """Timeout before a full turn: a table that turned while some frames lost the dots is not told to switch the
        table on (round 1 said "Is it switched on?" to a table that turned 301 deg with 11% of the frames lost); a table
        that stood still is."""
        rng = np.random.default_rng(25); M = random_map(rng); Bc0 = camera_bed(rng)
        cover = set(range(8, 80, 9))                                            # 11% of the frames show no dots
        dt = (B.LEARN_TIMEOUT + 5) / 79
        lr, st = self._learn(rng, M, Bc0, frames=80, step_deg=3.0, dt=dt, cover=cover)
        self.assertEqual(st["state"], "failed", st); self.assertNotIn("switched on", st["msg"]); self.assertIn("followed", st["msg"])
        lr, st = self._learn(rng, M, Bc0, frames=80, step_deg=0.0, dt=dt)
        self.assertEqual(st["state"], "failed", st); self.assertIn("switched on", st["msg"])


class HintTests(unittest.TestCase):
    def _sb(self):
        sb = B.SideBed(B.BedModel(dots=np.zeros((10, 4))))
        sb.state = "found"; sb.fixed = dict(n=np.array([0, 0, 1.0]), off=0.0, centre=np.zeros(3), x_axis=np.array([1.0, 0, 0]), B_scan=np.eye(4))
        return sb

    def _refused(self, sb, ps, O, T_hint):
        """One hinted frame whose fit was found and refused: the fit put the camera at O @ T_hint (mm; the tracker's
        result is in metres)."""
        sb.last_hint = (T_hint.copy(), T_hint.copy())                           # what hint_pose gave for this frame (no move yet)
        Tf = O @ T_hint; Tf = Tf.copy(); Tf[:3, 3] /= 1000.0
        return sb.hint_outcome(dict(ok=False, hinted=True, refused=True, refused_T=Tf), ps)

    @staticmethod
    def _pose(rng):
        T = np.eye(4); T[:3, :3] = rot3(rng.normal(size=3), rng.uniform(0, np.pi)); T[:3, 3] = rng.uniform(-300, 300, 3); return T

    def test_frames_too_thin_for_icp_never_stop_the_hints(self):
        """A depth dropout: hinted frames with no fit at all (too little depth for ICP) say nothing about the part. Round 1
        gave up after 3 of them (6 of 7 dropout replays), then locked the wrong way round (fix C)."""
        sb = self._sb(); rng = np.random.default_rng(20)
        for k in range(20):
            T = self._pose(rng); sb.last_hint = (T, T)
            self.assertIsNone(sb.hint_outcome(dict(ok=False, hinted=True, refused=False, refused_T=None), 0))
        self.assertFalse(sb.hints_off(0)); self.assertIsNone(sb.moved(0))

    def test_agreeing_refused_fits_move_the_hints_for_this_pass_only(self):
        """The part turned 30 deg and slid 15 mm on the table: two refused fits (from views the table turned between)
        that agree on that move make the hints of this pass come from where the part now sits, guard still on; never
        saved, and the next pass starts from the dots' own placement again."""
        sb = self._sb(); rng = np.random.default_rng(21)
        O = np.eye(4); O[:3, :3] = rot3([0, 0, 1], np.radians(30)); O[:3, 3] = [15.0, 4.0, 0.3]      # along the table (n = z)
        self.assertIsNone(self._refused(sb, 0, O, self._pose(rng)))
        self.assertIsNone(sb.hint_outcome(dict(ok=False, hinted=True, refused=False), 0))          # a thin frame in between: no effect
        O2 = O.copy(); O2[:3, 3] += [2.0, -3.0, 0.0]; O2[:3, :3] = rot3([0, 0, 1], np.radians(2)) @ O2[:3, :3]
        # default (HINT_ON_MOVE "pause"): the move is reported for the window to pause the pass; the hints are NOT moved
        # (following a guessed move fused wrong frames in replays) and stay on with their guard
        sbp = self._sb(); self._refused(sbp, 0, O, self._pose(rng))
        m0 = self._refused(sbp, 0, O2, self._pose(rng))
        self.assertIsNotNone(m0); self.assertIn("scanning paused", m0); self.assertEqual(sbp.move_pause, 0)
        self.assertIsNone(sbp.moved(0)); self.assertFalse(sbp.hints_off(0))
        sbp.new_pass(1); self.assertIsNone(sbp.moved(1))
        B.HINT_ON_MOVE = "follow"; B.HINT_FOLLOW = True                          # the follow mode, still available
        msg = self._refused(sb, 0, O2, self._pose(rng))
        self.assertIsNotNone(msg); self.assertIn("moved on the table", msg)
        self.assertLess(np.abs(sb.moved(0) - O2).max(), 1e-9); self.assertFalse(sb.hints_off(0))
        Bc = np.eye(4); Bc[:3, :3] = rot3([1, 0, 0], 2.5); Bc[:3, 3] = [30.0, 50.0, 260.0]           # where this frame's dots put the bed
        hit = B.Hit(n=Bc[:3, 2], off=-Bc[:3, 2] @ Bc[:3, 3], centre=Bc[:3, 3], x_axis=Bc[:3, 0], B_cam=Bc)
        fb = B.find_bed; B.find_bed = lambda *a, **k: hit
        try: Th = sb.hint_pose(None, None, CAL, 0)
        finally: B.find_bed = fb
        want = O2 @ sb.fixed["B_scan"] @ np.linalg.inv(Bc); want[:3, 3] /= 1000.0
        self.assertLess(np.abs(Th - want).max(), 1e-9)                          # the hint follows the part (metres)
        self.assertNotIn("hint_off", sb.to_json())                              # nothing of it is saved
        sb.new_pass(1); self.assertIsNone(sb.moved(1))                          # the next pass starts from the dots again
        # a fit in between starts over: refused, fit, refused -> no move taken
        sb2 = self._sb(); self._refused(sb2, 0, O, self._pose(rng)); sb2.hint_outcome(dict(ok=True, hinted=True), 0)
        self.assertIsNone(self._refused(sb2, 0, O2, self._pose(rng))); self.assertIsNone(sb2.moved(0))
        # the "stop" mode: the hints stop for the pass
        sb3 = self._sb(); B.HINT_ON_MOVE = "stop"; B.HINT_FOLLOW = False
        try:
            self._refused(sb3, 0, O, self._pose(rng)); self.assertIsNotNone(self._refused(sb3, 0, O2, self._pose(rng)))
            self.assertTrue(sb3.hints_off(0)); self.assertIsNone(sb3.hint_pose(None, None, CAL, 0))
        finally: B.HINT_ON_MOVE = "pause"; B.HINT_FOLLOW = False

    def test_refused_fits_that_disagree_or_turn_the_platter_over_keep_the_hints(self):
        """Refused fits that put the part in different places, or turn the platter over (a near-symmetric part matched
        the wrong way round: 170-179 deg seen on side 002535), are no evidence that the part moved on the table."""
        sb = self._sb(); rng = np.random.default_rng(22)
        for k in range(8):                                                       # a different move every time
            O = np.eye(4); O[:3, :3] = rot3([0, 0, 1], np.radians(25 * k + 10)); O[:3, 3] = [5.0 * k, -20.0, 0.0]
            self.assertIsNone(self._refused(sb, 0, O, self._pose(rng)))
        W = np.eye(4); W[:3, :3] = rot3([1, 0.2, 0], np.radians(176)); W[:3, 3] = [3.0, 1.0, 40.0]   # upside down, the same every time
        for k in range(6): self.assertIsNone(self._refused(sb, 0, W, self._pose(rng)))
        self.assertIsNone(sb.moved(0)); self.assertFalse(sb.hints_off(0))

    def test_refused_fits_from_nearly_the_same_view_are_not_enough(self):
        """ICP stopped part way from the hint lands in nearly the same wrong place in the next frame too (8 and 11 deg for
        a 30 deg move, frames 0 and 1 of side 233432; round 2's first try gave up on them and fused 18 frames 13 deg off).
        Agreeing fits must come from views the table turned at least HINT_APART_DEG between."""
        sb = self._sb(); rng = np.random.default_rng(23); T0 = self._pose(rng)
        O = np.eye(4); O[:3, :3] = rot3([0, 0, 1], np.radians(10)); O[:3, 3] = [8.0, 9.0, 0.0]
        turn = lambda deg: T0 @ np.block([[rot3([0.1, 1, 0.2], np.radians(deg)), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
        self.assertIsNone(self._refused(sb, 0, O, turn(0.0)))
        self.assertIsNone(self._refused(sb, 0, O, turn(3.3)))                  # the next frame: too close to the first
        self.assertIsNone(sb.moved(0))
        self.assertIsNotNone(self._refused(sb, 0, O, turn(12.0)))              # the table turned on: now it counts

    def test_signature_changes_when_a_plane_is_accepted(self):
        rng = np.random.default_rng(16); M = random_map(rng)
        bed = B.BedModel(dots=np.column_stack([M, np.full(len(M), 10), np.full(len(M), 0.5)]), radius_mm=97.0)
        sb = B.SideBed(bed); Bc = camera_bed(rng); s0 = sb.signature()
        for k in range(3):
            ir, d = render(Bc, M, rng); sb.add_hit(0, k, B.find_bed(d, ir, CAL, bed), np.eye(4))
        self.assertTrue(sb.found); self.assertNotEqual(sb.signature(), s0)


class WindowHelperTests(unittest.TestCase):
    def test_new_project_keeps_the_options(self):
        import pointyoink_scanner as P
        self.assertEqual(P.restart_argv(["--root", "/r", "--project", "/r/P1", "--near", "80"]), ["--root", "/r", "--near", "80"])
        self.assertEqual(P.restart_argv(["--last", "--param", "/x/param", "--far=500"]), ["--param", "/x/param", "--far=500"])
        self.assertEqual(P.restart_argv(["--project=/r/P1", "--root=/r"]), ["--root=/r"])
        self.assertEqual(P.restart_argv(["--proj", "/r/P1", "--la", "--root", "/r"]), ["--root", "/r"])

    def test_raw_frames_do_not_pile_up(self):
        import queue
        import pointyoink_scanner as P
        q = queue.Queue()
        for i in range(50): P.put_frame(q, i, recording=False)
        self.assertEqual(list(q.queue), [46, 47, 48, 49])                        # not recording: the newest few
        q = queue.Queue()
        for i in range(P.RAWQ_REC + 20): P.put_frame(q, i, recording=True)
        self.assertEqual(q.qsize(), P.RAWQ_REC); self.assertEqual(q.queue[0], 20)  # recording: bounded all the same


if __name__ == "__main__":
    unittest.main()
