"""Manual edits of the staged build (scanner/stages/tools/edits.py) on synthetic geometry: delete replay (T5), re-anchoring
when the model frame moves (T6), flip and nudge matrices (T7)."""
import os, sys, tempfile, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner"))

from stages.tools import edits as E        # noqa: E402


def plates(step, jitter=0.0, seed=0, gap=1.5):
    """Two parallel 40 x 40 mm plates gap mm apart. The 0.4 mm grid puts the 10 mm patch edge (x = +-5) midway between
    two samples, so the recorded selection is the square itself (a replay can only place an edge to half a spacing)."""
    g = np.arange(-20, 20, step); X, Y = np.meshgrid(g, g); n = X.size
    rng = np.random.default_rng(seed)
    top = np.stack([X.ravel(), Y.ravel(), np.full(n, gap)], 1); bot = np.stack([X.ravel(), Y.ravel(), np.zeros(n)], 1)
    if jitter:
        top = top + rng.normal(0, jitter, top.shape); bot = bot + rng.normal(0, jitter, bot.shape)
    return np.vstack([top, bot]), n


def patch(P, n, half=5.0):
    m = np.zeros(len(P), bool); m[:n] = (np.abs(P[:n, 0]) < half) & (np.abs(P[:n, 1]) < half); return m


def rigid(seed=3):
    rng = np.random.default_rng(seed); q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
    R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = rng.normal(0, 40, 3); return T


def apply(T, P): return P @ T[:3, :3].T + T[:3, 3]


class T5DeleteReplay(unittest.TestCase):
    def setUp(self):
        self.stages = tempfile.mkdtemp(prefix="stages_edits_")
        P0, n0 = plates(0.4)
        sel = patch(P0, n0)
        reg = E.make_region(P0, sel, "b2:old")
        self.op = {"tool": "delete", "frame": "model", "shape": "mask", "region": E.save_region(os.path.join(self.stages, "regions"), reg),
                   "radius": 0.1, "recorded": {"input_digest": "b2:old", "count": int(sel.sum())}}
        self.n_sel = int(sel.sum())

    def test_exact_when_unchanged(self):
        P0, n0 = plates(0.4)
        m, how = E.replay_mask(P0, self.op, "b2:old", None, self.stages)
        self.assertEqual(how, "exact"); self.assertTrue((m == patch(P0, n0)).all())

    def test_geometric_replay(self):
        P1, n1 = plates(0.3, jitter=0.05, seed=1)
        truth = patch(P1, n1)
        m, how = E.replay_mask(P1, self.op, "b2:new", None, self.stages)
        self.assertEqual(how, "geometric")
        got = (m & truth).sum() / truth.sum()
        self.assertGreaterEqual(got, 0.95, "removed %.3f of the top patch" % got)
        self.assertEqual(int(m[n1:].sum()), 0, "bottom plate points removed")
        self.assertLess((m & ~truth).sum(), 0.05 * truth.sum())

    def test_box_sphere_exact(self):
        P, n = plates(0.4)
        op = {"shape": "box", "box": {"min": [-5, -5, 1.0], "max": [5, 5, 2.0], "T": None}}
        m = E.select_shape(P, op)
        self.assertEqual(int(m.sum()), int(patch(P, n).sum())); self.assertEqual(int(m[n:].sum()), 0)
        op = {"shape": "sphere", "center": [0.0, 0.0, 1.5], "radius": 1.2}
        m = E.select_shape(P, op)
        self.assertEqual(int(m.sum()), int((np.linalg.norm(P - [0, 0, 1.5], axis=1) <= 1.2).sum()))
        self.assertEqual(int(m[n:].sum()), 0)
        # oriented box: rotated 90 deg about z, same set
        Tb = np.eye(4); Tb[:3, :3] = E.rot([0, 0, 1], 90.0)
        op = {"shape": "box", "box": {"min": [-5, -5, 1.0], "max": [5, 5, 2.0], "T": Tb.tolist()}}
        self.assertEqual(int(E.select_shape(P, op).sum()), int(patch(P, n).sum()))


class T6Anchor(T5DeleteReplay):
    def test_reanchored(self):
        M = rigid()
        self.op["anchor"] = {"order": ["gone", "A", "B"], "T": {"A": np.eye(4).tolist(), "B": np.eye(4).tolist(), "gone": np.eye(4).tolist()}}
        P1, n1 = plates(0.3, jitter=0.05, seed=2); truth = patch(P1, n1)
        Tnow = {"A": M.tolist(), "B": M.tolist()}                    # 'gone' left out: the next active id anchors
        m, how = E.replay_mask(apply(M, P1), self.op, "b2:new", Tnow, self.stages)
        self.assertGreaterEqual((m & truth).sum() / truth.sum(), 0.95); self.assertEqual(int(m[n1:].sum()), 0)
        op = {"shape": "sphere", "center": [0.0, 0.0, 1.5], "radius": 1.2, "frame": "model", "anchor": self.op["anchor"]}
        want = np.linalg.norm(P1 - [0, 0, 1.5], axis=1) <= 1.2
        m2, _ = E.replay_mask(apply(M, P1), op, "b2:new", Tnow, self.stages)
        self.assertEqual(int((m2 != want).sum()), 0)
        op = {"shape": "box", "box": {"min": [-5, -5, 1.0], "max": [5, 5, 2.0], "T": None}, "frame": "model", "anchor": self.op["anchor"]}
        m3, _ = E.replay_mask(apply(M, P1), op, "b2:new", Tnow, self.stages)
        want = (np.abs(P1[:, 0]) <= 5) & (np.abs(P1[:, 1]) <= 5) & (P1[:, 2] >= 1.0) & (P1[:, 2] <= 2.0)
        self.assertEqual(int((m3 != want).sum()), 0)

    def test_drift_check(self):
        P0, n0 = plates(0.4)
        op = dict(self.op, recorded={"input_digest": "b2:old", "count": self.n_sel * 3})
        job = {"op": op, "parent_digest": "b2:other", "T_now": None, "stages": self.stages}
        data = {"P": P0.astype(np.float32), "N": np.zeros_like(P0, np.float32), "W": np.ones(len(P0), np.float32),
                "NF": np.ones(len(P0), np.float32), "S": np.ones(len(P0), np.uint64), "side_ids": np.array(["A"]), "cell": 0.4}
        new, sel, st = E.delete_job(job, "points", data)
        self.assertTrue(st.get("check")); self.assertEqual(len(new["P"]), len(P0) - int(sel.sum()))


def l_cloud(seed=0):
    rng = np.random.default_rng(seed)
    a = np.stack([rng.uniform(0, 60, 4000), rng.uniform(0, 10, 4000), rng.uniform(0, 2, 4000)], 1)      # long arm
    b = np.stack([rng.uniform(0, 8, 2000), rng.uniform(0, 30, 2000), rng.uniform(0, 14, 2000)], 1)      # short upright arm
    return np.vstack([a, b])


class T7FlipNudge(unittest.TestCase):
    def test_flip_twice_identity(self):
        S = l_cloud(); c, ax = E.pca_frame(S)
        for k in range(3):
            Fl = E.flip_matrix(c, ax[:, k])
            self.assertLess(np.abs(Fl @ Fl - np.eye(4)).max(), 1e-9)

    def test_auto_axis(self):
        S = l_cloud(); c, ax = E.pca_frame(S); Ttrue = rigid(5)
        others = apply(Ttrue, S)                      # the reference: the same side placed right
        for k, name in enumerate("xyz"):
            Twrong = Ttrue @ E.flip_matrix(c, ax[:, k])
            Fl, got, sc = E.choose_flip(S, Twrong, others)
            self.assertEqual(got, name, sc)
            self.assertLess(np.abs(Twrong @ Fl - Ttrue).max(), 1e-6)

    def test_nudge(self):
        c = np.array([10.0, -4.0, 30.0]); t = [1.5, -2.0, 0.25]; r = [0.0, 0.0, 30.0]
        D = E.nudge_matrix(c, t, r)
        self.assertLess(np.abs(D[:3, :3] @ c + D[:3, 3] - (c + np.array(t))).max(), 1e-9)       # about the centroid
        Dinv = E.nudge_matrix(c + np.array(t), [-x for x in t], [0.0, 0.0, -30.0])
        self.assertLess(np.abs(Dinv @ D - np.eye(4)).max(), 1e-9)
        D0 = E.nudge_matrix(c, [0, 0, 0], [0, 0, 0]); self.assertLess(np.abs(D0 - np.eye(4)).max(), 1e-12)


def l_prism(h=25.0):
    """A closed L-shaped prism (mm), outward faces: the part the synthetic sides are cut from."""
    poly = np.array([(0, 0), (60, 0), (60, 15), (20, 15), (20, 40), (0, 40)], float); n = len(poly)
    V = np.vstack([np.c_[poly, np.zeros(n)], np.c_[poly, np.full(n, h)]]); F = []
    for i in range(n):
        j = (i + 1) % n; F += [(i, j, n + j), (i, n + j, n + i)]
    for a, b, c in ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)):      # the L as a fan from (0, 0): all inside it
        F += [(n + a, n + b, n + c), (a, c, b)]
    return V, np.array(F)


DIRS = [(0.4, 0.3, 1), (-0.4, -0.3, -1), (1, -0.4, 0.3), (-1, 0.4, 0.3)]


def box_sides(dirs=DIRS, seed=11):
    """Each side: the faces seen from one direction, in its own scan frame (true side->model pose G)."""
    from stages import plyio
    V, F = l_prism(); Nf = plyio.face_normals(V, F); out = []
    for k, d in enumerate(dirs):
        d = np.asarray(d, float) / np.linalg.norm(d); f = F[Nf @ d > 0.1]
        used = np.unique(f); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
        G = rigid(seed + k); out.append((apply(np.linalg.inv(G), V[used]), remap[f], G))
    return out


def moved_by(Ta, Tb):
    return E.moved(Ta, Tb, np.array([30.0, 20.0, 12.0]))


class T8FitFlipMatch(unittest.TestCase):
    """Flips and point matches land by a local fit and stay put when the placement around them changes."""
    @classmethod
    def setUpClass(cls):
        from stages import plyio
        cls.d = tempfile.mkdtemp(prefix="stages_fit_"); cls.sides = box_sides(); cls.meshes = {}
        for k, (V, F, G) in enumerate(cls.sides):
            p = os.path.join(cls.d, "s%d.ply" % k); plyio.write_mesh(p, V, F); cls.meshes["s%d" % k] = p
        cls.G = {"s%d" % k: s[2] for k, s in enumerate(cls.sides)}
        cls.samp = staticmethod(E._sampler({"meshes": cls.meshes}))

    def placement(self, T, name="pl.json"):
        p = os.path.join(self.d, name); order = list(T)
        E.jdump({"base": order[0], "order": order, "T": {k: np.asarray(v).tolist() for k, v in T.items()}, "score": {}, "flagged": []}, p)
        return p

    def fix(self, T, ops, meshes=None):
        out = tempfile.mkdtemp(dir=self.d)
        r = E.fix_job({"placement": self.placement(T), "flips": ops, "meshes": meshes or self.meshes, "out_dir": out})
        return {k: np.array(v) for k, v in E.jload(os.path.join(out, "placement.json"))["T"].items()}, r

    def wrong_flip(self, s, k):
        P, _N = self.samp(s); c, ax = E.pca_frame(P)
        return self.G[s] @ E.flip_matrix(c, ax[:, k])

    def test_fit_flip_auto_recovers(self):
        """Auto tries the three flips, fits each and keeps the best: the upside-down side lands where it belongs (the bottom
        face no other side sees caps its share near 0.6, as it does in the right pose)."""
        P, N = self.samp("s1"); right = E.fit_score(*E.probe(P, N), self.G["s1"], E._target(self.G, self.samp, "s1"))[0]
        for k in range(3):
            T = dict(self.G); T["s1"] = self.wrong_flip("s1", k)
            r = E.resolve_flip_job({"T": {a: b.tolist() for a, b in T.items()}, "side": "s1", "axis": "auto", "meshes": self.meshes})
            mm, deg = moved_by(self.G["s1"], np.array(r["T"]))
            self.assertLess(mm, 0.5, r["scores"]); self.assertLess(deg, 0.5, r["scores"]); self.assertGreater(r["share"], right - 0.02)

    def test_flip_survives_placement_change(self):
        """A flip kept against its anchors: base change (every pose times one rigid), refit (another side shifted) and a
        leave-out of its best anchor all leave the flipped side where it was (0.5 mm / 0.5 deg)."""
        T = dict(self.G); T["s1"] = self.wrong_flip("s1", 1)
        fit = E.resolve_flip_job({"T": {a: b.tolist() for a, b in T.items()}, "side": "s1", "axis": "auto", "meshes": self.meshes})
        op = {"tool": "flip", "side": "s1", "axis": "auto", "chose": fit["axis"], "rel": fit["rel"]}
        A, _ = self.fix(T, [op])
        self.assertLess(moved_by(self.G["s1"], A["s1"])[0], 0.5)
        W = rigid(42); B, _ = self.fix({k: W @ v for k, v in T.items()}, [op])        # the base changed
        mm, deg = moved_by(W @ A["s1"], B["s1"]); self.assertLess(mm, 0.5); self.assertLess(deg, 0.5)
        T2 = dict(T); T2["s1"] = self.G["s1"] @ E.flip_matrix([0, 0, 0], [0, 0, 1])   # a refit placed it differently
        C, _ = self.fix(T2, [op]); self.assertLess(moved_by(A["s1"], C["s1"])[0], 0.5)
        first = fit["rel"][0][0]; T3 = {k: v for k, v in T.items() if k != first}
        D, _ = self.fix(T3, [op], {k: v for k, v in self.meshes.items() if k != first}); self.assertLess(moved_by(A["s1"], D["s1"])[0], 0.5)

    def test_flip_two_sides_is_fitted(self):
        """With 2 sides (refine is an alias) the flip still ends with a local fit."""
        T = {"s0": self.G["s0"], "s3": self.wrong_flip("s3", 0) @ E.tr([0.8, -0.6, 0.5])}
        m = {k: self.meshes[k] for k in T}
        A, r = self.fix(T, [{"tool": "flip", "side": "s3", "axis": "x"}], m)
        self.assertLess(moved_by(self.G["s3"], A["s3"])[0], 0.5, r)

    def pairs(self, s, to, idx=(0, 2, 7, 9), noise=0.4, seed=4):
        V, _F = l_prism(); rng = np.random.default_rng(seed); X = V[list(idx)] + rng.normal(0, noise, (len(idx), 3))
        return [[apply(np.linalg.inv(self.G[to]), x[None])[0].tolist(), apply(np.linalg.inv(self.G[s]), x[None])[0].tolist()] for x in X]

    def test_match_recovers_known_transform(self):
        T = dict(self.G); T["s2"] = rigid(77)                      # side 2 placed anywhere
        A, r = self.fix(T, [{"tool": "match", "side": "s2", "to": "s0", "pairs": self.pairs("s2", "s0")}])
        mm, deg = moved_by(self.G["s2"], A["s2"]); self.assertLess(mm, 0.5, r); self.assertLess(deg, 0.5, r)
        self.assertTrue(r["matched"]["s2"]["icp"]); self.assertLess(max(r["matched"]["s2"]["residuals"]), 2.0)
        self.assertGreater(r["fits"]["s2"]["share"], 0.8)
        W = rigid(9); B, _ = self.fix({k: W @ v for k, v in T.items()}, [{"tool": "match", "side": "s2", "to": "s0", "pairs": self.pairs("s2", "s0")}])
        self.assertLess(moved_by(W @ A["s2"], B["s2"])[0], 0.5)      # absolute: follows its "to" side, not the old pose

    def test_match_bad_picks_keep_points(self):
        """ICP that would run far from the picks is refused: the rigid fit to the points stays, with a warning."""
        pr = self.pairs("s2", "s0", noise=0.0); T = dict(self.G)
        far = {"s0": E.tr(self.G["s0"][:3, :3] @ [0, 0, 5.5]) @ self.G["s0"]}
        r = E.solve_match(pr, *self.samp("s2"), T["s0"], E._target(far, self.samp))     # the surface sits 5.5 mm off the picks
        self.assertFalse(r["icp"]); self.assertIn("moved", r["warn"]); self.assertLess(moved_by(self.G["s2"], r["T"])[0], 0.1)
        with self.assertRaises(ValueError): E.solve_match(pr[:2], *self.samp("s2"), T["s0"], E._target(T, self.samp, only=["s0"]))
        with self.assertRaises(ValueError): E.solve_match([pr[0], pr[0], pr[1]], *self.samp("s2"), T["s0"], E._target(T, self.samp, only=["s0"]))

    def test_match_left_out_to_side_warns(self):
        T = {k: v for k, v in self.G.items() if k != "s0"}
        A, r = self.fix(T, [{"tool": "match", "side": "s2", "to": "s0", "pairs": self.pairs("s2", "s0")}], {k: v for k, v in self.meshes.items() if k != "s0"})
        self.assertTrue(r["warn"]); self.assertEqual(r["matched"], {})


class T9LineupOps(unittest.TestCase):
    """The engine side of flips and matches (fake line-up methods, nothing fitted): ops, graph, alias, history, CLI words."""
    def setUp(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import test_stages_core as C
        self.C = C; C.install_fakes(); self.tmp = tempfile.mkdtemp(prefix="stages_ops_"); self.ids = C.make_project(self.tmp)
        from stages.project import Project
        from stages.engine import Engine
        import io
        self.e = Engine(Project.open(self.tmp), out=io.StringIO()); self.e._post = lambda *a, **k: None

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True); self.C.remove_fakes()

    def fix_params(self): return self.e.graph().by_id["lineup.fix"]

    def test_match_op(self):
        self.assertEqual(self.fix_params().alias, "placement")
        pr = [[[0, 0, 0], [1, 0, 0]], [[10, 0, 0], [11, 0, 0]], [[0, 10, 0], [1, 10, 0]]]
        self.e.match(3, 1, pr)
        st = self.fix_params(); self.assertIsNone(st.alias)
        (op,) = st.params["flips"]; self.assertEqual((op["tool"], op["side"], op["to"]), ("match", self.ids[2], self.ids[0]))
        self.assertEqual(op["pairs"], [[[float(x) for x in p] for p in q] for q in pr])
        _lk, chain, _p = self.e.ops_of("lineup.fix"); self.assertEqual(len(chain), 1)
        ent, cur = self.e.hist.entries(); self.assertIn("match side 3 to side 1 (3 points)", ent[cur]["note"])
        with self.assertRaises(ValueError): self.e.set_op("lineup.fix", 0, {"pairs": pr})
        with self.assertRaises(ValueError): self.e.match(3, 3, pr)
        with self.assertRaises(ValueError): self.e.match(3, 1, pr[:2])
        with self.assertRaises(ValueError): self.e.match(3, 1, [[[0, 0, 0], [1, 0, 0]], [[1, 1, 1], [2, 1, 1]], [[2, 2, 2], [3, 2, 2]]])   # collinear
        self.e.leave_out(1); self.assertEqual(self.fix_params().params["flips"], [op])   # kept: fix_job warns that its "to" is out
        self.e.leave_out(3); self.assertEqual(self.fix_params().params.get("flips"), [])
        self.e.undo(); self.e.undo(); self.e.undo(); self.assertEqual(self.fix_params().alias, "placement")

    def test_flip_keeps_fitted_pose(self):
        """A flip made on a built line-up stores the fitted pose against its anchors (rel) and the axis auto chose;
        before any line-up it is only the axis (fitted when the line-up runs); changing the axis drops the kept pose."""
        self.e.flip(2, "auto"); op = self.fix_params().params["flips"][0]; self.assertNotIn("rel", op)
        self.e._fit_flip = lambda sid, axis: {"axis": "y", "rel": [[self.ids[0], np.eye(4).tolist()]], "T": np.eye(4).tolist()}
        self.e.flip(2, "auto"); op = self.fix_params().params["flips"][1]
        self.assertEqual((op["chose"], op["rel"][0][0]), ("y", self.ids[0]))
        self.assertEqual(self.e.flip(2, "x", preview=True)["axis"], "y"); self.assertEqual(len(self.fix_params().params["flips"]), 2)
        self.e.set_op("lineup.fix", 1, {"axis": "z"}); op = self.fix_params().params["flips"][1]
        self.assertEqual(op["axis"], "z"); self.assertNotIn("rel", op)

    def test_turn_op(self):
        """Nudge's rotate rings: the turn is stored in the side's frame (M, c) with its kept pose (rel), keep toggles by
        set_op, undo takes it back; a line-up that is not built refuses it."""
        R = E.rot([0, 0, 1], 30.0)
        with self.assertRaises(ValueError): self.e.turn(2, R.tolist())                       # nothing built in the fakes
        M = E.tr([1, 2, 3]) @ E.hom(R) @ E.tr([-1, -2, -3])
        self.e._resolve_turn = lambda sid, RR: {"M": M.tolist(), "c": [1.0, 2.0, 3.0], "rel": [[self.ids[0], np.eye(4).tolist()]]}
        with self.assertRaises(ValueError): self.e.turn(2, np.eye(3).tolist())               # no turn at all
        with self.assertRaises(ValueError): self.e.turn(2, (2 * np.eye(3)).tolist())         # not a rotation
        self.e.turn(2, R.tolist(), keep=False)
        st = self.fix_params(); self.assertIsNone(st.alias); (op,) = st.params["flips"]
        self.assertEqual((op["tool"], op["side"], op["deg"], op["keep"], op["rel"][0][0]), ("turn", self.ids[1], 30.0, False, self.ids[0]))
        self.assertLess(np.abs(np.array(op["M"]) - M).max(), 1e-8)
        ent, cur = self.e.hist.entries(); self.assertIn("turn side 2 by 30.0 deg", ent[cur]["note"])
        self.e.set_op("lineup.fix", 0, {"keep": True}); self.assertTrue(self.fix_params().params["flips"][0]["keep"])
        with self.assertRaises(ValueError): self.e.set_op("lineup.fix", 0, {"deg": 10})
        self.e.undo(); self.assertFalse(self.fix_params().params["flips"][0]["keep"])
        self.e.undo(); self.assertEqual(self.fix_params().alias, "placement")
        self.e.redo(); self.assertEqual(self.fix_params().params["flips"][0]["tool"], "turn")

    def test_turn_with_move_is_one_undo(self):
        """Nudge's Apply with a turn and a move: the turn op and the nudge op land in ONE history entry, so one undo takes
        both back and one redo brings both."""
        R = E.rot([0, 0, 1], 20.0)
        M = E.tr([1, 2, 3]) @ E.hom(R) @ E.tr([-1, -2, -3])
        self.e._resolve_turn = lambda sid, RR: {"M": M.tolist(), "c": [1.0, 2.0, 3.0]}
        ops = lambda: (self.e.recipe().get("ops") or {})
        cur0 = self.e.hist.entries()[1]; fix0 = len(ops().get("lineup.fix", [])); adj0 = len(ops().get("lineup.adjust", []))
        self.e.turn(2, R.tolist(), keep=True, move=[1.5, 0, -2])
        ent, cur = self.e.hist.entries(); self.assertEqual(cur, cur0 + 1)
        self.assertIn("turn side 2 by 20.0 deg", ent[cur]["note"]); self.assertIn("nudge side 2", ent[cur]["note"])
        fix, adj = ops()["lineup.fix"], ops()["lineup.adjust"]
        self.assertEqual((len(fix), len(adj)), (fix0 + 1, adj0 + 1))
        self.assertEqual((fix[-1]["tool"], fix[-1]["keep"], adj[-1]["tool"], adj[-1]["t"]), ("turn", True, "nudge", [1.5, 0.0, -2.0]))
        self.assertNotEqual(fix[-1]["id"], adj[-1]["id"])
        self.e.undo(); self.assertEqual(self.e.hist.entries()[1], cur0)
        self.assertEqual((len(ops().get("lineup.fix", [])), len(ops().get("lineup.adjust", []))), (fix0, adj0))
        self.e.redo()
        self.assertEqual((len(ops()["lineup.fix"]), len(ops()["lineup.adjust"])), (fix0 + 1, adj0 + 1))
        self.e.undo()
        self.e.turn(2, R.tolist(), move=[0, 0, 0])                                            # no move: the turn alone
        self.assertEqual((len(ops()["lineup.fix"]), len(ops().get("lineup.adjust", []))), (fix0 + 1, adj0))
        self.e.undo()


class T10Turn(unittest.TestCase):
    """A turn from Nudge's rotate rings: kept in the side's own frame and against the placed sides, then a LOCAL fit from
    the user's pose (the user's pose wins when the fit runs off); Keep exactly as placed skips the fit."""
    setUpClass = T8FitFlipMatch.__dict__["setUpClass"]; placement = T8FitFlipMatch.placement; fix = T8FitFlipMatch.fix

    def world_turn(self, T, s, R):
        """D @ T[s]: R about the side's placed vertex mean (what the view previews)."""
        from stages import plyio
        c = apply(T[s], plyio.vertex_mean(self.meshes[s])[None])[0]
        return E.tr(c) @ E.hom(R) @ E.tr(-c) @ T[s]

    def made(self, T, s, R, keep=False, disp=None):
        r = E.resolve_turn_job({"T": {k: v.tolist() for k, v in T.items()}, "T_disp": (disp if disp is not None else T[s]).tolist(),
                                "side": s, "R": R.tolist(), "meshes": self.meshes})
        return {"tool": "turn", "side": s, "M": r["M"], "c": r["c"], "keep": keep, "rel": r["rel"]}

    def test_side_matrix_is_the_world_turn(self):
        T = rigid(5); c0 = np.array([3.0, -2.0, 7.0]); R = E.rot([1, 2, 0.5], 37.0)
        M = E.turn_side_matrix(R, T, c0); c = apply(T, c0[None])[0]
        self.assertLess(np.abs(T @ M - E.tr(c) @ E.hom(R) @ E.tr(-c) @ T).max(), 1e-9)
        self.assertLess(np.abs(apply(M, c0[None])[0] - c0).max(), 1e-9)                      # about its own centre

    def test_turn_snaps_in_and_survives_base_change(self):
        """Side 1 sits 25 deg off; the user turns it back 23 deg (2 deg short): the local fit lands it; a base change
        (every pose times one rigid) and a refit that placed it elsewhere keep it there."""
        s = "s1"; T = dict(self.G); T[s] = self.world_turn(self.G, s, E.rot([0.3, 1, 0.2], 25.0))
        op = self.made(T, s, E.rot([0.3, 1, 0.2], -23.0)); self.assertTrue(op["rel"])
        A, r = self.fix(T, [op])
        mm, deg = moved_by(self.G[s], A[s]); self.assertLess(mm, 0.5, r); self.assertLess(deg, 0.5, r)
        self.assertTrue(r["turned"][s]["fit"]); self.assertFalse(r["warn"])
        W = rigid(42); B, _ = self.fix({k: W @ v for k, v in T.items()}, [op])
        mm, deg = moved_by(W @ A[s], B[s]); self.assertLess(mm, 0.5); self.assertLess(deg, 0.5)
        T2 = dict(T); T2[s] = E.tr([30, 0, 0]) @ T[s]                                          # placed somewhere else now
        C, _ = self.fix(T2, [op]); self.assertLess(moved_by(A[s], C[s])[0], 0.5)
        no_rel = dict(op, rel=[]); D, _ = self.fix(T, [no_rel])                                 # made before any line-up
        self.assertLess(moved_by(self.G[s], D[s])[0], 0.5)

    def test_keep_skips_the_fit(self):
        s = "s1"; T = dict(self.G); T[s] = self.world_turn(self.G, s, E.rot([0.3, 1, 0.2], 25.0))
        R = E.rot([0.3, 1, 0.2], -23.0); want = self.world_turn(T, s, R)
        A, r = self.fix(T, [self.made(T, s, R, keep=True)])
        self.assertLess(np.abs(A[s] - want).max(), 1e-6); self.assertEqual(r["turned"][s], {"fit": False})
        self.assertAlmostEqual(moved_by(self.G[s], A[s])[1], 2.0, delta=0.05)                   # exactly the 2 deg it is short
        A2, _ = self.fix(T, [dict(self.made(T, s, R, keep=True), rel=[])])
        self.assertLess(np.abs(A2[s] - want).max(), 1e-6)

    def test_fit_is_seeded_from_the_user_pose(self):
        """fix_job hands local_icp the user's pose (never a search); a fit that runs off keeps that pose, with a warning."""
        s = "s1"; T = dict(self.G); R = E.rot([0, 0, 1], 3.0); want = self.world_turn(T, s, R); seen = []
        real = E.local_icp
        E.local_icp = lambda P, N, tgt, T0, *a, **k: (seen.append(np.array(T0)), real(P, N, tgt, T0, *a, **k))[1]
        try: self.fix(T, [self.made(T, s, R)])
        finally: E.local_icp = real
        self.assertLess(np.abs(seen[0] - want).max(), 1e-6)
        tgt = E._target({k: v for k, v in self.G.items() if k != s}, self.samp)
        off = self.world_turn(self.G, s, E.rot([0, 0, 1], 16.0))                               # the fit would turn it 16 deg
        r = E.fit_turn(*self.samp(s), off, tgt)
        self.assertIsNotNone(r["warn"], r); self.assertLess(np.abs(r["T"] - off).max(), 1e-12)
        ok = E.fit_turn(*self.samp(s), self.world_turn(self.G, s, R), tgt)
        self.assertIsNone(ok["warn"]); self.assertLess(moved_by(self.G[s], ok["T"])[1], 0.5)

    def test_turn_made_on_a_nudged_view(self):
        """The view shows the line-up after nudges (T_disp); the turn is taken about the side's centre there, and the
        side-frame matrix is the same turn on the line-up before them."""
        s = "s2"; T = dict(self.G); N = E.tr([4, -2, 1]); R = E.rot([1, 0, 0], 10.0)
        op = self.made(T, s, R, keep=True, disp=N @ T[s])
        self.assertLess(np.abs(N @ T[s] @ np.array(op["M"]) - self.world_turn({s: N @ T[s]}, s, R)).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
