"""fused.plus (fuse.py --mode plus + drop_enclosed.py) and build_project.py --fuse plus.

P1 registry, params, preset          P2 argv (flag mapping, smooth in mm)      P3 engine run and script digests in keys
P4 fallback to classic on failure    P5 drop_enclosed.py on synthetic meshes   P6 fuse.py --mode plus on the CPU (2 sides)
P7 build_project.py step 4 wiring (classic argv unchanged, plus, plus -> classic fallback)
P1-P4 and P7 simulate the child processes; P5 and P6 run the real scripts on tiny synthetic data (CPU, a few seconds)."""
import glob, json, os, shutil, struct, subprocess, sys, tempfile, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_stages_core import Base, CALLS, SAVED, wb                          # noqa: E402
from test_stages_fixes import inproc, use, plates_mesh                       # noqa: E402
from test_stages_fixes3 import cut_ok                                        # noqa: E402
from stages import params as PR                                             # noqa: E402
from stages import plyio                                                    # noqa: E402
from stages.runner import StepError                                         # noqa: E402
from stages.methods import get_method                                       # noqa: E402
from stages.methods import fuse_plus as FP                                  # noqa: E402

PY = sys.executable


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ---------------------------------------------------------------------------------------------------------- P1
class P1Registered(Base):
    def test_method_and_params(self):
        m = SAVED["methods"][("fused", "plus")]()
        self.assertEqual((m.node, m.name, m.produces), ("fused", "plus", "mesh"))
        files, libs = m.code({})
        names = [n for n, _p in files]
        for n in ("fuse.py", "drop_enclosed.py", "stages/methods/fuse_plus.py"): self.assertIn(n, names)
        self.assertTrue(os.path.exists(dict(files)["drop_enclosed.py"]))
        self.assertIn("open3d", libs); self.assertIn("scipy", libs); self.assertTrue(any(l.startswith("argv=") for l in libs))
        p = PR.resolve({}, "fused", "plus")
        self.assertEqual((p["trunc_mm"], p["block_res"], p["min_weight"], p["smooth_mm"], p["smooth_min_sides"], p["drop_enclosed"]),
                         (1.2, 16, 2.0, 0.24, 3, True))
        self.assertEqual(PR.parse_path("fused.plus.smooth_mm")["method"], "plus")

    def test_preset_plus(self):
        r = PR.apply_preset({}, "plus")
        self.assertEqual(PR.method_of(r, "fused"), "plus")
        self.assertEqual(PR.resolve(r, "fused", "plus")["point_distance"], 0.4)
        self.e.set({"fused.method": "plus"}); g = self.e.graph()
        self.assertEqual(g.by_id["fused"].method.name, "plus")
        self.assertEqual([g.by_id[i].op["tool"] for i in g.lists["fused.plus"]],
                         ["isolation", "match_cleaned_sides"])     # cleanup is initially off
        self.assertFalse(g.by_id[g.lists["fused.plus"][-1]].op["on"])


# ---------------------------------------------------------------------------------------------------------- P2
class P2Argv(unittest.TestCase):
    def test_defaults_and_smooth_in_mm(self):
        p = PR.resolve({}, "fused", "plus", with_params={"fused.point_distance": 0.3})
        a = FP.plus_argv(p, "raw.ply", ["--set", "x"])
        self.assertEqual(a[1:3], ["--mode", "plus"]); self.assertEqual(arg(a, "--voxel"), "0.3"); self.assertIn("--gpu", a)
        self.assertEqual(arg(a, "--set-max-frames"), "300"); self.assertEqual(arg(a, "--plus-field-smooth"), "0.8")
        self.assertEqual(arg(a, "--plus-block-res"), "16")                       # fuse.py's own default is 0 (from --plus-block-mm)
        for f in ("--plus-trunc", "--plus-min-weight", "--plus-smooth-min-sides", "--trunc", "--min-weight"): self.assertNotIn(f, a)
        self.assertEqual(a[-2:], ["--set", "x"])
        p4 = PR.resolve({}, "fused", "plus")                                     # common default 0.4 mm: the tear-free recipe
        self.assertEqual((arg(FP.plus_argv(p4, "r", []), "--voxel"), arg(FP.plus_argv(p4, "r", []), "--plus-field-smooth")), ("0.4", "0.6"))
        p5 = PR.resolve({}, "fused", "plus", with_params={"fused.plus.trunc_mm": 1.5, "fused.plus.min_weight": 3})
        a5 = FP.plus_argv(p5, "r", []); self.assertEqual((arg(a5, "--plus-trunc"), arg(a5, "--plus-min-weight")), ("1.5", "3.0"))

    def test_classic_fallback_argv(self):
        p = PR.resolve({}, "fused", "plus", with_params={"fused.plus.min_weight": 5})
        a = FP.classic_argv(p, "raw.ply", ["--set", "x"])
        self.assertNotIn("--mode", a); self.assertFalse(any(x.startswith("--plus") for x in a))
        self.assertNotIn("--min-weight", a)                                      # plus's weight never leaks into classic
        self.assertEqual(a[1:], ["--out", "raw.ply", "--voxel", "0.4", "--gpu", "--set-max-frames", "300", "--set", "x"])


# ---------------------------------------------------------------------------------------------------------- P3 / P4
class _PlusRun(Base):
    def setUp(self):
        super().setUp(); self.cmds = []; self.fail_plus = 0
        use("fused", "plus"); use("side.platter", "auto"); inproc(self.e, self.fake_cmd)
        self.e.set({"fused.method": "plus"})
        self.ov = tempfile.mkdtemp(prefix="stages_repo_"); self.old = os.environ.get("POINTYOINK_STAGES_REPO")

    def tearDown(self):
        if self.old is None: os.environ.pop("POINTYOINK_STAGES_REPO", None)
        else: os.environ["POINTYOINK_STAGES_REPO"] = self.old
        shutil.rmtree(self.ov, ignore_errors=True); super().tearDown()

    def fake_cmd(self, cmd):
        name = os.path.basename(cmd[1]); self.cmds.append((name, cmd[2:]))
        if name == "strip_table.py": return cut_ok(cmd)
        if name == "fuse.py":
            if "--mode" in cmd and self.fail_plus:
                self.fail_plus -= 1
                raise StepError("fuse.py failed", tail="STAGE error {\"msg\": \"plus: GPU out of memory\", \"oom\": true}\n", rc=4)
            V, F = plates_mesh(); plyio.write_mesh(arg(cmd, "--out"), V, F)
            return 'STAGE integrate {"done": 10, "total": 10}\nSTAGE plus_grid {"blocks": 12, "grid_mb": 1}\nSTAGE done {"faces": %d, "verts": %d}\n' % (len(F), len(V))
        if name == "drop_enclosed.py":
            shutil.copy2(cmd[2], cmd[3])
            return 'STAGE drop_enclosed {"enclosed_cm2": 0.0, "tidy_triangles": 0, "triangles_out": 1}\ndropped 0 cm2\n'
        raise AssertionError("unexpected %s" % name)

    def fused_cmds(self):
        return [(n, a) for n, a in self.cmds if n in ("fuse.py", "drop_enclosed.py") and ("--set" in a or n == "drop_enclosed.py")]


class P3EngineRun(_PlusRun):
    def test_run_and_script_digests(self):
        self.run_()
        fc = self.fused_cmds()
        self.assertEqual([n for n, _a in fc], ["fuse.py", "drop_enclosed.py"])
        self.assertEqual(arg(fc[0][1], "--mode"), "plus"); self.assertEqual(len([x for x in fc[0][1] if x == "--set"]), len(self.ids))
        self.assertEqual(fc[1][1][0], arg(fc[0][1], "--out"))                   # drop_enclosed reads the plus fuse's output
        m = self.e.store.peek("fused", self.e.plan()[1]["fused"]["key"])
        self.assertEqual((m["stats"]["mode"], m["stats"]["sides"], m["stats"]["blocks"]), ("plus", len(self.ids), 12))
        # an identical copy of drop_enclosed.py: same keys; an edited one: only fused (and what follows) changes
        _g, i0, _r, _e = self.e.plan(); k0 = {k: v.get("key") for k, v in i0.items()}
        os.makedirs(os.path.join(self.ov, "scanner")); dst = os.path.join(self.ov, "scanner", "drop_enclosed.py")
        shutil.copy2(os.path.join(ROOT, "scanner", "drop_enclosed.py"), dst); os.environ["POINTYOINK_STAGES_REPO"] = self.ov
        _g, i1, _r, _e = self.e.plan(); self.assertEqual({k: v.get("key") for k, v in i1.items()}, k0)
        with open(dst, "a") as f: f.write("\n# edited\n")
        _g, i2, _r, _e = self.e.plan(); k2 = {k: v.get("key") for k, v in i2.items()}
        self.assertNotEqual(k2["fused"], k0["fused"]); self.assertEqual(k2["lineup.refine"], k0["lineup.refine"])
        self.cmds.clear(); self.run_()
        self.assertEqual([n for n, _a in self.fused_cmds()], ["fuse.py", "drop_enclosed.py"])     # only the fused node re-ran
        self.assertFalse(any(n == "strip_table.py" for n, _a in self.cmds))

    def test_drop_enclosed_off(self):
        self.e.set({"fused.plus.drop_enclosed": False}); self.run_()
        self.assertEqual([n for n, _a in self.fused_cmds()], ["fuse.py"])


class P4Fallback(_PlusRun):
    def test_classic_when_plus_fails_then_retry(self):
        self.fail_plus = 1; self.run_()
        fc = self.fused_cmds()
        self.assertEqual([("--mode" in a) for n, a in fc if n == "fuse.py"], [True, False])      # plus, then classic
        self.assertNotIn("drop_enclosed.py", [n for n, _a in fc])
        _g, info, _r, _e = self.e.plan()
        self.assertTrue(info["fused"].get("retry"), info["fused"])                               # flagged for retry
        self.cmds.clear(); self.run_()                                                            # next run: plus again, and it works
        self.assertEqual([n for n, _a in self.fused_cmds()], ["fuse.py", "drop_enclosed.py"])
        self.assertEqual(arg(self.fused_cmds()[0][1], "--mode"), "plus")

    def test_no_fallback(self):
        self.e.set({"fused.plus.fallback": False}); self.fail_plus = 1
        r = self.e.run(quiet=True)
        self.assertFalse(r.ok)
        self.assertEqual([("--mode" in a) for n, a in self.fused_cmds() if n == "fuse.py"], [True])


# ---------------------------------------------------------------------------------------------------------- P5
def box(c, s):
    """Closed axis-aligned box (outward normals), centre c, half sizes s: 8 vertices, 12 triangles."""
    c = np.asarray(c, float); s = np.asarray(s, float)
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s + c
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def subdivide(V, F, n=3):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)).subdivide_midpoint(n)
    return np.asarray(m.vertices), np.asarray(m.triangles)


class P5DropEnclosed(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(prefix="drop_enc_")

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def run_de(self, V, F, *extra):
        inp = os.path.join(self.tmp, "in.ply"); out = os.path.join(self.tmp, "out.ply"); plyio.write_mesh(inp, V, F)
        r = subprocess.run([PY, os.path.join(ROOT, "scanner", "drop_enclosed.py"), inp, out, "--min-share", "0"] + list(extra),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        st = json.loads([l for l in r.stdout.splitlines() if l.startswith("STAGE drop_enclosed ")][0].split(" ", 2)[2])
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(out); return np.asarray(m.vertices), np.asarray(m.triangles), st

    def test_sealed_flake_goes_outer_skin_stays(self):
        Vo, Fo = subdivide(*box((0, 0, 0), (10, 10, 10)))
        Vi, Fi = subdivide(*box((0, 0, 0), (2, 2, 0.3)))
        Vi2, Fi2 = Vi.copy(), Fi[:, ::-1]                                       # the flake's normals inward: still enclosed
        V = np.vstack([Vo, Vi2]); F = np.vstack([Fo, Fi2 + len(Vo)])
        V2, F2, st = self.run_de(V, F)
        self.assertEqual(st["enclosed_triangles"], len(Fi)); self.assertEqual(len(F2), len(Fo))
        self.assertLess(np.abs(V2).max(axis=0).min(), 10.0 + 1e-6); self.assertGreater(np.abs(V2).min(axis=0).max(), -1)
        self.assertTrue((np.abs(V2).max(1) > 9.99).all())                          # only the outer skin is left

    def test_face_seen_through_a_hole_stays_and_inward_skin_stays(self):
        Vo, Fo = box((0, 0, 0), (10, 10, 10)); Fo = Fo[[k for k in range(12) if k not in (10, 11)]]    # top (z = +10) open
        Vo, Fo = subdivide(Vo, Fo)
        Vi, Fi = subdivide(*box((0, 0, 5), (2, 2, 0.3)))
        V = np.vstack([Vo, Vi]); F = np.vstack([Fo, Fi + len(Vo)])
        V2, F2, st = self.run_de(V, F)
        C = V2[F2].mean(1)
        self.assertEqual(int((np.abs(C).max(1) > 9.99).sum()), len(Fo))            # the open box: all of it stays
        top = np.abs(C[:, 2] - 5.3) < 1e-6
        self.assertEqual(int(top.sum()), int((np.abs(Vi[Fi].mean(1)[:, 2] - 5.3) < 1e-6).sum()))   # seen from the opening: stays
        under = int((np.abs(Vi[Fi].mean(1)[:, 2] - 4.7) < 1e-6).sum())
        self.assertLess((np.abs(C[:, 2] - 4.7) < 1e-6).sum(), 0.5 * under)       # its underside can't be seen: most goes
        self.assertGreater(st["enclosed_triangles"], 0.5 * under)
        Vr, Fr = subdivide(*box((0, 0, 0), (10, 10, 10)))
        _V3, F3, st3 = self.run_de(Vr, Fr[:, ::-1])                                # a closed skin with its normals flipped
        self.assertEqual(st3["enclosed_triangles"], 0); self.assertEqual(len(F3), len(Fr))

    def test_rim_tidy(self):
        V, F = plates_mesh(step=1.0, half=3.0, gap=1.0); F = F[: len(F) // 2]          # one open plate
        spur = np.array([[3.0, 3.0, 1.0], [4.5, 3.2, 1.0]]); V = np.vstack([V, spur])
        corner = int(np.argmin(np.linalg.norm(V[:-2] - [3.0, 3.0, 1.0], axis=1)))
        F = np.vstack([F, [[corner, len(V) - 1, len(V) - 2]]])                    # a sliver hanging on at one vertex
        _V2, F2, st = self.run_de(V, F, "--rays", "0")
        self.assertGreaterEqual(st["tidy_triangles"], 1); self.assertLess(len(F2), len(F))
        _V3, F3, st3 = self.run_de(V, F, "--rays", "0", "--tidy-rounds", "0")
        self.assertEqual(len(F3), len(F))


# ---------------------------------------------------------------------------------------------------------- P6
FX = 1143.0; CX = 400.0; CY = 300.0; Z = 250.0; HALF = 30.0


def plate_frame(tx, ty, w=400, h=300):
    """400 x 300 depth (0.1 mm) of the plate z = -Z (scanner frame), |x|, |y| < HALF, seen by a camera translated (tx, ty)."""
    k = w / 800.0; fx, cx, cy = FX * k, CX * k, CY * k
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * Z / fx; Y = (v - cy) * Z / fx
    ok = (np.abs(X + tx) < HALF) & (np.abs(-Y + ty) < HALF)
    return np.where(ok, int(Z * 10), 0).astype(np.uint16)


def make_plate_project(root, sides=2, frames=4):
    sets = []
    for s in range(sides):
        d = os.path.join(root, "side%d" % s); c = os.path.join(d, "cache"); os.makedirs(c); os.makedirs(os.path.join(d, "param"))
        wb(os.path.join(d, "param", "Pl.bin"), struct.pack("<10f", 0, FX, 0, CX, 0, FX, CY, 0, 0, 1))
        pose = struct.pack("<i", frames)
        for f in range(frames):
            tx, ty = 3.0 * f - 4.5 + s, 2.0 * s - 1.0 + f
            P = np.eye(4); P[0, 3] = tx; P[1, 3] = ty
            wb(os.path.join(c, "frame_0_%d.dph" % f), plate_frame(tx, ty).tobytes())
            wb(os.path.join(c, "frame_0_%d.inf" % f), b"\0" * 16 + struct.pack("<16d", *P.ravel()))
            pose += struct.pack("<ii", 0, f) + struct.pack("<16d", *P.ravel())
        wb(os.path.join(c, "pointyoink_register_pose.pose"), pose)
        T = os.path.join(d, "T.json"); wb(T, json.dumps({"matrix": np.eye(4).tolist()}))
        sets += ["--set", "%s,%s,%s" % (c, os.path.join(d, "param", "Pl.bin"), T)]
    return sets


class P6FusePlusCpu(unittest.TestCase):
    """The real fuse.py --mode plus on the CPU: 2 sides x 4 frames of a flat plate; the surface lands on the plate, the
    classic fuse agrees with it, a rebuild is byte-identical and re-extraction from saved accumulators gives the same."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fuse_plus_"); self.sets = make_plate_project(self.tmp)

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def fuse(self, out, *extra):
        r = subprocess.run([PY, os.path.join(ROOT, "fuse.py"), "--out", out] + list(extra) + self.sets,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(out); return np.asarray(m.vertices), np.asarray(m.triangles), r.stdout

    def test_plate(self):
        p = os.path.join(self.tmp, "plus.ply"); acc = os.path.join(self.tmp, "acc.npz")
        V, F, out = self.fuse(p, "--mode", "plus", "--plus-acc", acc, "--plus-workers", "2")
        self.assertIn('"device": "CPU:0"', out); self.assertIn('"mode": "plus"', out)
        self.assertGreater(len(F), 1000)
        inner = (np.abs(V[:, 0]) < HALF - 3) & (np.abs(V[:, 1]) < HALF - 3)
        self.assertGreater(inner.mean(), 0.5)
        self.assertLess(np.median(np.abs(V[inner, 2] + Z)), 0.05)                # on the plate (z = -250 mm), not in metres
        self.assertLess(np.abs(V[inner, 2] + Z).max(), 0.3)
        Vc, _Fc, _o = self.fuse(os.path.join(self.tmp, "classic.ply"))          # classic CPU (0.3 mm, trunc 3 voxels)
        ic = (np.abs(Vc[:, 0]) < HALF - 3) & (np.abs(Vc[:, 1]) < HALF - 3)
        self.assertLess(abs(np.median(Vc[ic, 2]) - np.median(V[inner, 2])), 0.1)
        p2 = os.path.join(self.tmp, "plus2.ply"); self.fuse(p2, "--mode", "plus", "--plus-workers", "5")
        self.assertEqual(open(p, "rb").read(), open(p2, "rb").read())            # deterministic bytes (engine early cutoff)
        p3 = os.path.join(self.tmp, "plus3.ply"); self.fuse(p3, "--mode", "plus", "--plus-from-acc", acc)
        self.assertEqual(open(p, "rb").read(), open(p3, "rb").read())
        p4 = os.path.join(self.tmp, "plus4.ply"); self.fuse(p4, "--mode", "plus", "--plus-from-acc", acc, "--plus-smooth-min-sides", "1")
        self.assertNotEqual(open(p, "rb").read(), open(p4, "rb").read())         # 2 sides: blurred only when the gate allows it

    def test_grid_grows(self):
        _V, F, out = self.fuse(os.path.join(self.tmp, "g.ply"), "--mode", "plus", "--plus-block-count", "4")
        self.assertIn("block hash map grew", out); self.assertGreater(len(F), 1000)


# ---------------------------------------------------------------------------------------------------------- P7
class P7BuildProjectStep4(unittest.TestCase):
    """build_project.main() with every heavy step simulated: the classic argv is exactly today's; --fuse plus runs the
    plus fuse, keep_part, drop_enclosed; a failed plus fuse warns and runs today's classic command."""

    def setUp(self):
        import build_project as BP
        self.BP = BP; self.tmp = tempfile.mkdtemp(prefix="bp_plus_"); self.cmds = []; self.fail_plus = False
        for k in range(2):
            d = os.path.join(self.tmp, "data", "0927202600%04d" % k); os.makedirs(os.path.join(d, "cache")); os.makedirs(os.path.join(d, "param"))
            wb(os.path.join(d, "cache", "pointyoink_track.json"), "{}"); wb(os.path.join(d, "cache", "frame_0_0.dph"), b"\0\0")
        self.old_cfg = os.environ.get("POINTYOINK_CONFIG"); os.environ["POINTYOINK_CONFIG"] = os.path.join(self.tmp, "cfg", "config.json")
        self.saved = (BP.run, sys.argv)
        BP.run = self.fake_run

    def tearDown(self):
        self.BP.run, sys.argv = self.saved
        if self.old_cfg is None: os.environ.pop("POINTYOINK_CONFIG", None)
        else: os.environ["POINTYOINK_CONFIG"] = self.old_cfg
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_run(self, args, mem_gb, quiet=True, frames=None, relay=False):
        name = os.path.basename(args[0]); self.cmds.append((name, list(args[1:])))
        work = os.path.join(self.tmp, "build")
        if name == "pair_all.py":
            wb(os.path.join(work, "T_2_to_1.json"), json.dumps({"matrix": np.eye(4).tolist(), "fitness": 0.9})); return "pairs\n"
        if name == "place_sides.py":
            wb(os.path.join(work, "placement.json"), json.dumps({"order": [0, 1], "T": {"0": np.eye(4).tolist(), "1": np.eye(4).tolist()}}))
            return "placed\n"
        if name == "fuse.py" and "--mode" in args and self.fail_plus: raise RuntimeError("fuse.py failed (4): plus: GPU out of memory")
        if name == "drop_enclosed.py": return "STAGE drop_enclosed {}\ndropped 0 cm2\n"
        return "ok\n"

    def main(self, *extra):
        sys.argv = ["build_project.py", self.tmp] + list(extra)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = self.BP.main()
        self.assertEqual(rc, 0); return buf.getvalue()

    def step4(self):
        i = max(k for k, (n, a) in enumerate(self.cmds) if n == "place_sides.py")
        return [(n, a) for n, a in self.cmds[i + 1:] if n not in ("render_views.py", "needs_scan.py")]

    def expected_classic(self):
        work = os.path.join(self.tmp, "build"); sides = sorted(glob.glob(os.path.join(self.tmp, "data", "*")))
        a = ["--out", os.path.join(work, "all_raw.ply"), "--voxel", "0.4", "--gpu", "--set-max-frames", "300"]
        for i in (0, 1):
            a += ["--set", "%s,%s,%s" % (os.path.join(sides[i], "cache"), os.path.join(sides[i], "param", "Pl.bin"),
                                         os.path.join(work, "T_side%d_init.json" % (i + 1)))]
        return a

    def test_default_is_classic_unchanged(self):
        self.main()
        s4 = self.step4()
        self.assertEqual([n for n, _a in s4], ["fuse.py", "keep_part.py"])
        self.assertEqual(s4[0][1], self.expected_classic())

    def test_plus(self):
        out = self.main("--fuse", "plus")
        s4 = self.step4()
        self.assertEqual([n for n, _a in s4], ["fuse.py", "keep_part.py", "drop_enclosed.py"])
        a = s4[0][1]
        self.assertEqual(arg(a, "--mode"), "plus"); self.assertEqual(arg(a, "--voxel"), "0.3")
        self.assertEqual((arg(a, "--plus-trunc"), arg(a, "--plus-block-res"), arg(a, "--plus-min-weight"), arg(a, "--plus-field-smooth"),
                          arg(a, "--plus-smooth-min-sides")), ("1.2", "16", "2", "0.8", "3"))
        self.assertEqual(a[a.index("--set"):], self.expected_classic()[self.expected_classic().index("--set"):])
        model = os.path.join(self.tmp, "model.ply"); self.assertEqual(s4[2][1][:2], [model, model])
        self.assertIn('STAGE fused {"msg": "fused with plus", "mode": "plus"}', out)
        self.cmds.clear(); self.main("--fuse", "plus", "--plus-voxel", "0.4", "--no-drop-enclosed")
        s4 = self.step4(); self.assertEqual([n for n, _a in s4], ["fuse.py", "keep_part.py"])
        self.assertEqual((arg(s4[0][1], "--voxel"), arg(s4[0][1], "--plus-field-smooth")), ("0.4", "0.6"))

    def test_plus_fails_falls_back_to_classic(self):
        self.fail_plus = True
        out = self.main("--fuse", "plus")
        s4 = self.step4()
        self.assertEqual([n for n, _a in s4], ["fuse.py", "fuse.py", "keep_part.py"])
        self.assertEqual(s4[1][1], self.expected_classic())
        self.assertIn("plus fuse failed, using the classic fuse", out)


if __name__ == "__main__":
    unittest.main()
