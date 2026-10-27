"""Meshed tool "finish" (Finish for print: scanner/finish_mesh.py) and the "print" preset (Print-ready).

F1 registry, params, preset (kept manual deletes stay, before the finish)     F2 argv (every param reaches its flag)
F3 engine run on a fake script: progress, stats, script digest in the key, undo / redo from the cache
F4 One-tap print: cleaned points + finish in one run
F5 a real small run: finish_mesh.py on a sphere with a hole (Poisson depth 7, CPU, 6 cores at most, a few seconds), straight and through
   the engine: a closed mesh, simplified to the target."""
import os, shutil, subprocess, sys, tempfile, unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_stages_core import SAVED                                          # noqa: E402
from test_stages_plus import _PlusRun, arg                                   # noqa: E402
from stages import params as PR                                             # noqa: E402
from stages import plyio                                                    # noqa: E402
from stages.methods import get_tool                                         # noqa: E402
from stages.tools import finish as FIN                                      # noqa: E402

PY = sys.executable
HAS_PML = FIN.app_has_pymeshlab()


def sphere_with_hole(r=20.0, n_lat=60, n_lon=120, cap=0.95):
    """A UV sphere (mm) with the faces of its top cap (z > cap x r) removed: one round hole about 12 mm across."""
    th = np.linspace(0, np.pi, n_lat + 1)[1:-1]; ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.c_[(r * np.sin(T) * np.cos(P)).ravel(), (r * np.sin(T) * np.sin(P)).ravel(), (r * np.cos(T)).ravel()]
    V = np.vstack([V, [0, 0, r], [0, 0, -r]]); top, bot = len(V) - 2, len(V) - 1
    idx = np.arange((n_lat - 1) * n_lon).reshape(n_lat - 1, n_lon); F = []
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b, c, d = idx[i, j], idx[i, (j + 1) % n_lon], idx[i + 1, j], idx[i + 1, (j + 1) % n_lon]
            F += [[a, c, b], [b, c, d]]
    for j in range(n_lon):
        F += [[top, idx[0, j], idx[0, (j + 1) % n_lon]], [bot, idx[-1, (j + 1) % n_lon], idx[-1, j]]]
    F = np.asarray(F)
    keep = ~(V[F][:, :, 2] > cap * r).any(1)
    F = F[keep]; used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F]


def open_edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _u, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


# ---------------------------------------------------------------------------------------------------------- F1
class F1Registered(unittest.TestCase):
    def test_tool_params_code(self):
        t = get_tool("finish", "mesh")
        self.assertEqual((t.name, t.label, t.kinds), ("finish", "Finish for print", ("mesh",)))
        self.assertNotIn("points", get_tool("finish", "points").kinds)          # n/a on points
        files, libs = t.code({})
        self.assertEqual([n for n, _p in files], ["finish_mesh.py", "stages/tools/finish.py"])
        self.assertTrue(os.path.exists(dict(files)["finish_mesh.py"]))
        self.assertTrue(any(l.startswith("pymeshlab-") for l in libs), libs)
        p = PR.resolve_tool("finish", "mesh", {})
        self.assertEqual(p, {"faces": 250000, "depth": 9, "trim_mm": 3.0, "hole_edges": 400, "smooth": 5})
        with self.assertRaises(ValueError): PR.resolve_tool("finish", "mesh", {"face": 1})          # typos are errors
        with self.assertRaises(ValueError): PR.resolve_tool("finish", "mesh", {"faces": 5000})      # under the range
        self.assertIn("finish", PR.TOOLS_BY_KIND["mesh"])
        self.assertEqual(t.summary(p), "250k triangles, smooth 5, holes 400 edges")
        self.assertEqual((FIN.faces_text(1000000), FIN.faces_text(250000), FIN.faces_text(123456)), ("1M", "250k", "123456"))
        self.assertEqual(FIN.stl_bytes(250000), 12500084)

    def test_print_preset(self):
        r = PR.apply_preset({}, "print")
        self.assertEqual(PR.method_of(r, "fused"), "points")
        self.assertEqual(PR.resolve(r, "fused", "points")["point_distance"], 0.4)
        self.assertEqual(PR.resolve(r, "fused", "points")["frames_per_side"], 300)
        self.assertEqual([o["tool"] for o in r["ops"]["meshed"]], ["finish"])
        self.assertEqual([o["tool"] for o in r["ops"]["fused.points"]], ["isolation", "overlap", "smooth"])
        # a preset keeps the manual deletes of the lists it writes: they go first, the finish last
        r0 = {"next_op": 5, "ops": {"meshed": [{"id": "op-0001", "tool": "delete", "shape": "box", "on": True},
                                                {"id": "op-0002", "tool": "smooth", "on": True, "params": {}}]}}
        r1 = PR.apply_preset(r0, "print")
        self.assertEqual([(o["tool"], o["id"]) for o in r1["ops"]["meshed"]], [("delete", "op-0001"), ("finish", "op-0008")])
        r2 = PR.apply_preset(r1, "today")                                  # today drops the finish, keeps the delete
        self.assertEqual([o["tool"] for o in r2["ops"]["meshed"]], ["delete"])


# ---------------------------------------------------------------------------------------------------------- F2
class F2Argv(unittest.TestCase):
    def test_flags(self):
        a = FIN.argv(PR.resolve_tool("finish", "mesh", {}), "in.ply", "out.ply")
        self.assertEqual(os.path.basename(a[0]), "finish_mesh.py"); self.assertEqual(a[1:3], ["in.ply", "out.ply"])
        self.assertEqual([arg(a, f) for f in ("--faces", "--depth", "--trim", "--hole-edges", "--smooth")], ["250000", "9", "3", "400", "5"])
        a2 = FIN.argv(PR.resolve_tool("finish", "mesh", {"faces": 600000, "depth": 10, "trim_mm": 2.5, "hole_edges": 0, "smooth": 0}), "i", "o")
        self.assertEqual([arg(a2, f) for f in ("--faces", "--depth", "--trim", "--hole-edges", "--smooth")], ["600000", "10", "2.5", "0", "0"])
        self.assertNotIn("--stl", a2)
        self.assertGreater(FIN.cap_gb({"depth": 11}), FIN.cap_gb({"depth": 9}))


# ---------------------------------------------------------------------------------------------------------- F3 / F4
class _FinishRun(_PlusRun):
    real = False

    def fake_cmd(self, cmd):
        name = os.path.basename(cmd[1])
        if name == "finish_mesh.py":
            self.cmds.append((name, cmd[2:]))
            if self.real:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:]); return r.stdout
            from test_stages_fixes import plates_mesh
            V, F = plates_mesh(); plyio.write_mesh(cmd[3], V, F)
            return "".join('STAGE %s {}\n' % s for s in FIN.STAGES) + \
                'STAGE done {"out": "%s", "stl": null, "faces_in": %d, "faces": %d, "secs": 0.1, "mb_stl": null}\n' % (cmd[3], len(F), len(F))
        return super().fake_cmd(cmd)

    def finish_cmds(self): return [a for n, a in self.cmds if n == "finish_mesh.py"]

    def meshed_end(self):
        g = self.e.graph(); return g.final("meshed")


class F3EngineRun(_FinishRun):
    def test_run_key_undo(self):
        self.run_()
        self.assertEqual(self.finish_cmds(), [])
        self.e.add_op("meshed", {"tool": "finish", "params": {"faces": 300000, "smooth": 3}})
        end = self.meshed_end(); self.assertTrue(end.startswith("meshed#"), end)
        self.run_(to="meshed")
        fc = self.finish_cmds(); self.assertEqual(len(fc), 1)
        self.assertEqual((arg(fc[0], "--faces"), arg(fc[0], "--smooth"), arg(fc[0], "--depth")), ("300000", "3", "9"))
        g = self.e.graph(); parent = g.by_id[end].inputs["parent"]
        _g, info, _r, _e = self.e.plan()
        self.assertEqual(info[end]["status"], "fresh")
        m = self.e.store.peek(g.by_id[end].store, info[end]["key"])
        self.assertEqual(m["stats"]["faces"], m["stats"]["faces_in"]); self.assertIsNotNone(m["stats"]["stl_mb"])
        self.assertEqual(parent, "meshed")
        # the script's digest is in the key: an identical copy keeps every key, an edited one changes only the finish
        k0 = {k: v.get("key") for k, v in info.items()}
        os.makedirs(os.path.join(self.ov, "scanner")); dst = os.path.join(self.ov, "scanner", "finish_mesh.py")
        shutil.copy2(os.path.join(ROOT, "scanner", "finish_mesh.py"), dst); os.environ["POINTYOINK_STAGES_REPO"] = self.ov
        _g, i1, _r, _e = self.e.plan(); self.assertEqual({k: v.get("key") for k, v in i1.items()}, k0)
        with open(dst, "a") as f: f.write("\n# edited\n")
        _g, i2, _r, _e = self.e.plan(); k2 = {k: v.get("key") for k, v in i2.items()}
        self.assertEqual(sorted(k for k in k0 if k2[k] != k0[k]), sorted([end, "report"]))
        os.environ.pop("POINTYOINK_STAGES_REPO")
        # undo drops the finish (nothing runs), redo brings it back from the cache
        self.e.undo(); self.assertEqual(self.meshed_end(), "meshed")
        self.cmds.clear(); self.run_(to="meshed"); self.assertEqual(self.finish_cmds(), [])
        self.e.redo(); self.run_(to="meshed"); self.assertEqual(self.finish_cmds(), [])
        _g, i3, _r, _e = self.e.plan(); self.assertEqual(i3[end]["status"], "fresh")
        # turned off: passes the mesh through
        self.e.toggle_op("meshed", 0); _g, i4, _r, _e = self.e.plan(); self.assertEqual(i4[end]["status"], "off")

    def test_confirm_line(self):
        self.e.add_op("meshed", {"tool": "finish"})
        self.assertIn("Mesh tools: Finish for print 250k triangles, smooth 5, holes 400 edges", self.e.confirm_line())


class F4Preset(_FinishRun):
    def test_print_preset_runs_clean_points_and_finish(self):
        from test_stages_fixes3 import DensePointsMixin
        from test_stages_fixes import use, plates_mesh
        class MeshFixture:
            def execute(self, ctx):
                V, F = plates_mesh(); plyio.write_mesh(os.path.join(ctx.out_dir, 'mesh.ply'), V, F)
                return {'outputs': {'mesh': 'mesh.ply'}, 'stats': {}}
        use('fused', 'points', DensePointsMixin)
        use('meshed', 'poisson', MeshFixture)
        self.e.set({"fused.method": "tsdf"})
        self.e.preset("print"); self.run_()
        self.assertEqual(self.e.graph().by_id["fused"].method.name, "points")
        self.assertFalse(any(n == "fuse.py" and "--mode" in a for n, a in self.cmds))
        self.assertEqual(len(self.finish_cmds()), 1)
        g = self.e.graph(); self.assertEqual(g.by_id[g.final("meshed")].method.name, "finish")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model.ply")))


# ---------------------------------------------------------------------------------------------------------- F5
@unittest.skipUnless(HAS_PML, "PyMeshLab is not in this python")
class F5RealRun(_FinishRun):
    real = True

    def fake_cmd(self, cmd):
        if os.path.basename(cmd[1]) == "fuse.py" and self.real:           # the fused model: a sphere with a hole
            V, F = sphere_with_hole(); plyio.write_mesh(arg(cmd, "--out"), V, F)
            self.cmds.append(("fuse.py", cmd[2:]))
            return 'STAGE done {"faces": %d, "verts": %d}\n' % (len(F), len(V))
        return super().fake_cmd(cmd)

    def test_script_straight(self):
        d = tempfile.mkdtemp(prefix="finish_real_")
        try:
            V, F = sphere_with_hole(); src = os.path.join(d, "in.ply"); plyio.write_mesh(src, V, F)
            self.assertGreater(open_edges(F), 100)
            out = os.path.join(d, "out.ply"); stl = os.path.join(d, "out.stl")
            r = subprocess.run([PY, os.path.join(ROOT, "scanner", "finish_mesh.py"), src, out, "--stl", stl, "--depth", "7",
                                "--faces", "100000"], capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            V2, F2, _N = plyio.read_mesh(out)
            self.assertLessEqual(len(F2), 110000); self.assertGreater(len(F2), 20000)
            self.assertEqual(open_edges(F2), 0)                                  # the hole is closed
            rad = np.linalg.norm(V2, axis=1); self.assertLess(abs(np.median(rad) - 20.0), 0.5)   # same size (Taubin, no shrink)
            self.assertEqual(os.path.getsize(stl), FIN.stl_bytes(len(F2)))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_through_the_engine(self):
        self.e.clear_ops("fused")                                   # the finish reads the fused sphere itself
        self.e.add_op("meshed", {"tool": "finish", "params": {"faces": 100000, "depth": 7}})
        self.run_()
        g = self.e.graph(); end = g.final("meshed"); _g, info, _r, _e = self.e.plan()
        self.assertEqual(info[end]["status"], "fresh")
        V2, F2, _N = plyio.read_mesh(os.path.join(self.tmp, "model.ply"))       # published
        self.assertLessEqual(len(F2), 110000); self.assertEqual(open_edges(F2), 0)
        m = self.e.store.peek(g.by_id[end].store, info[end]["key"])
        self.assertEqual(m["stats"]["faces"], len(F2)); self.assertGreater(m["stats"]["faces_in"], 0)


if __name__ == "__main__":
    unittest.main()
