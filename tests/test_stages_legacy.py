"""Legacy side of the staged build: adopting build_project.py's caches (T8) and the legacy event stream read by a copy of
pointyoink_scanner.py's parser (T10). Synthetic projects; fake methods for the run; no subprocesses."""
import hashlib, io, json, os, shutil, sys, tempfile, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner")); sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_stages_core import make_project, install_fakes, remove_fakes, ident, wb   # noqa: E402
from stages.engine import Engine                                                   # noqa: E402
from stages.project import Project                                                  # noqa: E402
from stages.keys import md5_file                                                    # noqa: E402
from stages.methods.lineup import pair_version                                      # noqa: E402
from stages import compat                                                           # noqa: E402


def jd(obj, path):
    with open(path, "w") as f: json.dump(obj, f)


def fake_build(root, ids, voxel=0.4):
    """build/ as build_project.py leaves it (files written in its order, mtimes increasing)."""
    b = os.path.join(root, "build"); os.makedirs(b); cache = {}; V = pair_version(); t = time.time() - 500
    def touch(p):
        nonlocal t
        t += 1; os.utime(p, (t, t))
    for n, sid in enumerate(ids, 1):
        d = os.path.join(root, "data", sid); dph = [os.path.join(d, "cache", f) for f in os.listdir(os.path.join(d, "cache")) if f.endswith(".dph")]
        mt = max(os.stat(f).st_mtime_ns for f in dph)
        cache["side%d" % n] = "1|%s|%d|%d|%s|%s|%s" % (sid, len(dph), mt, md5_file(os.path.join(d, "cache", "pointyoink_register_pose.pose")),
                                                    md5_file(os.path.join(d, "table_plane.json")), voxel)
        wb(os.path.join(b, "side%d_raw.ply" % n), "raw %d" % n); touch(os.path.join(b, "side%d_raw.ply" % n))
        jd({"n": [0, 0, 1], "d": -10.0 - n, "margin": 1.0, "keep_above": True}, os.path.join(b, "side%d_plane.json" % n)); touch(os.path.join(b, "side%d_plane.json" % n))
        wb(os.path.join(b, "side%d_cut.ply" % n), "cut %d" % n); touch(os.path.join(b, "side%d_cut.ply" % n))
        wb(os.path.join(b, "side%d.ply" % n), "part %d" % n); touch(os.path.join(b, "side%d.ply" % n))
    jd(cache, os.path.join(b, "sides_cache.json"))

    def stamp(n):
        s = os.stat(os.path.join(b, "side%d.ply" % n)); return "%d:%d:%d" % (V, s.st_size, s.st_mtime_ns)

    def beds(n):
        pl = json.load(open(os.path.join(b, "side%d_plane.json" % n)))
        e = {"folder": ids[n - 1], "plane": {"n": pl["n"], "d": pl["d"]}, "B_scan": None, "bed_id": None}
        return hashlib.md5(json.dumps(e, sort_keys=True).encode()).hexdigest()[:12]
    pc = {}
    for i in range(1, len(ids) + 1):
        for j in range(i + 1, len(ids) + 1):
            pc["%d_to_%d" % (j, i)] = {"base": stamp(i), "moving": stamp(j), "matrix": ident(), "fitness": 0.8, "rmse": 0.5,
                                       "beds": beds(i) + beds(j), "support": None, "bed": None, "warns": []}
    jd(pc, os.path.join(b, "pairs_cache.json"))
    jd({"order": list(range(len(ids))), "T": {str(k): ident() for k in range(len(ids))}, "score": {str(k): 1.0 for k in range(len(ids))},
        "flagged": []}, os.path.join(b, "placement.json")); touch(os.path.join(b, "placement.json"))
    for k in range(len(ids)):
        jd({"matrix": ident()}, os.path.join(b, "T_refined_%d.json" % (k + 1))); touch(os.path.join(b, "T_refined_%d.json" % (k + 1)))
    wb(os.path.join(b, "all_raw.ply"), "all raw"); touch(os.path.join(b, "all_raw.ply"))


class T8Adopt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stages_legacy_"); self.ids = make_project(self.tmp)
        fake_build(self.tmp, self.ids)

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def status(self, e):
        g = e.graph(); _o, info, _ = e._walk(g, "plan", None); return {k: v["status"] for k, v in info.items()}

    def test_all_match(self):
        e = Engine(Project.open(self.tmp)); r = e.adopt_legacy()
        st = self.status(e)
        for s in self.ids:
            for n in ("side.fuse@", "side.platter@", "side.keep@"): self.assertEqual(st[n + s], "adopted", n + s)
        for n in ("lineup.pairs", "lineup.place", "lineup.refine", "fused"): self.assertEqual(st[n], "adopted", (n, r))
        self.assertEqual(st["lineup.fix"], "off"); self.assertEqual(st["lineup.adjust"], "fresh")
        g = e.graph(); self.assertEqual(st[g.lists["fused.tsdf"][0]], "stale")       # the fused isolation op runs next
        # the adopted files are copies with the build's bytes
        out = e.output("side.keep@" + self.ids[1]); self.assertEqual(open(out).read(), "part 2")
        self.assertNotEqual(os.path.realpath(out), os.path.realpath(os.path.join(self.tmp, "build", "side2.ply")))

    def test_one_changed(self):
        d = os.path.join(self.tmp, "data", self.ids[1], "cache"); f = os.path.join(d, sorted(x for x in os.listdir(d) if x.endswith(".dph"))[0])
        os.utime(f, (time.time() + 5, time.time() + 5))
        e = Engine(Project.open(self.tmp)); e.adopt_legacy(); st = self.status(e)
        self.assertEqual(st["side.fuse@" + self.ids[1]], "stale")
        self.assertEqual(st["side.keep@" + self.ids[1]], "waiting")
        for s in self.ids[:1] + self.ids[2:]: self.assertEqual(st["side.keep@" + s], "adopted")
        self.assertEqual(st["lineup.pairs"], "waiting")
        cache = json.load(open(os.path.join(self.tmp, "stages", "cache", "pairs.json")))
        self.assertEqual(len(cache), 3)                      # the 3 pairs without side 2
        for n in ("lineup.place", "fused"): self.assertEqual(st[n], "waiting")

    def test_voxel_mismatch(self):
        e = Engine(Project.open(self.tmp)); e.set({"side.fuse.voxel": 0.5}); e.adopt_legacy(); st = self.status(e)
        self.assertTrue(all(st["side.fuse@" + s] == "stale" for s in self.ids))


# the scanner's build-output parser (pointyoink_scanner.py build_model.worker), copied: SPAN table + frames branch
SPAN = {"side": (0.0, 0.35), "table": (0.0, 0.35), "align": (0.35, 0.70), "aligned": (0.35, 0.70), "base": (0.70, 0.72),
        "refine": (0.72, 0.80), "fuse": (0.80, 0.95), "done": (1.0, 1.0)}


def scanner_parse(text):
    st = {"build_frac": 0.0}; raw = []; done = None; logged = []
    for ln in text.splitlines(True):
        if ln.startswith(("median of pair medians", "kept ")): logged.append(ln.strip())
        if not ln.startswith("STAGE "): continue
        parts = ln.split(" ", 2)
        try: info = json.loads(parts[2])
        except Exception: info = {}
        if parts[1] == "frames":
            k, n = info.get("done", 0), max(1, info.get("total", 1))
            if info.get("step") == "all": v = 0.80 + 0.15 * k / n
            else:
                sn, ns = info.get("side", 1), max(1, info.get("sides", 1)); v = 0.35 * (sn - 1 + k / n) / ns
            raw.append((parts[1], v)); st["build_frac"] = max(st["build_frac"], v)
            continue
        if parts[1] in SPAN:
            lo, hi = SPAN[parts[1]]; k = info.get("done", info.get("side", 1) - 1 if "side" in info else 0); n = info.get("total", info.get("count", 1))
            if parts[1] == "side": st["sides_n"] = n
            if parts[1] == "table": n = st.get("sides_n", 1); k = info.get("side", 1)
            v = lo + (hi - lo) * min(1.0, k / max(1, n)); raw.append((parts[1], v)); st["build_frac"] = max(st["build_frac"], v)
        if parts[1] in ("warn", "error", "base", "done"): logged.append("%s: %s" % (parts[1], info.get("msg") or info.get("model", "")))
        elif "msg" in info and parts[1] not in ("align",): logged.append(info["msg"])
        if parts[1] == "done": done = info
    return st["build_frac"], raw, done, logged


class T10LegacyStream(unittest.TestCase):
    def setUp(self):
        install_fakes(); self.tmp = tempfile.mkdtemp(prefix="stages_compat_"); self.ids = make_project(self.tmp, n=5)
        self._post = Engine._post; Engine._post = lambda *a, **k: None

    def tearDown(self):
        Engine._post = self._post; remove_fakes(); shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stream(self):
        out = io.StringIO()
        rc = compat.main([self.tmp], out=out); text = out.getvalue()
        self.assertEqual(rc, 0, text[-2000:])
        frac, raw, done, logged = scanner_parse(text)
        vals = [v for _k, v in raw]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), raw)
        self.assertEqual(frac, 1.0)
        for k in ("model", "preview", "sides", "base", "secs"): self.assertIn(k, done)
        self.assertEqual(done["sides"], 5)
        for ln in text.splitlines():
            if not ln.startswith("STAGE "): continue
            name, body = ln.split(" ", 2)[1:]; info = json.loads(body)
            if name in ("plan", "node", "progress", "published"): self.assertNotIn("msg", info, ln)
            if name == "frames": self.assertNotIn("count", info, ln)
        self.assertIn("sides", [ln.split(" ")[1] for ln in text.splitlines() if ln.startswith("STAGE ")])
        self.assertTrue(any(l.startswith("median of pair medians") for l in logged)); self.assertTrue(any(l.startswith("kept ") for l in logged))
        # legacy files for bake/frames.py
        b = os.path.join(self.tmp, "build")
        pl = json.load(open(os.path.join(b, "placement.json"))); self.assertEqual(sorted(pl["order"]), list(range(5)))
        for k in range(5): self.assertTrue(os.path.exists(os.path.join(b, "T_refined_%d.json" % (k + 1))))
        self.assertTrue(os.path.exists(os.path.join(b, "side1_plane.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model.ply")))
        # a second Build: nothing runs, same stream shape
        out2 = io.StringIO(); self.assertEqual(compat.main([self.tmp], out=out2), 0)
        f2, _r, d2, _l = scanner_parse(out2.getvalue()); self.assertEqual(f2, 1.0); self.assertIsNotNone(d2)
        self.assertIn("side 1 unchanged, reusing it", out2.getvalue())


if __name__ == "__main__":
    unittest.main()
