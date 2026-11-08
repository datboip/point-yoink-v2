"""Simplify with a limit on how far the surface may move: measured against the original both ways, redone with more faces when needed."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def sphere(tmp_path_factory):
    import trimesh
    d = tmp_path_factory.mktemp("simp"); m = trimesh.creation.icosphere(subdivisions=5, radius=20)     # 20k faces
    m.vertices += np.random.default_rng(0).normal(0, 0.01, m.vertices.shape)
    src = str(d / "in.ply"); m.export(src); return d, src


def run(sphere, *extra):
    d, src = sphere; out = str(d / "out.ply")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "process.py"), src, out, "--clean", "--isolation-rate", "0", "--smooth-times", "0", *extra],
                       capture_output=True, text=True, timeout=250)
    stages = {ln.split(" ", 2)[1]: json.loads(ln.split(" ", 2)[2]) for ln in r.stdout.splitlines() if ln.startswith("STAGE ")}
    return r, stages


def test_tight_limit_keeps_more_faces_until_it_holds(sphere):
    r, st = run(sphere, "--simplify-pct", "5", "--simplify-tol", "0.03")
    c = st["simplify_check"]
    assert r.returncode == 0 and c["ok"] and c["within_pct"] >= 99.9 and c["max_mm"] <= 0.03 + 1e-6
    assert c["passes"] > 1 and c["keep_pct"] > 5


def test_loose_limit_keeps_the_asked_share(sphere):
    _r, st = run(sphere, "--simplify-pct", "10", "--simplify-tol", "5")
    assert st["simplify_check"]["passes"] == 1 and st["simplify_check"]["keep_pct"] == 10 and st["cleaned"]["faces"] < 3000


def test_no_limit_is_the_old_plain_percent(sphere):
    _r, st = run(sphere, "--simplify-pct", "10")
    assert "simplify_check" not in st and st["cleaned"]["faces"] < 3000


def test_tool_flags_and_report_text():
    from scanner.stages.tools.mesh_tools import Simplify
    from app2.qt import build_words as W
    t = Simplify()
    assert t.flags({"keep_pct": 40}) == ["--simplify-pct", "40"]
    assert t.flags({"keep_pct": 40, "tol_mm": 0.05}) == ["--simplify-pct", "40", "--simplify-tol", "0.05"]
    assert "within 0.05 mm" in t.summary({"keep_pct": 40, "tol_mm": 0.05})
    said = W.operation_stats("simplify", {"simplify_check": dict(within_pct=99.97, tol_mm=0.05, max_mm=0.048, keep_pct=62.5, ok=True)}, "mesh")
    assert said.startswith("99.97% within 0.05 mm, farthest 0.048 mm") and "limit not reached" not in said
