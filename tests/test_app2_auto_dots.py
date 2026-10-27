"""Show auto dots: the automatic fit's matches as spread-out pairs that really line up."""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink_dots_")) / "config.json")

import numpy as np
from app2.qt.build_step import BuildStepPage


def fake_page(pairs_json, items, calls):
    page = SimpleNamespace(gl=True, vp=SimpleNamespace(view=SimpleNamespace(items=items)),
                           snap=SimpleNamespace(out=lambda step: pairs_json), notice=lambda t, *a, **k: calls.append(("notice", t)),
                           _recipe_value=lambda p: 1.5, panel={"lineup": SimpleNamespace(set_pairs=lambda p: calls.append(("pairs", p)))})
    return page


def surface(rng, n):
    xy = rng.uniform(-60, 60, (n, 2)); z = 5 * np.sin(xy[:, 0] / 15) + 4 * np.cos(xy[:, 1] / 11)
    return np.column_stack([xy, z])


def test_pairs_line_up_under_the_fit_and_are_spread_out(tmp_path):
    rng = np.random.default_rng(1); base = surface(rng, 40000)
    a = np.radians(25); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]); t = np.array([5.0, -3.0, 2.0])
    moving = (base - t) @ R                                   # base = moving @ R.T + t
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    pj = tmp_path / "pairs.json"
    pj.write_text(json.dumps({"pairs": [{"base": "B", "moving": "M", "matrix": M.tolist(), "fitness": .9, "rmse": .5}]}))
    items = {"B": dict(kind="mesh", V=base), "M": dict(kind="mesh", V=moving)}; calls = []
    BuildStepPage.suggest_dots(fake_page(str(pj), items, calls), "M", "B", 12)
    pairs = [c[1] for c in calls if c[0] == "pairs"][0]
    assert len(pairs) == 12
    B = np.array([p[0] for p in pairs]); Mv = np.array([p[1] for p in pairs])
    assert np.linalg.norm(Mv @ R.T + t - B, axis=1).max() < 1.5            # each pair is a true match
    assert np.linalg.norm(B[:, None] - B[None], axis=2)[np.triu_indices(12, 1)].min() > 8      # and they are spread, not bunched


def test_a_pair_with_no_result_or_no_overlap_says_so(tmp_path):
    pj = tmp_path / "pairs.json"; pj.write_text(json.dumps({"pairs": []})); calls = []
    items = {"B": dict(kind="mesh", V=np.zeros((10, 3))), "M": dict(kind="mesh", V=np.ones((10, 3)))}
    BuildStepPage.suggest_dots(fake_page(str(pj), items, calls), "M", "B", 6)
    assert calls and calls[0][0] == "notice" and "no result" in calls[0][1]


def test_a_side_placed_far_from_its_own_auto_fit_is_flagged(tmp_path):
    def pose(deg, t):
        a = np.radians(deg); M = np.eye(4); M[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]; M[:3, 3] = t; return M
    fit = pose(10, [3, 0, 0])
    pj = tmp_path / "p.json"
    pj.write_text(json.dumps({"pairs": [{"base": "B", "moving": s, "matrix": fit.tolist(), "fitness": .9, "rmse": .5} for s in ("OK", "FLIP", "FAR")]}))
    page = SimpleNamespace(lineup={"T": {"B": np.eye(4).tolist(), "OK": fit.tolist(), "FLIP": pose(190, [3, 0, 0]).tolist(), "FAR": pose(10, [3, 0, 80]).tolist()}},
                           snap=SimpleNamespace(out=lambda step: str(pj)))
    got = BuildStepPage._auto_disagreements(page, "B", {})
    assert set(got) == {"FLIP", "FAR"} and "180 deg" in got["FLIP"] and "80 mm" in got["FAR"]


def test_orientation_never_touches_the_items_of_other_stages():
    """Regression: after every 3D load the orientation was applied to the FIRST item, which in Line-up is side 1 (it went back to identity)."""
    moved = []
    view = SimpleNamespace(items={"side1": {"kind": "mesh", "V": np.zeros((3, 3))}}, set_item_model=lambda k, M: moved.append(k), tool="orbit")
    page = SimpleNamespace(gl=True, orient_box=object(), stage="lineup", vp=SimpleNamespace(view=view))
    BuildStepPage._orient_view(page)
    assert moved == []
