"""Smooth area: smooth inside one shape on the surface, fading out, and leave everything else exactly as it was."""
import json
import numpy as np
import pytest
from scanner.stages.tools import local_fix as L


def spot(c, n=(0, 0, 1)): return json.dumps([list(c), list(n)])


P0 = {"spot": spot((0, 0, 0)), "shape": "cylinder", "diameter": 20.0, "width": 20.0, "height": 20.0, "depth": 6.0, "feather": 4.0, "strength": 0.8, "passes": 8}


def noisy_floor(seed=0):
    r = np.random.default_rng(seed); g = np.array([[x, y, 0.0] for x in np.arange(-40, 40.1, 1.0) for y in np.arange(-40, 40.1, 1.0)])
    g[:, 2] = r.normal(0, 0.3, len(g)); return g


def test_points_inside_get_smoother_and_points_outside_do_not_move():
    P = noisy_floor(); out, sel, st = L.apply("points", {"P": P}, P0); Q = out["P"]
    d = np.linalg.norm(P[:, :2], axis=1); inside = d < 6; outside = d > 12
    assert Q[inside, 2].std() < 0.5 * P[inside, 2].std()                       # much smoother in the middle
    assert np.array_equal(Q[outside], P[outside])                              # exactly untouched outside the shape
    assert st["moved"] == int(sel.sum()) and st["moved"] > 100


def test_the_edge_fades_instead_of_a_step():
    P = noisy_floor(); out, _, _ = L.apply("points", {"P": P}, P0); move = np.abs(out["P"][:, 2] - P[:, 2]); d = np.linalg.norm(P[:, :2], axis=1)
    ring = (d > 8.5) & (d < 10); core = d < 5
    assert 0 < move[ring].mean() < move[core].mean()                           # less change near the edge than in the middle


def test_taubin_does_not_shrink_a_curved_patch():
    t = np.linspace(0, np.pi, 41); P = np.array([[x, y, 6 * np.cos(np.radians(x * 4)) - 6] for x in np.arange(-10, 10.1, 0.5) for y in np.arange(-10, 10.1, 0.5)])
    out, _, _ = L.apply("points", {"P": P}, dict(P0, diameter=40.0, depth=20.0, passes=10)); Q = out["P"]
    assert abs((Q[:, 2].max() - Q[:, 2].min()) - (P[:, 2].max() - P[:, 2].min())) < 0.25 * (P[:, 2].max() - P[:, 2].min())


def test_mesh_smooths_only_vertices_inside():
    g = [(x, y) for x in range(-30, 31, 2) for y in range(-30, 31, 2)]; V = np.array([[x, y, 0.0] for x, y in g]); r = np.random.default_rng(1); V[:, 2] = r.normal(0, .4, len(V))
    n = 31; idx = lambda i, j: i * n + j
    F = np.array([f for i in range(n - 1) for j in range(n - 1) for f in ([idx(i, j), idx(i + 1, j), idx(i, j + 1)], [idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])])
    (V2, F2, _), sel, st = L.apply("mesh", (V, F, None), P0); d = np.linalg.norm(V[:, :2], axis=1)
    assert V2[d < 6, 2].std() < 0.6 * V[d < 6, 2].std() and np.array_equal(V2[d > 12], V[d > 12]) and np.array_equal(F2, F)


def test_box_shape_and_errors():
    P = noisy_floor(); out, _, _ = L.apply("points", {"P": P}, dict(P0, shape="box", width=10.0, height=30.0))
    moved = np.abs(out["P"][:, 2] - P[:, 2]) > 0; assert moved[(np.abs(P[:, 1]) < 3) & (np.abs(P[:, 0]) < 10)].any() and not moved[np.abs(P[:, 1]) > 6].any()      # width runs along the surface's first axis (y here)
    with pytest.raises(ValueError, match="area"): L.apply("points", {"P": P}, dict(P0, spot="[]"))
    with pytest.raises(ValueError, match="nothing to smooth"): L.apply("points", {"P": P}, dict(P0, spot=spot((500, 500, 500))))


def test_reach_flattens_a_wide_bump_that_fine_smoothing_leaves():
    g = np.array([[x, y, 0.0] for x in np.arange(-30, 30.1, 0.4) for y in np.arange(-30, 30.1, 0.4)])
    g[:, 2] = 0.6 * np.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) / (2 * 1.2 ** 2))                           # a bump 0.6 mm tall, about 3 mm wide
    narrow, _, _ = L.apply("points", {"P": g, "cell": 0.4}, dict(P0, reach=0.5, passes=4))
    wide, _, _ = L.apply("points", {"P": g, "cell": 0.4}, dict(P0, reach=4.0, passes=8))
    top = lambda X: X["P"][np.argmin(np.linalg.norm(g[:, :2], axis=1)), 2]
    assert top(narrow) > 0.9 * g[:, 2].max()                                                             # a small reach leaves a bump this wide
    assert top(wide) < 0.6 * g[:, 2].max()                                                               # a wide reach takes well over a third off the top
