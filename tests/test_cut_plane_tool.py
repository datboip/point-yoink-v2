"""Cut plane tool: everything below the plane goes, cuts stack, mesh and points behave the same."""
import numpy as np
from scanner.stages.tools import plane_cut as C

P0 = {"nx": 0, "ny": 0, "nz": 1, "d": 10.0, "margin": 0.0}

def test_points_below_plane_removed_and_arrays_stay_aligned():
    P = np.array([[0, 0, 0], [0, 0, 9.9], [0, 0, 10.1], [5, 5, 50]], float)
    data = {"P": P, "N": np.ones((4, 3)), "cell": 0.4}
    out, sel, st = C.apply("points", data, P0)
    assert sel.tolist() == [True, True, False, False] and len(out["P"]) == 2 and len(out["N"]) == 2 and out["cell"] == 0.4
    assert st["removed"] == 2 and st["removed_pct"] == 50.0

def test_margin_moves_the_plane_up():
    data = {"P": np.array([[0, 0, 10.5], [0, 0, 12.0]], float)}
    out, sel, _ = C.apply("points", data, dict(P0, margin=1.0))
    assert sel.tolist() == [True, False]

def test_mesh_faces_below_go_and_unused_vertices_drop():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 20], [1, 0, 20], [0, 1, 20]], float)
    F = np.array([[0, 1, 2], [3, 4, 5]])
    (V2, F2, _), sel, _ = C.apply("mesh", (V, F, None), P0)
    assert sel.tolist() == [True, False] and len(V2) == 3 and F2.shape == (1, 3) and V2[:, 2].min() == 20

def test_two_cuts_stack():
    data = {"P": np.array([[0, 0, z] for z in (1.0, 5.0, 9.0, 15.0, 30.0)], float)}
    a, _, _ = C.apply("points", data, dict(P0, d=4.0))
    b, _, _ = C.apply("points", a, dict(P0, d=12.0))
    assert b["P"][:, 2].tolist() == [15.0, 30.0]

def test_tilted_normal_is_normalized():
    data = {"P": np.array([[0, 0, 0], [10, 0, 0]], float)}
    out, sel, _ = C.apply("points", data, {"nx": 5, "ny": 0, "nz": 0, "d": 4.0})
    assert sel.tolist() == [True, False]


def test_cut_stays_inside_the_spots_outline():
    import json
    sq = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
    p = dict(P0, d=1.0, poly=C.outline(sq, [0, 0, 1]), pad=1.0)
    X = np.array([[5, 5, 0], [5, 5, 50], [50, 5, 0], [10.5, 5, 0], [-3, 5, 0]], float)
    assert C.below(X, p).tolist() == [True, False, False, True, False]       # below AND inside (pad 1 mm); far outside is kept

def test_without_an_outline_it_is_the_whole_plane():
    X = np.array([[500, 500, 0], [0, 0, 20]], float)
    assert C.below(X, P0).tolist() == [True, False]
