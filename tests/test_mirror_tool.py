"""Mirror: a half part gets its other half across a plane; what is already there is not doubled; the plane is found and refined."""
import numpy as np
import pytest
from scanner.stages.tools import mirror as M


def half_sheet(full=False, step=1.0):
    """A bent sheet z = 0.02 x^2 over x in [0, 30] (or [-30, 30]) and y in [-20, 20], 1 mm grid, as a mesh."""
    xs = np.arange(-30 if full else 0, 30.01, step); ys = np.arange(-20, 20.01, step); X, Y = np.meshgrid(xs, ys)
    V = np.c_[X.ravel(), Y.ravel(), 0.02 * X.ravel() ** 2]; n = len(xs); m = len(ys); ids = np.arange(n * m).reshape(m, n); F = []
    for i in range(m - 1):
        for j in range(n - 1):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]; F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


PLANE = dict(nx=1.0, ny=0.0, nz=0.0, d=0.0, fit="no", gap_mm=1.5)


def test_mesh_half_becomes_whole_and_is_marked():
    V, F = half_sheet(); (V2, F2, _), sel, st = M.apply("mesh", (V, F, None), PLANE)
    assert st["added"] > 0.8 * len(F) and st["guessed"] == len(F2) - len(F)
    assert V2[:, 0].min() < -29 and np.allclose(V2[len(V):, 2], 0.02 * V2[len(V):, 0] ** 2)
    assert not sel.any() and st["agree"] < 0.2                       # a half part barely agrees with its mirror image
    old = np.sign(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 2])
    nz = np.sign(np.cross(V2[F2[len(F):, 1]] - V2[F2[len(F):, 0]], V2[F2[len(F):, 2]] - V2[F2[len(F):, 0]])[:, 2])
    assert (nz == old[0]).mean() > 0.99                              # the new faces face the same way as the old ones


def test_whole_part_adds_nothing():
    V, F = half_sheet(full=True)
    with pytest.raises(ValueError, match="adds nothing"): M.apply("mesh", (V, F, None), PLANE)
    assert M.agreement(V, np.array([1.0, 0, 0]), 0.0) > 0.99


def test_points_half_becomes_whole():
    V, _ = half_sheet(); d = dict(P=V, N=np.tile([0, 0, 1.0], (len(V), 1)), W=np.ones(len(V)), NF=np.ones(len(V)), S=np.ones(len(V), np.uint64), side_ids=[], cell=1.0)
    out, _sel, st = M.apply("points", d, PLANE)
    assert len(out["P"]) == len(V) + st["added"] and st["added"] > 0.8 * len(V) and len(out["W"]) == len(out["P"])


def test_auto_plane_finds_the_symmetry_of_a_whole_part():
    V, _ = half_sheet(full=True); V = V + np.array([5.0, 3.0, 1.0])
    n, d, score = M.auto_plane(V)
    assert abs(abs(n[0]) - 1) < 0.02 and abs(d - 5.0 * np.sign(n[0])) < 0.3 and score > 0.95


def test_refine_fixes_a_plane_picked_a_little_off():
    V, _ = half_sheet(full=True, step=0.4)
    n, d = M.refine_plane(V, np.array([1.0, 0.05, 0.0]), 0.8)
    assert abs(abs(n[0]) - 1) < 0.01 and abs(d) < 0.25


def test_job_writes_the_guess_mask(tmp_path):
    V, F = half_sheet()
    new, _sel, st = M.job_apply(dict(params=PLANE, out_dir=str(tmp_path)), "mesh", (V, F, None))
    g = np.unpackbits(np.load(tmp_path / "guessed.npy"))[: len(new[1])].astype(bool)
    assert g.sum() == st["guessed"] and "_guessed" not in st
