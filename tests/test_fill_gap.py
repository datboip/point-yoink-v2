"""Fill gap: a flat sheet with a square hole gets a patch on the same plane; no hole, or no data, says so; the faces are marked as a guess."""
import numpy as np
import pytest
from scanner.stages.tools import local_fix as L
from scanner.stages.tools import guess_fill as G


def sheet(hole=True, curved=False):
    g = np.arange(-30, 30.01, 1.0); X, Y = np.meshgrid(g, g); Z = (0.01 * (X ** 2 + Y ** 2)) if curved else np.zeros_like(X)
    V = np.c_[X.ravel(), Y.ravel(), Z.ravel()]; n = len(g)
    ids = np.arange(n * n).reshape(n, n); F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            cen = V[[a, b, c, d]].mean(0)
            if hole and abs(cen[0]) < 6 and abs(cen[1]) < 6: continue
            F += [[a, b, c], [a, c, d]]
    F = np.array(F); used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F]


P = dict(spot="[[0,0,0],[0,0,1]]", shape="cylinder", diameter=26.0, width=20.0, height=20.0, depth=6.0, action="fill")


def test_fills_a_flat_hole_on_the_plane():
    V, F = sheet()
    (V2, F2, _), sel, st = L.apply("mesh", (V, F, None), P)
    new = V2[len(V):]
    assert len(F2) > len(F) and st["guessed"] == len(F2) - len(F) and len(new) > 10
    assert np.abs(new[:, 2]).max() < 0.05 and (np.abs(new[:, :2]).max(0) < 7).all()
    # the new faces face the same way as the sheet
    nz = np.cross(V2[F2[len(F):, 1]] - V2[F2[len(F):, 0]], V2[F2[len(F):, 2]] - V2[F2[len(F):, 0]])[:, 2]
    assert (nz > 0).all()


def test_the_patch_meets_the_edge_of_the_gap():
    V, F = sheet(); (V2, F2, _), _, _ = L.apply("mesh", (V, F, None), P)
    assert len(G.boundary_loops(F)) == 2                   # the sheet's outer edge and the hole's
    left = [l for l in G.boundary_loops(F2) if np.abs(V2[l]).max() < 20]
    assert left == []                                      # nothing of the hole is left open


def test_continues_a_curved_surface():
    V, F = sheet(curved=True)
    (V2, _, _), _, st = L.apply("mesh", (V, F, None), P)
    new = V2[len(V):]; want = 0.01 * (new[:, 0] ** 2 + new[:, 1] ** 2)
    assert np.abs(new[:, 2] - want).max() < 0.05


def test_no_gap_and_points_are_refused():
    V, F = sheet(hole=False)
    with pytest.raises(ValueError, match="no gap"): L.apply("mesh", (V, F, None), P)
    with pytest.raises(ValueError, match="mesh"): L.apply("points", dict(P=V, N=V, W=V[:, 0], NF=V[:, 0], S=V[:, 0], side_ids=[], cell=0.4), P)


def test_job_writes_the_guess_mask(tmp_path):
    V, F = sheet()
    new, sel, st = L.job_apply(dict(params=P, out_dir=str(tmp_path)), "mesh", (V, F, None))
    g = np.unpackbits(np.load(tmp_path / "guessed.npy"))[: len(new[1])].astype(bool)
    assert g.sum() == st["guessed"] and "_guessed" not in st
