"""Clone: copy what is in a shape at the source spot into the same shape at the destination spot."""
import json
import numpy as np
import pytest
from scanner.stages.tools import clone as C


def spot(c, n=(0, 0, 1)): return json.dumps([list(c), list(n)])


P0 = {"src": spot((0, 0, 0)), "dst": spot((100, 0, 0)), "shape": "cylinder", "diameter": 20.0, "width": 20.0, "height": 20.0, "depth": 8.0, "rotation": 0.0, "mirror": "no"}


def bump_cloud():
    """A flat floor at z=0 with a bump (a marker) at (3, 2, 4) near the source spot; a flat floor at the destination."""
    g = np.array([[x, y, 0.0] for x in range(-30, 131, 5) for y in range(-20, 21, 5)], float)
    return {"P": np.vstack([g, [[3.0, 2.0, 4.0]]]), "N": np.tile([0.0, 0, 1], (len(g) + 1, 1)), "W": np.ones(len(g) + 1)}


def test_the_bump_is_copied_to_the_destination_and_the_old_destination_is_replaced():
    d = bump_cloud(); out, sel, st = C.apply("points", d, P0)
    P = out["P"]; assert len(out["N"]) == len(P) == len(out["W"])
    assert np.any(np.linalg.norm(P - [103.0, 2.0, 4.0], axis=1) < 1e-9)                # the bump, moved by +100 in x
    assert np.any(np.linalg.norm(P - [3.0, 2.0, 4.0], axis=1) < 1e-9)                  # the original is still there
    assert st["removed"] == int(sel.sum()) and st["added"] > 0


def test_rotation_about_the_destination_normal_and_mirror():
    d = bump_cloud(); out, _, _ = C.apply("points", d, dict(P0, rotation=90.0))
    assert np.any(np.linalg.norm(out["P"] - [100 - 2.0, 3.0, 4.0], axis=1) < 1e-9)     # (3, 2) turned 90 degrees about the spot
    out, _, _ = C.apply("points", d, dict(P0, mirror="yes"))
    assert np.any(np.linalg.norm(out["P"] - [100 - 3.0, 2.0, 4.0], axis=1) < 1e-9)     # mirrored across the first axis


def test_a_tilted_destination_surface_turns_the_copy_with_it():
    d = bump_cloud(); n = np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])      # the destination surface leans 30 degrees
    out, _, _ = C.apply("points", d, dict(P0, dst=spot((100, 0, 0), n)))
    want = [103.0, 2 * np.cos(np.radians(30)) + 4 * np.sin(np.radians(30)), -2 * np.sin(np.radians(30)) + 4 * np.cos(np.radians(30))]
    assert np.any(np.linalg.norm(out["P"] - want, axis=1) < 1e-9)


def test_mesh_copy_and_mirror_keeps_faces_pointing_outward():
    V = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [100, 0, 0], [104, 0, 0], [100, 4, 0], [200, 0, 0], [204, 0, 0], [200, 4, 0]], float)
    F = np.array([[0, 1, 2], [3, 4, 5]])                                   # the source triangle, and the old destination triangle
    (V2, F2, _), sel, st = C.apply("mesh", (V, F, None), dict(P0, shape="cylinder", diameter=20.0, depth=4.0))
    assert st == {"removed": 1, "added": 1} and len(F2) == 2
    n = lambda V_, f: np.cross(V_[f[1]] - V_[f[0]], V_[f[2]] - V_[f[0]])[2]
    assert n(V2, F2[-1]) > 0                                               # copied unmirrored: same winding
    (V3, F3, _), _, _ = C.apply("mesh", (V, F, None), dict(P0, mirror="yes"))
    assert n(V3, F3[-1]) > 0                                               # a mirror reverses winding; the tool turns it back


def test_errors_are_plain():
    d = bump_cloud()
    with pytest.raises(ValueError, match="source"): C.apply("points", d, dict(P0, src="[]"))
    with pytest.raises(ValueError, match="too close"): C.apply("points", d, dict(P0, dst=spot((5, 0, 0))))
    with pytest.raises(ValueError, match="nothing to copy"): C.apply("points", {"P": np.array([[500.0, 500, 500]])}, P0)
