import numpy as np

from scanner.stages import plyio
from scanner.stages.graph import default_ops
from scanner.stages.methods.fuse_tsdf import FusedTsdf
from scanner.stages.tools.match_sides import apply


def test_match_cleaned_mesh_uses_placed_side_and_keeps_near_faces(tmp_path):
    side_vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    faces = np.array([[0, 1, 2]], np.int32)
    side_file = tmp_path / "side.ply"
    plyio.write_mesh(str(side_file), side_vertices, faces, None)
    transform = np.eye(4)
    transform[0, 3] = 10
    sides = [{"mesh": "side.ply", "T": transform.tolist()}]
    fused_vertices = np.vstack((side_vertices + [10, 0, 0], side_vertices + [20, 0, 0]))
    fused_faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    out, removed, stats = apply("mesh", (fused_vertices, fused_faces, None), sides, str(tmp_path), 1.5)
    assert removed.tolist() == [False, True]
    assert len(out[1]) == 1
    assert stats["removed_pct"] == 50


def test_match_cleaned_points_and_default_off(tmp_path):
    side_vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    side_file = tmp_path / "side.ply"
    plyio.write_mesh(str(side_file), side_vertices, np.array([[0, 1, 2]], np.int32), None)
    sides = [{"mesh": "side.ply", "T": np.eye(4).tolist()}]
    data = {"P": np.array([[0., 0., 0.], [10., 0., 0.]]), "N": np.zeros((2, 3)),
            "W": np.array([1., 1.]), "NF": np.array([1., 1.]), "S": np.array([1, 1], np.uint64),
            "side_ids": np.array(["one"]), "cell": np.array(0.4)}
    out, removed, stats = apply("points", data, sides, str(tmp_path), 1.5)
    assert removed.tolist() == [False, True]
    assert out["P"].shape == (1, 3)
    assert out["W"].tolist() == [1.]
    assert out["cell"].shape == ()
    assert default_ops(FusedTsdf())[-1]["on"] is False
