"""Side deletion must remain in Fused and follow the side's current pose."""

import numpy as np
import json
from types import SimpleNamespace
from fuse import parse_set_spec

from scanner.stages.graph import build
from scanner.stages.methods.fuse_plus import set_specs
from scanner.stages.project import Project
from scanner.stages.tools.edits import mask_side_depth
from test_stages_core import make_project


def test_active_side_delete_is_in_fused_inputs(tmp_path):
    ids = make_project(str(tmp_path), n=2)
    project = Project.open(str(tmp_path))
    op = {"id": "op-0001", "tool": "delete", "on": True, "frame": "side",
          "shape": "sphere", "center": [0, 0, 0], "radius": 2,
          "recorded": {"count": 12, "input_digest": "side-mesh"}}
    recipe = {"ops": {"side.keep@" + ids[0]: [op], "side.keep@" + ids[1]: [dict(op, on=False)]}}
    graph = build(project, recipe)
    fused = graph.by_id["fused"]
    assert fused.inputs["lineup"] == "lineup.adjust"
    assert fused.leaf["side_deletes"][ids[0]][0]["shape"] == "sphere"
    assert "recorded" not in fused.leaf["side_deletes"][ids[0]][0]
    assert ids[1] not in fused.leaf["side_deletes"]
    assert graph.by_id["meshed"].inputs["fused"] == graph.final("fused")
    assert "recorded" in op  # the saved side edit was not changed


def test_side_delete_removes_only_its_frame_pixels(tmp_path):
    pose = np.eye(4)
    pose[:3, 3] = [10, 0, 0]
    depth = np.full((2, 2), 100, np.uint16)
    op = {"frame": "side", "shape": "box",
          "box": {"min": [9, -1, -11], "max": [11, 1, -9], "T": None}}
    filtered = mask_side_depth(depth, pose, (1, 1, 0, 0), [op], str(tmp_path))
    assert filtered.tolist() == [[0, 100], [100, 100]]
    assert depth.tolist() == [[100, 100], [100, 100]]
    other_side = mask_side_depth(depth, pose, (1, 1, 0, 0), [], str(tmp_path))
    assert np.array_equal(other_side, depth)


def test_tsdf_and_plus_set_spec_passes_side_delete_file(tmp_path):
    project = SimpleNamespace(root=str(tmp_path), stages=str(tmp_path / "stages"))
    sides_path = tmp_path / "sides.json"
    sides_path.write_text(json.dumps({"sides": [{"id": "side-a", "frames": "frames", "calib": "Pl.bin",
                                                   "T": np.eye(4).tolist(), "plane": None}]}))
    op = {"shape": "sphere", "center": [0, 0, 0], "radius": 2, "tool": "delete"}
    ctx = SimpleNamespace(project=project, out_dir=str(tmp_path),
                          inputs={"lineup": SimpleNamespace(files={"sides": str(sides_path)})},
                          step=SimpleNamespace(leaf={"side_deletes": {"side-a": [op]}}))
    args, count = set_specs(ctx)
    assert count == 1
    parts = args[1].split(",")
    assert len(parts) == 5 and parts[3] == ""
    assert json.loads(open(parts[4]).read()) == {"ops": [op], "stages": project.stages}
    frames, calib, transform, plane, deletes = parse_set_spec(args[1])
    assert frames.endswith("/frames") and calib.endswith("/Pl.bin")
    assert np.allclose(transform, np.eye(4)) and plane is None
    assert deletes == {"ops": [op], "stages": project.stages}
