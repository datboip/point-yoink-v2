"""An empty base cut must stop the build instead of using the uncut side."""

import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import open3d as o3d

from test_stages_core import Base, CALLS
from test_stages_fixes import inproc, use
from stages.runner import StepError


def test_cut_script_rejects_margin_that_removes_all_faces(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    poses = struct.pack("<i", 4)
    for index, center in enumerate(((20, 0, 20), (0, 20, 20), (-20, 0, 20), (0, -20, 20))):
        matrix = np.eye(4)
        matrix[:3, 3] = center
        poses += struct.pack("<ii", 0, index) + struct.pack("<16d", *matrix.ravel())
    (cache / "pointyoink_register_pose.pose").write_bytes(poses)
    mesh = o3d.geometry.TriangleMesh.create_box(10, 10, 10)
    raw = tmp_path / "raw.ply"
    o3d.io.write_triangle_mesh(str(raw), mesh)
    plane = tmp_path / "plane.json"
    plane.write_text(json.dumps({"n": [0, 0, 1], "d": 0}))
    cut = tmp_path / "cut.ply"
    plane_out = tmp_path / "cut-plane.json"
    script = Path(__file__).resolve().parents[1] / "scanner" / "strip_table.py"
    result = subprocess.run([sys.executable, str(script), str(tmp_path), str(cut), "--mesh", str(raw),
                             "--plane", str(plane), "--plane-out", str(plane_out), "--margin", "16"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    marker = next(line for line in result.stdout.splitlines() if line.startswith("STAGE cut_rejected "))
    assert json.loads(marker[len("STAGE cut_rejected "):]) == {"reason": "empty_cut", "margin": 16.0}
    assert not cut.exists() and not plane_out.exists()


class TestEmptyBaseCut(Base):
    def test_empty_cut_stops_before_fused(self):
        use("side.platter", "auto")
        marker = "STAGE cut_rejected " + json.dumps({"reason": "empty_cut", "margin": 16}) + "\n"

        def fake_cut(_cmd):
            raise StepError("strip_table.py failed (exit 2)", marker, 2)

        inproc(self.e, fake_cut)
        result = self.e.run(to="fused", quiet=True)
        assert not result.ok
        assert result.failed == "side.platter@" + self.ids[0]
        assert "Side 1: base cut at 16 mm removed every triangle" in result.error
        assert "Lower the margin" in result.error
        assert "fused" not in CALLS
        assert self.e.output("fused") is None
