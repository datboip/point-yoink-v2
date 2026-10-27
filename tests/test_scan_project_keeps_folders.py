"""Closing a scan session never removes a folder it did not make (a scanner project has no PointYoink track file,
so it looks like "no sides"); a new, never-used project is still cleaned up."""
import os
from app2.scan.project import Project


def test_close_keeps_a_scanner_project(tmp_path):
    p = tmp_path / "Project10062026012338"; side = p / "data" / "10062026012338"; side.mkdir(parents=True)
    (side / "frame_000_0001.dph").write_bytes(b"x"); (p / "model.ply").write_bytes(b"ply")
    pr = Project(str(p), str(tmp_path))
    assert pr.sides == []
    pr.finish()
    assert (side / "frame_000_0001.dph").exists() and (p / "model.ply").exists()


def test_close_removes_an_unused_new_project(tmp_path):
    p = tmp_path / "ProjectLive10122026150000"
    Project(str(p), str(tmp_path)).finish()
    assert not os.path.exists(p)
