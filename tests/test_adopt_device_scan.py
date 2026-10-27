"""Adopting copied scanner poses does not read a device or alter raw frames."""
import hashlib
import json
import os
import subprocess
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-adopt-test-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XMODIFIERS"] = "@im=none"

from app2 import REPO
from app2.scan import fmt
from app2.scan.ui_model import ScanState
from app2.transfer.projects import import_projects
from scanner.adopt_device_scan import adopt_project, issue_summary, needs_preparation

sys.path.insert(0, str(Path(REPO) / "scanner"))
from stages.project import Project
from stages.engine import Engine
from fuse import read_global_poses


def pose(x=0):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def fake_side(project, name="side01", *, frames=3, global_poses=False):
    cache = project / "data" / name / "cache"
    cache.mkdir(parents=True)
    entries = []
    for index in range(frames):
        key = (index // 2, index % 2)
        stem = "frame_%03d_%04d" % key
        (cache / (stem + ".dph")).write_bytes(bytes([index + 1]) * 8)
        (cache / (stem + ".inf")).write_bytes(b"\0" * 16 + struct.pack("<16d", *pose(index).ravel()) +
                                               struct.pack("<16d", *np.eye(4).ravel()))
        entries.append((key, pose(index + 100 if key[0] else index)))
    if global_poses:
        raw = struct.pack("<i", len(entries))
        for key, matrix in entries:
            raw += struct.pack("<ii", *key) + struct.pack("<16d", *matrix.ravel())
        (cache / "global_register_pose.pose").write_bytes(raw)
    return cache


def pump(until=lambda: False, seconds=4):
    from PySide6.QtWidgets import QApplication
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if until():
            return True
        time.sleep(.01)
    return until()


def test_adopted_device_poses_load_as_real_build_sides(tmp_path):
    project = tmp_path / "Project09292026031500"
    cache = fake_side(project, global_poses=True)
    before = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in cache.glob("frame_*.*")}
    assert needs_preparation(project)
    report = adopt_project(project)
    assert (report["status"], report["frames"], report["tracked"], report["lost"]) == ("prepared", 3, 3, 0)
    track = json.loads((cache / "pointyoink_track.json").read_text())
    assert track["version"] == 1 and track["live"] is False
    assert track["frames"] == track["tracked"] == 3 and track["lost"] == []
    assert all(tuple(stamp) == fmt.STAMP_KEYS for stamp in track["stamps"])
    assert [stamp["frame"] for stamp in track["stamps"]] == ["frame_000_0000", "frame_000_0001", "frame_001_0000"]
    output = read_global_poses(cache / "pointyoink_register_pose.pose")
    assert len(output) == 3 and output[(1, 0)][0, 3] == 102
    build_project = Project.open(str(project), read_only=True)
    sides = build_project.sides(with_tracked=True)
    assert [(side.id, side.frames, side.tracked, side.lost) for side in sides] == [("side01", 3, 3, 0)]
    assert build_project.raw_stats("side01") == dict(frames=3, tracked=3, lost=0, live_plane=False)
    groups, graph, _estimate = Engine(build_project).status()
    assert [node.side for node in groups[0].nodes] == ["side01"]
    assert [side.id for side in graph.sides] == ["side01"]
    assert len(fmt.read_side(str(cache.parent))[0]) == 3
    assert {path.name: hashlib.sha256(path.read_bytes()).digest() for path in cache.glob("frame_*.*")} == before
    saved = {path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest())
             for path in (cache / "pointyoink_track.json", cache / "pointyoink_register_pose.pose")}
    assert adopt_project(project)["status"] == "nothing_to_prepare"
    assert {path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest())
            for path in (cache / "pointyoink_track.json", cache / "pointyoink_register_pose.pose")} == saved
    assert not needs_preparation(project)


def test_bad_or_missing_poses_pause_whole_project_without_writing(tmp_path):
    project = tmp_path / "ProjectRaw"
    first = fake_side(project, "side01", frames=2)
    second = fake_side(project, "side02", frames=2)
    (first / "frame_000_0001.inf").unlink()
    (second / "frame_000_0000.inf").write_bytes(b"\0" * 16 + struct.pack("<16d", *np.zeros((4, 4)).ravel()))
    report = adopt_project(project)
    assert report["status"] == "needs_decision" and report["lost"] == 2
    assert {(issue["side"], issue["frame"]) for issue in report["issues"]} == {
        ("side01", "frame_000_0001"), ("side02", "frame_000_0000")}
    assert "2 frames" in issue_summary(report) and "side01" in issue_summary(report)
    assert not list(project.glob("data/*/cache/pointyoink_track.json"))
    assert not list(project.glob("data/*/cache/pointyoink_register_pose.pose"))
    assert Project.open(str(project), read_only=True).sides() == []


def test_missing_global_pose_is_reported_without_falling_back_to_odometry(tmp_path):
    project = tmp_path / "ProjectWithResumedPass"
    cache = fake_side(project, global_poses=True)
    original = (cache / "global_register_pose.pose").read_bytes()
    (cache / "global_register_pose.pose").write_bytes(struct.pack("<i", 2) + original[4:4 + 2 * 136])
    report = adopt_project(project)
    assert report["status"] == "needs_decision" and report["lost"] == 1
    assert report["issues"] == [dict(side="side01", frame="frame_001_0000", reason="no global pose for this frame")]
    assert not (cache / "pointyoink_track.json").exists()


def test_command_line_reports_counts_and_prepares_copied_project(tmp_path):
    project = tmp_path / "ProjectCommand"
    fake_side(project, frames=2)
    result = subprocess.run([sys.executable, str(Path(REPO) / "scanner" / "adopt_device_scan.py"), str(project)],
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout)["tracked"] == 2
    assert len(Project.open(str(project), read_only=True).sides()) == 1


def test_models_only_skips_adoption_and_full_import_prepares_before_publish(tmp_path):
    source = tmp_path / "source" / "Projects" / "ProjectFull"
    fake_side(source)
    source_only = tmp_path / "models-only"
    fake_side(source_only)
    (source_only / ".pointyoink-models-only").touch()
    assert adopt_project(source_only)["status"] == "models_only"
    assert not needs_preparation(source_only)
    assert not (source_only / "data" / "side01" / "cache" / "pointyoink_track.json").exists()
    events = []
    import_projects(tmp_path / "source", tmp_path / "library", [source.name],
                    lambda kind, **data: events.append((kind, data)))
    saved = next(data for kind, data in events if kind == "saved")
    assert saved["prepared"]["status"] == "prepared"
    assert saved["prepared"]["frames"] == 3
    assert len(Project.open(saved["path"], read_only=True).sides()) == 1
    assert (source / "data" / "side01" / "cache" / "pointyoink_track.json").exists() is False


def test_wifi_placement_prepares_full_import(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app2.qt import theme
    from app2.qt.device_projects import DeviceProjects

    app = QApplication.instance() or QApplication([])
    theme.install(app)
    library = tmp_path / "library"
    stage = library / ".wifi-incoming" / "abc"
    fake_side(stage / "ProjectWifi")
    page = DeviceProjects(str(library))
    page._t0 = time.monotonic()
    page._place_received(str(stage))
    target = library / "ProjectWifi"
    assert len(Project.open(str(target), read_only=True).sides()) == 1
    assert "Received 1 project" in page.status.text()
    page.deleteLater()


def test_prepare_action_is_available_on_older_raw_import(tmp_path):
    from PySide6.QtWidgets import QApplication, QPushButton
    from app2.qt import theme
    from app2.qt.workspace_home import WorkspaceHome

    app = QApplication.instance() or QApplication([])
    theme.install(app)
    project = tmp_path / "ProjectOld"
    fake_side(project)
    home = WorkspaceHome(str(tmp_path))
    requested = []
    home.prepare_requested.connect(requested.append)
    home.refresh_projects()
    actions = [button for button in home.projects_popover.findChildren(QPushButton)
               if button.text() == "Prepare"]
    assert len(actions) == 1
    actions[0].click()
    assert requested == [str(project.resolve())]
    adopt_project(project)
    home.refresh_projects()
    assert not [button for button in home.projects_popover.findChildren(QPushButton)
                if button.isVisible() and button.text() == "Prepare"]
    home.deleteLater()


def test_prepare_card_opens_real_build_for_old_import(tmp_path):
    from PySide6.QtWidgets import QApplication, QPushButton
    from app2.qt import theme
    from app2.qt.shell import MainWindow

    app = QApplication.instance() or QApplication([])
    theme.install(app)
    project = tmp_path / "ProjectOld"
    fake_side(project)
    dummy = tmp_path / "Working"
    (dummy / "data").mkdir(parents=True)

    class Session:
        def __init__(self, bridge, path):
            self.bridge = bridge
            self.path = path

        def start(self):
            self.bridge("state", ScanState(project=self.path, n_sides=len(Project.open(self.path, read_only=True).sides()),
                                           any_frames=False, cam="gone", source_kind="live", source_label="", near=100,
                                           far=450, gain=4, auto_range=True, auto_gain=False, led=False, has_led=False,
                                           below_bed=False, remove_base=False, recording=False, building=False, busy=None,
                                           learning=False, capture_mode="continuous"))

        def close(self, then=None):
            self.bridge("closed", dict(then=then, project=self.path, lines=[], crashed=False))

        def cancel_build(self):
            pass

    win = MainWindow(lambda bridge, **over: Session(bridge, over.get("project") or str(dummy)),
                     gl=False, projects_dir=str(tmp_path))
    win.show()
    assert pump(lambda: win.state is not None)
    try:
        win.show_workspace()
        win.workspace.refresh_projects()
        button = next(button for button in win.workspace.projects_popover.findChildren(QPushButton)
                      if button.text() == "Prepare")
        button.click()
        assert pump(lambda: win.state is not None and win.state.project == str(project)
                    and win.stack.currentWidget() is win.build_slot, seconds=8)
        assert win.imported_project is None
        assert [side.id for side in Project.open(str(project), read_only=True).sides()] == ["side01"]
        assert win.workspace.ribbon_tools["Build"].isEnabled()
    finally:
        win._really = True
        win.build.shutdown()
        win.close()
        win.deleteLater()
        pump(seconds=.1)
