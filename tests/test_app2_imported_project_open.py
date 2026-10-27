"""Scanner-finished sides open for viewing and export without using a device."""
import hashlib
import os
import tempfile
import time
from pathlib import Path

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-imported-open-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XMODIFIERS"] = "@im=none"

from PIL import Image
from PySide6.QtWidgets import QApplication

from app2.imported_projects import RAW_ONLY_MESSAGE, imported_project
from app2.project_preview import project_rows
from app2.transfer.projects import _model_counts
from app2.qt import theme
from app2.qt.shell import MainWindow
from app2.scan.ui_model import ScanState


MESH = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        b"0 0 0\n10 0 0\n0 10 0\n3 0 1 2\n")
POINTS = (b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
          b"end_header\n0 0 0\n1 1 1\n")


def pump(until=lambda: False, seconds=2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if until(): return True
        time.sleep(.01)
    return until()


def scanner_project(root, name="Project09292026031500", *, models=("fuse_mesh.ply",), raw=False, marker=False):
    project = root / name
    for i, model in enumerate(models):
        side = project / "data" / ("side%02d" % (i + 1)); side.mkdir(parents=True)
        if model:
            (side / model).write_bytes(POINTS if model == "fuse.ply" else MESH)
        Image.new("RGB", (64, 48), (20 + i * 20, 30, 40)).save(side / "preview.png")
        if raw:
            cache = side / "cache"; cache.mkdir(); (cache / "frame_000_0001.dph").write_bytes(b"raw")
    if marker: (project / ".pointyoink-models-only").touch()
    return project


def test_scanner_models_and_raw_only_are_classified_without_tracker(tmp_path):
    models = scanner_project(tmp_path, models=("fuse_mesh.ply", "fuse.ply"), raw=True, marker=True)
    kind, sides = imported_project(models)
    assert kind == "models" and [side["name"] for side in sides] == ["side01", "side02"]
    assert sides[0]["faces"] and not sides[1]["faces"]
    assert all(Path(side["preview"]).is_file() for side in sides)
    assert project_rows(tmp_path)[0]["sides"] == 2
    assert _model_counts(models)[0] == 2
    raw = scanner_project(tmp_path, "RawOnly", models=(None,), raw=True)
    assert imported_project(raw) == ("raw_only", [])
    (models / "data" / "side01" / "cache" / "pointyoink_track.json").touch()
    assert imported_project(models) == ("tracked", [])


def test_models_open_in_read_only_build_view_and_export_without_changing_source(tmp_path):
    app = QApplication.instance() or QApplication([]); theme.install(app)
    project = scanner_project(tmp_path, models=("fuse_mesh.ply", "fuse_mesh.ply"), marker=True)
    dummy = tmp_path / "Working"; (dummy / "data").mkdir(parents=True)
    made = []

    class Session:
        def __init__(self, bridge, path): self.bridge = bridge; self.path = path; self.closes = []
        def start(self):
            self.bridge("state", ScanState(project=self.path, n_sides=0, any_frames=False, cam="gone", source_kind="live",
                                           source_label="", near=100, far=450, gain=4, auto_range=True, auto_gain=False,
                                           led=False, has_led=False, below_bed=False, remove_base=False, recording=False,
                                           building=False, busy=None, learning=False, capture_mode="continuous"))
        def close(self, then=None):
            self.closes.append(then); self.bridge("closed", dict(then=then, project=self.path, lines=[], crashed=False))
        def cancel_build(self): pass

    def make(bridge, **over):
        session = Session(bridge, over.get("project") or str(dummy)); made.append(session); return session

    win = MainWindow(make, gl=False, projects_dir=str(tmp_path)); win.show(); pump()
    try:
        win._workspace_project(str(project))
        assert pump(lambda: win.imported_project == str(project) and win.stack.currentWidget() is win.build_slot)
        page = win.imported_models
        assert made[0].closes == [("imported", str(project))]
        assert win.build_layout.currentWidget() is page
        assert page.current_side()["name"] == "side01" and page.vp.main.tile.img is not None
        assert pump(lambda: page.vp.main.tile.img.width() == 900)
        assert win.workspace.ribbon_tools["Export"].isEnabled()
        assert not win.workspace.ribbon_tools["Edit"].isEnabled()
        assert not win.workspace.build_tools.isVisible()
        page.buttons[1].click(); assert page.current_side()["name"] == "side02"
        assert pump(lambda: page.vp.main.tile.img.width() == 900)
        source = Path(page.current_side()["path"]); before = hashlib.sha256(source.read_bytes()).digest()
        out = tmp_path / "output.stl"; page.ask_save = lambda *_args: str(out)
        win._menu_action("export")
        assert pump(lambda: out.is_file() and page.export_future is None)
        assert out.read_bytes().startswith(b"PointYoink binary STL")
        ply = tmp_path / "output.ply"; page.ask_save = lambda *_args: str(ply)
        page._choose_export("ply")
        assert pump(lambda: ply.is_file() and page.export_future is None), win.toast.text.text()
        assert hashlib.sha256(source.read_bytes()).digest() == before
        win.show_workspace(); win._workspace_continue()
        assert win.stack.currentWidget() is win.build_slot and page.current_side()["name"] == "side02"
        win.show_workspace(); assert win.workspace.continue_b.text() == "View scanner model"
        win._menu_action("new")
        assert win.imported_project is None and win.stack.currentWidget() is win.scan
        assert win.build_layout.currentWidget() is win.build and len(made) == 2
        assert pump(lambda: win.state is not None)
        win.show_step(2); assert win.edit.build is win.build
        win._workspace_project(str(project))
        assert pump(lambda: win.imported_project == str(project))
        assert win.edit.build is None and win.build_layout.currentWidget() is page
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); pump(seconds=.1)


def test_point_cloud_exports_ply_and_raw_only_stays_on_projects(tmp_path):
    app = QApplication.instance() or QApplication([]); theme.install(app)
    project = scanner_project(tmp_path, models=("fuse.ply",))
    raw = scanner_project(tmp_path, "RawOnly", models=(None,), raw=True)
    win = MainWindow(lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("offline")), gl=False, projects_dir=str(tmp_path))
    win.show(); win.startup.ready(); pump(seconds=.1)
    try:
        win._workspace_project(str(raw))
        assert win.stack.currentWidget() is win.workspace
        assert RAW_ONLY_MESSAGE == win.toast.text.text()
        win.pick_folder = lambda: str(raw)
        win._workspace_open()
        assert win.stack.currentWidget() is win.workspace and RAW_ONLY_MESSAGE == win.toast.text.text()
        win._workspace_project(str(project)); pump()
        page = win.imported_models
        assert win.stack.currentWidget() is win.build_slot
        assert not page.export_stl_b.isEnabled() and page.export_as_b.isEnabled()
        assert pump(lambda: page.vp.main.tile.img.width() == 560)
        out = tmp_path / "points.ply"; page.ask_save = lambda *_args: str(out)
        page._choose_export("ply")
        assert pump(lambda: out.is_file() and page.export_future is None)
        assert out.read_bytes() == POINTS
        assert (project / "data" / "side01" / "fuse.ply").read_bytes() == POINTS
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); pump(seconds=.1)
