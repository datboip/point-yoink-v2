"""Workflow fixes: a saved project opens and builds without a scanner, lands on Build when it has sides, the header's
scanner pill only says connected once frames come in, Export works before Build was opened (and says so when it falls
back to an older model.ply), no sides means no rebuild and a reason, and YOLO Yoink closes when the page changes.
Offscreen, no scanner (POINTYOINK_NO_DEVICE=1), an isolated config, fake sessions and a fake build engine in temp
folders: nothing outside them is read or written."""
import os
import tempfile
import time
from pathlib import Path

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-workflow-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["XMODIFIERS"] = "@im=none"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app2.qt import theme
from app2.scan.ui_model import ScanState
from app2.builder.stagehost import Snapshot


def app():
    a = QApplication.instance() or QApplication([])
    theme.install(a); return a


def pump(cond=lambda: False, secs=0.3):
    t0 = time.time()
    while time.time() - t0 < secs:
        QApplication.processEvents(); time.sleep(0.01)
        if cond(): return True
    return cond()


def project(root, name, sides=0, model=False):
    p = Path(root) / name; (p / "data").mkdir(parents=True)
    for i in range(sides): (p / "data" / ("0926202601552%d" % i) / "cache").mkdir(parents=True)
    if model: (p / "model.ply").write_bytes(b"ply\n")
    return str(p)


def state(path, n_sides=0, cam=None, **kw):
    base = dict(project=path, n_sides=n_sides, any_frames=bool(n_sides), cam=cam, source_kind="live", source_label="Scanner in PC mode",
                near=100, far=450, gain=4, auto_range=True, auto_gain=False, led=False, has_led=False, below_bed=False,
                remove_base=False, recording=False, building=False, busy=None, learning=False, capture_mode="continuous")
    base.update(kw); return ScanState(**base)


class FakeSession:
    """What the window needs of a ScanSession: start sends a state, close says closed, build is recorded."""
    def __init__(self, bridge, path, root):
        self.bridge = bridge; self.path = path or project(root, "ProjectLiveNew%d" % time.time_ns()); self.runs = []; self.closes = []

    def start(self):
        n = sum(1 for x in os.listdir(os.path.join(self.path, "data")) if os.path.isdir(os.path.join(self.path, "data", x, "cache")))
        self.bridge("state", state(self.path, n)); return self

    def close(self, then=None):
        self.closes.append(then); self.bridge("closed", dict(then=then, project=self.path, lines=[], crashed=False))

    def build(self, backend): self.runs.append(backend)
    def new_side(self): pass
    def cancel_build(self): pass


def raw_node(n=1): return dict(id="side.fuse@s%d" % n, status="fresh", stats=dict(frames=10))


class FakeEngine:
    """The build engine's answers: a Snapshot (raw sides, Meshed's status and file) and the h_* helpers used here."""
    def __init__(self, root, raw=1, meshed="fresh", mesh_file=None):
        self.root = root; self.raw = raw; self.meshed = meshed; self.exported = []
        model = os.path.join(root, "model.ply")             # a built Meshed has its file (the project's model here)
        self.mesh_file = mesh_file or (model if meshed == "fresh" and os.path.exists(model) else None)

    def snapshot(self):
        raw = [raw_node(i + 1) for i in range(self.raw)]
        files = {"meshed": {"mesh": self.mesh_file}} if self.mesh_file else {}
        return Snapshot(groups=[dict(name="raw", nodes=raw), dict(name="meshed", nodes=[dict(id="meshed", status=self.meshed, stats={})])],
                        sides=[dict(id="s%d" % (i + 1), n=i + 1, active=True) for i in range(self.raw)], files=files,
                        chains={}, finals={}, entries=[dict(note="start")], cursor=0, running=False, recipe={}, est_s=60,
                        fused_method="tsdf", meshed_method="auto", methods={})

    def h_export_stl(self, src, dst):
        self.exported.append((src, dst)); Path(dst).write_bytes(b"solid\n"); return dict(bytes=2_000_000, faces=1234)

    def h_export(self, src, dst, fmt="stl", zip_root=None):
        self.exported.append((src, dst, fmt, zip_root)); Path(dst).write_bytes(b"solid\n")
        return dict(bytes=2_000_000, faces=1234, dims="10.0 x 20.0 x 30.0 mm", sides=1)

    def h_model_dims(self, src): return dict(w=10.0, d=20.0, h=30.0, text="10.0 x 20.0 x 30.0 mm")

    def h_onetap(self, preset):
        return dict(preset=preset, fusion="Standard", ftools="", mesh="Auto", mtools="", deletes=0, to_run=3, est_s=60)

    def h_custom_onetap(self, fragment, preset):
        return self.h_onetap(preset)

    def h_plan(self, *a, **k): return dict(to_run=3, est_s=60)


@pytest.fixture
def window(tmp_path):
    from app2.qt.shell import MainWindow
    app(); made = []; engines = {}

    def make(bridge, **over):
        s = FakeSession(bridge, over.get("project"), str(tmp_path)); made.append(s); return s
    win = MainWindow(make, gl=False, projects_dir=str(tmp_path))
    win.build.engine_factory = lambda root: engines.setdefault(root, FakeEngine(root))
    win.resize(1688, 1044); win.show(); pump()
    win.made, win.engines = made, engines
    yield win
    # Keep the unparented bridges alive until their owned workers and queued
    # deliveries have finished. close(0) is deliberately nonblocking in the app.
    from PySide6.QtCore import QCoreApplication, QEvent
    bridges = [win.build.sb, win.build.vb, win.imported_models.bridge]
    workers = [win.build.host, win.build.loader, win.imported_models.loader]
    for worker in workers:
        if worker is not None:
            worker.close(3.0)
            assert not worker.thread.is_alive(), "Owned fixture worker did not stop"
    win.build.shutdown(0.5)
    win.imported_models.shutdown()
    for executor in (win.imported_models.executor, win.imported_models.picture_executor,
                     win._prepare_executor):
        executor.shutdown(wait=True, cancel_futures=True)
    for bridge in bridges:
        if bridge is not None:
            QCoreApplication.sendPostedEvents(bridge, QEvent.Type.MetaCall)
    app().processEvents()
    win._really = True; win.close()
    assert not win.isVisible(), "Owned fixture window refused to close"
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app().processEvents()


def on_build(win): return win.stack.currentWidget() is win.build_slot


# ------------------------------------------------------------------------------------------------ 1 and 2: no scanner
def test_offline_source_opens_without_a_scanner_and_says_not_connected():
    from app2.__main__ import offline_safe_source
    src = offline_safe_source(); got = []
    src.start(got.append)                                     # POINTYOINK_NO_DEVICE=1: refused, and the session lives on
    assert src.offline and src.dead and not src.present()
    assert src.gone_text().startswith("Scanner not connected") and "built" in src.gone_text()
    src.stop(); assert not src.running and got == []


def test_calibration_comes_from_the_project_when_the_default_is_missing(tmp_path):
    from app2.__main__ import find_param, NO_CALIBRATION
    p = project(tmp_path, "ProjectLiveA", sides=1); side = os.path.join(p, "data", os.listdir(os.path.join(p, "data"))[0])
    os.makedirs(os.path.join(side, "param")); Path(side, "param", "Pl.bin").write_bytes(b"\0" * 40)
    missing = str(tmp_path / "no-such-param")
    assert find_param(None, p, str(tmp_path), fallback=missing) == os.path.join(side, "param")
    assert find_param(None, None, str(tmp_path), fallback=missing) == os.path.join(side, "param")     # a new project: the newest side's
    empty = tmp_path / "empty-root"; empty.mkdir()
    assert find_param(None, None, str(empty), fallback=missing) is None
    assert "no camera files yet" in NO_CALIBRATION and "Small part" in NO_CALIBRATION   # the text was rewritten with the device steps (2026-10-24)


def test_build_stays_usable_without_a_scanner_and_says_why_when_the_session_stopped(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLive1", sides=1)
    win._workspace_project(p); pump(lambda: win.build.snap is not None, 5)
    b = win.build
    assert on_build(win) and b.snap is not None and b.session is not None
    assert b.onetap.isEnabled() and b.vp.rebuild.btn.isEnabled() and b.off_note.isHidden()
    win.bridge("closed", dict(then=None, project=p, lines=[], crashed=True)); pump()
    assert not b.onetap.isEnabled() and not b.off_note.isHidden() and "Projects" in b.off_note.text()
    assert "Projects" in win.scan.banner.more.text() and "New project or Open project" not in win.scan.banner.more.text()
    assert win.header._source[0] == "Session stopped"


# ------------------------------------------------------------------------------------------------ 3: where Open lands
def test_open_lands_on_build_with_sides_on_scan_without_and_the_open_project_keeps_its_session(window, tmp_path):
    win = window
    with_sides = project(tmp_path, "ProjectLiveS", sides=2); empty = project(tmp_path, "ProjectLiveE")
    win._workspace_project(with_sides); pump(lambda: getattr(win.state, "project", None) == with_sides, 3)
    assert on_build(win)
    n = len(win.made); closes = list(win.made[-1].closes)
    win.show_workspace(); win._workspace_project(with_sides); pump()
    assert on_build(win) and len(win.made) == n and win.made[-1].closes == closes     # already open: no new session
    win.pick_folder = lambda: with_sides; win.show_workspace(); win._workspace_open(); pump()
    assert on_build(win) and len(win.made) == n
    win._workspace_project(empty); pump(lambda: getattr(win.state, "project", None) == empty, 3)
    assert win.stack.currentWidget() is win.scan
    win.show_workspace(); win._workspace_continue(); assert win.stack.currentWidget() is win.scan


def test_refused_open_and_non_project_folder_say_so(window, tmp_path):
    win = window
    win.pick_folder = lambda: str(tmp_path)                    # no data folder
    win._workspace_open(); assert win.toast.isVisible() and "not a scan project" in win.toast.text.text()
    win.state.busy = "Saving side 1..."
    win._workspace_project(project(tmp_path, "ProjectLiveB", sides=1))
    assert "Wait for" in win.toast.text.text()


# ------------------------------------------------------------------------------------------------ 4: the header
def test_scanner_pill_says_connected_only_once_frames_come(window, tmp_path):
    win = window; pump(lambda: win.state is not None, 3)
    assert win.header._source == ("Scanner not connected", "ink3")          # a state, no frames yet: not green
    win.bridge("state", state(win.state.project, cam="gone")); pump()
    assert win.header._source[0] == "Scanner not connected"
    win.bridge("state", state(win.state.project)); win.bridge("live", object()); pump()
    assert win.header._source == ("Scanner in PC mode", "kept")
    win.bridge("state", state(win.state.project, cam="gone")); pump()
    assert win.header._source[0] == "Scanner disconnected"


def test_import_hides_the_project_name_and_pill_and_the_name_saves_on_focus_out(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveN", sides=1)
    win._workspace_project(p); pump(lambda: getattr(win.state, "project", None) == p, 3)
    win.show_import(); pump()
    assert win.header.isVisible() and not win.header.folder.isVisible() and not win.header.source.isVisible()
    win.device_projects.dismiss(); pump()
    win.show_step(1); pump()
    assert win.header.folder.isVisible() and win.header.source.isVisible()
    named = []; win.header.project_named.connect(named.append)
    win.activateWindow(); win.header.folder.setFocus(); win.header.folder.setText("Headrest"); pump()
    win.header.folder.clearFocus(); pump()
    assert named == ["Headrest"]


# ------------------------------------------------------------------------------------------------ 6: Export
def test_export_is_on_before_build_was_shown_and_its_result_shows_on_any_page(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveX", sides=1, model=True)
    win.pick_folder = lambda: p
    win._workspace_open(); pump(lambda: win.build.snap is not None, 5)
    win.show_step(0); pump()
    assert win.workspace.ribbon_tools["Export"].isEnabled() and win.menu_actions["export"].isEnabled()
    win.build.ask_save = lambda default, *a: str(tmp_path / "out.stl")
    win.workspace.ribbon_tools["Export"].click()                          # the Export sheet, STL picked
    d = win.build.export_dlg; assert d.isVisible() and d.choice() == ("stl", False)
    assert pump(lambda: "10.0 x 20.0 x 30.0 mm" in d.size.text(), 5), d.size.text()
    d.ok_b.click()
    assert pump(lambda: "Saved out.stl" in win.toast.text.text(), 5), win.toast.text.text()
    assert win.stack.currentWidget() is win.scan and win.toast.isVisible()


def test_export_from_a_meshed_result_before_build_opens(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveM", sides=1)          # no model.ply: only Meshed's own file
    mesh = Path(p) / "stages" / "meshed.ply"; mesh.parent.mkdir(); mesh.write_bytes(b"ply\n")
    win.engines[p] = FakeEngine(p, mesh_file=str(mesh))
    win.bridge("state", state(p, 1)); pump(lambda: win.build.snap is not None, 5)
    assert not win.build.isVisible() and win.workspace.ribbon_tools["Export"].isEnabled()


def test_export_refuses_an_older_model_while_meshed_is_out_of_date(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveO", sides=1, model=True)
    win.engines[p] = FakeEngine(p, meshed="stale", mesh_file=str(Path(p) / "model.ply"))
    win.bridge("state", state(p, 1)); pump(lambda: win.build.snap is not None, 5)
    win.show_workspace(); pump(); saved = []; win.build.ask_save = lambda default, *a: saved.append(default) or str(tmp_path / "old.stl")
    assert not win._menu_states()["export"] and not win.workspace.ribbon_tools["Export"].isEnabled()
    win._menu_action("export"); win.build.export_stl(); pump()               # greyed, and asked anyway: it says why
    assert "Rebuild Meshed before exporting" in win.toast.text.text(), win.toast.text.text()
    assert not saved and not win.engines[p].exported


# ------------------------------------------------------------------------------------------------ 7: no sides
def test_no_sides_no_rebuild_bar_and_a_reason(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveZ")
    win.engines[p] = FakeEngine(p, raw=0, meshed="waiting")
    win._workspace_project(p); pump(lambda: getattr(win.state, "project", None) == p, 3)
    win.show_step(1); pump(lambda: win.build.snap is not None, 5)
    b = win.build
    assert b.vp.rebuild.isHidden()
    assert not b.onetap.isEnabled() and not b.vp.rebuild.btn.isEnabled()
    assert b.off_note.isVisible() and b.off_note.text().startswith("No sides yet")
    assert b._stage_sub("raw") == "0 sides, 0 frames"
    win.engines[p].raw = 1; b.host.status(); pump(lambda: b.onetap.isEnabled(), 5)
    assert b._stage_sub("raw") == "1 side, 10 frames" and b._stage_sub("lineup").startswith("1 side in")


# ------------------------------------------------------------------------------------------------ 8: YOLO Yoink
def test_yolo_opens_on_a_picked_preset_and_closes_when_the_page_changes(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveY", sides=1)
    win._workspace_project(p); pump(lambda: win.build.onetap.isEnabled(), 5)
    d = win.build.onetap_dlg
    win.build.open_onetap(); pump()
    assert d.isVisible() and d.current and d.profile_cards[d.current].isChecked()
    win.show_step(0); pump()
    assert not d.isVisible() and not win.build.scrim.isVisible()
    win.show_step(1); win.build.open_onetap(); pump(); assert d.isVisible()
    win.show_workspace(); pump(); assert not d.isVisible()
    d.current = None; d.run()                                   # nothing picked: no crash, nothing sent


# ------------------------------------------------------------------------------------------------ 10: no placeholder pages
def test_no_navigation_reaches_the_placeholder_steps(window):
    win = window
    names = [a.text().split("\t")[0] for a in win.menus["View"].actions()[-1].menu().actions()]
    assert names == ["Scan", "Build", "Edit"]
    assert not any(s.key().toString() in ("Ctrl+4", "Ctrl+5", "Ctrl+6") for s in win._keys)
    for k in (3, 4): win.show_step(k); assert win.stack.currentWidget() not in win.placeholders


def test_export_sheet_format_goes_to_the_host_and_is_remembered_and_meshed_shows_the_size(window, tmp_path):
    win = window; p = project(tmp_path, "ProjectLiveF", sides=1, model=True)
    win.bridge("state", state(p, 1)); pump(lambda: win.build.snap is not None, 5)
    saved = []
    win.build.ask_save = lambda default, *a: saved.append((default,) + a) or str(tmp_path / "part")
    win._menu_action("export_as"); d = win.build.export_dlg; assert d.isVisible()
    d.fmt_b["obj"].click(); d.ok_b.click()
    assert pump(lambda: "Saved part.obj" in win.toast.text.text(), 5), win.toast.text.text()
    assert "10.0 x 20.0 x 30.0 mm" in win.toast.text.text() and not d.isVisible()
    assert saved[0][0].endswith("ProjectLiveF_print.obj") and "OBJ" in saved[0][1]
    assert win.engines[p].exported[-1][2:] == ("obj", None)
    win.workspace.ribbon_tools["Export"].click(); assert d.isVisible() and d.choice() == ("obj", False)
    d.zip.setChecked(True); d.ok_b.click()
    assert pump(lambda: "Saved part.zip" in win.toast.text.text(), 5), win.toast.text.text()
    assert win.engines[p].exported[-1][2:] == ("obj", p)
    win.show_step(1); win.build.show_stage("meshed")
    assert pump(lambda: "Size: 10.0 x 20.0 x 30.0 mm" in win.build.info.lines[1].text(), 5), win.build.info.lines[1].text()
