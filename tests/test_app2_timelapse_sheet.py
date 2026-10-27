"""The Export scan timelapse sheet and its wiring: File > Export scan timelapse... and the Export sheet's link open it,
its choices make the child's command line, Export runs the child (python -m app2.scan.timelapse) to a clip in the
project folder with progress, Stop leaves nothing behind, and the window's notice offers Open folder. Offscreen, no
scanner, an isolated config, a synthetic project in a temp folder."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-tlsheet-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["XMODIFIERS"] = "@im=none"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt import theme
from app2.qt import build_words as W
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_app2_timelapse import make_project                     # noqa: E402 (the synthetic ball side)


def app():
    a = QApplication.instance() or QApplication([]); theme.install(a); return a


def pump(cond=lambda: False, secs=0.3):
    t0 = time.time()
    while time.time() - t0 < secs:
        QApplication.processEvents(); time.sleep(0.01)
        if cond(): return True
    return cond()


@pytest.fixture
def proj(tmp_path): return make_project(str(tmp_path))


@pytest.fixture
def sheet():
    from app2.qt.timelapse_sheet import TimelapseSheet
    app(); host = QWidget(); host.resize(1280, 800); host.show(); d = TimelapseSheet(host)
    yield d
    if d.busy: d.proc.kill(); d.proc.waitForFinished(3000)
    host.close(); host.deleteLater(); app().processEvents()


def test_choices_make_the_command_line(sheet, proj):
    sheet.present(proj, 3, formats=("mp4", "gif"))
    assert sheet.isVisible() and [sheet.sides.itemText(i) for i in range(4)] == ["All sides", "Side 1", "Side 2", "Side 3"]
    assert sheet.choice == dict(layout="model", size="720p", length=8, format="mp4") and sheet.orbit.isChecked() and sheet.mark.isChecked()
    sheet.pick("format", "gif"); assert sheet.choice["size"] == "480p" and sheet.seg["size"]["480p"].isChecked()
    assert sheet.where.text() == "Saves as ProjectLive01012026000000_timelapse.gif in the project folder."
    sheet.pick("layout", "camera+model"); sheet.pick("length", 15); sheet.sides.setCurrentIndex(2); sheet.mark.setChecked(False); sheet.orbit.setChecked(False)
    a = sheet.args()
    assert a[:3] == ["-m", "app2.scan.timelapse", proj] and a[a.index("-o") + 1] == os.path.join(proj, "ProjectLive01012026000000_timelapse.gif")
    for k, v in (("--format", "gif"), ("--size", "480p"), ("--layout", "camera+model"), ("--seconds", "15"), ("--turns", "0"), ("--side", "2")):
        assert a[a.index(k) + 1] == v
    assert "--no-watermark" in a
    sheet.pick("format", "mp4"); assert sheet.choice["size"] == "720p"
    sheet.cancel(); assert not sheet.isVisible()


def test_no_ffmpeg_means_gif_only(sheet, proj):
    sheet.present(proj, 1, formats=("gif",))
    assert sheet.choice["format"] == "gif" and not sheet.seg["format"]["mp4"].isEnabled()
    assert sheet.hint.text() == W.TIMELAPSE_SHEET["no_mp4"]


def test_export_runs_the_child_and_stop_leaves_nothing(sheet, proj):
    sheet.present(proj, 1, formats=("mp4", "gif")); sheet.pick("format", "gif"); sheet.pick("length", 5); sheet.orbit.setChecked(False)
    done = []; sheet.done.connect(done.append); seen = []
    sheet.bar.valueChanged.connect(seen.append)
    sheet.run(); assert sheet.busy and not sheet.ok_b.isEnabled() and sheet.cancel_b.text() == "Stop"
    assert pump(lambda: done, 180), sheet.status.text()
    r = done[0]; out = os.path.join(proj, "ProjectLive01012026000000_timelapse.gif")
    assert r["path"] == out and os.path.getsize(out) == r["bytes"] and r["frames"] == 60 and not sheet.isVisible()
    assert seen and max(seen) >= 900 and not sheet.busy
    # again, stopped as soon as it reports progress: no file, no .part file, the sheet stays open
    sheet.present(proj, 1, formats=("mp4", "gif")); sheet.pick("format", "gif"); sheet.pick("length", 30)
    stopped = []; sheet.stopped.connect(lambda: stopped.append(1))
    sheet.run(); nxt = sheet.out
    assert pump(lambda: sheet.bar.value() > 0, 120)
    sheet.cancel(); assert pump(lambda: stopped, 30)
    assert sheet.isVisible() and not sheet.busy and sheet.status.text() == W.TIMELAPSE_SHEET["stopped"]
    assert not os.path.exists(nxt) and not os.path.exists(nxt + ".part")
    sheet.cancel(); assert not sheet.isVisible()


def test_export_sheet_links_to_the_timelapse():
    from app2.qt.export_sheet import ExportDialog
    app(); host = QWidget(); host.resize(1280, 800); host.show(); d = ExportDialog(host); got = []
    d.timelapse.connect(lambda: got.append(1)); d.present(["stl"], "stl")
    assert d.timelapse_b.text() == "Scan timelapse..." and d.timelapse_b.isVisible()
    d.timelapse_b.click(); assert got == [1]
    host.close(); host.deleteLater()


def test_file_menu_opens_the_sheet_and_the_notice_opens_the_folder(tmp_path, proj, monkeypatch):
    from app2.qt.shell import MainWindow
    from test_app2_workflow_fixes import FakeSession, state
    app(); made = []
    win = MainWindow(lambda bridge, **over: made.append(FakeSession(bridge, over.get("project"), str(tmp_path))) or made[-1],
                     gl=False, projects_dir=str(tmp_path))
    try:
        win.resize(1688, 1044); win.show(); pump()
        act = win.menu_actions["timelapse"]; assert act.text() == "Export scan timelapse..."
        assert act in win.menus["File"].actions()
        win.bridge("state", state(proj, 1)); pump(lambda: getattr(win.state, "project", None) == proj, 3)
        assert win._menu_states()["timelapse"]
        win._menu_action("timelapse"); d = win.timelapse_dlg
        assert d.isVisible() and d.project == proj and d.sides.count() == 2
        win.hide_sheets(); assert not d.isVisible()
        win.build.timelapse_requested.emit(); assert d.isVisible(); d.cancel()
        opened = []
        from app2.qt import shell
        monkeypatch.setattr(shell.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
        win._timelapse_done(dict(path=os.path.join(proj, "x_timelapse.mp4"), bytes=2_500_000))
        assert win.toast.isVisible() and win.toast.text.text() == "Saved x_timelapse.mp4, 2.5 MB."
        assert win.toast.action.isVisible() and win.toast.action.text() == "Open folder"
        win.toast.action.click(); assert opened == [proj] and not win.toast.isVisible()
        win.say("plain"); assert not win.toast.action.isVisible()
        # a project with no saved side: a notice, no sheet
        empty = os.path.join(str(tmp_path), "ProjectLiveEmpty"); os.makedirs(os.path.join(empty, "data"))
        win.bridge("state", state(empty, 0)); pump(lambda: getattr(win.state, "project", None) == empty, 3)
        win._menu_action("timelapse"); assert not d.isVisible() and win.toast.text.text() == W.TIMELAPSE_SHEET["no_sides"]
    finally:
        win._really = True; win.build.shutdown(0.5); win.close(); win.deleteLater(); app().processEvents()
