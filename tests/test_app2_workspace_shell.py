"""The 2.0 pages stay inside the shared Workspace navigation."""
import os
import tempfile
from pathlib import Path

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-shell-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QFrame

from app2.qt import theme
from app2.qt.shell import MainWindow


def test_workspace_project_dropdown_opens_a_local_project(tmp_path):
    from app2.qt.workspace_home import WorkspaceHome
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    project = tmp_path / 'ProjectLive123'; (project / 'data' / 'side1').mkdir(parents=True)   # a side: empty projects are not listed
    home = WorkspaceHome(str(tmp_path)); opened = []
    home.project_requested.connect(opened.append)
    home.resize(1024, 740); home.refresh_projects(); home.show(); app.processEvents()
    try:
        assert home.projects_dropdown.text() == 'Your projects (1)  ▾'
        home.projects_dropdown.click(); app.processEvents()
        assert home.projects_popover.isVisible()
        assert home.projects_popover.geometry().left() == 0
        assert home.projects_popover.geometry().top() == home.page_header.geometry().bottom() + 1
        assert home.projects_popover.geometry().bottom() + 1 == home.filmstrip.geometry().top()
        home.resize(1500, 900); app.processEvents()
        assert home.projects_popover.geometry().left() == 0
        assert home.projects_popover.geometry().bottom() + 1 == home.filmstrip.geometry().top()
        assert sum(card.isVisible() for card in home.projects_popover.findChildren(type(home.projects_popover), 'WorkspaceProjectCard')) == 1
        card = home.project_list.itemAt(0).widget()
        card.findChild(type(home.open), 'WorkspaceProjectOpen').click()
        assert opened == [str(project.resolve())]
        assert not home.projects_popover.isVisible()
    finally:
        home.close()


def test_workspace_project_gate_and_show_all(tmp_path):
    from app2.qt.workspace_home import WorkspaceHome
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    for i in range(23): (tmp_path / ('ProjectLive%03d' % i) / 'data' / 'side1').mkdir(parents=True)
    home = WorkspaceHome(str(tmp_path)); home.resize(1024, 740); home.show(); app.processEvents()
    try:
        assert 'Line-up' in home.ribbon_tools and 'Align' not in home.ribbon_tools and 'Combine' not in home.ribbon_tools
        modes = []; home.mode_requested.connect(modes.append)
        home.ribbon_tools['Line-up'].click(); assert modes == ['Align']
        home.set_project_open(False)
        assert not any(home.ribbon_tools[n].isEnabled() for n in ('Build', 'Edit', 'Export', 'Cut base', 'Smooth', 'Simplify', 'Line-up'))
        assert all(home.ribbon_tools[n].isEnabled() for n in ('Projects', 'Import', 'Scan'))
        home.set_project_open(True, export=False)
        assert home.ribbon_tools['Build'].isEnabled() and not home.ribbon_tools['Export'].isEnabled()
        home.toggle_projects_popover(); app.processEvents()
        cards = lambda: sum(c.isVisible() for c in home.projects_popover.findChildren(QFrame, 'WorkspaceProjectCard'))
        app.processEvents(); assert cards() == 20 and home.show_all_b.isVisible() and '20 of 23' in home.count_note.text()
        home.show_all_b.click(); app.processEvents(); assert cards() == 23
    finally:
        home.close()


@pytest.mark.parametrize("size", [(1024, 640), (1688, 1044)])
def test_new_scan_and_home_keep_one_shared_shell(size, tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)

    def offline(*args, **kwargs):
        raise RuntimeError("No device in shared-shell test")

    win = MainWindow(offline, gl=False, projects_dir=str(tmp_path))
    win.resize(*size); win.show(); app.processEvents()
    try:
        assert win.stack.currentWidget() is win.workspace
        assert not win.scan.status.isVisible()
        assert not win.scan.status.log_btn.isVisible()
        assert win.workspace.ribbon_scroll.isVisible()
        assert win.workspace.page_header.height() == win.header.height()
        assert win.workspace.findChild(type(win.header.step), "WorkspaceTitle").font().pointSize() == win.header.step.font().pointSize()
        assert not win.rail.isVisible()
        assert not win.workspace.ribbon_tools["Export"].isEnabled()
        assert not win.workspace.build_tools.isVisible()
        win.workspace.new.click(); app.processEvents()
        assert win.stack.currentWidget() is win.scan
        assert win.scan.status.isVisible()
        assert win.scan.status.log_btn.isVisible()
        assert win.workspace.ribbon_scroll.isVisible()
        assert not win.rail.isVisible()
        assert win.workspace.ribbon_tools["Scan"].objectName() == "WorkspaceToolPrimary"
        assert not win.workspace.build_tools.isVisible()
        assert not win.workspace.ribbon_tools["Build"].isEnabled()          # offline New: no project, so no Build
        real_gate = win.workspace.set_project_open                             # from here on, act as if a project is open
        win.workspace.set_project_open = lambda is_open, export=None: real_gate(True, export); real_gate(True)
        win.workspace.ribbon_tools["Build"].click(); app.processEvents()
        assert win.stack.currentWidget() is win.build_slot
        assert not win.scan.status.isVisible()
        assert not win.scan.status.log_btn.isVisible()
        assert not win.build.log_b.isVisible()
        win.build.log.appendPlainText('Build output'); app.processEvents()
        assert win.build.log_b.isVisible()
        win.build.log_b.click(); app.processEvents()
        assert win.build.log_pop.isVisible()
        assert win.workspace.ribbon_tools["Build"].objectName() == "WorkspaceToolPrimary"
        assert not win.workspace.build_tools.isVisible()          # its tools are rows in Build's rail now
        win._workspace_mode("Smooth"); app.processEvents()
        assert win.stack.currentWidget() is win.build_slot
        assert win.workspace.ribbon_tools["Build"].objectName() == "WorkspaceToolPrimary"
        win.workspace.ribbon_tools["Edit"].click(); app.processEvents()
        assert win.stack.currentWidget() is win.edit
        assert win.scan.status.isVisible()
        assert not win.scan.status.pill.isVisible()
        assert not win.build.log_pop.isVisible()                  # the Log button itself shows on Edit too, once there is output
        assert not win.workspace.build_tools.isVisible()
        win.workspace.ribbon_tools["Projects"].click(); app.processEvents()
        assert win.stack.currentWidget() is win.workspace
        assert not win.scan.status.isVisible()
        assert not win.scan.status.log_btn.isVisible()
        assert win.workspace.ribbon_tools["Projects"].objectName() == "WorkspaceToolPrimary"
        assert win.workspace.projects_dropdown.isVisible()
        assert 'No local projects yet' in win.workspace.project_list.itemAt(0).widget().text()
        center_before = win.workspace.start.mapTo(win.workspace, win.workspace.start.rect().center())
        assert abs(center_before.x() - win.workspace.width() / 2) <= 2
        win.workspace.ribbon_tools['Import'].click(); app.processEvents()
        assert win.stack.currentWidget() is win.device_projects
        assert not win.scan.status.isVisible()
        assert not win.scan.status.log_btn.isVisible()
        assert win.workspace.ribbon_tools['Import'].objectName() == 'WorkspaceToolPrimary'
        win.device_projects.dismiss(); app.processEvents()
        assert win.stack.currentWidget() is win.workspace
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()


def test_scan_camera_layout_adapts_inside_shared_shell(tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    win = MainWindow(lambda *a, **k: None, gl=False, projects_dir=str(tmp_path))
    try:
        for size in ((1280, 800), (1688, 1044)):
            win.resize(*size); win.show(); win.show_step(0)
            for _ in range(8): app.processEvents()
            assert win.scan.maximized_camera is None
            assert not win.scan.one_pic
            assert all(tile.isVisible() and tile.height() >= tile.minimumHeight() for tile in
                       (win.scan.ir, win.scan.depth, win.scan.colour))
            assert win.scan.camera_scroll.verticalScrollBar().maximum() == 0
            assert all(abs(tile.width() * .75 - (tile.height()-tile.header.height())) <= 4 for tile in
                       (win.scan.ir, win.scan.depth, win.scan.colour)), [(t.width(),t.height()) for t in (win.scan.ir,win.scan.depth,win.scan.colour)]
            assert win.scan.go.isVisible()
            # Settings live in the right tab at every shell size.
            # (a new scan with no sides opens on Settings; this loop resizes one session, so set the tab explicitly)
            win.scan.sides.tabs.setCurrentIndex(0)
            for _ in range(8): app.processEvents()
            assert not win.scan.controls.isVisible()
            actions = (win.scan.go.geometry(), win.scan.sides.buildbox.geometry())
            win.scan.sides.tabs.setCurrentIndex(1)
            for _ in range(8): app.processEvents()
            assert win.scan.controls.isVisible()
            assert actions == (win.scan.go.geometry(), win.scan.sides.buildbox.geometry())
            win.scan.sides.tabs.setCurrentIndex(0)
            win.scan.maximize_camera("Color")
            for _ in range(8): app.processEvents()
            assert win.scan._camera_order()[0] is win.scan.colour
            assert not win.scan.switch.isVisible()
            win.scan.restore_cameras()
            for _ in range(8): app.processEvents()
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()


def test_scanner_project_import_opens_inside_shared_app(tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    win = MainWindow(lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')), gl=False, projects_dir=str(tmp_path))
    win.show(); app.processEvents()
    try:
        win.workspace.ribbon_tools['Import'].click(); app.processEvents()
        assert win.device_projects.isVisible()
        assert win.stack.currentWidget() is win.device_projects
        assert win.header.isVisible() and win.header.step.text() == 'Import'
        assert win.device_projects.library == str(tmp_path)
        win.device_projects.transport.setCurrentIndex(1)
        assert 'Wi-Fi' in win.device_projects.help.text()
        win.device_projects.dismiss(); app.processEvents()
        assert not win.device_projects.isVisible()
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()
