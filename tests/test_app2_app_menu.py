"""Top menus use existing actions; About stays within the app window."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-menu-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt.app_menu import AboutCard
from app2.version import APP_TITLE


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_about_fixed_inside_window_and_outside_close(size):
    app=QApplication.instance() or QApplication([])
    parent=QWidget(); parent.resize(*size); card=AboutCard(parent)
    parent.show(); card.show(); app.processEvents()
    try:
        assert not card.isWindow() and not card.panel.isWindow()
        assert card.rect().contains(card.panel.geometry())
        assert card.panel.width()==420 and card.panel.height()==360
        assert card.version.text()==APP_TITLE and not card.logo.pixmap().isNull()
        QTest.mouseClick(card,Qt.MouseButton.LeftButton,pos=QPoint(parent.width()-10,10))
        assert not card.isVisible()
    finally: parent.close()


def test_top_menus_busy_gates_actions_and_about_escape(tmp_path,monkeypatch):
    from app2.qt.shell import MainWindow
    from app2.scan.ui_model import ScanState
    app=QApplication.instance() or QApplication([])
    def offline(*args,**kwargs): raise RuntimeError('No device in menu test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path)); win.resize(1280,800); win.show(); app.processEvents()
    called=[]
    monkeypatch.setattr(win,'ask_new_project',lambda:called.append('new'))
    monkeypatch.setattr(win,'show_step',lambda index:called.append(('step',index)))
    monkeypatch.setattr(win.build,'_export_src',lambda:None)
    try:
        assert [a.text() for a in win.menuBar().actions()]==['File','Edit','View','Help']
        assert win.menu_actions['keys'] in win.menus['Help'].actions()
        assert win.menu_actions['open'].text().endswith('\tCtrl+O')
        assert win.menu_actions['home'].text().endswith('\tH')
        assert win.rail.keys.isHidden() and not hasattr(win.rail,'menu')
        assert all(button.isHidden() for button in (win.header.new,win.header.open,win.header.finish,win.header.motion))
        win.state=ScanState(project=str(tmp_path),building=True)
        win._refresh_menus()
        for action in ('new','open','export','close','undo','redo','edit','rename'):
            assert not win.menu_actions[action].isEnabled()
        win._menu_action('new'); assert called==[]
        win.state.building=False; win._refresh_menus()
        win.menu_actions['new'].trigger(); assert called==['new']
        win.menu_actions['edit'].trigger(); assert called[-1]==('step',2)
        win.menu_actions['about'].trigger(); app.processEvents()
        assert win.about_card.isVisible() and not win.about_card.isWindow()
        win.hide_sheets(); assert not win.about_card.isVisible()
        win.menu_actions["guide"].trigger(); app.processEvents()
        assert win.guide_card.isVisible()
        win.hide_sheets(); assert not win.guide_card.isVisible()
    finally:
        win._really=True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_local_guide_fixed_scrollable_no_external_links(size):
    from app2.qt.app_menu import GuideCard
    app=QApplication.instance() or QApplication([])
    parent=QWidget(); parent.resize(*size); guide=GuideCard(parent)
    parent.show(); guide.show(); app.processEvents()
    try:
        assert not guide.isWindow()
        assert guide.panel.width()>=640 and guide.panel.height()>=520 and guide.panel.width()<=size[0] and guide.panel.height()<=size[1]
        assert guide.rect().contains(guide.panel.geometry())
        assert guide.text.verticalScrollBar().maximum()>0
        assert not guide.text.openExternalLinks() and not guide.text.openLinks()
        text=guide.text.toPlainText()
        assert 'Camera setup' in text and 'File → Export STL' in text
        guide.done.click(); assert not guide.isVisible()
    finally: parent.close()


def test_projection_and_orientation_menu_routes_to_active_view(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app2.qt.shell import MainWindow
    app = QApplication.instance() or QApplication([])
    def offline(*args, **kwargs): raise RuntimeError("No device in menu test")
    win = MainWindow(offline, gl=False, projects_dir=str(tmp_path))
    called = []
    view = SimpleNamespace(cam=SimpleNamespace(projection_mode="orthographic"),
                           set_projection_mode=lambda mode: called.append(("scan", mode)))
    cube = SimpleNamespace(set_reference=lambda face: called.append(("reference", face)))
    monkeypatch.setattr(win, "_active_view", lambda: view)
    monkeypatch.setattr(win, "_active_cube", lambda: cube)
    monkeypatch.setattr(win.build.vp, "set_projection_mode", lambda mode: called.append(("build", mode)))
    try:
        win._refresh_menus()
        assert win.menu_actions["orthographic"].isChecked()
        assert not win.menu_actions["perspective"].isChecked()
        win._menu_action("perspective")
        assert called[-1] == ("scan", "perspective")
        win.show_step(1)
        win._menu_action("orthographic")
        assert called[-1] == ("build", "orthographic")
        win._menu_action("front")
        assert called[-1] == ("reference", "front")
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()


def test_report_a_problem_gathers_with_no_project(tmp_path):
    from app2.qt.shell import MainWindow
    app=QApplication.instance() or QApplication([])
    def offline(*args,**kwargs): raise RuntimeError('No device in menu test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path)); win.show(); app.processEvents()
    try:
        win.scan.log.append('a scan log line')
        data=win._report_data()                      # crashed: the scan log panel had no toPlainText
        assert data is not None
    finally: win.close()
