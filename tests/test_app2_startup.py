"""Startup remains a child in the same window with fixed status regions."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-startup-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt.startup import StartupOverlay, EmptyViewBackground, splash_background


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_startup_same_window_stable_status_and_immediate_ready(size):
    app=QApplication.instance() or QApplication([])
    root=QWidget(); root.resize(*size); overlay=StartupOverlay(root)
    root.show(); overlay.show(); app.processEvents()
    assert not overlay.isWindow() and not overlay.logo.isNull()
    assert overlay.rect().contains(overlay.panel.toRect())
    statusrect=overlay.status.geometry(); buttonrect=overlay.skip.geometry()
    for text in ('Opening project','Connecting camera','Ready','Very long device failure '*30):
        overlay.set_status(text); app.processEvents()
        assert overlay.status.geometry()==statusrect and overlay.skip.geometry()==buttonrect
        assert overlay.status.fontMetrics().horizontalAdvance(overlay.status.text())<=424
        assert overlay.rect().contains(buttonrect)
    overlay.fail('Camera did not connect. Open workspace to continue.'); assert not overlay.timer.isActive()
    assert overlay.skip.isEnabled()
    image=overlay.grab().toImage(); assert not image.isNull()
    overlay.ready(); assert not overlay.isVisible() and not overlay.timer.isActive()
    root.close()


def test_original_gradient_color_formula():
    image=splash_background()
    assert image.size().width()==480
    assert image.pixelColor(240,144).getRgb()[:3]==(35,40,55)


def test_background_is_child_and_mouse_transparent():
    from PySide6.QtCore import Qt
    app=QApplication.instance() or QApplication([])
    parent=QWidget(); background=EmptyViewBackground(parent); background.resize(492,616)
    assert not background.isWindow()
    assert background.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not background.grab().isNull()
    parent.close()


def test_shell_startup_waits_for_first_state_and_dismisses(tmp_path):
    from types import SimpleNamespace
    from app2.qt.shell import MainWindow
    from app2.scan.ui_model import ScanState
    app=QApplication.instance() or QApplication([])
    called=[]
    session=SimpleNamespace(start=lambda:called.append('start'))
    window=MainWindow(lambda bridge,**kw:session,gl=False,projects_dir=str(tmp_path))
    window.show(); app.processEvents()
    try:
        assert called==['start']
        assert window.startup.isVisible() and not window.startup.isWindow()
        assert window.startup.status.text()=='Connecting camera'
        assert all(not key.isEnabled() for key in window._keys if not key._allow_edit)
        state=ScanState(project=str(tmp_path),source_kind='replay',source_label='Replay',light_remaining=None)
        window.on_state(state); app.processEvents()
        assert not window.startup.isVisible() and not window.startup.timer.isActive()
        assert window.state is state
        window.menuBar().setFocus(); app.processEvents()
        assert all(key.isEnabled() for key in window._keys)
    finally:
        window._really=True; window.build.shutdown(); window.close(); window.deleteLater(); app.processEvents()


@pytest.mark.parametrize('during_start',[False,True])
def test_shell_startup_errors_leave_workspace_usable(tmp_path,during_start):
    from types import SimpleNamespace
    from app2.qt.shell import MainWindow
    app=QApplication.instance() or QApplication([])
    def broken(): raise RuntimeError('Isolated startup failure')
    def factory(bridge,**kwargs):
        if not during_start: broken()
        return SimpleNamespace(start=broken)
    window=MainWindow(factory,gl=False,projects_dir=str(tmp_path))
    window.show(); app.processEvents()
    try:
        assert not window.startup.isVisible()
        assert window.startup.error and not window.startup.timer.isActive()
        assert window.header.open.isEnabled() and window.header.new.isEnabled()
        window.menuBar().setFocus(); app.processEvents()
        assert all(key.isEnabled() for key in window._keys)
    finally:
        window._really=True; window.build.shutdown(); window.close(); window.deleteLater(); app.processEvents()


def test_model_and_edit_empty_backgrounds_follow_geometry():
    import numpy as np
    from app2.qt.view3d import ModelView
    from app2.qt.edit_view import EditView
    from PySide6.QtWidgets import QWidget
    app=QApplication.instance() or QApplication([])
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    normals=np.tile(np.array([0,0,1],dtype=np.float32),(3,1)); faces=np.array([[0,1,2]],dtype=np.int32)
    model=ModelView(); edit=EditView()
    try:
        for view in (model,edit):
            assert not view.empty_background.isHidden()
            children=[child for child in view.children() if isinstance(child,QWidget)]
            assert children.index(view.empty_background)<children.index(view.nav_cube)
        model.set_points(vertices,normals); assert model.empty_background.isHidden()
        model.set_mesh(vertices,normals,faces=faces)
        model.set_points(None); assert model.empty_background.isHidden()
        model.set_mesh(None); assert not model.empty_background.isHidden()
        edit.show_items([dict(key='part',kind='mesh',V=vertices,N=normals,F=faces)])
        assert edit.empty_background.isHidden()
        edit.clear(); assert not edit.empty_background.isHidden()
        edit.show_items([]); assert not edit.empty_background.isHidden()
        for view in (model,edit):
            children=[child for child in view.children() if isinstance(child,QWidget)]
            assert children.index(view.empty_background)<children.index(view.nav_cube)
    finally:
        model.close(); edit.close(); model.deleteLater(); edit.deleteLater(); app.processEvents()
