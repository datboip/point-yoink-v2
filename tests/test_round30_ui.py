"""Tile controls stay reachable without moving the workspace or covering images."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
import pytest
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtWidgets import QApplication,QComboBox,QCheckBox
from PySide6.QtTest import QTest


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_headers_maximize_restore_and_no_workspace_jump(app,tmp_path,size):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state,settle
    p=ScanPage(SessionBridge(),False);p.resize(*size);p.show();p.on_state(scene_state(tmp_path));settle(app,p)
    try:
        assert not hasattr(p,'camera_arrangement')
        anchors=(p.left,p.model,p.sides,p.go)
        def geometry():return [(w.mapTo(p,QPoint()),w.size()) for w in anchors]
        before=geometry()
        for tile in (p.ir,p.depth,p.colour):
            h=tile.header
            assert h.max_button.focusPolicy()==Qt.FocusPolicy.StrongFocus
            QTest.mouseClick(h.max_button,Qt.MouseButton.LeftButton);app.processEvents()
            assert p._camera_order()[0] is tile
            assert geometry()==before
            for small in (p.ir,p.depth,p.colour):
                sh=small.header
                for child in sh.findChildren(type(sh.max_button)):
                    if child.isVisible():assert child.width()>=child.fontMetrics().horizontalAdvance(child.text())
                if sh.ir is not None:
                    for b in sh.ir.buttons:assert b.width()>=b.fontMetrics().horizontalAdvance(b.text())
            tile.set_image(np.zeros((600,800,3),np.uint8))
            assert tile.image_rect(QRectF(tile.rect())).top()>=h.height()-0.01   # float rounding
            assert all(b.geometry().right()<=h.width() for b in h.findChildren(type(h.max_button)) if b.isVisible())
            h.max_button.setFocus();QTest.keyClick(h.max_button,Qt.Key.Key_Escape);app.processEvents()
            assert p.maximized_camera is None and p._camera_order()[0] is p.ir
            assert geometry()==before
            QTest.keyClick(h.max_button,Qt.Key.Key_Return);app.processEvents()
            assert p._camera_order()[0] is tile
            QTest.keyClick(h.max_button,Qt.Key.Key_Return);app.processEvents()
            assert p._camera_order()[0] is tile and geometry()==before   # the last camera picked stays on top
        assert p.camera_options.ir.parent() is p.ir.header
        assert p.camera_options.show.parent() is p.colour.header
    finally:p.close()


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_popout_maximize_all_tiles_and_shared_switches(app,size):
    from app2.qt.camera_options import CameraPreferences
    from app2.qt.camera_popout import CameraPopout
    from app2.qt.scan_page import CAMERA_PARTS
    prefs=CameraPreferences();p=CameraPopout(CAMERA_PARTS,prefs=prefs);p.resize(*size);p.show();app.processEvents()
    try:
        before=p.size()
        for tile in (p.ir,p.depth,p.colour):
            tile.header.max_button.click();app.processEvents()
            assert p.grid.itemAtPosition(0,0).widget() is tile
            assert p.size()==before
            tile.header.max_button.setFocus();QTest.keyClick(tile.header.max_button,Qt.Key.Key_Escape);app.processEvents()
            assert p.maximized_camera is None
        p.options.show.click();app.processEvents()
        assert p.options.menu.isVisible()
        assert isinstance(p.options.actions['depth_edge'],QCheckBox)
        assert prefs.show['depth_edge']
        p.options.actions['depth_edge'].click();assert not prefs.show['depth_edge']
        QTest.keyClick(p.options.actions['depth_edge'],Qt.Key.Key_Escape);app.processEvents()
        assert not p.options.menu.isVisible()
        assert p.options.ir.findChildren(QComboBox)==[]
        p.options.ir.buttons[2].click();assert prefs.ir=='Both'
    finally:p.options.menu.close();p.close()


def test_short_choices_are_buttons_and_send_the_selected_value(app,tmp_path):
    from app2.qt.choices import Choices
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state
    p=ScanPage(SessionBridge(),False);calls=[]
    class Session:
        def __getattr__(self,name):return lambda *a:calls.append((name,a))
    p.session=Session();p.on_state(scene_state(tmp_path))
    try:
        for widget in (p.model.distance_mode,p.controls.live_build.detail,p.controls.live_build.angle):
            assert isinstance(widget,Choices)
            assert len(widget.buttons)==widget.count()
        p.controls.capture_mode.setCurrentIndex(p.controls.capture_mode.findData('smart'))   # Capture, Object, Surface stay dropdowns (user's call)
        assert ('set_capture_mode',('smart',)) in calls
        p.controls.surface.setCurrentIndex(2);assert ('set_surface',('Shiny',)) in calls
    finally:p.close()


def test_color_estimate_is_marked_and_kept_arc_is_not_a_live_angle(app):
    from app2.qt.camera_tiles import TurnReadout
    readout=TurnReadout()
    readout.set_data(dict(degrees=90,live_angle_deg=None))
    assert readout.text()=='' and 'unknown' in readout.toolTip()
    readout.set_estimate(dict(angle_deg=42))
    assert readout.text()=='~42°' and readout.toolTip().startswith('Estimated')
    readout.set_estimate(None);readout.set_data(dict(live_angle_deg=51,bed_seen=True))
    assert readout.text()=='51°'
    readout.close()


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_shared_build_edit_model_display_has_visible_choices(app,size):
    from app2.qt.build_view import BuildViewPanel
    from app2.qt.view_modes import ViewModes
    from app2.qt import theme
    theme.install(app)
    panel=BuildViewPanel(False);panel.resize(*size);panel.show()
    modes=[('shaded','Shaded'),('flat','Flat facets'),('points','Points')]
    panel.set_colours(modes,'shaded');app.processEvents();calls=[];panel.colour_picked.connect(calls.append)
    try:
        assert isinstance(panel.colour_btn,ViewModes)
        before=panel.main.geometry()
        for index,b in enumerate(panel.colour_btn.look_buttons()):
            assert b.width()>=b.fontMetrics().horizontalAdvance(b.text())
            b.click();app.processEvents();assert calls[-1]==('shaded','flat')[index]
            panel.set_colours(modes,('shaded','flat')[index]);app.processEvents()
            assert panel.main.geometry()==before
        panel.resize(380,size[1]);app.processEvents()
        assert panel.colour_btn.geometry().right()<panel.area.width()
        assert not panel.colour_menu.isVisible()
    finally:panel.close()


def test_view_modes_one_row_with_pair_buttons_and_overlay_menu(app):
    """One row: two look buttons (Shaded flips with Scan color, Flat facets with Points), an Overlay drop-down (None,
    What is left, Frame support, Capture distance); the What is left sub-pair shows only while it is on; None goes back
    to the last look."""
    from app2.qt.build_view import BuildViewPanel
    panel=BuildViewPanel(False);panel.resize(1400,800);panel.show()
    modes=[('shaded','Shaded'),('scan_color','Scan color'),('flat','Flat facets'),('points','Points'),('scan_guide','What is left: table turns'),
           ('scan_guide_handheld','What is left: scanner moves'),('quality','Frame support'),('capture_distance','Capture distance')]
    calls=[];panel.colour_picked.connect(calls.append)
    try:
        panel.set_colours(modes,'flat');app.processEvents()
        vm=panel.colour_btn;a,b=vm.look_buttons()
        assert [a.text(),b.text()]==['Shaded','Flat facets'] and b.isChecked() and not a.isChecked()
        assert vm.overlay_texts()==['None','What is left','Frame support','Capture distance']
        assert vm.overlays.text().startswith('None') and not vm.left_kind.isVisible()
        b.click();app.processEvents();assert calls[-1]=='points'                                  # lit: flips to the other member
        panel.set_colours(modes,'points');app.processEvents();assert b.text()=='Points' and b.isChecked()
        a.click();app.processEvents();assert calls[-1]=='shaded'                                  # not lit: picks what it shows
        panel.set_colours(modes,'shaded');app.processEvents();assert a.isChecked() and not b.isChecked() and b.text()=='Points'
        a.click();app.processEvents();assert calls[-1]=='scan_color'
        panel.set_colours(modes,'scan_color');app.processEvents();assert a.text()=='Scan color' and a.isChecked()
        vm.pick_overlay('what_is_left');app.processEvents();assert calls[-1]=='scan_guide'
        panel.set_colours(modes,'scan_guide');app.processEvents()
        assert vm.left_kind.isVisible() and vm.overlays.text().startswith('What is left') and not a.isChecked() and not b.isChecked()
        vm.left_kind.buttons[1].click();app.processEvents();assert calls[-1]=='scan_guide_handheld'
        panel.set_colours(modes,'scan_guide_handheld');app.processEvents()
        vm.pick_overlay('__none__');app.processEvents();assert calls[-1]=='scan_color'            # None: the look from before
        vm.pick_overlay('quality');app.processEvents();assert calls[-1]=='quality'
        panel.set_colours([('shaded','Shaded'),('sides','Sides')],'shaded');app.processEvents()
        assert not vm.overlays.isVisible() and [x.text() for x in vm.look_buttons()]==['Shaded','Sides']
        g=panel.colour_btn.geometry();assert g.right()<panel.area.width() and g.bottom()<=panel.area.height()-8 and g.top()>panel.area.height()/2
        panel.set_colours(modes,'scan_guide');panel._place();app.processEvents()
    finally:panel.close()


def test_notice_sits_under_the_view_buttons(app):
    """The notice bar never overlaps the Look row or Compare / Home view: those sit along the bottom, the notice top right."""
    from app2.qt.build_view import BuildViewPanel
    panel=BuildViewPanel(False);panel.resize(1400,800);panel.show()
    modes=[('shaded','Shaded'),('flat','Flat facets'),('points','Points'),('scan_guide','What is left: table turns'),
           ('scan_guide_handheld','What is left: scanner moves'),('quality','Frame support'),('capture_distance','Capture distance')]
    try:
        panel.set_colours(modes,'scan_guide');panel.notice.show_text('Saved ProjectLive10222026200528_print.stl, 34.4 MB, 687039 triangles.')
        panel._place();app.processEvents()
        n=panel.notice.geometry();assert panel.notice.isVisible()
        for b in (panel.colour_btn,panel.compare,panel.home):
            if not b.isHidden():assert not n.intersects(b.geometry()),(b.geometry(),n)
        assert n.right()<=panel.area.width()-8 and n.bottom()<=panel.colour_btn.geometry().top()
    finally:panel.close()
