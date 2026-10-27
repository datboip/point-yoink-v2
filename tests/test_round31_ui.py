"""Pass actions stay outside the images and work through the existing commands."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
import pytest
import numpy as np
from PySide6.QtCore import QPoint,QRectF
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_pass_controls_visible_in_both_tabs_and_clear_of_views(app,tmp_path,size):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state,settle
    p=ScanPage(SessionBridge(),False);p.resize(*size);p.show();p.on_state(scene_state(tmp_path));settle(app,p)
    try:
        for tab in (0,1):
            p.sides.tabs.setCurrentIndex(tab);settle(app,p)
            controls=(p.go,p.sides.new,p.controls.led,p.controls.gain_row,p.controls.range_row,p.model.distance_mode)
            for control in controls:   # strip under the cameras, Start and + New side in the right column
                assert control.isVisible() and (p.left.isAncestorOf(control) or p.sides.isAncestorOf(control))
                pos=control.mapTo(p,QPoint())
                assert pos.x()>=0 and pos.x()+control.width()<=p.width()
            assert (p.width(),p.height())==size
            for tile in (p.ir,p.depth,p.colour,p.model.image):
                tile.set_image(np.zeros((600,800,3),np.uint8))
                r=tile.image_rect(QRectF(tile.rect()))
                assert r.width()/r.height()==pytest.approx(4/3)
            header_bottom=p.model.header.mapTo(p.model,QPoint()).y()+p.model.header.height()
            assert header_bottom<=p.model.viewport.y()
        for control,key in ((p.go,'Space'),(p.sides.new,'N'),(p.controls.led,'L'),(p.controls.auto_gain,'G'),(p.controls.auto_range,'R')):
            assert key in control.toolTip()
    finally:p.close()


def test_popout_does_not_rewind_color(app):
    from app2.qt.camera_popout import CameraPopout
    from app2.qt.scan_page import CAMERA_PARTS
    from app2.scan.preview import LiveView
    p=CameraPopout(CAMERA_PARTS)
    try:
        new=np.full((30,40,3),90,np.uint8);old=np.full_like(new,30)
        estimate=dict(angle_deg=17,source='Color bed dots')
        p.set_live(LiveView(colour=new,colour_only=True,color_token=2,bed_angle=estimate),CAMERA_PARTS[0])
        p.set_live(LiveView(colour=old,colour_only=True,color_token=1),CAMERA_PARTS[0])
        assert p.colour.img.pixelColor(0,0).red()==90
        assert p.options.turn.estimate is estimate
    finally:p.close()


def test_strip_commands_and_shortcuts_share_state(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state
    calls=[]
    class Session:
        def __getattr__(self,name):return lambda *args:calls.append((name,args))
    p=ScanPage(SessionBridge(),False);p.session=Session();p.on_state(scene_state(tmp_path));p.streamed=True;p._apply_go()
    try:
        p.go.click();assert calls[-1][0]=='toggle_scan'
        p.sides.new.click();assert calls[-1][0]=='new_side'
        p.key_gain_up();assert calls[-1]==('set_gain',(p.state.gain+1,))
        p.key_gain_down();assert calls[-1]==('set_gain',(p.state.gain-1,))
        p.live_strip.ir.setCurrentText('Both');assert p.camera_prefs.ir=='Both'
        p.key_ir_camera();assert p.live_strip.ir.currentText()=='Left'
        p.key_view_color();assert p.model.distance_mode.currentText()=='Coverage'
        p.controls.outside_below.setValue(60);assert calls[-1]==('set_outside_below',(60,))
    finally:p.close()


def test_build_fps_expires_without_a_stream(app):
    from app2.qt.live_rates import LiveRates
    from app2.scan.preview import LiveView
    r=LiveRates();r.active=True
    try:
        for i in range(31):r.add(LiveView(colour=np.full((2,2,3),i,np.uint8),colour_only=bool(i%10)),now=10+i/30)
        r.refresh(now=11)
        assert r.isVisible() and 'Color 30.0 fps' in r.text() and 'Depth 3.0 fps' in r.text()
        r.refresh(now=15);assert not r.isVisible()
    finally:r.close()


def test_late_depth_preview_cannot_replace_newer_color(app):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from app2.scan.preview import LiveView
    p=ScanPage(SessionBridge(),False)
    try:
        new=np.full((30,40,3),90,np.uint8);old=np.full_like(new,30)
        p._show_model_color(LiveView(colour=new,color_token=2))
        p._show_model_color(LiveView(colour=old,color_token=1))
        assert p.model.live_img is new
    finally:p.close()
