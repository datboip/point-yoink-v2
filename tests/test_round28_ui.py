"""Camera controls, retained marks, fixed screen strokes and shared preferences."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QImage,QPainter
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    return QApplication.instance() or QApplication([])


def frame():
    from app2.scan.preview import LiveView
    image=np.full((60,80,3),20,np.uint8)
    marks=dict(size=[80,60],box=[[10,10],[70,10],[70,50],[10,50],[10,10]],aim=[40,30],part_center=[20,30],color=[79,196,111],depth_outline=[[5,5],[75,5],[75,55],[5,55],[5,5]])
    return LiveView(ir=image,depth=image,colour=image,colour_clean=np.full_like(image,30),colour_guides=marks,ir_guides=marks,right_guides=marks,ir_right=np.full_like(image,100),ir_both=np.full_like(image,60),colour_only=False,seq=1)


def test_show_defaults_all_seven_options_and_ir_mode_persist(app):
    from app2.qt.camera_options import CameraPreferences,CameraOptions,DEFAULTS
    prefs=CameraPreferences();controls=CameraOptions(prefs)
    assert {k:prefs.show[k] for k in DEFAULTS}==dict(dots=True,box=True,aim=True,part_center=False,depth_edge=True,left_fov=False,right_fov=False)
    for key in DEFAULTS:controls.actions[key].click()
    controls.ir.setCurrentText('Both')
    restored=CameraPreferences()
    assert {k:restored.show[k] for k in DEFAULTS}=={k:not v for k,v in DEFAULTS.items()} and restored.ir=='Both'
    controls.close()


def test_follow_range_has_no_lock_button_or_f_shortcut_and_fixed_width(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from app2.qt.shell import SCAN_KEYS
    from app2.qt.widgets import KeysSheet
    from test_round25_ui import scene_state
    page=ScanPage(SessionBridge(),False);page.resize(1280,800);page.show();page.sides.tabs.setCurrentIndex(1)
    try:
        s=scene_state(tmp_path);page.on_state(s);app.processEvents()
        assert not hasattr(page.controls,'frame_lock') and not hasattr(page,'key_frame')
        assert all(key!='F' for key,_ in SCAN_KEYS)
        assert all(key!='F' for key,_ in KeysSheet.KEYS)
        tool=page.controls.frame_range;old=tool.width()
        s.range_undo=True;page.on_state(s);app.processEvents()
        assert tool.width()==old and tool.minimumWidth()==tool.maximumWidth()
        assert not page.colour.toolTip() and not page.model.toolTip()
        assert 'Part Range applied' in page.controls.frame_note.toolTip()
    finally:page.close()


def test_same_preferences_apply_to_color_model_focus_and_popout(app):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    page=ScanPage(SessionBridge(),False);page.resize(1280,800);page.show()
    try:
        page.maximize_camera('Color')   # the small tile is empty while the big view shows live color
        v=frame();page.on_live(v);page.camera_prefs.set_show('dots',False);page.camera_prefs.set_show('depth_edge',True)
        assert page.colour.img.pixelColor(0,0).red()==30
        assert page.model.image.guide_options['depth_edge']
        page.pop_out_cameras();pop=page.camera_window
        assert pop.prefs is page.camera_prefs and pop.colour.guide_options['depth_edge']
        pop.options.ir.setCurrentText('Right');assert page.camera_prefs.ir=='Right'
        assert pop.ir.img.pixelColor(0,0).red()==100
        page.return_cameras();page.one_pic=True;page._switch_camera(2)
        assert page.ir.guide_options['depth_edge'] and page.ir.img.pixelColor(0,0).red()==30
    finally:
        if page.camera_window is not None:page.camera_window.close()
        page.close();app.processEvents()


def test_camera_only_color_updates_do_not_increment_ir_fps(app,monkeypatch):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    page=ScanPage(SessionBridge(),False);page.show();page.maximize_camera('Color')
    try:
        a=frame();page.on_live(a);b=frame();page.on_live(b)
        ir_stamps=list(page.ir._stamps)
        for _ in range(4):
            c=frame();c.colour_only=True;page.on_live(c)
        assert page.ir._stamps==ir_stamps and len(page.colour._stamps)==6
        before=len(page.colour._stamps);page.camera_prefs.set_show('part_center',True)
        assert len(page.colour._stamps)==before
    finally:page.close()


def test_vectors_keep_screen_stroke_width_on_white_and_black_at_two_scales(app):
    from app2.qt.camera_guides import paint
    guides=dict(size=[100,100],box=[[10,10],[90,10],[90,90],[10,90],[10,10]],color=[79,196,111])
    widths=[]
    for size in (400,800):
        for background in (0,255):
            image=QImage(size,size,QImage.Format.Format_RGB888);image.fill(Qt.GlobalColor.black if background==0 else Qt.GlobalColor.white)
            painter=QPainter(image);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            paint(painter,QRectF(0,0,size,size),guides,dict(aim=False),18)
            painter.end()
            x=size//2;ys=[y for y in range(size//10-8,size//10+8) if image.pixelColor(x,y).red()!=background]
            widths.append(len(ys))
            assert image.pixelColor(size//2,size//2).red()==background,'No fill or dimming outside edges'
    assert max(widths)-min(widths)<=1 and min(widths)>=3


def test_each_show_toggle_has_visible_effect_and_changes_no_image_pixels(app):
    from app2.qt.widgets import ImageTile
    from app2.qt.camera_options import DEFAULTS
    v=frame();v.colour_guides.update(left_fov=[[2,2],[78,2],[78,58],[2,58],[2,2]],right_fov=[[4,1],[77,1],[77,59],[4,59],[4,1]])
    tile=ImageTile();tile.resize(800,600);tile.show();tile.set_image(v.colour)
    original=tile.img.copy()
    try:
        for key in DEFAULTS:
            if key=='dots':continue
            none={k:False for k in DEFAULTS};tile.set_guides(v.colour_guides,none);app.processEvents();base=tile.grab().toImage()
            none[key]=True;tile.set_guides(v.colour_guides,none);app.processEvents();after=tile.grab().toImage()
            assert base!=after,key
            assert tile.img==original
    finally:tile.close()


def test_legacy_right_missing_does_not_show_left_under_right_name(app):
    from app2.qt.widgets import ImageTile
    from app2.qt.camera_options import CameraPreferences,apply_frame
    prefs=CameraPreferences();prefs.set_ir('Right');v=frame();v.ir_right=None;v.right_guides=None
    tile=ImageTile();apply_frame(tile,v,'ir',prefs)
    assert tile.img is None and tile.camera_guides is None
    tile.close()


def test_full_missing_depth_hint_has_its_own_readable_footer_line(app):
    from app2.qt.widgets import DistanceBar
    bar=DistanceBar();bar.resize(600,bar.sizeHint().height())
    bar.frame_hint='Part of the part shows no depth (dark or shiny surface)'
    bar.set_guide(dict(center_mm=300,zones=[],working_range=[100,600],edges=[100,600],counts=[1]),(100,450),(0,1000))
    layout=bar.text_layout()
    assert not bar.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert layout['status'][0]==bar.frame_hint
    assert not layout['status'][1].intersects(layout['distance'][1])
    assert 'Framing:' not in layout['distance'][0]
    bar.close()
