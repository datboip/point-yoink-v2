"""Fixed Scan controls, explicit Range review, shared dots, and calm startup."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');os.environ['POINTYOINK_NO_DEVICE']='1'
import numpy as np
import pytest
from PySide6.QtCore import Qt,QPoint,QTimer,QRectF
from PySide6.QtWidgets import QApplication,QAbstractButton,QComboBox,QDialog,QFileDialog
from PySide6.QtGui import QImage,QPainter
from PySide6.QtTest import QTest


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'));return QApplication.instance() or QApplication([])


def test_range_wizard_has_four_steps_and_requires_apply(app,tmp_path):
    from app2.qt.capture_tools import CaptureTools
    from test_round25_ui import scene_state
    c=CaptureTools();c.show();s=scene_state(tmp_path);s.bed_size=30;c.set_state(s,False);calls=[];c.range_apply.connect(lambda:calls.append('apply'));c.open_wizard()
    try:
        assert c.wizard.pages.count()==4 and not c.wizard.accept.isEnabled()
        c.wizard.next.click();assert c.wizard.pages.currentIndex()==1
        s.range_test=dict(active=False,result=dict(complete=True,camera_pair_confirmed=True,bins=[],distance_bands=[dict(lo=200,hi=400,state='strong depth')],observed_distance_mm=[200,400]))
        c.set_state(s,False);assert calls==[];c.wizard.step(3);c.wizard.accept.click();assert calls==['apply']
    finally:c.wizard.close();c.close()


def test_one_depth_edge_toggle_drives_ir_color_and_popout(app):
    from app2.qt.camera_options import CameraPreferences,apply_frame,CameraOptions
    from app2.qt.widgets import ImageTile
    from test_round28_ui import frame
    prefs=CameraPreferences();assert prefs.show['depth_edge'];v=frame();ir=ImageTile();color=ImageTile()
    prefs.set_show('depth_edge',False);apply_frame(ir,v,'ir',prefs);apply_frame(color,v,'color',prefs)
    assert ir.guide_options['depth_edge'] is False and color.guide_options['depth_edge'] is False
    control=CameraOptions(prefs);assert control.menu.findChildren(QComboBox)==[]
    assert control.menu.color_angle.isChecked() is False
    ir.close();color.close();control.close()


def test_dots_stay_crisp_and_shared_palette_survives_reload(app):
    from app2.qt.camera_guides import paint
    from app2.qt.camera_options import CameraPreferences
    from app2.scan.range_guide import BAND_RGB,SAFE_PALETTE
    prefs=CameraPreferences();prefs.set_option('safe_palette',True);assert BAND_RGB==SAFE_PALETTE and CameraPreferences().show['safe_palette']
    widths=[]
    for size in (300,600):
        im=QImage(size,size,QImage.Format.Format_RGB888);im.fill(Qt.GlobalColor.white);p=QPainter(im)
        paint(p,QRectF(0,0,size,size),dict(size=[100,100],dots_xy=np.array([[50,50]]),dots_uv=np.array([[48,48]]),dots_levels=np.array([2]),dots_removed=np.array([False])),dict(dots=True,aim=False),18);p.end()
        widths.append(sum(im.pixelColor(x,size//2)!=Qt.GlobalColor.white for x in range(size//2-6,size//2+7)))
        assert im.pixelColor(size//2,size//2).blue()>100
    assert max(widths)-min(widths)<=1


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_all_scan_controls_keep_unrelated_geometry(app,tmp_path,monkeypatch,size):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state,settle
    from types import SimpleNamespace as NS
    page=ScanPage(SessionBridge(),False);page.resize(*size);page.show();commands=[]
    class Session:
        def __getattr__(self,name):return lambda *a:commands.append((name,a))
    page.session=Session();page.confirm=lambda *_:False;page.ask_text=lambda *_:None
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *_a,**_k:'')
    state=scene_state(tmp_path,has_led=True,bed_available=True,bed_size=30);state.capture_mode='live';state.live_undo=True;state.range_undo=True;page.on_state(state);page.sides.tabs.setCurrentIndex(1);settle(app,page)
    # Openings live in separate windows. Their children may change. The camera
    # arrangement intentionally changes camera tiles, while Start and Build stay.
    anchors=[page.left,page.model,page.sides,page.go,page.sides.buildbox,page.camera_bar,page.camera_options,page.controls.frame_range,page.controls.range_undo,page.controls.range_row,page.controls.gain_row,page.controls.led,page.controls.capture_tools.range_button]
    def geometry():return [(w.mapTo(page,QPoint()),w.size()) for w in anchors]
    clicked=[]
    try:
        assert not hasattr(page,"camera_arrangement")
        controls=[w for w in page.findChildren(QAbstractButton) if w.isVisible() and w.isEnabled() and w.window() is page.window()]
        for w in controls:
            if not w.isVisible():continue
            before=geometry();QTimer.singleShot(30,lambda:[p.close() for p in QApplication.topLevelWidgets() if p is not page and p.windowType()==Qt.WindowType.Popup])
            QTest.mouseClick(w,Qt.MouseButton.LeftButton);app.processEvents();assert geometry()==before,w.text();clicked.append(w.text())
            for dialog in page.findChildren(QDialog):dialog.close()
            page.sides.tabs.setCurrentIndex(1);app.processEvents()
        for combo in (page.controls.capture_mode,page.controls.scan_mode,page.controls.surface,page.camera_options.ir):
            before=geometry()
            for i in range(combo.count()):combo.setCurrentIndex(i);app.processEvents();assert geometry()==before,combo.objectName() or combo.currentText()
        before=geometry();page.maximize_camera("Color");app.processEvents()
        assert [(w.mapTo(page,QPoint()),w.size()) for w in (page.model,page.sides,page.go)]==[before[anchors.index(w)] for w in (page.model,page.sides,page.go)]
        assert 'Use part Range' in clicked and 'Undo' in clicked and 'Range test' in clicked and 'LED' in clicked
    finally:
        page.return_cameras()
        for dialog in page.findChildren(QDialog):dialog.close()
        page.close();app.processEvents()


def test_live_status_keeps_height_and_full_details_on_hover(app):
    from app2.qt.status_chip import LiveStatus
    c=LiveStatus();c.resize(220,c.height());old=c.size()
    for state,text in [('ok','Part framed'),('warn','Part goes past what depth can see: center it in the view'),('problem','Camera calibration missing')]:
        c.set_status(state,text,text+' More detail.');assert c.size()==old and c.toolTip().endswith('More detail.')
    c.close()


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_scan_tabs_tracks_show_actions_and_popup_controls_keep_page_geometry(app,tmp_path,monkeypatch,size):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_round25_ui import scene_state,settle
    from PySide6.QtWidgets import QDoubleSpinBox
    p=ScanPage(SessionBridge(),False);p.resize(*size);p.show()
    class Session:
        def __getattr__(self,name):return lambda *a:None
    p.session=Session();p.confirm=lambda *_:False;p.ask_text=lambda *_:None
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *_a,**_k:'')
    s=scene_state(tmp_path,has_led=True,bed_available=True,bed_size=30);s.capture_mode='live';s.live_undo=True;s.range_undo=True
    p.on_state(s);p.sides.tabs.setCurrentIndex(1);settle(app,p)
    anchors=(p.left,p.model,p.sides,p.go,p.camera_bar,p.controls.range_row,p.controls.gain_row,p.controls.led)
    def geometry():return [(w.mapTo(p,QPoint()),w.size()) for w in anchors]
    checked=[]
    def click(w):
        before=geometry();QTest.mouseClick(w,Qt.MouseButton.LeftButton);app.processEvents();assert geometry()==before,w.text();checked.append(w.text())
    try:
        before=geometry()
        for i in range(p.sides.tabs.count()):
            QTest.mouseClick(p.sides.tabs,Qt.MouseButton.LeftButton,pos=p.sides.tabs.tabRect(i).center());app.processEvents();assert geometry()==before
            checked.append('Tab '+p.sides.tabs.tabText(i))
        for track in (p.controls.range,p.controls.gain):
            before=geometry();QTest.mouseClick(track,Qt.MouseButton.LeftButton,pos=track.rect().center());app.processEvents();assert geometry()==before;checked.append(track.accessibleName() or type(track).__name__)
        from app2.qt.choices import Choices
        from PySide6.QtWidgets import QCheckBox
        for choice in p.camera_options.menu.findChildren(Choices):
            for option in choice.buttons:
                before=geometry();option.click();app.processEvents();assert geometry()==before;checked.append('Show '+option.text())
        for option in p.camera_options.menu.findChildren(QCheckBox):
            before=geometry();option.click();app.processEvents();assert geometry()==before;checked.append('Show '+option.text())
        for dialog in (p.controls.calibration_tools,p.controls.live_build.window,p.controls.capture_tools.wizard):
            dialog.show();app.processEvents()
            for field in dialog.findChildren(QDoubleSpinBox):
                before=geometry();field.stepUp();app.processEvents();assert geometry()==before;checked.append('Nudge '+field.prefix())
            for button in dialog.findChildren(QAbstractButton):
                if button.isVisible() and button.isEnabled():click(button)
            dialog.close();app.processEvents()
        assert len(checked)>20
        if os.getenv('POINTYOINK_CLICK_EVIDENCE'):
            import json
            from pathlib import Path
            root=Path(os.environ['POINTYOINK_CLICK_EVIDENCE']);root.mkdir(exist_ok=True,parents=True);(root/('controls-'+str(size[0])+'.json')).write_text(json.dumps(checked,indent=2))
    finally:
        for dialog in p.findChildren(QDialog):dialog.close()
        p.close();app.processEvents()
