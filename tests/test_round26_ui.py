import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');os.environ['POINTYOINK_NO_DEVICE']='1'
from types import SimpleNamespace as NS
import pytest
from PySide6.QtWidgets import QApplication,QLabel
from PySide6.QtCore import QPoint
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge
from test_round25_ui import scene_state


@pytest.fixture
def page(tmp_path):
    app=QApplication.instance() or QApplication([]);p=ScanPage(SessionBridge(),False);p.resize(1280,800);p.show();s=scene_state(tmp_path);p.on_state(s)
    for _ in range(8):app.processEvents()
    p.relayout(p.width(),p.height())
    yield app,p,s
    p.close();app.processEvents()


@pytest.mark.parametrize('size',[(1280,800),(1920,1080),(1280,586)])
def test_settings_live_on_right_and_actions_stay_pinned_on_both_tabs(page,size):
    app,p,s=page;p.resize(*size);p._size=None;p.relayout(*size)
    assert not hasattr(p,'settings_toggle')
    assert p.sides.pages.isAncestorOf(p.controls) and not p.left.isAncestorOf(p.controls)
    assert [p.sides.tabs.tabText(i) for i in range(2)]==['Sides','Settings']
    positions=[]
    for index in (0,1):
        p.sides.tabs.setCurrentIndex(index)
        for _ in range(8):app.processEvents()
        positions.append((p.go.mapTo(p,QPoint(0,0)),p.sides.buildbox.mapTo(p,QPoint(0,0))))
        assert p.go.isVisible() and p.sides.buildbox.isVisible()
        assert p.sides.new.isVisible()   # + New side under both tabs
        assert p.sides.scroll.isVisible()==(index==0)
    assert positions[0]==positions[1]
    p.controls.range.dragging=True;p.on_state(s);assert p.sides.tabs.currentIndex()==1
    p.controls.range.dragging=False


def test_compact_framing_has_one_status_and_details_only_on_hover(page):
    app,p,s=page;p.sides.tabs.setCurrentIndex(1);s.scene_setup=dict(points=169274,fill=.92,dots=None,problems=['Part touches picture edge.'],hint='')
    s.range_undo=True;p.on_state(s)
    for _ in range(8):app.processEvents()
    c=p.controls
    assert c.frame_range.parentWidget() is c.range_undo.parentWidget()
    assert not hasattr(c,'frame_lock')
    assert c.range_undo.isVisible() and '169274' not in c.frame_note.text()
    assert c.frame_note.toolTip()=='Part touches picture edge.' and '169274' not in c.frame_note.toolTip()   # hover text is only what to do next
    assert c.scene_note.isHidden() and c.distance_status.isHidden() and c.calibration_info.isHidden()
    s.range_undo=False;p.on_state(s);assert not c.range_undo.isHidden() and not c.range_undo.isEnabled()   # stays in place, disabled


def test_keyboard_commands_work_from_the_settings_tab(page):
    app,p,s=page;p.sides.tabs.setCurrentIndex(1);calls=[];p.session=NS(toggle_scan=lambda:calls.append('scan'))
    p.key_space();assert calls==['scan'] and not hasattr(p,'key_frame')
    p.on_state(s);assert p.sides.tabs.currentIndex()==1


def test_distance_labels_do_not_overlap_each_other_marker_row_or_range_values(page):
    app,p,s=page;b=p.distance_bar
    s.range_guide['zones']=[dict(lo=100,hi=700,state='good depth'),dict(lo=200,hi=500,state='strong depth',live=True)]
    b.set_guide(s.range_guide,(100,900),(0,1000))
    for width in (260,600,1000):
        b.resize(width,b.sizeHint().height());layout=b.text_layout();labels=layout['labels']
        for i,(_,_,rect) in enumerate(labels):
            assert rect.bottom()<=layout['line']
            assert all(not rect.intersects(other) for _,_,other in labels[i+1:])
        assert not layout['distance'][1].intersects(layout['cut'][1])


def test_framing_row_does_not_resize_the_panel_when_undo_appears(page):
    app,p,s=page;p.sides.tabs.setCurrentIndex(1);p._size=None;p.relayout(p.width(),p.height());before=p.sides.width()
    s.range_undo=True;p.on_state(s)
    for _ in range(8):app.processEvents()
    assert p.sides.width()==before


def test_normal_settings_fit_their_viewport_without_horizontal_clipping(page):
    app,p,s=page;p.sides.tabs.setCurrentIndex(1)
    for _ in range(8):app.processEvents()
    assert p.controls.width()<=p.controls_scroll.viewport().width()
    assert p.controls.save_color.text()=='Save color' and '770 MB' in p.controls.save_color.toolTip()
    p.controls.set_auto_msg('Trying gain 8 after a full turn')
    assert p.controls.auto_msg.isHidden() and 'gain 8' in p.controls.gain_row.toolTip()


def test_camera_check_is_not_a_request_to_choose_calibration_files(page):
    app,p,s=page;s.calibration_missing=False;s.calibration_error='Checking camera mode. Show learned dots.';s.mode_evidence={}
    p.on_state(s)
    assert p.go.text_now()=='Checking camera' and not p.go.isEnabled()
    assert not p.controls.calibration_setup.isChecked() and 'dots' in p.controls.frame_note.text()


def test_real_missing_calibration_surfaces_settings_once_without_interrupting_a_drag(page):
    app,p,s=page;s.calibration_missing=True;s.calibration_error='Small part calibration is unavailable.'
    p.on_state(s);assert p.sides.tabs.currentIndex()==1
    p.sides.tabs.setCurrentIndex(0);p.on_state(s);assert p.sides.tabs.currentIndex()==0
    s.calibration_missing=False;s.calibration_error='';p.on_state(s)
    p.controls.range.dragging=True;s.calibration_missing=True;s.calibration_error='Calibration is unavailable.'
    p.on_state(s);assert p.sides.tabs.currentIndex()==0
    p.controls.range.dragging=False


def test_camera_mismatch_has_one_click_fix_on_the_camera_view(page):
    app,p,s=page;calls=[];p.session=NS(switch_camera_match=lambda:calls.append('switch'))
    s.scan_mode='far';s.mode_evidence=dict(mode='near');p.on_state(s)
    for _ in range(4):app.processEvents()
    assert p.camera_match.isVisible() and p.camera_match.text()=='Use Small part'
    assert p.camera_match.parentWidget() is p.model
    p.camera_match.click();assert calls==['switch']
    s.scan_mode='near';p.on_state(s);assert p.camera_match.isHidden()


def test_overlay_nudge_and_range_test_actions_are_wired(page):
    app,p,s=page;calls=[]
    p.session=NS(set_overlay=lambda values:calls.append(('nudge',values)),auto_overlay=lambda:calls.append(('auto',)),reset_overlay=lambda:calls.append(('reset',)),range_test=lambda:calls.append(('test',)),reset_range_test=lambda:calls.append(('reset-test',)))
    p.sides.tabs.setCurrentIndex(1);p.controls.calibration_setup.setChecked(True);tools=p.controls.capture_tools
    tools.fields['x'].setValue(2);tools.auto.click();tools.reset.click();tools.range_button.click();tools.wizard.start.emit();tools.reset_range.click()
    assert calls[0][0]=='nudge' and calls[0][1]['x']==2
    assert calls[1:]==[('auto',),('reset',),('test',),('reset-test',)]
    s.range_test=dict(active=True,hint='Keep going slowly');p.on_state(s)
    assert tools.range_button.text()=='Range test' and tools.wizard.pages.currentIndex()==2 and not p.go.isEnabled()


def test_capture_distance_display_uses_shared_palette_and_unknown_gray(page):
    app,p,s=page;m=p.model
    m.view=NS(point_palette=None)
    try:
        c=dict(P=__import__('numpy').zeros((3,3)),L=__import__('numpy').array([0,0,0]),B=__import__('numpy').array([255,1,2]))
        levels=m._display_levels(c);assert levels.tolist()==[4,1,2] and m.overlay.distance_legend
        from app2.scan.range_guide import BAND_RGB
        assert __import__('numpy').allclose(m.view.point_palette[2],__import__('numpy').array(BAND_RGB['best'])/255)
        m.distance_mode.blockSignals(True);m.distance_mode.setCurrentIndex(2);assert m._display_levels(c).tolist()==[4,4,4]
    finally:m.view=None


def test_pending_distance_measurement_keeps_the_last_colored_surface(page):
    import numpy as np
    app,p,s=page;m=p.model;m.view=NS()
    try:
        old=dict(side_no=1,P=np.zeros((2,3)),B=np.array([1,2]),seq=1);m._last_cloud=old
        pending=dict(side_no=1,P=np.ones((2,3)),B=np.array([255,255]),seq=2);m.on_cloud(pending)
        assert m._last_cloud is old and m.overlay.distance_pending
    finally:m.view=None


def test_capture_distance_selector_leaves_room_for_touch_up_guide(page):
    app,p,s=page;m=p.model;m.guide_target.addItem('Camera above the bed: fill the remaining area');m.guide_target.show();m.distance_mode.show()
    for width in (500,1000):
        m.resize(width,600);m._refresh_overlay()
        from PySide6.QtCore import QRect,QPoint
        guide=QRect(m.guide_target.mapTo(p,QPoint()),m.guide_target.size())
        selector=QRect(m.distance_mode.mapTo(p,QPoint()),m.distance_mode.size())
        assert not guide.intersects(selector)
        assert m.guide_target.width()>0
