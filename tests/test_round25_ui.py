import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');os.environ['POINTYOINK_NO_DEVICE']='1'
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge
from app2.scan.range_guide import from_scene,COLORS
from app2.scan.framing import analyze,Framing
from test_app2_workflow_fixes import state


@pytest.fixture
def page(tmp_path):
    app=QApplication.instance() or QApplication([])
    p=ScanPage(SessionBridge(),False);p.resize(1280,800);p.show();p._settings_user_choice=True
    yield app,p,tmp_path
    p.close();app.processEvents()


def scene_state(path,**kw):
    d=np.full((80,100),4000,np.uint16);d[25:55,30:70]=3000
    part=analyze(d,(100,100,50,40),(np.array([0,0,-1]),400,np.array([0,0,400]),60),(0,1000))
    g=from_scene(d,'near',part);fr=Framing(3).update(part,g,0)
    return state(str(path),scan_mode='near',range_limits=(0,1000),range_guide=g,framing=fr,**kw)


def settle(app,p):
    p._size=None;p.relayout(p.width(),p.height())
    for _ in range(6):app.processEvents()
    p.place_distance()


@pytest.mark.parametrize('size',[(1280,800),(1920,1080),(1280,586),(1280,1000)])
def test_footer_has_its_own_space_and_rewind_wins(page,size):
    app,p,path=page;p.resize(*size);s=scene_state(path);p.on_state(s);settle(app,p)
    assert p.distance_bar.isVisible()
    viewport=p.model.image.geometry();image_bottom=p.model.image.mapTo(p,QPoint(0,0)).y()+viewport.height()
    assert image_bottom<=p.distance_bar.y()
    assert p.model.overlay.geometry().bottom()<p.model.height()-p.model._footer_height if hasattr(p.model,'_footer_height') else True
    assert p.go.parentWidget() is p.sides
    buildtop=p.sides.buildbox.mapTo(p.sides,p.sides.buildbox.rect().topLeft()).y()
    assert buildtop-p.go.geometry().bottom()<=p.sides.layout().spacing()+1
    assert p.left.isAncestorOf(p.live_strip)   # pass controls under the camera views
    s.picked=1;s.picked_has_frames=True;p.on_state(s);settle(app,p)
    assert p.rewind_bar.isVisible() and not p.distance_bar.isVisible()
    m=p.model.layout().contentsMargins();assert m.bottom()==m.top()   # no footer space reserved under the model while the rewind bar is shown (the ring margin is the same all round)


def test_measured_zones_live_support_and_old_internal_keys(page):
    app,p,path=page;s=scene_state(path);p.on_state(s);settle(app,p)
    b=p.distance_bar
    best=next(z for z in b.guide['zones'] if z['state']=='strong depth')
    assert b.zone_at((best['lo']+best['hi'])/2)=='strong depth'
    assert b.zone_at(900)=='too far' and b.cut==(100,450)
    assert len(set(COLORS[z['state']] for z in b.guide['zones']))>1
    assert [p.controls.scan_mode.itemData(i) for i in range(2)]==['near','far']
    assert [p.controls.scan_mode.itemText(i) for i in range(2)]==['Small part','Large part']
    assert p.controls.range.guide is None


def test_calibration_chooser_is_setup_only_until_problem(page):
    app,p,path=page;s=scene_state(path);p.sides.tabs.setCurrentIndex(1);p.on_state(s);settle(app,p)
    assert not p.controls.choose_far.isVisible()
    p.controls.calibration_setup.click();settle(app,p);assert p.controls.choose_far.isVisible()
    p.controls.calibration_setup.click();s.calibration_error='Calibration is missing.';p.on_state(s);settle(app,p)
    assert p.controls.choose_far.isVisible()


def test_mismatch_button_explains_and_dispatches_fix_without_capture(page):
    app,p,path=page;s=scene_state(path);s.scan_mode='far';s.mode_evidence=dict(mode='near');s.calibration_error='Frames match Small part.'
    calls=[];p.session=NS(switch_camera_match=lambda:calls.append('fix'),go=lambda:calls.append('capture'))
    p.on_state(s);settle(app,p);assert p.go.text_now()=='Use Small part' and p.go.isEnabled()
    assert p.go.property('kind')=='pause'
    assert p.go.title.fontMetrics().horizontalAdvance(p.go.text_now())<=p.go.title.width()
    p.go.click();assert calls==['fix']


def test_range_actions_and_busy_guard_without_lock(page):
    app,p,path=page;s=scene_state(path);calls=[]
    p.session=NS(use_frame_range=lambda:calls.append('range'),undo_frame_range=lambda:calls.append('undo'),reframe=lambda:calls.append('reframe'))
    s.range_undo=True;p.on_state(s);settle(app,p)
    for w in (p.controls.frame_range,p.controls.range_undo):w.click()
    assert calls==['range','undo'] and 'Part Range applied' in p.controls.frame_note.text()
    s.recording=True;p.on_state(s)
    assert not p.controls.frame_range.isEnabled() and not hasattr(p.controls,"frame_lock")


def test_idle_build_progress_has_no_reserved_block(page):
    app,p,path=page;p.on_state(scene_state(path));settle(app,p);box=p.sides.buildbox
    idle=box.sizeHint().height();assert not box.progress.isVisible()
    prog=NS(finished=False,frac=.5,step='Meshing',elapsed_text=lambda:'50%')
    box.set_progress(prog);settle(app,p);active=box.sizeHint().height()
    assert active>idle and box.progress.isVisible()
    box.set_progress(None);settle(app,p);assert box.sizeHint().height()==idle


def test_build_progress_preserves_the_cards_being_viewed(page):
    from PySide6.QtWidgets import QLabel
    app,p,path=page;p.on_state(scene_state(path));settle(app,p)
    for i in range(10):
        card=QLabel('Side '+str(i+1));card.setFixedHeight(card.fontMetrics().lineSpacing()*6);p.sides.lv.insertWidget(i,card)
    p.sides.picked_card=card;settle(app,p)
    sb=p.sides.scroll.verticalScrollBar();sb.setValue(sb.maximum()//2);before=sb.value()
    box=p.sides.buildbox;box.set_progress(NS(finished=False,frac=.5,step='Fusing',elapsed_text=lambda:'50%'))
    for _ in range(8):app.processEvents()
    assert sb.value()==before
    box.set_progress(None)
    for _ in range(8):app.processEvents()
    assert sb.value()==before
