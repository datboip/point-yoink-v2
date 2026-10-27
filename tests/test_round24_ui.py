import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
from types import SimpleNamespace as NS
import pytest
from PySide6.QtWidgets import QApplication,QWidget


@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def test_all_three_fill_width_and_do_not_move_between_modes(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    p=ScanPage(SessionBridge(),False);p.resize(1280,800);p.show();p._settings_user_choice=True
    try:
        shapes=[]
        for mode in ('near','far'):
            original=state(str(tmp_path),scan_mode=mode,picked_has_frames=False)
            p.on_state(original);p.relayout(1280,800);app.processEvents();p.relayout(1280,800)
            shapes.append([(x.width(),x.height()) for x in (p.ir,p.depth,p.colour)])
            assert p.ir.width()==p.left.width()
            from app2.qt.scan_page import GAP
            assert p.depth.width()+p.colour.width()+GAP==p.left.width()
        assert shapes[0]==shapes[1]
        assert not hasattr(p,"settings_toggle")
        assert p.go.parentWidget() is p.sides and p.sides.pages.isAncestorOf(p.controls_scroll)
    finally:p.close()


def test_near_chooser_dispatches_explicit_near_bundle(app,tmp_path,monkeypatch):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    p=ScanPage(SessionBridge(),False);calls=[];p.state=state(str(tmp_path),scan_mode='near')
    p.session=NS(set_near_calibration=lambda x:calls.append(x))
    monkeypatch.setattr('app2.qt.scan_page.QFileDialog.getExistingDirectory',lambda *a:str(tmp_path))
    try:p.choose_far_calibration();assert calls==[str(tmp_path)]
    finally:p.close()




def test_device_does_not_claim_queued_or_unacknowledged_light_is_on():
    from app2.scan.sources import DeviceControl
    import threading
    gate=threading.Event();done=threading.Event()
    d=DeviceControl(True,gain_fn=lambda x:None,led_fn=lambda x:gate.wait())
    try:
        d.led(True);assert d.led_state=='pending'
        gate.set();d.q.put((None,None,done));assert done.wait(2)
        # A setter return value alone does not prove a physical light state.
        d.led_fn=lambda x:None;done.clear();d.led(True);d.q.put((None,None,done));assert done.wait(2)
        assert d.led_state=='unverified'
        d.led_fn=lambda x:(_ for _ in ()).throw(OSError('fixture'));done.clear();d.led(True);d.q.put((None,None,done));assert done.wait(2)
        assert d.led_state=='failed'
    finally:gate.set();d.close()


def test_small_shell_settings_remain_reachable(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    p=ScanPage(SessionBridge(),False);p.resize(1280,586);p.show()
    try:
        p.on_state(state(str(tmp_path)));p.relayout(1280,586);app.processEvents()
        assert not p.controls.isVisible()
        p.sides.tabs.setCurrentIndex(1);app.processEvents()
        assert p.controls_scroll.height()>=p.controls.range_row.sizeHint().height()
        from PySide6.QtCore import QPoint
        center=p.controls.range.mapTo(p.live_strip,p.controls.range.rect().center())
        assert p.live_strip.rect().contains(center) and p.controls.range.isVisible()
        p.controls_scroll.ensureWidgetVisible(p.controls.outside_below);app.processEvents()
        center=p.controls.outside_below.mapTo(p.controls_scroll.viewport(),p.controls.outside_below.rect().center())
        assert p.controls_scroll.viewport().rect().contains(center)
        assert not p.one_pic and p.go.isVisible()
    finally:p.close()
