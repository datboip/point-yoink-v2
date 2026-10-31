"""Motion stream tests use loopback only and an isolated settings file."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink-motion-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import math
import socket
import struct
import threading
import time
import pytest
from app2.scan.telemetry import TelemetryClient, configured_ip, decode_packet

_APP = None


def _qapp():
    """One QApplication for the whole module: a test-local one is deleted when the test ends, while Qt objects live on."""
    global _APP
    from PySide6.QtWidgets import QApplication
    _APP = _APP or QApplication.instance() or QApplication([])
    return _APP


def packet(quaternion=(0.,0.,0.,1.)):
    return b'12345678'+struct.pack('<13f', *quaternion, 1,2,3,4,5,6,7,8,9)


def wait_for(check):
    deadline = time.monotonic()+3
    while time.monotonic()<deadline:
        if check(): return
        time.sleep(.01)
    assert check()


def test_packet_angles_and_fields():
    sample = decode_packet(packet((0,0,math.sqrt(.5),math.sqrt(.5))), received=123)
    assert sample.angles == pytest.approx((90,0,0))
    assert sample.position == (7,8,9)
    assert sample.gyro == (1,2,3)
    assert sample.acceleration == (4,5,6)
    assert sample.received == 123


@pytest.mark.parametrize('bad',[b'', packet((0,0,0,0)), packet((float('nan'),0,0,1))])
def test_invalid_packet(bad):
    with pytest.raises(ValueError): decode_packet(bad)


def test_config_read_only(tmp_path, monkeypatch):
    cfg=tmp_path/'config.json'; cfg.write_text('{"scanner_ip":"192.0.2.20"}')
    monkeypatch.setenv('POINTYOINK_CONFIG',str(cfg)); before=cfg.read_bytes()
    assert configured_ip() == '192.0.2.20'
    assert cfg.read_bytes() == before
    cfg.write_text('broken'); assert configured_ip() == ''


def test_fragmented_stream_latest_sample_stale_and_disconnect():
    server=socket.socket(); server.bind(('127.0.0.1',0)); server.listen(1)
    release=threading.Event()
    def serve():
        with server:
            conn,_=server.accept()
            with conn:
                payload=packet((0,0,0,0))+packet()*100
                conn.sendall(payload[:17]); conn.sendall(payload[17:])
                release.wait(3)
    worker=threading.Thread(target=serve,daemon=True); worker.start()
    client=TelemetryClient()
    try:
        client.connect('127.0.0.1',server.getsockname()[1])
        wait_for(lambda:client.snapshot()[2]==100)
        state,sample,count,age=client.snapshot()
        assert state=='Receiving' and count==100
        assert client.snapshot(now=sample.received+3)[0]=='Stale'
        client.disconnect(); assert client.snapshot()[0]=='Disconnected'
    finally:
        client.close(); release.set(); worker.join(3)


def test_invalid_address_does_not_start_network():
    client=TelemetryClient()
    with pytest.raises(ValueError): client.connect('not an IP')
    assert client.snapshot()[0]=='Disconnected'


def test_panel_is_a_tab_page_with_stable_fields_and_no_second_window():
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.telemetry_panel import TelemetryPanel
    app=_qapp()
    parent=QWidget(); parent.resize(1280,800)
    client=TelemetryClient(); panel=TelemetryPanel(parent,client); panel.resize(480,560)
    parent.show(); panel.show(); app.processEvents()
    sizes=[w.size() for row in panel.fields.values() for w in row]
    assert not panel.isWindow()
    with client._lock:
        client._sample=decode_packet(packet(),received=time.monotonic())
        client._state='Receiving'; client._count=22
    panel.refresh(); app.processEvents()
    assert sizes==[w.size() for row in panel.fields.values() for w in row]
    for widget in (panel.age,panel.status,panel.connect_button,*[cell for row in panel.fields.values() for cell in row]):
        assert panel.rect().contains(widget.geometry())
    panel.toggle_connection(); assert panel.manual_off and client.snapshot()[0]=='Disconnected'
    panel.shutdown(); parent.close()


def test_finder_returns_the_address_that_really_streams_motion():
    import socket, threading
    from app2.scan import scanner_ip
    server=socket.socket(); server.bind(('127.0.0.1',0)); server.listen(4); port=server.getsockname()[1]
    def serve():
        while True:
            try: conn,_=server.accept()
            except OSError: return
            try:
                for _ in range(5): conn.sendall(packet()); time.sleep(.02)
            except OSError: pass
            conn.close()
    threading.Thread(target=serve,daemon=True).start()
    try:
        assert scanner_ip.speaks_motion('127.0.0.1',port)
        assert not scanner_ip.speaks_motion('127.0.0.1',1)
        assert scanner_ip.find('127.0.0.1',port=port)=='127.0.0.1'
        assert scanner_ip.find('',port=port,own='10.9.9.9',arp=['8.8.8.8','10.9.9.7'])==''                 # a public address is never tried
        real = scanner_ip.is_local; scanner_ip.is_local = lambda ip: True      # the test server is on loopback
        try: assert scanner_ip.find('',port=port,own='10.9.9.9',arp=['127.0.0.1','10.9.9.7'])=='127.0.0.1'
        finally: scanner_ip.is_local = real
        assert scanner_ip.find('',port=port,own='10.9.9.9',arp=[])==''
    finally: server.close()


def test_remember_keeps_other_settings_and_rejects_junk(tmp_path,monkeypatch):
    import json
    from app2.scan import scanner_ip
    cfg=tmp_path/'config.json'; cfg.write_text(json.dumps({'general':{'open_last':True}}))
    monkeypatch.setenv('POINTYOINK_CONFIG',str(cfg))
    assert not scanner_ip.remember('not an ip')
    assert scanner_ip.remember('192.168.1.50')
    assert not scanner_ip.remember('8.8.8.8') and not scanner_ip.remember('127.0.0.1')
    data=json.loads(cfg.read_text())
    assert data['scanner_ip']=='192.168.1.50' and data['general']['open_last'] is True
    from app2.scan.telemetry import configured_ip
    assert configured_ip()=='192.168.1.50'


def test_motion_link_connects_by_itself_and_asks_once_when_nothing_answers(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.telemetry_panel import TelemetryPanel
    from app2.qt.motion_link import MotionLink
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=_qapp()
    parent=QWidget(); calls=[]; live=[False]
    client=TelemetryClient(); client.connect=lambda ip,port=9999:calls.append(ip)
    panel=TelemetryPanel(parent,client); answers=['192.168.1.50','']
    link=MotionLink(panel,lambda:live[0],parent,find=lambda saved:answers.pop(0))
    asked=[]; link.need_address.connect(lambda:asked.append(1))
    link.tick(); assert calls==[]                       # no scanner yet: nothing to look for
    live[0]=True; link.tick()
    for _ in range(50):
        app.processEvents(); time.sleep(.01)
        if calls: break
    assert calls==['192.168.1.50'] and not asked and panel.address.text()=='192.168.1.50'
    link._next=0.; client.connect=lambda ip,port=9999:calls.append(ip); link.tick()
    for _ in range(50):
        app.processEvents(); time.sleep(.01)
        if asked: break
    assert asked==[1] and 'Could not find' in panel.notice.text()
    link._next=0.; answers.append(''); link.tick()
    for _ in range(50): app.processEvents(); time.sleep(.01)
    assert asked==[1]                                   # asks once per run
    panel.manual_off=True; link._next=0.; before=len(answers); link.tick(); assert len(answers)==before   # Disconnect is respected
    link.shutdown(); panel.shutdown(); parent.close()


def test_scanner_panel_has_one_tab_for_use_and_one_for_motion(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    from PySide6.QtWidgets import QApplication, QMainWindow
    from app2.qt.telemetry_panel import TelemetryPanel
    from app2.qt.resource_panel import ResourcePanel
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=_qapp()
    win=QMainWindow(); win.session=None; win.motion_panel=TelemetryPanel(win,TelemetryClient())
    panel=ResourcePanel('SCANNER',win,sampler=NS(last={},samples=[],build_usage=NS(average={})))
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())]==['Use and limits','Motion']
    panel.show_tab('Motion'); assert panel.tabs.currentIndex()==1
    assert win.motion_panel.parentWidget() is not win
    panel.close(); assert win.motion_panel.parentWidget() is win
    win.motion_panel.shutdown(); win.close()


def test_motion_view_draws_with_and_without_data():
    from app2.qt.telemetry_panel import MotionView
    app=_qapp(); view=MotionView(); view.resize(420,160); view.show()
    view.grab()                                           # nothing yet: "no data"
    for i in range(30): view.set_angles((170+i*.5,-5+i,20-i),now=100+i*.1)
    view.set_angles((-175,10,0),now=103.1)                # heading wraps through 180: no line across the graph
    view.grab(); assert len(view.trace)>2
    view.set_angles(None); assert view.angles is None
    view.set_angles((0,0,0),now=200.); assert len(view.trace)==1   # older than ten seconds is dropped
    view.close()


def test_level_pose_becomes_zero_and_is_remembered(tmp_path,monkeypatch):
    from app2.scan import telemetry as T
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    tilted=decode_packet(packet((0.2,0.1,0.05,0.97)),received=1.)
    assert max(abs(v) for v in tilted.angles[1:])>1                          # raw angles are off level
    zero=tilted.quaternion
    assert all(abs(v)<1e-6 for v in tilted.angles_from(zero))
    nudge=decode_packet(packet((0.2,0.1,0.05+0.02,0.97)),received=2.)
    assert 0<max(abs(v) for v in nudge.angles_from(zero))<10                  # a small turn from level reads small
    assert T.configured_zero() is None
    assert T.save_zero(zero) and T.configured_zero()==pytest.approx(zero)
    assert T.save_zero(None) and T.configured_zero() is None


def test_axes_from_two_poses_give_pitch_roll_and_heading():
    from app2.scan.telemetry import axes_from, body_angles
    axes=axes_from((0,0,1),(-1,0,0))                  # sensor z is up when flat; pointed down, gravity reads against sensor x
    assert axes['up']==pytest.approx((0,0,1)) and axes['forward']==pytest.approx((1,0,0))
    assert body_angles(axes,(0,0,1),(0,0,0,1),(0,0,0,1))==pytest.approx((0,0,0))
    h,p,r=body_angles(axes,(-.5,0,math.sqrt(.75)),(0,0,0,1),(0,0,0,1)); assert p==pytest.approx(-30) and r==pytest.approx(0,abs=1e-6)   # pointing down 30
    h,p,r=body_angles(axes,(0,.5,math.sqrt(.75)),(0,0,0,1),(0,0,0,1)); assert p==pytest.approx(0,abs=1e-6) and r==pytest.approx(30)       # right side down
    h,p,r=body_angles(axes,(0,0,1),(0,0,math.sin(.4),math.cos(.4)),(0,0,0,1)); assert h==pytest.approx(math.degrees(.8))                # turned 0.8 rad
    assert axes_from((0,0,1),(0,0,-1)) is None         # upside down is not a quarter turn from flat
    assert axes_from((0,0,0),(1,0,0)) is None


def test_panel_calibrates_in_two_poses_and_remembers(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QWidget
    from app2.qt.telemetry_panel import TelemetryPanel
    from app2.scan import telemetry as T
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=_qapp(); parent=QWidget(); client=TelemetryClient(); panel=TelemetryPanel(parent,client)
    def hold(accel,quaternion=(0.,0.,0.,1.)):
        now=time.monotonic()
        with client._lock: client._history.clear()
        for i in range(8):
            client.ingest(b'12345678'+struct.pack('<13f',*quaternion,0,0,0,*accel,0,0,0),now-.5+i*.05)
    panel.calibrate(); assert 'No motion data' in panel.zero_note.text()
    hold((0,0,1)); panel.calibrate(); assert panel.step==2 and 'step 2' in panel.zero_note.text() and panel.capture_down.isEnabled()
    hold((0,0,-1)); panel.calibrate(); assert panel.step==2 and 'does not look' in panel.zero_note.text() and panel.axes is None   # not a quarter turn
    hold((-1,0,0)); panel.calibrate(); app.processEvents()
    assert panel.step==0 and panel.axes is not None and 'Calibrated' in panel.zero_note.text()
    assert T.configured_axes()['forward']==pytest.approx((1,0,0)) and T.configured_zero()==pytest.approx((0,0,0,1))
    hold((-.5,0,math.sqrt(.75))); panel.refresh(); assert panel.view.angles[1]==pytest.approx(-30)
    moving=TelemetryClient(); panel.client=moving; now=time.monotonic()
    for i in range(8): moving.ingest(packet((0.2+i*.02,0.1,0.05,0.97)),now-.5+i*.05)
    panel.calibrate(); assert 'moved' in panel.zero_note.text()
    panel.forget_level(); assert panel.axes is None and T.configured_axes() is None and not panel.clear_level.isEnabled()
    panel.shutdown(); parent.close()


def test_set_heading_here_zeroes_heading_only(tmp_path, monkeypatch):
    import time as _t
    from app2.qt import telemetry_panel as tp
    from app2.scan.telemetry import Sample
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'c.json'))
    panel = tp.TelemetryPanel()
    panel.axes = {'up': (0., 0., 1.), 'forward': (1., 0., 0.), 'right': (0., -1., 0.)}
    q = (0., 0., .3826834, .9238795)                       # turned 45 degrees about up
    s = Sample(quaternion=q, gyro=(0., 0., 0.), acceleration=(0., 0., 1.), position=(0., 0., 0.), received=0.)
    panel.client.recent = lambda seconds: [s] * 5
    panel.client.snapshot = lambda: ('Receiving', s, 5, 0.)
    panel.set_heading_here()
    heading, pitch, roll = tp.body_angles(panel.axes, s.acceleration, s.quaternion, panel.zero)
    assert abs(heading) < .5 and abs(pitch) < .5 and abs(roll) < .5
