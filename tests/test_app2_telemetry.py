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


def test_panel_fixed_geometry_and_no_second_window():
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.telemetry_panel import TelemetryPanel
    app=QApplication.instance() or QApplication([])
    parent=QWidget(); parent.resize(1280,800)
    client=TelemetryClient(); panel=TelemetryPanel(parent,client)
    parent.show(); panel.show(); app.processEvents()
    sizes=[w.geometry() for row in panel.fields.values() for w in row]
    assert not panel.isWindow()
    with client._lock:
        client._sample=decode_packet(packet(),received=time.monotonic())
        client._state='Receiving'; client._count=22
    panel.refresh(); app.processEvents()
    assert panel.width()==380 and panel.height()==462
    assert sizes==[w.geometry() for row in panel.fields.values() for w in row]
    for widget in (panel.done,panel.age,panel.status,panel.connect_button,
                   *[cell for row in panel.fields.values() for cell in row]):
        assert panel.rect().contains(widget.geometry())
    panel.dismiss(); assert client.snapshot()[0]=='Disconnected'
    panel.shutdown(); parent.close()
