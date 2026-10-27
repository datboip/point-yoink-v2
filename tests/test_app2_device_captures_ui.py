import os,tempfile,time
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QWidget
from PySide6.QtTest import QTest
from app2.qt.device_captures import DeviceCaptures

def wait(panel,app):
    end=time.monotonic()+8
    while panel.proc and time.monotonic()<end:app.processEvents();QTest.qWait(10)
    assert panel.proc is None,panel.status.text()

def test_capture_worker_roundtrip_without_device(tmp_path):
    app=QApplication.instance() or QApplication([]);parent=QWidget();parent.resize(1024,640)
    panel=DeviceCaptures(parent);root=tmp_path/'source';(root/'Screenshots').mkdir(parents=True);(root/'Movies').mkdir()
    (root/'Screenshots/a.png').write_bytes(b'png fixture');(root/'Movies/a.mp4').write_bytes(b'video fixture')
    (root/'Projects').mkdir();(root/'Projects/private.mp4').write_bytes(b'not a capture')
    parent.show();panel.present();panel.root=str(root);panel.scan_files();wait(panel,app)
    assert panel.files.topLevelItemCount()==2,panel.status.text()
    assert parent.rect().contains(panel.geometry()) and not panel.isWindow()
    panel.select_all(True);dest=tmp_path/'saved';panel.destination.setText(str(dest));panel.copy();wait(panel,app)
    assert (dest/'Screenshots/a.png').read_bytes()==b'png fixture'
    assert (dest/'Movies/a.mp4').read_bytes()==b'video fixture'
    assert (root/'Screenshots/a.png').exists()
    assert 'Saved 2' in panel.status.text()
    panel.dismiss();parent.close()

def test_scan_session_closed_before_connect_and_restored(tmp_path,monkeypatch):
    from app2.qt.shell import MainWindow
    from app2.scan.ui_model import ScanState
    from types import SimpleNamespace
    app=QApplication.instance() or QApplication([])
    def offline(*a,**k):raise RuntimeError('test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path));calls=[]
    win._crashed=False;win.state=ScanState(project=str(tmp_path));win.session=SimpleNamespace(close=lambda **kw:calls.append(kw))
    monkeypatch.setattr(win.captures,'connect_usb',lambda:calls.append('connect'))
    monkeypatch.setattr(win,'start_session',lambda **kw:calls.append(('resume',kw)))
    win.connect_captures();assert calls==[{'then':'captures'}]
    win.on_closed({'then':'captures'});assert calls[-1]=='connect' and win.session is None
    win.captures_done();assert calls[-1]==('resume',{'project':str(tmp_path),'last':False})
    win._really=True;win.build.shutdown();win.close()
