from pathlib import Path
import json,struct
import numpy as np
import pytest
from app2.scan import calibration


def bundle(path,fx=800):
    path.mkdir(parents=True)
    (path/'Pl.bin').write_bytes(struct.pack('<HH9f',800,600,fx,0,400,0,fx,300,0,0,1))
    return path


def test_near_registry_is_separate_and_rejects_far(tmp_path):
    near=calibration.store(bundle(tmp_path/'n'),tmp_path/'config','near')
    far=calibration.store(bundle(tmp_path/'f',500),tmp_path/'config','far')
    assert near['folder']!=far['folder']
    assert calibration.load(near['folder'],'near')['intrinsics'][0]==800
    with pytest.raises(calibration.CalibrationUnavailable):calibration.load(far['folder'],'near')


def test_detector_needs_repeated_unambiguous_evidence(monkeypatch):
    import app2.scan.mode_detection as m
    from scanner import bed
    rows={'near':dict(bed_fit=True),'far':dict(bed_fit=False)}
    monkeypatch.setattr(m,'compare',lambda *a,**kw:rows)
    d=m.Detector({'near':{},'far':{}})
    try:
        for i in range(bed.ACCEPT_HITS):
            d.offer(np.zeros((2,2)),np.zeros((2,2)));d.future.result();e=d.poll()
            assert e['mode']==('near' if i+1==bed.ACCEPT_HITS else None)
        rows['far']['bed_fit']=True
        for _ in range(bed.ACCEPT_HITS):
            d.offer(np.zeros((2,2)),np.zeros((2,2)));d.future.result();e=d.poll()
        assert e['mode'] is None
    finally:d.close()


def test_capture_refuses_contrary_frame_evidence():
    from app2.scan.session import ScanSession
    s=object.__new__(ScanSession);s.scan_mode='far';s.mode_evidence=dict(mode='near',state='matched')
    assert 'Small part' in s._capture_calibration_error()


def test_check_in_progress_does_not_start_capture():
    from app2.scan.session import ScanSession
    from types import SimpleNamespace
    s=object.__new__(ScanSession);s.scan_mode='near';s.mode_evidence={};s.calibration_error=''
    s.mode_detector=SimpleNamespace(evidence=dict(state='checking'))
    assert not s._capture_calibration_error()  # the Object setting stands until frames contradict it


def test_startup_newest_far_is_not_registered_as_near(tmp_path,monkeypatch):
    from app2 import __main__ as main
    from app2.scan import session,bedhook
    from types import SimpleNamespace
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'settings.json'))
    root=tmp_path/'projects';project=root/'ProjectLiveFixture';far=bundle(project/'data'/'side'/'param',500)
    monkeypatch.setattr(main,'offline_safe_source',lambda:SimpleNamespace(kind='live'))
    monkeypatch.setattr(bedhook,'load_bed_hook',lambda:None)
    monkeypatch.setattr(session,'ScanSession',lambda cfg,*a,**k:cfg)
    cfg=main.session_factory(main.parse(['--project',str(project),'--root',str(root)]),'image')(lambda *a:None)
    assert Path(cfg.param_dir)==far and cfg.near_param is None
