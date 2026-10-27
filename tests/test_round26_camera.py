from types import SimpleNamespace as NS
from test_round25_camera import session,frame,evidence


def test_unclear_two_to_zero_votes_keep_confirmed_notice_and_start(monkeypatch):
    s,sig=session(monkeypatch);s.mode_evidence=evidence('near',sig);s.mode_notice='Camera match: Small part at 342 mm (7 dots).';s.calibration_error=''
    s.mode_detector.result=dict(mode=None,state='uncertain',votes=dict(near=2,far=0),frames=[])
    notice=s.mode_notice;s._check_camera_mode(frame(1))
    assert s.mode_evidence['mode']=='near' and s.mode_notice==notice and not s._capture_calibration_error()
    assert any('held' in e for e in s.events) and 'state' not in s.events


def test_new_background_checks_do_not_replace_the_confirmed_notice(monkeypatch):
    s,sig=session(monkeypatch);s.mode_evidence=evidence('near',sig);s.mode_notice='Camera match: Small part.'
    s._check_camera_mode(frame(5));assert s.mode_notice=='Camera match: Small part.' and s._mode_background


def test_losing_dots_does_not_move_the_confirmed_scene_reference(monkeypatch):
    s,sig=session(monkeypatch);s.mode_evidence=evidence('near',sig);s.mode_notice='Camera match: Small part.';old=dict(s._mode_signature)
    sig['distance']=350;s.mode_detector.result=dict(mode=None,state='uncertain',votes=dict(near=0,far=0),frames=[])
    s._check_camera_mode(frame(1));assert s._mode_signature==old
    sig['distance']=370;s.mode_detector.evidence=dict(state='uncertain');s._check_camera_mode(frame(5))
    assert not s.mode_evidence and not s._capture_calibration_error()


def test_strong_contrary_votes_do_not_keep_the_old_match(monkeypatch):
    s,sig=session(monkeypatch);s.mode_evidence=evidence('near',sig);s.mode_notice='Camera match: Small part.'
    s.mode_detector.result=dict(mode=None,state='uncertain',votes=dict(near=2,far=3),frames=[])
    s._check_camera_mode(frame(1));assert not s.mode_evidence.get('mode') and not s._capture_calibration_error()
