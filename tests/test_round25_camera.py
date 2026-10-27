from types import SimpleNamespace as NS
import numpy as np
from app2.scan.session import ScanSession
from app2.scan.mode_detection import changed


def session(monkeypatch,mode='near',recording=False):
    s=object.__new__(ScanSession);s.scan_mode=mode;s.recording=recording;s.busy=None;s.building=False;s.picked=None;s.bed=NS(learning=False)
    s.mode_evidence={};s._mode_has_result=True;s._mode_checked_at=0;s._mode_signature=dict(distance=330.,span=30.,radius=6.,radius_mad=.2);s._mode_background=False
    s.source=NS(colour=lambda:None);s.events=[];s._emit_state=lambda **k:s.events.append('state');s._log=lambda x:s.events.append(x)
    s._pause=lambda side,why:(s.events.append('paused'),setattr(s,'recording',False));s._set_scan_mode=lambda x:s.events.append('switched '+x)
    sig=dict(s._mode_signature);monkeypatch.setattr('app2.scan.mode_detection.signature',lambda *a:dict(sig))
    class Check:
        evidence=dict(state='matched',mode=mode);future=None;result=None
        def poll(self):r=self.result;self.result=None;return r
        def begin(self):self.evidence=dict(state='checking',mode=None);s.events.append('recheck')
        def offer(self,*a,**kw):s.events.append('offered')
    s.mode_detector=Check();return s,sig


def frame(t):return NS(t=t,depth=np.full((20,20),3300,np.uint16),irl=np.zeros((20,20),np.uint8))


def evidence(mode,sig):return dict(mode=mode,state='matched',frames=[{'_signature':dict(sig),mode:dict(inliers=12,rms_mm=.2)}])


def test_idle_rechecks_in_background_without_blocking_confirmed_start(monkeypatch):
    s,sig=session(monkeypatch);s.calibration_error='';s._check_camera_mode(frame(4))
    assert 'recheck' in s.events and 'offered' in s.events and s._mode_background
    assert not s._capture_calibration_error()


def test_large_scene_change_blocks_start_until_new_evidence(monkeypatch):
    s,sig=session(monkeypatch);sig['distance']=660.;s._check_camera_mode(frame(1))
    assert 'recheck' in s.events and not s._mode_background
    assert not s._capture_calibration_error()


def test_stale_async_result_cannot_switch_to_previous_scene(monkeypatch):
    s,sig=session(monkeypatch);s._mode_has_result=False;s.mode_detector.result=evidence('far',sig);sig['distance']=660
    s._check_camera_mode(frame(1));assert not any(x.startswith('switched') for x in s.events)
    assert not s.mode_evidence and 'recheck' in s.events


def test_contrary_capture_result_pauses_and_never_auto_switches(monkeypatch):
    s,sig=session(monkeypatch,recording=True);s.mode_detector.result=evidence('far',sig)
    s._check_camera_mode(frame(1));assert 'paused' in s.events and not s.recording
    assert not any(x.startswith('switched') for x in s.events)
    assert 'Large part' in s._capture_calibration_error()
    assert any('Evidence:' in x for x in s.events)


def test_one_click_fix_switches_without_starting_capture(monkeypatch):
    s,sig=session(monkeypatch,mode='far');s.mode_evidence=evidence('near',sig)
    s._switch_camera_match();assert 'switched near' in s.events and not s.recording
    s.events.clear();s.recording=True;s._switch_camera_match();assert s.events==[]


def test_signature_uses_measured_scene_spread_and_dot_uncertainty():
    p=dict(distance=330,span=40,radius=6,radius_mad=.2)
    assert not changed(p,dict(p,distance=350,radius=6.1))
    assert changed(p,dict(p,distance=660)) and changed(p,dict(p,radius=9))


def test_uncertain_initial_or_changed_scene_keeps_capture_blocked(monkeypatch):
    s,sig=session(monkeypatch);s.mode_detector.result=dict(mode=None,state='uncertain',frames=[])
    s._check_camera_mode(frame(1));assert not s._capture_calibration_error() and 'Learn bed' in s.mode_notice


def test_failed_background_check_preserves_prior_match_and_retries(monkeypatch):
    s,sig=session(monkeypatch);s.calibration_error='';s.mode_evidence=evidence('near',sig)
    s._mode_background=True;s.mode_detector.evidence=dict(state='checking');s.mode_detector.result=dict(mode=None,state='unavailable',frames=[])
    s._check_camera_mode(frame(1));assert s.mode_evidence['mode']=='near' and not s._capture_calibration_error()
    s.mode_detector.evidence=dict(state='unavailable');s.events.clear();s._check_camera_mode(frame(5))
    assert 'recheck' in s.events


def test_first_contrary_result_during_capture_cannot_auto_switch_after_pause(monkeypatch):
    s,sig=session(monkeypatch,recording=True);s._mode_has_result=False;s.mode_detector.result=evidence('far',sig)
    s._check_camera_mode(frame(1));assert 'paused' in s.events and not any(x.startswith('switched') for x in s.events)
