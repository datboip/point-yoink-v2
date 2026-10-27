"""New-side lighting policy and gain retention using temporary projects without connecting a scanner."""
import os,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
import pytest
from app2.scan.session import ScanSession
from app2.scan.light_timer import LightTimer
from app2.scan.assist import AutoGain

@pytest.mark.parametrize('auto',[False,True])
@pytest.mark.parametrize('save_color',[False,True])
@pytest.mark.parametrize('app_light',[False,True])
def test_new_side_keeps_gain_and_applies_light_policy_before_preparing(auto,save_color,app_light,tmp_path):
    from app2.scan.project import Project
    from app2.scan.scan_mode import read
    import hashlib
    params=tmp_path/'params';params.mkdir();calibration=b'fixture calibration identity'
    (params/'Pl.bin').write_bytes(calibration)
    s=ScanSession.__new__(ScanSession);calls=[]
    s.cfg=NS(param_dir=str(params),w=800,h=600,motion_calibration=None)
    s.cal=(1000.,1000.,400.,300.);s.near=100;s.far=450;s.scan_mode='near'
    s.project=Project(str(tmp_path/'project'),str(params))
    def prepare(side):
        saved=read(side.folder)
        assert saved==side.scan_profile
        assert saved['mode']=='near' and (saved['near'],saved['far'])==(100,450)
        assert saved['calibration_sha256']==hashlib.sha256(calibration).hexdigest()
        assert saved['intrinsics']==list(s.cal) and saved['depth_size']==[800,600]
        assert saved['sensor_mode_verified'] is False
        calls.append(('prepare',s.gain,s.led))
    s.recording=s.busy=s.building=False
    s.bed=NS(learning=False,side_opened=lambda *a:None)
    s.picked=object();s._save=lambda side:calls.append('save')
    s.source=NS(has_led=True,device=NS(led=lambda on:calls.append(('led',on)),gain=lambda value:calls.append(('gain',value))))
    s.host=NS(release=lambda:None,new=prepare)
    s._home=lambda:None
    s._emit_cloud=lambda *a,**kw:None;s._emit_image=lambda *a:None
    s._emit=lambda *a:None;s._emit_state=lambda:None;s._no=lambda side:1
    s._busy=lambda note,fn:fn()
    s.save_color=save_color
    s.led=True;s.gain=9;s.auto_gain=auto;s.auto_range=True;s.capture_mode='single'
    s.autogain=AutoGain();s.autogain.turn={'old':True}
    s.light_timer=LightTimer();s.light_timer.start(0);s._light_seen=12;s._led_auto=app_light
    s._new_side()
    off=app_light and not save_color          # only a light the app turned on for color goes off; the user's own light stays
    assert calls==(['save',('led',False)] if off else ['save'])+[('gain',9),('prepare',9,not off)]
    assert s.gain==9 and s.led is (not off)
    assert s.light_timer.ask_at==(None if off else 300)
    assert s.autogain.turn is None and s.auto_gain is auto
    assert s.auto_range and s.capture_mode=='single'

@pytest.mark.parametrize('blocked',['recording','busy','building','learning'])
def test_rejected_new_side_does_not_reset_settings(blocked):
    s=ScanSession.__new__(ScanSession)
    s.recording=s.busy=s.building=False;s.bed=NS(learning=blocked=='learning')
    if blocked!='learning':setattr(s,blocked,True)
    s.gain=9;s.led=True
    s._new_side()
    assert s.gain==9 and s.led
