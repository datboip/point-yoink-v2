"""Live editing, saved scan controls, import details, and private driver regressions."""
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit
from PySide6.QtGui import QShortcut, QKeySequence
from app2.qt import build_step as B

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])

@pytest.fixture(autouse=True)
def private_config(monkeypatch, tmp_path):
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path/'config.json'))

def test_unbuilt_edit_stage_does_not_use_published_model():
    p=NS(published='/old-model.ply', stage='meshed', _all_mode=lambda:False,
         _platter_preview=lambda:False, _view_path=lambda:None, flush_pending_edits=lambda:False)
    from app2.qt.edit_page import EditPage
    calls=[]
    p.running=lambda:False
    p.show_stage=lambda key,**kw:calls.append((key,kw))
    e=NS(build=p,sync=lambda:None)
    EditPage.pick_stage(e,'meshed')
    assert calls[0][1].get('keep_published') is False

def test_saved_gain_and_range_survive_session_restart(tmp_path):
    from app2.scan.session import ScanConfig, ScanSession
    cfg=ScanConfig(str(tmp_path),str(tmp_path),gain=9,near=180,far=320)
    geometry=[]
    p=ScanSession.__new__(ScanSession)
    p.gain,p.near,p.far=cfg.gain,cfg.near,cfg.far
    p.scan_mode='near';p._emit_state=lambda:None;p._log=lambda m:None
    p.composer=NS(set_scan_geometry=lambda *values:geometry.append(values))
    ScanSession._set(p,gain=11,near=190,far=330)
    assert geometry==[(190,330,'near')]
    next_cfg=ScanConfig(str(tmp_path),str(tmp_path))
    assert (next_cfg.gain,next_cfg.near,next_cfg.far)==(11,190,330)

def test_first_scan_enables_auto_gain(tmp_path):
    from app2.scan.session import ScanConfig
    assert ScanConfig(str(tmp_path),str(tmp_path)).auto_gain

def test_scan_button_is_findable_by_private_driver(app,tmp_path):
    from app2.qt.widgets import BigButton
    from app2.diagnostics.driver import Driver
    w=QWidget();layout=QVBoxLayout(w);b=BigButton();b.set_action('Start scan','primary');layout.addWidget(b)
    w.show();app.processEvents();d=Driver(w,str(tmp_path/'own.sock'))
    try:
        assert d.cmd_tree({'contains':'Start scan'})['count']==1
        hits=[];b.clicked.connect(lambda:hits.append(True));d.cmd_click({'text':'Start scan','exact':True})
        assert hits==[True]
    finally:d.server.close();w.close()

def test_driver_space_reaches_window_shortcut(app,tmp_path):
    from app2.diagnostics.driver import Driver
    w=QWidget();layout=QVBoxLayout(w);entry=QLineEdit();layout.addWidget(entry)
    hits=[];shortcut=QShortcut(QKeySequence('Space'),w);shortcut.activated.connect(lambda:hits.append(True))
    w.show();w.activateWindow();entry.setFocus();app.processEvents();d=Driver(w,str(tmp_path/'key.sock'))
    try:
        d.cmd_key({'keys':'Space'})
        assert hits==[True] and entry.text()==''
    finally:d.server.close();w.close()

def test_import_scan_details_include_settings_and_frames(tmp_path):
    from app2.qt.device_projects import scan_details
    side=tmp_path/'10062026011852';side.mkdir();(side/'cache').mkdir()
    (side/'cache'/'0.dph').write_bytes(b'')
    (side/'property.rvproj').write_text(json.dumps({'scan_param':{'accuracy_type':1,'scan_object':4,'color_type':1,'depth_width':640,'depth_height':400}}))
    line=scan_details(side)
    assert all(x in line for x in ['High','Dark','Color on','640','400','1 frame'])


def test_first_gain_is_chosen_from_ir_brightness():
    from app2.scan.assist import AutoGain
    assert AutoGain.initial(np.full((20,20),16,np.uint8),1)==8
    assert AutoGain.initial(np.full((20,20),220,np.uint8),1)==1
    assert AutoGain.initial(np.zeros((20,20),np.uint8),1) is None

def test_saved_gain_keeps_other_app_config(tmp_path):
    from app2.scan.settings import save,load,path
    path().write_text(json.dumps({'other':'kept'}))
    save(gain=12,near=175,far=390)
    assert load()['gain']==12
    assert json.loads(path().read_text())['other']=='kept'

def test_meshed_summary_explains_alias():
    snap=NS(meshed_method='as_fused',fused_kind='mesh',chains={})
    assert 'Same as Fused (As scanned). Pick Watertight for a closed surface.' in B.BuildStepPage._stage_sub(NS(snap=snap),'meshed')

def test_wifi_waiting_hides_progress(app,tmp_path):
    from app2.qt.device_projects import DeviceProjects
    p=DeviceProjects(library=str(tmp_path))
    try:
        p.progress.show();p.progress.setRange(0,0)
        p.worker_event(dict(event='listening',code='1234',ip='127.0.0.1',stage=str(tmp_path)))
        assert p.progress.isHidden() and p.status.text()=='Waiting for the scanner'
    finally:p.close()

def test_wifi_file_arrivals_list_projects_scans_current_file(app,tmp_path):
    from app2.qt.device_projects import DeviceProjects
    p=DeviceProjects(library=str(tmp_path));p.mode='wifi';p._t0=0
    try:
        p.worker_event(dict(event='received_file',stage=str(tmp_path),name='Project10062026011852',scan='10062026011852',relpath='Project10062026011852/data/10062026011852/cache/0.dph'))
        assert p.files.topLevelItemCount()==1
        assert p.files.topLevelItem(0).text(2)=='1'
        p.worker_event(dict(event='progress',bytes=1024,total=2048,rate=512,relpath='cache/0.dph'))
        assert all(x in p.status.text() for x in ('cache/0.dph','MB/s','left'))
        assert p.progress.value()==50
    finally:p.close()

def test_quality_job_has_visible_working_and_done_state(app):
    from PySide6.QtWidgets import QLabel,QProgressBar
    p=NS(_shut=False,host=NS(call=lambda *a,**k:23),tickets={},_one_shot_jobs={},job_state=QLabel(),job_progress=QProgressBar(),_refresh_tools=lambda:None,_refresh_progress=lambda:None,_needs_rebuild_first=lambda why:False,notice=lambda *a,**k:None)
    t=B.BuildStepPage.call(p,'h_quality','copy/model.ply')
    assert t in p._one_shot_jobs
    assert 'Measuring scan quality' in p.job_state.text()
    assert not p.job_progress.isHidden()
    B.BuildStepPage.on_result(p,dict(ticket=t,name='h_quality',ok=True,value={}))
    assert p.job_progress.isHidden() and p.job_state.text().startswith('Done:')
    t=B.BuildStepPage.call(p,'h_quality','copy/model.ply')
    B.BuildStepPage.on_result(p,dict(ticket=t,name='h_quality',ok=False,error='missing frames'))
    assert p.job_progress.isHidden() and p.job_state.text().startswith('Failed:')

def test_new_scan_menu_enabled_on_scan_page_while_recording():
    from app2.qt.shell import MainWindow
    q=NS(state=NS(busy=False,building=False,recording=True,learning=False,project=None),_closing=False,
      build=NS(busy=False,busy_call=False,can_export=lambda:False,undo_b=NS(isEnabled=lambda:False),redo_b=NS(isEnabled=lambda:False),vp=NS(sel_delete=NS(isEnabled=lambda:False))),
      imported_models=NS(export_future=None),_prepare_future=None,imported_project=None,stack=NS(currentWidget=lambda:None),build_slot=None,edit=object(),
      header=NS(new=NS(isEnabled=lambda:False),open=NS(isEnabled=lambda:False),folder=NS(isVisible=lambda:False)),_active_cube=lambda:None,_active_view=lambda:None)
    assert MainWindow._menu_states(q)['new']

def test_new_scan_asks_before_stopping_recording():
    from app2.qt.shell import MainWindow
    closed=[];asked=[]
    p=NS(state=NS(busy=False,building=False,recording=True,n_sides=1),imported_project=None,session=NS(close=lambda **k:closed.append(k)),_crashed=False,_closing=False,
         confirm=lambda title,text:(asked.append(text) or False))
    MainWindow.ask_new_project(p)
    assert asked and 'Stop recording' in asked[0] and closed==[]

def test_new_side_keeps_gain(tmp_path):
    from app2.scan.session import ScanSession
    from app2.scan.project import Project
    from app2.scan.scan_mode import read
    import hashlib
    params=tmp_path/'params';params.mkdir();calibration=b'fixture calibration identity'
    (params/'Pl.bin').write_bytes(calibration)
    gains=[];prepared=[]
    p=ScanSession.__new__(ScanSession)
    p.recording=p.busy=p.building=False
    p.bed=NS(learning=False,side_opened=lambda *a:None)
    p._forget_rewound=lambda:None;p._drop_empty_touch_side=lambda side:False;p.picked=None
    p.save_color=True;p.source=NS(device=NS(gain=gains.append));p.gain=11
    p.autogain=NS(reset=lambda:None);p._emit=lambda *a:None;p._emit_state=lambda:None
    p.host=NS(release=lambda:None,new=prepared.append)
    p.project=Project(str(tmp_path/'project'),str(params))
    p.cfg=NS(param_dir=str(params),w=800,h=600,motion_calibration=None)
    p.cal=(1000.,1000.,400.,300.);p.near=100;p.far=450;p.scan_mode='near'
    p._home=lambda:None;p._emit_cloud=lambda *a,**k:None;p._emit_image=lambda *a:None
    p._busy=lambda text,fn:fn();p._no=lambda side:1
    p._new_side()
    assert p.gain==11 and gains==[11] and prepared==[p.picked]
    saved=read(p.picked.folder)
    assert saved==p.picked.scan_profile
    assert saved['mode']=='near' and (saved['near'],saved['far'])==(100,450)
    assert saved['calibration_sha256']==hashlib.sha256(calibration).hexdigest()
    assert saved['intrinsics']==list(p.cal) and saved['depth_size']==[800,600]
    assert saved['sensor_mode_verified'] is False

def test_receiver_passes_real_current_file_and_arrival_events(monkeypatch,tmp_path):
    import sys
    from app2.transfer.projects import receive_wifi
    class Receiver:
        def __init__(self,dest,on_event):
            self.stage=dest;self.on_event=on_event;self._on=True;self.code='1234';self.ip='127.0.0.1'
        def _file(self,h,body):
            dest=tmp_path/h['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
            self.on_event('progress',dict(bytes=len(body),total=100,files=1,rate=4))
        def start(self):
            self._file(dict(path='Project10062026011852/data/10062026011852/cache/0.dph'),b'123')
            self.on_event('done',dict(projects=['Project10062026011852']))
        def stop(self):self._on=False
    monkeypatch.setitem(sys.modules,'wifi',NS(Receiver=Receiver));events=[]
    receive_wifi(tmp_path,notify=lambda kind,**info:events.append((kind,info)))
    arrived=next(info for kind,info in events if kind=='received_file')
    progress=next(info for kind,info in events if kind=='progress')
    assert arrived['scan']=='10062026011852' and arrived['name']=='Project10062026011852'
    assert progress['relpath'].endswith('cache/0.dph')


def test_first_run_explicit_manual_gain_is_respected(tmp_path):
    from app2.scan.session import ScanConfig,ScanSession
    cfg=ScanConfig(str(tmp_path),str(tmp_path),auto_gain=False)
    session=ScanSession(cfg,NS(),lambda *a:None)
    assert not session.auto_gain and not session._initial_auto_gain
    session._emit_state=lambda:None
    session._initial_auto_gain=True
    session._set_auto_gain(False)
    assert not session._initial_auto_gain
    session._set(gain=13)
    assert session.gain==13 and not session._initial_auto_gain


def test_received_project_disclosure_reads_saved_path_with_no_mount(app,tmp_path,monkeypatch):
    import time
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
    from app2.qt import device_projects as D
    stage=tmp_path/'incoming';side=stage/'Project10062026011852'/'data'/'10062026011852'
    side.mkdir(parents=True);(side/'cache').mkdir();(side/'cache'/'0.dph').write_bytes(b'')
    (side/'property.rvproj').write_text(json.dumps({'scan_param':{'accuracy_type':1,'scan_object':4,'color_type':1,'depth_width':640,'depth_height':400}}))
    monkeypatch.setattr(D,'adopt_project',lambda p:{'issues':[]})
    library=tmp_path/'library';library.mkdir();p=D.DeviceProjects(library=str(library));p._t0=time.monotonic()
    try:
        p._place_received(stage);assert p.root is None
        item=p.files.topLevelItem(0)
        assert item.childIndicatorPolicy()==D.QTreeWidgetItem.ShowIndicator
        p._expanded(item)
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
            labels=[label.text() for label in p.files.findChildren(QLabel)]
            if any('High | Dark | Color on' in t for t in labels):break
        assert any('High | Dark | Color on' in t for t in labels)
    finally:p.close()

def test_wifi_unknown_total_never_claims_zero_time_left(app,tmp_path):
    import time
    from app2.qt.device_projects import DeviceProjects
    p=DeviceProjects(library=str(tmp_path));p.mode='wifi';p._t0=time.monotonic()
    try:
        p.worker_event(dict(event='progress',bytes=200,total=200,rate=100))
        assert 'time left unknown' in p.status.text()
        assert 'Known files' in p.progress.format()
    finally:p.close()


def test_new_project_can_discard_the_open_one(tmp_path):
    """New project while sides exist asks Save / Discard / Cancel; Discard closes with the folder to delete, then "new" (2026-10-24)."""
    from app2.qt.shell import MainWindow
    from types import SimpleNamespace as NS
    closed=[]
    p=NS(state=NS(busy=False,building=False,recording=False,n_sides=2,project=str(tmp_path)),imported_project=None,session=NS(close=lambda **k:closed.append(k)),_crashed=False,_closing=False,
         ask_keep_or_discard=lambda title,text:'discard')
    assert MainWindow.ask_new_project(p) is True and closed==[dict(then=('discard',str(tmp_path),'new'))]
    closed.clear(); p.ask_keep_or_discard=lambda title,text:None; p._closing=False
    assert MainWindow.ask_new_project(p) is None and closed==[]
    closed.clear(); p.ask_keep_or_discard=lambda title,text:'keep'; p._closing=False
    MainWindow.ask_new_project(p); assert closed==[dict(then='new')]
