"""Explicit Far calibration and safe identity checks for incoming projects."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import struct
import numpy as np
import pytest


def parameters(root, focal, guid):
    folder=root/guid/'param';folder.mkdir(parents=True)
    (folder/'Pl.bin').write_bytes(struct.pack('<HH9f',800,600,focal,0,400,0,focal,300,0,0,1))
    (folder/'Prgb.bin').write_bytes(b'fixture RGB identity')
    (folder.parent/'property.rvproj').write_text(json.dumps(dict(guid=guid,scan_param={'scanner_sn':'fixture scanner'})))
    return folder


def bare_session(tmp_path, far=True):
    from app2.scan.session import ScanSession
    near=parameters(tmp_path,1000.,'near-side');far_path=parameters(tmp_path,700.,'far-side')
    s=ScanSession.__new__(ScanSession)
    s.cfg=NS(param_dir=str(near),near_param=str(near),far_param=str(far_path) if far else None,w=800,h=600)
    s.cal=(1000.,1000.,400.,300.);s.scan_mode='near';s.calibration_source={}
    s.recording=s.busy=s.building=False;s.bed=NS(learning=False);s.picked=None;s.project=None;s.host=None
    s.near=100;s.far=450;s._near_range=(100,450);s._scan_depth=None
    s.motion_log=None;s.composer=None;s.messages=[];s._log=s.messages.append
    s._emit=lambda *args:None;s._emit_state=lambda **kw:None;s._emit_sides=lambda:None
    return s,near,far_path


def test_far_selects_the_explicit_bundle_before_the_tracker(tmp_path):
    s,near,far=bare_session(tmp_path)
    s._set_scan_mode('far')
    assert s.cal[:2]==(700.,700.) and Path(s.cfg.param_dir)==far
    assert s.calibration_source['declared_mode']=='far'
    assert s.calibration_source['source_side']=='far-side'
    assert s.calibration_source['hardware_mode_verified'] is False
    s._set_scan_mode('near')
    assert s.cal[:2]==(1000.,1000.) and Path(s.cfg.param_dir)==near


def test_missing_far_calibration_blocks_start_without_recording_a_side(tmp_path):
    s,_,_=bare_session(tmp_path,False)
    s._set_scan_mode('far')
    assert 'Large part camera' in s.calibration_error
    made=[];s._new_side=lambda:made.append(True)
    s._go()
    assert not made and not s.recording


def test_bundle_is_saved_with_source_and_resolution(tmp_path):
    from app2.scan.calibration import load,copy_parameters
    folder=parameters(tmp_path,700.,'export-side')
    selected=load(folder,'far',(400,300));target=tmp_path/'side'/'param'
    copy_parameters(selected,target)
    assert (target/'Pl.bin').read_bytes()==(folder/'Pl.bin').read_bytes()
    assert (target/'Prgb.bin').read_bytes()==(folder/'Prgb.bin').read_bytes()
    restored=load(target,'far',(400,300))
    assert restored['intrinsics']==(350.,350.,200.,150.)
    assert restored['source']['source_side']=='export-side'
    assert restored['source']['calibration_size']==[800,600]


def test_malformed_and_changed_mode_provenance_are_refused(tmp_path):
    from app2.scan.calibration import load,CalibrationUnavailable
    folder=parameters(tmp_path,700.,'export-side')
    selected=load(folder,'far')
    (folder/'calibration-source.json').write_text(json.dumps(dict(selected['source'],declared_mode='near')))
    with pytest.raises(CalibrationUnavailable):load(folder,'far')
    (folder/'Pl.bin').write_bytes(b'bad')
    with pytest.raises(CalibrationUnavailable):load(folder,'far')


def incoming(root,guid,side='scan-side'):
    root.mkdir();(root/'data'/side).mkdir(parents=True)
    (root/(root.name+'.revo')).write_text(json.dumps(dict(guid=guid,nodes=[dict(type=2,guid=side)])))
    (root/'data'/side/'property.rvproj').write_text(json.dumps(dict(guid=side)))
    return root


def test_same_name_different_project_gets_a_numbered_copy(tmp_path):
    from app2.transfer.incoming import received_target,install_received
    existing=incoming(tmp_path/'Project','original-guid')
    (existing/'data'/'scan-side'/'preview.png').write_bytes(b'original')
    source=incoming(tmp_path/'arrival','other-guid')
    (source/'data'/'scan-side'/'preview.png').write_bytes(b'new')
    target=received_target(source,existing)
    assert target.name=='Project (1)'
    install_received(source,target)
    assert (existing/'data'/'scan-side'/'preview.png').read_bytes()==b'original'
    assert (target/'data'/'scan-side'/'preview.png').read_bytes()==b'new'


def test_proven_reshare_preserves_pc_edits(tmp_path):
    from app2.transfer.incoming import received_target,install_received
    existing=incoming(tmp_path/'Project','same-guid');(existing/'stages').mkdir()
    (existing/'stages'/'recipe.json').write_text('PC edits')
    source=incoming(tmp_path/'arrival','same-guid');(source/'data'/'scan-side'/'result.ply').write_bytes(b'finished')
    assert received_target(source,existing)==existing
    assert install_received(source,existing)
    assert (existing/'stages'/'recipe.json').read_text()=='PC edits'
    assert (existing/'data'/'scan-side'/'result.ply').read_bytes()==b'finished'


@pytest.mark.parametrize('guid,side',[(None,'scan-side'),('same-guid','different-side')])
def test_uncertain_project_identity_does_not_overwrite(tmp_path,guid,side):
    from app2.transfer.incoming import received_target,install_received
    existing=incoming(tmp_path/'Project','same-guid');source=incoming(tmp_path/'arrival',guid,side)
    assert received_target(source,existing)!=existing
    with pytest.raises(ValueError):install_received(source,existing)


def test_far_warning_disables_capture_even_for_shots():
    from app2.scan.ui_model import ScanState,buttons
    state=ScanState(scan_mode='far',calibration_error='Far calibration is unavailable.',capture_mode='single',
                    recording=False,busy=False,building=False,learning=False,picked=None,n_sides=0)
    assert buttons(state)['go'][2] is False


def test_copy_rebuild_refuses_an_existing_output_before_modifying_it(tmp_path):
    from app2.scan.recalibrate import rebuild_copy
    source=tmp_path/'source';source.mkdir();target=tmp_path/'copy';target.mkdir()
    (target/'keep.txt').write_text('keep')
    with pytest.raises(ValueError):rebuild_copy(source,1,str(tmp_path/'none'),target)
    assert (target/'keep.txt').read_text()=='keep'


def test_color_images_and_timing_survive_retracking(tmp_path):
    from app2.scan import fmt
    from app2.scan.recalibrate import restore_saved_color
    source=tmp_path/'original';target=tmp_path/'retracked'
    (source/'cache').mkdir(parents=True);(target/'cache').mkdir(parents=True)
    image=b'recorded image bytes';(source/'cache'/'frame_002_0042.img').write_bytes(image)
    row=dict(frame='frame_000_0000',t=2.,ok=True)
    fmt.write_side_report(str(target/'cache'),[(0,0,np.eye(4))],[row])
    original=dict(color_saved=True,color_t=1.99,color_offset_ms=-10.,color_sync='recorded host clock')
    plan=[(1,str(source),'frame_002_0042',original)]
    assert restore_saved_color(source,target,plan)==1
    assert (target/'cache'/'frame_000_0000.img').read_bytes()==image
    restored=fmt.read_side(str(target))[1][0]
    assert restored['color_saved'] is True and restored['color_t']==1.99 and restored['color_offset_ms']==-10.


def test_far_controls_dispatch_and_rebuild_only_a_new_copy(tmp_path,monkeypatch):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    from app2.qt import scan_page
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    app=QApplication.instance() or QApplication([])
    project=tmp_path/'Project';project.mkdir();parent=tmp_path/'copies';parent.mkdir()
    (parent/'Project_LargePart').mkdir()
    page=scan_page.ScanPage(SessionBridge(),False);calls=[];started=[]
    class OwnedProcess(QProcess):
        def start(self,program,args):started.append((program,args,self.processEnvironment()))
        def terminate(self):calls.append('stop owned copy')
    monkeypatch.setattr(scan_page,'QProcess',OwnedProcess)
    monkeypatch.setattr(scan_page.QFileDialog,'getExistingDirectory',lambda *args:str(tmp_path/'far-param'))
    page.session=NS(set_far_calibration=lambda folder:calls.append(folder))
    try:
        s=state(str(project));page.on_state(s)
        assert page.controls.calibration_info.isHidden() and page.controls.choose_far.text()=="Choose calibration files..."
        s.scan_mode='far';s.calibration_error='Far calibration is unavailable. Choose a Far scanner export.'
        page.on_state(s)
        assert page.controls.calibration_info.text()==s.calibration_error
        page.controls.choose_far.click();assert calls==[str(tmp_path/'far-param')]
        assert not page.controls.rebuild_far.isEnabled()
        s.picked=1;s.picked_has_frames=True;s.far_calibration_path=str(tmp_path/'far-param')
        s.calibration_error=None;s.calibration_source=dict(declared_mode='far',source_side='far-export')
        page.on_state(s)
        assert 'far-export' in page.controls.calibration_info.text() and page.controls.rebuild_far.isEnabled()
        monkeypatch.setattr(scan_page.QFileDialog,'getExistingDirectory',lambda *args:str(parent))
        page.rebuild_far_copy();assert len(started)==1
        _program,args,env=started[0]
        assert args[:3]==['-m','app2.scan.recalibrate',str(project)]
        assert args[args.index('--output')+1]==str(parent/'Project_LargePart (1)')
        assert env.value('POINTYOINK_NO_DEVICE')=='1' and not env.contains('POINTYOINK_DRIVER')
        assert not (parent/'Project_LargePart (1)').exists()
        page.rebuild_far_copy();assert calls[-1]=='stop owned copy'
        process=page._far_process;process.finished.emit(0,QProcess.ExitStatus.NormalExit)
        assert page._far_process is None and 'Project_LargePart (1)' in page.controls.far_copy_note.text()
    finally:page.close();page.deleteLater();app.processEvents()


def test_calibration_text_growth_keeps_camera_size_and_reflows_settings(tmp_path):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    app=QApplication.instance() or QApplication([]);page=ScanPage(SessionBridge(),False)
    try:
        page.resize(1280,640);page.sides.tabs.setCurrentIndex(1);page.show()
        s=state(str(tmp_path));page.on_state(s);page.relayout(1280,640);app.processEvents()
        old=page.camera_scroll.height()
        s.calibration_error='Far calibration is unavailable. Choose a calibration folder from a Far scanner export before starting a new side.'
        page.on_state(s);app.processEvents()
        assert page.camera_scroll.height()==old
        page.controls.calibration_setup.setChecked(True);app.processEvents()
        from PySide6.QtCore import QPoint
        popup=page.controls.calibration_tools
        center=page.controls.choose_far.mapTo(popup,page.controls.choose_far.rect().center())
        assert popup.isVisible() and popup.rect().contains(center)
        assert page.camera_scroll.height()==old
        assert page.controls.calibration_info.isHidden()
        assert s.calibration_error in page.controls.calibration_setup.toolTip()
    finally:page.close();page.deleteLater();app.processEvents()


def test_missing_far_calibration_blocks_board_learning(tmp_path):
    s,_,_=bare_session(tmp_path,False);s.gain=4;called=[]
    s.bed.learn_toggle=lambda *args:called.append(args)
    s._set_scan_mode('far');s._learn('fixture board')
    assert not called and 'Large part camera' in s.messages[-1]


def test_missing_far_disables_learn_but_keeps_stop_learning():
    from app2.scan.ui_model import ScanState,buttons
    state=ScanState(scan_mode='far',calibration_error='Far calibration is unavailable.',
                    bed_available=True,n_sides=0,learning=False,recording=False,busy=False,building=False)
    assert buttons(state)['learn'][2] is False
    state.learning=True
    assert buttons(state)['learn'][2] is True and buttons(state)['go'][2] is True


def test_legacy_side_cannot_inherit_far_without_far_provenance(tmp_path,monkeypatch):
    from app2.scan import preview
    s,near,_far=bare_session(tmp_path)
    s._set_scan_mode('far')
    side=NS(folder=str(near.parent),scan_profile={},model_thumb=None)
    s.project=NS(sides=[side]);s.host=NS(cal=s.cal)
    s.composer=NS(cal=s.cal,area=None,scan_geometry=None)
    for name in ('_forget_rewound','_home','_emit_cloud','_emit_image','_show_saved','_sync_scan_geometry'):
        setattr(s,name,lambda *args,**kwargs:None)
    monkeypatch.setattr(preview,'set_rgb_calib',lambda folder:None)
    s._pick(side)
    assert s.scan_mode=='far' and s.calibration_error
    assert 'copy' in s.calibration_error.lower()
    s._go();assert not s.recording
