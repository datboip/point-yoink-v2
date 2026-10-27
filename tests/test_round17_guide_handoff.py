import json
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication,QLabel

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def sample():
    return dict(spots=27,poses=[dict(rank=1,text='Tilt down about 30 degrees, one turn: fills about 70 percent',color='#50cfed')],needs_flip=[0],reason='')


def test_report_panel_uses_exact_shared_spot_colors(app):
    from app2.qt.report_panel import ReportPanel
    p=ReportPanel();p.set_guide(sample())
    labels=[q.text() for q in p.legend]
    assert '#ff4632' in labels[0] and 'Not scanned yet (holes)' in labels[0]
    assert '#b45ae6' in labels[1] and 'Rough or doubled, scan again' in labels[1]
    assert '#46d778' in labels[2] and 'Done' in labels[2]
    assert not any('Getting there' in q.text() for q in p.findChildren(QLabel))
    p.close()


def test_report_summary_and_positions_use_same_guide_after_refresh(app):
    from app2.qt.report_panel import ReportPanel
    p=ReportPanel();p.set_guide(sample())
    p.set_report(dict(open_edge_mm=100,runs=2,pinholes=3,places=[]),8)
    assert p.head.text()=='27 missing or rough spots'
    assert 'Hole details: 100 mm' in p.hole_metrics.text()
    assert p.places.itemAt(0).widget().text()=='1. '+sample()['poses'][0]['text']
    p.close()


def test_guide_panel_reason_is_visible_without_positions(app):
    from app2.qt.report_panel import ReportPanel
    p=ReportPanel();p.set_guide(dict(sample(),poses=[],reason='Saved bed plane is needed to suggest positions'))
    assert 'Saved bed plane' in p.sub.text() and '27' in p.head.text()
    p.close()


def test_report_settings_do_not_claim_to_change_missing_map(app):
    from app2.qt.report_panel import ReportPanel
    p=ReportPanel()
    assert [c.c['path'] for c in p.settings.controls]==['report.min_mm']
    assert any('only the hole details' in q.text() for q in p.findChildren(QLabel))
    p.close()


def test_project_mode_survives_reopen_and_is_project_local(tmp_path):
    from app2.scan.guide_mode import selected_mode,select_mode
    assert selected_mode(tmp_path)=='turntable'
    select_mode(tmp_path,'handheld');assert selected_mode(tmp_path)=='handheld'
    assert selected_mode(tmp_path/'other')=='turntable'
    (tmp_path/'build'/'scan-guide-mode.json').write_text('{broken')
    assert selected_mode(tmp_path)=='turntable'
    with pytest.raises(ValueError):select_mode(tmp_path,'spinny')


def test_build_choice_persists_handoff_and_report_reopen(app,tmp_path,monkeypatch):
    from app2.scan.guide_mode import selected_mode
    from app2.qt.bridge import SessionBridge
    from app2.qt.build_step import BuildStepPage
    p=BuildStepPage(SessionBridge(),gl=False);p.host_root=str(tmp_path)
    monkeypatch.setattr(p,'_show_scan_guide',lambda mode:None)
    p.pick_colour('scan_guide_handheld');assert selected_mode(tmp_path)=='handheld'
    p.show_stage('report');assert p.colour_mode['report']=='scan_guide_handheld'
    p.pick_colour('scan_guide');assert selected_mode(tmp_path)=='turntable'
    p.shutdown();p.close()


def test_touchup_reload_uses_project_mode_and_current_range(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app2.scan.guide_mode import select_mode
    from app2.scan.session import ScanSession
    from app2.scan import touchup
    from app2.builder import scan_guide
    select_mode(tmp_path,'handheld')
    model=tmp_path/'model.ply';model.write_bytes(b'fixture');st=model.stat()
    sd=SimpleNamespace(folder=str(tmp_path),touch_info=dict(model=dict(mtime=st.st_mtime,size=st.st_size),ref={}))
    tm=SimpleNamespace()
    monkeypatch.setattr(touchup.TouchModel,'load',lambda *_:tm)
    monkeypatch.setattr(touchup,'start_poses',lambda *_:[])
    monkeypatch.setattr(touchup,'platter_priors',lambda *_:[])
    calls=[]
    def guide(project,src,mode='turntable',distance=None,frames=None,live=False):
        assert live
        calls.append((mode,distance));return dict(mode=mode)
    monkeypatch.setattr(scan_guide,'guide',guide)
    session=ScanSession.__new__(ScanSession);session.project=SimpleNamespace(path=str(tmp_path))
    session._log=lambda *_:None;session.near=120;session.far=280
    session.cfg=SimpleNamespace(touch_factory=lambda *_:'factory')
    assert session._load_touch(sd)=='factory'
    assert calls==[('handheld',(120,280))] and sd.touch_model.scan_guide['mode']=='handheld'


def test_new_touchup_uses_persisted_handheld_mode(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app2.scan.guide_mode import select_mode
    from app2.scan.session import ScanSession
    from app2.scan import touchup
    from app2.builder import scan_guide
    import model_frames
    select_mode(tmp_path,'handheld');(tmp_path/'model.ply').write_bytes(b'fixture')
    sd=SimpleNamespace(folder=str(tmp_path))
    tm=SimpleNamespace(counts=lambda:(0,27))
    monkeypatch.setattr(touchup.TouchModel,'load',lambda *_:tm)
    monkeypatch.setattr(touchup,'start_poses',lambda *_:[])
    monkeypatch.setattr(touchup,'platter_priors',lambda *_:[])
    monkeypatch.setattr(model_frames,'read',lambda *_:{'side':[]})
    monkeypatch.setattr(model_frames,'write_touch',lambda *_:None)
    monkeypatch.setattr(model_frames,'touch_info',lambda *_:{})
    calls=[]
    monkeypatch.setattr(scan_guide,'guide',lambda project,src,mode='turntable',distance=None,frames=None,live=False:(calls.append((mode,live)) or dict(mode=mode)))
    s=ScanSession.__new__(ScanSession);s.project=SimpleNamespace(path=str(tmp_path),sides=[],make_side=lambda:sd)
    s.recording=False;s.busy=False;s.building=False;s.picked=None;s.near=120;s.far=280
    s.bed=SimpleNamespace(learning=False,side_opened=lambda *_:None)
    s.host=SimpleNamespace(release=lambda:None,new=lambda *_args,**_kwargs:None)
    s.cfg=SimpleNamespace(touch_factory=lambda *_:'factory')
    s._log=lambda *_:None;s._forget_rewound=lambda:None;s._home=lambda:None;s._no=lambda _:1
    s._busy=lambda _,work:work();s._emit_image=lambda *_:None;s._emit_touch=lambda *_:None
    s._touch_home=lambda _:None;s._emit_sides=lambda:None;s._emit_state=lambda:None
    events=[];s._emit=lambda kind,value:events.append((kind,value))
    s._touch_up()
    assert [value for kind,value in events if kind=="turn"][-1]["coverage"]["frames"]==0
    assert calls==[('handheld',True)] and s.picked.touch_model.scan_guide['mode']=='handheld'
