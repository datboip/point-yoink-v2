"""Precision, capture limits, light leases and real joined fin geometry."""
import os,tempfile,json
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ['QT_QPA_PLATFORM']='offscreen'
import app2
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt.tool_panel import SliderRow
from app2.scan.light_timer import LightTimer


def test_numeric_value_units_and_fixed_width():
    app=QApplication.instance() or QApplication([])
    c=dict(label='Fill holes',lo=0,hi=2000,step=50,default=400)
    row=SliderRow(c); before=row.value_l.size()
    row.set_numeric_units(.1,' mm',2);row.value_l.setValue(6.4)
    assert row.value()==64
    row.value_l.stepUp(); assert row.value()==114
    row.track.changed.emit([8]); assert row.value()==400 and row.value_l.value()==40
    row.set_numeric_units(1,' edges',0)
    assert row.value_l.size()==before and row.value_l.value()==400


def test_light_lease_and_repeated_extensions():
    t=LightTimer();t.start(10)
    assert not t.tick(309) and t.remaining(309) is None
    assert not t.tick(310) and t.remaining(310)==120
    assert t.extend(320);assert not t.tick(619) and t.remaining(619) is None
    assert not t.tick(620) and t.extend(630)
    assert not t.tick(930) and t.remaining(930)==120
    assert t.tick(1050) and not t.extend(1050)
    t.stop();assert t.remaining(1100) is None and not t.tick(1500)


def _shot_session(mode='single'):
    from app2.scan.session import ScanSession
    s=ScanSession.__new__(ScanSession);s.capture_mode=mode;s.recording=True;s.shot_deadline=8;s.picked=object()
    s.shot_need=3;s.shot_got=0;s.shot_presses={};s.calls=[]
    def pause(sd,why): s.calls.append(why);s.recording=False
    s._pause=pause
    return s


def test_single_shot_takes_its_frames_and_times_out(monkeypatch):
    monkeypatch.setattr('app2.scan.session.time.monotonic',lambda:2)
    s=_shot_session()
    s._check_shot(True);s._check_shot(True);assert not s.calls
    s._check_shot(True);assert len(s.calls)==1 and 'Saved 3' in s.calls[0]
    s.recording=True;s.shot_got=1;s.shot_deadline=1;s._check_shot(False)
    assert len(s.calls)==2 and 'Only 1 of 3' in s.calls[-1]
    s.recording=True;s.shot_got=0;s.shot_deadline=1;s._check_shot(False);assert 'No tracked frame' in s.calls[-1]
    s.recording=True;s.capture_mode='continuous';s._check_shot(True);assert len(s.calls)==3


def test_single_shot_ramps_and_burst_is_fixed():
    s=_shot_session(); sd=object()
    assert [s._shot_count(sd) for _ in range(5)]==[10,15,20,25,25]
    assert s._shot_count(object())==10                     # another side starts over
    s.capture_mode='burst'; assert s._shot_count(sd)==25


@pytest.mark.parametrize('shape',['triangle','rounded'])
def test_fin_is_closed_joined_and_source_unchanged(shape):
    import trimesh,numpy as np
    from stages.tools.fin import join_fin
    mesh=trimesh.creation.box(extents=[20,20,4]);before=mesh.vertices.copy()
    p=dict(anchors=json.dumps([[[-6,0,2],[0,0,1]],[[6,0,2],[0,0,1]]]),thickness=2,reach=8,shape=shape)
    V,F=join_fin(mesh.vertices,mesh.faces,p);result=trimesh.Trimesh(V,F)
    assert result.is_volume and len(result.split())==1 and result.volume>mesh.volume
    assert np.array_equal(mesh.vertices,before)
    with pytest.raises(ValueError,match='closed mesh'): join_fin(mesh.vertices,mesh.faces[:-1],p)


def test_fin_recipe_undo_and_redo(tmp_path):
    from stages.project import Project
    from stages.engine import Engine
    (tmp_path/"project"/"data").mkdir(parents=True)
    e=Engine(Project.open(str(tmp_path/'project'),create=True))
    anchors=json.dumps([[[-6,0,2],[0,0,1]],[[6,0,2],[0,0,1]]])
    e.add_op('meshed',{'tool':'fin','params':{'anchors':anchors,'thickness':2.,'reach':8.,'shape':'triangle'}})
    assert e.chain_ops('meshed')[-1][2]['params']['anchors']==anchors
    e.undo();assert not e.chain_ops('meshed')
    e.redo();assert e.chain_ops('meshed')[-1][2]['tool']=='fin'


def test_scan_progress_visible_for_build_started_elsewhere():
    from app2.qt.bridge import SessionBridge
    from app2.qt.scan_page import ScanPage
    from app2.scan.ui_model import ScanState
    from app2.builder.runner import BuildProgress
    app=QApplication.instance() or QApplication([])
    page=ScanPage(SessionBridge(),gl=False);page.resize(1280,748)
    page.state=ScanState(building=True)
    prog=BuildProgress();prog.text='Building: Meshing';prog.frac=.5
    page.on_build(prog)
    assert not page.build_card.isHidden() and page.build_card.title.text()
    page.hide();page._show_build_card();assert not page.build_card.isHidden()
    prog.finished=True;page.on_build(prog);assert page.build_card.isHidden()
    page.deleteLater();app.processEvents()
