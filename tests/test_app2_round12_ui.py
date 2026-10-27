"""UI regressions from the NewScan editor walk. No device or shared app config."""
import os,tempfile,types,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='round12-ui-test-'))/'config.json')
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget,QPushButton,QVBoxLayout,QLabel
from PySide6.QtCore import QRect,Qt,QPointF
from PySide6.QtGui import QFontMetrics
from app2.qt import build_step as B,build_words as W
from app2.qt.bridge import SessionBridge

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])

def test_mirror_help_wraps_at_default_panel_width(app):
    p=B.BuildStepPage(SessionBridge(),False)
    p.stage='meshed';p.panel['meshed'].show_tab('Mirror');p.resize(1280,800);p.show();app.processEvents()
    label=p.mirror_ui['meshed']['box'].findChildren(QLabel)[0]
    label.resize(356,label.sizeHint().height());app.processEvents()
    assert label.wordWrap() and 'Measured surfaces stay.' in label.text()
    needed=label.fontMetrics().boundingRect(QRect(0,0,label.width(),10000),Qt.TextFlag.TextWordWrap,label.text()).height()
    assert label.height()>=needed
    p.shutdown();p.close()

def progress_probe(app):
    from app2.qt.build_view import ProgressCard
    from app2.builder.runner import BuildProgress
    from app2.builder.stagehost import Snapshot
    p=B.BuildStepPage(SessionBridge(),False)
    p.state=types.SimpleNamespace(building=True);p.run_mine=types.SimpleNamespace(what=None)
    p.prog=BuildProgress();p.prog.t0-=12;p.prog.current='meshed#op-0001';p.prog.plan={'to_run':1,'est_s':26,'cached':42}
    node=dict(id=p.prog.current,method='local_fix',tool='local_fix',params={'action':'smooth'},status='waiting')
    p.snap=Snapshot(groups=[dict(name='meshed',nodes=[node])])
    p.prog.nodes={p.prog.current:dict(id=p.prog.current,state='start')}
    return p

def test_local_edit_without_percent_shows_phase_and_elapsed(app):
    p=progress_probe(app);p._refresh_progress();card=p.vp.card
    assert 'Smoothing the area' in card.title.text() and card.pct.text()=='Working...'
    assert card.bar.maximum()==0 and '0:12' in card.elapsed.text()
    p.prog.nodes[p.prog.current].update(frac=.5,text='Filling the gap');p.prog.frac=.5;p._refresh_progress()
    assert 'Filling the gap' in card.l2.text() and card.bar.value()==500     # the step's live detail: under the plain headline
    p.state=None;p.snap=None;p.shutdown();p.close()

def test_same_tool_time_is_used_not_the_last_delete(tmp_path):
    from scanner.stages.engine import Engine
    e=object.__new__(Engine);e.store=types.SimpleNamespace(nodes=str(tmp_path))
    step=types.SimpleNamespace(kind='op',op={'tool':'local_fix'},params={'action':'smooth'},store='meshed.ops')
    for name,tool,secs,created in [('old','local_fix',26,1),('new','delete',5,2)]:
        d=tmp_path/'meshed.ops'/name;d.mkdir(parents=True)
        (d/'meta.json').write_text(json.dumps(dict(status='ok',created=created,secs=secs,op=dict(tool=tool,params={'action':'smooth'}))))
    assert e._weight(step,{'nodes':{'meshed#op-0001':{'secs':5}}})==26
    assert e._weight(step,{'secs':{'op.local_fix.smooth':[26,29]}})==29

@pytest.mark.parametrize('tool,stats,kind,expected',[
 ('local_fix',{'moved':13894,'removed':0},'mesh','Moved 13,894 points'),
 ('local_fix',{'added_faces':120,'removed':0},'mesh','Filled 120 faces'),
 ('mirror',{'added_points':90},'points','Added 90 points'),
 ('mirror',{'added_faces':44},'mesh','Added 44 faces'),
 ('delete',{'removed':81},'mesh','Removed 81 faces'),
 ('overlap',{'removed':81},'points','Removed 81 points')])
def test_applied_counts_describe_actual_result(tool,stats,kind,expected):
    assert W.operation_stats(tool,stats,kind)==expected

def test_context_restore_requeues_mesh_and_blocks_picks_until_draw(app):
    from app2.qt.edit_view import EditView
    v=EditView();P=np.array([[0,0,0],[10,0,0]],np.float32)
    v.items={'part':dict(key='part',kind='points',P=P,N=None)};v.elements=(P,None)
    v._todo={};v.render_ready=True;v._restore_scene()
    assert not v.render_ready and v._todo['items'][0]['key']=='part'
    assert v._todo['elements'] is v.elements
    assert v.pick_front(QPointF(10,10)) is None
    v.close()

def test_leaving_edit_resets_tool_panels_even_during_build(app):
    from app2.qt.edit_page import EditPage
    p=B.BuildStepPage(SessionBridge(),False);edit=EditPage();p.stage='meshed'
    edit.attach(p);p.panel['meshed'].show_tab('Smooth area');p.state=types.SimpleNamespace(building=True)
    edit.detach();assert p.panel['meshed'].current=='Mesh'
    p.state=None;p.shutdown();p.close();edit.close()

def test_touch_up_reason_and_connected_button_without_a_device(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    p=ScanPage(SessionBridge(),False);(tmp_path/'model.ply').write_text('mesh fixture')
    p.state=types.SimpleNamespace(project=str(tmp_path),cam='gone',source_kind='live',recording=False,building=False,busy=False,learning=False)
    got=[];p.session=types.SimpleNamespace(touch_up=lambda:got.append('touch'))
    p._refresh_touch_up()
    assert not p.sides.touch.isHidden() and not p.sides.touch.isEnabled()
    assert p.sides.touch_reason.text()=='Connect the scanner to touch up'
    p.streamed=True;p.state.cam=None;p._refresh_touch_up()
    assert p.sides.touch.isEnabled() and p.sides.touch_reason.isHidden()
    p.sides.touch.click();assert got==['touch']
    p.state.building=True;p._refresh_touch_up();assert not p.sides.touch.isEnabled()
    p.ended=True;p._refresh_touch_up();assert not p.sides.touch.isEnabled()
    assert p.sides.touch_reason.text()=="Connect the scanner to touch up"
    p.close()

def test_cut_view_without_gl_is_safe(app):
    p=B.BuildStepPage(SessionBridge(),False);p.stage='meshed'
    assert p.vp.view is None and p._cut_view() is False
    assert p._clone_view() is False and p._fix_view() is False and p._mirror_view() is False
    assert p._spots_view() is None
    p.shutdown();p.close()

def test_sharp_card_uses_engine_params_and_tracks_a_change(app,monkeypatch):
    from app2.qt.onetap import preset_facts,facts_line
    from stages import params as PR
    assert preset_facts('plus')['detail']==PR.PRESETS['plus']['set']['fused.point_distance']==.4
    monkeypatch.setitem(PR.PRESETS['plus']['set'],'fused.point_distance',.55)
    assert '0.55 mm' in facts_line(preset_facts('plus'))

def test_driver_reply_survives_button_deleted_by_click(app):
    from app2.diagnostics.driver import Driver
    import shiboken6
    w=QPushButton('Open');w.clicked.connect(lambda:shiboken6.delete(w));w.show()
    driver=types.SimpleNamespace(_find=lambda req:(w,None),_click=lambda widget:widget.click())
    assert Driver.cmd_click(driver,{'wait':0})=={'clicked':'Open'}

def test_lineup_subtitle_wraps_in_narrow_rail(app):
    from app2.qt.stage_list import StageRow
    row=StageRow('lineup','Line-up');row.setFixedWidth(248)
    text='11 sides in, 1 left out (side 6), base side 2';row.set_info(text);row.show();app.processEvents()
    assert row.sub.wordWrap() and row.sub.text()==text
    h=row.sub.fontMetrics().boundingRect(QRect(0,0,row.sub.width(),10000),Qt.TextFlag.TextWordWrap,text).height()
    assert row.sub.height()>=h
    row.close()
