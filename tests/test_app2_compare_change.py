"""Compare keeps the cached result before a confirmed edit, not model.ply."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-compare-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import copy
from types import SimpleNamespace
import pytest
from app2.qt.build_step import BuildStepPage
from app2.builder.stagehost import Snapshot


class Probe:
    _clear_compare_change=BuildStepPage._clear_compare_change
    _remember_compare_change=BuildStepPage._remember_compare_change
    _sync_compare_change=BuildStepPage._sync_compare_change
    _before_path=BuildStepPage._before_path
    def _view_path(self): return self.snap.out(self.snap.final('meshed'))
    def _list_token(self): return 'meshed'
    def _node_for_view(self): return self.snap.node(self.snap.final('meshed'))
    def _chain(self): return self.snap.chains['meshed']
    def _side(self): return None
    def _all_mode(self): return False


def setup(tmp_path):
    p=Probe(); p.host_root=str(tmp_path); p.state=None; p.stage='meshed'; p.before_kind='tool'; p.change_baseline=None; p.comparing=False
    p.action=SimpleNamespace(enabled=False)
    p.vp=SimpleNamespace(before_change_action=SimpleNamespace(setEnabled=lambda on:setattr(p.action,'enabled',on)))
    ops=[{'id':str(i),'tool':'smooth','on':True} for i in (1,2,3)]
    files={}
    for node in ('meshed','meshed#1','meshed#2','meshed#3'):
        path=tmp_path/'stages'/'nodes'/'meshed.ops'/node/'mesh.ply'; path.parent.mkdir(parents=True); path.write_bytes(b'immutable fixture')
        files[node]={'mesh':str(path)}
    recipe={'ops':{'meshed':copy.deepcopy(ops)}}
    p.snap=Snapshot(recipe=recipe,files=files,finals={'meshed':'meshed#3'},chains={'meshed':[('meshed',i,o) for i,o in enumerate(ops)]},groups=[{'name':'meshed','nodes':[{'id':'meshed#3','status':'done'}]}])
    return p


def test_before_change_keeps_cached_path_after_rebuild(tmp_path):
    p=setup(tmp_path); old=p._view_path(); before=copy.deepcopy(p.snap.recipe)
    after=copy.deepcopy(before); after['ops']['meshed'][1]['on']=False
    assert p._remember_compare_change(after,'Before turning off edit 2','Now: edit 2 off')
    assert p._before_path()==(old,'Before turning off edit 2') and p.action.enabled
    p.snap.recipe=after
    nextfile=tmp_path/'stages'/'nodes'/'meshed.ops'/'newkey'/'mesh.ply'; nextfile.parent.mkdir(); nextfile.write_bytes(b'new result')
    p.snap.files['meshed#3']['mesh']=str(nextfile)
    assert p._view_path()!=old and p._before_path()[0]==old
    assert p.change_baseline['changed']
    assert p.change_baseline['before_recipe']==before


@pytest.mark.parametrize('change',['recipe','project','stage','missing'])
def test_baseline_invalidated_when_context_changes(tmp_path,change):
    p=setup(tmp_path); expected=copy.deepcopy(p.snap.recipe); expected['ops']['meshed'][1]['on']=False
    p._remember_compare_change(expected,'Before','Now')
    if change=='recipe': p.snap.recipe={'other':'edit'}
    elif change=='project': p.host_root=str(tmp_path/'other')
    elif change=='stage': p.stage='fused'
    else: Path(p.change_baseline['path']).unlink()
    assert p._sync_compare_change() is None and not p.action.enabled
    assert p.before_kind=='tool'


def test_published_mutable_file_is_not_used(tmp_path):
    p=setup(tmp_path); published=tmp_path/'model.ply'; published.write_bytes(b'mutable')
    p.snap.files['meshed#3']['mesh']=str(published)
    assert not p._remember_compare_change(p.snap.recipe,'Before','Now')
    assert not p.action.enabled


def test_before_last_tool_skips_disabled_rows(tmp_path):
    p=setup(tmp_path)
    p.snap.chains['meshed'][1][2]['on']=False
    assert p._before_path()[0]==p.snap.out('meshed#1')
    p.snap.chains['meshed'][2][2]['on']=False
    assert p._before_path()[0]==p.snap.out('meshed')


def test_confirmed_toggle_and_batch_capture_before_mutation(tmp_path):
    p=setup(tmp_path); calls=[]; p.busy_call=None; p.running=lambda:False
    p._close_edit_confirm=lambda:setattr(p,'_pending_edit',None)
    p.op_call=lambda name,value,token:calls.append((name,value,token))
    p.notice=lambda message:pytest.fail(message)
    original=copy.deepcopy(p.snap.recipe)
    p._pending_edit=('toggle_op','meshed','2',copy.deepcopy(original),'meshed')
    BuildStepPage._accept_edit_confirm(p)
    assert p.change_baseline['before_label']=='Before turning off edit 2'
    assert p.change_baseline['after_label']=='Now: edit 2 off'
    assert p.snap.recipe==original
    p._pending_edit=('h_toggle_many','meshed',{'1':False,'3':False},copy.deepcopy(original),'meshed')
    BuildStepPage._accept_edit_confirm(p)
    assert p.change_baseline['before_label']=='Before edit changes'
    assert [o['on'] for o in p.change_baseline['after_recipe']['ops']['meshed']]==[False,True,False]
    assert p.snap.recipe==original
    assert calls[-1]==('h_toggle_many',{'1':False,'3':False},'meshed')


def test_right_label_waits_for_finished_loaded_result(tmp_path):
    from app2.qt import build_words as W
    p=setup(tmp_path); expected=copy.deepcopy(p.snap.recipe); expected['ops']['meshed'][1]['on']=False
    p._remember_compare_change(expected,'Before turning off edit 2','Now: edit 2 off')
    p.comparing=True; p.detected=None; p.peek=False; p.gl=True; p.busy_call=None
    p.shown={'path':p._view_path()}; p.view_label=lambda:[('Built model',None)]
    p._platter_preview=lambda:False; p._row_state=lambda sid:('Ready',None)
    parts=[]
    p.vp.main=SimpleNamespace(setToolTip=lambda text:None,set_chips=lambda chips,note,center:parts.append(chips))
    p.running=lambda:True
    p.snap.recipe=expected
    BuildStepPage._refresh_chips(p)
    assert parts[-1][0][0]=='Rebuilding change'
    p.running=lambda:False
    BuildStepPage._refresh_chips(p)
    assert parts[-1][0][0]=='Now: edit 2 off'
    p.shown={'path':'old picture still loading'}
    BuildStepPage._refresh_chips(p)
    assert parts[-1][0][0]=='Rebuilding change'
