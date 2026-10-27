"""Applied changes require review before mutating a recipe."""
import os,tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt.bridge import SessionBridge
from app2.qt.build_step import BuildStepPage

@pytest.mark.parametrize('size',[(1196,748),(1604,992)])
def test_cancel_confirm_and_stale_recipe(size):
    app=QApplication.instance() or QApplication([])
    p=BuildStepPage(SessionBridge(),gl=False);p.resize(*size);p.show_stage('meshed');p.show()
    p.snap=SimpleNamespace(recipe={'test':1},chains={'meshed':[('meshed',0,{'id':'a','tool':'finish','on':True}),('meshed',1,{'id':'b','tool':'smooth','on':True})]})
    p._applied_ids=[('meshed','a')];p.running=lambda:False
    calls=[];p.op_call=lambda *a,**kw:calls.append((a,kw));p.notice=lambda *a:None
    p.row_call('toggle_op',0);app.processEvents()
    assert not calls and p.edit_confirm.isVisible()
    assert '1 later edit' in p.edit_confirm.text.text()
    assert p.rect().contains(p.edit_confirm.geometry())
    p.edit_confirm.cancel.click();assert not calls
    p.row_call('toggle_op',0);p.edit_confirm.go.click();assert len(calls)==1
    p.row_call('drop_op',0);p.snap.recipe['test']=2;p.edit_confirm.go.click();assert len(calls)==1
    p.snap=None;p.shutdown();p.close();p.deleteLater();app.processEvents()

def test_checkbox_batch_waits_for_apply_and_builds_once(tmp_path):
    import app2
    from stages.project import Project
    from stages.engine import Engine
    from app2.builder.stagehost import StageHost
    app=QApplication.instance() or QApplication([])
    (tmp_path/'data').mkdir()
    h=StageHost.__new__(StageHost);h.e=Engine(Project.open(str(tmp_path),create=True))
    for value in (5,8):h.e.add_op('meshed',{'tool':'finish','params':{'smooth':value}})
    before=h.e.recipe();chain=h.e.chain_ops('meshed')
    p=BuildStepPage(SessionBridge(),gl=False);p.show_stage('meshed');p.resize(1196,748);p.show()
    p.snap=SimpleNamespace(recipe=before,chains={'meshed':chain},sides=[])
    p._applied_ids=[('meshed',o['id']) for _,_,o in chain];p.running=lambda:False
    p._remember_compare_change=lambda *a:False
    calls=[];runs=[]
    def call(name,*args,then=None,**kwargs):
        calls.append(name);result=getattr(h,name)(*args)
        if then:then(result)
    p.call=call;p.start_run=runs.append
    p.applied.set_rows('Meshed',[dict(id=o['id'],label='Finish',on=True) for _,_,o in chain])
    p.applied.set_tools_enabled(True)
    for checkbox in p.applied.checks.values():checkbox.click()
    assert len(p.applied.pending)==2
    assert h.e.recipe()==before and not runs
    p.applied.apply_b.click();assert p.edit_confirm.isVisible() and not runs
    p.edit_confirm.go.click()
    assert calls==['h_toggle_many'] and len(runs)==1
    assert all(not o['on'] for _,_,o in h.e.chain_ops('meshed'))
    h.e.undo();assert h.e.recipe()==before
    p.snap=None;p.shutdown();p.close();p.deleteLater();app.processEvents()
