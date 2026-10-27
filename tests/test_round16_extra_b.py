import json,os,time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
@pytest.fixture
def app():return QApplication.instance() or QApplication([])
@pytest.fixture(autouse=True)
def config(monkeypatch,tmp_path):monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))

def test_finish_summary_has_short_numbered_parts():
    from app2.builder.stagehost import tool_words
    assert tool_words('finish',{'faces':250000,'smooth':5},'mesh')=='Finish for print · 250k triangles · smooth 5'

def test_applied_names_elide_instead_of_splitting_numbers(app):
    from app2.qt.tool_panel import AppliedList
    from app2.qt.widgets import ElidedLabel
    p=AppliedList();p.set_rows('Meshed',[dict(id='f',label='Finish for print · 250k triangles · smooth 5')])
    label=p.findChild(ElidedLabel,'AppliedEntryText')
    assert label is not None and not label.wordWrap()
    assert label.toolTip()=='Finish for print · 250k triangles · smooth 5'

def test_touch_header_names_empty_and_recorded_sides(app):
    from app2.qt.model_panel import ModelPanel
    p=ModelPanel(False);p.set_context(False,7,0);p.set_touch_context(True,dict(done=0,total=6,left=6,holes=4,rough=2))
    assert p.title_parts()[0][0]=='Touch up · press Start, then aim at the red and purple spots (or any area you want better)'
    p.set_context(False,7,120);p.set_touch_context(True,dict(done=4,total=6,left=2,holes=1,rough=1))
    assert p.title_parts()[0][0]=='Touch up · 120 frames · 4 of 6 spots done'
    assert p.overlay.touch['left']==2

def test_touch_counts_drop_when_holes_and_rough_turn_done():
    from app2.scan.touchup import TouchModel
    p=NS(spots=[{'kind':'hole'}]*4+[{'kind':'rough'}]*2,done=np.zeros(6,bool))
    assert TouchModel.remaining_counts(p)==dict(left=6,holes=4,rough=2)
    p.done[[0,1,4]]=True
    assert TouchModel.remaining_counts(p)==dict(left=3,holes=2,rough=1)

def test_repeated_touch_up_reuses_empty_side_and_restores_model_tracker(tmp_path):
    from app2.scan.session import ScanSession
    side=NS(touch_info={},frames=0,entries=[],cache=str(tmp_path),touch_model=None);called=[];factory=object()
    s=NS(recording=False,busy=False,building=False,bed=NS(learning=False),project=NS(sides=[side]),picked=side,
         host=NS(has=lambda sd:False,new=lambda sd,**kw:called.append((sd,kw['factory']))),
         _busy=lambda label,fn:fn(),_load_touch=lambda sd:factory,_touch_model=lambda sd:None,
         _emit_sides=lambda:None,_emit_state=lambda:None)
    ScanSession._touch_up(s)
    assert called==[(side,factory)] and len(s.project.sides)==1

def test_empty_touch_cleanup_never_removes_existing_or_recorded_sides(tmp_path):
    from app2.scan.session import ScanSession
    side=NS(touch_info={},frames=0,entries=[],cache=str(tmp_path));removed=[]
    p=NS(project=NS(made_sides=[],delete_side=removed.append),picked=side,host=NS(has=lambda sd:False))
    assert not ScanSession._drop_empty_touch_side(p,side)
    p.project.made_sides=[side];side.frames=1
    assert not ScanSession._drop_empty_touch_side(p,side)
    side.frames=0;(tmp_path/'lost.dph').write_bytes(b'raw')
    assert not ScanSession._drop_empty_touch_side(p,side)
    (tmp_path/'lost.dph').unlink()
    assert ScanSession._drop_empty_touch_side(p,side) and removed==[side] and p.picked is None

def platter_context(tmp_path,touch=False,no_base=False):
    side=tmp_path/'side';side.mkdir();(side/'cache').mkdir()
    if touch:(side/'touchup.json').write_text('{}')
    calls=[];notes=[]
    ctx=NS(step=NS(side='s',n=7),params={'margin':1.,'max_tilt':8.},out_dir=str(tmp_path),inputs={'raw':NS(files={'mesh':'raw.ply'})},
        project=NS(folder=lambda sid:str(side),frame_size=lambda sid:[800,600],board_plane=lambda sid:None),
        call=lambda *a,**k:(calls.append(a) or 'platter: 120 support points, 0.2 deg off the spin axis; kept 5 triangles in 1 pieces, part height 40 mm'),
        job=lambda *a,**k:dict(no_base=no_base,frames=32),legacy=lambda *a,**k:notes.append(k.get('msg')),warn=lambda t:None)
    return ctx,calls,notes

def test_touchup_base_cut_skips_without_bed_search_or_warning(tmp_path):
    from scanner.stages.methods.side import PlatterAuto
    ctx,calls,notes=platter_context(tmp_path,touch=True)
    r=PlatterAuto().execute(ctx)
    assert not calls and r['alias_of']=='raw' and r['stats']['lineup']=='shape'
    assert notes==['No base in view: nothing to cut']

def test_observed_no_bed_skips_but_unknown_keeps_normal_cut(tmp_path,monkeypatch):
    from scanner.stages.methods.side import PlatterAuto
    monkeypatch.setattr(PlatterAuto,'board_plane',lambda self,c:None)
    ctx,calls,notes=platter_context(tmp_path,no_base=True)
    assert PlatterAuto().execute(ctx)['alias_of']=='raw' and calls==[]
    ctx.job=lambda *a,**k:dict(no_base=False)
    assert PlatterAuto().execute(ctx)['outputs']['mesh']=='cut.ply' and len(calls)==1

def test_gpu_capacity_depends_on_actual_keys_not_frame_count():
    from scanner.fusion_memory import allocation_plan,GIB
    keys={(i,0,0) for i in range(800)}
    plan=allocation_plan(keys,400,7*GIB)
    assert plan['capacity']==1072 and plan['chunk_blocks']==256
    assert plan['capacity']<100000 and plan['estimated_peak_bytes']<=plan['budget_bytes']
    with pytest.raises(RuntimeError,match='complete grid cannot fit'):
        allocation_plan({(i,0,0) for i in range(50000)},800,4*GIB)

def test_peak_vram_log_counts_own_process_not_desktop(monkeypatch):
    from scanner.fusion_memory import Peak,MIB
    monkeypatch.setattr('scanner.fusion_memory.subprocess.check_output',lambda *a,**k:f'123, 9000\n{os.getpid()}, 1200\n')
    events=[];p=Peak('CUDA:0',lambda event,**data:events.append((event,data)));p.sample()
    assert p.peak==1200*MIB
    p.closed=False;p.thread=NS(join=lambda _:None);p.finish()
    assert events[-1][0]=='fusion_memory' and events[-1][1]['peak_vram_mib']==1200


def test_no_base_status_is_calm_and_never_asks_for_a_rebuild():
    from app2.qt.build_step import BuildStepPage
    node=dict(id='side.platter@s',method='auto',status='done',stats=dict(source='no base in view',base_cut_skipped=True))
    self=NS(snap=NS(out=lambda *a:None),_nodes=lambda group:[node])
    assert BuildStepPage._base_state(self)==('No base in view: nothing to cut','',None,None)
    assert BuildStepPage._cut_status(self,node,'auto')==('No base in view: nothing to cut',None)


def test_no_bed_observer_unknown_does_not_guess_when_view_files_are_missing(tmp_path):
    from scanner.no_base import observe
    assert observe(dict(folder=str(tmp_path),size=[400,300]))['no_base'] is False

def test_no_base_job_uses_the_stage_runner_module_namespace(tmp_path):
    import importlib
    from scanner.stages.methods.side import PlatterAuto
    ctx,calls,notes=platter_context(tmp_path)
    def job(fn,payload,**kwargs):
        module,name=fn.split(':')
        return getattr(importlib.import_module('stages.'+module),name)(payload)
    ctx.job=job
    assert PlatterAuto().execute(ctx)['outputs']['mesh']=='cut.ply'


def test_no_base_job_dispatches_in_a_real_subprocess(tmp_path):
    import subprocess,sys
    from pathlib import Path
    from scanner.stages.methods.side import PlatterAuto
    ctx,calls,notes=platter_context(tmp_path)
    names=[]
    ctx.job=lambda fn,payload,**kwargs:(names.append(fn) or dict(no_base=False))
    PlatterAuto().execute(ctx)
    source=Path(__file__).resolve().parents[1]
    job=tmp_path/'job.json';result=tmp_path/'result.json'
    job.write_text(json.dumps(dict(fn=names[0],folder=str(tmp_path/'side'),size=[400,300],result=str(result))))
    run=subprocess.run([sys.executable,str(source/'scanner/stages/job.py'),str(job)],cwd=source,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(result.read_text())['no_base'] is False
