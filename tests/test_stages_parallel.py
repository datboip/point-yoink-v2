"""Ready-node overlap, exact output replay, branch failures, and cancellation."""
import io,json,threading,time
from pathlib import Path
import pytest
from test_stages_core import make_project,install_fakes,remove_fakes,FakeMixin
from stages.engine import Engine
from stages.project import Project
from stages.parallel import Admission,Jobs,gpu_slots,GPU_JOB_BYTES,GPU_RESERVE
from stages.resources import Budget,GIB
from stages.runner import Runner,Cancelled

@pytest.fixture
def fakes():
    install_fakes()
    try:yield
    finally:remove_fakes()

def engine(root):
    make_project(str(root),n=3)
    e=Engine(Project.open(str(root)),out=io.StringIO());e._post=lambda *a:None;return e

def digests(e):
    return {meta['node']:meta['digest'] for _,_,_,meta in e.store.artifacts() if meta and meta.get('digest')}

def test_ready_nodes_overlap_and_match_serial_bytes(fakes,tmp_path,monkeypatch):
    original=FakeMixin.execute;barrier=threading.Barrier(2);overlap=[];parallel=False;events=[]
    def execute(self,ctx):
        if parallel and ctx.step.kind=='side.fuse' and ctx.step.n<=2:
            overlap.append(ctx.step.n);barrier.wait(5)
        ctx.progress(1,2,'half complete')
        return original(self,ctx)
    monkeypatch.setattr(FakeMixin,'execute',execute)
    monkeypatch.setattr('stages.parallel.gpu_slots',lambda requested=2:(min(2,requested),None))   # not the desktop's free VRAM
    from stages import resources as RS
    monkeypatch.setattr(RS,'budget',lambda memory_gb=None:RS.Budget(32*RS.GIB,32.0,16,1600))    # nor its free RAM (two jobs must fit)
    monkeypatch.setenv('POINTYOINK_BUILD_WORKERS','1');a=engine(tmp_path/'serial');assert a.run(to='lineup',publish=False).ok
    parallel=True;monkeypatch.setenv('POINTYOINK_BUILD_WORKERS','24');b=engine(tmp_path/'parallel')
    b.on_event=lambda kind,name,value:events.append((name,dict(value)))
    result=b.run(to='lineup',publish=False);assert result.ok,result.error
    assert sorted(overlap)==[1,2] and digests(a)==digests(b)
    fractions=[x['overall'] for name,x in events if name=='progress'];assert fractions==sorted(fractions) and all(0<=x<=1 for x in fractions)
    assert result.ran==[s.id for s in b.graph().steps if s.id in result.ran]
    for nid in result.ran:
        states=[x['state'] for name,x in events if name=='node' and x['id']==nid]
        assert states==['start','done']

def test_failed_side_does_not_stop_independent_sides(fakes,tmp_path,monkeypatch):
    original=FakeMixin.execute
    def execute(self,ctx):
        if ctx.step.kind=='side.fuse' and ctx.step.n==1:raise RuntimeError('forced failure')
        return original(self,ctx)
    monkeypatch.setattr(FakeMixin,'execute',execute);e=engine(tmp_path/'failed');result=e.run(to='lineup',publish=False)
    assert not result.ok and result.failed=='side.fuse@'+e.p.sides()[0].id
    nodes=e.store.state()['nodes'];assert 'side.keep@'+e.p.sides()[1].id in nodes
    assert 'side.platter@'+e.p.sides()[0].id not in nodes
    assert 'lineup.pairs' not in nodes

def test_worker_inherits_cancel_even_in_nested_entry_scope(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    runner=Runner();entered=threading.Event();release=threading.Event();observed=[]
    with runner.scope():
        sid=runner._tl.scope
        def worker():
            with runner.worker(sid,None,'side.fuse'):
                with runner.scope():entered.set();release.wait(3);observed.append(runner.cancelled)
                observed.append(runner.cancelled)
        t=threading.Thread(target=worker);t.start();assert entered.wait(3);runner.cancel();release.set();t.join(3)
    assert observed==[True,True]
    with runner.scope():assert not runner.cancelled

def test_child_slices_never_multiply_the_budget(monkeypatch):
    budget=Budget(8*GIB,8,8,800);runner=Runner();jobs=Jobs(runner,budget,2)
    monkeypatch.setattr('stages.parallel.resources.budget',lambda *a:budget)
    first=threading.Event();second=threading.Event();release=threading.Event();caps=[]
    def take(event):
        with jobs.lease(5,False,'cpu') as b:caps.append(b.memory_bytes);event.set();release.wait(3)
    a=threading.Thread(target=take,args=(first,));b=threading.Thread(target=take,args=(second,));a.start();assert first.wait(2);b.start()
    assert not second.wait(.1);release.set();a.join(3);b.join(3)
    assert caps==[5*GIB,5*GIB] and jobs.peak_mem==5*GIB and jobs.peak_cpu<=8

def test_a_whole_budget_job_runs_when_another_program_took_some_ram(monkeypatch):
    # The final fusion asks for the whole budget; with the app open less is free than at launch. It must run on what
    # is free (as the serial build did), not wait forever.
    budget=Budget(16*GIB,16,8,800);jobs=Jobs(Runner(),budget,2)
    monkeypatch.setattr('stages.parallel.resources.budget',lambda *a:Budget(15*GIB,16,8,800))
    done=[]
    def take():
        with jobs.lease(None,False,'fused') as b:done.append(b.memory_bytes)
    t=threading.Thread(target=take,daemon=True);t.start();t.join(3)
    assert done==[15*GIB]

def test_three_gpu_grids_are_refused_before_launch(monkeypatch):
    monkeypatch.setattr('stages.parallel.subprocess.check_output',lambda *a,**k:str(8736))
    slots,free=gpu_slots(3);assert slots==2 and 3*GPU_JOB_BYTES+GPU_RESERVE>free


def test_cancel_from_plan_does_not_report_success(fakes,tmp_path):
    e=engine(tmp_path/'cancelled')
    e.on_event=lambda kind,name,value:e.cancel() if name=='plan' else None
    result=e.run(to='lineup',publish=False)
    assert not result.ok and result.error=='cancelled' and not result.ran


def test_one_cancel_stops_both_owned_child_groups(tmp_path,monkeypatch):
    import sys
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    budget=Budget(8*GIB,8,16,1600)
    monkeypatch.setattr('stages.runner.resources.budget',lambda *a:budget)
    runner=Runner();runner.argv=lambda cmd,*a,**k:list(cmd)
    gate=threading.Barrier(3);outcomes=[]
    with runner.scope():
        scope=runner._tl.scope
        with runner.batch(1) as jobs:
            def work():
                with runner.worker(scope,jobs,'side.keep'):
                    try:runner.run([sys.executable,'-c','import time;print("ready",flush=True);time.sleep(30)'],1,on_line=lambda line:gate.wait(5) if line=='ready' else None)
                    except Cancelled:outcomes.append('cancelled')
            threads=[threading.Thread(target=work) for _ in range(2)]
            for t in threads:t.start()
            gate.wait(5);assert len(runner.pgids)==2;runner.cancel()
            for t in threads:t.join(5);assert not t.is_alive()
    assert outcomes==['cancelled','cancelled'] and not runner.pgids
