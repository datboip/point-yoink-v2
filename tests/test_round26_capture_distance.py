import json,numpy as np
from app2.scan.capture_distance import merge,query,build_side,load,UNKNOWN,rgb


def test_voxel_best_band_only_upgrades_and_unknown_is_separate():
    keys=np.array([[0,0,0]]);levels=np.array([0],np.uint8)
    for level in [1,2,0,1]:keys,levels=merge(keys,levels,np.array([[0,0,0]]),np.array([level],np.uint8))
    assert levels.tolist()==[2]
    assert query((keys,levels,1),[[.1,.1,.1],[9,9,9]]).tolist()==[2,255]
    assert not np.array_equal(rgb(np.array([0])),rgb(np.array([255])))


def test_side_roundtrip_and_rewind_remove_discarded_evidence(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();cal=(100,100,0,0);entries=[];stamps=[]
    for i,(z,shift,state) in enumerate([(2500,50,'usable depth'),(3000,0,'strong depth'),(3000,0,'usable depth')]):
        name='frame_000_%04d'%i;d=np.zeros((2,2),np.uint16);d[0,0]=z;d.tofile(cache/(name+'.dph'));M=np.diag([1.,-1.,-1.,1.]);M[2,3]=shift
        entries.append((0,i,M));stamps.append(dict(frame=name,capture_bands=[dict(lo=100,hi=600,state=state)]))
    field=build_side(tmp_path,entries,cal,(2,2),'near',stamps)
    assert query(field,[[0,0,300]]).tolist()==[2]
    assert query(load(tmp_path),[[0,0,300]]).tolist()==[2]
    field=build_side(tmp_path,entries[:1],cal,(2,2),'near',stamps[:1]);assert query(field,[[0,0,300]]).tolist()==[1]
    assert json.loads((tmp_path/'capture-distance.json').read_text())['tracked_frames']==1


def test_placed_project_combines_best_without_coloring_unseen_voxels(tmp_path):
    (tmp_path/'data/s').mkdir(parents=True);(tmp_path/'build').mkdir()
    with (tmp_path/'data/s/capture-distance.npz').open('wb') as f:np.savez(f,keys=np.array([[0,0,0]]),levels=np.array([2],np.uint8),step=1.)
    M=np.eye(4);M[0,3]=100;(tmp_path/'build/frames.json').write_text(json.dumps(dict(sides={'s':M.tolist()})))
    from app2.scan.capture_distance import project_levels
    assert project_levels(tmp_path,np.array([[100.1,.1,.1],[0,0,0]])).tolist()==[2,255]


def test_distance_worker_waits_for_a_real_gap_between_frames(monkeypatch):
    import threading,time
    from app2.scan import capture_distance as c
    gate=threading.Event();clock=[time.monotonic(),.02];done=threading.Event()
    monkeypatch.setattr(c,'_CAPTURE_IDLE',gate);monkeypatch.setattr(c,'_CAPTURE_CLOCK',clock)
    thread=threading.Thread(target=lambda:(c.wait_for_budget(),done.set()));thread.start()
    assert not done.wait(.01)
    clock[0]=time.monotonic();gate.set();assert done.wait(.5);thread.join()
    assert time.monotonic()-clock[0]>=clock[1]


def test_frame_budget_gate_reopens_even_when_processing_raises():
    import threading
    from types import SimpleNamespace as NS
    from app2.scan.session_distance import distance_frame
    import pytest
    state=NS(_distance_idle=threading.Event(),_distance_clock=[0,0]);state._distance_idle.set()
    @distance_frame
    def fail(s,frame):
        assert not s._distance_idle.is_set();raise ValueError('bad frame')
    with pytest.raises(ValueError):fail(state,None)
    assert state._distance_idle.is_set() and state._distance_clock[1]>0


def test_worker_preparation_finishes_before_recording():
    from types import SimpleNamespace as NS
    from app2.scan.session_distance import DistanceSession
    class Ready:
        def result(self):calls.append('ready');return 1
    calls=[]
    s=NS(_ensure_distance_pool=lambda:calls.append('pool'),_distance_pool=NS(submit=lambda fn:Ready()))
    DistanceSession._prepare_distance(s)
    assert calls==['pool','ready'] and s._distance_ready_ms>=0
