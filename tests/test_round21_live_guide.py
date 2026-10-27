"""Covered Touch up spots update cached suggestions outside the frame path."""
import threading
from types import SimpleNamespace as NS
import numpy as np
from app2.scan.touchup import TouchModel, COVER_GOOD
from app2.scan.session import ScanSession


def model():
    points=np.array([[0,0,100],[10,0,100],[20,0,100]],np.float32)
    spots=[dict(centre=p,kind='hole') for p in points]
    tm=TouchModel(points,np.tile([0,0,-1],(3,1)),np.ones(3),np.arange(3),spots)
    tm.scan_guide=dict(poses=[dict(rank=1,indices=[1000,1001],area_mm2=9,share=.9,text='First, fills about 90 percent'),
                              dict(rank=2,indices=[1002],area_mm2=1,share=.1,text='Second, fills about 10 percent')],
        live_samples=dict(vertex_ids=[1000,1001,1002],areas=[8,1,2],spot_ids=[0,1,2],
                          spot_centers=points.tolist(),spot_kinds=['hole']*3))
    return tm


def test_completed_spot_reorders_remaining_poses_using_original_vertex_ids():
    tm=model(); changed=threading.Event(); tm.on_guide_changed=changed.set
    depth=np.zeros((1,21),np.uint16);depth[0,0]=1000
    for _ in range(COVER_GOOD):tm.update(depth,np.eye(4),(100,100,0,0))
    assert changed.wait(5)
    assert tm.done.tolist()==[True,False,False]
    poses=tm.scan_guide['poses']
    assert [p['indices'] for p in poses]==[[1002],[1001]]
    assert [p['rank'] for p in poses]==[1,2]
    assert [p['area_mm2'] for p in poses]==[2,1]
    assert '67 percent' in poses[0]['text']
    assert tm.scan_guide['covered_spots']==1


def test_finished_targets_drop_and_repeated_frames_do_not_spawn_more_work():
    tm=model(); changed=threading.Event();tm.on_guide_changed=changed.set
    depth=np.full((1,21),1000,np.uint16)
    for _ in range(COVER_GOOD):tm.update(depth,np.eye(4),(100,100,0,0))
    assert changed.wait(5)
    assert tm.scan_guide['poses']==[]
    revision=tm.scan_guide['live_revision']
    for _ in range(20):tm.update(depth,np.eye(4),(100,100,0,0))
    assert tm.scan_guide['live_revision']==revision


def test_slow_rerank_does_not_block_frame_tracking():
    tm=model();entered=threading.Event();release=threading.Event();done=threading.Event()
    rank=tm._rank_live_guidance
    def slow(snapshot):
        entered.set();assert release.wait(5);return rank(snapshot)
    tm._rank_live_guidance=slow;tm.on_guide_changed=done.set
    try:
        tm.done[0]=True;tm._request_guide_refresh()
        assert entered.wait(5)
        # The ranking worker is blocked, but the same tracking-thread update returns.
        tm.update(np.zeros((1,21),np.uint16),np.eye(4),(100,100,0,0))
        assert not done.is_set()
    finally:release.set()
    assert done.wait(5)


def test_worker_refresh_posts_to_session_and_emits_without_another_frame():
    tm=model();sd=NS(touch_model=tm);queued=[];clouds=[]
    session=NS(_stop=False,picked=sd,_post=lambda fn,*args:queued.append((fn,args)),
               _touch_model=lambda s:s.touch_model,_emit_touch=lambda s,t:clouds.append(t.scan_guide))
    session._touch_guide_ready=lambda *args:ScanSession._touch_guide_ready(session,*args)
    ScanSession._bind_touch_guide(session,sd,tm)
    tm.on_guide_changed()
    assert not clouds and len(queued)==1
    fn,args=queued.pop();fn(*args)
    assert clouds==[tm.scan_guide]
    session.picked=None;fn(*args);assert len(clouds)==1
    session._stop=True;tm.on_guide_changed();assert not queued


def test_attached_guide_removes_already_covered_counts_and_unreachable_targets():
    tm=model();guide=dict(tm.scan_guide,remaining=[1000,1001],needs_flip=[1000],unreachable=[1000,1002],
                         missing_vertices=6,rough_vertices=0,spots=3,surface_mm2=11)
    guide['live_samples']=dict(guide['live_samples'],counts=[3,1,2],kinds=[1,1,1])
    tm.done[0]=True;changed=threading.Event();tm.on_guide_changed=changed.set
    tm.scan_guide=guide
    assert changed.wait(5)
    result=tm.scan_guide
    assert result['remaining']==[1001] and result['needs_flip']==[]
    assert result['unreachable']==[1002]
    assert result['missing_vertices']==3 and result['rough_vertices']==0
    assert result['spots']==2 and result['surface_mm2']==3


def test_failed_refresh_keeps_previous_guidance():
    tm=model();previous=tm.scan_guide;finished=threading.Event()
    def fail(done):
        finished.set();raise ValueError('invalid suggestion')
    tm._rank_live_guidance=fail;tm.done[0]=True;tm._request_guide_refresh()
    assert finished.wait(5)
    assert tm.scan_guide is previous


def test_coverage_changes_during_refresh_coalesce_into_latest_result():
    tm=model();entered=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
    rank=tm._rank_live_guidance
    def wait_first(done):
        calls.append(done.copy())
        if len(calls)==1:entered.set();assert release.wait(5)
        return rank(done)
    tm._rank_live_guidance=wait_first;tm.on_guide_changed=finished.set
    try:
        tm.done[0]=True;tm._request_guide_refresh();assert entered.wait(5)
        tm.done[2]=True;tm._request_guide_refresh()
    finally:release.set()
    assert finished.wait(5)
    assert len(calls)==2 and tm.scan_guide['covered_spots']==2
    assert [p['indices'] for p in tm.scan_guide['poses']]==[[1001]]
