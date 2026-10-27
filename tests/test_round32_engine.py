"""Accepted-model bed seeds and evidence-based early color learning."""
from types import SimpleNamespace as NS
import numpy as np
import pytest


def test_reference_needs_three_agreeing_accepted_placements():
    from app2.scan.bed_resume import Reference
    r=Reference();bad=np.eye(4);bad[0,3]=100
    r.add(bad);r.add(np.eye(4));r.add(np.eye(4));assert r.value is None
    r.add(np.eye(4));assert np.allclose(r.value,np.eye(4))


def test_conflicting_saved_placements_do_not_leave_a_stale_reference():
    from app2.scan.bed_resume import Reference
    r=Reference();other=np.eye(4);other[0,3]=100
    for _ in range(3):r.add(np.eye(4))
    assert r.value is not None
    for _ in range(3):r.add(other)
    assert r.value is None


def test_refused_or_unobserved_frame_cannot_set_reference():
    from app2.scan.bed_pose import BedPoseGuide
    g=BedPoseGuide();g.step_hit=NS(B_cam=np.eye(4))
    for _ in range(4):g.accepted(dict(ok=False,T=np.eye(4)))
    assert g.reference is None
    g.step_hit=None
    for _ in range(4):g.accepted(dict(ok=True,T=np.eye(4)))
    assert g.reference is None
    g.step_hit=NS(B_cam=np.eye(4))
    for _ in range(3):g.accepted(dict(ok=True,T=np.eye(4)))
    assert np.allclose(g.reference,np.eye(4))


def test_resume_hint_requires_local_depth_fit():
    from app2.scan.tracking import TrackerHost
    calls=[];host=TrackerHost((1,1,0,0),20,20)
    host.trk=NS(_hint_local_only=False,hint=lambda *a,**kw:calls.append((a,kw)))
    host.resume_hint(np.eye(4));assert calls[0][1]['local_only'] is True
    assert calls[0][1]['guard_mm']>0 and calls[0][1]['guard_deg']>0


def test_known_resume_reference_without_dots_waits_instead_of_search():
    from app2.scan.bed_adapter import ScannerBed
    calls=[]
    guide=NS(reference=np.eye(4),last_hit=None,step=lambda *a,**kw:False)
    sb=NS(bed=object(),hint_pose=lambda *a:calls.append('fallback'))
    side=NS(ps=1,idx=0,bed=NS(off_pass=None,sb=sb,pose_guide=guide),scan_profile={})
    tracker=NS(marker_required=False,marker_hint=lambda *a:None)
    ScannerBed.__new__(ScannerBed).before_step(side,None,None,tracker,None)
    assert tracker.marker_required and not calls


def test_color_resume_requires_paired_bias_and_current_depth_plane(monkeypatch):
    from app2.scan.bed_pose import BedPoseGuide
    from app2.scan import bed_pose,color_bed,preview
    model=NS();sb=NS(bed=model,hints_off=lambda ps:False,prior_for=lambda ps:None)
    side=NS(ps=1,bed=NS(sb=sb));adapter=NS(cal=(1,1,0,0),color_packet=dict(jpeg=b'frame'))
    calls=[];tracker=NS(resume_hint=lambda T:calls.append(T))
    monkeypatch.setattr(bed_pose.bed,'has_dots',lambda model:True)
    monkeypatch.setattr(bed_pose.bed,'find_bed',lambda *a,**kw:None)
    monkeypatch.setattr(color_bed,'depth_pose',lambda *a:dict(B_cam=np.eye(4)))
    monkeypatch.setattr(preview,'decoded',lambda *a:np.zeros((2,2,3)))
    monkeypatch.setattr(color_bed,'solve_pose',lambda *a:dict(B_cam=np.eye(4)))
    adapter.color_observer=NS(latest=lambda now:None)
    g=BedPoseGuide();g.reference=np.eye(4)
    assert not g.step(adapter,side,None,None,tracker,resume=True) and not calls
    g.correction=np.eye(4)
    assert g.step(adapter,side,None,None,tracker,resume=True) and len(calls)==1
    monkeypatch.setattr(color_bed,'depth_pose',lambda *a:None)
    assert not g.step(adapter,side,None,None,tracker,resume=True) and len(calls)==1


def test_color_resume_cannot_create_a_plane_without_depth():
    from app2.scan.color_bed import depth_pose
    assert depth_pose(np.zeros((600,800,3),np.uint8),None,np.zeros((600,800),np.uint16),(700,700,400,300),None) is None


def learner_for_gate():
    from app2.scan.color_bed import ColorBedLearner
    l=ColorBedLearner((700,700,400,300),radius_mm=100)
    dots=np.array([[i*10,j*10] for i in range(4) for j in range(3)],float)
    l.turned=lambda:100.;l._kept=lambda:(dots,np.ones(len(dots)),np.zeros(len(dots)))
    l.full_views=[(np.radians(a),dots.copy(),np.zeros(2)) for a in (0,45,100)]
    def check():l.map_check=dict(dots=12,seen=1.,halves=1.)
    l._check_map=check
    return l


def test_color_completion_uses_three_rotated_full_views_per_dot():
    l=learner_for_gate();assert l.ready()
    assert l.fast_check['minimum_dot_views']==3
    l.full_views[-1]=(l.full_views[-1][0],l.full_views[-1][1][:-1],np.zeros(2))
    assert not l.ready()


@pytest.mark.parametrize('missing',['color','rotation','map_check'])
def test_fast_learning_keeps_the_full_turn_fallback(missing):
    l=learner_for_gate()
    if missing=='color':l.full_views=[]
    if missing=='rotation':l.turned=lambda:20.
    if missing=='map_check':l._check_map=lambda:setattr(l,'map_check',dict(seen=.1,halves=.2))
    assert not l.ready()
    l.turned=lambda:370.;assert l.ready()


def test_cropped_color_bed_does_not_certify_the_whole_map():
    from app2.scan.color_bed import ColorBedLearner
    l=ColorBedLearner((700,700,400,300),radius_mm=100)
    l.c_est=np.zeros(2);l.plane=(np.array([0,0,-1.]),300.);l.e=(np.array([1,0,0.]),np.array([0,1,0.]))
    K=np.array([[700,0,400],[0,700,300],[0,0,1.]])
    assert l._full_color_view((K,np.eye(3),np.zeros(3)),(600,800,3))
    K[0,2]=700
    assert not l._full_color_view((K,np.eye(3),np.zeros(3)),(600,800,3))
    l.radius_mm=None
    assert not l._full_color_view((K,np.eye(3),np.zeros(3)),(600,800,3))
