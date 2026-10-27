import numpy as np
from types import SimpleNamespace as NS
from app2.scan.overlay_alignment import apply,refine,manual_matrix
from app2.scan.session_guides import ScanGuides


def fixture():
    p=np.array([[20,20],[100,20],[180,20],[20,100],[100,100],[180,100],[100,180]],float)
    M=np.eye(3);M[:2,2]=[2,-1]
    return dict(accepted=True,projected_points=p.tolist(),circle_points=apply(p,M).tolist(),radii=[5]*len(p))


def test_bounded_good_dot_fit_reports_before_and_after():
    fit=refine(fixture());assert fit and fit['before_px']>2 and fit['after_px']<1e-8 and fit['matches']==7


def test_alignment_rejects_insufficient_degenerate_or_excessive_matches():
    c=fixture();c['radii']=[1]*7;assert refine(c) is None
    c=fixture();c['projected_points']=c['projected_points'][:3];c['circle_points']=c['circle_points'][:3];assert refine(c) is None
    c=fixture();c['accepted']=False;assert refine(c) is None


def test_wrong_mode_or_contrary_votes_cannot_be_hidden_by_fine_tune():
    class S(ScanGuides):pass
    s=S();s.scan_mode='near';row=dict(color=fixture())
    s._refine_overlay(row,dict(mode='far',votes={'far':3,'near':0}));assert not hasattr(s,'_overlay_auto')
    s._refine_overlay(row,dict(mode='near',votes={'far':1,'near':3}));assert not hasattr(s,'_overlay_auto')


def test_manual_nudge_is_per_calibration_and_reset_is_identity(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    from app2.scan import preview
    monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    class S(ScanGuides):
        def _emit_state(self,**kw):pass
        def _log(self,text):pass
    s=S();s._guide_calibration_key='one';s._overlay_auto=None;s._set_overlay(dict(x=3,y=-2,angle=0,scale=1))
    assert np.allclose(apply([[400,300]],preview.OVERLAY_MATRIX),[[403,298]])
    from app2.scan.settings import load
    assert load()['overlay_nudges']['one']['x']==3
    s.scan_mode='near';s._setup_guide_calibration(dict(source=dict(calibration_sha256='one')))
    assert s._overlay_manual['x']==3
    s._setup_guide_calibration(dict(source=dict(calibration_sha256='two')));assert s._overlay_manual=={}
    s._reset_overlay();assert np.array_equal(preview.OVERLAY_MATRIX,np.eye(3))
