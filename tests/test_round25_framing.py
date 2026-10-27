import json
from types import SimpleNamespace as NS
import numpy as np
from app2.scan.framing import analyze,Framing
from app2.scan.range_guide import from_scene,SteadyGuide


def scene(offset=0,z=3000):
    d=np.full((80,100),4000,np.uint16);d[25:55,30+offset:70+offset]=z
    return d,(100.,100.,50.,40.),(np.array([0.,0.,-1.]),400.,np.array([0.,0.,400.]),60.)


def test_part_plane_excludes_mat_and_unconnected_background():
    d,cal,geo=scene();d[0:20,0:30]=3000
    part=analyze(d,cal,geo,(0,1000))
    assert part['source']=='part' and part['points']==1200
    assert np.allclose(part['rect'],[30,25,69,54])
    assert part['background_pixels']==6800 and part['centered'] and not part['clipped']
    assert np.count_nonzero(d)==8000


def test_no_bed_is_labeled_coverage_and_empty_mat_is_not_a_part():
    d,cal,geo=scene();coverage=analyze(d,cal,None,(0,1000));assert coverage['source']=='coverage'
    empty=analyze(np.full_like(d,4000),cal,geo,(0,1000));assert empty['rect'] is None and empty['points']==0
    f=Framing(3);f.update(coverage,from_scene(d,'near',coverage),0);assert f.union["source"]=="coverage" and "locked" not in f.update(coverage,None,1)


def test_connected_overhang_is_preserved():
    d,cal,geo=scene();geo=(*geo[:3],10.)
    part=analyze(d,cal,geo,(0,1000));assert part['points']==1200


def test_steady_window_keeps_following_and_never_freezes_the_box():
    d,cal,geo=scene();f=Framing(3)
    for t in (0,.5,1,1.5,2,2.5,3,3.5):
        part=analyze(d,cal,geo,(0,1000));view=f.update(part,from_scene(d,'near',part),t)
    assert f.steady and 'locked' not in view
    frozen=view['rect'];depth=view['depth_mm']
    d2,_,_=scene(offset=1,z=3010);part=analyze(d2,cal,geo,(0,1000));view=f.update(part,from_scene(d2,'near',part),4)
    assert view['rect'][2]>frozen[2] and view['observed_range_mm']==[300,301]
    f.reset();assert f.union is None


def test_663_mm_with_good_depth_is_not_too_far():
    d,cal,_=scene(z=6630);geo=(np.array([0.,0.,-1.]),800.,np.array([0.,0.,800.]),300.)
    d[d==4000]=8000;part=analyze(d,cal,geo,(0,1000));g=from_scene(d,'near',part)
    assert g['center_mm']==663 and g['state']=='strong depth'
    assert not g['reference_distance_observed'] and g['observed_distance']
    # A good current frame does not invent a new distance band beyond the
    # measured profile. It remains a fill observation with its own state.
    assert all(not x.get('live') for x in g['zones'])
    assert all(x['hi']<663 for x in g['zones'])
    f=Framing(3);view=f.update(part,g,0);assert view['quality']==g['state']


def test_range_action_is_undoable_without_a_lock(tmp_path,monkeypatch):
    from app2.scan.session import ScanSession
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    s=object.__new__(ScanSession);s.busy=None;s.building=False;s.recording=False;s.bed=NS(learning=False)
    s.near=100;s.far=450;s.auto_range=True;s.gain=7;s._sync_scan_geometry=lambda:None;s._emit=lambda *a:None;s._emit_state=lambda **k:None
    (tmp_path/'side').mkdir();s.picked=NS(folder=str(tmp_path/'side'),scan_profile={});s._ensure_framing()
    d,cal,geo=scene();part=analyze(d,cal,geo,(100,450));s.framer.update(part,from_scene(d,'near',part),0)
    s._use_frame_range();assert (s.near,s.far,s.auto_range)==(295,305,False)
    s._undo_frame_range();assert (s.near,s.far,s.auto_range)==(100,450,True)
    assert not (tmp_path/'side/property.rvproj').exists(), 'Framing changes never save a lock'


def test_quality_state_does_not_blink_on_a_single_dropout():
    d,cal,geo=scene();s=analyze(d,cal,geo,(0,1000));good=from_scene(d,'near',s);hold=SteadyGuide(3)
    for t in (0,.3,.6):g=hold.update(good,t)
    empty=analyze(np.zeros_like(d),cal,geo,(0,1000));bad=from_scene(np.zeros_like(d),'near',empty)
    assert hold.update(bad,.9)['state']==good['state']


def test_reset_restarts_the_steady_window_and_live_colors_are_held():
    d,cal,geo=scene();part=analyze(d,cal,geo,(0,1000));g=from_scene(d,'near',part);f=Framing(3)
    for t in range(5):f.update(part,g,t)
    assert f.steady
    f.reset()
    for t in (5,5.3):
        f.update(part,g,t);assert not f.steady
    hold=SteadyGuide(3)
    for t in (0,.3,.6):hold.update(g,t)
    sparse=dict(g,state='sparse depth',zones=[dict(lo=290,hi=310,state='sparse depth',live=True)])
    held=hold.update(sparse,.9)
    assert held['zones'][0]['state']==held['state']==g['state']


def test_reframe_preserves_the_pending_range_undo():
    from app2.scan.session import ScanSession
    s=object.__new__(ScanSession);s.recording=False;s.busy=None;s.building=False;s.picked=None
    s._ensure_framing();s.frame_range_undo=(100,450,True);s._emit_state=lambda **kw:None
    s._reframe();assert s.frame_range_undo==(100,450,True) and s.framer.union is None


def test_distance_bars_show_the_middle_object_not_the_room_without_a_bed():
    """2026-10-25: no bed known: the histogram follows the depth cluster in the middle, so a far wall does not fill the bars."""
    from app2.scan.range_guide import foreground_guess
    d,cal,geo=scene()                                     # part at 300 mm in the middle, background at 400 mm everywhere else
    m=foreground_guess(d);assert m is not None and m[40,50] and not m[5,5]
    coverage=analyze(d,cal,None,(0,1000));g=from_scene(d,'near',coverage)
    edges=np.array(g['edges']);counts=np.array(g['counts']);mid=(edges[:-1]+edges[1:])/2
    assert counts[mid>350].sum()==0 and counts[(mid>290)&(mid<310)].sum()>0
