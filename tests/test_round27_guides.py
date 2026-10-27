import json
import numpy as np
from app2.scan.range_guide import band_edges,distance_levels,distance_zones
from app2.scan.capture_distance import query


def test_fallback_best_is_the_requested_middle_40_percent_and_measured_bands_win():
    assert band_edges(100,600)==(100,250,450,600)
    p=dict(observed_distance_mm=[100,600]);assert distance_levels([99,100,249,250,450,451,600,601],p).tolist()==[0,1,1,2,2,1,1,0]
    p['distance_bands']=[dict(lo=115,hi=135,state='strong depth')]
    assert distance_levels([120,350],p).tolist()==[2,0]


def test_reconstructed_surface_matches_saved_support_across_voxel_boundary_only():
    field=(np.array([[0,0,0],[10,0,0]]),np.array([2,0],np.uint8),.5)
    P=np.array([[.51,.25,.25],[5.01,.25,.25],[2.5,0,0]])
    assert query(field,P).tolist()==[255,0,255]
    from app2.scan.capture_distance import surface_query
    assert surface_query(field,P).tolist()==[2,0,255]


def test_manual_nudge_updates_measured_error_without_moving_depth_or_poses(tmp_path,monkeypatch):
    from app2.scan import preview
    from app2.scan.session_guides import ScanGuides
    from test_round26_alignment import fixture
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'settings.json'));monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    class S(ScanGuides):
        def _emit_state(self,**kw):pass
        def _log(self,text):pass
    s=S();s.scan_mode='near';s._guide_calibration_key='test';s._overlay_manual={};s._overlay_auto=None
    s._refine_overlay(dict(color=fixture()),dict(mode='near',votes={'near':3,'far':0}))
    assert s._overlay_fit['after_px']<1e-8
    s._set_overlay(dict(x=1,y=0,angle=0,scale=1))
    assert np.isclose(s._overlay_fit['after_px'],1)
    assert s._overlay_fit['before_px']>2


def test_old_side_with_saved_mode_can_restore_seen_distance_evidence(tmp_path,monkeypatch):
    from app2.scan import capture_distance as c,fmt
    from app2.scan.scan_mode import write
    folder=tmp_path/'side';folder.mkdir();(folder/'cache').mkdir();(folder/'cache'/fmt.TRACK_JSON).write_text(json.dumps({'stamps':[]}))
    write(folder,dict(mode='near'));entries=[(0,0,np.eye(4))];calls=[]
    monkeypatch.setattr(fmt,'read_side',lambda path:(entries,None))
    import fuse
    monkeypatch.setattr(fuse,'read_calib',lambda path:(1,1,0,0));monkeypatch.setattr(c,'build_side',lambda *args:calls.append(args))
    c.ensure_side(folder)
    assert len(calls)==1 and calls[0][4]=='near' and calls[0][1] is entries


def test_fallback_policy_change_invalidates_saved_distance_field(tmp_path,monkeypatch):
    from app2.scan import capture_distance as c
    (tmp_path/'cache').mkdir();np.uint16([3000]).tofile(tmp_path/'cache/frame_000_0000.dph')
    entries=[(0,0,np.diag([1.,-1.,-1.,1.]))]
    monkeypatch.setattr(c,'measured_profile',lambda mode:dict(observed_distance_mm=[100,500]))
    f=c.build_side(tmp_path,entries,(1,1,0,0),(1,1),'near');assert query(f,[[0,0,300]])[0]==2
    import app2.scan.range_guide as guide
    monkeypatch.setattr(guide,'band_edges',lambda lo,hi:(lo,400,450,hi))
    f=c.build_side(tmp_path,entries,(1,1,0,0),(1,1),'near');assert query(f,[[0,0,300]])[0]==1


def test_better_neighboring_surface_view_upgrades_an_existing_red_voxel():
    from app2.scan.capture_distance import surface_query
    field=(np.array([[0,0,0],[1,0,0]]),np.array([0,2],np.uint8),.5)
    assert surface_query(field,[[.49,.25,.25]])[0]==2


def test_build_uses_facet_centers_and_keeps_unsupported_hole_gray(tmp_path,monkeypatch):
    from app2.builder.capture_distance import colors
    from app2.builder import view_data
    import app2.builder.capture_distance as capture
    side=tmp_path/'data'/'saved';side.mkdir(parents=True)
    with (side/'capture-distance.npz').open('wb') as f:np.savez(f,keys=np.array([[0,0,0]]),levels=np.array([2],np.uint8),step=.5)
    V=np.float32([[0,0,0],[.5,0,0],[0,.5,0],[10,0,0],[10.5,0,0],[10,.5,0]]);F=np.uint32([[0,1,2],[3,4,5]])
    monkeypatch.setattr(view_data,'load_file',lambda path:dict(V=V,F=F));monkeypatch.setattr(capture,'ensure_side',lambda path:None)
    C,counts=colors(tmp_path,'fixture',placements={'saved':np.eye(4)})
    from app2.scan.capture_distance import rgb
    assert np.array_equal(C,rgb(np.uint8([2,255]))) and counts['unmeasured']==1


def test_build_loader_preserves_facet_bands_instead_of_first_vertex_or_patch_color(tmp_path,monkeypatch):
    from app2.builder.view_data import ViewLoader
    from app2.scan.capture_distance import rgb
    path=tmp_path/'fixture.ply';path.touch();np.save(tmp_path/'guessed.npy',np.packbits([True,False]))
    V=np.float32([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]);F=np.uint32([[0,1,2],[1,3,2]]);N=np.tile([0,0,1],(4,1)).astype(np.float32)
    loader=ViewLoader(lambda data:None)
    try:
        monkeypatch.setattr(loader,'_get',lambda path:dict(kind='mesh',V=V,N=N,F=F))
        C=rgb(np.uint8([2,255]));out=loader._load(dict(items=[dict(key='main',path=str(path),rgb=C)],mode='capture_distance',edit=None,check_mem=False))
        assert np.array_equal(out['items'][0]['face_rgb'],C)
    finally:loader.close()


def test_live_aim_is_a_plain_plus_and_center_marks_are_optional_without_labels(monkeypatch):
    from app2.scan import preview
    monkeypatch.setattr(preview,'RGB_CAL',(np.eye(3),np.eye(3),np.zeros(3)))
    monkeypatch.setattr(preview,'SHOW_PART_CENTER',True)
    color=(79,196,111);area=dict(center=np.array([100,100]),part_center=np.array([75,75]),outline=np.array([[30,30],[180,30],[180,180],[30,180],[30,30]]),color=color,label='Part')
    image=preview.depth_guides(np.zeros((600,800,3),np.uint8),np.zeros((600,800),np.uint16),(500,500,400,300),300,color,area)
    assert np.array_equal(image[75,75],color)
    assert np.allclose(image[100,100],color,atol=1),'The aim remains a plain plus'
    assert np.all(image[100,120:175]==0),'No long cross line across the part box'
    from app2.scan.framing import Framing
    from app2.scan.range_guide import from_scene
    c=object.__new__(preview.Composer);c.cal=(500,500,400,300);c.framing=dict(source='part',rect=[100,100,500,450],depth_mm=300,depth_size=(800,600),observed_range_mm=[290,310],state='off center',hint='Bring the dot to the aim mark',quality='usable depth',locked=False)
    assert 'label' not in c.guide_area()
    c.framing['locked']=True;assert 'label' not in c.guide_area(), 'An old lock cannot reintroduce image words'
