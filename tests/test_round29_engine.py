"""Live fusion evidence, reversible updates, explicit bands, and measured surface guards."""
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest


def field(**kw):
    from app2.scan.live_build import QualityField
    return QualityField(1,dict(observed_distance_mm=[100,500]),**kw)


def test_live_repeats_keep_only_until_saved_support_target():
    assert field().target==12
    f=field(support=3,min_share=0);p=np.array([[0,0,300],[1,0,300]],float);z=np.full(2,300.)
    assert [f.observe(p,z)['keep'] for _ in range(6)]==[True,True,True,False,False,False]
    assert f.state()['seen']==6 and f.state()['kept']==3 and f.state()['pass_done']
    assert np.array_equal(f.levels(p)[1],[2,2])


def test_live_new_surface_and_better_distance_upgrade_without_dilution():
    f=field(support=1,min_share=0);p=np.array([[0,0,300]],float)
    assert f.observe(p,[110])['keep'];assert f.observe(p,[300])['upgrades']==1
    assert not f.observe(p,[110])['keep'] and f.levels(p)[0][0]==2
    assert f.observe(p+3,[300])['new']==1


def test_live_refuses_unsure_tracking_and_records_tunable_rule(tmp_path):
    f=field(support=2,min_share=.5);p=np.zeros((1,3))
    assert not f.observe(p,[300],ok=False)['keep'];assert not len(f.keys)
    f.save(tmp_path);saved=json.loads((tmp_path/'live-build.json').read_text());assert saved['refused']==1 and saved['min_share']==.5 and saved['support_target']==2


def test_detail_uses_measured_noise_and_pixel_pitch():
    from app2.scan.live_build import detail_levels
    from app2.scan.surface import profiles
    for mode in ('near','far'):
        values=detail_levels(mode);p=profiles(mode)
        assert values['Fine']>=max(2*p['noise_sigma_mm'],p['pixel_pitch_mm'])
        assert values['Balanced']==2*values['Fine'] and values['Coarse']==4*values['Fine']


def test_recent_fusion_undo_removes_unsaved_repeats_exactly():
    from app2.scan.live_build import RecentFusion
    class Tracker:
        def __init__(self):self.values=np.zeros(3);self.T=np.eye(4);self.fused=0;self.n=0;self.model=None
        def snapshot(self):return self.values.copy()
        def restore(self,s):self.values=s.copy()
    trk=Tracker();undo=RecentFusion(3,6)
    for i in range(4):
        undo.before(trk,i,[],[]);trk.values+=np.array([i+1,2,3]);trk.fused+=1;trk.n+=1
    undo.restore(trk)
    assert np.array_equal(trk.values,np.zeros(3)) and trk.fused==0 and trk.n==0


def test_manual_light_ownership_survives_auto_pause(tmp_path,monkeypatch):
    from app2.scan.session_live_build import LiveBuildSession
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    s=LiveBuildSession();s._init_live_build();s.source=NS(has_led=True);calls=[];s._set_led=calls.append;s._emit_state=lambda **k:None
    s.recording=True;s.light_during_scans=True;s._manual_led=True;s._start_scan_light();s.recording=False;s._sync_scan_light();assert calls==[]
    s._manual_led=False;s.recording=True;s._start_scan_light();s.recording=False;s._sync_scan_light();assert calls==[True,False]
    s._set_light_during_scans(True);from app2.scan.settings import load
    assert load()['light_during_scans']


def test_finished_range_test_does_not_apply_without_command():
    from app2.scan.session_guides import ScanGuides
    s=ScanGuides();profile=dict(complete=True,camera_pair_confirmed=True,bins=[{}]);s._range_sweep=NS(close=lambda:profile);old=object();s.range_profile=old;s._log=lambda *_:None;s._emit_state=lambda **_:None
    s._finish_range_test();assert s.range_profile is old and s.range_test_result is profile


def test_range_requires_repeated_frames_per_bin_and_half_shipped_span():
    from app2.scan.range_test import result
    def sample(z):return dict(accepted=True,distance_mm=z,fill=.9,plane_rms_mm=.1,dots=20,dot_rms_mm=.1,overlay_error_px=1,camera_match='near')
    rows=[sample(z) for z in range(170,401,5)]
    assert not result(rows,'near','cal')['complete']
    assert result(rows*3,'near','cal')['complete']
    assert not result([sample(300),sample(305),sample(310)]*3,'near','cal')['complete']


def test_apply_rejects_wrong_calibration_and_accepts_complete_supported_result(tmp_path,monkeypatch):
    from app2.scan.session_guides import ScanGuides
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'));s=ScanGuides();s.recording=False;s.scan_mode='near';s._guide_calibration_key='expected';s._sync_scan_geometry=lambda:None;s._log=lambda *_:None;s._emit_state=lambda **_:None
    p=dict(complete=True,camera_pair_confirmed=True,mode='near',calibration_sha256='wrong',observed_distance_mm=[170,400],bins=[dict(frames=3)]*47)
    s.range_test_result=p;s.range_profile=None;s._apply_range_test();assert s.range_profile is None
    p['calibration_sha256']='expected';s._apply_range_test();assert s.range_profile is p


def test_surface_detects_dark_shiny_and_uniform_without_switching():
    from app2.scan.surface import hint
    d=np.full((100,100),3000,np.uint16);mask=np.ones(d.shape,bool);ir=np.full(d.shape,30,np.uint8);scene=dict(source='part',mask=mask)
    assert hint(d,ir,scene)['kind']=='good'
    d[30:60,30:60]=0;ir[30:60,30:60]=0;mask[30:60,30:60]=False
    assert hint(d,ir,scene)['kind']=='dark'
    ir[20:60,20:60]=255;assert hint(d,ir,scene)['kind']=='shiny'


def test_sparse_part_leaves_range_to_the_histogram_and_connected_span_is_never_cut():
    from app2.scan.surface import safe_range
    s=dict(range_mm=[390,460],noise_mm=.4,fill=.05)
    assert safe_range(s,'near',(100,600)) is None            # too little depth to say: the caller's histogram decides
    assert safe_range(dict(fill=1),'near',(100,600)) is None  # no span at all: same
    s['fill']=1;s['range_mm']=[300,300];a,b=safe_range(s,'near',(100,600));assert b-a>=6*.4-1e-10


def test_single_stereo_edge_does_not_enclose_dark_background():
    from app2.scan.depth_footprint import DepthFootprint
    yy,xx=np.indices((100,140));edge=12+yy/5;d=np.where(xx>=edge,3000,0).astype(np.uint16);d[30:60,50:90]=0
    f=DepthFootprint(3);outline=None
    for i in range(5):outline=f.update(d,dict(depth_mm=300,noise_mm=.1),i/3)
    assert outline is not None and len(outline)==100 and not np.array_equal(outline[0],outline[-1])
    assert np.max(np.abs(outline[:,0]-(12+outline[:,1]/5)))<2
    assert f.update(np.zeros_like(d),{},9) is None


def test_rim_markers_learn_single_start_and_pair_opposite():
    from app2.scan.rim import RimMarkers
    m=RimMarkers()
    for angle in (0,15,30,45):m.observe([dict(angle_deg=angle+20,reference_deg=0,height_mm=-8)],angle,True)
    learned=m.finish(.1);assert learned and np.isclose(learned['offset_deg'],20) and learned['band_mm'][0]<-8<learned['band_mm'][1]
    assert np.isclose(m.observe([dict(angle_deg=250,reference_deg=180,height_mm=-8)])['angle_deg'],50)
    assert m.observe([]) is None


def test_rim_fields_round_trip_without_changing_learned_outline():
    from scanner.bed import BedModel
    b=BedModel(radius_mm=100,rim=dict(radius_mm=102,spread_mm=.2),rim_markers=dict(offset_deg=5,band_mm=[-9,-7]));j=b.to_json();other=BedModel.from_json(j)
    assert other.rim==b.rim and other.rim_markers==b.rim_markers and other.radius_mm==100


def test_color_extension_marks_the_part_without_a_convex_depth_envelope(monkeypatch):
    from app2.scan.color_surface import ColorSurface
    from app2.scan import preview
    monkeypatch.setattr(preview,'RGB_CAL',(np.array([[100,0,70],[0,100,50],[0,0,1]]),np.eye(3),np.zeros(3)));monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    rgb=np.full((100,140,3),180,np.uint8);d=np.full((100,140),4000,np.uint16);mask=np.zeros(d.shape,bool);mask[20:80,60:100]=True;d[mask]=3000;rgb[mask]=30
    rgb[35:65,20:60]=30;d[35:65,20:60]=0
    helper=ColorSurface();result=None;context=dict(bed_plane=dict(normal=[0,0,-1],offset_mm=400,center_mm=[0,0,400],radius_mm=200))
    for i in range(4):result=helper.update(rgb,d,mask,(100,100,70,50),i/3,context)
    assert result is not None and result['pixels']>500
    assert result['mask'][40,30] and not result['mask'][5,5]
    rgb[35:65,20:60]=180
    current=helper.update(rgb,d,mask,(100,100,70,50),4/3,context)
    assert current is None or not current['mask'][40,30], 'An earlier silhouette must not persist after current color evidence disappears'


def test_missing_black_mat_depth_is_not_mislabeled_as_a_same_color_part(monkeypatch):
    import cv2
    from app2.scan.color_surface import ColorSurface
    from app2.scan import preview
    cal=(200.,200.,120.,100.);monkeypatch.setattr(preview,'RGB_CAL',(np.array([[200.,0,120],[0,200.,100],[0,0,1]]),np.eye(3),np.zeros(3)));monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    rgb=np.full((200,240,3),180,np.uint8);mat=np.zeros(rgb.shape[:2],np.uint8);cv2.circle(mat,(120,125),45,1,-1);rgb[mat.astype(bool)]=40
    depth=np.full(mat.shape,4000,np.uint16);depth[mat.astype(bool)]=0;part=np.zeros(mat.shape,bool);part[50:125,100:140]=True;depth[part]=3000
    rng=np.random.default_rng(0);values=np.clip(rng.normal(40,4,int(part.sum())),0,255).astype(np.uint8);rgb[part]=values[:,None]
    scene=dict(bed_plane=dict(normal=[0,0,-1],offset_mm=400,center_mm=[0,50,400],radius_mm=90))
    generic=ColorSurface();aware=ColorSurface()
    for i in range(4):
        false_candidate=generic.update(rgb,depth,part,cal,i/3);unknown=aware.update(rgb,depth,part,cal,i/3,scene)
    assert false_candidate is not None and false_candidate['pixels']>500
    assert unknown is None, 'No independent color contrast from the actual mat means unknown'


def test_three_far_apart_range_holds_do_not_claim_the_gaps():
    from app2.scan.range_test import result
    rows=[dict(accepted=True,distance_mm=z,fill=.9,plane_rms_mm=.1,dots=20,camera_match='near') for z in (170,300,450)]*5
    assert not result(rows,'near','cal')['complete']


def test_measured_rim_fit_holds_radius_over_noisy_frames():
    from app2.scan.rim import RimFit
    yy,xx=np.indices((160,200));cal=(200.,200.,100.,80.);z=np.full(xx.shape,400.);x=(xx-100)*z/200;y=(yy-80)*z/200;disk=x*x+y*y<=100**2
    depth=np.where(disk,4000,0).astype(np.uint16);fit=RimFit(3);geo=(np.array([0.,0.,-1.]),400.,np.array([0.,0.,400.]),99.)
    rng=np.random.default_rng(0)
    for i in range(6):
        noisy=depth.copy();noisy[disk]+=rng.integers(-1,2,int(disk.sum()),dtype=np.int16).astype(np.uint16);fit.update(noisy,cal,geo,.1,i/3)
    pitch=400/min(cal[:2]);uncertainty=3*np.hypot(.1,pitch/np.sqrt(12))
    assert fit.result is not None and abs(fit.result['radius_mm']-100)<=uncertainty and fit.result['spread_mm']<=uncertainty


def test_rim_rejects_interior_boundaries_in_occluded_sectors():
    from app2.scan.rim import RimFit
    yy,xx=np.indices((220,220));x=(xx-110)*2.;y=(yy-110)*2.;radius=np.hypot(x,y);theta=np.arctan2(y,x)
    disk=(radius<=100)&((theta>-.3)|(radius<75))
    depth=np.where(disk,4000,0).astype(np.uint16);fit=RimFit(3);geo=(np.array([0.,0.,-1.]),400.,np.array([0.,0.,400.]),99.5)
    for i in range(6):fit.update(depth,(200.,200.,110.,110.),geo,.1,i/3)
    assert fit.result is not None and abs(fit.result['radius_mm']-100)<3
    interior=np.where(radius<85,4000,0).astype(np.uint16);unknown=RimFit(3)
    for i in range(6):unknown.update(interior,(200.,200.,110.,110.),geo,.1,i/3)
    assert unknown.result is None, 'Visible interior must not shrink the learned rim'


def test_synthetic_rim_images_detect_one_and_adjacent_pair_below_the_plane():
    import cv2
    from app2.scan.rim import detect_markers,RimMarkers
    cal=(400.,400.,320.,240.);geo=(np.array([0.,0.,-1.]),400.,np.array([0.,0.,400.]),100.)
    def image(angles):
        d=np.zeros((480,640),np.uint16);ir=np.zeros(d.shape,np.uint8)
        for angle in angles:
            a=np.radians(angle);u=round(320+400*100*np.cos(a)/408);v=round(240-400*100*np.sin(a)/408)
            cv2.circle(ir,(u,v),1,255,-1);cv2.circle(d,(u,v),1,4080,-1)
        return d,ir
    learned=RimMarkers()
    for _ in range(3):
        d,ir=image([35]);markers=detect_markers(d,ir,cal,geo,.1,6)
        assert len(markers)==1 and markers[0]['reference_deg']==0
        learned.observe(markers,top_angle=20,learn=True)
    assert learned.finish(.1) is not None
    d,ir=image([213,217]);markers=detect_markers(d,ir,cal,geo,.1,6,learned.learned['band_mm'])
    assert len(markers)==1 and markers[0]['reference_deg']==180
    assert abs(learned.observe(markers)['angle_deg']-20)<1
    assert detect_markers(*image([]),cal,geo,.1,6)==[]
