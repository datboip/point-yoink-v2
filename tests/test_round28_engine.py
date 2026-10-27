"""Regressions for the following box, measured footprint, IR pair and retained turns."""
import json
import struct
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from app2.scan.framing import analyze,Framing
from app2.scan.range_guide import from_scene


def scene(offset=0,z=3000):
    d=np.full((80,100),4000,np.uint16);d[25:55,30+offset:70+offset]=z
    cal=(100.,100.,50.,40.);geo=(np.array([0.,0.,-1.]),400.,np.array([0.,0.,400.]),60.)
    return d,cal,geo


def test_follow_box_grows_across_turn_and_range_uses_all_observed_depths():
    f=Framing(3)
    for t,(dx,z) in enumerate(((0,3000),(10,3100),(-10,2900),(0,3000))):
        d,cal,geo=scene(dx,z);s=analyze(d,cal,geo,(0,1000));v=f.update(s,from_scene(d,'near',s),t)
        assert v['rect'][0]<=s['rect'][0] and v['rect'][2]>=s['rect'][2]
    assert v['rect']==[20,25,79,54] and v['observed_range_mm']==[290,310]
    assert v['range_mm'][0]<290 and v['range_mm'][1]>310
    assert 'locked' not in v and not hasattr(f,'lock')


def test_big_scene_change_and_new_side_reset_follow_union():
    f=Framing(3);d,cal,geo=scene();s=analyze(d,cal,geo,(0,1000));f.update(s,None,0)
    moved=dict(s,rect=np.array([300.,25.,339.,54.]));v=f.update(moved,None,1)
    assert v['rect']==moved['rect'].tolist()
    f=Framing(3);f.update(s,None,0)
    d2,_,_=scene(z=9000);d2[d2==4000]=10000
    far=analyze(d2,cal,(geo[0],1000.,np.array([0.,0.,1000.]),200.),(0,2000))
    view=f.update(far,None,2);assert view["observed_range_mm"]==[900,900]
    f.reset();assert f.union is None


def test_old_side_lock_fields_are_ignored_and_not_written_again(tmp_path):
    from app2.scan.scan_mode import read,write
    path=tmp_path/'property.rvproj';old=dict(mode='near',framing_lock=dict(rect=[1,2,3,4]),framing_locks=[{}],near=100)
    path.write_text(json.dumps(dict(pointyoink_scan=old)))
    assert read(tmp_path)==dict(mode='near',near=100)
    write(tmp_path,old);assert 'framing_lock' not in path.read_text()


def test_curved_observed_depth_edge_is_inside_sensor_and_fills_internal_holes():
    from app2.scan.depth_footprint import DepthFootprint,assess
    import cv2
    d=np.zeros((100,140),np.uint16);yy,xx=np.indices(d.shape)
    support=((xx-70)**2/60**2+(yy-50)**2/45**2<=1);d[support]=3500
    d[30:70,45:95]=3000;d[35:65,55:85]=0
    cal=(100.,100.,70.,50.);geo=(np.array([0.,0.,-1.]),350.,np.array([0.,0.,350.]),150.)
    s=analyze(d,cal,geo,(0,1000));f=DepthFootprint(3)
    for i in range(3):edge=f.update(d,s,i/3)
    assert edge is not None and len(edge)>5
    assert edge[:,0].min()>0 and edge[:,0].max()<d.shape[1]-1
    assess(s,d,edge,np.zeros(d.shape,np.uint8))
    assert s['missing_depth_pixels']>=900 and s['missing_depth'] and not s['clipped']
    view=Framing(3).update(s,from_scene(d,'near',s),0)
    assert 'dark or shiny' in view['hint'] and 'past the depth view' not in view['hint']
    assert not np.array_equal(edge[0],edge[-1]),'The overlap edge is a single open boundary'


def test_footprint_expires_and_never_invents_full_sensor_edge():
    from app2.scan.depth_footprint import DepthFootprint
    d=np.zeros((80,100),np.uint16);d[10:70,20:80]=3000
    f=DepthFootprint(3)
    for i in range(3):edge=f.update(d,dict(range_mm=[290,310]),i/3)
    assert edge is not None
    assert f.update(np.zeros_like(d),dict(range_mm=[290,310]),4) is None


def test_follow_color_box_keeps_parallax_of_both_turn_depth_extremes(monkeypatch):
    from app2.scan import preview
    K=np.array([[500,0,400],[0,500,300],[0,0,1]],float)
    monkeypatch.setattr(preview,'RGB_CAL',(K,np.eye(3),np.array([25.,0,0])))
    monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    frame=dict(rect=[100,100,600,500],depth_mm=300,depth_size=[800,600],observed_range_mm=[200,400])
    box=preview.part_area((500,500,400,300),frame)['outline']
    for z in frame['observed_range_mm']:
        x,y=preview.depth_px_to_rgb(np.array([100,600]),np.array([100,500]),np.full(2,z),(500,500,400,300))
        assert box[:,0].min()<=x.min() and box[:,0].max()>=x.max()
        assert box[:,1].min()<=y.min() and box[:,1].max()>=y.max()


def test_right_ir_optional_file_is_exact_and_moves_with_undo(tmp_path):
    from app2.scan import fmt
    cache=tmp_path/'cache';cache.mkdir();d=np.full((600,800),3000,np.uint16);l=np.zeros(d.shape,np.uint8);r=np.full(d.shape,123,np.uint8)
    fmt.write_frame(str(cache),'frame_000_0000',d,l,np.eye(4),irr=r)
    assert (cache/'frame_000_0000.irr').stat().st_size==480000
    assert np.array_equal(np.fromfile(cache/'frame_000_0000.irr',np.uint8).reshape(d.shape),r)
    fmt.move_frames(str(cache),str(tmp_path/'discarded'),['frame_000_0000'])
    assert not (cache/'frame_000_0000.irr').exists()
    fmt.move_frames(str(tmp_path/'discarded'),str(cache),['frame_000_0000'])
    assert (cache/'frame_000_0000.irr').is_file()
    fmt.write_frame(str(cache),'frame_000_0001',d,l,np.eye(4))
    assert not (cache/'frame_000_0001.irr').exists()


def test_rectified_right_projection_and_both_use_depth_not_rgb_baseline():
    from app2.scan.stereo_view import Stereo
    s=Stereo((400,400,100,50),20,100)
    assert np.allclose(s.right([[80,40],[80,40]],[200,400]),[[40,40],[60,40]])
    assert np.allclose(s.left(s.right([[80,40]],200),200),[[80,40]])
    l=np.zeros((100,200),np.uint8);r=np.zeros_like(l);l[:,80]=200;r[:,40]=200
    blend=s.blend(l,r,np.full(l.shape,2000,np.uint16),200)
    assert blend[50,80,2]>blend[50,80,0]          # the matched line: blue, the two cameras agree


def test_missing_right_ir_is_not_a_second_copy_of_left():
    from app2.scan.stereo_view import select_ir
    from app2.scan.preview import LiveView
    v=LiveView(ir=np.ones((10,10,3),np.uint8))
    assert select_ir(v,'Left')[0] is v.ir
    assert select_ir(v,'Right')==(None,None) and select_ir(v,'Both')==(None,None)


def test_turn_coverage_only_fills_kept_angles_and_keeps_best_band(tmp_path):
    from app2.scan.turn_coverage import snapshot,save
    from track import F
    (tmp_path/'param').mkdir();(tmp_path/'cache').mkdir()
    (tmp_path/'param/Pl.bin').write_bytes(struct.pack('<HH9f',800,600,400,0,400,0,400,300,0,0,1))
    bed=dict(state='found',B_scan=np.eye(4).tolist(),n=[0,0,1],off=-400)
    (tmp_path/'bed_pose.json').write_text(json.dumps(bed))
    side=NS(folder=str(tmp_path),cache=str(tmp_path/'cache'),entries=[],stamps=[],scan_profile=dict(mode='near'),bed=None)
    for i,(angle,z) in enumerate(((0,190),(16,500),(16,190),(32,190))):
        a=np.radians(angle);T=np.eye(4);T[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        side.entries.append((0,i,T@F));side.stamps.append(dict(frame=f'frame_000_{i:04d}',ok=True,part_distance_mm=z))
    side.stamps.append(dict(frame='frame_000_0004',ok=False,part_distance_mm=190))
    out=snapshot(side)
    assert out['frames']==4 and np.isclose(out['degrees'],32)
    assert sum(x>=0 for x in out['levels'])==3 and out['levels'][0]==2
    assert max(out['counts'])==2 and sum(x<0 for x in out['levels'])>0
    assert all(x==2 for x in out['levels'] if x>=0)
    save(side);assert json.loads((tmp_path/'turn-coverage.json').read_text())==out
    side.entries.pop();side.stamps.pop(-2)
    rewound=snapshot(side);assert rewound['frames']==3 and np.isclose(rewound['degrees'],16)


def test_preview_keeps_vector_marks_out_of_camera_pixels(monkeypatch):
    from app2.scan import preview
    monkeypatch.setattr(preview,'RGB_CAL',(np.array([[100,0,50],[0,100,40],[0,0,1]]),np.eye(3),np.zeros(3)))
    monkeypatch.setattr(preview,'PARAM_DIR',None)
    monkeypatch.setattr(preview,'OVERLAY_MATRIX',np.eye(3))
    c=preview.Composer((100,100,50,40),lambda *a:None)
    try:
        c.rgb=np.full((600,800,3),71,np.uint8);d,cal,geo=scene();s=analyze(d,cal,geo,(0,1000));v=Framing(3).update(s,None,0)
        v['depth_outline']=np.array([[5,0],[6,79]],float)
        frame=c.compose(d,np.zeros(d.shape,np.uint8),False,framing=v,guide_depth=np.zeros_like(d))
        assert np.array_equal(frame.colour,c.rgb)
        assert frame.colour_guides['box'] is not None and frame.ir_guides['depth_outline'] is not None
        assert 'label' not in frame.colour_guides
    finally:c.stop();c.thread.join(2)


def test_depthless_dark_extension_inside_observed_view_gets_a_cautious_hint():
    from app2.scan.depth_footprint import DepthFootprint,assess
    d,cal,geo=scene();ir=np.full(d.shape,150,np.uint8)
    d[24:56,5:35]=0;ir[24:56,5:35]=0
    part=analyze(d,cal,geo,(0,1000),ir=ir)
    edge=DepthFootprint(3).update(d,part,0);assess(part,d,edge,ir)
    assert not part['missing_extension_possible'],'IR alone cannot establish a missing color part extension'
    assert not part['clipped']
    view=Framing(3).update(part,from_scene(d,'near',part),0)
    assert 'past the depth view' not in view['hint']


def test_sensor_clipping_keeps_its_own_hint_even_when_there_are_interior_holes():
    d,cal,geo=scene();d[25:55,:40]=3000;d[30:50,5:35]=0
    part=analyze(d,cal,geo,(0,1000));part['missing_depth']=True
    view=Framing(3).update(part,from_scene(d,'near',part),0)
    assert 'past the depth view' in view['hint'] and 'missing inside' in view['hint']


def test_turn_keeps_gray_views_without_guessing_a_camera_mode(tmp_path):
    from app2.scan.turn_coverage import snapshot
    from track import F
    (tmp_path/'bed_pose.json').write_text(json.dumps(dict(state='found',B_scan=np.eye(4).tolist(),n=[0,0,1],off=-400)))
    side=NS(folder=str(tmp_path),cache=str(tmp_path/'cache'),entries=[(0,0,F.copy())],stamps=[dict(frame='frame_000_0000',ok=True,part_distance_mm=300)],scan_profile={},bed=None)
    out=snapshot(side)
    assert out['frames']==1 and out['levels'][0]==-2 and out['counts'][0]==1
    assert sum(x==-1 for x in out['levels'])==len(out['levels'])-1


def test_saving_an_old_side_removes_legacy_lock_fields(tmp_path):
    from app2.scan.project import Project,Side
    from test_app2_scan_rewind import saved_side
    project,side=saved_side(str(tmp_path),[True,True])
    prop=Path(side.folder)/'property.rvproj';data=json.loads(prop.read_text());data['pointyoink_scan']=dict(mode='near',framing_lock=dict(rect=[1,2,3,4]),framing_locks=[{}]);prop.write_text(json.dumps(data))
    reopened=Side(side.folder);reopened.dirty=True;project.sides=[reopened]
    project.save_side(reopened,log=lambda *a:None)
    assert 'framing_lock' not in prop.read_text()


def test_replay_delivers_saved_right_ir_and_none_for_an_old_frame(tmp_path):
    from test_app2_scan_rewind import replay_project
    from app2.scan.sources import ReplaySource
    project=replay_project(str(tmp_path),2)
    right=next(Path(project).glob('data/*/cache/frame_000_0001.irr'));right.unlink()
    source=ReplaySource(project,fps=0);frames=[]
    try:
        source.start(frames.append);source.join(3)
        assert source.done.is_set() and len(frames)==2
        assert np.all(frames[0].irr==123) and frames[1].irr is None
        assert np.all(frames[0].irl==0)
    finally:source.stop();source.join(2)


def test_part_does_not_prove_its_own_view_boundary_but_background_can():
    from app2.scan.depth_footprint import assess
    d,cal,geo=scene();part=analyze(d,cal,geo,(0,1000))
    own=np.array([[30,25],[69,25],[69,54],[30,54],[30,25]],float)
    assess(part,d,own)
    assert not part['clipped'],'A contour made only from this part is not an optical boundary'
    d[:,71:]=0;d[25:55,30:71]=3000;part=analyze(d,cal,geo,(0,1000))
    measured=np.array([[70,0],[70,79]],float)
    assess(part,d,measured)
    assert part['clipped']
    assert 'past what depth can see' in Framing(3).update(part,from_scene(d,'near',part),0)['hint']


def test_new_side_immediately_clears_preparation_bounds_and_previous_turn(tmp_path):
    from app2.scan.session import ScanSession
    from app2.scan.project import Project
    from test_app2_scan_rewind import param_dir
    params=param_dir(str(tmp_path));project=Project(str(tmp_path/'project'),params)
    session=object.__new__(ScanSession);session.project=project;session.picked=None
    session.recording=False;session.busy=None;session.building=False
    session.bed=NS(learning=False,side_opened=lambda *a:None)
    session._ensure_framing();d,cal,geo=scene();session.framer.update(analyze(d,cal,geo,(0,1000)),None,0)
    session.framing_view=dict(rect=[1,2,3,4]);session._forget_rewound=lambda:None
    session.save_color=True;session.autogain=NS(reset=lambda:None);session.source=NS(device=NS(gain=lambda value:None));session.gain=7
    events=[];session._emit=lambda kind,value:events.append((kind,value));session._emit_state=lambda **kw:None
    session.host=NS(release=lambda:None,new=lambda side:None);session.scan_mode='near';session.calibration_source=dict(declared_mode='near')
    session._reset_mode_detection=lambda **kw:None;session.cfg=NS(param_dir=params,w=800,h=600);session.cal=(500,500,400,300);session.near=100;session.far=450
    session._home=lambda:None;session._emit_cloud=lambda *a,**kw:None;session._emit_image=lambda *a:None;session._busy=lambda text,fn:fn()
    session._new_side()
    assert session.framer.union is None and session.framing_view is None
    turns=[value for kind,value in events if kind=='turn']
    assert turns and turns[-1]['coverage']['frames']==0 and all(x==-1 for x in turns[-1]['coverage']['levels'])


def test_late_bed_result_refreshes_saved_turn_and_selected_ring(tmp_path):
    from app2.scan.turn_coverage import save
    from app2.scan.bed_adapter import ScannerBed
    from app2.scan.session import ScanSession
    from track import F
    side=NS(folder=str(tmp_path),cache=str(tmp_path/'cache'),entries=[(0,0,F.copy())],stamps=[dict(frame='frame_000_0000',ok=True,part_distance_mm=300)],scan_profile=dict(mode='near'),bed=None)
    assert save(side)['frames']==0
    late=object.__new__(ScannerBed)
    def install(side,path):
        (tmp_path/'bed_pose.json').write_text(json.dumps(dict(state='found',B_scan=np.eye(4).tolist(),n=[0,0,1],off=-400)))
        return 'bed'
    late._write_platter=install;late._resave(side)
    assert json.loads((tmp_path/'turn-coverage.json').read_text())['frames']==1
    session=object.__new__(ScanSession);session.picked=side;session.project=NS(sides=[side]);session.bed=NS(drain=lambda *a:([],True),take_message=lambda:None)
    session._no=lambda side:1;session._log=lambda text:None;session._platter=lambda side:None;session._emit_sides=lambda:None
    events=[];session._emit=lambda kind,value:events.append((kind,value));session.gain=7
    session._drain_bed()
    assert [value for kind,value in events if kind=='turn'][-1]['coverage']['frames']==1


def test_unmeasured_points_take_their_band_from_the_camera_distance():
    from app2.scan.session import depth_band_fill
    from app2.scan.range_guide import measured_profile
    T=np.eye(4);T[:3,3]=[0,0,50]                       # camera 50 mm along z of the scan frame, looking down +z
    P=np.array([[0,0,240],[0,0,330],[0,0,900],[5,5,240]],float)
    B=np.array([255,255,255,1],np.uint8)
    out=depth_band_fill(P,B,T,measured_profile('near'))
    assert out[0]==2 and out[1]==1 and out[2]==0 and out[3]==1   # 190 best, 280 ok, 850 off, measured one kept
    assert np.array_equal(depth_band_fill(P,B.copy(),np.eye(4),measured_profile('near')),depth_band_fill(P,B,np.eye(4),measured_profile('near')))


def test_surface_presets_are_four_different_starts_backed_by_measured_gains():
    from app2.scan.surface import preset,profiles
    measured={g['gain'] for g in profiles('near')['gains']}
    starts={name:preset('near',name) for name in ('Normal','Dark','Shiny','Sprayed')}
    assert starts['Dark']['gain']>starts['Normal']['gain']>starts['Shiny']['gain']
    assert starts['Shiny']['auto_gain_bounds'][1]<=starts['Dark']['auto_gain_bounds'][0]
    for v in starts.values():assert v['gain'] in measured and v['evidence']['gain']==v['gain'] and v['auto_gain_bounds'][0]<=v['gain']<=v['auto_gain_bounds'][1]


def test_points_the_camera_sees_now_move_up_to_this_frames_band():
    from app2.scan.session import depth_band_fill
    from app2.scan.range_guide import measured_profile
    cal=(400,400,320,240);depth=np.zeros((480,640),np.uint16);depth[240,320]=1900   # the center pixel sees 190 mm
    P=np.array([[0,0,190],[0,0,190],[100,0,190]],float)      # seen at 190 (best); seen but tolerance miss; off the pixel
    depth2=depth.copy();depth2[240,320]=2600
    B=np.array([0,0,0],np.uint8)                                 # all first seen too close
    out=depth_band_fill(P,B,np.eye(4),measured_profile('near'),depth,cal)
    assert out[0]==2 and out[2]==0                               # seen now in best: green; not seen: kept
    assert depth_band_fill(P,B,np.eye(4),measured_profile('near'),depth2,cal)[0]==0   # the pixel shows something else: kept
    assert depth_band_fill(P,np.array([2,2,2],np.uint8),np.eye(4),measured_profile('near'),depth,cal).tolist()==[2,2,2]   # never down


def test_composer_builds_only_the_shown_ir_picture(tmp_path):
    from app2.scan.preview import Composer
    cal=(400,400,320,240);d=np.zeros((480,640),np.uint16);d[200:280,280:360]=2000;ir=np.full(d.shape,90,np.uint8)
    c=Composer(cal,lambda *a:None);c.set_scan_geometry(100,400,'near')
    class S:
        def right(self,p,z):return p
        def blend(self,l,r,d,z):return np.dstack([l,r,l])
    c.stereo=S()
    job=dict(d=d,irl=ir,irr=ir,lost_now=False)
    v=c.compose(**job);assert v.ir is not None and v.ir_right is None and v.ir_both is None
    c.ir_mode='Right';v=c.compose(**job);assert v.ir_right is not None and v.ir_both is None and v.right_guides is not None
    c.ir_mode='Both';v=c.compose(**job);assert v.ir_both is not None


def test_lockstep_color_moves_with_the_depth_frame_while_recording():
    import cv2
    from app2.scan.preview import Composer
    cal=(400,400,320,240);d=np.zeros((480,640),np.uint16);d[200:280,280:360]=2000;ir=np.full(d.shape,90,np.uint8)
    a=cv2.imencode('.jpg',np.full((48,64,3),40,np.uint8))[1].tobytes();b=cv2.imencode('.jpg',np.full((48,64,3),200,np.uint8))[1].tobytes()
    newest={'jpeg':b}
    c=Composer(cal,lambda *a:None,colour_fn=lambda:newest['jpeg']);c.set_scan_geometry(100,400,'near')
    c.color_source=type('S',(),{'latest_color':staticmethod(lambda:dict(newest,t=0.0))})()
    c.lockstep=True
    v=c.compose(d=d,irl=ir,lost_now=False,jpeg=a)
    assert v.colour is not None and int(v.colour.mean())<100           # the matched frame (dark), not the newest (bright)
    assert c.colour_only() is None                                     # no free-running color while recording
    c.lockstep=False
    v=c.colour_only();assert v is not None and int(v.colour.mean())>150   # idle again: the newest color frame, free running


def test_save_color_choice_is_remembered():
    from types import SimpleNamespace
    from app2.scan.session import ScanSession
    from app2.scan import settings
    s=ScanSession.__new__(ScanSession);s.cfg=SimpleNamespace(save_color=True);s._emit_state=lambda **k:None
    s._set_save_color(False)
    assert settings.load().get('save_color') is False and s.cfg.save_color is False
    s._set_save_color(True);assert settings.load().get('save_color') is True


def test_cut_platter_follows_the_newest_pose_when_the_scanner_moves():
    """Cut below bed while the finder looks for the dots again: the platter is seen from the newest tracked pose, not
    averaged over the pass's last 30 poses (a hand-moved scanner got a tilted plane through the part)."""
    from app2.scan.session import ScanSession, F
    from scanner.table_plane import to_camera
    s=ScanSession.__new__(ScanSession); s._cut_poses=[]
    tb=(np.array([0.,0.,1.]),0.0,np.array([0.,0.,0.]),100.0,"bed")          # the scan frame's z=0 plane
    poses=[]
    for k in range(30):                                                         # the scanner sweeps 60 degrees down
        a=np.radians(2*k); R=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]],float)
        T=np.eye(4); T[:3,:3]=R; T[:3,3]=(0,-200.0,150.0); poses.append(T)
    s._cut_poses=poses
    n_avg,off_avg,_,_=s._camera_platter(tb)                                     # the old average over the sweep
    n_now,off_now,_,_=s._camera_platter(tb,[poses[-1]])
    want_n,want_off=to_camera(tb[0],tb[1],poses[-1])
    assert np.allclose(n_now,want_n) and abs(off_now-want_off)<1e-9
    assert np.degrees(np.arccos(np.clip(n_avg@want_n,-1,1)))>20                 # the average was far off
    # the session's fallback uses the newest pose (last['T'] in meters, F flips the axes)
    class Bed:
        def camera_geo(self,sd,rec): return None
    class Cut:
        def apply(self,d,geo): Cut.geo=geo; return np.arange(3)
    sd=NS(ps=0,idx=5); s.below_bed=True; s.recording=True; s.busy=None; s.bed=Bed(); s.bedcut=Cut(); s.table=tb
    s._cut_geo_kept=None; s._cut_said=None; s._log=lambda *a: None; s._no=lambda sd: 1
    Tm=poses[-1].copy(); Tr=Tm@np.linalg.inv(F); Tr[:3,3]/=1000.0; s.last=dict(ok=True,T=Tr)
    idx=s._cut_below_bed(sd,np.ones((4,4),np.uint16))
    assert len(idx)==3 and np.allclose(Cut.geo[0],want_n) and abs(Cut.geo[1]-want_off)<1e-6
    s.last=dict(ok=False,T=None); Cut.geo=None                                  # a lost frame keeps the last platter
    s._cut_below_bed(sd,np.ones((4,4),np.uint16)); assert np.allclose(Cut.geo[0],want_n)


def test_bed_height_lifts_the_live_cut_plane_and_its_tint():
    """Bed height moves the Cut below bed plane (and the purple tint) up or down by mm, while recording too."""
    from app2.scan.session import ScanSession, F, BED_LIFT_MAX_MM
    s=ScanSession.__new__(ScanSession); s.bed_lift_mm=0.0; s._log=lambda *a: None; s._emit_state=lambda *a,**k: None
    geo=(np.array([0.,0.,-1.]),300.0,None,100.0)                                  # camera 300 mm over the plane
    class Bed:
        def camera_geo(self,sd,rec): return geo
    class Cut:
        def apply(self,d,g): Cut.geo=g; return np.arange(2)
    sd=NS(ps=0,idx=5); s.below_bed=True; s.recording=True; s.busy=None; s.bed=Bed(); s.bedcut=Cut(); s.table=None
    s._cut_geo_kept=None; s._cut_said=None; s._no=lambda sd: 1; s.last=None; s.recording=True
    s._cut_below_bed(sd,np.ones((4,4),np.uint16)); assert Cut.geo[1]==300.0
    s._set_bed_lift(-8)                                                           # the plane 8 mm lower: more of the part stays
    s._cut_below_bed(sd,np.ones((4,4),np.uint16)); assert Cut.geo[1]==308.0 and s._overlay_geo(sd)[1]==308.0
    s._set_bed_lift(500); assert s.bed_lift_mm==BED_LIFT_MAX_MM                   # clamped
    s._set_bed_lift("x"); assert s.bed_lift_mm==BED_LIFT_MAX_MM
    s.recording=False; assert s._overlay_geo(sd) is None


def test_touch_up_cloud_carries_the_built_surface_for_the_surface_style():
    """A touch-up side's cloud brings the built mesh (V, VN, F) colored like its dots, so Surface / Dots and the
    Distance / Coverage / Plain buttons work there too (they did nothing in touch-up, 2026-10-22)."""
    from app2.scan.session import ScanSession
    s=ScanSession.__new__(ScanSession); sent=[]
    s._no=lambda sd: 6; s._next_seq=lambda: 1; s.home_seq=0; s._view_bed_up=lambda sd: None
    s._publish_distance=lambda sd,payload: None; s._emit=lambda kind,payload: sent.append((kind,payload))
    P=np.random.rand(10,3).astype(np.float32); V=np.random.rand(25,3).astype(np.float32)
    L=np.arange(10,dtype=np.uint8); B=np.full(10,255,np.uint8); B[3]=2; vi=np.arange(25)%10
    tm=NS(P=P,N=P,levels=lambda: L,capture_distance=B,pivot=P.mean(0),scan_guide=None,counts=lambda: (1,4),remaining_counts=lambda: {},
          mesh=(V,V,np.zeros((3,3),np.int32)),v_index=vi)
    s._emit_touch(NS(),tm)
    kind,c=sent[-1]; assert kind=='cloud' and c['touch']=={'done':1,'total':4}
    assert c['V'] is V and c['F'].shape==(3,3) and np.array_equal(c['VL'],L[vi]) and np.array_equal(c['VB'],B[vi])
    tm.mesh=None; s._emit_touch(NS(),tm); assert 'V' not in sent[-1][1]                  # no mesh: dots as before
