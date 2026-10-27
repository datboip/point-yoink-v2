"""Offline motion clocks, honest calibration, sparse prediction, and side presets."""
import json,struct
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def calibration():return dict(sensor_to_camera=np.eye(3).tolist(),quaternion_frame='sensor_to_world',quaternion_order='xyzw',timestamp_basis='host_monotonic_seconds',received_to_depth_offset_s=0.,gyro_scale_rad_s=1.,gravity_norm=1.,gravity_sigma=.01)
def pose(angle):
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix();return T

def snapshot(t,angle=0):
    from app2.scan.telemetry import Sample
    q=tuple(Rotation.from_euler('x',angle,degrees=True).as_quat())
    return 'Receiving',Sample(q,(0,0,0),(0,0,1),(0,0,0),t),1,0


def test_raw_packet_logging_and_replay_are_offline_and_timestamped(tmp_path,monkeypatch):
    from app2.scan.telemetry import TelemetryClient
    from app2.scan.motion_seed import MotionLog,MotionReplay
    monkeypatch.setattr('socket.create_connection',lambda *a,**kw:pytest.fail('No hardware socket is allowed'))
    client=TelemetryClient();log=MotionLog(tmp_path/'motion.jsonl',calibration(),provenance='synthetic');client.set_recorder(log.append)
    packet=b'OBSERVED'+struct.pack('<13f',0,0,0,1,0,0,0,0,0,1,10,20,30)
    for index in range(20):client.ingest(packet,100+index*.1)
    client.set_recorder(None);log.close()
    assert client.snapshot_at(100.05)[1].received==100
    replay=MotionReplay(log.path)
    assert replay.header['provenance']=='synthetic' and replay.header['calibration']==calibration()
    assert replay.snapshot(99)[1] is None and replay.snapshot(100.05)[1].received==100
    assert replay.snapshot(110)[0]=='Stale'


def test_constant_object_spin_is_not_canceled_by_static_camera_imu():
    from app2.scan.motion_seed import PoseSeeder
    seed=PoseSeeder(calibration())
    for index in range(3):seed.accepted(pose(index*2),index/3,snapshot(index/3))
    predicted,sources=seed.predict(1,snapshot(1))
    assert np.allclose(predicted,pose(6)) and 'calibrated camera pose' in sources


def test_camera_and_object_rotation_are_separate_components():
    from app2.scan.motion_seed import PoseSeeder
    c=calibration();c.pop('gyro_scale_rad_s');c.pop('gravity_norm');c.pop('gravity_sigma');seed=PoseSeeder(c)
    for index in range(3):
        T=pose(index*2);T[:3,:3]@=Rotation.from_euler('x',index,degrees=True).as_matrix()
        seed.accepted(T,index/3,snapshot(index/3,index))
    expected=pose(6);expected[:3,:3]@=Rotation.from_euler('x',3,degrees=True).as_matrix()
    predicted,sources=seed.predict(1,snapshot(1,3));assert np.allclose(predicted,expected,atol=1e-8)


@pytest.mark.parametrize('cal,sample',[({},snapshot(1)),(calibration(),snapshot(-10)),(dict(sensor_to_camera=np.zeros((3,3)).tolist()),snapshot(1))])
def test_unknown_or_stale_motion_is_geometric_fallback(cal,sample):
    from app2.scan.motion_seed import PoseSeeder
    seed=PoseSeeder(cal)
    for index in range(3):seed.accepted(pose(index*2),index/3)
    predicted,sources=seed.predict(1,sample)
    assert np.allclose(predicted,pose(6)) and sources==['steady fitted motion']


def test_sparse_motion_uses_elapsed_time_not_frame_count_and_stops_after_gap():
    from app2.scan.motion_seed import PoseSeeder
    seed=PoseSeeder()
    for t in (0,.2,.6):seed.accepted(pose(t*12),t)
    predicted,_=seed.predict(.9);assert np.allclose(predicted,pose(.9*12))
    assert seed.predict(5)[0] is None


def test_far_preset_uses_measured_depth_noise_and_preserves_near(tmp_path):
    from app2.scan.scan_mode import profile,read,write
    near=profile('near',(1000,1000,400,300),np.full((600,800),6000,np.uint16),120,350)
    assert near==dict(mode='near',near=120,far=350,hardware_switch=False)
    far=profile('far',(1000,1000,400,300),np.full((600,800),6000,np.uint16),120,350)
    assert far['far']>350 and far['voxel_mm']>.4 and far['distance_mm']==600
    write(tmp_path,far);assert read(tmp_path)==far
    waiting=profile('far',(1000,1000,400,300),None,120,350)
    from app2.scan.tracking import LIVE_MAX_DEPTH_MM
    assert waiting['far']==LIVE_MAX_DEPTH_MM and 'voxel_mm' not in waiting


def test_session_mode_changes_and_geometry_are_real_backend_calls(tmp_path,monkeypatch):
    from app2.scan.session import ScanSession
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    s=ScanSession.__new__(ScanSession);events=[];s.scan_mode='near';s.near=120;s.far=350;s._near_range=(120,350)
    s._scan_depth=np.full((600,800),6000,np.uint16);s.cal=(1000,1000,400,300)
    far_params=tmp_path/"far";far_params.mkdir()
    import struct
    (tmp_path/"Pl.bin").write_bytes(struct.pack("<HH9f",800,600,1000,0,400,0,1000,300,0,0,1))
    (far_params/"Pl.bin").write_bytes(struct.pack("<HH9f",800,600,700,0,400,0,700,300,0,0,1))
    s.cfg=NS(param_dir=str(tmp_path),near_param=str(tmp_path),far_param=str(far_params),w=800,h=600);s.recording=False;s.busy=None;s.building=False;s.bed=NS(learning=False)
    s.picked=NS(frames=0,folder=str(tmp_path/'side'));(tmp_path/'side').mkdir()
    s.host=NS(has=lambda side:False,release=lambda:None);s.composer=NS(set_scan_geometry=lambda *a:events.append(a))
    s._emit=lambda *a:None;s._emit_sides=lambda:None;s._emit_state=lambda:None;s._log=lambda text:events.append(text)
    s._set_scan_mode('far');assert s.scan_mode=='far' and s.far>350
    assert s.picked.scan_profile['sensor_mode_verified'] is False and events[-1][2]=='far'
    s._set_scan_mode('near');assert (s.near,s.far)==(120,350)
    s.picked.frames=1;s._set_scan_mode('far');assert s.scan_mode=='near' and 'new side' in events[-1]


def test_far_host_routes_seed_before_icp_while_near_is_unchanged():
    from app2.scan.tracking import TrackerHost
    class Fake:
        def __init__(self,*a):self.fused=0;self.streak=0;self._hint=None;self.hints=[];self.T=np.eye(4);self.k=0
        def hint(self,T,**kw):self._hint=T;self.hints.append(T)
        def step(self,d):
            T=pose(self.k*2);self.k+=1;self.fused+=1;self._hint=None
            return dict(ok=True,T=T,fit=1,rmse=0)
    owner=NS(scan_profile={'mode':'far'},motion_calibration=None);host=TrackerHost((1,1,0,0),1,1,factory=Fake,release=lambda:None);host.new(owner)
    for index in range(4):host.step(np.ones((1,1)),timestamp=index/3)
    assert len(host.trk.hints)==1 and np.allclose(host.trk.hints[0],pose(6))
    owner.scan_profile={'mode':'near'};host.new(owner)
    for index in range(4):host.step(np.ones((1,1)),timestamp=index/3)
    assert host.trk.hints==[]


def test_measured_turntable_orbit_preserves_center_and_rejects_freehand():
    from app2.scan.motion_seed import TurntableOrbit
    center=np.array([.03,.02,.4]);orbit=TurntableOrbit(1.5)
    for index in range(40):
        T=pose(index*2);T[:3,3]=(np.eye(3)-T[:3,:3])@center;orbit.accepted(T)
    assert orbit.model is not None and orbit.model['residual_mm']<1e-8
    wanted=pose(30);wanted[:3,3]=(np.eye(3)-wanted[:3,:3])@center
    slipped=wanted.copy();slipped[:3,3]+=np.array([.001,-.002,.001])
    assert orbit.project(slipped) is None and orbit.model is None
    for index in range(40):
        T=pose(index*2);T[:3,3]=(np.eye(3)-T[:3,:3])@center;orbit.accepted(T)
    assert np.allclose(orbit.project(wanted),wanted)
    tiny=TurntableOrbit(1.5)
    for index in range(40):
        T=pose(index*.1);T[:3,3]=(np.eye(3)-T[:3,:3])@center;tiny.accepted(T)
    assert tiny.model is None
    freehand=TurntableOrbit(1.5)
    for index in range(12):
        T=pose(index*2);T[:3,3]=[index*.003,0,index*.005];freehand.accepted(T)
    assert freehand.model is None and freehand.project(T) is None


def test_late_session_motion_attachment_logs_and_routes_raw_packet(tmp_path,monkeypatch):
    from app2.scan.session import ScanSession
    from app2.scan.telemetry import TelemetryClient
    from app2.scan.motion_seed import MotionReplay
    monkeypatch.setattr('socket.create_connection',lambda *a,**k:pytest.fail('No socket'))
    s=ScanSession.__new__(ScanSession);s.motion_provider=None;s.motion_log=None
    s.project=NS(path=str(tmp_path));s.cfg=NS(motion_log=None,motion_calibration=calibration(),motion_provenance='synthetic',param_dir=str(tmp_path),w=800,h=600);s.cal=(1000,1000,400,300);s.scan_mode='far'
    client=TelemetryClient();s._set_motion_provider(client.snapshot_at)
    assert s.motion_log is not None
    packet=b'OBSERVED'+struct.pack('<13f',0,0,0,1,0,0,0,0,0,1,0,0,0)
    client.ingest(packet,10.)
    received={}
    def step(d,**kw):
        received.update(kw);return dict(ok=False,T=np.eye(4),fit=0,rmse=1,reloc=False,seed_sources=[])
    s.host=NS(step=step);s._no=lambda sd:1;s.bed=NS(before_step=lambda *a:None,camera_geo=lambda *a:None,after_step=lambda *a:([],None));s.last=None;s.recording=True;s.capture_mode='continuous';s.save_color=False;s.gl=True;s.led=True;s.source=NS(colour=lambda:None,device=NS(led_state='unknown'));s._check_shot=lambda ok:None;s._emit=lambda *a:None;s.home_seq=0
    side=NS(scan_profile={'mode':'far'},ps=0,idx=0,pass_ok=1,cache=str(tmp_path/'cache'),entries=[],stamps=[],lost=0,rgb_thumb=None,frames=0)
    Path=__import__('pathlib').Path;Path(side.cache).mkdir()
    s._record(side,np.ones((2,2),np.uint16),np.zeros((2,2),np.uint8),10.1,0,1000,0)
    assert received['timestamp']==10.1 and received['motion'][1].received==10.
    assert side.stamps[0]['scan_mode']=='far' and side.stamps[0]['t']==10.1
    client.set_recorder(None);s.motion_log.close();replay=MotionReplay(s.motion_log.path)
    assert replay.header['provenance']=='synthetic' and len(replay.samples)==1


def test_saved_far_profile_drives_build_voxel_range_and_respects_overrides(tmp_path):
    from app2.scan.scan_mode import write
    from stages.project import Project
    from stages.engine import Engine
    folder=tmp_path/'data'/'side';(folder/'cache').mkdir(parents=True);(folder/'param').mkdir()
    (folder/'cache'/'pointyoink_track.json').write_text('{}')
    write(folder,dict(mode='far',voxel_mm=.75,far=1000))
    eng=Engine(Project.open(tmp_path));node='side.fuse@side'
    params=eng.graph().by_id[node].params
    assert params['voxel']==.75 and params['max_depth']==1000
    explicit=eng.graph(with_params={'side.fuse.voxel':.4,'side.fuse.max_depth':600}).by_id[node].params
    assert explicit['voxel']==.4 and explicit['max_depth']==600
    from stages.recipe import new_recipe
    saved=new_recipe();saved['params']['side.fuse']={'voxel':.4,'max_depth':600}
    global_manual=eng.graph(recipe=saved).by_id[node].params
    assert global_manual['voxel']==.4 and global_manual['max_depth']==600
    write(folder,dict(mode='near'));eng.p.refresh()
    near=eng.graph().by_id[node].params
    assert near['voxel']==.4 and near['max_depth']==600


def test_strong_current_marker_quality_anchors_only_supported_native_hint():
    from app2.scan.tracking import TrackerHost
    from track import Tracker
    class HintReceiver:
        _marker_anchor=False
        hint=Tracker.hint
    host=TrackerHost((1,1,0,0),1,1,factory=lambda *a:HintReceiver(),release=lambda:None);host.new(NS(scan_profile={'mode':'near'}))
    quality=dict(inliers=13,runner_up=3,rms_mm=.3,span_mm=50.)
    host.marker_hint(np.eye(4),quality)
    assert host.trk._marker_anchor and np.isclose(host.trk._guard[1],.9)
    assert np.isclose(host.trk._guard[2],np.degrees(np.arctan2(.9,50.)))
    for bad in [dict(quality,inliers=4),dict(quality,runner_up=8),dict(quality,rms_mm=float('nan'))]:
        host.marker_hint(np.eye(4),bad);assert not host.trk._marker_anchor


def test_authorized_bed_current_hit_exports_quality_without_bypassing_acquisition(monkeypatch):
    import bed
    sb=bed.SideBed(None,None);sb.B_scan=lambda:np.eye(4);sb.camera_bed=lambda ps:None;sb.prior_for=lambda ps:None
    h=NS(B_cam=np.eye(4),dots_px=np.array([[0,0,1],[20,20,1],[40,40,1]]),matched=np.array([True,True,True]),centre=np.array([0,0,500]),inliers=13,runner_up=3,rms=.3)
    monkeypatch.setattr(bed,'find_bed',lambda *a,**k:h)
    T=sb.hint_pose(np.ones((2,2)),np.zeros((2,2)),(1000,1000,0,0),0)
    assert T is not None and sb.last_hint_quality['rms_mm']==.3 and sb.last_hint_quality['span_mm']==20
    sb.B_scan=lambda:None
    assert sb.hint_pose(np.ones((2,2)),np.zeros((2,2)),(1000,1000,0,0),0) is None and sb.last_hint_quality is None


@pytest.mark.parametrize('bad',[[],dict(calibration(),sensor_to_camera='unknown'),dict(calibration(),received_to_depth_offset_s='unknown'),dict(calibration(),gravity_norm='unknown')])
def test_malformed_motion_calibration_falls_back_without_stopping_frames(bad):
    from app2.scan.motion_seed import PoseSeeder
    seed=PoseSeeder(bad)
    from app2.scan.session import ScanSession
    session=ScanSession.__new__(ScanSession);session.cfg=NS(motion_calibration=bad);session.motion_provider=lambda t:snapshot(t)
    assert session._motion_snapshot(1)[1].received==1
    assert seed.sensor(snapshot(1),1) is None
    for index in range(3):seed.accepted(pose(index*2),index/3)
    predicted,_=seed.predict(1,snapshot(1))
    assert np.allclose(predicted,pose(6))


def test_turntable_orbit_constraint_is_installed_in_small_part_mode_too():
    """2026-10-25: the orbit constraint used to be Large part only; a turntable scan in Small part mode slid along smooth skins."""
    from types import SimpleNamespace as NS
    from app2.scan.tracking import TrackerHost
    from app2.scan.motion_seed import TurntableOrbit
    class Fake:
        def __init__(self,*a):self.fused=1;self._hint=None;self.pose_constraint=None;self.max_rmse=1.5;self.streak=0;self.last_good=np.eye(4)
        def step(self,raw,**k):return dict(ok=True,T=np.eye(4),fit=1,rmse=0)
    host=TrackerHost((1,1,0,0),1,1,factory=Fake,release=lambda:None);host.new(NS(scan_profile={'mode':'near'},motion_calibration=None))
    host.step(np.ones((1,1)))
    assert isinstance(host.trk.pose_constraint,TurntableOrbit)
