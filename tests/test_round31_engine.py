"""New geometry and frame scheduling contracts, without a scanner."""
import os
os.environ['POINTYOINK_NO_DEVICE']='1'
import time
from threading import Event
import numpy as np
import cv2
import pytest


def test_color_delivery_wakes_composer_while_depth_is_busy():
    from app2.scan.sources import FrameSource
    from app2.scan.preview import Composer
    source=FrameSource();seen=Event();blocked=Event();release=Event()
    c=Composer((200,200,100,60),lambda kind,v:seen.set() if v.colour_only else None,source.colour)
    depth=np.full((120,200),2000,np.uint16);ir=np.full(depth.shape,60,np.uint8)
    try:
        c.compose(depth,ir,False)
        original=c.compose
        def slow(**kwargs):blocked.set();release.wait(2);return original(**kwargs)
        c.compose=slow;c.post(dict(d=depth,irl=ir,lost_now=False));assert blocked.wait(1)
        ok,jpeg=cv2.imencode('.jpg',np.full((600,800,3),80,np.uint8));assert ok
        source.color_arrived(jpeg.tobytes(),time.monotonic())
        assert seen.wait(1), 'Color must not wait for the depth preview'
    finally:release.set();c.stop();c.thread.join(2);source.device.close()


def test_color_subscription_is_removed_on_stop():
    from app2.scan.sources import FrameSource
    from app2.scan.preview import Composer
    s=FrameSource();c=Composer((1,1,0,0),lambda *a:None,s.colour)
    assert len(s._color_listeners)==1
    c.stop();c.thread.join(2);assert not s._color_listeners;s.device.close()


def test_live_cut_off_keeps_deep_overhang_and_changed_value_invalidates_cache():
    from app2.scan.bedcut import BedCut
    d=np.full((20,60),3000,np.uint16);geo=(np.array([0,0,-1.]),200.,np.array([0,0,200.]),20.)
    cut=BedCut((100,100,30,10),60,20,outside_below_mm=0)
    off=cut.cut(d,geo);assert len(off)<d.size and 0 not in off
    cut.outside_below_mm=25
    on=cut.cut(d,geo);assert len(on)==d.size
    assert np.all(d==3000)


@pytest.mark.parametrize('outline',[None,dict(shape='rect',centre=[0,0],width=100,depth=80,angle=0)])
def test_build_cut_preserves_overhang_with_known_outline(outline):
    from scanner.base_cut import removed
    p=dict(n=[0,0,1],d=0,margin=1,keep_above=True,centre=[0,0,0],radius_mm=50,x_axis=[1,0,0])
    if outline:p['outline']=outline
    points=np.array([[0,0,-30],[90,0,-60],[90,0,5],[0,0,5]],float)
    assert removed(points,p).tolist()==[True,False,False,False]
    assert removed(points.reshape(2,2,3),p).ravel().tolist()==[True,False,False,False]
    p['outside_below_mm']=25;assert removed(points,p).tolist()==[True,True,False,False]
    del p['centre'];assert removed(points,p).tolist()==[True,True,False,False]


def test_stereo_limit_uses_calibration_and_preserves_measured_edge_unknown(monkeypatch):
    from app2.scan import preview
    from app2.scan.stereo_view import Stereo
    cal=(700,700,300,150)
    monkeypatch.setattr(preview,'RGB_CAL',(np.array([[700,0,300],[0,700,150],[0,0,1.]]),np.eye(3),np.zeros(3)))
    s=Stereo(cal,30,300)
    edge=preview.stereo_limit(s,cal,(300,600),300)
    assert edge is not None and np.isfinite(edge).all()
    assert np.min(edge[:,0])==pytest.approx(70)
    assert np.max(edge[:,0])==pytest.approx(599)
    assert preview.stereo_limit(None,cal,(300,600),300) is None


def test_color_pose_has_rigid_transform_and_rejects_ambiguity():
    from app2.scan.color_bed import solve_pose
    from scanner.bed import BedModel
    rng=np.random.default_rng(31);dots=[];image=np.full((600,800,3),30,np.uint8)
    while len(dots)<20:
        point=rng.uniform(-60,60,2)
        if not dots or np.linalg.norm(np.asarray(dots)-point,axis=1).min()>12:dots.append(point)
    dots=np.array(dots)
    from scanner.bed import basis
    x,y=basis([0,0,-1])
    points=dots[:,0,None]*x+dots[:,1,None]*y+[0,0,300]
    for point in points:cv2.circle(image,(round(400+point[0]*700/300),round(300+point[1]*700/300)),4,(240,240,240),-1)
    model=BedModel(dots=np.column_stack([dots,np.ones((len(dots),2))]))
    result=solve_pose(image,(np.array([[700,0,400],[0,700,300],[0,0,1.]]),np.eye(3),np.zeros(3)),(np.array([0,0,-1.]),300),model)
    assert result is not None
    B=result['B_cam'];assert B.shape==(4,4) and np.linalg.det(B[:3,:3])==pytest.approx(1)
    assert result['inliers']>=5
    assert solve_pose(image,None,(np.array([0,0,-1.]),300),model) is None


def test_cut_setting_defaults_off_persists_and_rejects_nonfinite(tmp_path,monkeypatch):
    from app2.scan.session import ScanConfig,ScanSession
    from app2.scan.settings import save
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    cfg=ScanConfig(str(tmp_path),str(tmp_path));assert cfg.outside_below_mm==0
    save(outside_below_mm=80);assert ScanConfig(str(tmp_path),str(tmp_path)).outside_below_mm==80
    s=ScanSession.__new__(ScanSession);s.cfg=cfg;s.outside_below_mm=0;s.bedcut=s._preview_cut=s.composer=None;s._emit_state=lambda:None
    s._set_outside_below(float('nan'));assert s.outside_below_mm==0
    s._set_outside_below(60);assert s.outside_below_mm==60


def test_color_learner_accepts_real_six_field_detections_without_double_counting():
    from app2.scan.color_bed import ColorBedLearner
    from scanner.bed import basis
    rng=np.random.default_rng(13);dots=[]
    while len(dots)<20:
        point=rng.uniform(-60,60,2)
        if not dots or np.linalg.norm(np.asarray(dots)-point,axis=1).min()>12:dots.append(point)
    depth=np.full((600,800),3000,np.uint16);ir=np.full(depth.shape,30,np.uint8);rgb=np.full((*depth.shape,3),30,np.uint8)
    for i,(x,y) in enumerate(dots):
        center=(round(400+x*700/300),round(300+y*700/300))
        cv2.circle(rgb,center,4,(240,240,240),-1)
        if i>=5:cv2.circle(ir,center,4,240,-1)
    ok,jpeg=cv2.imencode('.jpg',rgb);assert ok
    learner=ColorBedLearner((700,700,400,300));learner.plane=(np.array([0,0,-1.]),300.);learner.e=basis(learner.plane[0]);learner.state='learning'
    cal=(np.array([[700,0,400],[0,700,300],[0,0,1.]]),np.eye(3),np.zeros(3))
    learner.add_color(depth,ir,jpeg.tobytes(),cal,t=1,gain=4)
    assert learner.color_stats[-1]['extra_dots']==5
    assert len(learner.cs)==20


def test_color_pose_worker_limits_redundant_fits_but_keeps_latest_result():
    from app2.scan.color_tracking import ColorObserver
    calls=[]
    def measure(*args):calls.append(1);return dict(angle_deg=10)
    o=ColorObserver(3,measure,interval=.1);image=np.zeros((20,20,3),np.uint8)
    try:
        o.anchor(object(),(np.eye(3),np.eye(3),np.zeros(3)),([0,0,1],-300),3,now=1)
        o.offer(image,now=1);o.future.result(2)
        assert o.offer(image,now=1.02)['angle_deg']==10
        assert o.future is None and len(calls)==1
        o.offer(image,now=1.12);o.future.result(2);assert len(calls)==2
    finally:o.close()


def test_flat_scene_without_fresh_dots_cannot_silently_fuse():
    from types import SimpleNamespace as NS
    from app2.scan.tracking import TrackerHost
    calls=[];tracker=NS(fused=10,last_good=np.eye(4),streak=0,_hint=None,vel=np.eye(4),step=lambda d:calls.append(d))
    host=TrackerHost((1,1,0,0),20,20);host.trk=tracker;host.marker_required=True
    result=host.step(np.full((20,20),3000,np.uint16))
    assert not result['ok'] and not calls
    assert 'dots' in result['refusal_reason'] and tracker.streak==1
    assert not host.marker_required


def test_actual_mesh_cut_does_not_flag_preserved_overhang_as_wrong_plane(tmp_path):
    import json,struct,subprocess,sys
    from pathlib import Path
    import open3d as o3d
    from app2 import REPO
    cache=tmp_path/'cache';cache.mkdir()
    poses=[]
    for i,angle in enumerate((0,120,240)):
        a=np.radians(angle);T=np.eye(4);T[:3,3]=[100*np.cos(a),100*np.sin(a),300]
        poses.append(struct.pack('<ii16d',0,i,*T.ravel()))
    (cache/'pointyoink_register_pose.pose').write_bytes(struct.pack('<i',len(poses))+b''.join(poses))
    def box(size,origin):return o3d.geometry.TriangleMesh.create_box(*size).translate(origin)
    mesh=box((40,40,1),(-20,-20,-1))+box((40,20,55),(60,-10,-50)).subdivide_midpoint(1)+box((100,20,10),(0,-10,5))
    raw=tmp_path/'raw.ply';out=tmp_path/'cut.ply';plane=tmp_path/'plane.json'
    o3d.io.write_triangle_mesh(str(raw),mesh)
    plane.write_text(json.dumps(dict(n=[0,0,1],d=0,centre=[0,0,0],radius_mm=30,source='bed',x_axis=[1,0,0])))
    result=subprocess.run([sys.executable,str(Path(REPO)/'scanner/strip_table.py'),str(tmp_path),str(out),'--mesh',str(raw),'--plane',str(plane)],capture_output=True,text=True,check=True)
    assert 'STAGE wrong_plane' not in result.stdout
    kept=o3d.io.read_triangle_mesh(str(out))
    assert np.min(np.asarray(kept.vertices)[:,2])<-40
