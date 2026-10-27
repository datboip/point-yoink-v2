"""Color guidance expires and cannot enqueue work or alter measured geometry."""
from threading import Event
import numpy as np
import cv2
import pytest


def test_color_worker_drops_busy_frames_and_expires_anchor():
    from app2.scan.color_tracking import ColorObserver
    started=Event();release=Event();calls=[]
    def measure(rgb,*args):
        calls.append(int(rgb[0,0,0]));started.set();release.wait(2)
        return dict(angle_deg=17,source='Estimated from color dots')
    observer=ColorObserver(3,measure)
    rgb=np.zeros((20,20,3),np.uint8)
    try:
        observer.anchor(object(),(np.eye(3),np.eye(3),np.zeros(3)),([0,0,1],-300),3,now=10)
        assert observer.offer(rgb,now=10) is None and started.wait(1)
        rgb[:]=20
        assert observer.offer(rgb,now=10.1) is None and observer.dropped==1
        release.set();observer.future.result(timeout=2)
        value=observer.offer(rgb,now=10.2)
        assert value['angle_deg']==17 and calls[0]==0
        assert observer.offer(rgb,now=14) is None
        assert len(calls)<=2
    finally:release.set();observer.close()


def test_anchor_change_and_disable_discard_inflight_result():
    from app2.scan.color_tracking import ColorObserver
    released=Event()
    def measure(*args):released.wait(2);return dict(angle_deg=99)
    o=ColorObserver(3,measure)
    try:
        o.anchor(object(),(np.eye(3),np.eye(3),np.zeros(3)),([0,0,1],-300),3,now=1)
        o.offer(np.zeros((10,10,3),np.uint8),now=1);o.clear();released.set();o.future.result(2)
        assert o.offer(np.zeros((10,10,3),np.uint8),now=1.2) is None
    finally:released.set();o.close()


def test_refreshing_a_steady_anchor_keeps_a_completed_observation():
    from app2.scan.color_tracking import ColorObserver
    o=ColorObserver(3,lambda *args:dict(angle_deg=42))
    model=object();cal=(np.eye(3),np.eye(3),np.zeros(3));image=np.zeros((10,10,3),np.uint8)
    try:
        o.anchor(model,cal,([0,0,1],-300),3,now=1)
        o.offer(image,now=1);o.future.result(2)
        o.anchor(model,cal,([0,0,1],-300.1),3,now=1.1)
        assert o.offer(image,now=1.2)['angle_deg']==42
        o.future.result(2)
        o.anchor(model,cal,([0,0,1],-320),3,now=1.3)
        assert o.offer(image,now=1.4) is None
    finally:o.close()


def test_color_solver_rejects_no_map_and_an_ambiguous_grid():
    from app2.scan.color_tracking import solve
    from scanner.bed import BedModel
    image=np.zeros((600,800,3),np.uint8)
    cal=(np.array([[700,0,400],[0,700,300],[0,0,1.]]),np.eye(3),np.zeros(3))
    assert solve(image,cal,([0,0,-1],300),None) is None
    assert solve(image,None,([0,0,-1],300),BedModel()) is None
    grid=np.array([[i*20,j*20,1,0] for i in range(-2,3) for j in range(-2,3)])
    for x,y,*_ in grid:cv2.circle(image,(round(400+x*700/300),round(300+y*700/300)),4,(240,240,240),-1)
    assert solve(image,cal,([0,0,-1],300),BedModel(dots=grid)) is None


def test_color_solver_recovers_a_known_rotation_from_image_dots():
    from app2.scan.color_tracking import solve
    from scanner.bed import BedModel
    image=np.full((600,800,3),30,np.uint8)
    rng=np.random.default_rng(7);points=[]
    while len(points)<20:
        p=rng.uniform(-60,60,2)
        if not points or np.linalg.norm(np.asarray(points)-p,axis=1).min()>12:points.append(p)
    points=np.array(points);a=np.radians(45);x=np.array([np.cos(a),-np.sin(a),0]);y=np.cross([0,0,-1],x)
    P=points[:,0,None]*x+points[:,1,None]*y+np.array([0,0,300])
    for p in P:cv2.circle(image,(round(400+p[0]*700/300),round(300+p[1]*700/300)),4,(240,240,240),-1)
    model=BedModel(dots=np.column_stack([points,np.ones((len(points),2))]))
    result=solve(image,(np.array([[700,0,400],[0,700,300],[0,0,1.]]),np.eye(3),np.zeros(3)),([0,0,-1],300),model)
    assert result is not None and abs(result['angle_deg']-45)<.5
    assert result['source'].startswith('Estimated')
    assert set(result)=={'angle_deg','inliers','rms_mm','source'}


@pytest.mark.parametrize('distance',[200.,350.,600.])
def test_stereo_holes_use_bed_depth_and_residual_is_bounded(distance):
    from app2.scan.stereo_view import Stereo
    left=np.full((300,600),30,np.uint8);depth=np.full(left.shape,round(distance*10),np.uint16)
    for j in range(3):
        for i in range(5):
            c=(180+70*i,65+80*j);cv2.circle(left,c,5,240,-1);cv2.circle(depth,c,4,0,-1)
    cal=(700,700,300,150);s=Stereo(cal,30,300,refine=True);yy,xx=np.indices(left.shape,dtype=np.float32);right=cv2.remap(left,xx+np.float32(700*30/distance-2),yy,cv2.INTER_LINEAR)
    original=depth.copy()
    for _ in range(3):value=s.blend(left,right,depth,distance*.7)
    assert np.array_equal(depth,original)
    assert s.alignment_stats is not None
    assert np.linalg.norm(s.alignment_stats['shift_px'])<=s.alignment_stats['bound_px']
    assert np.median(s.alignment_stats['after_px'])<.5
    assert not (value[:,:,0].astype(int)>value[:,:,2].astype(int)).any()   # aligned everywhere: blue (agree), no orange (differ)


def test_unmeasured_and_one_camera_only_pixels_stay_unblended():
    from app2.scan.stereo_view import Stereo
    left=np.full((120,200),20,np.uint8);right=left.copy();cv2.rectangle(left,(100,50),(120,70),200,-1)
    s=Stereo((200,200,100,60),10,100)
    result=s.blend(left,right,np.zeros(left.shape,np.uint16),200)       # no depth: only the left camera, plain gray
    assert np.array_equal(result[:,:,0],result[:,:,2]) and np.array_equal(result[:,:,2],left)
    result=s.blend(left,right,np.full(left.shape,2000,np.uint16),200)   # the patch only the left camera sees: orange, never blue
    patch=result[50:71,100:121].astype(int)
    assert (patch[...,0]>patch[...,2]).any() and not (patch[...,2]>patch[...,0]+40).any()
    assert (result[90:110,150:180,2].astype(int)>result[90:110,150:180,0]).all()


def test_color_angle_is_off_by_default_and_never_changes_depth(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    from app2.scan.preview import Composer
    c=Composer((200,200,100,60),lambda *a:None)
    depth=np.full((120,200),2000,np.uint16);original=depth.copy()
    try:
        assert c.color_angle_enabled is False
        result=c.compose(depth,np.full(depth.shape,80,np.uint8),False)
        assert result.color_angle is None
        assert np.array_equal(depth,original)
        assert c.color_observer.future is None
    finally:c.stop();c.thread.join(2)
