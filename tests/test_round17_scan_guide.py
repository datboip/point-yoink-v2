import numpy as np
import pytest
try:
    from scanner.scan_guide import Lens, Reachability, look_at, axis_from_poses, plan, guidance
except ModuleNotFoundError:
    class Missing:
        def __call__(self,*args,**kwargs): raise AssertionError('Scan guide is not implemented')
        def __getattr__(self,name): raise AssertionError('Scan guide is not implemented')
    Lens=Reachability=look_at=axis_from_poses=plan=guidance=Missing()


def square(x, z=100, half=10):
    V=np.array([[x-half,-half,z],[x+half,-half,z],[x+half,half,z],[x-half,half,z]],float)
    return V,np.array([[0,1,2],[0,2,3]])


def checker(V,F,P,N,second=(0,0,0),bed=None):
    return Reachability(V,F,P,N,Lens(100,100,100,100,200,200,10,500,second),bed=bed,axis=[0,1,0],center=[0,0,100])


def test_camera_shadow_blocks_surface():
    V,F=square(0,z=50)
    c=checker(V,F,[[0,0,100]],[[0,0,-1]])
    assert not c.reachable(np.eye(4),False)[0]


def test_second_camera_shadow_even_when_depth_sees_surface():
    V,F=square(25,z=50,half=8)
    c=checker(V,F,[[0,0,100]],[[0,0,-1]],second=(50,0,0))
    assert c.clear(np.zeros(3),c.P)[0]
    assert not c.reachable(np.eye(4),False)[0]


def test_fov_range_and_signed_facing():
    V,F=square(1000)
    c=checker(V,F,[[0,0,100],[300,0,100],[0,0,600],[0,0,100]],[[0,0,-1]]*3+[[0,0,1]])
    assert c.reachable(np.eye(4),False).tolist()==[True,False,False,False]


def test_bed_blocks_bottom_from_above():
    V,F=square(1000)
    c=checker(V,F,[[0,-10,100]],[[0,1,0]],bed=([0,1,0],0))
    T=look_at(np.array([0,100,0]),[0,-10,100],[0,1,0])
    assert not c.reachable(T,False)[0]


def test_turntable_rotates_mesh_equivalently_to_camera():
    V,F=square(1000)
    c=checker(V,F,[[0,0,100]],[[0,0,1]])
    assert not c.reachable(np.eye(4),False)[0]
    assert c.reachable(np.eye(4),True,5)[0]


def test_tilt_guidance_ignores_turntable_yaw():
    T=look_at(np.array([0,80,200]),[0,0,0],[0,1,0])
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
    C=T.copy(); C[:3,:3]=R@T[:3,:3];C[:3,3]=R@T[:3,3]
    assert guidance(C,T,axis=[0,1,0])['ready']


def test_live_move_direction_and_green_target():
    T=look_at(np.array([0,100,200]),[0,0,0],[0,1,0]); C=T.copy();C[:3,3]*=1.2
    assert 'closer' in guidance(C,T)['text']
    assert guidance(T,T)['ready']


def test_circle_axis_uses_saved_pose_geometry():
    n=np.array([0,1,1])/np.sqrt(2); x=np.array([1,0,0]);y=np.cross(n,x)
    poses=[]
    for a in np.linspace(0,2*np.pi,36,endpoint=False):
        T=np.eye(4);T[:3,3]=[3,4,5]+100*(np.cos(a)*x+np.sin(a)*y);poses.append(T)
    axis,c=axis_from_poses(poses,n)
    assert np.allclose(axis,n); assert np.allclose(c,[3,4,5])


def banana():
    # A curved tube, its inner surface hidden by its far arm at several views.
    V=[];F=[]
    for a in np.linspace(-1.25,1.25,48):
        for b in np.linspace(0,2*np.pi,24,endpoint=False):
            r=60+10*np.cos(b);V.append([r*np.sin(a),10*np.sin(b),r*np.cos(a)])
    for i in range(47):
        for j in range(24):
            a=i*24+j;b=i*24+(j+1)%24;c=b+24;d=a+24;F.extend([[a,b,c],[a,c,d]])
    return np.array(V),np.array(F)


def test_curved_banana_self_occlusion():
    V,F=banana()
    c=checker(V,F,[V[23*24+12]],[[0,0,-1]])
    assert c.clear([0,0,0],c.P)[0]
    assert not c.clear([0,0,150],c.P)[0]


def test_greedy_selects_area_and_only_labels_bed_side_flip(monkeypatch):
    from scanner import scan_guide as g
    class C:
        N=np.array([[0,-1,0],[1,0,0],[0,1,0]]);axis=np.array([0,1,0])
        def visibility(self,T,*args):return np.array([0]),np.array([[False,False,True]])
        def phase_pose(self,T,angle):return T
    monkeypatch.setattr(g,'candidates',lambda *_:[dict(T=np.eye(4),tilt=30,distance=200,height=100,mode='turntable')])
    r=plan(C(),[10,20,70]);assert r['poses'][0]['share']==.7
    assert r['needs_flip']==[0];assert r['unreachable']==[1]
    assert r['poses'][0]['capture']=='Single'


def test_stereo_calibration_uses_q_not_rgb(tmp_path):
    v=np.array([0,100,0,100,0,100,100,0,0,1],'<f4');(tmp_path/'Pl.bin').write_bytes(v.tobytes())
    q=np.eye(4,dtype='<f4');q[3,2]=1/80.;(tmp_path/'Q.bin').write_bytes(q.tobytes())
    for n in ('mapparamL.bin','mapparamR.bin'):(tmp_path/n).write_bytes(bytes(12)+np.eye(3,dtype='<f4').tobytes())
    (tmp_path/'LC_RT.bin').write_bytes(bytes(48))
    lens=Lens.from_param(tmp_path);assert lens.second==pytest.approx((80,0,0))
    high=Lens.from_param(tmp_path,size=(800,600))
    assert high.fx==2*lens.fx and high.cx==2*lens.cx
    assert high.width/high.fx==lens.width/lens.fx


def test_build_and_touchup_hooks_exist():
    from app2.builder.stagehost import StageHost
    assert callable(StageHost.h_scan_guide)
    from app2.scan.touchup import TouchModel
    tm=TouchModel(np.zeros((1,3)),np.ones((1,3)),np.zeros(1),np.array([-1]),[])
    assert tm.scan_guide is None


def test_thin_occluder_is_not_endpoint_tolerance():
    V,F=square(0,z=99.8)
    c=checker(V,F,[[0,0,100]],[[0,0,-1]])
    assert not c.clear([0,0,0],c.P)[0]


def test_live_height_and_handheld_position():
    T=look_at(np.array([0,100,200]),[0,0,0],[0,1,0])
    C=T.copy();C[:3,3]=[0,-100,200]
    assert not guidance(C,T)['ready']
    C=T.copy();C[:3,3]=[200,100,0]
    assert not guidance(C,T,turntable=False)['ready']


def test_turntable_uses_calibrated_gravity():
    T=look_at(np.array([0,100,200]),[0,0,0],[0,1,0])
    C=T.copy();C[:3,3]+=[30,0,0]
    r=guidance(C,T,gravity=[0,-9.8,0],axis=[1,0,0])
    assert r['ready']


def test_missing_map_available_without_scanner_metadata(tmp_path,monkeypatch):
    import open3d as o3d
    from app2.builder.scan_guide import guide
    from scanner import quality_map as qm
    V,F=square(0); m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    src=tmp_path/'model.ply';o3d.io.write_triangle_mesh(str(src),m)
    monkeypatch.setattr(qm,'quality_map',lambda *_:dict(kind=np.array([1,2,0,0]),spots=[]))
    r=guide(tmp_path,src)
    rgb=np.load(r['rgb'])
    assert list(rgb[0])==[255,70,50] and list(rgb[1])==[180,90,230]
    assert r['poses']==[] and 'calibration' in r['reason']


def test_ghost_scanner_has_body_and_view_cone():
    from app2.qt.scan_guide import lines
    pose=dict(T=np.eye(4).tolist(),distance=200)
    lens=dict(fx=100,fy=100,cx=100,cy=100,width=200,height=200,far=400)
    seg=lines(pose,lens)
    assert len(seg)==20 and sum(cone for _,_,cone in seg)==8
    assert all(np.isfinite(a).all() and np.isfinite(b).all() for a,b,_ in seg)


def test_live_model_panel_selects_target_and_reuses_done_counts():
    from PySide6.QtWidgets import QApplication
    from app2.qt.model_panel import ModelPanel
    app=QApplication.instance() or QApplication([])
    panel=ModelPanel(gl=False)
    T=look_at(np.array([0,100,200]),[0,0,0],[0,1,0])
    panel.scan_guide=dict(poses=[dict(T=T.tolist(),text='Camera 100 mm above the bed, hold still')],axis=[0,1,0],center=[0,0,0],mode='turntable')
    panel._guide_done=(2,3);panel._guide_update(T)
    assert panel.overlay.guidance['ready']
    assert '2 of 3 spots done' in panel.overlay.guidance['text']
    panel.close()


def test_second_camera_fov_rejects_untriangulatable_spot():
    V,F=square(1000)
    c=checker(V,F,[[0,0,20]],[[0,0,-1]],second=(80,0,0))
    assert not c.reachable(np.eye(4),False)[0]


def test_gravity_alignment_requires_tilt_and_rejects_inconsistent_motion():
    from scanner.scan_guide import GravityAlignment
    fit=GravityAlignment();up=np.array([0,1,0])
    for _ in range(8):assert fit.add(np.eye(4),[0,-9.8,0],up) is None
    for a in np.linspace(0,.8,10):
        T=np.eye(4);T[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
        fit.add(T,T[:3,:3].T@-up,up)
    assert np.allclose(fit.rotation,np.eye(3),atol=1e-6)


def test_oriented_model_transforms_ghost_and_highlight_coordinates():
    from app2.qt.scan_guide import display_points
    M=np.eye(4);M[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];M[:3,3]=[4,5,6]
    class View:items={'main':dict(model=M)}
    assert np.allclose(display_points(View(),[[1,0,0],[0,2,0]]),[[4,6,6],[2,5,6]])


def test_lost_tracking_never_claims_target_reached():
    r=guidance(np.eye(4),np.eye(4),tracking_ok=False)
    assert not r['ready'] and 'Find the part' in r['text']
