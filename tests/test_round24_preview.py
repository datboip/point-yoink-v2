import numpy as np
import pytest


def test_center_distance_does_not_report_background_median():
    from app2.scan.range_guide import center_distance,describe
    d=np.full((60,80),9000,np.uint16);d[15:45,20:60]=3000
    assert center_distance(d)==300
    g=describe(d,(200,500));assert g['state']=='in range' and sum(g['counts'])==d.size
    widths=np.diff(g['edges']);assert np.allclose(widths,widths[0]) and abs(widths[0]*10-round(widths[0]*10))<1e-6   # whole 0.1 mm depth units
    assert describe(d,(350,500))['state']=='too close'
    assert describe(d,(100,250))['state']=='too far'
    assert describe(np.zeros_like(d),(100,250))['center_mm'] is None


def test_lost_depth_tint_is_stable():
    from app2.scan.preview import ir_view
    d=np.full((600,800),3000,np.uint16);ir=np.full(d.shape,100,np.uint8);cal=(800,800,400,300)
    assert np.array_equal(ir_view(d,ir,cal,True,None,None),ir_view(d,ir,cal,False,None,None))


def test_center_cross_uses_exact_ir_pixel(monkeypatch):
    from app2.scan import preview as p
    monkeypatch.setattr(p,'RGB_CAL',(np.eye(3),np.eye(3),np.zeros(3)))
    area=p.live_area((1,1,0,0),(0,0,799,599),300)
    assert np.allclose(area['center'],[400,300])


def test_repeating_depth_pulses_warn_but_normal_stream_does_not():
    from app2.scan.stream_health import StreamHealth
    h=StreamHealth(64)
    for i in range(40):assert not h.add(i/10,10000+i)
    h=StreamHealth(64)
    for i in range(40):warning=h.add(i/10,20 if i%10==0 else 10000)
    assert warning


def test_active_scan_cannot_expire_light_and_long_scan_sets_reply_time():
    from app2.scan.light_timer import LightTimer
    t=LightTimer();t.start(0);t.activity(0,True);t.activity(900,True)
    assert not t.tick(900) and t.remaining(900) is None
    t.activity(901,False);t.tick(t.ask_at)
    assert t.reply_wait>=901 and t.remaining(t.ask_at)>=901




def test_missing_center_depth_uses_confirmed_mat_plane(monkeypatch):
    from app2.scan import preview as p
    monkeypatch.setattr(p,'RGB_CAL',None)
    c=p.Composer((800,800,400,300),lambda *a:None)
    c.reference_plane=(np.array([0.,0.,-1.]),413.)
    d=np.zeros((600,800),np.uint16);d[500:520,100:700]=3000
    try:
        frame=c.compose(d,np.zeros_like(d,np.uint8),False)
        assert frame.zref_mm==413 and frame.distance_source=='mat estimate'
        assert p.distance_note(frame)=='Mat estimate 413 mm'
    finally:c.stop()


def test_lower_resolution_saved_depth_keeps_native_picture_center(monkeypatch):
    from app2.scan import preview as p
    monkeypatch.setattr(p,'RGB_CAL',None)
    c=p.Composer((400,400,200,150),lambda *a:None)
    try:
        v=c.compose(np.full((300,400),3000,np.uint16),np.full((300,400),100,np.uint8),False)
        assert v.ir.shape==(300,400,3) and v.depth.shape==(300,400,3)
        assert v.zref_mm==300
    finally:c.stop()


def test_bed_tint_stays_over_the_bed():
    # A part hanging past the rim, a little under the bed's height, is not tinted as bed; the same height over the bed is.
    import numpy as np
    from app2.scan.preview import ir_view
    cal=(100.,100.,50.,50.);H=W=101
    d=np.full((H,W),3000,np.uint16)                       # flat 300 mm away
    nc=np.array([0.,0.,-1.]);oc=299.                      # bed plane at 299 mm, normal toward the camera: 1 mm below
    out=ir_view(d,np.full((H,W),100,np.uint8),cal,False,(nc,oc,np.array([0.,0.,299.]),30.),None,aim=False)
    plain=ir_view(d,np.full((H,W),100,np.uint8),cal,False,None,None,aim=False)
    # 30 mm rim: the centre is over the bed (violet), the corner is not (just the depth tint)
    assert (out[50,50]!=plain[50,50]).any() and (out[2,2]==plain[2,2]).all()


def test_color_dots_stay_with_their_picture_until_the_next_one(monkeypatch):
    """2026-10-25: the dots on the color picture come from the depth frame that picture arrived with, not the newest depth."""
    import cv2
    from app2.scan import preview as p
    K=np.array([[800.,0,400],[0,800.,300],[0,0,1]]);monkeypatch.setattr(p,'RGB_CAL',(K,np.eye(3),np.zeros(3)))
    c=p.Composer((800,800,400,300),lambda *a:None)
    ok,buf=cv2.imencode('.jpg',np.zeros((600,800,3),np.uint8));jpeg1=buf.tobytes()
    ok,buf=cv2.imencode('.jpg',np.full((600,800,3),9,np.uint8));jpeg2=buf.tobytes()
    a=np.zeros((600,800),np.uint16);a[200:400,300:500]=2500
    b=np.zeros((600,800),np.uint16);b[200:400,450:650]=2500          # the part moved on in the next depth frame
    try:
        c.compose(a,np.zeros_like(a,np.uint8),False,jpeg=jpeg1);xa=c.dot_data['dots_xy'][:,0].mean()
        c.compose(b,np.zeros_like(b,np.uint8),False,jpeg=jpeg1);xb=c.dot_data['dots_xy'][:,0].mean()
        assert abs(xb-xa)<1e-6                                      # same picture: the dots did not move with the newer depth
        c.compose(b,np.zeros_like(b,np.uint8),False,jpeg=jpeg2);xc=c.dot_data['dots_xy'][:,0].mean()
        assert xc>xa+100                                            # new picture: its own depth frame
    finally:c.stop()
