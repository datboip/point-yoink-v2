from types import SimpleNamespace as NS
import threading
import numpy as np
from app2.scan.framing import analyze,Framing
from app2.scan import preview


def scene():
    d=np.full((100,140),5000,np.uint16);d[30:70,40:90]=3000;d[45:47,90:130]=3000
    geo=(np.array([0.,0.,-1.]),500.,np.array([0.,0.,500.]),300.)
    return d,(100.,100.,70.,50.),geo


def test_connected_thin_overhang_is_inside_the_full_part_bounds():
    d,cal,geo=scene();d[0:5,0:5]=2500;s=analyze(d,cal,geo,(0,1000))
    assert s['points']==2080 and list(s['rect'])==[40,30,129,69]
    v,u=np.nonzero(s['mask']);assert np.all(u<=s['rect'][2]) and np.all(v<=s['rect'][3])


def test_color_bounds_include_depth_parallax_and_marks_have_separate_meaning(monkeypatch):
    d,cal,geo=scene();d[45:47,90:130]=4000;s=analyze(d,cal,geo,(0,1000))
    from app2.scan.depth_footprint import DepthFootprint
    footprint=DepthFootprint(3)
    for i in range(3):s['depth_outline']=footprint.update(d,s,i/3)
    f=Framing(3);view=f.update(s,dict(state='good depth',center_mm=s['depth_mm']),0,automatic=False)
    monkeypatch.setattr(preview,'RGB_CAL',(np.array([[100.,0,70],[0,100.,50],[0,0,1]]),np.eye(3),np.array([40.,0,0])))
    area=preview.part_area(cal,view);v,u=np.nonzero(s['mask']);x,y=preview.depth_px_to_rgb(u,v,d[v,u]*.1,cal)
    assert x.min()>=area['outline'][:,0].min() and x.max()<=area['outline'][:,0].max()
    assert area['part_center'] is not None and area['depth_outline'] is not None
    assert not np.allclose(area['part_center'],area['center'])


def test_framing_labels_do_not_modify_the_camera_image(monkeypatch):
    monkeypatch.setattr(preview,'RGB_CAL',(np.eye(3),np.eye(3),np.zeros(3)))
    rgb=np.full((600,800,3),200,np.uint8);depth=np.zeros((600,800),np.uint16)
    area=dict(outline=np.array([[200,200],[500,200],[500,450],[200,450],[200,200]]),center=np.array([350,325]))
    plain=preview.depth_guides(rgb.copy(),depth,(1,1,0,0),300,(255,220,0),area)
    labeled=preview.depth_guides(rgb.copy(),depth,(1,1,0,0),300,(255,220,0),dict(area,label='Past depth view: move back'))
    assert np.array_equal(plain,labeled)
    assert not np.array_equal(plain,rgb), 'The geometry path must actually draw for this check'


def test_light_failure_before_frames_does_not_stick_and_late_failure_is_ignored():
    from app2.scan.sources import DeviceControl
    calls=[];gate=threading.Event();entered=threading.Event()
    def led(on):
        calls.append(on)
        if len(calls)==1:entered.set();gate.wait(2);raise OSError('device absent')
        return dict(readback=True,on=on)
    d=DeviceControl(True,gain_fn=lambda x:None,led_fn=led,present=False)
    try:
        d.led(False);assert entered.wait(2);assert d.led_state!='failed'
        d.stream_arrived();gate.set();assert d.flush(2)
        assert calls==[False,False] and d.led_state=='off'
        d.stream_arrived();assert d.flush(2) and len(calls)==2
        d.led_fn=lambda x:(_ for _ in ()).throw(OSError('device present but command failed'))
        d.led(True);assert d.flush(2) and d.led_state=='failed'
    finally:gate.set();d.close()


def test_live_source_marks_device_present_only_after_a_complete_fake_frame():
    import io
    from app2.scan.sources import LiveSource,FS
    calls=[];frames=[]
    s=LiveSource(depth_argv=['owned-test-stream'],colour_argv=['owned-test-color'],gain_fn=lambda x:None,led_fn=lambda on:(calls.append(on) or dict(readback=True,on=on)))
    try:
        assert not s.device.present;s.running=True;s.sink=frames.append
        s._pump(NS(stdout=io.BytesIO(bytes(FS))),s.gen)
        assert s.device.flush(2) and s.device.present and len(frames)==1
        assert calls==[False]
        s._pump(NS(stdout=io.BytesIO(bytes(FS))),s.gen)
        assert s.device.flush(2) and calls==[False]
    finally:s.running=False;s.device.close()
