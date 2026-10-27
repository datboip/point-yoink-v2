import numpy as np
from app2.scan.assist import AutoRange
from app2.scan.framing import analyze
from app2.scan.stream_health import StreamHealth


def test_connected_auto_range_includes_curved_ends_outside_center_crop():
    d=np.zeros((600,800),np.uint16);d[200:400,220:580]=3400
    d[290:310,50:230]=2400;d[290:310,570:750]=5000
    geo=(np.array([0,0,-1]),600,np.array([0,0,600]),500)
    scene=analyze(d,(600,600,400,300),geo,(250,415))
    r=AutoRange();r.rng=(250,415);near,far=r.connected(scene['range_mm'])
    assert near<=240 and far>=500
    assert np.all((d[scene['mask']]*.1>=near)&(d[scene['mask']]*.1<=far))
    assert near<scene['observed_range_mm'][0] and far>scene['observed_range_mm'][1]


def test_live_full_depth_trace_is_log_only_but_real_collapses_still_warn():
    h=StreamHealth(64);t=0
    for i in range(35):
        t+=.348+(2*.348 if i%6==5 else 0)
        assert not h.add(t,245585 if i%2 else 244986)
    assert h.gap_note and h.gap_note['kind']=='delivery gaps' and h.gap_note['events']>=4
    assert abs(h.gap_note['period_s']-2.784)<.01
    loss=StreamHealth(64)
    for i in range(35):warning=loss.add(i*.348,0 if i%8==0 else 244986)
    assert warning and loss.evidence['kind']=='near-empty depth'


def test_dots_outside_range_remain_visible_in_a_dim_second_color(monkeypatch):
    from app2.scan import preview as p
    monkeypatch.setattr(p,'RGB_CAL',(np.eye(3),np.eye(3),np.zeros(3)));monkeypatch.setattr(p,'OVERLAY_MATRIX',np.eye(3))
    d=np.zeros((18,18),np.uint16);d[6,6]=1500;d[12,12]=3000
    monkeypatch.setattr(p,'dot_samples',lambda depth:np.nonzero(depth>0))   # sample exactly the two set pixels
    rgb=p.depth_guides(np.zeros((18,18,3),np.uint8),d,(1,1,0,0),300,(60,255,60),bands=(200,400),cut=(200,400))
    from app2.scan.range_guide import BAND_RGB
    assert np.array_equal(rgb[6,6],np.array(p.REMOVED_RGB))
    assert np.array_equal(rgb[12,12],BAND_RGB['best'])


def test_five_stable_bands_and_axis_include_cut_and_part():
    from app2.scan.range_guide import distance_zones,measured_profile,distance_action
    zones=distance_zones(measured_profile('near'));assert len(zones)==3
    assert all(a['hi']==b['lo'] for a,b in zip(zones,zones[1:]))
    g=dict(center_mm=zones[1]['lo']-50,zones=zones)
    assert distance_action(g).startswith('Move about')
    g['center_mm']=(zones[1]['lo']+zones[1]['hi'])/2;assert distance_action(g)=='Good distance'


def test_best_follows_the_measured_peak_not_the_middle_of_everything_seen():
    from app2.scan.range_guide import distance_zones
    z=lambda lo,hi,frames,fill:dict(lo=lo,hi=hi,frames=frames,median_fill=fill)
    profile=dict(observed_distance_mm=[150,500],fill_quartiles=[.6,.7,.9],
                 zones=[z(150,160,100,.65),z(160,170,5,.4),z(200,210,200,.93),z(210,220,120,.7),z(220,230,4,.3),z(480,490,30,.62)])
    ok_low,best,ok_high=distance_zones(profile)
    assert (best['lo'],best['hi'])==(200,220)                     # the few far frames and thin zones do not drag Best out
    assert ok_low['hi']==best['lo'] and ok_high['lo']==best['hi'] and ok_low['lo']==180 and ok_high['hi']==240
