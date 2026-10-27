import numpy as np
from types import SimpleNamespace as NS
from PySide6.QtCore import QRectF
from PySide6.QtGui import QFontMetrics
from test_round26_ui import page


def test_camera_match_action_reserves_space_outside_all_image_marks(page):
    app,p,s=page;s.scan_mode='far';s.mode_evidence=dict(mode='near');p.on_state(s)
    for size in ((1280,800),(1920,1080)):
        p.resize(*size)
        for _ in range(8):app.processEvents()
        assert p.camera_match.isVisible()
        assert not p.camera_match.geometry().intersects(p.model.viewport.geometry())
        assert p.camera_match.geometry().bottom()<p.model.viewport.geometry().top()
    s.scan_mode='near';p.on_state(s)
    assert p.model.layout().contentsMargins().top()==p.model.ring_w     # no notice row; only the outline ring


def test_partial_and_empty_range_tests_explain_how_to_continue(page):
    from app2.scan.range_test import result
    from test_round26_range_test import sample
    app,p,s=page;p.sides.tabs.setCurrentIndex(1)
    for bins in ([],[350],[350,355]):
        s.range_test=dict(active=False,attempted=True,result=result([sample(z) for z in bins],'near','test'));p.on_state(s)
        tools=p.controls.capture_tools
        assert tools.detail.text()=='Not enough distances yet, keep moving'
        assert not tools.detail.isHidden()
        if bins:
            p.controls_scroll.ensureWidgetVisible(tools.chart)
            for _ in range(4):app.processEvents()
            assert tools.chart.plot_rect().width()>QFontMetrics(tools.chart.font()).horizontalAdvance('350 to 355 mm')
            assert tools.chart.plot_rect().height()>=QFontMetrics(tools.chart.font()).lineSpacing()


def test_nudge_steps_and_before_after_error_are_visible_in_camera_setup(page):
    app,p,s=page;p.sides.tabs.setCurrentIndex(1);p.controls.calibration_setup.setChecked(True)
    s.overlay=dict(manual=dict(x=1),fit=dict(before_px=2.24,after_px=1,matches=7));p.on_state(s)
    tools=p.controls.capture_tools;p.controls_scroll.ensureWidgetVisible(tools.nudge)
    for _ in range(8):app.processEvents()
    assert '1 px' in tools.steps.text() and tools.fields['x'].singleStep()==1
    assert tools.fit.isVisible() and '2.24 to 1.00 px' in tools.fit.text()
    viewport=p.controls.calibration_tools;q=tools.fit.mapTo(viewport,tools.fit.rect().topLeft())
    assert viewport.rect().contains(q) and viewport.rect().contains(q+tools.fit.rect().bottomRight())


def test_distance_axis_stays_fixed_and_histogram_counts_do_not_alias(page):
    app,p,s=page;b=p.distance_bar
    g=dict(working_range=[200,400],center_mm=250,zones=[dict(lo=200,hi=400,state='strong depth')],counts=[10,20,30],edges=[200,220,240,260])
    b.set_guide(g,(0,1000),(0,1000));assert b.view_span()==(100,500)
    g=dict(g,center_mm=800);b.set_guide(g,(750,900),(0,1000));assert b.view_span()==(100,500)
    for n in (7,97,211):assert np.isclose(b.histogram_slots(n).sum(),60)
    b.set_guide(dict(g,working_range=None),(0,1000),(0,1000));assert b.isHidden()


def test_capture_key_wraps_every_entry_and_palette_matches_all_three_views(page):
    from app2.qt.capture_key import layout
    from app2.scan.capture_distance import rgb
    from app2.scan.range_guide import BAND_RGB
    app,p,s=page;m=p.model
    for width in (240,440,850):
        box,entries=layout(width,500)
        assert len(entries)==4
        for text,color,rect in entries:
            assert box.contains(rect) and rect.width()>=QFontMetrics(m.font()).horizontalAdvance(text)
        assert all(not rect.intersects(other) for i,(_,_,rect) in enumerate(entries) for _,_,other in entries[i+1:])
    m.view=NS(point_palette=None)
    try:
        m._display_levels(dict(P=np.zeros((4,3)),L=np.zeros(4),B=np.array([0,1,2,255])))
        assert m.view.capture_colors
        assert np.allclose(m.view.point_palette[[0,1,2,4]]*255,rgb(np.array([0,1,2,255])))
    finally:m.view=None


def test_bar_and_dots_agree_at_both_inclusive_best_edges(page):
    from app2.scan.range_guide import band_edges,distance_levels,distance_zones
    app,p,s=page;b=p.distance_bar;profile=dict(observed_distance_mm=[100,600]);edges=band_edges(100,600)
    b.set_guide(dict(working_range=[100,600],zones=distance_zones(profile),center_mm=300,counts=[1],edges=[100,600]),(0,1000),(0,1000))
    rank={'usable depth':1,'strong depth':2,'good depth':2,'too close':0,'too far':0,'unmeasured':0}
    assert [rank[b.zone_at(z)] for z in edges]==distance_levels(edges,profile).tolist()
