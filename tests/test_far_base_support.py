"""Round support is evidence, not permission to remove arbitrary round parts."""
import numpy as np
import pytest
from scanner.table_plane import SidePlane


def geometry(round_part=False, shift=0., empty=False):
    angle = np.linspace(0, 2*np.pi, 720, endpoint=False)
    radii = np.arange(1., 101.)
    a, r = np.meshgrid(angle, radii)
    top = np.column_stack((r.ravel()*np.cos(a.ravel())+shift, r.ravel()*np.sin(a.ravel()), np.zeros(a.size)))
    a, z = np.meshgrid(angle, np.arange(-100., 0., 2.))
    stand = np.column_stack((95*np.cos(a.ravel())+shift, 95*np.sin(a.ravel()), z.ravel()))
    t = np.linspace(-1, 1, 300)
    xy = np.concatenate([np.column_stack((60*t, np.full_like(t, y))) for y in (-30.,30.)]
                        + [np.column_stack((np.full_like(t, x),30*t)) for x in (-60.,60.)])
    if round_part: xy = np.column_stack((50*np.cos(angle),50*np.sin(angle)))
    upper = np.concatenate([np.column_stack((xy, np.full(len(xy), z))) for z in np.arange(10.,61.,5.)])
    points = np.concatenate([top,stand] + ([] if empty else [upper]))
    centers = np.column_stack((300*np.cos(angle[::30]),300*np.sin(angle[::30]),np.full(24,300.)))
    return points, centers


def test_stand_below_table_does_not_make_the_real_top_absent():
    points, centers = geometry()
    p = SidePlane()
    p.pts = list(np.array_split(points, len(centers)))
    p.centres = list(centers)
    p._sampling_mm = 1.
    result = p.plane()
    assert result is not None
    assert abs(result[1]) <= 2.


@pytest.mark.parametrize('kwargs',[{'round_part':True},{'shift':25.},{'empty':True}])
def test_round_flange_displaced_support_or_no_object_remains_uncertain(kwargs):
    from scanner.base_geometry import verify_top
    points, centers = geometry(**kwargs)
    assert not verify_top(points,centers,[0,0,1],0.,1.)['verified']


def test_tilted_plane_and_short_motion_keep_the_guard():
    from scanner.base_geometry import verify_top
    points, centers = geometry()
    assert not verify_top(points,centers,[0,.4,1],0.,1.)['verified']
    assert not verify_top(points,centers[:3],[0,0,1],0.,1.)['verified']


def test_observed_sampling_comes_from_distance_and_focal_length():
    from scanner.base_geometry import depth_spacing
    assert depth_spacing(np.full((4,4),4000,np.uint16),[800,800,2,2]) == .5
    assert depth_spacing(np.zeros((4,4),np.uint16),[800,800,2,2]) is None


def test_live_remove_base_uses_the_supported_top_without_moving_the_part():
    from types import SimpleNamespace
    from app2.scan.session import ScanSession
    points, centers = geometry()
    table = SidePlane()
    table.pts = list(np.array_split(points, len(centers)))
    table.centres = list(centers)
    table._sampling_mm = 1.
    side = type('SavedSide', (), {})()
    side.table = table
    session = ScanSession.__new__(ScanSession)
    session.gl = True
    session.recording = False
    session.picked = side
    session.table = None
    session.host = SimpleNamespace(has=lambda s: True, fused=True,
                                   cloud=lambda: (points, np.zeros_like(points), np.full(len(points), 2, np.uint8), 1.))
    session.bed = SimpleNamespace(platter=lambda s: None, request_auto=lambda s: None)
    table.plane()  # A completed finder result is already available to this display fixture.
    session._side_clouds = {}
    session.home_seq = 0
    session._no = lambda s: 1
    session._next_seq = lambda: 1
    session._touch_model = lambda s: None
    emitted = []
    session._emit = lambda kind, value: emitted.append(value)
    session.remove_base = True
    session._emit_cloud(side)
    kept = emitted[-1]['P']
    assert len(kept) < len(points)
    assert np.array_equal(kept, points[points[:, 2] > 1.])


def test_live_support_fitting_is_bounded_but_keeps_all_motion_centers(monkeypatch):
    import scanner.base_geometry as geometry_module
    points, centers = geometry()
    table = SidePlane()
    table.pts = [points[:1500].copy() for _ in range(320)]
    table.centres = [centers[i % len(centers)] for i in range(320)]
    table._sampling_mm = 1.
    seen = []
    def support(observed, motion, spacing, **kwargs):
        seen.append((len(observed), len(motion)))
        return np.array([0.,0.,1.]), 0., len(observed), {'tilt_deg': 0.}
    monkeypatch.setattr(geometry_module, 'find_top', support)
    assert table.plane() is not None
    assert seen == [(geometry_module.OBSERVED_FRAME_BUDGET*1500, 320)]
    assert table.plane() is not None
    assert len(seen) == 1


def test_certified_plane_exposes_supported_top_only():
    points, centers = geometry()
    table = SidePlane()
    table.pts = list(np.array_split(points, len(centers)))
    table.centres = list(centers)
    table._sampling_mm = 1.
    result = table.certified_plane()
    assert result is not None
    assert abs(result[1]) <= 2.
    assert table._top_support['verified']


def test_legacy_flat_plane_is_not_a_certified_registration_cut():
    _, centers = geometry()
    x,y = np.mgrid[-100:101:2, -100:101:2]
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    table = SidePlane()
    table.pts = list(np.array_split(points, len(centers)))
    table.centres = list(centers)
    table._sampling_mm = 1.
    assert table.plane() is not None
    assert table.certified_plane() is None
