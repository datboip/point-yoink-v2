"""Bed hook: dots in view that match no dot of the picked bed, three finder looks in a row, say so once per pass (2026-10-25)."""
import numpy as np
from types import SimpleNamespace as NS


def _bed():
    from app2.scan.bed_adapter import ScannerBed, SideState
    b = ScannerBed.__new__(ScannerBed); b.profile = 'Round dot mat'
    st = SideState(NS(found=False, bed=NS(name='Round dot mat'))); return b, st


def test_three_misses_say_it_once_per_pass():
    b, st = _bed(); sb = st.sb
    assert b._mismatch(st, sb, 12, None, 1, 1) is None and b._mismatch(st, sb, 12, None, 1, 1) is None
    m = b._mismatch(st, sb, 12, None, 1, 1)
    assert m and 'none match the bed Round dot mat' in m and 'Learn bed' in m
    assert b._mismatch(st, sb, 12, None, 1, 1) is None                # said for this pass
    assert b._mismatch(st, sb, 12, None, 2, 1)                        # the next pass may say it again


def test_few_dots_or_a_match_reset_the_count():
    b, st = _bed(); sb = st.sb
    for _ in range(2): b._mismatch(st, sb, 12, None, 1, 1)
    assert b._mismatch(st, sb, 3, None, 1, 1) is None and st.find_miss == 2            # too few dots to judge: no count
    hit = NS(matched=np.ones(9, bool))
    assert b._mismatch(st, sb, 12, hit, 1, 1) is None and st.find_miss == 0           # a real match: start over
    sb.found = True; assert b._mismatch(st, sb, 12, None, 1, 1) is None and st.find_miss == 0


def test_bed_out_of_view_is_said_once_a_pass_after_it_was_found():
    b, st = _bed(); sb = st.sb; sb.found = True
    assert b._mismatch(st, sb, 2, None, 1, 1) is None and b._mismatch(st, sb, 1, None, 1, 1) is None
    m = b._mismatch(st, sb, 0, None, 1, 1); assert m and 'out of view' in m
    assert b._mismatch(st, sb, 0, None, 1, 1) is None
    assert b._mismatch(st, sb, 12, None, 1, 1) is None and st.find_miss == 0          # dots back, bed found: reset
    b2, st2 = _bed()                                                                   # never found: no out-of-view talk
    for _ in range(5): assert b2._mismatch(st2, st2.sb, 1, None, 1, 1) is None


def test_hits_off_the_camera_bed_drop_it_after_three_and_the_rim_follows_the_dots():
    from app2.scan.bed_adapter import ScannerBed, SideState
    import bed as bedlib
    b = ScannerBed.__new__(ScannerBed); drops = []
    cam = dict(n=np.array([0., 0., -1.]), off=300.0, centre=np.array([0., 0., 300.]), x_axis=np.array([1., 0., 0.]))
    sb = NS(found=True, bed=NS(name='mat'), camera_bed=lambda ps: cam, drop_camera=lambda ps, idx, why: drops.append((ps, idx, why)) or ['dropped'])
    st = SideState(sb)
    def hit(cx):
        h = bedlib.Hit.__new__(bedlib.Hit); h.n = np.array([0., 0., -1.]); h.off = 300.0; h.centre = np.array([cx, 0., 300.]); h.x_axis = np.array([1., 0., 0.]); h.matched = np.ones(9, bool); return h
    far, near = hit(40.), hit(1.)
    assert bedlib.agree(near, cam) and not bedlib.agree(far, cam)
    assert b._rim_check(st, sb, far, 0, 10) == [] and b._rim_check(st, sb, far, 0, 20) == [] and st.rim_bad == 2
    assert b._rim_check(st, sb, near, 0, 30) == [] and st.rim_bad == 0                      # one agreeing look resets
    for i in (40, 50): b._rim_check(st, sb, far, 0, i)
    assert b._rim_check(st, sb, far, 0, 60) == ['dropped'] and drops[0][:2] == (0, 60) and '40 mm' in drops[0][2]
    # the rim: a fresh disagreeing hit within 15 frames is drawn instead of the camera bed
    side = NS(ps=0, idx=70, bed=st); b.bed = None
    import app2.scan.bed_adapter as A
    A._shape = lambda bed, x: 'shape'
    geo = ScannerBed.camera_geo(b, side, True); assert np.allclose(geo[2], far.centre)
    side.idx = 90; geo = ScannerBed.camera_geo(b, side, True); assert np.allclose(geo[2], cam['centre'])   # stale: back to the camera bed
