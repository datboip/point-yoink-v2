"""The matched bed supplies the live side's up direction; without one the table fit does, and nothing guesses otherwise."""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-bed-up-")) / "config.json")
import numpy as np
import pytest
from app2.scan.session import ScanSession


@pytest.mark.parametrize("normal,expected", [([0, 3, 4], (0, .6, .8)), ([0, -2, 0], (0, -1, 0)), ([0, 0, 0], None), ([float("nan"), 1, 0], None)])
def test_bed_up_uses_signed_matched_side_normal(normal, expected):
    session = ScanSession.__new__(ScanSession)
    session.bed = SimpleNamespace(platter=lambda side: (normal, 12, [0, 0, 0], 100, "bed"))
    result = session._view_bed_up(object())
    if expected is None: assert result is None
    else: assert np.allclose(result, expected)


def test_unmatched_bed_falls_back_to_the_table_fit_or_nothing():
    session = ScanSession.__new__(ScanSession)
    session.bed = SimpleNamespace(platter=lambda side: None); session.picked = None; session.recording = False
    assert session._view_bed_up(SimpleNamespace(table=None)) is None
    assert session._view_bed_up(None) is None
    side = SimpleNamespace(table=SimpleNamespace(plane=lambda: ([0, 0, 2], 5, [0, 0, 0], 50)))
    assert np.allclose(session._view_bed_up(side), (0, 0, 1))


def test_cloud_payload_carries_bed_direction_and_empty_clears_it():
    session = ScanSession.__new__(ScanSession)
    side = object(); events = []
    session.gl = True; session.home_seq = 4; session.remove_base = False; session.picked = side; session.recording = False
    session._touch_model = lambda sd: None
    session.bed = SimpleNamespace(platter=lambda sd: ([0, -1, 0], 12, [0, 0, 0], 100, 'bed'))
    session.host = SimpleNamespace(has=lambda sd: True, fused=True,
        cloud=lambda: (np.ones((3, 3), np.float32), np.ones((3, 3), np.float32), np.ones(3, np.uint8), .5))
    session._side_clouds = {}; session._no = lambda sd: 1; session._next_seq = lambda: 7
    session._emit = lambda event, payload: events.append((event, payload))
    session._emit_cloud(side)
    assert events[-1][1]['bed_up'] == (0., -1., 0.)
    session._emit_cloud(side, empty=True)
    assert events[-1][1]['bed_up'] is None
    assert len(events[-1][1]['P']) == 0


def test_saved_mesh_uses_its_own_side_bed():
    session = ScanSession.__new__(ScanSession)
    side = object(); events = []
    session.picked = side; session.recording = False; session.home_seq = 5
    session.host = SimpleNamespace(has=lambda sd: False)
    session.bed = SimpleNamespace(platter=lambda sd: ([1, 0, 0], 10, [0, 0, 0], 100, 'bed') if sd is side else None)
    session._next_seq = lambda: 8; session._emit = lambda event, payload: events.append((event, payload))
    session._side_mesh_ready(side, {'side_no': 2})
    assert events[-1][0] == 'side_mesh'
    assert events[-1][1]['bed_up'] == (1., 0., 0.)


def test_first_frames_take_up_from_the_bed_seen_while_framing():
    session = ScanSession.__new__(ScanSession)
    side = SimpleNamespace(table=None)
    session.bed = SimpleNamespace(platter=lambda s: None); session.picked = side; session.recording = True; session.table = None
    session._bed_live = dict(geo=([0, 0.6, 0.8], 300, [0, 0, 0], 100))     # the dot bed, normal pointing away from the camera
    session.last = dict(T=np.eye(4), ok=True)
    assert np.allclose(session._view_bed_up(side), (0, -0.6, -0.8))          # flipped to point back at the camera
    R = np.eye(4); R[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]; session.last = dict(T=R, ok=True)
    assert np.allclose(session._view_bed_up(side), R[:3, :3] @ [0, -0.6, -0.8])   # carried by the current pose
    session._bed_live = {}
    assert session._view_bed_up(side) is None
