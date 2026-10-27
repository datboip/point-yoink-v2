"""Smart frames: a frame is dropped only when the view is nearly all solid, one in ten is kept, and frames after a gain change are skipped."""
import time
import numpy as np
from app2.scan.session import ScanSession


class _Pc:
    def __init__(self, P, N): self.point = type("p", (), dict(positions=_T(P), normals=_T(N)))()


class _T:
    def __init__(self, a): self.a = a
    def cpu(self): return self
    def numpy(self): return self.a


class _Trk:
    def __init__(self, P, lv): self.model = _Pc(P, np.tile([0, 0, -1.0], (len(P), 1))); self.lv = lv
    def coverage_levels(self, P): return self.lv


def _session(levels):
    rng = np.random.default_rng(1); n = 4000
    P = np.c_[rng.uniform(-0.02, 0.02, n), rng.uniform(-0.02, 0.02, n), np.full(n, 0.3)]
    s = ScanSession.__new__(ScanSession); s.cal = (900.0, 900.0, 400.0, 300.0); s._gain_t = 0.0
    s._smart = dict(seen=0, skipped=0, lv=None, at=0); s._log = lambda *_: None
    s.host = type("h", (), dict(raw=lambda self: _Trk(P, levels)))()
    return s


R = dict(ok=True, T=np.eye(4))


def test_saturated_view_is_dropped_but_one_in_ten_is_kept():
    s = _session(np.full(4000, 2, np.int8))
    kept = [not s._smart_skip(None, R) for _ in range(30)]
    assert kept.count(True) == 3 and kept[9] and kept[19] and kept[29]


def test_thin_view_is_always_kept():
    s = _session(np.zeros(4000, np.int8))
    assert all(not s._smart_skip(None, R) for _ in range(20))


def test_lost_frames_and_a_gain_change_window():
    s = _session(np.zeros(4000, np.int8))
    assert s._smart_skip(None, dict(ok=False, T=np.eye(4))) is False
    s._gain_t = time.monotonic(); assert s._smart_skip(None, R) is True
    s._gain_t = time.monotonic() - 2; assert s._smart_skip(None, R) is False


def test_little_model_keeps_everything():
    s = _session(np.full(4000, 2, np.int8)); s.host = type("h", (), dict(raw=lambda self: None))()
    assert not s._smart_skip(None, R)
