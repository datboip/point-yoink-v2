"""A failed relearn must not present the retained map as a fresh success."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='bed-status-')) / 'config.json')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scanner'))
from app2.scan.bed_adapter import ScannerBed
from app2.scan.ui_model import ScanState, bed_card, buttons


def test_failed_relearn_keeps_map_and_failure():
    adapter = ScannerBed.__new__(ScannerBed)
    learner = object(); old = object()
    adapter.learner = learner; adapter.closing = False; adapter.bed = old
    failure = dict(state='failed', msg='The table turned only 243 of 360 deg.')
    lines = adapter._learn_result(dict(learner=learner, res=failure), [])
    assert adapter.bed is old and adapter.learner is None
    assert adapter.learn_status == failure and failure['msg'] in lines[0]
    state = ScanState(bed_available=True, bed=old, learn_status=failure, n_sides=0)
    assert bed_card(state)['sub'] == 'Learn failed; using the last learned bed'
    assert 'Learn bed failed' in buttons(state)['hint']
    state.bed = None
    assert bed_card(state)['sub'] == 'Learn failed; try Learn bed again'


def test_success_clears_failure_and_stale_results_are_ignored():
    adapter = ScannerBed.__new__(ScannerBed)
    learner = object(); new = list(range(29))
    class Model(list):
        radius_mm = 100
        self_ambiguity = None
    new = Model(new)
    adapter.learner = learner; adapter.closing = False; adapter.bed = None
    adapter.learn_status = dict(state='failed')
    adapter._install = lambda bed, sides: setattr(adapter, 'bed', bed)
    assert adapter._learn_result(dict(learner=object(), res=dict(state='failed')), []) == []
    adapter._learn_result(dict(learner=learner, res=dict(state='done'), bed=new), [])
    assert adapter.bed is new and adapter.learn_status is None


def test_learning_hint_says_the_table_speed_and_whether_it_is_enough():
    from app2.scan import ui_model as U
    def hint(speed):
        s = U.ScanState.__new__(U.ScanState)
        for k in U.ScanState.__slots__: setattr(s, k, None)
        s.busy = None; s.learning = True; s.learn_status = dict(deg=120, dots=14, state="learn", speed=speed, need=2.5)
        s.recording = False; s.building = False; s.capture_mode = "continuous"; s.picked = None; s.n_sides = 0
        try: return U.buttons(s)["hint"]
        except AttributeError: return U.buttons_for(s)["hint"]
    assert "too slow" in hint(1.4) and "Turn the table up" in hint(1.4)
    assert "fine" in hint(4.0) and "too slow" not in hint(4.0)
    assert "deg/s" not in hint(None)


def test_counted_dots_give_a_verdict():
    """2026-10-25: the user may say how many dots the bed has; the learning line counts toward it, well short is not saved."""
    class Model(list):
        radius_mm = 100; self_ambiguity = None; nominal = None
    def adapter(want):
        a = ScannerBed.__new__(ScannerBed); a.learner = object(); a.closing = False; a.bed = None; a.learn_want = want; a.learn_status = None
        a._install = lambda bed, sides: setattr(a, 'bed', bed); return a
    a = adapter(32); a._learn_result(dict(learner=a.learner, res=dict(state='learning', deg=120, dots=18)), [])
    st = ScanState(bed_available=True, bed=None, learning=True, learn_status=a.learn_status, n_sides=0)
    assert bed_card(st)['sub'] == 'Learning: 120 of 360°, 18 of 32 dots'
    a = adapter(32); lines = a._learn_result(dict(learner=a.learner, res=dict(state='done'), bed=Model(range(20))), [])
    assert a.bed is None and 'Found 20 of the 32' in lines[0] and a.learn_status['state'] == 'failed'
    a = adapter(32); lines = a._learn_result(dict(learner=a.learner, res=dict(state='done'), bed=Model(range(30))), [])
    assert a.bed is not None and '30 of the 32' in lines[0]
    a = adapter(32); lines = a._learn_result(dict(learner=a.learner, res=dict(state='done'), bed=Model(range(35))), [])
    assert a.bed is not None and 'more than the 32' in lines[0]
