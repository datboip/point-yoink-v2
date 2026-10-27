"""A click on the model reaches By hand and Orient (the connection was once lost in an edit)."""
from app2.qt import build_step as B

class P:
    _surface_pick = B.BuildStepPage._surface_pick
    def _near_model(s, w): return True
    def notice(s, *a, **k): s.got.append(("notice",))
    def __init__(s, spots, orient, cut=False): s.a = spots; s.o = orient; s.c = cut; s.got = []
    def _cut_on(s): return s.c
    def _clone_on(s): return False
    def _fix_on(s): return False
    def _mirror_on(s): return False
    def _cut_add(s, w): s.got.append(("cut", w))
    def _spots_on(s): return s.a
    def _orient_on(s): return s.o
    def _spot_add(s, w): s.got.append(("spot", w))
    def _orient_add(s, w): s.got.append(("orient", w))

def test_dispatch():
    p = P(True, True); p._surface_pick([1, 2, 3]); assert p.got == [("spot", [1, 2, 3])]
    p = P(False, True); p._surface_pick([1, 2, 3]); assert p.got == [("orient", [1, 2, 3])]
    p = P(False, False); p._surface_pick([1, 2, 3]); assert p.got == []
    p = P(True, True, cut=True); p._surface_pick([1, 2, 3]); assert p.got == [("cut", [1, 2, 3])]

def test_connected_in_source():
    import inspect
    assert "vp.main.view.surface_picked.connect(self._surface_pick)" in inspect.getsource(B.BuildStepPage.__init__)


def test_click_off_the_model_is_dropped_and_only_the_second_one_says_so():
    p = P(True, False); p._near_model = lambda w: False
    p._surface_pick([1, 2, 3]); assert p.got == []
    p._surface_pick([1, 2, 3]); assert p.got == [("notice",)]
    p._near_model = lambda w: True; p._surface_pick([1, 2, 3]); p.got.clear()
    p._near_model = lambda w: False; p._surface_pick([1, 2, 3]); assert p.got == []        # a good click resets the count

def test_near_model_distance():
    import numpy as np, types
    class Q:
        _near_model = B.BuildStepPage._near_model; stage = "side"
        def _cut_local(s, w): return w
    q = Q(); q.vp = types.SimpleNamespace(view=types.SimpleNamespace(elements=(np.array([[0, 0, 0], [10, 0, 0]], float),)))
    assert q._near_model([0, 0, 5.0]) and not q._near_model([0, 0, 50.0])


def test_blank_pick_click_notices_from_the_second_one():
    class Q:
        _pick_missed = B.BuildStepPage._pick_missed
        got = []
        def _cut_on(s): return True
        def _clone_on(s): return False
        def _fix_on(s): return False
        def _mirror_on(s): return False
        def _spots_on(s): return False
        def _orient_on(s): return False
        def notice(s, *a, **k): s.got.append(a)
    q = Q(); q.got = []
    q._pick_missed(); assert q.got == []
    q._pick_missed(); assert len(q.got) == 1 and "model" in q.got[0][0]
    q._pick_missed(); assert len(q.got) == 2

def test_report_is_always_in_the_menu_states():
    import inspect
    from app2.qt import shell
    assert "report=True" in inspect.getsource(shell.MainWindow._menu_states)
