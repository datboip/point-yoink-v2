"""Mirror tab: spots make the plane, Apply adds one mirror op with the plane and the sliders, Auto-find sets a plane without spots."""
import types
import numpy as np
from app2.qt import build_step as B, build_words as W, edit_page
from scanner.stages.tools import mirror as M


class Probe:
    _edit_run = B.BuildStepPage._edit_run
    _mirror_plane = B.BuildStepPage._mirror_plane; _mirror_apply = B.BuildStepPage._mirror_apply; _mirror_params = B.BuildStepPage._mirror_params
    _mirror_add = B.BuildStepPage._mirror_add; _mirror_undo = B.BuildStepPage._mirror_undo; _mirror_clear = B.BuildStepPage._mirror_clear; _ctl = B.BuildStepPage._ctl
    def __init__(self):
        self.stage = "fused"; self._mspots = {"fused": []}; self._mplane = {"fused": None}; self.calls = []; self.notes = []; self._tools_on = True; self.busy_call = None
        ctl = lambda name, val: types.SimpleNamespace(c={"param": name}, value=lambda: val)
        self.panel = {"fused": types.SimpleNamespace(bodies={"Mirror": types.SimpleNamespace(controls=[ctl("fit", "yes"), ctl("gap_mm", 2.0)])})}
        self.vp = types.SimpleNamespace(view=types.SimpleNamespace(tools_on=True))
    def running(self): return False
    def _mirror_on(self): return True
    def _cut_local(self, w): return list(w)
    def _mirror_view(self): pass
    def _list_token(self, st): return st
    def call(self, name, *a, then=None, **k): self.calls.append((name, a)); then and then(None)
    def notice(self, t, **k): self.notes.append(t)
    def start_run(self, r): self.went = r


def test_three_spots_make_the_plane_and_apply_adds_one_op():
    p = Probe()
    for s in ([0, 0, 0], [0, 10, 0], [0, 0, 10]): p._mirror_add(s)
    n, d = p._mirror_plane("fused"); assert abs(abs(n[0]) - 1) < 1e-9 and abs(d) < 1e-9
    p._mirror_apply("fused"); name, args = p.calls[-1]
    assert name == "add_op" and args[1]["tool"] == "mirror"
    prm = args[1]["params"]; assert abs(abs(prm["nx"]) - 1) < 1e-9 and prm["gap_mm"] == 2.0 and prm["fit"] == "yes"
    assert p._mspots["fused"] == [] and p.notes == [W.SPOTS["mirror_done"]]
    assert p.went.publish and p.went.to is None and p.published is None


def test_two_spots_or_spots_in_a_line_are_not_a_plane():
    p = Probe(); p._mirror_add([0, 0, 0]); p._mirror_add([1, 0, 0]); assert p._mirror_plane("fused") is None
    p._mirror_add([2, 0, 0]); assert p._mirror_plane("fused") is None
    p._mirror_apply("fused"); assert p.calls == [] and p.notes == [W.SPOTS["mirror_need"]]


def test_auto_plane_is_used_when_there_are_no_spots_and_a_spot_replaces_it():
    p = Probe(); p._mplane["fused"] = (np.array([0, 1.0, 0]), 3.0)
    n, d = p._mirror_plane("fused"); assert n[1] == 1.0 and d == 3.0
    p._mirror_add([0, 0, 0]); assert p._mplane["fused"] is None
    p._mirror_undo("fused"); p._mirror_clear("fused"); assert p._mirror_plane("fused") is None


def test_edit_has_a_mirror_button_that_works_on_points_too():
    import inspect
    src = inspect.getsource(edit_page)
    assert '"Mirror", "Mirror"' in src and 'if label in ("Clone", "Smooth area", "Mirror")' in src
    assert W.MORE_TOOLS["mirror"]["name"] == "Mirror"
