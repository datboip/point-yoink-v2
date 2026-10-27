"""Smooth area tab: one pick sets the spot, Apply adds one local_fix op, the Edit button exists."""
import json, types, inspect
import numpy as np
from app2.qt import build_step as B, build_words as W, edit_page


class Probe:
    _edit_run = B.BuildStepPage._edit_run
    _fix_pick = B.BuildStepPage._fix_pick; _fix_clear = B.BuildStepPage._fix_clear; _fix_apply = B.BuildStepPage._fix_apply
    _fix_params = B.BuildStepPage._fix_params; _ctl = B.BuildStepPage._ctl
    def __init__(self):
        self.stage = "fused"; self._fix = {"fused": {"spot": None}}; self.calls = []; self.notes = []
        ctl = lambda name, val: types.SimpleNamespace(c={"param": name}, value=lambda: val)
        self.panel = {"fused": types.SimpleNamespace(bodies={"Smooth area": types.SimpleNamespace(controls=[ctl("diameter", 20.0), ctl("strength", 0.8), ctl("passes", 8)])})}
    def _cut_local(self, w): return list(w)
    def _clone_normal(self, p): return np.array([0.0, 0.0, 1.0])
    def _fix_view(self): pass
    def _list_token(self, st): return st
    def call(self, name, *a, then=None, **k): self.calls.append((name, a)); then and then(None)
    def notice(self, t, **k): self.notes.append(t)
    def start_run(self, r): self.went = r


def test_pick_then_apply_adds_one_op_with_the_slider_values_and_resets():
    p = Probe(); p._fix_pick([5, 6, 7]); assert json.loads(p._fix["fused"]["spot"])[0] == [5, 6, 7]
    p._fix_apply("fused"); name, args = p.calls[-1]
    assert name == "add_op" and args[1]["tool"] == "local_fix"
    prm = args[1]["params"]; assert prm["diameter"] == 20.0 and prm["strength"] == 0.8 and prm["passes"] == 8 and "spot" in prm
    assert p._fix["fused"]["spot"] is None
    assert p.went.publish and p.went.to is None and p.published is None


def test_apply_without_a_spot_says_so():
    p = Probe(); p._fix_apply("fused"); assert p.calls == [] and p.notes == [W.SPOTS["fix_need"]]


def test_it_is_an_edit_button_and_an_open_on_demand_tab_not_a_rail_row():
    assert W.MORE_TOOLS["local_fix"]["name"] == "Smooth area"
    assert [r for r in W.STAGES if "fix" in r["id"]] == [] and "Smooth area" in inspect.getsource(edit_page)
