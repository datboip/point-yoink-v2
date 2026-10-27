"""The Clone tab: picks fill source then destination, Apply adds one op, and the layout still builds."""
import json, types
import numpy as np
from app2.qt import build_step as B
from app2.qt import build_words as W


class Probe:
    _edit_run = B.BuildStepPage._edit_run
    _clone_pick = B.BuildStepPage._clone_pick; _clone_next = B.BuildStepPage._clone_next; _clone_clear = B.BuildStepPage._clone_clear
    _clone_apply = B.BuildStepPage._clone_apply; _clone_params = B.BuildStepPage._clone_params; _ctl = B.BuildStepPage._ctl
    def __init__(self):
        self.stage = "fused"; self._clone = {"fused": {"src": None, "dst": None, "next": "src"}}; self.calls = []; self.notes = []; self.views = 0
        self.panel = {"fused": types.SimpleNamespace(bodies={"Clone": types.SimpleNamespace(controls=[
            types.SimpleNamespace(c={"param": "diameter"}, value=lambda: 20.0), types.SimpleNamespace(c={"param": "mirror"}, value=lambda: "no")])})}
    def _cut_local(self, w): return list(w)
    def _clone_normal(self, p): return np.array([0.0, 0.0, 1.0])
    def _clone_view(self): self.views += 1
    def _list_token(self, st): return st
    def call(self, name, *a, then=None, **k): self.calls.append((name, a)); then and then(None)
    def notice(self, t, **k): self.notes.append(t)
    def start_run(self, r): self.went = r


def test_picks_fill_source_then_destination_and_apply_adds_one_op_and_resets():
    p = Probe(); p._clone_pick([1, 2, 3]); p._clone_pick([50, 2, 3])
    st = p._clone["fused"]; assert json.loads(st["src"])[0] == [1, 2, 3] and json.loads(st["dst"])[0] == [50, 2, 3]
    p._clone_apply("fused")
    name, args = p.calls[-1]; assert name == "add_op" and args[0] == "fused" and args[1]["tool"] == "clone"
    prm = args[1]["params"]; assert prm["diameter"] == 20.0 and prm["mirror"] == "no" and "src" in prm and "dst" in prm
    assert p._clone["fused"] == {"src": None, "dst": None, "next": "src"}
    assert p.went.publish and p.went.to is None and p.published is None


def test_apply_needs_both_spots():
    p = Probe(); p._clone_pick([1, 2, 3]); p._clone_apply("fused")
    assert p.calls == [] and p.notes == [W.SPOTS["clone_need"]]


def test_set_destination_button_redirects_the_next_pick():
    p = Probe(); p._clone_next("fused", "dst"); p._clone_pick([9, 9, 9])
    assert p._clone["fused"]["src"] is None and p._clone["fused"]["dst"] is not None


def test_outlines_are_drawn_at_both_spots_in_their_colors():
    class Q:
        _clone_outlines = B.BuildStepPage._clone_outlines; _clone_params = B.BuildStepPage._clone_params; _ctl = B.BuildStepPage._ctl
        stage = "fused"
        def _orient_item(self): return None, None
    q = Q(); q.panel = {"fused": types.SimpleNamespace(bodies={"Clone": types.SimpleNamespace(controls=[
        types.SimpleNamespace(c={"param": "diameter"}, value=lambda: 20.0), types.SimpleNamespace(c={"param": "shape"}, value=lambda: "cylinder")])})}
    q._clone = {"fused": {"src": None, "dst": None}}
    spot = lambda c: json.dumps([c, [0, 0, 1]])
    out = q._clone_outlines({"src": spot([0, 0, 0]), "dst": spot([100, 0, 0])})
    assert [m["colour"] for m in out] == ["#2fd06a", "#f0a030"] and [m["text"] for m in out] == ["Copy from here", "Copy goes here"]
    c = np.array(out[1]["corners"]); assert len(c) == 28 and abs(np.linalg.norm(c[:, :2] - [100, 0], axis=1).mean() - 10.0) < 1e-6   # a 20 mm circle at the destination
    assert q._clone_outlines({"src": None, "dst": None}) == []
