"""By hand: sides already cut by hand are marked and Next side skips them."""
import numpy as np
import json, os, tempfile, types
from app2.qt import build_step as B

class Probe:
    _hand_cut_ids = B.BuildStepPage._hand_cut_ids
    _pager_note = B.BuildStepPage._pager_note
    def __init__(self, root, sides, k):
        self.host_root = root; self.side_k = k; self.msgs = []
        self.snap = types.SimpleNamespace(sides=sides)

def mk(manual):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "stages"))
    json.dump({"sides": {"platter": {i: {"method": "manual"} for i in manual}}}, open(os.path.join(d, "stages", "recipe.json"), "w"))
    return d

def test_hand_cut_ids_and_note():
    sides = [{"id": "a", "active": True}, {"id": "b", "active": True}, {"id": "c", "active": False}]
    p = Probe(mk(["a"]), sides, 1)
    assert p._hand_cut_ids() == {"a"}
    assert p._pager_note(1) == "base cut by hand" and p._pager_note(2) == "" and p._pager_note(3) == "left out"

def test_missing_recipe_is_empty():
    assert Probe(None, [], 1)._hand_cut_ids() == set()


def test_cut_points_without_a_model_matrix_stay_as_clicked():
    class Q:
        _cut_local = B.BuildStepPage._cut_local; _cut_display = B.BuildStepPage._cut_display
        def _orient_item(self): return "main", {}
    q = Q()
    assert q._cut_local([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0] and q._cut_display([[1, 2, 3]]) == [[1, 2, 3]]


def test_cut_warning_when_the_plane_runs_through_the_part():
    class Q: _cut_in_the_part = B.BuildStepPage._cut_in_the_part
    E = np.array([[0, 0, z] for z in range(0, 101)], float); plane = {"nx": 0, "ny": 0, "nz": 1, "d": 50.0}
    assert Q()._cut_in_the_part(E, E[:, 2] < 50, plane)                       # removes 0..49: deep under the plane
    assert not Q()._cut_in_the_part(E, (E[:, 2] < 50) & (E[:, 2] > 45), plane)   # a thin layer under it


from types import SimpleNamespace


class _PendingProbe:
    """Just enough of the page for _platter_pending: a done platter node, no edits in the panel."""
    _platter_pending = B.BuildStepPage._platter_pending

    def __init__(self, spots, by_hand=True, changed=False, status="done"):
        self._spots = {"s1": spots}; self._by_hand = by_hand
        self.snap = SimpleNamespace(node=lambda key: {"status": status})
        self.panel = {"side": SimpleNamespace(bodies={"Platter": SimpleNamespace(changed=lambda: changed)})}

    def _side(self): return {"id": "s1"}
    def _spots_on(self): return self._by_hand


def test_by_hand_keeps_the_base_on_screen_until_three_spots_are_picked_and_cut():
    assert _PendingProbe([])._platter_pending() is True                      # nothing to click on otherwise
    assert _PendingProbe([[0, 0, 0], [1, 0, 0]])._platter_pending() is True
    assert _PendingProbe([[0, 0, 0], [1, 0, 0], [0, 1, 0]])._platter_pending() is False   # cut run from these spots: show the result
    assert _PendingProbe([[0, 0, 0], [1, 0, 0], [0, 1, 0]], changed=True)._platter_pending() is True
    assert _PendingProbe([], by_hand=False)._platter_pending() is False      # Auto, done and unchanged: the cut result
    assert _PendingProbe([], by_hand=False, status="stale")._platter_pending() is False   # Auto, waiting to rebuild: still the clean object, never the base
    assert _PendingProbe([], by_hand=False, changed=True)._platter_pending() is True      # a Margin being set: the violet
