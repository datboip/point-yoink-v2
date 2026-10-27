"""YOLO leaves out a side that clearly does not line up, says so, and the Line-up card keeps saying so until Use is turned back on."""
import json, os, tempfile, types
from app2.qt import build_step as B
from app2.qt import build_words as W


class Probe:
    _onetap_flagged = B.BuildStepPage._onetap_flagged
    _auto_out_path = B.BuildStepPage._auto_out_path
    _auto_out_load = B.BuildStepPage._auto_out_load
    _auto_out_save = B.BuildStepPage._auto_out_save
    def __init__(self, active):
        self.host_root = tempfile.mkdtemp(); self.snap = types.SimpleNamespace(sides=[{"id": k, "active": v} for k, v in active.items()])
        self.calls = []; self.notes = []; self.went = 0
    def call(self, name, *a, then=None, fail=None, **k): self.calls.append((name, a)); then and then(None)
    def notice(self, text, **k): self.notes.append((text, k))


R = {"bad": ["s11"], "bad_n": [11], "sides": {"11": {"agree": 0.721, "alone": 0.104}, "2": {"agree": 0.95, "alone": 0.0}}}


def test_a_flagged_side_is_left_out_warned_about_and_recorded_then_the_build_goes_on():
    p = Probe({"s11": True, "s2": True}); go = lambda: setattr(p, "went", p.went + 1)
    p._onetap_flagged(R, go)
    assert p.calls == [("h_apply", ([("leave_out", ("s11",), None)],))] and p.went == 1
    text, k = p.notes[0]; assert "Side 11 was left out" in text and "72%" in text and "Use back on" in text and k["secs"] >= 30
    p.snap.sides[0]["active"] = False                                       # the engine now has it out
    assert p._auto_out_load() == {"s11": {"n": 11, "agree": 72, "alone": 10.4}}
    p.snap.sides[0]["active"] = True                                        # the user turned Use back on: the record goes
    assert p._auto_out_load() == {}


def test_nothing_flagged_just_goes_on_with_no_warning():
    p = Probe({"s11": True}); go = lambda: setattr(p, "went", p.went + 1)
    p._onetap_flagged({"bad": [], "bad_n": [], "sides": {}}, go); assert p.went == 1 and p.calls == [] and p.notes == []


def test_several_sides_are_named_together():
    p = Probe({"a": True, "b": True}); go = lambda: None
    p._onetap_flagged({"bad": ["a", "b"], "bad_n": [3, 7], "sides": {"3": {"agree": 0.8, "alone": 0.0}, "7": {"agree": 0.6, "alone": 0.0}}}, go)
    assert "Sides 3 and 7 were left out" in p.notes[0][0] and "60%" in p.notes[0][0]


def test_line_up_card_badge_follows_the_record_and_use():
    from PySide6.QtWidgets import QApplication
    from app2.qt.lineup_panel import LineupPanel
    app = QApplication.instance() or QApplication([]); lp = LineupPanel(); cols = ["#f00", "#0f0"]
    sides = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": False}]
    lp.set_sides(sides, {"a": 1.0}, "a", None, [], [], cols); lp.set_auto_out({"b": {"n": 2, "agree": 72, "alone": 10.4}}); app.processEvents()
    assert lp.rows["b"].auto.isVisibleTo(lp.rows["b"]) and "72%" in lp.rows["b"].auto.text() and not lp.rows["a"].auto.isVisibleTo(lp.rows["a"])
    lp.rows["b"].use_c.setChecked(True); lp.set_auto_out({"b": {"n": 2, "agree": 72, "alone": 10.4}})
    assert not lp.rows["b"].auto.isVisibleTo(lp.rows["b"])
