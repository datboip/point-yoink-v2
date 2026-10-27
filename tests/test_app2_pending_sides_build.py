"""A build started from YOLO Yoink or Rebuild applies Line-up's pending Use / How scanned choices first."""
import types
from app2.qt import build_step as B


class Probe:
    _with_pending_sides = B.BuildStepPage._with_pending_sides
    def __init__(self, pending, hints, sides):
        self.panel = {"lineup": types.SimpleNamespace(pending_sides=lambda: pending, hints_dirty=lambda: hints is not None, pending_hints=lambda: hints or "")}
        self.snap = types.SimpleNamespace(sides=sides); self.calls = []; self.notes = []; self.went = 0
    def call(self, name, *a, then=None, fail=None, **k): self.calls.append((name, a)); then and then(None)
    def notice(self, text, **k): self.notes.append(text)


SIDES = [{"id": "a", "active": True}, {"id": "b", "active": True}, {"id": "c", "active": True}]


def test_nothing_pending_just_goes_on():
    p = Probe([], None, SIDES); p._with_pending_sides(lambda: setattr(p, "went", p.went + 1))
    assert p.went == 1 and p.calls == []


def test_use_off_is_applied_before_the_build():
    p = Probe([("c", False)], None, SIDES); p._with_pending_sides(lambda: setattr(p, "went", p.went + 1))
    assert p.calls == [("h_apply", ([("leave_out", ("c",), None)],))] and p.went == 1


def test_hints_are_applied_too():
    p = Probe([], "b:flipped", SIDES); p._with_pending_sides(lambda: setattr(p, "went", p.went + 1))
    assert p.calls[0][1][0] == [("set", ({"lineup.place.hints": "b:flipped"},), None)] and p.went == 1


def test_leaving_every_side_out_is_refused():
    p = Probe([("a", False), ("b", False), ("c", False)], None, SIDES); p._with_pending_sides(lambda: setattr(p, "went", p.went + 1))
    assert p.went == 0 and p.calls == [] and p.notes
