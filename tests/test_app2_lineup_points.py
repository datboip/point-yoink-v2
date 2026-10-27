"""Line-up Points tab (align via points): the pick state machine, the panel's list and button, the view's
pick tool and dots, the words. Offscreen, isolated config, no scanner."""
import os
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XMODIFIERS'] = '@im=none'
os.environ['POINTYOINK_NO_DEVICE'] = '1'
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_points_')) / 'config.json')

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt import theme, build_words as W
from app2.qt.lineup_panel import LineupPanel, PairPicks, TABS

A, B, C, D = [0, 0, 0], [40, 0, 0], [0, 30, 0], [10, 10, 25]


def rigid(p, t=(5.0, -3.0, 2.0)):
    c, s = np.cos(0.4), np.sin(0.4); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return list(R @ np.asarray(p, float) + t)


def test_pick_state_machine_like_1x():
    k = PairPicks()
    assert k.pick(1, B) == "half"                       # the first half on EITHER side
    assert k.pick(1, C) == "moved"                      # same side again: the half moves
    assert k.pick(0, A) == "pair"                       # the other side completes it, stored "to" first
    assert k.pairs == [[A, C]] and k.pending is None
    k.pick(0, B); assert k.dots()[-1] == (0, B, 2)      # the pending half is drawn as pair 2
    k.undo(); assert k.pending is None and len(k.pairs) == 1       # Undo point: the pending half first
    k.undo(); assert k.pairs == []                                 # then the last pair
    for p in (A, B, C): k.pick(0, p); k.pick(1, rigid(p))
    k.start_redo(1, 0)
    assert k.pick(1, D) == "wrong_side" and k.redo == (1, 0)       # a redo waits for its own side
    assert k.pick(0, [41, 0, 0]) == "redone" and k.pairs[1][0] == [41, 0, 0] and k.redo is None
    k.delete(0); assert len(k.pairs) == 2
    k.clear(); assert (k.pairs, k.pending, k.redo) == ([], None, None)


def test_problems_and_residuals():
    k = PairPicks()
    for p in (A, B): k.pick(0, p); k.pick(1, rigid(p))
    assert k.problem() == "need" and k.residuals() is None
    k.pick(0, [80, 0, 0]); k.pick(1, rigid([80, 0, 0])); assert k.problem() == "collinear"
    k.delete(2); k.pick(0, [0, 0, 0.1]); k.pick(1, rigid([0, 0, 0.1])); assert k.problem() == "duplicate"
    k.delete(2); k.pick(0, C); k.pick(1, rigid(C)); k.pick(0, D); k.pick(1, rigid(D))
    assert k.problem() is None and max(k.residuals()) < 1e-9          # an exact rigid move fits every pair
    k.pairs[3][1] = list(np.asarray(k.pairs[3][1]) + [0, 0, 20])       # one bad pick stands out
    r = k.residuals(); assert int(np.argmax(r)) == 3


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([]); theme.install(app)
    p = LineupPanel(); p.resize(420, 900)
    sides = [dict(id="s1", n=1, active=True), dict(id="s2", n=2, active=True), dict(id="s3", n=3, active=True)]
    ops = []
    p.set_sides(sides, {"s1": 1.0, "s2": 0.9, "s3": 0.3}, "s1", None, ops, [], W.SIDE_COLOURS); p.show_tab("Points")
    yield p
    p.deleteLater(); app.processEvents()


def test_points_panel_flow(panel):
    got = []; panel.points_apply.connect(lambda s, t, pr: got.append((s, t, pr)))
    assert "Points" in TABS and list(TABS).index("Points") == list(TABS).index("Flip") + 1
    panel.pt_side.box.setCurrentIndex(2); panel.pt_to.box.setCurrentIndex(0)
    assert panel.points_sides() == ("s3", "s1")
    assert not panel.pt_btn.isEnabled() and "3 more pairs" in panel.pt_why.text()
    assert "pair 1" in panel.pt_next.text()
    panel.add_pick(0, A); assert "side 3" in panel.pt_next.text()
    panel.add_pick(1, rigid(A))
    for p in (B, C): panel.add_pick(1, rigid(p)); panel.add_pick(0, p)
    assert panel.pt_rows.count() == 3 and panel.pt_btn.isEnabled() and "mm on average" in panel.pt_why.text()
    panel.pt_btn.click()
    assert got and got[0][0:2] == ("s3", "s1") and len(got[0][2]) == 3 and np.allclose(got[0][2][2][0], C)
    panel.set_enabled_tools(False); assert not panel.pt_btn.isEnabled()
    panel.set_enabled_tools(True); assert panel.pt_btn.isEnabled()
    panel.pt_to.box.setCurrentIndex(2)                                  # the same side twice: refused, picks cleared
    assert not panel.pt_btn.isEnabled() and panel.picks.pairs == [] and panel.pt_why.text() == TABS["Points"]["same"]
    panel.pt_to.box.setCurrentIndex(0); panel.add_pick(0, A); panel.pt_undo.click(); assert panel.picks.pending is None


def test_flip_words_are_true():
    f = TABS["Flip"]
    assert "lines everything up again" not in f["hint"] and "fits" in f["hint"]
    auto = f["controls"][0]["choices"][0][2]; assert "each flip" in auto and "fit" in auto   # it fits every flip, keeps the best
    for t in (TABS["Points"], f):
        for v in t.values():
            if isinstance(v, str): assert "—" not in v and "–" not in v and "colour" not in v.lower()


def test_fix_words():
    from app2.qt.build_step import fix_words
    pos = {"a": 1, "b": 3}
    assert fix_words({"tool": "flip", "side": "b", "axis": "auto", "chose": "y"}, pos) == "Side 3, auto: y axis"
    assert fix_words({"tool": "match", "side": "b", "to": "a", "pairs": [1, 2, 3, 4]}, pos, True) == "Match side 3 to side 1 (4 points)"


def test_view_pick_tool_and_dots():
    app = QApplication.instance() or QApplication([])
    from app2.qt.edit_view import EditView
    v = EditView(); v.resize(400, 300)
    v.set_tool("pairpick"); assert v.tool == "pairpick"
    v.set_markers([([0, 0, 0], "#ff5252", 1)]); assert len(v.markers) == 1
    v.overlay.resize(v.size()); v.overlay.repaint()                     # the dots paint without a GL scene
    v.set_only(["a"]); assert v.only == {"a"}; v.set_only(None); assert v.only is None
    v.deleteLater(); app.processEvents()


def test_matrix_recovers_the_transform_the_pairs_came_from():
    rng = np.random.default_rng(3)
    base = rng.uniform(-50, 50, (5, 3))
    a = np.radians(37); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]); t = np.array([12.0, -7.0, 30.0])
    moving = (base - t) @ R            # inverse of base = moving @ R.T + t
    p = PairPicks()
    for b, m in zip(base, moving): p.pairs.append([list(b), list(m)])
    T = p.matrix()
    assert T is not None
    assert np.allclose(moving @ T[:3, :3].T + T[:3, 3], base, atol=1e-6)
    assert PairPicks().matrix() is None


def test_sides_rows_say_what_was_already_done_by_hand(panel):
    sides = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": True}, {"id": "c", "n": 3, "active": True}]
    panel.set_sides(sides, {"a": 1.0, "b": .9, "c": .8}, "a", None, [], [], ["#e8744f", "#5b9df0", "#4cc86a"])
    panel.set_tags({"b": ["matched by points to side 1", "nudged"]})
    assert panel.rows["b"].tag.text() == "matched by points to side 1 · nudged" and not panel.rows["b"].tag.isHidden()
    assert panel.rows["a"].tag.isHidden() and panel.rows["c"].tag.isHidden()
    panel.set_tags({})
    assert panel.rows["b"].tag.isHidden()


def test_auto_dots_become_ordinary_editable_pairs(panel):
    panel.set_pairs([[[0, 0, 0], [1, 1, 1]], [[1, 0, 0], [2, 1, 1]], [[0, 1, 0], [1, 2, 1]]])
    assert len(panel.picks.pairs) == 3 and panel.picks.pending is None
    panel.picks.delete(1); assert len(panel.picks.pairs) == 2           # a bad one can be removed like any clicked pair
    got = []; panel.auto_points.connect(lambda *a: got.append(a)); panel.pt_auto.click()
    assert got and got[0][2] == 12


def test_fine_is_a_mark_on_a_pair_set_in_the_points_tab(panel):
    sides = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": True}, {"id": "c", "n": 3, "active": True}]
    panel.set_sides(sides, {"a": 1.0, "b": .9, "c": .8}, "a", None, [], [], ["#e8744f", "#5b9df0", "#4cc86a"])
    assert not hasattr(panel.rows["b"], "ok_c")                               # no per-row mark: fine only means something against a side
    got = []; panel.checked_changed.connect(got.append)
    panel.pt_side.box.setCurrentIndex(panel.pt_side.box.findData("b")); panel.pt_to.box.setCurrentIndex(panel.pt_to.box.findData("a"))
    panel.pt_fine.click()
    assert got[-1] == {"b": "a"} and "Checked 1 of 3" in panel.checked_l.text()
    assert panel.pt_side.value() == "c" and panel.pt_fine.text() == "This one is fine"                 # on to the next side, whose pair is not marked
    panel.pt_side.box.setCurrentIndex(panel.pt_side.box.findData("b")); assert panel.pt_fine.text() == "Fine against side 1 (tap to undo)"
    panel.pt_fine.click(); assert got[-1] == {} and panel.pt_fine.text() == "This one is fine"          # undo: no advance
    panel.set_checked({"c": "a"}); assert panel.checked_map() == {"c": "a"} and "Checked 1 of 3" in panel.checked_l.text()


def test_pending_use_and_how_scanned_survive_a_refresh():
    from PySide6.QtWidgets import QApplication
    from app2.qt.lineup_panel import LineupPanel
    app = QApplication.instance() or QApplication([])
    p = LineupPanel(); sides = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": True}]
    cols = ["#f00", "#0f0"]
    p.set_sides(sides, {"a": 1.0, "b": 0.9}, "a", None, [], [], cols); p.set_hints({})
    p.rows["b"].use_c.setChecked(False); p.rows["b"].up.setCurrentIndex(p.rows["b"].up.findData("flipped"))
    assert p.pending_sides() == [("b", False)] and p.hints_dirty()
    p.set_sides(sides, {"a": 1.0, "b": 0.9}, "a", None, [], [], cols); p.set_hints({})       # a refresh: nothing saved changed
    assert not p.rows["b"].use_c.isChecked() and p.pending_sides() == [("b", False)] and p.rows["b"].up.currentData() == "flipped"
    sides2 = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": False}]       # now the saved value changed: it is shown, no longer pending
    p.set_sides(sides2, {"a": 1.0}, "a", None, [], [], cols)
    assert not p.rows["b"].use_c.isChecked() and p.pending_sides() == []
    sides3 = [{"id": "a", "n": 1, "active": True}, {"id": "b", "n": 2, "active": True}]         # undone elsewhere: back on
    p.set_sides(sides3, {"a": 1.0, "b": 0.9}, "a", None, [], [], cols); assert p.rows["b"].use_c.isChecked()
