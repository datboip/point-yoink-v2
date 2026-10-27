"""Line-up Nudge's rotate rings (Bambu Studio style): ring hit test, drag angle from screen motion, Shift snap, the Rotate
(absolute) angles (pure math, no GL); the panel's Rotate boxes, Keep exactly as placed and Apply; the view's rings and
mouse routing (offscreen, no GL scene). Isolated config, no scanner."""
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XMODIFIERS'] = '@im=none'
os.environ['POINTYOINK_NO_DEVICE'] = '1'
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_rings_')) / 'config.json')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scanner"))

import numpy as np
import pytest
from app2.gl_core import Camera
from app2.qt import rotate_gizmo as RG

W_, H_ = 800, 600
R0 = 20.0


def cam(elev=0.0, turn=0.0):
    c = Camera(); c.frame(np.array([-30.0, -30, -30]), np.array([30.0, 30, 30]))
    if elev or turn: c.orbit(turn / 0.35, elev / 0.35)
    return c


def px(c, P):
    x, y, _z = c.project(np.atleast_2d(np.asarray(P, float)), W_, H_); return float(x[0]), float(y[0])


def on_ring(k, deg, r=R0):
    u, v = RG.basis(k); t = np.radians(deg); return r * (np.cos(t) * u + np.sin(t) * v)


def test_ring_hit():
    c = cam()                                        # looking down -Z: Z ring face on, X and Y rings edge-on lines
    assert RG.ring_hit(c, np.zeros(3), R0, px(c, on_ring(2, 45)), W_, H_) == 2
    assert RG.ring_hit(c, np.zeros(3), R0, px(c, on_ring(0, 60)), W_, H_) == 0       # the X ring seen as a vertical line
    assert RG.ring_hit(c, np.zeros(3), R0, px(c, on_ring(1, 150)), W_, H_) == 1      # the Y ring as a horizontal line
    x, y = px(c, on_ring(2, 45)); assert RG.ring_hit(c, np.zeros(3), R0, (x + 30, y + 30), W_, H_) is None
    assert RG.ring_hit(c, np.zeros(3), R0, (x + 5, y), W_, H_) == 2                   # within HIT_PX


def test_drag_angle_face_on_unwraps_and_snaps():
    c = cam(); d = RG.RingDrag(c, np.zeros(3), R0, 2, px(c, on_ring(2, 0)), W_, H_)
    assert not d.edge
    assert d.angle(px(c, on_ring(2, 30))) == pytest.approx(30.0, abs=0.1)            # right hand about +Z
    assert d.angle(px(c, on_ring(2, -40, 3 * R0))) == pytest.approx(-40.0, abs=0.1)  # outside the ring: free, any distance
    for a in range(0, 351, 50): got = d.angle(px(c, on_ring(2, a)))
    assert got == pytest.approx(350.0, abs=0.1)                                       # past 180: unwrapped
    d2 = RG.RingDrag(c, np.zeros(3), R0, 2, px(c, on_ring(2, 0)), W_, H_)
    assert d2.angle(px(c, on_ring(2, 37)), snap_on=True) == 30.0 and d2.angle(px(c, on_ring(2, 38)), snap_on=True) == 45.0
    assert RG.snap(-7.4) == 0.0 and RG.snap(-8) == -15.0


def test_snap_rule_by_cursor_radius():
    """On the ring or anywhere outside it: free; inward over the tick band: 5 deg; further in by the spokes: 30 deg."""
    for f in (1.0, 1.2, 3.0, 50.0, RG.FREE_IN, None): assert RG.snap_step(f) == 0.0
    for f in (RG.FREE_IN - 1e-9, 0.7, RG.SPOKE_OUT): assert RG.snap_step(f) == 5.0
    for f in (RG.SPOKE_OUT - 1e-9, 0.3, 0.0): assert RG.snap_step(f) == 30.0
    assert (RG.snap_step(1.5, True), RG.snap_step(0.7, True), RG.snap_step(0.3, True)) == (15.0, 15.0, 30.0)
    assert RG.snap(27.4, 0.0) == 27.4 and RG.snap(27.4, 5.0) == 25.0 and RG.snap(46.0, 30.0) == 60.0


def test_drag_snaps_by_where_the_cursor_is():
    c = cam(); d = RG.RingDrag(c, np.zeros(3), R0, 2, px(c, on_ring(2, 0)), W_, H_)
    assert d.angle(px(c, on_ring(2, 27.3))) == pytest.approx(27.3, abs=0.1)                  # on the ring: free
    assert d.angle(px(c, on_ring(2, 27.3, 2.5 * R0))) == pytest.approx(27.3, abs=0.1)        # outside: free
    assert d.angle(px(c, on_ring(2, 27.3, 0.7 * R0))) == 25.0 and d.angle(px(c, on_ring(2, 28.2, 0.7 * R0))) == 30.0   # ticks
    assert d.angle(px(c, on_ring(2, 44.0, 0.3 * R0))) == 30.0 and d.angle(px(c, on_ring(2, 46.0, 0.3 * R0))) == 60.0   # spokes
    assert d.frac == pytest.approx(0.3, abs=0.01)
    assert RG.label(1, -5.0) == "Y: 355.00" and RG.label(2, 30.0) == "Z: 30.00"


def test_protractor_ticks_and_spokes():
    segs = RG.protractor(np.zeros(3), R0, 2, a0=10.0); kinds = [k for _a, _b, k in segs]
    assert kinds.count("spoke") == 12 and kinds.count("tick") == 72 and kinds.count("circle") == 72
    assert all(abs(a[2]) < 1e-12 and abs(b[2]) < 1e-12 for a, b, _k in segs)                # in the Z ring's plane
    ticks = [(a, b) for a, b, k in segs if k == "tick"]
    ln = [round(float(np.linalg.norm(b - a)) / R0, 2) for a, b in ticks]
    assert ln[0] == ln[6] == round(RG.FREE_IN - RG.SPOKE_OUT, 2) and ln[3] == round(RG.FREE_IN - 0.68, 2) and ln[1] == round(RG.FREE_IN - 0.74, 2)
    ang = np.degrees(np.arctan2(ticks[0][1][1], ticks[0][1][0])); assert ang == pytest.approx(10.0)                  # 0 at the grab


def test_drag_angle_edge_on_follows_the_tangent():
    c = cam(elev=10.0)                               # the X ring still edge-on, its front and back apart on screen
    front = on_ring(0, 90); d = RG.RingDrag(c, np.zeros(3), R0, 0, px(c, front), W_, H_)
    assert d.edge
    assert d.angle(px(c, RG.axis_rot(0, 5.0) @ front)) == pytest.approx(5.0, abs=0.5)
    assert d.angle(px(c, RG.axis_rot(0, -5.0) @ front)) == pytest.approx(-5.0, abs=0.5)


def test_off_centre_and_tilted_view():
    c = cam(elev=35.0, turn=-50.0); ce = np.array([4.0, -3.0, 2.0])
    for k in range(3):
        p0 = ce + on_ring(k, 20); assert RG.ring_hit(c, ce, R0, px(c, p0), W_, H_) == k
        d = RG.RingDrag(c, ce, R0, k, px(c, p0), W_, H_)
        if not d.edge: assert d.angle(px(c, ce + on_ring(k, 65))) == pytest.approx(45.0, abs=0.2)


def test_euler_round_trip():
    from stages.tools import edits as E
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.uniform([-179, -89, -179], [179, 89, 179]); R = RG.from_euler(*a)
        assert np.allclose(R, E.r_xyz(a), atol=1e-12) and np.allclose(RG.euler_xyz(R), a, atol=1e-6)
    assert np.allclose(RG.from_euler(*RG.euler_xyz(RG.from_euler(10, 90, 20))), RG.from_euler(10, 90, 20), atol=1e-9)   # gimbal
    assert np.allclose(RG.axis_rot(2, 90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


# ------------------------------------------------------------------------------------------------ panel and view (offscreen)
@pytest.fixture
def panel():
    from PySide6.QtWidgets import QApplication
    from app2.qt import theme, build_words as W
    from app2.qt.lineup_panel import LineupPanel
    app = QApplication.instance() or QApplication([]); theme.install(app)
    p = LineupPanel(); p.resize(420, 1000)
    sides = [dict(id="s1", n=1, active=True), dict(id="s2", n=2, active=True)]
    p.set_sides(sides, {}, "s1", None, [("Side 2, auto: y axis", "lineup.fix", 0)],
                [("Side 2: turned 30 deg", "lineup.fix", 1), ("Side 1: 1,0,0 mm, 0,0,0 deg", "lineup.adjust", 0)], W.SIDE_COLOURS)
    p.show_tab("Nudge")
    yield p
    p.deleteLater(); app.processEvents()


def test_panel_rotate_boxes_and_apply(panel):
    prev, done, gone = [], [], []
    panel.nudge_preview.connect(lambda s, t, R: prev.append((s, t, np.array(R))))
    panel.nudge_apply.connect(lambda s, t, R, keep: done.append((s, t, np.array(R), keep)))
    panel.remove_op.connect(lambda tok, i: gone.append((tok, i)))
    panel.nudge_side.box.setCurrentIndex(1); assert prev[-1][0] == "s2" and np.allclose(prev[-1][2], np.eye(3))
    panel.ring_turn(2, 30.0, False)                                  # a drag in progress: previewed, not kept yet
    assert np.allclose(prev[-1][2], RG.axis_rot(2, 30)) and np.allclose(panel.rot, np.eye(3))
    assert [round(b.value(), 1) for b in panel.abs_] == [0.0, 0.0, 30.0] and "30.0" in panel.turned.text()
    panel.ring_turn(2, 30.0, True); assert np.allclose(panel.rot, RG.axis_rot(2, 30))
    panel.rel[0].setValue(90.0); panel.rel[0].editingFinished.emit()                   # Rotate (relative): about world X
    assert np.allclose(panel.rot, RG.axis_rot(0, 90) @ RG.axis_rot(2, 30)) and panel.rel[0].value() == 0.0
    for b, x in zip(panel.abs_, (10.0, 20.0, 30.0)): b.setValue(x)
    panel.abs_[2].editingFinished.emit(); assert np.allclose(panel.rot, RG.from_euler(10, 20, 30))    # Rotate (absolute)
    panel.move.set_value([1.5, 0, 0], emit=True); assert prev[-1][1] == [1.5, 0, 0]  # Move steppers still preview
    panel.keep.setChecked(True); panel.nudge_btn.click()
    s, t, R, keep = done[-1]; assert (s, t, keep) == ("s2", [1.5, 0, 0], True) and np.allclose(R, RG.from_euler(10, 20, 30))
    panel.nudge_reset.click(); assert np.allclose(panel.rot, np.eye(3)) and panel.move.value() == [0, 0, 0]
    assert np.allclose(prev[-1][2], np.eye(3))
    rm = panel.removes["lineup.adjust"]; rm[0].click(); rm[1].click(); assert gone == [("lineup.fix", 1), ("lineup.adjust", 0)]
    panel.set_enabled_tools(False); assert not panel.nudge_btn.isEnabled()


def test_words_are_plain():
    from app2.qt.lineup_panel import TABS
    for v in TABS["Nudge"].values():
        if isinstance(v, str): assert "—" not in v and "–" not in v and "colour" not in v.lower()
    from app2.qt.build_step import fix_words
    pos = {"a": 2}
    assert fix_words({"tool": "turn", "side": "a", "deg": 30.0}, pos) == "Side 2: turned 30 deg"
    assert fix_words({"tool": "turn", "side": "a", "deg": 12.5, "keep": True}, pos, True) == "Turn side 2 (12.5 deg, kept as placed)"


def test_view_rings_and_drag():
    from PySide6.QtCore import QPointF, Qt, QEvent
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from app2.qt.edit_view import EditView
    app = QApplication.instance() or QApplication([])
    v = EditView(); v.resize(W_, H_); v.cam = cam(); v.set_gizmo(np.zeros(3), R0); got = []
    v.ring_turn.connect(lambda k, deg, fin: got.append((k, round(deg, 1), fin)))
    v.overlay.resize(v.size()); v.overlay.repaint()                  # the rings paint without a GL scene

    def ev(kind, p, btn=Qt.MouseButton.LeftButton, mods=Qt.KeyboardModifier.NoModifier):
        return QMouseEvent(kind, QPointF(*p), QPointF(*p), btn, btn if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton, mods)
    v.mouseMoveEvent(ev(QEvent.Type.MouseMove, px(v.cam, on_ring(2, 45)), Qt.MouseButton.NoButton)); assert v.gizmo["hover"] == 2
    view0 = v.cam.view.copy(); assert v.protractor() is None                  # ticks only while a ring is dragged
    v.mousePressEvent(ev(QEvent.Type.MouseButtonPress, px(v.cam, on_ring(2, 45))))
    v.mouseMoveEvent(ev(QEvent.Type.MouseMove, px(v.cam, on_ring(2, 70))))
    assert v.protractor() and {k for _a, _b, k in v.protractor()} == {"circle", "tick", "spoke"}
    v.overlay.repaint()                                              # the protractor and the "Z: 25.00" readout
    v.mouseMoveEvent(ev(QEvent.Type.MouseMove, px(v.cam, on_ring(2, 82)), mods=Qt.KeyboardModifier.ShiftModifier))
    v.mouseReleaseEvent(ev(QEvent.Type.MouseButtonRelease, px(v.cam, on_ring(2, 82))))
    assert got == [(2, 25.0, False), (2, 30.0, False), (2, 30.0, True)]
    assert v.protractor() is None
    assert np.allclose(v.cam.view, view0) and v.gizmo.get("drag") is None     # the drag never turned the camera
    v.mousePressEvent(ev(QEvent.Type.MouseButtonPress, (5, 5))); v.mouseMoveEvent(ev(QEvent.Type.MouseMove, (40, 5)))
    v.mouseReleaseEvent(ev(QEvent.Type.MouseButtonRelease, (40, 5)))
    assert len(got) == 3 and not np.allclose(v.cam.view, view0)       # off the rings: the camera turns as before
    v.tools_on = False; v.mousePressEvent(ev(QEvent.Type.MouseButtonPress, px(v.cam, on_ring(2, 0)))); assert v.gizmo.get("drag") is None
    v.set_gizmo(None); assert v.gizmo is None
    v.deleteLater(); app.processEvents()


def test_nudge_apply_turn_and_move_is_one_engine_call():
    """Apply with a turn and a move sends ONE engine call (turn with move=), so it is one history entry and one undo."""
    from types import SimpleNamespace
    from app2.qt.build_step import BuildStepPage as BuildStep
    sent = []
    fake = SimpleNamespace(panel={"lineup": SimpleNamespace(reset_nudge=lambda emit=False: None)}, last_changed=None,
                           call=lambda name, calls, then=None, fail=None: sent.append((name, calls)),
                           start_run=lambda run: None, notice=lambda text: None)
    a = np.radians(30.0); Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    BuildStep.do_nudge(fake, "s2", [1.0, 0, 0], Rz, keep=True)
    (name, calls), = sent
    assert name == "h_apply" and len(calls) == 1
    (op, args, _kw), = calls
    assert op == "turn" and args[0] == "s2" and args[2] is True and args[3] == [1.0, 0, 0]
    sent.clear(); BuildStep.do_nudge(fake, "s2", [0, 0, 0], Rz)
    assert sent[0][1][0][0] == "turn" and sent[0][1][0][1][3] is None
    sent.clear(); BuildStep.do_nudge(fake, "s2", [0, 2.0, 0], np.eye(3))
    assert sent[0][1] == [("nudge", ("s2", [0, 2.0, 0], [0, 0, 0]), None)]
