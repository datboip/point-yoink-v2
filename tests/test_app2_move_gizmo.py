"""The move arrows (Bambu style): hit test, drag distance along an axis, the view's routing, Nudge's move, Cut shape."""
import os, sys, tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'; os.environ['XMODIFIERS'] = '@im=none'; os.environ['POINTYOINK_NO_DEVICE'] = '1'
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_arrows_')) / 'config.json')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scanner"))
import numpy as np
from app2.gl_core import Camera
from app2.qt import move_gizmo as MG

W_, H_ = 800, 600


def cam():
    c = Camera(); c.frame(np.array([-30.0, -30, -30]), np.array([30.0, 30, 30])); return c


def test_arrow_hit_and_miss():
    c = cam(); o = np.zeros(3); L = 20.0
    for k in range(3):
        xy, z = MG.project_arrow(c, o, L, k, W_, H_)
        if not (z > 0).all() or np.linalg.norm(xy[1] - xy[0]) < 12: continue       # an axis seen end on has no shaft to grab
        mid = (xy[0] + xy[1]) / 2
        assert MG.arrow_hit(c, o, L, mid, W_, H_) == k
    assert MG.arrow_hit(c, o, L, (5, 5), W_, H_) is None


def test_drag_slides_along_the_axis_only():
    c = cam(); o = np.zeros(3); L = 20.0
    for k in range(3):
        xy, z = MG.project_arrow(c, o, L, k, W_, H_)
        if np.linalg.norm(xy[1] - xy[0]) < 12: continue
        d = MG.ArrowDrag(c, o, k, xy[0], W_, H_, L)
        mm = d.moved(xy[1])                                   # the cursor at the tip: one arrow length along +axis
        assert abs(mm - L) < 0.5, (k, mm)
        assert abs(d.moved(xy[1], snap_on=True) - round(mm)) <= 0.5
        assert abs(d.moved(xy[0])) < 1e-6


def test_view_routes_arrows_and_rings():
    from PySide6.QtWidgets import QApplication
    from app2.qt.edit_view import EditView
    app = QApplication.instance() or QApplication([])
    v = EditView(); v.tools_on = True
    v.set_gizmo(np.zeros(3), 10.0, rings=False, arrows=True)
    assert v.gizmo["arrows"] and not v.gizmo["rings"] and abs(v.arrow_length() - 13.5) < 1e-9
    from PySide6.QtCore import QPointF
    assert v._ring_at(QPointF(1, 1)) is None                  # rings off: never grabbed
    v.set_gizmo(None); assert v.gizmo is None
    v.close()


def test_nudge_arrow_move_adds_to_the_move_boxes():
    from PySide6.QtWidgets import QApplication
    from app2.qt.lineup_panel import LineupPanel
    app = QApplication.instance() or QApplication([])
    p = LineupPanel(); got = []
    p.nudge_preview.connect(lambda sid, t, R: got.append(list(t)))
    p.arrow_move(0, 2.5, False); assert np.allclose(got[-1], [2.5, 0, 0])
    p.arrow_move(0, 4.0, False); assert np.allclose(got[-1], [4.0, 0, 0])        # live: from the press, not cumulative
    p.arrow_move(0, 4.0, True); assert np.allclose(p.move.value(), [4.0, 0, 0])
    p.arrow_move(2, -1.0, True); assert np.allclose(p.move.value(), [4.0, 0, -1.0])
    p.close()


def test_fin_tool_arrow_moves_the_last_spot(monkeypatch):
    """A drag on the arrows slides the last picked spot along the world axis, in the model's own coordinates."""
    from PySide6.QtWidgets import QApplication
    from app2.qt.edit_view import EditView
    from app2.qt.fin_tool import FinTool
    app = QApplication.instance() or QApplication([])
    v = EditView(); v.tools_on = True
    from PySide6.QtCore import QObject
    class Page(QObject): gl = True; stage = 'meshed'; comparing = False
    page = Page(); page.vp = type('VP', (), {})(); page.vp.view = v; page.panel = {}
    t = FinTool(page); t.minimum_picks = 1
    M = np.eye(4); M[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]           # the model turned 90 about Z
    t.source = dict(key='m', kind='mesh', model=M); t.anchors = [[[1.0, 2.0, 3.0], [0, 0, 1]]]
    t.preview = lambda: None
    t.show_arrows(); assert v.gizmo['owner'] == 'fin-preview' and not v.gizmo['rings']
    assert np.allclose(v.gizmo['c'], M[:3, :3] @ [1, 2, 3])
    t.arrow_move(0, 5.0, False)                                             # world +X is local -Y here... M^-1 x = (0,-1,0)? check numerically
    local = np.linalg.solve(M[:3, :3], np.array([5.0, 0, 0]))
    assert np.allclose(t.anchors[0][0], np.array([1, 2, 3]) + local)
    t.arrow_move(0, 2.0, True); assert np.allclose(t.anchors[0][0], np.array([1, 2, 3]) + local * 0.4)   # from the press, not cumulative
    t.clear(); assert v.gizmo is None
    v.close()
