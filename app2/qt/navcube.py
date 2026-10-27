# SPDX-License-Identifier: AGPL-3.0-or-later
"""The orientation cube in the bottom-left corner of every 3D view (the Build step's views, the Scan step's model): a
small shaded cube with its faces named (Front, Back, Left, Right, Top, Bottom) and the X / Y / Z axes in red, green and
blue along the edges of one corner. It turns with the view's camera; a click on a face turns the camera to look at that
face (about the same pivot, from the same distance); a drag that starts on it turns the view like a drag on the model.

The face math, the hit-testing and the face-to-camera presets are in viewnav.py at the repo root (renderer
agnostic); here they are drawn with QPainter on a child widget of the view, a 96 px cube with reserved axis room, so it sits inside the scene and
never changes the layout. The camera is gl_core.Camera: view is world -> eye (OpenGL eye: x right, y up, z towards the
viewer), which is exactly the rotation viewnav draws with.

  cube = NavCube(view)             # view: a widget with .cam (gl_core.Camera); it may have a camera_moved signal
  cube.face_at(x, y) -> key        # the face under a point of the cube widget, or None
  cube.go_face(key)                # what a click on that face does"""
import os, sys
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QEvent, Signal
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen, QFont, QTransform, QFontMetricsF
from PySide6.QtWidgets import QWidget, QMenu

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path: sys.path.append(_ROOT)
import viewnav as VN                                           # noqa: E402  (the cube math)

SIZE = 160                    # 96 px cube plus fixed room for axis arrows at every rotation
CANVAS = 620                  # viewnav sizes its cube from its canvas (15.5% of it, at most 112): a 96 px layout box
_X, _Y, _S = VN.layout(CANVAS, CANVAS)["cube"]
OFFX, OFFY = _X - 32, _Y - 36    # canvas = widget + (OFFX, OFFY): room under the cube for the axes
OFF = OFFY                    # (the tests' name)
CLICK_PX = 4                  # a press that moves further is a drag (turns the view), not a click


def _c(rgb, a=255): return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), a)


def preset(key):
    """The eye rotation (3 x 3, world -> eye) that looks straight at face `key`."""
    az, el = VN.FACE_BY_KEY[key][2]
    return VN.rot_from_az_el(az, el)


def look_at_face(cam, key, basis=None):
    """Turn cam to face `key` about its pivot, keeping the pivot where it is on screen (same distance)."""
    R = preset(key) @ (np.eye(3) if basis is None else basis); pe = cam.view[:3, :3] @ np.asarray(cam.pivot, float) + cam.view[:3, 3]
    V = np.eye(4); V[:3, :3] = R; V[:3, 3] = pe - R @ np.asarray(cam.pivot, float)
    cam.view = V


class NavCube(QWidget):
    face_picked = Signal(str)
    basis_changed = Signal(object)

    def __init__(self, view, parent=None, bottom_inset=0):
        super().__init__(parent or view); self.view = view; self.hover = None; self._press = None; self._last = None
        self.basis = np.eye(3)
        self.bottom_inset = bottom_inset; self._last_rotation = None
        self.setFixedSize(SIZE, SIZE); self.setMouseTracking(True); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground); self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip("Click a face to look at that side. Drag to turn. Right-click: set Front or Top, Perspective or Orthographic.")
        view._cube = self; view.installEventFilter(self); self._place()

    def set_bottom_inset(self, pixels):
        self.bottom_inset = max(0, int(pixels)); self._place()

    def set_basis(self, basis):
        from app2.view_orientation import validate_basis
        self.basis = validate_basis(basis); self.update()

    def set_reference(self, key):
        if key not in ('front', 'top'): raise ValueError('Choose Front or Top.')
        rotation = np.asarray(self.view.cam.view, float)[:3, :3]
        u, _, vt = np.linalg.svd(rotation)
        self.set_basis(preset(key).T @ (u @ vt))
        self.basis_changed.emit(self.basis.copy())

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        for key, name in [('front', 'Front'), ('top', 'Top')]:
            action = menu.addAction('Set this view as ' + name)
            action.triggered.connect(lambda _=False, face=key: self.set_reference(face))
        menu.addSeparator()                       # the view's projection lives here too (it sat in the View menu only)
        mode = getattr(self.view.cam, "projection_mode", "perspective")
        for key, name in [('perspective', 'Perspective'), ('orthographic', 'Orthographic (true shapes, no depth)')]:
            action = menu.addAction(name); action.setCheckable(True); action.setChecked(mode == key)
            action.triggered.connect(lambda _=False, m=key: self._projection(m))
        menu.exec(event.globalPos())

    def _projection(self, mode):
        """Same as View > Perspective / Orthographic; the view's own setter when it has one (it also repaints)."""
        setter = getattr(self.view, "set_projection_mode", None)
        if setter is not None: setter(mode)
        else: self.view.cam.set_projection_mode(mode); self.view.update()
        sig = getattr(self.view, "projection_changed", None)
        if sig is not None: sig.emit(mode)

    def sync(self):
        rotation = np.asarray(self.view.cam.view)[:3, :3]
        if self._last_rotation is None or not np.array_equal(rotation, self._last_rotation):
            self._last_rotation = rotation.copy(); self.update()

    # ---------------- geometry ----------------
    def rot(self):
        R = np.asarray(self.view.cam.view, float)[:3, :3] @ self.basis.T
        u, _s, vt = np.linalg.svd(R); return u @ vt                 # a pure rotation (a view may carry a tiny scale)

    def _regions(self):
        """viewnav's regions for a SIZE x SIZE canvas (its layout puts the cube at the bottom left of that canvas)."""
        return VN.regions(self.rot(), CANVAS, CANVAS)[1]

    def face_at(self, x, y):
        """The face under widget point (x, y) (viewnav's canvas point is ours + (OFFX, OFFY))."""
        for r in reversed(self._regions()):
            if VN.point_in_poly(x + OFFX, y + OFFY, r["poly"]): return r["key"]
        return None

    def go_face(self, key):
        look_at_face(self.view.cam, key, self.basis); self.view.update(); self.update()
        sig = getattr(self.view, "camera_moved", None)
        if sig is not None: sig.emit()
        fol = getattr(self.view, "_set_following", None)             # the Scan view stops riding with the scanner
        if fol is not None: fol(False)
        self.face_picked.emit(key)

    def _place(self):
        self.move(0, max(0, self.view.height() - SIZE - self.bottom_inset)); self.raise_()

    def eventFilter(self, obj, e):
        if obj is self.view and e.type() in (QEvent.Type.Resize, QEvent.Type.Show): self._place()
        return False

    # ---------------- mouse ----------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton: e.accept(); return
        if e.button() != Qt.MouseButton.LeftButton: e.ignore(); return
        self._press = (self.face_at(e.position().x(), e.position().y()), e.position()); self._last = e.position()

    def mouseMoveEvent(self, e):
        if self._press is None:
            h = self.face_at(e.position().x(), e.position().y())
            if h != self.hover: self.hover = h; self.update()
            return
        d0 = e.position() - self._press[1]
        if max(abs(d0.x()), abs(d0.y())) > CLICK_PX:                  # a drag: turn the view
            d = e.position() - self._last; self._last = e.position()
            follow = getattr(self.view, '_set_following', None)
            if follow is not None: follow(False)
            self.view.cam.orbit(d.x(), d.y()); self.view.update(); self.update()
            sig = getattr(self.view, "camera_moved", None)
            if sig is not None: sig.emit()

    def mouseReleaseEvent(self, e):
        p, self._press = self._press, None
        if p is None: return
        d0 = e.position() - p[1]
        if p[0] and max(abs(d0.x()), abs(d0.y())) <= CLICK_PX: self.go_face(p[0])

    def leaveEvent(self, e):
        if self.hover is not None: self.hover = None; self.update()

    # ---------------- drawing ----------------
    def paintEvent(self, e):
        R = self.rot(); p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(-OFFX, -OFFY); regs = VN.regions(R, CANVAS, CANVAS)[1]
        edge = QPen(_c(VN.EDGE), 1)
        for r in regs:
            p.setPen(edge); p.setBrush(_c(VN.HOVER if self.hover == r["key"] else VN.COLORS.get(r["key"], VN.COLORS["front"]), 190))
            p.drawPolygon(QPolygonF([QPointF(*q) for q in r["poly"]]))
        _lo, axes = VN.axis_segments(R, CANVAS, CANVAS)
        f = QFont(); f.setPixelSize(10); f.setBold(True); p.setFont(f)
        for k in ("x", "y", "z"):
            a, b = (np.asarray(x, float) for x in axes[k]); v = b - a; n = float(np.linalg.norm(v))
            if n < 2:
                col = _c(VN.AXIS[k]); p.setPen(QPen(col, 1.5)); p.setBrush(col)
                p.drawEllipse(QPointF(*a), 2, 2)
                p.drawText(QRectF(a[0] - 18, a[1] + 4, 12, 12), int(Qt.AlignmentFlag.AlignCenter), k.upper())
                continue
            u = v / n; q = np.array([-u[1], u[0]]); col = _c(VN.AXIS[k])
            p.setPen(QPen(col, 2)); p.drawLine(QPointF(*a), QPointF(*b))
            head = min(7.0, max(4.0, n * 0.3)); p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
            p.drawPolygon(QPolygonF([QPointF(*b), QPointF(*(b - u * head + q * head * 0.55)), QPointF(*(b - u * head - q * head * 0.55))]))
            t = b + u * 7; p.setPen(col); p.drawText(QRectF(t[0] - 6, t[1] - 6, 12, 12), int(Qt.AlignmentFlag.AlignCenter), k.upper())
        # the face names, printed on the faces (a label mapped onto its face's quad)
        lf = QFont(); lf.setPixelSize(20); lf.setBold(True); fm = QFontMetricsF(lf)
        for r in regs:
            if r["normal_z"] < 0.25: continue                         # too edge-on to read
            text = r["label"]; tw = fm.horizontalAdvance(text) + 8; th = fm.height()
            src = QPolygonF([QPointF(0, 0), QPointF(tw, 0), QPointF(tw, th), QPointF(0, th)])
            dst = QPolygonF([QPointF(*q) for q in r["label_quad"]])
            tr = QTransform()
            if not QTransform.quadToQuad(src, dst, tr): continue
            p.save(); p.setTransform(tr * QTransform.fromTranslate(-OFFX, -OFFY)); p.setFont(lf); p.setPen(_c(VN.TEXT))
            p.drawText(QRectF(0, 0, tw, th), int(Qt.AlignmentFlag.AlignCenter), text); p.restore()
        p.end()
