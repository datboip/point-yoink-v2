# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 3D view of PointYoink 2.0: a QOpenGLWidget over app2.gl_core (OpenGL 3.3 core, PyOpenGL).

  view = ModelView()
  view.set_points(P, N, levels)      # the live model, ~3x a second: tracker cloud (mm), normals, coverage 0/1/2
  view.set_levels(levels)            # recolour only (Clean marks picked points with level 3)
  view.set_mesh(V, N, C, faces)      # the built model, once
  view.set_lens(fx, fy, cx, cy)      # the depth camera's own lens (Pl.bin), so Home lines up with the IR picture
  view.set_home(T, pivot)            # the scanner's own view (pose OpenCV camera -> world, mm); Home goes back to it
  view.follow_scanner(T, pivot)      # every tracked frame: the view rides along with the scanner until the user turns it
  view.select_in_mask(P, mask)       # Clean: indices of P inside a lasso / box mask and not hidden behind the surface

Mouse: left-drag turns about the pivot, right- or middle-drag pans, wheel zooms towards the point under the cursor.
Double-click: Home (dbl_click="home", the Scan step) or make the point under the cursor the pivot (dbl_click="pivot",
for Review). Nothing redraws on its own: a new cloud or a drag asks for one frame (update()), so an idle view costs
nothing. set_* only keep the arrays (safe with arrays made on a worker thread and handed over by a signal); they go
to the GPU in paintGL, where the context is current. The arrays must not be changed afterwards (hand over fresh ones)."""
import math, time
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from app2 import gl_core


def default_format():
    """Call before QApplication: OpenGL 3.3 core, 24-bit depth, no multisampling (the depth under the cursor is read
    back for picking, which a multisampled buffer does not allow), vsync on."""
    f = QSurfaceFormat(); f.setVersion(3, 3); f.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    f.setDepthBufferSize(24); f.setSamples(0); f.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(f)


def scanner_fov(fy, height=600):
    """The depth camera's vertical field of view in degrees: 2 atan((height / 2) / fy)."""
    return math.degrees(2.0 * math.atan((height / 2.0) / fy))


class ModelView(QOpenGLWidget):
    picked = Signal(object)                  # world point under a double-click (dbl_click="pivot"), or None
    following_changed = Signal(bool)         # riding with the scanner (True) or turned by the user (False)
    home_asked = Signal()

    def __init__(self, parent=None, dbl_click="home"):
        super().__init__(parent)
        self.cam = gl_core.Camera(); self.scene = None; self.dbl_click = dbl_click
        self._todo = {}                          # what paintGL still has to send: "points", "levels", "mesh"
        self._drag = None; self._btn = None; self._have_points = False; self._have_mesh = False
        self.paint_ms = 0.0                      # the last paintGL, CPU side (for the tests)
        self.following = True                    # riding along with the scanner (Home); a drag or the wheel stops it
        self.ride = True                         # turned away: keep riding with the scanner from your angle (False: stay put in the world)
        from app2.qt.startup import EmptyViewBackground
        self.empty_background = EmptyViewBackground(self)
        self.empty_background.setGeometry(self.rect()); self.empty_background.show()
        from app2.qt.navcube import NavCube
        self.nav_cube = NavCube(self)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.setMinimumSize(80, 60)

    def _sync_empty_background(self):
        self.empty_background.setVisible(not (self._have_points or self._have_mesh))
        self.empty_background.lower(); self.nav_cube.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.empty_background.setGeometry(self.rect())

    def set_projection_mode(self, mode):
        self.cam.set_projection_mode(mode); self.update()

    def _set_following(self, on):
        if on != self.following: self.following = on; self.following_changed.emit(on)

    # ---------------- data in ----------------
    def set_lens(self, fx, fy, cx, cy, width=800, height=600):
        """The depth camera's intrinsics (pixels of its 800 x 600 picture): the view uses the scanner's own lens, off
        centre where its principal point is, so at Home the model lands where the IR picture shows the part."""
        self.cam.set_intrinsics(fx, fy, cx, cy, width, height); self.update()

    def set_points(self, P, N=None, levels=None):
        """P (n,3) float32, N (n,3) unit normals, levels (n,) uint8 0-3; P None or empty clears. Frames the view on the
        first cloud when there is no Home."""
        if P is None or not len(P):
            self._todo["points"] = None; self._todo.pop("levels", None); self._have_points = False; self._sync_empty_background(); self.update(); return
        if N is None: N = np.tile(np.float32([0, 0, 1]), (len(P), 1))
        if levels is None: levels = np.full(len(P), 2, np.uint8)
        self._todo["points"] = (P, N, levels); self._todo.pop("levels", None)
        if not self._have_points and self.cam.home is None: self.cam.frame(*self._box(P))
        else: self.cam.fit_depth(*self._box(P))                 # the cloud grew: keep it inside near / far
        self._have_points = True; self._sync_empty_background(); self.update()

    def has_points(self): return self._have_points

    def set_levels(self, levels):
        if self._todo.get("points") is not None:
            P, N, _ = self._todo["points"]; self._todo["points"] = (P, N, levels)
        else: self._todo["levels"] = levels
        self.update()

    def set_mesh(self, V, N=None, C=None, faces=None, keep_view=False):
        """V (n,3), N (n,3), C (n,3) uint8 or None, faces (m,3); V None clears. keep_view: a live surface that keeps
        growing leaves the camera where the user put it."""
        if V is None or not len(V): self._todo["mesh"] = None; self._have_mesh = False; self._sync_empty_background(); self.update(); return
        if N is None:
            import open3d as o3d                                     # normals for a mesh that came without them
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
            m.compute_vertex_normals(); N = np.asarray(m.vertex_normals, np.float32)
        self._todo["mesh"] = (V, N, C, faces); self._have_mesh = True; self._sync_empty_background()
        if keep_view: self.cam.fit_depth(*self._box(V))
        else: self.cam.frame(*self._box(V))
        self.update()

    def has_mesh(self): return self._have_mesh

    def set_home(self, T, pivot):
        self.cam.set_home(T, pivot); self._set_following(True); self.update()

    def go_home(self):
        self._set_following(True); self.cam.go_home(); self.update()

    def follow_scanner(self, T, pivot):
        """The scanner moved (a tracked frame): Home becomes its new pose, and the view goes along unless the user has
        turned it away (the Tk scanner's model panel did the same)."""
        v = self.cam.view.copy(); p = self.cam.pivot.copy(); old = self.cam.home
        self.cam.set_home(T, pivot)
        if not self.following:
            if self.ride and old is not None:        # 2026-10-24 "keep spinning": your angle rides along with the scanner, so the part keeps turning
                T0 = old[0]; T1 = np.asarray(T, float)
                O = v @ T0                           # your eye, in the scanner's own frame
                self.cam.view = O @ np.linalg.inv(T1); self.cam.pivot = (T1 @ np.linalg.inv(T0) @ np.append(p, 1.0))[:3]
            else: self.cam.view, self.cam.pivot = v, p
        self.update()

    def select_in_mask(self, P, mask, tol=None):
        """Clean's box / lasso: mask is a widget-sized bool image (device pixels). Reads the last frame's depth once."""
        if self.scene is None: return np.zeros(0, np.int64)
        w, h = self._px(); self.makeCurrent()
        try: depth = self.scene.read_depth_image(w, h)
        finally: self.doneCurrent()
        return self.cam.select(P, mask, depth, 2.0 * self.scene.point_size if tol is None else tol)

    @staticmethod
    def _box(P):
        s = P[::max(1, len(P) // 200_000)]                          # a sample is plenty for the bounding box
        return s.min(0), s.max(0)

    # ---------------- GL ----------------
    def initializeGL(self):
        self.scene = gl_core.SceneGL(); self.scene.point_size = gl_core.POINT_MM * getattr(self, 'point_scale', 1.0)

    def set_point_scale(self, scale):
        """Dot size from Show marks (Small, Normal, Large) for the 3D dots too."""
        self.point_scale = float(scale)
        if self.scene is not None: self.scene.point_size = gl_core.POINT_MM * self.point_scale
        self.update()

    def paintGL(self):
        from app2.scan import timing; timing.mark('gl_paint_start')
        try: self._paint_gl()
        finally: timing.mark('gl_paint_end')

    def _paint_gl(self):
        self.nav_cube.sync()
        t0 = time.perf_counter()
        todo, self._todo = self._todo, {}
        if "points" in todo:
            if todo["points"] is None: self.scene.points.count = 0
            else: self.scene.set_points(*todo["points"])
        if "levels" in todo and self.scene.points.count == len(todo["levels"]): self.scene.set_levels(todo["levels"])
        if "mesh" in todo:
            if todo["mesh"] is None: self.scene.mesh.count = 0
            else: self.scene.set_mesh(*todo["mesh"])
        self.scene.capture_colors=getattr(self,"capture_colors",False)
        self.scene.palette=getattr(self,"point_palette",gl_core.PALETTE)
        w, h = self._px()
        self.scene.draw(self.cam.view, self.cam.projection(w / max(h, 1)), w, h)
        self.paint_ms = (time.perf_counter() - t0) * 1000.0

    def _px(self):
        r = self.devicePixelRatioF(); return max(1, int(self.width() * r)), max(1, int(self.height() * r))

    def world_at(self, pos: QPointF):
        """The model point under a widget position, or None over the background. Reads one depth value of the last frame."""
        if self.scene is None: return None
        r = self.devicePixelRatioF(); w, h = self._px(); x, y = pos.x() * r, pos.y() * r
        self.makeCurrent()                                       # binds this widget's framebuffer
        try: d = self.scene.read_depth(x, y, h)
        finally: self.doneCurrent()
        return None if d >= 1.0 else self.cam.unproject(x, y, d, w, h)

    # ---------------- mouse ----------------
    def mousePressEvent(self, e):
        self._drag = e.position(); self._btn = e.button()

    def mouseMoveEvent(self, e):
        if self._drag is None: return
        d = e.position() - self._drag; self._drag = e.position(); self._set_following(False)
        if self._btn == Qt.MouseButton.LeftButton: self.cam.orbit(d.x(), d.y())
        else:
            w, h = self._px(); self.cam.pan(d.x() * self.devicePixelRatioF(), d.y() * self.devicePixelRatioF(), h, w / h)
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0; self._set_following(False)
        self.cam.zoom(steps, toward=self.world_at(e.position())); self.update()

    def mouseDoubleClickEvent(self, e):
        if self.dbl_click == "home":
            self.go_home(); self.home_asked.emit(); return
        p = self.world_at(e.position())
        if p is None: self.go_home()
        else: self.cam.pivot = p; self.update()
        self.picked.emit(p)

