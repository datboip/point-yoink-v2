# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's 3D view: a QOpenGLWidget over edit_gl.EditScene + gpu_select.GpuSelector (numpy select_core as the
fallback), with Rectangle and Lasso.

  v = EditView()
  v.show_items([dict(key, kind="mesh", V, N, F, face_rgb=None, model=None, colour=None) | dict(key, kind="points", P, N, rgb)],
               edit="main", elements=(X, Nrm))  # a stage result; edit = the item the tools select on (engine order)
  v.set_tool("orbit" | "rect" | "lasso");  v.through = True / False       # also select what is hidden behind
  v.selection -> bool per element (engine order); selection_changed(int count)
  v.show_would_go(mask, platter=False) / clear_would_go()                 # a Detect's mask: red (or violet), no editing
  v.set_item_model(key, M)                                                # Line-up nudge / flip previews: no re-upload
  v.set_tool("pairpick"); v.set_markers([(xyz, "#rrggbb", "1"), ...])     # Line-up Points: a click gives surface_picked,
                                                                          # numbered dots drawn where the points are
  v.set_gizmo(centre, radius) / v.set_gizmo(None)                         # Line-up Nudge: rotate rings (rotate_gizmo);
                                                                          # a drag gives ring_turn(axis, deg, done)
  v.go_home()                                                             # frame the result

Mouse: no tool armed = the Scan step's view (left turns, right or middle pans, wheel zooms towards the cursor). A tool
armed: left draws (Ctrl takes away), right turns, middle pans, wheel zooms; turning keeps the selection. Esc / Del
come from the page (this widget never takes focus). Arrays are handed over and never changed afterwards; they go to
the GPU in paintGL, where the context is current. No MSAA: the depth buffer is read back for visible-only selection."""
import time
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor, QImage, QPolygonF
from PySide6.QtWidgets import QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from app2 import gl_core
from app2.builder import select_core as sc
from app2.qt.theme import T

RED, INK = QColor(T["lost"]), QColor(T["ink"])


class _Overlay(QWidget):
    """The rectangle or lasso being drawn, above the GL picture (QPainter, device independent pixels)."""
    def __init__(self, view):
        super().__init__(view); self.v = view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, e):
        v = self.v
        from app2.qt.scan_guide import paint
        paint(v)
        if getattr(v,"capture_colors",False):
            from app2.qt.capture_key import paint as paint_key
            self.capture_key_inset=getattr(v,"capture_key_inset",0)
            paint_key(self)
        if v.plane_marks: self._planes()
        if v.markers: self._dots()
        if v.gizmo is not None:
            if v.gizmo.get("rings", True): self._rings()
            if v.gizmo.get("arrows"): self._arrows()
        if not v._stroke and not (v.tool == "brush" and v._hover is not None and v.tools_on): return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = QColor(INK) if not v._subtract else QColor(RED)
        p.setPen(QPen(col, 1.5, Qt.PenStyle.DashLine)); fill = QColor(col); fill.setAlphaF(0.08); p.setBrush(fill)
        pts = v._stroke or [v._hover]
        if v.tool == "rect": p.drawRect(QRectF(pts[0], pts[-1]).normalized())
        elif v.tool == "brush":
            p.drawEllipse(pts[-1], v.brush_px, v.brush_px)
        elif v.tool == "lasso":
            path = QPainterPath(pts[0])
            for q in pts[1:]: path.lineTo(q)
            path.closeSubpath(); p.drawPath(path)
        p.end()

    def _rings(self):
        """Nudge's rotate rings: X / Y / Z circles, the far half faint, the hovered or dragged ring lit, the angle by the cursor."""
        from app2.qt import rotate_gizmo as RG
        v = self.v; g = v.gizmo; w, h = max(1, v.width()), max(1, v.height())
        ce = v.cam.project(np.asarray(g["c"], float)[None], w, h)[2][0]
        on = g["drag"].k if g.get("drag") is not None else g.get("hover")
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for k in range(3):
            xy, ze, _P = RG.project_ring(v.cam, g["c"], g["r"], k, w, h)
            if not (ze > 0).all() or not np.isfinite(xy).all(): continue
            col = QColor(RG.HOVER if k == on else RG.COLOURS[k])
            for front in (False, True):
                c2 = QColor(col); c2.setAlphaF(1.0 if front else 0.35); p.setPen(QPen(c2, (3.0 if k == on else 2.0) if front else 1.2))
                for i in range(len(xy)):
                    j = (i + 1) % len(xy)
                    if ((ze[i] + ze[j]) / 2 <= ce) == front: p.drawLine(QPointF(*xy[i]), QPointF(*xy[j]))
        segs = v.protractor()
        if segs:
            for A, B, kind in segs:                                  # the dragged ring's protractor, far half faint
                (x, y, z) = v.cam.project(np.array([A, B]), w, h)
                if not (z > 0).all() or not np.isfinite(x).all() or not np.isfinite(y).all(): continue
                c2 = QColor("#ffffff"); c2.setAlphaF((0.9 if kind != "spoke" else 0.6) if (z[0] + z[1]) / 2 <= ce + 1e-6 or kind == "spoke" else 0.3)
                p.setPen(QPen(c2, 1.6 if kind == "circle" else 1.2)); p.drawLine(QPointF(x[0], y[0]), QPointF(x[1], y[1]))
        if g.get("drag") is not None and g.get("pos") is not None:
            f = p.font(); f.setBold(True); f.setPixelSize(13); p.setFont(f)
            txt = RG.label(g["drag"].k, g.get("deg", 0.0)); q = g["pos"] + QPointF(14, -10)
            r = p.fontMetrics().boundingRect(txt).adjusted(-5, -3, 5, 3).translated(q.toPoint())
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(QRectF(r), 4, 4)
            p.setPen(QColor(RG.COLOURS[g["drag"].k])); p.drawText(q, txt)
        p.end()

    def _arrows(self):
        """The move arrows: X / Y / Z from the centre, the hovered or dragged one lit, the mm slid by the cursor."""
        from app2.qt import move_gizmo as MG
        v = self.v; g = v.gizmo; w, h = max(1, v.width()), max(1, v.height()); L = v.arrow_length()
        on = g["adrag"].k if g.get("adrag") is not None else g.get("ahover")
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for k in range(3):
            xy, z = MG.project_arrow(v.cam, g["c"], L, k, w, h)
            if not (z > 0).all() or not np.isfinite(xy).all(): continue
            col = QColor(MG.HOVER if k == on else MG.COLOURS[k]); tail, tip, base = (QPointF(*q) for q in xy)
            d = tip - base; n = np.hypot(d.x(), d.y())
            p.setPen(QPen(col, 3.5 if k == on else 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)); p.drawLine(tail, base)
            if n > 1e-6:
                side = QPointF(-d.y(), d.x()) * (max(5.0, n * 0.45) / n)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(col); p.drawPolygon(QPolygonF([tip, base + side, base - side]))
        if g.get("adrag") is not None and g.get("pos") is not None:
            f = p.font(); f.setBold(True); f.setPixelSize(13); p.setFont(f)
            txt = MG.label(g["adrag"].k, g.get("mm", 0.0)); q = g["pos"] + QPointF(14, -10)
            r = p.fontMetrics().boundingRect(txt).adjusted(-5, -3, 5, 3).translated(q.toPoint())
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(QRectF(r), 4, 4)
            p.setPen(QColor(MG.COLOURS[g["adrag"].k])); p.drawText(q, txt)
        p.end()

    def _planes(self):
        """Each side's cut plane as an outline and a faint fill in the side's color, its number at one corner: what Auto cut, seen in place."""
        v = self.v; w, h = max(1, v.width()), max(1, v.height()); p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = p.font(); f.setBold(True); f.setPixelSize(12); p.setFont(f)
        for m in v.plane_marks:
            x, y, ze = v.cam.project(np.asarray(m["corners"], float), w, h)
            if not (ze > 0).all() or not np.isfinite(x).all() or not np.isfinite(y).all(): continue
            poly = QPolygonF([QPointF(float(a), float(b)) for a, b in zip(x, y)]); col = QColor(m["colour"])
            fill = QColor(col); fill.setAlphaF(0.12); p.setBrush(fill); p.setPen(QPen(col, 2.0)); p.drawPolygon(poly)
            q = QPointF(float(x[0]), float(y[0])); txt = m.get("text") or "Side %s base" % m["label"]
            r = p.fontMetrics().boundingRect(txt).adjusted(-4, -2, 4, 2).translated(int(q.x()) + 4, int(q.y()) - 18)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(QRectF(r), 4, 4)
            p.setPen(col); p.drawText(QPointF(r.x() + 4, r.y() + r.height() - 5), txt)
        p.end()

    def _dots(self):
        """The Points tab's numbered dots, projected each frame (points behind the camera skipped)."""
        v = self.v; P = np.array([m[0] for m in v.markers], float)
        x, y, ze = v.cam.project(P, max(1, v.width()), max(1, v.height()))
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = p.font(); f.setBold(True); f.setPixelSize(11); p.setFont(f)
        for (_xyz, col, label), xx, yy, z in zip(v.markers, x, y, ze):
            if z <= 0 or not np.isfinite(xx + yy): continue
            c = QPointF(float(xx), float(yy)); p.setPen(QPen(QColor("#000000"), 1.5)); p.setBrush(QColor(col)); p.drawEllipse(c, 8, 8)
            p.setPen(QColor("#000000")); p.drawText(QRectF(c.x() - 8, c.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(label))
        p.end()


class EditView(QOpenGLWidget):
    surface_picked = Signal(object)
    marker_clicked = Signal(object)        # a click on one of the numbered dots (its label), outside a pick tool
    pick_missed = Signal()                 # a pick click that hit no part of the model
    selected_nothing = Signal(str)         # a box, loop or Piece click that took no faces (the tool's name)
    ring_turn = Signal(int, float, bool)          # Nudge's rotate rings: axis (0 X, 1 Y, 2 Z), deg since the press, done
    arrow_move = Signal(int, float, bool)         # the move arrows: axis, mm slid since the press, done
    selection_changed = Signal(int)
    camera_moved = Signal()
    marked_only_changed = Signal(bool)
    gl_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = gl_core.Camera(); self.scene = None; self.gsel = None; self.gpu_ok = True
        self._hover = None; self.hover_mask = None; self.mesh_points = False
        self._hover_timer = QTimer(self); self._hover_timer.setSingleShot(True); self._hover_timer.setInterval(180); self._hover_timer.timeout.connect(self._update_hover)
        self._todo = {}; self._drag = None; self._btn = None; self._stroke = []; self._subtract = False
        self.selection_mode = "add"; self.brush_px = 24; self.magic_angle = 49; self._face_graph_cache = None
        self.tool = "orbit"; self.through = False; self.tools_on = True
        self.edit_key = None; self.n = 0; self.selection = np.zeros(0, bool); self.would_go = None; self.elements = None
        self.gone = None; self.gone_stack = []           # faces deleted in Edit and hidden here until the one rebuild (2026-10-25)
        self._np_elems = None; self.last_select_ms = 0.0; self.select_path = None; self.flat = False; self.marked_only = False
        self.items = {}; self._box = None; self.paint_ms = 0.0; self.error = None; self.markers = []; self.plane_marks = []; self._press_at = None; self.only = None
        self.render_ready = False
        self.gizmo = None
        from app2.qt.startup import EmptyViewBackground
        self.empty_background = EmptyViewBackground(self)
        self.empty_background.setGeometry(self.rect()); self.empty_background.show()
        self.overlay = _Overlay(self)
        from app2.qt.navcube import NavCube
        self.nav_cube = NavCube(self)
        self.camera_moved.connect(self.nav_cube.sync); self.camera_moved.connect(self.clear_hover); self.camera_moved.connect(self.overlay.update)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.setMinimumSize(80, 60); self.setMouseTracking(True)

    # ---------------- data in ----------------
    def show_items(self, items, edit=None, elements=None, frame=True):
        """items: dicts (see the module doc). elements: (X, Nrm) of the edit item (points, or face centroids and
        normals) in engine order; made here when not given. frame: look at the whole result."""
        self.render_ready = False
        self.scan_guide = None
        self.clear_hover(); self.items = {it["key"]: it for it in items}; self.edit_key = edit
        self._todo["items"] = list(items); self._np_elems = None; self._face_graph_cache = None
        if edit is not None and edit in self.items:
            it = self.items[edit]
            if elements is None:
                elements = sc.centroids(it["V"], it["F"]) if it["kind"] == "mesh" else (np.asarray(it["P"], np.float32), it.get("N"))
            self.elements = elements; self._todo["elements"] = elements; self.n = len(elements[0])
        else: self.elements = None; self.n = 0
        self.selection = np.zeros(self.n, bool); self.would_go = None; self.gone = None; self.gone_stack = []
        lo = hi = None
        for it in items:
            A = it["V"] if it["kind"] == "mesh" else it["P"]
            if not len(A): continue
            s = np.asarray(A)[::max(1, len(A) // 200_000)]
            M = np.asarray(it.get("model")) if it.get("model") is not None else None
            if M is not None: s = s @ M[:3, :3].T + M[:3, 3]
            l, h = s.min(0), s.max(0); lo = l if lo is None else np.minimum(lo, l); hi = h if hi is None else np.maximum(hi, h)
        self._box = (lo, hi) if lo is not None else None
        if self._box is not None:
            if frame: self.cam.frame(*self._box)
            else: self.cam.fit_depth(*self._box)
        self.empty_background.setVisible(self._box is None)
        self.empty_background.lower(); self.nav_cube.raise_()
        self._push_flags(); self.update()

    def clear(self):
        self.scan_guide=None; self.overlay.update()
        self.empty_background.show(); self.empty_background.lower(); self.nav_cube.raise_()
        self.items = {}; self._todo["items"] = []; self.edit_key = None; self.elements = None; self.n = 0
        self.selection = np.zeros(0, bool); self.would_go = None; self.gone = None; self.gone_stack = []; self.update()

    def go_home(self):
        if self._box is not None: self.cam.frame(*self._box); self.camera_moved.emit(); self.update()

    def set_fill_preview(self, item):
        self._fill_preview = item
        self._todo["fill_preview"] = item
        self.update()

    def set_item_model(self, key, M):
        it = self.items.get(key)
        if it is None: return
        it["model"] = np.asarray(M, float); self._todo.setdefault("models", {})[key] = it["model"]; self.update()

    def set_only(self, keys):
        """Show only these items (None: every item by its own "visible" flag); survives show_items (Line-up Points)."""
        self.only = None if keys is None else set(keys); self._todo["vis"] = True; self.update()

    def set_lines(self, parts):
        """[(segments (k,2,3) float32 world, (r,g,b,a))] drawn over the model, solid in front and faint behind (the Fill
        holes preview's hole outlines); [] or None clears."""
        self._lines = list(parts or []); self._todo["lines"] = self._lines; self.update()

    def set_plane_marks(self, marks):
        """[{corners (4x3 world), colour "#rrggbb", label}]: the cut planes drawn over the picture (Side > All sides)."""
        self.plane_marks = list(marks or []); self.overlay.update()

    def set_markers(self, markers):
        """[(world xyz, "#rrggbb", label)]: numbered dots over the picture (Line-up Points)."""
        self.markers = list(markers or []); self.overlay.update()

    def set_gizmo(self, centre, radius=None, rings=True, arrows=True):
        """Nudge's rotate rings and the move arrows about the world axes through centre (radius in mm); None hides them. A drag in progress
        keeps going (the page moves the centre with a Move step, never mid-drag)."""
        if centre is None: self.gizmo = None
        else:
            g = self.gizmo or {}; g.update(c=np.asarray(centre, float), r=float(radius or g.get("r") or 10.0), rings=rings, arrows=arrows); self.gizmo = g
        self.overlay.update()

    def arrow_length(self):
        g = self.gizmo; return 0.0 if g is None else float(g["r"]) * 1.35

    def _arrow_at(self, pos):
        from app2.qt import move_gizmo as MG
        g = self.gizmo
        if g is None or not g.get("arrows") or not self.tools_on: return None
        return MG.arrow_hit(self.cam, g["c"], self.arrow_length(), (pos.x(), pos.y()), max(1, self.width()), max(1, self.height()))

    def protractor(self):
        """The protractor segments (rotate_gizmo.protractor) of the ring being dragged; None when no drag."""
        from app2.qt import rotate_gizmo as RG
        g = self.gizmo
        if g is None or g.get("drag") is None: return None
        return RG.protractor(g["c"], g["r"], g["drag"].k, g["drag"].a0)

    def _ring_at(self, pos):
        from app2.qt import rotate_gizmo as RG
        g = self.gizmo
        if g is None or not g.get("rings", True) or not self.tools_on: return None
        return RG.ring_hit(self.cam, g["c"], g["r"], (pos.x(), pos.y()), max(1, self.width()), max(1, self.height()))

    def set_item_colour(self, key, rgb01, picked=False):
        it = self.items.get(key)
        if it is None: return
        self._todo.setdefault("colours", {})[key] = (np.float32(rgb01), picked); self.update()

    def set_projection_mode(self, mode):
        self.cam.set_projection_mode(mode); self.camera_moved.emit(); self.update()

    def set_mesh_points(self, on):
        self.mesh_points = bool(on); self.update()

    def set_flat(self, on): self.flat = bool(on); self.update()

    def set_marked_only(self, on):
        on = bool(on and self.would_go is not None and np.any(self.would_go))
        if self.marked_only != on:
            self.marked_only = on; self.marked_only_changed.emit(on)
        self.update()

    def show_would_go(self, mask, platter=False):
        """What a tool or the platter cut would remove (engine order): red (violet for the platter), over any selection."""
        self.would_go = None if mask is None else np.asarray(mask, bool)[:self.n]; self._platter = platter; self._push_flags()

    def clear_would_go(self): self.show_would_go(None)

    def _push_flags(self):
        if self.would_go is None or not np.any(self.would_go): self.set_marked_only(False)
        m = self.would_go if self.would_go is not None else self.selection
        flags = None if m is None else m.astype(np.uint8)
        if self.would_go is None and flags is not None and self.hover_mask is not None and len(self.hover_mask) == len(flags):
            flags = flags | (self.hover_mask.astype(np.uint8) << 2)
        gone = self.gone
        if gone is not None and len(gone) == self.n:
            flags = (gone.astype(np.uint8) << 3) if flags is None else (flags | (gone.astype(np.uint8) << 3))
        self._todo["flags"] = flags if flags is not None and flags.any() else None
        self._todo["platter"] = bool(getattr(self, "_platter", False)) and self.would_go is not None
        self.selection_changed.emit(int(self.selection.sum()) if self.selection is not None else 0)
        self.update()

    def set_tool(self, t):
        self.clear_hover(); self.tool = t; self._stroke = []
        self.setToolTip(("This model is too big for a hover preview. Click to select." if self.n > 1_000_000 else "Amber is a hover preview. Click to select; selected faces are red.") if t in ("magic", "piece", "brush") else "")
        self.setCursor(Qt.CursorShape.CrossCursor if t in ("rect", "lasso", "brush", "magic", "piece", "fin", "pairpick") else Qt.CursorShape.OpenHandCursor)
        if t == "orbit" and self.selection is not None and self.selection.any(): self.selection[:] = False; self._push_flags()   # orbit clears the selection
        self.overlay.update()

    def clear_selection(self):
        if self.selection is not None: self.selection[:] = False
        self.clear_hover(); self._push_flags()

    def hide_selected(self):
        """The selection is gone from the picture (a delete the engine has recorded; the rebuild comes later). Returns
        how many elements went."""
        if self.selection is None or not self.selection.any(): return 0
        m = self.selection.copy(); self.gone = m if self.gone is None else (self.gone | m); self.gone_stack.append(m)
        self.selection[:] = False; self.clear_hover(); self._push_flags(); return int(m.sum())

    def unhide_last(self):
        """The last hidden group shows again (its delete was undone before the rebuild)."""
        if not self.gone_stack: return
        self.gone_stack.pop(); self.gone = np.any(self.gone_stack, axis=0) if self.gone_stack else None; self._push_flags()

    def hidden_count(self): return int(self.gone.sum()) if self.gone is not None else 0

    def invert_selection(self):
        if self.selection is not None and len(self.selection): self.selection = ~self.selection; self.clear_hover(); self._push_flags()

    def topology_selection(self, operation):
        """Use the same face topology operations as the original mesh editor."""
        import mesh_ops
        item = self.items.get(self.edit_key, {})
        if item.get("kind") != "mesh" or not self.tools_on: return False
        faces = item["F"]
        if self._face_graph_cache is None: self._face_graph_cache = mesh_ops.FaceGraph(faces)
        fn = {"grow": mesh_ops.grow_face_selection, "shrink": mesh_ops.shrink_face_selection,
              "connected": mesh_ops.connected_face_selection}[operation]
        changed = fn(faces, self.selection, graph=self._face_graph_cache)
        did_change = not np.array_equal(changed, self.selection)
        self.selection = changed; self.would_go = None; self._push_flags()
        return did_change

    def _apply_selection(self, got):
        if self.gone is not None and len(self.gone) == len(got): got = got & ~self.gone      # deleted, rebuild pending: not pickable
        if self._subtract: self.selection &= ~got
        elif self.selection_mode == "replace" and not getattr(self, "_force_add", False): self.selection = got.copy()
        else: self.selection |= got                          # Add mode, or Shift held
        self.would_go = None; self._push_flags()

    def clear_hover(self):
        self._hover_timer.stop()
        if self.hover_mask is not None: self.hover_mask = None; self._push_flags()

    def _update_hover(self):
        # Hover never changes committed selection. Bound preview work on large scans.
        if self._hover is None or self._drag is not None or not self.tools_on or self.n > 1_000_000:
            self.clear_hover(); return
        pos = self._hover
        if self.tool == "magic": mask = self._magic_region_at(pos)
        elif self.tool == "piece": mask = self._piece_region_at(pos)
        elif self.tool == "brush":
            w, h = self._px(); ratio = self.devicePixelRatioF(); yy, xx = np.ogrid[:h, :w]
            mask = self.select_mask((xx-pos.x()*ratio)**2+(yy-pos.y()*ratio)**2 <= (self.brush_px*ratio)**2, self.through)
        else: mask = None
        self.hover_mask = mask; self._push_flags()

    def _magic_pick(self, pos):
        mask = self._magic_region_at(pos)
        if mask is not None: self._apply_selection(mask)

    PIECE_PX = 6                           # how far from the click a face may be and still count as clicked

    def _piece_region_at(self, pos):
        """The whole connected piece under pos (2026-10-25, the Piece tool: "easy clicking attached is 80 percent of it"),
        or None when nothing is there. Mesh only."""
        import mesh_ops
        item = self.items.get(self.edit_key, {})
        if item.get("kind") != "mesh" or self.selection is None: return None
        w, h = self._px(); r = self.devicePixelRatioF(); yy, xx = np.ogrid[:h, :w]
        seed = self.select_mask((xx - pos.x() * r) ** 2 + (yy - pos.y() * r) ** 2 <= (self.PIECE_PX * r) ** 2, False)
        if len(seed) != len(self.selection) or not seed.any(): return None
        faces = item["F"]
        if self._face_graph_cache is None: self._face_graph_cache = mesh_ops.FaceGraph(faces)
        return mesh_ops.connected_face_selection(faces, seed, graph=self._face_graph_cache)

    def _piece_pick(self, pos):
        mask = self._piece_region_at(pos)
        if mask is None: self.selected_nothing.emit("piece"); return
        self._apply_selection(mask)

    def _magic_region_at(self, pos):
        from app2.builder.mesh_selection import magic_region
        import mesh_ops
        item = self.items.get(self.edit_key, {})
        if item.get("kind") != "mesh": return
        w, h = self._px(); r = self.devicePixelRatioF()
        yy, xx = np.ogrid[:h, :w]
        candidates = self.select_mask((xx-pos.x()*r)**2+(yy-pos.y()*r)**2 <= (self.brush_px*r)**2, False)
        ids = np.flatnonzero(candidates)
        if not len(ids): return
        world = self.world_at(pos)
        if world is None: return
        M = item.get("model")
        if M is not None: Mi = np.linalg.inv(np.asarray(M, float)); world = Mi[:3, :3] @ np.asarray(world, float) + Mi[:3, 3]
        seed = ids[np.argmin(np.linalg.norm(self.elements[0][ids]-world, axis=1))]
        visible = None if self.through else self.select_mask(np.ones((h,w), bool), False)
        if self._face_graph_cache is None: self._face_graph_cache = mesh_ops.FaceGraph(item["F"])
        return magic_region(item["V"], item["F"], seed, self.magic_angle, visible, self._face_graph_cache)

    def mask_points(self):
        """World positions of the selected elements (engine order) for the delete-as-mask PLY."""
        return np.asarray(self.elements[0], np.float32)[self.selection]

    # ---------------- GL ----------------
    def initializeGL(self):
        from app2 import edit_gl
        self.render_ready = False
        try: self.scene = edit_gl.EditScene()
        except Exception as e: self.error = str(e); self.gl_failed.emit(str(e)); return
        try:
            from app2 import gpu_select
            self.gsel = gpu_select.GpuSelector()
        except Exception as e:                           # transform feedback did not link: numpy selection instead
            self.gsel = None; self.gpu_ok = False; print("gpu selection off: %s" % e, flush=True)
        self._restore_scene()

    def _restore_scene(self):
        """A new GL context needs the saved mesh and selection uploaded again."""
        self.render_ready = False
        self._todo["items"] = list(self.items.values())
        if self.elements is not None: self._todo["elements"] = self.elements
        self._push_flags()
        self._todo["fill_preview"] = getattr(self, "_fill_preview", None)
        self._todo["lines"] = getattr(self, "_lines", [])

    def paintGL(self):
        self.nav_cube.sync()
        if self.scene is None: return
        t0 = time.perf_counter()
        todo, self._todo = self._todo, {}
        if "items" in todo:
            for k in list(self.scene.items): self.scene.remove(k)
            for it in todo["items"]:
                try:
                    if it["kind"] == "mesh":
                        self.scene.set_mesh(it["key"], it["V"], it["N"], it["F"], face_rgb=it.get("face_rgb"), vertex_rgb=it.get("vertex_rgb"), model=it.get("model"), colour=it.get("colour"))
                    else:
                        self.scene.set_points(it["key"], it["P"], it["N"], rgb=it.get("rgb"), model=it.get("model"), colour=it.get("colour"))
                except Exception as e: self.error = str(e); print("view:", e, flush=True)
        if "fill_preview" in todo:
            self.scene.remove("fill-preview")
            it = todo["fill_preview"]
            if it is not None:
                self.scene.set_mesh("fill-preview", it["V"], it["N"], it["F"], model=it.get("model"), colour=np.float32([.72, .40, .95]))
        if "lines" in todo and hasattr(self.scene, "set_lines"):
            try: self.scene.set_lines(todo["lines"] or [])
            except Exception as e: print("view lines:", e, flush=True)
        if "items" in todo or "vis" in todo or self.only is not None:
            for k, si in self.scene.items.items(): si.visible = (k in self.only) if self.only is not None else self.items.get(k, {}).get("visible", True)
        for k, M in (todo.get("models") or {}).items():
            if k in self.scene.items: self.scene.items[k].model = M
        for k, (c, _picked) in (todo.get("colours") or {}).items():
            if k in self.scene.items: self.scene.items[k].colour = c
        if "elements" in todo and self.gsel is not None:
            try: self.gsel.set_elements(*todo["elements"])
            except Exception as e: self.gsel = None; self.gpu_ok = False; print("gpu selection off: %s" % e, flush=True)
        if "flags" in todo and self.edit_key in self.scene.items:
            self.scene.redbit_platter = todo.get("platter", False)
            self.scene.set_flags(self.edit_key, todo["flags"])
        self.scene.capture_colors=getattr(self,"capture_colors",False)
        if self.scene.capture_colors:
            from app2.qt.capture_key import layout
            self.nav_cube.set_bottom_inset(getattr(self,"capture_key_inset",0)+int(layout(self.width(),self.height())[0].height())+self.fontMetrics().lineSpacing())
        self.scene.only_marked = self.marked_only
        self.scene.flat = self.flat; self.scene.mesh_points = self.mesh_points
        w, h = self._px(); self.scene.draw(self.cam.view, self.cam.projection(w / max(h, 1)), w, h)
        self.render_ready = bool(self.items) and not self.error and all(k in self.scene.items for k in self.items)
        self.paint_ms = (time.perf_counter() - t0) * 1000.0

    def _px(self):
        r = self.devicePixelRatioF(); return max(1, int(self.width() * r)), max(1, int(self.height() * r))

    def resizeEvent(self, e):
        super().resizeEvent(e); self.overlay.resize(self.size())
        self.empty_background.setGeometry(self.rect())

    @staticmethod
    def front_pixel(img, xi, yi, k):
        """(yi, xi, depth) of the surface nearest the camera within k pixels of (xi, yi): a click through a gap in a point cloud
        or a thin mesh must hit the front, not the far side seen through the gap. Depth 1.0 means no surface."""
        h, w = img.shape[:2]; y0, y1, x0, x1 = max(0, yi - k), min(h, yi + k + 1), max(0, xi - k), min(w, xi + k + 1)
        win = np.asarray(img[y0:y1, x0:x1], float)
        if not (win < 1.0).any(): return yi, xi, float(img[yi, xi])
        j = int(np.argmin(np.where(win < 1.0, win, np.inf))); dy, dx = divmod(j, win.shape[1])
        return y0 + dy, x0 + dx, float(win[dy, dx])

    @staticmethod
    def front_point(X, V, pj, px, py, w, h, r, band=3.0):
        """The point of X (n x 3, world) under screen (px, py) within r px, among the front-most layer (within band mm of the nearest to
        the camera), the one closest to the click on screen. None when no point is there."""
        eye = X @ V[:3, :3].T + V[:3, 3]; cw = eye @ pj[3, :3] + pj[3, 3]; ok = cw > 1e-6
        clip = eye @ pj[:3, :3].T + pj[:3, 3]; den = np.where(ok, cw, 1.0)
        sx = (clip[:, 0] / den * 0.5 + 0.5) * w; sy = (0.5 - clip[:, 1] / den * 0.5) * h
        near = ok & (np.abs(sx - px) <= r) & (np.abs(sy - py) <= r)
        if not near.any(): return None
        idx = np.nonzero(near)[0]; depth = -eye[idx, 2]
        front = idx[depth <= depth.min() + band]
        j = front[int(np.argmin((sx[front] - px) ** 2 + (sy[front] - py) ** 2))]
        return X[j]

    def _marker_at(self, pos, r=11):
        """The label of the numbered dot under pos (device independent pixels), or None."""
        w, h = max(1, self.width()), max(1, self.height()); P = np.array([m[0] for m in self.markers], float)
        x, y, ze = self.cam.project(P, w, h); best = None
        for (_xyz, _col, label), xx, yy, z in zip(self.markers, x, y, ze):
            if z <= 0 or not np.isfinite(xx + yy): continue
            d = (xx - pos.x()) ** 2 + (yy - pos.y()) ** 2
            if d <= r * r and (best is None or d < best[0]): best = (d, label)
        return None if best is None else best[1]

    def pick_front(self, pos, r=6):
        """The shown model's own point nearest the camera within r px of pos (None when none is there). Unlike the depth buffer it
        cannot return empty space: a hole in a sparse point cloud gave the far plane."""
        if not self.render_ready or self.elements is None or not len(self.elements[0]): return None
        w, h = self._px(); k = self.devicePixelRatioF()
        it = self.items.get(self.edit_key) or {}; M = it.get("model")
        E = np.asarray(self.elements[0], np.float64)
        E = E[::max(1, len(E) // 600000)] if len(E) > 600000 else E
        X = E if M is None else E @ np.asarray(M, np.float64)[:3, :3].T + np.asarray(M, np.float64)[:3, 3]
        return self.front_point(X, np.asarray(self.cam.view, np.float64), np.asarray(self.cam.projection(w / h), np.float64),
                                pos.x() * k, pos.y() * k, w, h, r * k)

    def world_at(self, pos, reach=0, front=0):
        """The surface point under pos. reach (px): when the pixel itself shows no surface (a point cloud has gaps), snap to the
        nearest pixel within reach that does."""
        if self.scene is None: return None
        r = self.devicePixelRatioF(); w, h = self._px(); x, y = pos.x() * r, pos.y() * r
        self.makeCurrent()
        try: img = self.scene.read_depth_image(w, h)
        finally: self.doneCurrent()
        xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1)); d = img[yi, xi]
        if d >= 1.0 and reach:
            k = int(reach * r); y0, y1, x0, x1 = max(0, yi - k), min(h, yi + k + 1), max(0, xi - k), min(w, xi + k + 1)
            win = np.asarray(img[y0:y1, x0:x1]); ys, xs = np.nonzero(win < 1.0)
            if len(ys):
                j = int(np.argmin((ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2)); yi, xi = int(ys[j] + y0), int(xs[j] + x0); d = img[yi, xi]; x, y = float(xi), float(yi)
        if front and d < 1.0:
            yi, xi, d = self.front_pixel(img, xi, yi, int(front * r)); x, y = float(xi), float(yi)
        return None if d >= 1.0 else self.cam.unproject(x, y, d, w, h)

    # ---------------- selecting ----------------
    def _mask(self):
        """The drawn rectangle / lasso as a device-pixel bool image (QPainterPath filled into a QImage)."""
        w, h = self._px(); r = self.devicePixelRatioF()
        img = QImage(w, h, QImage.Format.Format_Grayscale8); img.fill(0)
        p = QPainter(img); p.scale(r, r); p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(255, 255, 255))
        if self.tool == "brush":
            p.setPen(QPen(QColor(255, 255, 255), self.brush_px * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            for a, b in zip(self._stroke, self._stroke[1:]): p.drawLine(a,b)
            p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(self._stroke[0], self.brush_px, self.brush_px)
        elif self.tool == "rect": p.drawRect(QRectF(self._stroke[0], self._stroke[-1]).normalized())
        else:
            path = QPainterPath(self._stroke[0])
            for q in self._stroke[1:]: path.lineTo(q)
            path.closeSubpath(); p.drawPath(path)
        p.end()
        a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w]
        return a > 127

    def _edit_view(self):
        """world -> eye for the edit item's own coordinates: the camera's view times the item's model matrix. The
        elements (face centroids) are the file's coordinates; the item is drawn turned by its saved orientation. With
        the plain camera view the brush picked faces a turn away from the circle."""
        it = self.items.get(self.edit_key) or {}
        M = it.get("model")
        return self.cam.view if M is None else np.asarray(self.cam.view, float) @ np.asarray(M, float)

    def select_mask(self, mask, through):
        """bool per element (engine order) inside a device-pixel mask: GPU first, numpy if it fails."""
        if self.elements is None or self.scene is None: return np.zeros(self.n, bool)
        w, h = self._px(); pj = self.cam.projection(w / h)
        it = self.items.get(self.edit_key) or {}
        reach = float(getattr(self.scene, "point_size", 0.0)) if it.get("kind") == "points" else 0.0    # a point's sprite
        view = self._edit_view()
        t0 = time.perf_counter(); self.makeCurrent(); got = None
        try:
            depth = None if through else self.scene.read_depth_image(w, h)
            # the GPU holds the shown result's elements only after the next paint: until then, numpy
            gpu_fresh = self.gsel is not None and "elements" not in self._todo and self.gsel.n == self.n
            if gpu_fresh and self.cam.projection_mode != "orthographic":
                try: got = self.gsel.select(mask, view, pj, self.cam.near, self.cam.far, depth, through=through, reach_mm=reach); self.select_path = "gpu"
                except Exception as e: print("gpu selection failed, numpy instead: %s" % e, flush=True); self.gsel = None
            if got is None:
                if self._np_elems is None: self._np_elems = sc.Elements(*self.elements)
                got = self._np_elems.select_screen(mask, view, pj, self.cam.near, self.cam.far, depth, through=through, reach_mm=reach)
                self.select_path = "numpy"
        finally: self.doneCurrent()
        self.last_select_ms = (time.perf_counter() - t0) * 1000
        return got

    def _finish_stroke(self):
        if self.tool in ("rect", "lasso", "brush") and len(self._stroke) >= {"rect":2,"lasso":3,"brush":1}[self.tool] and self.n:
            got = self.select_mask(self._mask(), self.through)
            if len(got) != len(self.selection): got = np.zeros(len(self.selection), bool)     # never mix two results
            if not got.any() and self.tool in ("rect", "lasso"): self.selected_nothing.emit(self.tool)
            self._apply_selection(got)
        self._stroke = []; self.overlay.update(); self._push_flags()

    # ---------------- mouse ----------------
    def _armed(self): return self.tool in ("rect", "lasso", "brush", "magic", "piece") and self.tools_on and self.n > 0

    def mousePressEvent(self, e):
        self.clear_hover()
        self._press_at = e.position(); self._pick_at = e.position() if (self.tool == "pairpick" and self.tools_on and e.button() == Qt.MouseButton.LeftButton) else None
        if self.tool == "fin" and self.tools_on and e.button() == Qt.MouseButton.LeftButton and self._arrow_at(e.position()) is None and self._ring_at(e.position()) is None:
            point = self.world_at(e.position(), reach=4)                # the gizmo's arrows win over a surface pick under them
            if point is not None: self.surface_picked.emit(point)
            return
        if self.gizmo is not None and e.button() == Qt.MouseButton.LeftButton:
            ka = self._arrow_at(e.position())
            if ka is not None:
                from app2.qt import move_gizmo as MG
                g = self.gizmo; g.update(adrag=MG.ArrowDrag(self.cam, g["c"], ka, (e.position().x(), e.position().y()),
                                                            max(1, self.width()), max(1, self.height()), self.arrow_length()), mm=0.0, pos=e.position())
                self.overlay.update(); return
            k = self._ring_at(e.position())
            if k is not None:
                from app2.qt import rotate_gizmo as RG
                g = self.gizmo; g.update(drag=RG.RingDrag(self.cam, g["c"], g["r"], k, (e.position().x(), e.position().y()),
                                                          max(1, self.width()), max(1, self.height())), deg=0.0, pos=e.position())
                self.overlay.update(); return
        self._drag = e.position(); self._btn = e.button()
        if e.button() == Qt.MouseButton.LeftButton and self._armed():
            self._subtract = self.selection_mode == "subtract" or bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._force_add = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier) and not self._subtract      # Shift: add to what is selected
            if self.tool == "magic": self._magic_pick(e.position()); return
            if self.tool == "piece": self._piece_pick(e.position()); return
            self._stroke = [e.position()]

    def mouseMoveEvent(self, e):
        self._hover = e.position()
        if self.tool == "brush": self.overlay.update()
        g = self.gizmo
        if g is not None and g.get("adrag") is not None:
            g["mm"] = g["adrag"].moved((e.position().x(), e.position().y()), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            g["pos"] = e.position(); self.arrow_move.emit(g["adrag"].k, g["mm"], False); self.overlay.update(); return
        if g is not None and g.get("drag") is not None:
            g["deg"] = g["drag"].angle((e.position().x(), e.position().y()), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            g["pos"] = e.position(); self.ring_turn.emit(g["drag"].k, g["deg"], False); self.overlay.update(); return
        if g is not None and self._drag is None:
            ka = self._arrow_at(e.position()); k = None if ka is not None else self._ring_at(e.position())
            if ka != g.get("ahover") or k != g.get("hover"): g["ahover"] = ka; g["hover"] = k; self.overlay.update()
            self.setCursor(Qt.CursorShape.PointingHandCursor if (k is not None or ka is not None) else Qt.CursorShape.CrossCursor if self.tool != "orbit" else Qt.CursorShape.OpenHandCursor)
        if self._drag is None:
            if self.tool in ("brush", "magic", "piece"):
                self.clear_hover(); self._hover_timer.start()
            return
        if self._btn == Qt.MouseButton.LeftButton and self.tool in ("magic", "piece") and self._armed(): return
        d = e.position() - self._drag; self._drag = e.position()
        if self._btn == Qt.MouseButton.LeftButton and self._stroke:
            if self.tool == "rect": self._stroke[1:] = [e.position()]
            else: self._stroke.append(e.position())
            self.overlay.update(); return
        turn = self._btn == (Qt.MouseButton.RightButton if self._armed() else Qt.MouseButton.LeftButton)
        if turn: self.cam.orbit(d.x(), d.y())
        else:
            w, h = self._px(); r = self.devicePixelRatioF(); self.cam.pan(d.x() * r, d.y() * r, h, w / h)
        self.camera_moved.emit(); self.update()

    def mouseReleaseEvent(self, e):
        g = self.gizmo
        if g is not None and g.get("adrag") is not None:
            k, mm = g["adrag"].k, g.get("mm", 0.0); g.update(adrag=None, mm=0.0, pos=None); self.overlay.update()
            self.arrow_move.emit(k, mm, True); return
        if g is not None and g.get("drag") is not None:
            k, deg = g["drag"].k, g.get("deg", 0.0); g.update(drag=None, deg=0.0, pos=None); self.overlay.update()
            self.ring_turn.emit(k, deg, True); return
        if self._btn == Qt.MouseButton.LeftButton and self._stroke: self._finish_stroke()
        if (self.markers and self.tool != "pairpick" and self._btn == Qt.MouseButton.LeftButton and self._press_at is not None
                and (e.position() - self._press_at).manhattanLength() < 5):
            hit = self._marker_at(e.position())
            if hit is not None: self.marker_clicked.emit(hit)
        pa, self._pick_at = getattr(self, "_pick_at", None), None
        if pa is not None and self.tool == "pairpick" and (e.position() - pa).manhattanLength() < 5:    # a click, not a turn
            point = self.pick_front(e.position())
            if point is not None: self.surface_picked.emit(point)
            else: self.pick_missed.emit()
        self._drag = None

    def leaveEvent(self, e):
        self._hover = None; self.clear_hover(); self.overlay.update(); super().leaveEvent(e)

    def wheelEvent(self, e):
        self.clear_hover()
        steps = e.angleDelta().y() / 120.0
        self.cam.zoom(steps, toward=self.world_at(e.position())); self.camera_moved.emit(); self.update()

    def mouseDoubleClickEvent(self, e):
        if not self._armed(): self.go_home()
