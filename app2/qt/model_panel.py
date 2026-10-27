# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Scan step's model panel: this scan session's model, big, in the middle of the window. Only this session's
sides show here (the combined model is the Build step's): the side being scanned, live, or the picked side's own model.

A stack of two pages under one overlay:
  GL page     ModelView (view3d.py) drawing the tracker's own cloud at the panel's size; it rides with the scanner
              (pose events) until the user turns it; Home, H or a double-click snap back. Only in points mode.
              A saved side picked shows its model here too: its cloud from this session, or its own mesh from the
              last Build model (side_mesh), turned the same way.
  image page  a picture: the worker's render in image mode (Tk's picture, turned through the session) or a saved
              side's thumb_model.png. Thumbs keep their pixel size (a 400x300 thumb blown up to the panel looked
              broken); the others letterbox.
Which page shows: whichever of "a cloud with points" and "a picture" came last (both carry the session's sequence
number); neither: the colour camera, live (show_live; the biggest area of the window is never an empty well), or the
empty well before the first colour frame.
The GL view goes Home when a cloud or pose carries a newer home_seq than it has seen (the session counts New side,
pick, delete and reloading a side; a latest-wins delivery that replaces one payload with a newer one still carries it).
ModelOverlay is a transparent sibling raised over both (a QOpenGLWidget composites through the backing store, so a
sibling draws over it correctly): the state ring, the title chip, the coverage key and the platter dial. The title chip
says exactly what is shown (title_parts): "Live: side 7", "Side 3 (saved) · 420 frames", "Live: color camera". It lets
every mouse event through; the Home view button is its own sibling above it."""
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QWidget, QStackedLayout, QVBoxLayout, QHBoxLayout, QLabel
from app2.qt.theme import qc, ui_font
from app2.qt.widgets import ImageTile, chip, button, fit_parts


class ModelImage(ImageTile):
    """The image page. In image mode a drag turns the model (through the session), the wheel zooms, a double-click
    goes Home; dx, dy are in the Tk panel's pixels (it was 600 px wide), so a drag turns as far as it did there."""
    dragged = Signal(float, float)
    wheeled = Signal(float)
    home_asked = Signal()

    def __init__(self, parent=None):
        super().__init__((), parent, radius=12); self._last = None; self.setMinimumSize(80, 60)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self._last = e.position()

    def mouseMoveEvent(self, e):
        if self._last is None: return
        d = e.position() - self._last; self._last = e.position()
        r = self.image_rect(QRectF(self.rect())) if self.img is not None else QRectF(self.rect())
        k = 600.0 / max(1.0, r.width())
        self.dragged.emit(d.x() * k, d.y() * k)

    def mouseReleaseEvent(self, e): self._last = None

    def wheelEvent(self, e): self.wheeled.emit(0.01 if e.angleDelta().y() > 0 else -0.01)

    def mouseDoubleClickEvent(self, e): self.home_asked.emit()


class ModelOverlay(QWidget):
    """Ring, title, coverage key and platter dial over the model. Paints only; mouse events go through."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.ring = None; self.parts = [("Live: color camera", None)]; self.covered = None; self.turn = None; self.empty_hint = None
        self.touch=None;self.distance_legend=False
        self.max_w = 10 ** 6                                # the title chip's room (the Home view button's left edge)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shown = fit_parts(self.parts, self.max_w, 13)
        box = None if getattr(self.parentWidget(),'external_header',False) else chip(p, r.x() + 12, r.y() + 12, shown, 13)
        rest = self.parts[len(shown):]                     # what did not fit goes on a second line, not over the picture
        if rest and box is not None: chip(p, r.x() + 12, box.bottom() + 4, fit_parts(rest, self.max_w, 13), 13)
        compact = self.width() < 560
        if self.empty_hint:
            f = ui_font(14); p.setFont(f); p.setPen(qc("ink3")); fm = QFontMetrics(f)
            p.drawText(QPointF(r.center().x() - fm.horizontalAdvance(self.empty_hint) / 2, r.center().y()), self.empty_hint)
        if self.touch is not None:
            fm=QFontMetrics(ui_font(12));line=fm.height()+5;y=r.bottom()-line*6-12
            x=r.left()+14;box=QRectF(x-5,y-14,max(240,min(r.width()-28,600)),line*6+16)
            p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(10,12,15,232));p.drawRoundedRect(box,6,6);p.setFont(ui_font(12))
            t=self.touch
            texts=[("Not scanned yet (holes)","lost"),("Rough or doubled, scan again","platter"),("Done","kept"),
                   ("%d spots left: %d holes, %d rough" % (t.get("left",0),t.get("holes",0),t.get("rough",0)),"ink"),
                   ("When the spots are green, Build again:","ink2"),("only the new frames are added.","ink2")]
            for text,color in texts:p.setPen(qc(color));p.drawText(QPointF(x,y),text);y+=line
        if self.distance_legend:
            from app2.qt.capture_key import paint
            paint(self,p,pending=getattr(self,'distance_pending',False))
        if self.covered is not None and self.touch is None and not self.distance_legend:                        # coverage key: the three colours of the model
            f_big = ui_font(22 if not compact else 17, QFont.Weight.Bold); f = ui_font(12)
            from app2.gl_core import COVERAGE
            keys = [("scan here", QColor(*COVERAGE[0])), ("getting there", QColor(*COVERAGE[1])), ("well covered", QColor(*COVERAGE[2]))]
            fm, fmb = QFontMetrics(f), QFontMetrics(f_big)
            head = "%d%% well covered" % round(100 * self.covered)
            w = max(fmb.horizontalAdvance(head), sum(fm.horizontalAdvance(k) + 26 for k, _ in keys)) + 24
            h = fmb.height() + fm.height() + 22
            box = QRectF(r.x() + 12, r.bottom() - 12 - h, w, h)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(10, 12, 15, 200)); p.drawRoundedRect(box, 8, 8)
            p.setFont(f_big); p.setPen(qc("ink")); p.drawText(QPointF(box.x() + 12, box.y() + 8 + fmb.ascent()), head)
            x = box.x() + 12; y = box.y() + 12 + fmb.height()
            p.setFont(f)
            for k, c in keys:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(c); p.drawRoundedRect(QRectF(x, y + fm.height() / 2 - 5, 14, 10), 3, 3)
                p.setPen(qc("ink2")); p.drawText(QPointF(x + 19, y + fm.ascent()), k); x += fm.horizontalAdvance(k) + 26
        if self.turn is not None:                           # the platter dial: this turn's degrees, turn number, gain
            deg, n, gain = self.turn; R = 34 if not compact else 26
            c = QPointF(r.right() - 16 - R - 6, r.bottom() - 16 - R - 6)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(10, 12, 15, 200)); p.drawEllipse(c, R + 8, R + 8)
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(qc("line"), 5)); p.drawEllipse(c, R, R)
            pen = QPen(qc("platter"), 5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
            p.drawArc(QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R), 90 * 16, -int(deg % 360 * 16))
            f = ui_font(15 if not compact else 12, QFont.Weight.Bold); p.setFont(f); p.setPen(qc("ink"))
            fm = QFontMetrics(f); t = "turn %d" % n
            p.drawText(QPointF(c.x() - fm.horizontalAdvance(t) / 2, c.y() + 2), t)
            f2 = ui_font(11); p.setFont(f2); fm2 = QFontMetrics(f2); t2 = "gain %d" % gain; p.setPen(qc("ink2"))
            p.drawText(QPointF(c.x() - fm2.horizontalAdvance(t2) / 2, c.y() + 4 + fm2.ascent()), t2)
        guide=getattr(self,"guidance",None)
        if guide:
            p.setPen(QColor(guide["color"])); p.setFont(ui_font(14))
            from app2.qt.scan_guide import guidance_rect
            p.drawText(guidance_rect(self),Qt.TextFlag.TextWordWrap.value,guide["text"])
        panel=self.parentWidget()
        sg=getattr(panel,"scan_guide",None)
        view=getattr(panel,"view",None)
        if sg and view is not None:
            from app2.qt.scan_guide import lines
            for pose in sg.get("poses",[]):
                for a,b,cone in lines(pose,sg["lens"]):
                    points=np.array([a,b])
                    depth=-(points@view.cam.view[2,:3]+view.cam.view[2,3])
                    if not (depth>1e-3).all():continue
                    x,y,z=view.cam.project(points,max(1,self.width()),max(1,self.height()))
                    if (z>0).all() and np.isfinite(x).all():
                        p.setPen(QPen(QColor(pose["color"]),1 if cone else 2,Qt.PenStyle.DashLine if cone else Qt.PenStyle.SolidLine))
                        p.drawLine(QPointF(x[0],y[0]),QPointF(x[1],y[1]))
        if getattr(self.parent(), 'frame', None) is not None: pass   # the panel's frame draws the outline round the title too
        elif self.ring:
            p.setPen(QPen(qc(self.ring), 3)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(1.5, 1.5, -1.5, -1.5), 11, 11)
        else:
            p.setPen(QPen(qc("line"), 1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRoundedRect(r, 12, 12)


from PySide6.QtWidgets import QComboBox


class PanelFrame(QWidget):
    """The 3D view's outline around its title row and picture both, like a camera tile's: paints only, the mouse
    goes through. The title row used to sit outside the outline, so the corners and the tops never lined up."""
    def __init__(self, parent=None):
        super().__init__(parent); self.ring = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, e):
        from app2.qt.widgets import RING_WIDTH
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); r = QRectF(self.rect())
        if self.ring: p.setPen(QPen(qc(self.ring), RING_WIDTH)); half = RING_WIDTH / 2
        else: p.setPen(QPen(qc("line"), 1)); half = 0.5
        p.setBrush(Qt.BrushStyle.NoBrush); p.drawRoundedRect(r.adjusted(half, half, -half, -half), 11, 11)


class ModelPanel(QWidget):
    """gl: points mode (a ModelView page); otherwise the image page only. home_asked: image mode's Home (the session
    re-renders); in GL mode Home is handled here."""
    home_asked = Signal()
    dragged = Signal(float, float)
    wheeled = Signal(float)

    def __init__(self, gl=False, parent=None):
        super().__init__(parent)
        self._project_basis = np.eye(3); self._bed_up = None; self._bed_context = None
        self.gl = gl
        outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        self.viewport=QWidget();outer.addWidget(self.viewport,1)
        self.stack=QStackedLayout(self.viewport);self.stack.setContentsMargins(0,0,0,0)
        self.view = None
        if gl:
            from app2.qt.view3d import ModelView
            self.view = ModelView(dbl_click="home"); self.stack.addWidget(self.view)
            self.view.following_changed.connect(self._refresh_overlay); self.view.home_asked.connect(self._refresh_overlay)
        self.image = ModelImage(); self.stack.addWidget(self.image)
        self.image.dragged.connect(self.dragged); self.image.wheeled.connect(self.wheeled); self.image.home_asked.connect(self.home)
        self.overlay = ModelOverlay(self)
        # color fps in the panel's top-right corner, off the picture (the small Color tile is empty while this shows it)
        self.rate_label = QLabel(self); self.rate_label.setProperty("role", "hint"); self.rate_label.setToolTip("Color camera frames shown per second")
        self.rate_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.rate_label.hide()
        self.rate_timer = QTimer(self); self.rate_timer.setInterval(1000); self.rate_timer.timeout.connect(self._show_rate); self.rate_timer.start()
        self.scan_guide=None; self.motion_provider=None
        from scanner.scan_guide import GravityAlignment
        self.gravity_alignment=GravityAlignment()
        self.guide_target=QComboBox(self); self.guide_target.setAccessibleName("Scanner target")
        self.guide_target.hide(); self.guide_target.currentIndexChanged.connect(self._guide_changed)
        self.home_b = button("Home view", "float", "Back to the scanner's view (H, or double-click the model)", self)
        self.home_b.clicked.connect(self.home)
        from app2.qt.choices import Choices
        self.distance_mode=Choices(self);self.distance_mode.addItems(['Distance','Coverage','Plain']);self.distance_mode.hide()
        for b,tip in zip(self.distance_mode.buttons,('Color by how far the scanner was when each spot was saved. Fills in once a pause saves the views.',
                                                     'Color by how many frames saw each spot: purple scan here, light blue getting there, deep blue well covered.',
                                                     'One plain color, to judge the shape.')):b.setToolTip(tip)
        self.distance_mode.currentIndexChanged.connect(self._distance_mode_changed);self._last_cloud=None;self._last_mesh=None
        self.cloud_seq = -1; self.img_seq = -1; self.cloud_covered = None; self.pivot = None
        self.img_payload = None; self.turn = None; self.page = "empty"; self.home_seq = 0; self.live_img = None;self.live_guides=None;self.live_options={}
        self.recording=False;self.picked=None;self.frames=0;self.mesh_shown=False;self.touch_context=False;self.touch_progress={}
        self._pick_page()

    def reserve_header(self):
        """Scan titles, Home and fps use layout space above the viewport."""
        from app2.qt.widgets import ElidedLabel
        self.external_header=True
        from app2.qt.widgets import RING_WIDTH
        self.header=QWidget();self.header.setObjectName('ModelHeader');row=QHBoxLayout(self.header)
        self.header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.header.setStyleSheet('#ModelHeader {background:#20262e;border-top-left-radius:9px;border-top-right-radius:9px;}')   # a camera tile's band
        line=self.fontMetrics().lineSpacing();row.setContentsMargins(line//2,line//6,line//6,line//6)
        self.header_title=ElidedLabel();self.header_title.setMinimumWidth(1)
        row.addWidget(self.header_title,1);row.addWidget(self.rate_label);row.addWidget(self.home_b)
        self.layout().insertWidget(0,self.header)
        self.ring_w=RING_WIDTH;self.frame=PanelFrame(self);self.frame.ring=self.overlay.ring   # the outline goes round the title too
        self.image.radius=11-RING_WIDTH                     # the picture's corners follow the outline's inner curve
        m=self.layout().contentsMargins();self.layout().setContentsMargins(RING_WIDTH,m.top()+RING_WIDTH,RING_WIDTH,m.bottom()+RING_WIDTH)
        self._refresh_overlay()

    def set_context(self, recording, picked, frames):
        """What the session is doing, for the title: recording or not, the picked side (or None), its frames."""
        c = (bool(recording), picked, int(frames or 0))
        if c != (self.recording, self.picked, self.frames):
            self.recording, self.picked, self.frames = c; self._refresh_overlay()

    def set_touch_context(self,on,progress=None):
        self.touch_context=bool(on)
        if progress is not None:self.touch_progress=dict(progress)
        if not on:self.touch_progress={}
        self._refresh_overlay()

    def set_project_basis(self, basis, reset_bed=False):
        from app2.view_orientation import validate_basis
        self._project_basis = validate_basis(basis)
        if reset_bed: self._bed_up = None; self._bed_context = None
        self._apply_bed_basis()

    def _apply_bed_basis(self):
        if self.view is None: return
        from app2.view_orientation import basis_with_bed_up, basis_with_bed_up_and_front
        basis = self._project_basis
        if self._bed_up is not None:
            try:
                front = getattr(self, '_bed_front', None)
                if front is not None and np.allclose(basis, np.eye(3)):        # no saved choice yet: Front is where the scanner started
                    basis = basis_with_bed_up_and_front(self._bed_up, front)
                else: basis = basis_with_bed_up(self._bed_up, basis)
            except (TypeError, ValueError): self._bed_up = None
        self.view.nav_cube.set_basis(basis)

    def _bed_from(self, payload):
        self._bed_context = (payload.get('side_no'), payload.get('home_seq', self.home_seq))
        self._bed_up = payload.get('bed_up'); self._bed_front = payload.get('bed_front')
        self._apply_bed_basis()

    def _home_from(self, payload):
        hs = payload.get("home_seq")
        if hs is not None and hs > self.home_seq:
            self.home_seq = hs
            self._bed_up = None; self._bed_context = None; self._apply_bed_basis()
            if self.view is not None: self.view.go_home()

    # ---- session events
    def _guide_changed(self, *_):
        if getattr(self,"_guide_pose",None) is not None: self._guide_update(self._guide_pose)

    def _guide_update(self, T):
        self._guide_pose=T
        guide=self.scan_guide
        if not guide or not guide.get("poses"): self.overlay.guidance=None; return
        from scanner.scan_guide import guidance
        target=guide["poses"][max(0,self.guide_target.currentIndex())]
        self.guide_target.setToolTip(target["text"])
        gravity=None
        if self.motion_provider is not None:
            state,sample,*_=self.motion_provider()
            # Device axes have not been calibrated. Use motion only if a saved mapping is supplied.
            mapping=getattr(self,"motion_to_camera",None)
            if state=="Receiving" and sample is not None and mapping is None:
                mapping=self.gravity_alignment.add(T,sample.acceleration,guide["axis"])
            if state=="Receiving" and sample is not None and mapping is not None:
                gravity=np.asarray(T)[:3,:3] @ (np.asarray(mapping) @ np.asarray(sample.acceleration))
        self.overlay.guidance=guidance(T,target["T"],gravity,turntable=guide.get("mode")=="turntable",axis=guide["axis"],center=np.asarray(guide["center"]),hold_still=target.get("hold_still",False))
        done=getattr(self,"_guide_done",None)
        if done:
            if not self.overlay.guidance["text"].endswith("."):self.overlay.guidance["text"]+="."
            self.overlay.guidance["text"]+=f" {done[0]} of {done[1]} spots done."
        self.overlay.update()

    def on_cloud(self, c):
        """GL mode: the model so far (mm). Empty arrays: the side changed; forget the cloud."""
        from app2.scan import timing; timing.mark('gui_cloud_start')
        try: self._on_cloud(c)
        finally: timing.mark('gui_cloud_end')

    def _on_cloud(self, c):
        # Keep the last colored surface while its replacement is measured.
        # Pending evidence is never painted as a downgrade to gray.
        previous=self._last_cloud
        if self.view is not None and self.distance_mode.currentIndex()==0 and c.get('B') is not None and previous is not None and c.get('side_no')==previous.get('side_no') and previous.get('B') is not None and np.any(previous['B']!=255) and np.all(c['B']==255):
            self._pending_distance_cloud=c;self.overlay.distance_pending=True;self.overlay.update();return
        self.overlay.distance_pending=False;self._pending_distance_cloud=None
        if c.get('live_build') and not getattr(self,'_live_quality',False):
            self.distance_mode.blockSignals(True);self.distance_mode.setCurrentIndex(1);self.distance_mode.blockSignals(False)
        self._live_quality=bool(c.get('live_build'))
        self._last_cloud=c;self._last_mesh=None
        self.set_touch_context(c.get("touch") is not None,c.get("touch"))
        if self.view is None: return
        guide=c.get("scan_guide")
        if guide is not self.scan_guide:
            self.scan_guide=guide; self.overlay.guidance=None; self._guide_pose=None; self.guide_target.blockSignals(True); self.guide_target.clear()
            if guide:
                self.guide_target.addItems([f'{p["rank"]}. {p["text"]}' for p in guide["poses"]])
            self.guide_target.blockSignals(False); self.guide_target.setVisible(bool(guide and guide["poses"]))
            self.guide_target.setGeometry(15,45,max(100,min(600,self.width()-30)),30); self.guide_target.raise_()
        touch=c.get("touch"); self._guide_done=(touch["done"],touch["total"]) if touch else None
        self._home_from(c)
        self._bed_from(c if len(c["P"]) else {})
        n = len(c["P"]); style = getattr(self, 'view_style', 'Surface')
        if style == 'Surface' and c.get("F") is None and c.get('surface_soon') and n:   # the first frames of a side: the surface is on its way; no dots flash first
            if self.mesh_shown: self.view.set_mesh(None); self.mesh_shown = False
            self.view.set_points(None); self.cloud_covered = c["covered"]; self.pivot = c["pivot"]
            self.cloud_seq = c.get("seq", self.cloud_seq + 1); self._pick_page(); return
        if style == 'Surface' and c.get("F") is not None and n:  # the live surface: a real shape, colored like the dots were
            levels=self._display_levels(dict(c,P=c["V"],L=c["VL"],B=c["VB"]))
            colors=(np.asarray(self.view.point_palette)[levels]*255).round().astype(np.uint8)
            # keep the camera whenever something was already shown or Home is set (the view follows the scanner from the
            # first pose): a first mesh that reframed to the saved Home rotation flipped the view 180 for a second, then
            # the next pose snapped it back; Dots to Surface spun the same way
            shown = self.mesh_shown or getattr(self.view, "_have_points", False) or getattr(self.view.cam, "home", None) is not None
            self.view.set_points(None); self.view.set_mesh(c["V"], c["VN"], colors, c["F"], keep_view=shown); self.mesh_shown = True
            self.cloud_covered = c["covered"]; self.pivot = c["pivot"]
            self.cloud_seq = c.get("seq", self.cloud_seq + 1); self._pick_page(); return
        if self.mesh_shown: self.view.set_mesh(None); self.mesh_shown = False     # a saved side's mesh gives way
        if n:
            self.view.set_points(c["P"], c["N"], self._display_levels(c)); self.cloud_covered = c["covered"]; self.pivot = c["pivot"]
        else:
            self.view.set_points(None); self.cloud_covered = None; self.pivot = None; self.turn = None
        self.cloud_seq = c.get("seq", self.cloud_seq + 1); self._pick_page()

    def _display_levels(self,c):
        from app2 import gl_core
        palette=gl_core.PALETTE.copy();mode=self.distance_mode.currentIndex();band=c.get('B')
        self.distance_mode.setVisible(getattr(self, 'external_distance_mode', False) or band is not None)
        # Distance needs saved views; before the first pause it would be all gray, so show coverage meanwhile.
        saved=band is not None and bool(np.any(np.asarray(band)!=255))
        self.distance_mode.setItemEnabled(0,band is None or saved)
        self.distance_mode.buttons[0].setToolTip('Color by how far the scanner was when each spot was saved. Fills in once a pause saves the views.' if (band is None or saved)
                                                 else 'No saved views yet for this model: Distance fills in after a pause saves them. Showing Coverage meanwhile.')
        if mode==0 and band is not None and not saved:mode=1
        self.overlay.distance_legend=bool(mode==0 and band is not None)
        if mode==0 and band is not None:
            from app2.scan.range_guide import BAND_RGB,UNSEEN_RGB
            palette[:3]=np.asarray([BAND_RGB[k] for k in ('off','ok','best')])/255;palette[4]=np.asarray(UNSEEN_RGB)/255
            levels=np.where(np.asarray(band)==255,4,band).astype(np.uint8)
        elif mode==2:levels=np.full(len(c['P']),4,np.uint8)
        else:
            levels=c['L']
            if c.get('live_build'):
                from app2.scan.range_guide import BAND_RGB
                palette[:3]=np.asarray([BAND_RGB[k] for k in ('off','ok','best')])/255
        self.view.capture_colors=bool(mode==0 and band is not None)
        self.view.point_palette=palette;return levels
    def set_view_style(self, style):
        """'Surface' (the live mesh) or 'Dots' (the splats): redraw the last cloud the new way."""
        if style not in ('Surface', 'Dots') or style == getattr(self, 'view_style', 'Surface'): return
        self.view_style = style
        if self._last_cloud is not None and self.view is not None: self.on_cloud(self._last_cloud)

    def _distance_mode_changed(self):
        if self.view is None:return
        pending=getattr(self,"_pending_distance_cloud",None)
        if pending is not None and self.distance_mode.currentIndex()!=0:self.on_cloud(pending);return
        if self._last_cloud is not None:self.view.set_levels(self._display_levels(self._last_cloud))
        elif self._last_mesh is not None:self.on_side_mesh(self._last_mesh)
        self._refresh_overlay()

    def on_side_mesh(self, m):
        """GL mode: a saved side's own mesh from the last build (mm, its scan frame), with the scanner's last pose on
        it as Home."""
        if self.view is None: return
        self._home_from(m)
        self._bed_from(m)
        self.view.set_points(None); self.cloud_covered = None; self.turn = None
        self._last_mesh=m;self._last_cloud=None;self.distance_mode.setVisible(getattr(self,'external_distance_mode',False) or m.get('B') is not None);self.overlay.distance_legend=bool(m.get('B') is not None and self.distance_mode.currentIndex()==0)
        from app2.scan.capture_distance import rgb
        color=rgb(m["B"]) if m.get("B") is not None and self.distance_mode.currentIndex()==0 else None
        self.view.capture_colors=bool(color is not None)
        self.view.set_mesh(m["V"], m["N"], color, m["F"]); self.mesh_shown = True; self.pivot = m["pivot"]
        if m.get("T_mm") is not None: self.view.set_home(m["T_mm"], m["pivot"])
        self.cloud_seq = m.get("seq", self.cloud_seq + 1); self._pick_page()

    def on_pose(self, p):
        """GL mode: the scanner's pose (camera -> world, mm): Home follows it, and so does the view at Home."""
        if self.view is None: return
        self._home_from(p)
        T = p["T_mm"]
        if p.get("ok",True):self._guide_update(T)
        elif self.scan_guide:
            self._guide_pose = None
            from scanner.scan_guide import guidance
            self.overlay.guidance=guidance(T,T,tracking_ok=False);self.overlay.update()
        pivot = self.pivot if self.pivot is not None else T[:3, 3] + T[:3, 2] * 250.0
        self.view.follow_scanner(T, pivot)

    def on_image(self, m):
        self.img_payload = m if m.get("img") is not None else None
        self.img_seq = m.get("seq", self.img_seq + 1)
        if m["kind"] == "thumb" or m.get("img") is None: self.turn = None
        self._pick_page()

    def show_live(self, img, guides=None, options=None):
        """The colour camera (with the depth outline), shown here while there is no model to show."""
        self.live_img = img;self.live_guides=guides;self.live_options=dict(options or {})
        if self.page in ("empty", "live"): self._pick_page()

    def on_turn(self, t):
        self.turn = None if t.get("deg") is None else (t["deg"], t["n"], t["gain"]); self._refresh_overlay()

    def set_ring(self, colour):
        if colour != self.overlay.ring: self.overlay.ring = colour; self.overlay.update()
        if getattr(self, 'frame', None) is not None and colour != self.frame.ring: self.frame.ring = colour; self.frame.update()

    def set_lens(self, cal):
        """cal: (fx, fy, cx, cy) of the depth camera (Pl.bin)."""
        if self.view is not None: self.view.set_lens(*cal)

    def home(self):
        if self.page == "gl": self.view.go_home()
        else: self.home_asked.emit()
        self._refresh_overlay()

    def your_view(self):
        if self.page == "gl": return not self.view.following
        return bool(self.img_payload and self.img_payload.get("your_view"))

    # ---- pages
    def _pick_page(self):
        has_pts = self.view is not None and (self.view.has_points() or self.view.has_mesh())
        img = self.img_payload is not None
        if has_pts and (not img or self.cloud_seq > self.img_seq): page = "gl"
        elif img: page = "image"
        elif self.live_img is not None: page = "live"
        else: page = "empty"
        m = self.img_payload
        if page == "image": self.image.set_image(m["img"], fit="native" if m["kind"] == "thumb" else "fit")
        elif page == "live": self.image.set_image(self.live_img, fit="fit")
        elif page == "empty": self.image.set_image(None)
        self.image.set_guides(self.live_guides if page=="live" else None,self.live_options)
        self.page = page
        if page == "gl": self.stack.setCurrentWidget(self.view)
        else: self.stack.setCurrentWidget(self.image)
        self.overlay.raise_(); self.home_b.raise_()
        self._refresh_overlay()

    def title_parts(self):
        """The title chip: exactly what the panel shows. [(text, colour or None)]; the first part is the title."""
        kind=self.img_payload["kind"] if (self.page=="image" and self.img_payload) else None
        if self.touch_context:
            if not self.frames:
                if self.touch_progress.get("total",0)==0:return [("Touch up · no weak spots found · press Start and scan the area you want better",None)]
                return [("Touch up · press Start, then aim at the red and purple spots (or any area you want better)",None)]
            return [("Touch up · %d frames · %d of %d spots done" % (self.frames,self.touch_progress.get("done",0),self.touch_progress.get("total",0)),None)]
        no=self.picked
        if self.page in ("live", "empty"):
            if self.recording and no is not None: return [("Live: side %d" % no, None), ("color camera until the model starts", None)]
            if no is not None: return [("Live: color camera", None), ("side %d starts when you press Start" % no, None)]
            if self.page == "empty": return [("Live: color camera", None)]       # the centered hint says the rest
            return [("Live: color camera", None), ("the model grows here once you scan", None)]
        if self.recording and no is not None:
            return [("Live: side %d" % no, None)] + ([("your view", None)] if self.your_view() else [])
        if no is None: return [("Model", None)]
        head = "Side %d (saved) · %d frame%s" % (no, self.frames, "" if self.frames == 1 else "s")
        if kind == "thumb": return [(head, None), ("still picture", None)]
        extra = [("from the last build", None)] if (self.page == "gl" and self.mesh_shown) else []
        return [(head, None)] + extra + ([("your view", None)] if self.your_view() else [])

    def _refresh_overlay(self, *_):
        o = self.overlay; kind = self.img_payload["kind"] if (self.page == "image" and self.img_payload) else None
        if getattr(self,'external_header',False):
            self.header_title.setText(' · '.join(text for text,_ in self.title_parts()))
        o.parts=self.title_parts();o.touch=self.touch_progress if self.touch_context and not o.distance_legend else None
        o.covered = self.cloud_covered if self.page == "gl" else (self.img_payload.get("covered") if kind == "live" else None)
        o.turn = self.turn if (self.page == "gl" or kind == "live") else None
        o.empty_hint = "The model grows here while you scan." if self.page == "empty" else None
        o.max_w = max(120, self.width() - (self.home_b.width() + 36 if (self.page == "gl" or kind == "live") else 24))
        self.home_b.setVisible(self.page == "gl" or kind == "live")
        if self.page == "live" and self.image.img is not None:   # keep the chip in the dark margin beside the picture, never on it
            margin = self.image.image_rect(QRectF(self.image.rect())).left() - 24
            if margin >= 120: o.max_w = min(o.max_w, margin)
        if self.view is not None:
            compact = self.width() < 560
            f_big = ui_font(22 if not compact else 17, QFont.Weight.Bold)
            f = ui_font(12)
            clearance = QFontMetrics(f_big).height() + QFontMetrics(f).height() + 40 if o.covered is not None else 0
            if getattr(o,"touch",None) is not None:
                from app2.qt.scan_guide import touch_legend_height
                clearance=max(clearance,touch_legend_height(o)+2*QFontMetrics(ui_font(14)).height()+26)
            if o.distance_legend:
                from app2.qt.capture_key import layout
                clearance=max(clearance,int(layout(o.width(),o.height())[0].height())+QFontMetrics(ui_font()).lineSpacing())
            self.view.nav_cube.set_bottom_inset(clearance)
        if self.guide_target.isVisible():
            self.guide_target.setGeometry(15,45,max(100,min(700,self.width()-30)),30)
            self.guide_target.raise_()
        if self.distance_mode.isVisible() and not getattr(self, 'external_distance_mode', False):
            self.distance_mode.adjustSize();line=self.fontMetrics().lineSpacing();self.distance_mode.move(max(line,self.home_b.x()-self.distance_mode.width()-line//2),self.home_b.y()+(self.home_b.height()-self.distance_mode.height())//2);self.distance_mode.raise_()
            if self.guide_target.isVisible() and self.guide_target.geometry().intersects(self.distance_mode.geometry()):
                box=self.guide_target.geometry();box.setWidth(max(1,self.distance_mode.x()-line-box.x()));self.guide_target.setGeometry(box)
        o.update()

    def set_camera_notice(self,button):
        """Reserve a real row so a camera action never covers the image or its box."""
        self.camera_notice=button
        self._place_camera_notice()
    def _place_camera_notice(self):
        button=getattr(self,'camera_notice',None);line=self.fontMetrics().lineSpacing()
        top=0
        if button is not None:
            button.setMaximumWidth(max(1,self.width()-line));button.adjustSize()
            button.move(line//2,line//2);button.raise_();top=button.height()+line
        top+=getattr(self,'ring_w',0)
        margins=self.layout().contentsMargins()
        if margins.top()!=top:
            self.layout().setContentsMargins(margins.left(),top,margins.right(),margins.bottom());self.layout().activate()
        self.overlay.setGeometry(self.viewport.geometry())
    def set_footer(self,height):
        ring=getattr(self,'ring_w',0);height=max(0,int(height))+ring
        if self.layout().contentsMargins().bottom()!=height:
            self.layout().setContentsMargins(ring,self.layout().contentsMargins().top(),ring,height);self.layout().activate()
        self.image.inset_bottom=0
        self.overlay.setGeometry(self.viewport.geometry());self.overlay.raise_()

    def _show_rate(self):
        rate = self.image.fps() if self.page == "live" else None
        if not rate: self.rate_label.hide(); return
        self.rate_label.setText("%.1f fps" % rate); self.rate_label.adjustSize()
        if getattr(self,'external_header',False):self.rate_label.show();return
        right = self.home_b.x() - 8 if self.home_b.isVisible() else self.width() - 12
        self.rate_label.move(right - self.rate_label.width(), 12); self.rate_label.show(); self.rate_label.raise_()

    def resizeEvent(self, e):
        self._place_camera_notice()
        self.overlay.setGeometry(self.viewport.geometry()); self.overlay.raise_()
        if getattr(self, 'frame', None) is not None: self.frame.setGeometry(self.rect()); self.frame.raise_()
        if not getattr(self,'external_header',False):
            self.home_b.adjustSize(); self.home_b.move(self.width() - self.home_b.width() - 12, 12); self.home_b.raise_()
        self._refresh_overlay()
        super().resizeEvent(e)

