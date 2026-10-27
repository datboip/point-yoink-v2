# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 2.0 shell and Scan step widgets. Only looks and layout live here: every widget takes plain values (numpy
pictures, numbers, strings, the session's snapshots) through set_* calls and reports what the user did through
signals, so nothing in here knows where the values come from. Every button is Qt.NoFocus: Space must always reach
Start / Pause, never "click" whatever was clicked last."""
import collections
from pathlib import Path
from app2 import REPO
from app2.version import APP_TITLE
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSize, QEvent, QTimer
from PySide6.QtGui import QPainter, QPainterPath, QPen, QImage, QColor, QFont, QFontMetrics, QPixmap, QPolygonF
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
                               QProgressBar, QPlainTextEdit, QSizePolicy, QLineEdit, QComboBox, QCheckBox, QTabBar, QStackedWidget, QToolButton)
from app2.qt.theme import T, qc, ui_font
from app2.scan.ui_model import KIND_COLOUR
from app2.scan.object_modes import label as object_label
from app2.scan.assist import RANGE_STEP_MM

STEPS = ["Scan", "Build", "Edit", "Review", "Measure", "Export"]


def to_qimage(a):
    """HxWx3 uint8 RGB -> a QImage that owns its pixels (the array may be reused after this)."""
    if a is None: return None
    if isinstance(a, QImage): return a
    a = np.ascontiguousarray(a[..., :3], np.uint8); h, w, _ = a.shape
    return QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def repolish(w):
    """After a style property changes at run time (picked, kind, level): let the style sheet see it."""
    w.style().unpolish(w); w.style().polish(w); w.update()


def button(text="", kind=None, tip=None, parent=None):
    """A push button that never takes keyboard focus."""
    b = QPushButton(text, parent); b.setFocusPolicy(Qt.FocusPolicy.NoFocus); b.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind: b.setProperty("kind", kind)
    if tip: b.setToolTip(tip)
    return b


def set_kind(b, kind):
    if b.property("kind") != kind: b.setProperty("kind", kind); repolish(b)


def chip(p, x, y, parts, px=12, bg=None, pad=(8, 4)):
    """A dark pill with text over a picture: parts = [(text, colour name or None), ...]; a colour draws a dot first.
    Returns the pill's rect."""
    f = ui_font(px, QFont.Weight.DemiBold); fm = QFontMetrics(f); p.setFont(f)
    dot = px * 0.62; gap = 10; w = pad[0] * 2; h = fm.height() + pad[1] * 2
    for i, (t, c) in enumerate(parts): w += fm.horizontalAdvance(t) + (dot + 5 if c else 0) + (gap if i else 0)
    r = QRectF(x, y, w, h)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(bg or QColor(10, 12, 15, 232)); p.drawRoundedRect(r, 6, 6)
    cx = x + pad[0]
    for i, (t, c) in enumerate(parts):
        if i: cx += gap
        if c:
            p.setBrush(qc(c)); p.drawEllipse(QRectF(cx, y + h / 2 - dot / 2, dot, dot)); cx += dot + 5
        p.setPen(qc("ink") if i == 0 else qc("ink2")); p.drawText(QPointF(cx, y + pad[1] + fm.ascent()), t)
        p.setPen(Qt.PenStyle.NoPen); cx += fm.horizontalAdvance(t)
    return r


def chip_width(parts, px=12, pad=(8, 4)):
    """How wide chip() draws these parts."""
    fm = QFontMetrics(ui_font(px, QFont.Weight.DemiBold)); dot = px * 0.62
    return pad[0] * 2 + sum(fm.horizontalAdvance(t) + (dot + 5 if c else 0) + (10 if i else 0) for i, (t, c) in enumerate(parts))


def fit_parts(parts, max_w, px=12):
    """A title chip's parts cut to max_w: the extra parts go from the end first, then the title ends in "..." (a
    label is never drawn over a button or off the view)."""
    parts = list(parts)
    while len(parts) > 1 and chip_width(parts, px) > max_w: parts.pop()
    if parts and chip_width(parts, px) > max_w:
        t, c = parts[0]; fm = QFontMetrics(ui_font(px, QFont.Weight.DemiBold))
        room = max(10, int(max_w - chip_width([("", c)], px)))
        parts[0] = (fm.elidedText(t, Qt.TextElideMode.ElideRight, room), c)
    return parts


RING_WIDTH = 3   # the state ring round a camera picture; its header sits inside it


def row_label(text, role="section"):
    l = QLabel(text); l.setProperty("role", role); return l


CAMERA_SETUP_STEPS=("<b>The scanner is calibrated at the factory.</b> The app only needs a copy of its camera files, once per camera.<br>"
    "1. On the MIRACO itself (not PC mode), scan anything with <b>Small part</b> and save it. Color on or off, either is fine.<br>"
    "2. Do the same scan with <b>Large part</b>.<br>"
    "3. Plug the USB in, pick <b>File Transfer</b> on the scanner, then here File, <b>Import scanner projects</b>, and import both.<br>"
    "Each import brings its camera's files. Or press Choose below and pick the <b>param</b> folder inside such a project.")

class ElidedLabel(QLabel):
    """One line of text that ends in "..." when it does not fit (a plain QLabel cuts it off at one end)."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent); self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self): return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, e):
        p = QPainter(self); fm = self.fontMetrics(); r = self.contentsRect()
        p.setPen(self.palette().color(self.foregroundRole()))
        shown = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, r.width())
        p.drawText(r, int(self.alignment()), shown)
        cut = shown != self.text()                            # cut short: the whole text on hover
        if cut != getattr(self, "_cut", None) and not getattr(self,"keep_tip",False): self._cut = cut; self.setToolTip(getattr(self,"detail_tooltip",self.text() if cut else ""))


class FitLabel(ElidedLabel):
    """One line that shows the longest of its texts that fits ("Side 10 · touch up", else "Side 10"); never narrower
    than the last (shortest) one."""
    def set_texts(self, *texts):
        self.texts = [t for t in texts if t]; self.setText(self.texts[0] if self.texts else ""); self.updateGeometry(); self.update()

    def minimumSizeHint(self):
        t = (getattr(self, "texts", None) or [self.text()])[-1]
        return QSize(self.fontMetrics().horizontalAdvance(t) + 2, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 2, super().sizeHint().height())

    def paintEvent(self, e):
        p = QPainter(self); fm = self.fontMetrics(); r = self.contentsRect()
        p.setPen(self.palette().color(self.foregroundRole()))
        texts = getattr(self, "texts", None) or [self.text()]
        shown = next((t for t in texts if fm.horizontalAdvance(t) <= r.width()), None)
        if shown is None: shown = fm.elidedText(texts[-1], Qt.TextElideMode.ElideRight, r.width())
        p.drawText(r, int(self.alignment()), shown)
        cut = shown != texts[0]
        if cut != getattr(self, "_cut", None): self._cut = cut; self.setToolTip(texts[0] if cut else "")


class StableLabel(QLabel):
    """A word-wrapped label whose height never follows its text, so nothing under it moves when a message comes, goes
    or changes: as tall as `lines` lines, or the tallest of `alts` (every text it may show) at its width when taller.
    Blank keeps the height. A text taller than that (an engine error) ends in "..." with the whole text on hover."""
    def __init__(self, text="", lines=1, alts=(), grow=False, parent=None):
        super().__init__(text, parent); self.lines = lines; self.alts = [a for a in alts if a]; self.grow = grow   # grow: text taller wins
        self.setWordWrap(True); self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        sp = self.sizePolicy(); sp.setHeightForWidth(True); sp.setVerticalPolicy(QSizePolicy.Policy.Fixed); self.setSizePolicy(sp)

    def set_alts(self, alts): self.alts = [a for a in alts if a]; self.updateGeometry()

    def _text_h(self, t, w):
        return self.fontMetrics().boundingRect(0, 0, max(1, w), 100000, int(Qt.TextFlag.TextWordWrap), t).height()

    def hasHeightForWidth(self): return True

    def heightForWidth(self, w):
        m = self.contentsMargins(); iw = w - m.left() - m.right()
        alts = self.alts + ([self.text()] if self.grow and self.text() else [])
        return max([self.fontMetrics().lineSpacing() * self.lines] + [self._text_h(t, iw) for t in alts]) + m.top() + m.bottom()

    def sizeHint(self):
        w = self.width() if self.width() > 20 else super().sizeHint().width()
        return QSize(0, self.heightForWidth(w))

    def minimumSizeHint(self): return QSize(0, self.sizeHint().height())

    def paintEvent(self, e):
        r = self.contentsRect(); t = self.text()
        fits = not t or self._text_h(t, r.width()) <= r.height()
        tip = "" if fits else t
        if self.toolTip() != tip and (tip or getattr(self, "_tip_ours", False)): self.setToolTip(tip); self._tip_ours = bool(tip)
        if fits: super().paintEvent(e); return
        words = t.split(" "); lo, hi = 0, len(words)          # the most words that fit with "..." after them
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._text_h(" ".join(words[:mid]) + "...", r.width()) <= r.height(): lo = mid
            else: hi = mid - 1
        p = QPainter(self); p.setFont(self.font()); p.setPen(self.palette().color(self.foregroundRole()))
        p.drawText(r, int(self.alignment()) | int(Qt.TextFlag.TextWordWrap), " ".join(words[:lo]).rstrip(" .,:;") + "...")


def keep_room(w, on=True):
    """A widget that hides keeps its room in the layout (nothing next to or under it moves)."""
    sp = w.sizePolicy(); sp.setRetainSizeWhenHidden(on); w.setSizePolicy(sp); return w


def fit_width(b, texts, pad=None):
    """A button as wide as its longest text (bold or not), so a text change never resizes it or moves its row."""
    f = QFont(b.font()); f.setBold(True); fm = QFontMetrics(f)
    b.setMinimumWidth(max(fm.horizontalAdvance(t) for t in texts) + (pad if pad is not None else 32)); return b


# ---------------------------------------------------------------- pictures
class ImageTile(QWidget):
    """A camera picture (or the model) in a dark rounded well, with its name on a pill, a note bottom right and an
    optional state ring. fit: "fit" (whole picture, letterboxed), "cover" (fills the well) or "native" (its own pixel
    size, centred, only ever scaled down: small thumbs blown up looked broken)."""
    def __init__(self, parts=(), parent=None, radius=10, fit="fit"):
        super().__init__(parent)
        self.parts = list(parts); self.img = None; self.ring = None; self.radius = radius; self.fit = fit; self.note = None
        self.camera_guides=None;self.guide_options={};self.guide_ir=False
        self.target_h = None; self.max_w = None            # max_w: the name pill's room (None: the tile's width)
        self.center = None                                  # a line in the middle (nothing to show yet, and why)
        self.note_lift = 0                                  # the note's gap above the bottom (buttons over the tile)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(80, 60)

    def set_target_height(self, h):
        """Camera tiles: the height their 4:3 picture wants. They get it when there is room and shrink (never below
        60 px) when there is not; a fixed height instead would raise the window's minimum size."""
        self.target_h = int(h); self.setMaximumHeight(max(60, self.target_h))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum); self.updateGeometry()

    def sizeHint(self):
        return QSize(4 * self.target_h // 3, self.target_h) if self.target_h else super().sizeHint()

    def set_image(self, a, fit=None, sample=True):
        self.img = to_qimage(a)
        if fit:self.fit=fit
        if a is not None:
            import time
            now=time.monotonic()
            if sample and a is not getattr(self,'_sample_image',None):
                stamps=self.__dict__.setdefault('_stamps',[])
                stamps.append(now)
                while len(stamps)>1 and stamps[1]<now-3:stamps.pop(0)
                self._sample_image=a
        else:
            self._stamps=[];self._sample_image=None;self.camera_guides=None
        self.update()

    def fps(self):
        import time
        stamps=getattr(self,'_stamps',[])
        if len(stamps)<2:return None
        if time.monotonic()-stamps[-1]>3:return 0.
        return (len(stamps)-1)/max(stamps[-1]-stamps[0],.001)

    def set_guides(self,data,options=None,ir=False):
        self.camera_guides=data;self.guide_options=dict(options or {});self.guide_ir=ir;self.update()

    def set_ring(self, colour):                                         # a theme colour name or None
        if colour != self.ring: self.ring = colour; self.update()

    def set_note(self, text):
        if text != self.note: self.note = text; self.update()

    inset_bottom = 0                                       # room kept free under the picture (the distance bar)
    def image_rect(self, r):
        iw, ih = self.img.width(), self.img.height(); r = QRectF(r).adjusted(0, getattr(getattr(self,"header",None),"height",lambda:0)(), 0, -self.inset_bottom)
        if self.fit == "cover": s = max(r.width() / iw, r.height() / ih)
        elif self.fit == "native": s = min(1.0, r.width() / iw, r.height() / ih)
        else: s = min(r.width() / iw, r.height() / ih)
        w, h = iw * s, ih * s
        return QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self,'header'):self.header.adapt(self.width())

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath(); path.addRoundedRect(r, self.radius, self.radius)
        p.fillPath(path, qc("well"))
        if self.img is not None:
            p.save(); p.setClipPath(path); image_rect=self.image_rect(r);p.drawImage(image_rect,self.img)
            from app2.qt.camera_guides import paint
            paint(p,image_rect,self.camera_guides,self.guide_options,self.fontMetrics().lineSpacing(),ir=self.guide_ir)
            p.restore()
        self.overlay(p, r)
        from app2.scan import timing; timing.mark('paint_'+(self.parts[0][0] if self.parts else 'tile'))
        if self.ring:
            p.setPen(QPen(qc(self.ring), RING_WIDTH)); p.setBrush(Qt.BrushStyle.NoBrush); half = RING_WIDTH / 2
            p.drawRoundedRect(r.adjusted(half, half, -half, -half), self.radius - 1, self.radius - 1)
        else:
            p.setPen(QPen(qc("line"), 1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(path)

    def set_center(self, text):
        if text != self.center: self.center = text; self.update()

    def overlay(self, p, r):
        if self.center:
            p.setFont(ui_font(14)); p.setPen(qc("ink3"))
            p.drawText(r.adjusted(24, 40, -24, -40), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.center)
        if hasattr(self,"header"):
            self.header.refresh()
            return
        if not self.parts: return
        s = 10 if self.width() > 300 else 7
        px = 12 if self.width() > 300 else 11
        chip(p, r.x() + s, r.y() + s, fit_parts(self.parts, self.max_w or r.width() - 2 * s, px), px)
        rate = self.fps()
        if rate is not None:                                # how often this picture really updates
            t = "%.1f fps" % rate; f = ui_font(px); p.setFont(f); w = QFontMetrics(f).horizontalAdvance(t) + 16
            chip(p, r.right() - s - w, r.y() + s, [(t, None)], px)
        if self.note:
            f = ui_font(11); p.setFont(f); fm = QFontMetrics(f); w = fm.horizontalAdvance(self.note) + 16
            chip(p, r.right() - s - w, r.bottom() - s - fm.height() - 8 - self.note_lift, [(self.note, None)], 11)


# ---------------------------------------------------------------- controls
class Track(QWidget):
    """A slider track with one handle (gain) or two (Near / Far range). The kept part of the range is filled green,
    the same green as the kept depth on the IR view. Drag a handle (the nearest one to the press), or with the track
    focused (click it): Left / Right one step, Page Up / Page Down a big step, Home / End the ends; Tab moves from the
    near handle to the far one. changed(values) while it moves, released(values) at the end of a drag."""
    changed = Signal(object)
    released = Signal(object)
    KEYS = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Home, Qt.Key.Key_End)

    def __init__(self, lo, hi, values, step=1, page=None, fill="kept", marks=(), name="", unit="", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.values, self.step, self.page = lo, hi, [int(v) for v in values], step, page or step * 4
        self.fill, self.marks, self.unit = fill, marks, unit
        self.active = 0; self.dragging = False; self.enabled_look = True;self.guide=None
        self.compact = False; self.setMinimumHeight(self._grip()[1] + (4 if self.compact else 8))   # the grip and its focus ring
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)                            # the handle under the pointer lights up
        # focus by a click only: a track that took the window's first focus showed a focus ring and moved on arrow keys
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(name); self._describe()

    def _grip(self):
        """Handle width, handle height and groove thickness: chunky enough to grab without looking; smaller in compact
        windows, where the camera pictures need the height."""
        line = self.fontMetrics().lineSpacing()
        if self.compact: return max(10, line * 2 // 3), max(18, line + line // 3), max(6, line // 3)
        return max(16, line), max(26, line * 8 // 5), max(10, line * 3 // 5)

    def set_compact(self, compact):
        self.compact = compact; self.setMinimumHeight(self._grip()[1] + (4 if self.compact else 8)); self.updateGeometry(); self.update()

    def _describe(self):
        u = " " + self.unit if self.unit else ""
        self.setAccessibleDescription(("%d to %d%s" % tuple(self.values + [u])) if len(self.values) == 2 else "%d%s" % (self.values[0], u))

    PAD = 13                                               # room at each end for the handle and its focus ring (11 + pen)
    def x_of(self, v): return self.PAD + (self.width() - 2 * self.PAD) * (v - self.lo) / (self.hi - self.lo)

    def v_of(self, x):
        v = self.lo + (x - self.PAD) / max(1.0, self.width() - 2 * self.PAD) * (self.hi - self.lo)
        return int(round(v / self.step) * self.step)

    def set_values(self, values):
        """From the session: ignored while the user is moving a handle (the command is on its way)."""
        if self.dragging: return
        values = [int(v) for v in values]
        if values != self.values: self.values = values; self._describe(); self.update()

    def _move(self, k, v):
        v = int(min(self.hi, max(self.lo, v)))
        if len(self.values) == 2:                          # near stays below far
            v = min(v, self.values[1] - self.step) if k == 0 else max(v, self.values[0] + self.step)
        if v == self.values[k]: return
        self.values[k] = v; self._describe(); self.update(); self.changed.emit(list(self.values))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton: return
        x = e.position().x()
        self.active = int(np.argmin([abs(self.x_of(v) - x) for v in self.values]))
        self.dragging = True; self._move(self.active, self.v_of(x))

    def mouseMoveEvent(self, e):
        if self.dragging: self._move(self.active, self.v_of(e.position().x())); return
        x = e.position().x(); near = min(range(len(self.values)), key=lambda k: abs(self.x_of(self.values[k]) - x))
        hover = near if abs(self.x_of(self.values[near]) - x) <= self.fontMetrics().lineSpacing() else None
        if hover != getattr(self, "hover", None): self.hover = hover; self.update()

    def leaveEvent(self, e): self.hover = None; self.update(); super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.dragging: self.dragging = False; self.released.emit(list(self.values))

    def event(self, e):
        if e.type() == QEvent.Type.ShortcutOverride and e.key() in self.KEYS:
            e.accept(); return True                        # these keys move the handle, not a window shortcut
        return super().event(e)

    def keyPressEvent(self, e):
        k = e.key(); v = self.values[self.active]
        step = {Qt.Key.Key_Left: -self.step, Qt.Key.Key_Right: self.step, Qt.Key.Key_PageDown: -self.page, Qt.Key.Key_PageUp: self.page}.get(k)
        if step is not None: self._move(self.active, v + step); self.released.emit(list(self.values)); return
        if k == Qt.Key.Key_Home: self._move(self.active, self.lo); self.released.emit(list(self.values)); return
        if k == Qt.Key.Key_End: self._move(self.active, self.hi); self.released.emit(list(self.values)); return
        super().keyPressEvent(e)

    def focusNextPrevChild(self, forward):
        if len(self.values) == 2 and forward and self.active == 0: self.active = 1; self.update(); return True
        if len(self.values) == 2 and not forward and self.active == 1: self.active = 0; self.update(); return True
        return super().focusNextPrevChild(forward)

    def focusInEvent(self, e): self.update(); super().focusInEvent(e)
    def focusOutEvent(self, e): self.update(); super().focusOutEvent(e)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        guide=getattr(self,'guide',None)
        if guide:
            colors={'in range':'accent','too close':'warn','too far':'warn','no center depth':'ink3'}
            color=qc(colors.get(guide['state'],'ink3'));color.setAlpha(100)
            lo,hi=guide['working_range']
            p.setBrush(qc('panel'));p.drawRect(QRectF(self.x_of(lo),0,self.x_of(hi)-self.x_of(lo),self.height()))
            counts=guide['counts'];edges=guide['edges'];peak=max(counts,default=0)
            p.setBrush(color)
            if peak:
                for left,right,count in zip(edges,edges[1:],counts):
                    x=max(self.PAD,self.x_of(left));end=min(self.width()-self.PAD,self.x_of(right))
                    height=(self.height()-4)*count/peak
                    if end>x:p.drawRect(QRectF(x,self.height()-height,end-x,height))
        line = self.fontMetrics().lineSpacing(); w, h, groove = self._grip(); r = groove / 2
        # a scale under the groove: a tick each page step, so a handle's place reads at a glance
        span = self.hi - self.lo; every = self.page
        while every and (self.width() - 2 * self.PAD) * every / span < line: every *= 2   # no closer than a line apart
        ticks = int(span // every) if every else 0
        if ticks > 0:
            p.setBrush(qc("ink3"))
            for i in range(ticks + 1):
                x = self.x_of(self.lo + i * every); big = i % 2 == 0
                p.drawRect(QRectF(x - 1, y + r + 3, 2, line // 2 if big else line // 3))
        p.setBrush(qc("well")); p.drawRoundedRect(QRectF(self.PAD, y - r, self.width() - 2 * self.PAD, groove), r, r)   # over the guide
        for m in self.marks:                                 # e.g. where Auto gain stops
            p.setBrush(qc("ink3")); p.drawRect(QRectF(self.x_of(m) - 1, y - r - 3, 2, groove + 6))
        a = self.x_of(self.values[0]) if len(self.values) == 2 else self.PAD
        b = self.x_of(self.values[-1])
        fill = qc(self.fill) if (self.enabled_look and self.isEnabled()) else qc("ink3")
        p.setBrush(fill); p.drawRoundedRect(QRectF(a, y - r, b - a, groove), r, r)
        # handles: upright grips in the fill color, so they read as something to drag rather than two dots
        for k, v in enumerate(self.values):
            box = QRectF(self.x_of(v) - w / 2, y - h / 2, w, h)
            if self.hasFocus() and k == self.active:
                p.setPen(QPen(qc("accent"), 2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRoundedRect(box.adjusted(-3, -3, 3, 3), w / 2 + 3, w / 2 + 3)
            hot = k == getattr(self, "hover", None) or (self.dragging and k == self.active)
            p.setBrush(qc("ink") if hot else fill.lighter(125)); p.setPen(QPen(qc("window"), 2)); p.drawRoundedRect(box, w / 2, w / 2)
            p.setPen(QPen(qc("window"), 1))
            for dx in (-2, 2): p.drawLine(QPointF(box.center().x() + dx, y - h / 5), QPointF(box.center().x() + dx, y + h / 5))


class DistanceBar(QWidget):
    defaults_asked=Signal()
    """A separate footer: recorded support, current part depth, Range, and distance."""
    def __init__(self,parent=None):
        super().__init__(parent);self.guide=None;self.cut=None;self.limits=(0,1)
        self.defaults=QPushButton('Use default bands',self);self.defaults.clicked.connect(self.defaults_asked);self.defaults.hide()
        self.setToolTip('Distance bands come from your Range test, or the shipped saved-scan profile. Green is best, yellow is OK, red is outside supported distances. Faded bars are outside Range. These are capture guides, not a precision rating.')
        self.setMinimumHeight(self.sizeHint().height())
    def sizeHint(self):
        fm=QFontMetrics(ui_font());return QSize(fm.horizontalAdvance('Part distance and Range'),fm.lineSpacing()*7)
    def set_guide(self,guide,cut,limits):
        if guide and not guide.get('working_range'):guide=None   # no placeholder axis before the mode's span is known
        self.guide,self.cut,self.limits=guide,cut,limits;self.setVisible(bool(guide))
        if guide:   # the advice holds; the millimeters stay live
            from app2.scan.range_guide import distance_action
            from app2.qt.steady import Steady
            if not hasattr(self,'steady'):self.steady=Steady(lambda a:(setattr(self,'action',a),self.update()),self)
            action=distance_action(guide);self.steady.offer(action,action)
        self.defaults.setVisible(bool((guide or {}).get('bands_date')));self.defaults.adjustSize();line=self.fontMetrics().lineSpacing();self.defaults.move(self.width()-self.defaults.width()-line//2,0);self.update()
    def view_span(self):
        """Keep the live mode axis steady while the part and Range move."""
        span=(self.guide or {}).get('working_range')
        if not span:return self.limits
        lo,hi=span;margin=(hi-lo)/2
        return max(self.limits[0],lo-margin),min(self.limits[1],hi+margin)
    def histogram_slots(self,n):
        """Split source counts by overlap, preserving counts at any display width."""
        g=self.guide or {};counts=np.asarray(g.get('counts',[]),float);edges=np.asarray(g.get('edges',[]),float)
        if not len(counts):return np.zeros(n)
        lo,hi=self.view_span();cumulative=np.r_[0,np.cumsum(counts)]
        return np.diff(np.interp(np.linspace(lo,hi,n+1),edges,cumulative))
    def bounds(self):
        """Where too close ends and too far starts: the outer edges of the bands, else the working range. No dead gap."""
        g=self.guide or {};zones=g.get('zones') or [];span=g.get('working_range')
        if zones:return min(z['lo'] for z in zones),max(z['hi'] for z in zones)
        return span
    def display_zones(self):
        g=self.guide or {};zones=[dict(x) for x in g.get('zones',[])];span=self.bounds() if g.get('working_range') else None
        if span:
            lo,hi=self.view_span();zones=[dict(lo=lo,hi=span[0],state='too close')]+zones+[dict(lo=span[1],hi=hi,state='too far')]
        return [x for x in zones if x['hi']>x['lo']]
    def zone_at(self,value):
        span=self.bounds() if (self.guide or {}).get('working_range') else None
        if span and value<span[0]:return 'too close'
        if span and value>span[1]:return 'too far'
        from app2.scan.range_guide import BAND_LEVELS
        zones=[z for z in self.display_zones() if z['lo']<=value<=z['hi']]
        return max(zones,key=lambda z:BAND_LEVELS.get(z['state'],0))['state'] if zones else 'unmeasured'
    def text_layout(self):
        """Nonoverlapping labels after applying live-band priority, sized by the font."""
        from app2.scan.range_guide import LABELS
        g=self.guide or {};fm=QFontMetrics(self.font());line=fm.lineSpacing();pad=fm.horizontalAdvance('0000 mm')
        left,right=pad,max(pad,self.width()-pad);lo,hi=self.view_span()
        x=lambda value:left+(right-left)*np.clip((value-lo)/max(np.finfo(float).eps,hi-lo),0,1)
        edges=sorted(set([lo,hi]+[float(z[k]) for z in self.display_zones() for k in ('lo','hi') if lo<float(z[k])<hi]))
        bands=[]
        for a,b in zip(edges,edges[1:]):
            state=self.zone_at((a+b)/2)
            if bands and bands[-1][2]==state:bands[-1]=(bands[-1][0],b,state)
            else:bands.append((a,b,state))
        labels=[]
        for a,b,state in bands:
            if state=='unmeasured':continue
            text={'strong depth':'Best','usable depth':'OK'}.get(state,LABELS.get(state,state));width=fm.horizontalAdvance(text)
            if x(b)-x(a)<width+line/2:continue
            rect=QRectF((x(a)+x(b)-width)/2,0,width,line)
            labels.append((text,state,rect))
        bold=QFont(self.font());bold.setBold(True);bfm=QFontMetrics(bold)
        from app2.scan.range_guide import distance_action
        center=g.get('center_mm');distance=('No part depth yet' if center is None else 'Part %.0f mm: %s' % (center,getattr(self,'action',None) or distance_action(g)))
        self.setToolTip(distance+'\n'+getattr(self,'frame_hint','')+'\nGray bars are outside Range. Distance bands use a saved Range test when available, otherwise the shipped profile.')
        cut='Range %g to %g mm (gray is left out)' % tuple(self.cut) if self.cut else ''
        text_left=line/2;text_right=self.width()-line/2
        available=max(0,text_right-text_left);cut_width=min(fm.horizontalAdvance(cut)+line/2,max(0,(available-line)/2)) if cut else 0
        distance_width=max(0,available-cut_width-(line if cut else 0))
        y=self.height()-line;distance_rect=QRectF(text_left,y,distance_width,line);cut_rect=QRectF(text_right-cut_width,y,cut_width,line)
        status_rect=QRectF(text_left,y-line,available,line)
        return dict(labels=labels,distance=(bfm.elidedText(distance,Qt.TextElideMode.ElideRight,int(distance_width)),distance_rect),
                    cut=(fm.elidedText(cut,Qt.TextElideMode.ElideRight,int(cut_width)),cut_rect),status=(fm.elidedText(getattr(self,'frame_hint',''),Qt.TextElideMode.ElideRight,int(available)),status_rect),left=left,right=right,line=line,x=x)
    def paintEvent(self,event):
        g=self.guide
        if not g:return
        from app2.scan.range_guide import COLORS,BAND_RGB
        band_colors=dict(COLORS);band_colors.update({'strong depth':'#%02x%02x%02x'%BAND_RGB['best'],'usable depth':'#%02x%02x%02x'%BAND_RGB['ok'],'sparse depth':'#%02x%02x%02x'%BAND_RGB['off'],'too close':'#%02x%02x%02x'%BAND_RGB['off'],'too far':'#%02x%02x%02x'%BAND_RGB['off']})
        layout=self.text_layout();line=layout['line'];left,right=layout['left'],layout['right'];x=layout['x']
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);fm=QFontMetrics(self.font())
        p.setPen(Qt.PenStyle.NoPen);p.setBrush(qc('well'));p.drawRoundedRect(QRectF(self.rect()),line/3,line/3)
        strip=line/3;top=line;bottom=self.height()-4*line-strip
        for zone in self.display_zones():
            a,b=x(zone['lo']),x(zone['hi'])
            if b<=a:continue
            c=QColor(band_colors.get(zone['state'],COLORS['unmeasured']));tint=QColor(c);tint.setAlpha(45 if zone.get('live') else 30)
            p.fillRect(QRectF(a,top,b-a,bottom-top),tint);p.fillRect(QRectF(a,bottom,b-a,strip),c)
        slot=max(2,line//6);n=max(1,int((right-left)//slot))
        e=np.asarray(g.get('edges',[0,1]),float);lo_,hi_=self.view_span()
        if len(e)>1:n=max(1,min(n,int((hi_-lo_)/max(np.diff(e).min(),np.finfo(float).eps))))   # never a column narrower than a source bin (that combs)
        counts=self.histogram_slots(n)
        levels=counts/max(max(counts,default=0),np.finfo(float).eps)
        lo,hi=self.view_span();slot_width=(right-left)/n
        for i,value in enumerate(levels):
            if value<=0:continue
            mm=lo+(i+.5)*(hi-lo)/n
            color=QColor(band_colors.get(self.zone_at(mm),COLORS['unmeasured']))
            if self.cut and not self.cut[0]<=mm<=self.cut[1]:color=QColor(*__import__('app2.scan.preview',fromlist=['REMOVED_RGB']).REMOVED_RGB)   # left out by the Range: gray, same as the dots
            height=(bottom-top)*value
            p.fillRect(QRectF(left+i*slot_width,bottom-height,max(1,slot_width-1),height),color)
        p.setPen(qc('ink'))
        for text,state,rect in layout['labels']:p.drawText(rect,Qt.AlignmentFlag.AlignCenter,text)
        center=g.get('center_mm')
        if center is not None:
            cx=x(center);p.setPen(QPen(qc('ink'),2));p.drawLine(QPointF(cx,top+strip),QPointF(cx,bottom+strip))
            p.setPen(Qt.PenStyle.NoPen);p.setBrush(qc('ink'))
            p.drawPolygon(QPolygonF([QPointF(cx-strip,top),QPointF(cx+strip,top),QPointF(cx,top+strip)]))
        # Tick density follows the font width; the values follow the Range step.
        from app2.scan.assist import RANGE_STEP_MM
        lo,hi=self.view_span();label_width=fm.horizontalAdvance('%g mm' % hi)+line
        count=max(1,int((right-left)/label_width));raw=(hi-lo)/count
        magnitude=10**np.floor(np.log10(max(raw,RANGE_STEP_MM)))
        step=max(RANGE_STEP_MM,next((k*magnitude for k in (1,2,5,10) if k*magnitude>=raw),10*magnitude))
        if g.get('bands_date'):
            p.setPen(qc('ink2'));p.drawText(QPointF(line/2,line),'Bands from your Range test, '+g['bands_date'][:10])
        p.setPen(qc('ink3'));pad=fm.horizontalAdvance('%g mm' % hi)
        for value in np.arange(np.ceil(lo/step)*step,hi+np.finfo(float).eps,step):
            tx=x(value);p.drawLine(QPointF(tx,bottom+strip),QPointF(tx,bottom+strip+line/4))
            text='%g mm' % value if value+step>hi else '%g' % value
            p.drawText(QRectF(tx-pad/2,bottom+strip+line/4,pad,line),Qt.AlignmentFlag.AlignCenter,text)
        bold=QFont(self.font());bold.setBold(True);p.setFont(bold);p.setPen(qc('ink'))
        text,rect=layout['distance'];p.drawText(rect,Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,text)
        p.setFont(self.font());p.setPen(qc('ink2'))
        text,rect=layout['cut'];p.drawText(rect,Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,text)
        text,rect=layout['status']
        if text:
            from app2.qt.status_chip import paint_status
            paint_status(p,rect,fm,'warn',getattr(self,'frame_hint',''))



class BigButton(QPushButton):
    """The primary button: Start / Continue / Pause, with its key on it. kind: primary (green) or pause (amber)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setMinimumHeight(52); self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay = QHBoxLayout(self); lay.setContentsMargins(20, 0, 14, 0)
        self.title = ElidedLabel(); self.title.setFont(ui_font(19, QFont.Weight.Bold))       # a long build step ends in ...
        self.key = QLabel("Space"); self.key.setProperty("role", "keycap"); self.key.setFont(ui_font(11, QFont.Weight.DemiBold))
        for w in (self.title, self.key): w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(self.title, 1); lay.addWidget(self.key, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setProperty("kind", None); self.set_action("", "primary", False)

    def text_now(self): return self.title.text()

    def set_action(self, text, kind, enabled=True):
        changed = self.property("kind") != kind or self.isEnabled() != bool(enabled)
        if self.title.text() != text: self.title.setText(text); self.setAccessibleName(text)
        self.setEnabled(enabled); self.key.setVisible(enabled)
        if changed:
            self.setProperty("kind", kind); repolish(self)
            # the labels get their colour here: a style sheet rule on "a disabled button's label" also matched the
            # labels of enabled buttons (Qt ignores the ancestor's state there)
            ink = T["on_color"] if enabled else T["ink2"]
            self.title.setStyleSheet("color: %s;" % ink); self.key.setStyleSheet("color: %s;" % ink)


class ControlRow(QWidget):
    """A name, its value, its Auto chip and its track under them."""
    def __init__(self, name, track, chip_tip, parent=None):
        super().__init__(parent)
        self.name = row_label(name); self.value = row_label("", "value"); self.track = track
        self.auto = button("Auto", "chip", chip_tip); self.auto.setCheckable(True)
        self.g = QGridLayout(self); self.g.setContentsMargins(0, 0, 0, 0); self.g.setHorizontalSpacing(10)
        from app2.qt.help_text import HelpButton
        self.help = HelpButton(name)
        name_row = QHBoxLayout(); name_row.setContentsMargins(0, 0, 0, 0); name_row.setSpacing(5)
        name_row.addWidget(self.name); name_row.addWidget(self.help); name_row.addStretch(1)
        self.g.addLayout(name_row, 0, 0); self.g.addWidget(self.value, 0, 1, Qt.AlignmentFlag.AlignRight); self.g.addWidget(self.auto, 0, 2)
        self.g.addWidget(self.track, 1, 0, 1, 3); self.g.setColumnStretch(0, 1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_compact(self, compact): self.g.setVerticalSpacing(0 if compact else 2)


BELOW_BED_TIP = ("Cut below bed: leaves out the floor and the turntable stand under the bed. They don't turn with the "
                 "part and can confuse tracking. The bed top stays. Leave this on.")
REMOVE_BASE_TIP = ("Remove base: hides the bed in the 3D view, the way the finished model will look. "
                   "The build always removes the bed either way.")
KEEP_BASE_TIP = ("The 3D view shows the bed. The build still removes it from the finished model "
                 "(turn Remove base on to hide it here too).")


class Controls(QFrame):
    """Range, gain, their Auto chips (each right beside the value it moves), Fill LED, Cut below bed, Remove base and
    what Auto gain is doing."""
    range_changed = Signal(int, int)
    gain_changed = Signal(int)
    auto_range_toggled = Signal(bool)
    auto_gain_toggled = Signal(bool)
    save_color_toggled = Signal(bool)
    led_toggled = Signal(bool)
    below_bed_toggled = Signal(bool)
    remove_base_toggled = Signal(bool)
    outside_below_changed = Signal(int)
    bed_lift_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("ScanControls");self.setProperty("role", "panel")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        v = QVBoxLayout(self); v.setContentsMargins(14, 10, 14, 12); v.setSpacing(8)
        from app2.qt.help_text import HelpButton
        head_row = QHBoxLayout(); self.capture_label = QLabel("Capture"); head_row.addWidget(self.capture_label)
        self.capture_help = HelpButton("Capture"); head_row.addWidget(self.capture_help); head_row.addStretch(1); v.addLayout(head_row)
        mode_row = QHBoxLayout()
        from app2.qt.choice_group import ChoiceGroup
        self.capture_mode = ChoiceGroup(3); self.capture_mode.addItem("Continuous", "continuous"); self.capture_mode.addItem("Single shot", "single"); self.capture_mode.addItem("Burst", "burst"); self.capture_mode.addItem("Smart frames", "smart");self.capture_mode.addItem("Live build", "live")
        mode_row.addWidget(self.capture_mode, 1); v.addLayout(mode_row)
        distance_row = QGridLayout();distance_row.setContentsMargins(0,0,0,0)
        for col,(name,tip) in enumerate((("Object","How big is the part? Small part: hold the scanner close for fine detail. Large part: hold it farther back so a bigger part fits. The app checks that the camera matches your pick."),("Surface","What does the part look like? Normal: most parts. Dark: black or very dark parts. Shiny: glossy or metal parts that reflect light. Sprayed: parts coated with scanning spray. This picks a good starting brightness. The line below says how well the camera sees the part."))):
            cell=QHBoxLayout();cell.setSpacing(4);cell.addWidget(QLabel(name));cell.addWidget(HelpButton(name,tip));cell.addStretch(1);distance_row.addLayout(cell,col*2,0,1,2)
        self.scan_mode = ChoiceGroup(2); self.scan_mode.setObjectName("ScanDistanceMode")
        self.scan_mode.addItem("Small part", "near"); self.scan_mode.addItem("Large part", "far")
        self.scan_mode.setToolTip("Small part: scanner close, finest detail. Large part: scanner farther back, bigger parts fit.")
        self.scan_mode.setItemTip(0,"Small part: hold the scanner close for the finest detail.");self.scan_mode.setItemTip(1,"Large part: hold the scanner farther back so a bigger part fits.")
        self.surface=ChoiceGroup(2);self.surface.addItems(("Normal","Dark","Shiny","Sprayed"));[self.surface.setItemTip(i,t) for i,t in enumerate(("Most parts.","Black or very dark parts.","Glossy or metal parts that reflect light.","Parts coated with scanning spray."))];distance_row.addWidget(self.scan_mode,1,0,1,2);distance_row.addWidget(self.surface,3,0,1,2);distance_row.setColumnStretch(0,1);distance_row.setColumnStretch(1,1);v.addLayout(distance_row)
        from app2.qt.status_chip import LiveStatus
        self.surface_note=LiveStatus(lines=2);v.addWidget(self.surface_note)
        from app2.qt.live_build_controls import LiveBuildControls
        self.live_build=LiveBuildControls();v.addWidget(self.live_build)
        self.calibration_info=ElidedLabel("Calibration source not saved")   # one steady line; the full text on hover
        self.calibration_info.setProperty("role","hint");v.addWidget(self.calibration_info)
        self.calibration_setup=button('Camera setup', 'plain', 'Once per camera: the app keeps a copy of the scanner\'s own camera files. Opens the steps.')
        self.calibration_setup.setCheckable(True);v.addWidget(self.calibration_setup)
        from PySide6.QtWidgets import QDialog
        self.calibration_tools=QDialog(self);self.calibration_tools.setWindowTitle('Camera setup');cv=QVBoxLayout(self.calibration_tools)
        self.setup_steps=QLabel(CAMERA_SETUP_STEPS);self.setup_steps.setWordWrap(True);self.setup_steps.setStyleSheet('color:#e7edf5;');self.setup_steps.setTextFormat(Qt.TextFormat.RichText);cv.addWidget(self.setup_steps)
        self.setup_status=QLabel('');self.setup_status.setWordWrap(True);self.setup_status.setStyleSheet('color:#e7edf5;');self.setup_status.setTextFormat(Qt.TextFormat.RichText);cv.addWidget(self.setup_status)
        self.choose_far=button("Choose Large part calibration...");cv.addWidget(self.choose_far)
        self.rebuild_far=button("Rebuild Large part copy");cv.addWidget(self.rebuild_far)
        self.far_copy_note=QLabel("");self.far_copy_note.setWordWrap(True);cv.addWidget(self.far_copy_note)
        self.calibration_tools.hide()
        def setup_visible(on):
            self.calibration_tools.setVisible(on)
        self.calibration_setup.toggled.connect(setup_visible)
        self.framing_row=QWidget();actions=QHBoxLayout(self.framing_row);actions.setContentsMargins(0,0,0,0);actions.setSpacing(v.spacing()//2)
        self.frame_range=QToolButton();self.frame_range.setText('Use part Range');self.frame_range.setToolTip('Set Range to fit the part seen so far, so the bed and background are left out.')
        self.range_undo=QToolButton();self.range_undo.setText('Undo');self.range_undo.setToolTip('Put back the Range you had before.')
        for tool in (self.frame_range,self.range_undo):
            tool.setFocusPolicy(Qt.FocusPolicy.NoFocus);tool.setFont(ui_font());actions.addWidget(tool)
        def widest(tool,*texts):                                  # one fixed width for every label, so a press never resizes the row
            old=tool.text();sizes=[]
            for t in texts:tool.setText(t);sizes.append(tool.sizeHint())
            tool.setText(old);tool.setFixedSize(max(z.width() for z in sizes),max(z.height() for z in sizes))
        widest(self.frame_range,'Use part Range');widest(self.range_undo,'Undo')
        self.frame_help=HelpButton('Part Range','Sets Range to fit the part seen so far on this side, so the bed and what is behind it are left out. Undo puts back the Range you had before.');actions.addWidget(self.frame_help)
        actions.addStretch(1)
        v.addWidget(self.framing_row)                   # Undo always shown (disabled until there is one), so the row never reflows
        self.frame_note=LiveStatus(lines=2);self.frame_note.setProperty('role','hint');v.addWidget(self.frame_note)
        self.scene_note=QLabel('');self.scene_note.hide()
        self.range = Track(50, 800, (100, 450), step=RANGE_STEP_MM, page=25, name="Range: front and back cut", unit="mm")
        self.range_default_limits = (self.range.lo, self.range.hi)
        self.distance_status=QLabel("Center depth not available");self.distance_status.setProperty("role","hint");self.distance_status.hide()
        self.gain = Track(1, 15, (4,), step=1, page=3, fill="accent", marks=(12,), name="IR gain")
        self.range_row = ControlRow("Range", self.range, "Auto: Range follows the part while you scan (R)")
        self.gain_row = ControlRow("IR gain", self.gain, "Auto: picks the camera brightness for you each turn (G)")
        from app2.scan.tracking import LIVE_MAX_DEPTH_MM
        widest_range='%d to %d mm' % (LIVE_MAX_DEPTH_MM,LIVE_MAX_DEPTH_MM)
        self.range_row.value.setFixedWidth(self.range_row.value.fontMetrics().horizontalAdvance(widest_range))
        self.auto_range, self.auto_gain = self.range_row.auto, self.gain_row.auto
        v.addWidget(self.range_row); v.addWidget(self.gain_row)
        bottom = QHBoxLayout(); bottom.setSpacing(8)
        self.led = button("Fill LED", "lamp", "Turn the scanner's white light on or off. Helps on dark parts and in dim rooms (L)"); self.led.setCheckable(True)
        sizes=[]                                                # one width for every light label, so a click never moves the row
        for t in ("LED",):self.led.setText(t);sizes.append(self.led.sizeHint().width())
        self.led.setText("LED");self.led.setFixedSize(max(sizes),self.led.sizeHint().height())   # fixed both ways: on, off and sent styles never move the rows
        self.below_bed = button("Cut below bed", "choice", BELOW_BED_TIP); self.below_bed.setCheckable(True); self.below_bed.setChecked(True)
        self.remove_base = button("Remove base", "choice", REMOVE_BASE_TIP); self.remove_base.setCheckable(True); self.remove_base.setChecked(True)
        # what Auto gain is doing: a line of its own under the buttons, full width, wrapping, hidden when empty (beside
        # the buttons it was squeezed to "Auto g...", seen live)
        self.auto_msg = QLabel(); self.auto_msg.setProperty("role", "hint"); self.auto_msg.setWordWrap(True); self.auto_msg.hide()
        self.led_help = HelpButton("Fill LED");self.led_help.hide()
        bottom.addWidget(self.led); bottom.addWidget(self.led_help); bottom.addWidget(self.below_bed); bottom.addWidget(self.remove_base); bottom.addStretch(1)
        v.addLayout(bottom)
        self.light_during_scans=QCheckBox('Light during scans');self.light_during_scans.setAccessibleName('Light during scans');self.light_during_scans.setToolTip('Turns the white light on when you press Start and off when you Pause. If you turn it on yourself, it stays on.');bottom.insertWidget(1,self.light_during_scans)
        self.save_color=QCheckBox("Save color");self.save_color.setChecked(True)
        self.save_color.setToolTip("Saves a color photo with each frame so the model can have real colors. About 770 MB per 1,000 frames. Turn off to save disk space.")
        self.save_color.toggled.connect(self.save_color_toggled)
        bottom.removeWidget(self.light_during_scans)                # the two checkboxes share one row
        checks=QHBoxLayout();checks.addWidget(self.light_during_scans,1);checks.addWidget(self.save_color,1);v.addLayout(checks)
        from PySide6.QtWidgets import QSpinBox
        outside_row=QHBoxLayout();self.outside_label=QLabel('Cut beside bed');outside_row.addWidget(self.outside_label)
        self.outside_below=QSpinBox();self.outside_below.setRange(0,LIVE_MAX_DEPTH_MM);self.outside_below.setSuffix(' mm');self.outside_below.setSpecialValueText('Off')
        self.outside_below.setAccessibleName('Cut depth beside bed');self.outside_below.setToolTip('Cuts the table or floor beside the bed. Off keeps parts that hang over the bed edge. Try 80 mm if the table shows up in your scan.')
        self.outside_below.valueChanged.connect(self.outside_below_changed);outside_row.addWidget(self.outside_below)
        self.outside_help=HelpButton('Cut beside bed','Off: keeps everything beside the bed, including parts that hang over the edge. A number: cuts anything that far below the bed, beside its edge, like the table. Try 80 mm if the table shows up. Needs a learned bed. What is cut while scanning cannot be brought back.');outside_row.addWidget(self.outside_help)
        v.addLayout(outside_row)
        lift_row=QHBoxLayout();self.lift_label=QLabel('Bed height');lift_row.addWidget(self.lift_label)
        self.bed_lift=QSpinBox();self.bed_lift.setRange(-50,50);self.bed_lift.setSuffix(' mm');self.bed_lift.setSpecialValueText('');self.bed_lift.setPrefix('')
        self.bed_lift.setAccessibleName('Bed height');self.bed_lift.setToolTip('Moves the Cut below bed plane up (+) or down (-) while you scan. Raise it if the table shows in the model; lower it if the purple cut sits on the part.')
        self.bed_lift.valueChanged.connect(self.bed_lift_changed);lift_row.addWidget(self.bed_lift)
        self.lift_help=HelpButton('Bed height','The cut plane comes from the table dots or Auto remove base. When it lands above the part (purple over the part, nothing building), lower it a few mm; when the table shows in the model, raise it. Works while scanning. Goes back to 0 when the app starts.',self);lift_row.addWidget(self.lift_help)
        v.addLayout(lift_row)
        # Cut below bed and Remove base sit beside Fill LED when the column has room, else on a row of their own (fit_width)
        self.bottom = bottom; self.bed_row = QHBoxLayout(); self.bed_row.setSpacing(8); v.addLayout(self.bed_row)
        v.addWidget(self.auto_msg)
        self.two_rows = False; self.layout_dirty = False; self.compact = False
        self.range.changed.connect(lambda vals: self._show_range(*vals))
        self.range.changed.connect(lambda vals: self.range_changed.emit(*vals))
        self.gain.changed.connect(lambda vals: (self.gain_row.value.setText(str(vals[0])), self.gain_changed.emit(vals[0])))
        self.auto_range.toggled.connect(self.auto_range_toggled); self.auto_gain.toggled.connect(self.auto_gain_toggled)
        self.led.toggled.connect(self.led_toggled)
        self.below_bed.toggled.connect(self.below_bed_toggled); self.remove_base.toggled.connect(self.remove_base_toggled)
        from app2.qt.capture_tools import CaptureTools
        self.capture_tools=CaptureTools();v.addWidget(self.capture_tools)
        cv.insertWidget(0,self.capture_tools.nudge)
        v.removeWidget(self.calibration_setup);section=self.setup_section=QLabel('Camera setup and tests');section.setProperty('role','hint');v.insertWidget(v.indexOf(self.capture_tools),section);v.insertWidget(v.indexOf(self.capture_tools),self.calibration_setup)
        self._show_range(100, 450); self.gain_row.value.setText("4")

    def _show_range(self, near, far): self.range_row.value.setText("%d to %d mm" % (near, far))

    def fit_width(self, width):
        """Cut below bed and Remove base beside Fill LED when all three fit in `width` (the column), else on their own
        row under it. Returns whether that changed (the page then works its sizes out again)."""
        two = True   # their own row, half the width each, like Range test and Reset test
        if two == self.two_rows: return False
        self.two_rows = two
        for b in (self.below_bed, self.remove_base): (self.bottom if two else self.bed_row).removeWidget(b)   # from where they are
        if two:
            self.bed_row.addWidget(self.below_bed, 1); self.bed_row.addWidget(self.remove_base, 1)
        else:
            while self.bed_row.count(): self.bed_row.takeAt(0)
            self.bottom.insertWidget(2, self.below_bed); self.bottom.insertWidget(3, self.remove_base)
        self.layout().activate()
        return True

    def set_compact(self, compact):
        """Short windows: tighter padding, and the Auto gain line goes (the status line says it instead)."""
        self.range_row.set_compact(compact); self.gain_row.set_compact(compact)
        self.layout().setContentsMargins(14, 6 if compact else 10, 14, 8 if compact else 12)
        self.layout().setSpacing(4 if compact else 8)
        self.compact = compact; self._show_auto_msg()

    def _show_auto_msg(self):
        """Gain details stay on hover; the framing row owns the single status line."""
        self.auto_msg.hide()
        self.gain_row.setToolTip(self.auto_msg.text() or 'Auto gain tries a new gain after a full turn.')

    def height_for(self, width):
        """The controls' height at this width (the Auto gain line wraps)."""
        lay = self.layout(); lay.activate()
        return lay.heightForWidth(width) if lay.hasHeightForWidth() else self.sizeHint().height()

    @staticmethod
    def _set_checked(b, on):
        if b.isChecked() != bool(on): b.blockSignals(True); b.setChecked(bool(on)); b.blockSignals(False)

    def set_state(self, s, btns):
        """From a ScanState and ui_model.buttons()."""
        self.live_build.set_state(s)
        outside=getattr(s,'outside_below_mm',None)
        if outside is not None:
            self.outside_below.blockSignals(True);self.outside_below.setValue(round(outside));self.outside_below.blockSignals(False)
        self.outside_below.setEnabled(not (s.recording or s.busy or s.building or s.learning))
        lift=getattr(s,'bed_lift_mm',None)
        if lift is not None and round(lift)!=self.bed_lift.value():
            self.bed_lift.blockSignals(True);self.bed_lift.setValue(round(lift));self.bed_lift.blockSignals(False)
        name=getattr(s,'surface_name',None) or 'Normal';self.surface.blockSignals(True);self.surface.setCurrentText(name);self.surface.blockSignals(False);self.surface.setEnabled(not (s.recording or s.busy or s.building))
        helper=getattr(s,'surface_hint',None) or {};kind=helper.get('kind','unknown')
        # one line from the surface check: what the camera sees on the part, and whether the Surface choice already fits it
        picked=self.surface.currentText();match={'dark':'Dark','shiny':'Shiny'}.get(kind)
        text={'good':'Surface: good','unknown':'Surface: waiting for the part',
              'dark':'Surface: dark spots, no depth','shiny':'Surface: shiny glare'}.get(kind,'Surface: '+helper.get('text',''))
        if match:text+=(', %s is set' % match) if picked in (match,'Sprayed') else ', try %s or spray' % match
        state='ok' if kind=='good' or (match and picked in (match,'Sprayed')) else 'idle' if kind=='unknown' else 'warn'
        self.surface_note.set_status(state,text,'Watches the IR picture of the part while you frame it. Dark: parts of the part return no depth because they soak up IR. Shiny: the IR picture is blown out by glare. Pick the matching Surface, or dust the part with scanning spray and pick Sprayed.')
        self._set_checked(self.light_during_scans,getattr(s,'light_during_scans',False))
        guide=getattr(s,'range_guide',None);self.range.guide=None;self.range.update()   # drawn big under the live picture
        if guide:self.distance_status.setText(guide.get('label',guide['state']).capitalize())
        frame=getattr(s,'framing',None) or {};busy=bool(s.recording or s.busy or s.building or s.learning or (getattr(s,'range_test',None) or {}).get('active'))
        self.frame_range.setEnabled(frame.get('source')=='part' and frame.get('range_mm') is not None and not busy)
        undo=bool(getattr(s,'range_undo',False));self.range_undo.setEnabled(undo and not busy)
        setup=getattr(s,'scene_setup',None) or {};details=[]
        # hover text: only what to do next, at most three lines (counts and the box legend were a wall of text)
        if setup:details.extend(setup.get('problems',[]))
        hint=('Check camera setup' if getattr(s,'calibration_error',None) else 'Part Range applied' if undo else frame.get('hint','Show the part above the bed'))
        if getattr(s,'calibration_error',None) and getattr(s,'calibration_missing',None) is False:
            hint='Use the matching Object setting' if (getattr(s,'mode_evidence',None) or {}).get('mode') else 'Show learned dots for the camera check'
        if getattr(s,'calibration_error',None):details.insert(0,s.calibration_error)
        details=[d for d in dict.fromkeys(details) if d!=hint][:3] or [hint]
        state='problem' if getattr(s,'calibration_error',None) else 'ok' if frame.get('state')=='framed' or undo else 'warn'   # an applied Range is a done thing, not a warning
        self.frame_note.set_status(state,hint);self.frame_note.detail_tooltip='\n'.join(details);self.frame_note.setToolTip(self.frame_note.detail_tooltip)
        self.calibration_info.hide();self.scene_note.hide();self.distance_status.hide()
        limits = getattr(s, "range_limits", None) or self.range_default_limits
        if (self.range.lo, self.range.hi) != tuple(limits):
            self.range.lo, self.range.hi = limits; self.range.update()
        if not self.range.dragging: self.range.set_values((s.near, s.far)); self._show_range(*self.range.values)
        if not self.gain.dragging: self.gain.set_values((s.gain,)); self.gain_row.value.setText(str(self.gain.values[0]))
        self._set_checked(self.auto_range, s.auto_range); self._set_checked(self.auto_gain, s.auto_gain)
        self._set_checked(self.save_color,True if getattr(s,"save_color",None) is None else s.save_color);self.save_color.setEnabled(not s.busy and not s.recording)
        self._set_checked(self.led, bool(getattr(s,'led_requested',s.led)) and s.has_led)   # what was asked; the label says if it is confirmed
        self.auto_range.setEnabled(btns["auto_range"][2]); self.auto_gain.setEnabled(btns["auto_gain"][2])
        self.range.setEnabled(not s.busy); self.gain.setEnabled(not s.busy)
        text, _, on = btns["led"]
        status=getattr(s,'led_status',None)
        # the button always says LED; its color shows on or off, and the hover text says the rest
        words={'pending':'Sending to the scanner.','unverified':'The scanner got the request but cannot confirm the light is on.','failed':'The scanner did not take the request.','on':'On.','off':'Off.'}
        if self.led.text()!='LED':self.led.setText('LED')
        self.led.setEnabled(on)
        self.led.setToolTip(("Turn the scanner's white light on or off (L). "+words.get(status,'')).strip() if s.has_led else "The light needs the local add-on to be set up.")
        for b, key in ((self.below_bed, "below_bed"), (self.remove_base, "remove_base")):
            if key in btns and getattr(s, key, None) is not None:
                self._set_checked(b, getattr(s, key)); b.setEnabled(btns[key][2])
        self.remove_base.setToolTip(REMOVE_BASE_TIP if self.remove_base.isChecked() else KEEP_BASE_TIP)

    def set_auto_msg(self, text):
        text = text[:1].upper() + text[1:] if text else ""
        if text != self.auto_msg.text():
            self.auto_msg.setText(text); self.auto_msg.setToolTip(text)
            if not self.auto_msg.isHidden(): self.layout_dirty = True     # may wrap onto more lines
        self._show_auto_msg()


class CameraSwitch(QWidget):
    """IR / Depth / Colour as one segmented switch: on short windows only one camera picture fits."""
    picked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        self.buttons = []; self.current = 0
        for i, name in enumerate(("IR", "Depth", "Color")):
            b = button(name, "seg"); b.setCheckable(True)
            b.setProperty("pos", "first" if i == 0 else ("last" if i == 2 else "mid"))
            b.clicked.connect(lambda _=False, k=i: self.pick(k)); h.addWidget(b, 1); self.buttons.append(b)
        self.pick(0)

    def pick(self, k):
        self.current = k
        for i, b in enumerate(self.buttons): b.setChecked(i == k)
        self.picked.emit(k)


# ---------------------------------------------------------------- sides
class Thumbs(QWidget):
    """A side's two pictures: the colour camera at its start and its model so far. Narrow: the model alone, with the
    colour picture as a small inset."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rgb = None; self.model = None; self.single = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_images(self, rgb, model): self.rgb, self.model = rgb, model; self.update()

    def draw(self, p, img, r, rad=6):
        path = QPainterPath(); path.addRoundedRect(r, rad, rad); p.fillPath(path, qc("well"))
        if img is None: return
        s = max(r.width() / img.width(), r.height() / img.height()); w, h = img.width() * s, img.height() * s
        p.save(); p.setClipPath(path); p.drawImage(QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h), img); p.restore()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        if self.single and self.model is None:          # a side's first pass: no model picture until it pauses
            self.draw(p, self.rgb, QRectF(0, 0, W, H))
        elif self.single:
            self.draw(p, self.model, QRectF(0, 0, W, H))
            ins = QRectF(W - W * 0.38 - 5, H - H * 0.38 - 5, W * 0.38, H * 0.38)
            p.setPen(QPen(qc("raised"), 3)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRoundedRect(ins, 4, 4)
            self.draw(p, self.rgb, ins, 4)
        else:
            w = (W - 6) / 2
            self.draw(p, self.rgb, QRectF(0, 0, w, H)); self.draw(p, self.model, QRectF(w + 6, 0, w, H))


class SideCard(QFrame):
    """One side: its two pictures, "Side N", frames, lost (red), where its base cut comes from. Picked: blue outline
    and Delete on it; scanning: green outline and "Scanning"."""
    picked = Signal(int)
    delete_asked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "card"); self.no = 0; self.name_key = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        v = QVBoxLayout(self); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
        self.thumbs = Thumbs(); v.addWidget(self.thumbs)
        r1 = QHBoxLayout(); r1.setSpacing(8)
        self.name = FitLabel(); self.name.setFont(ui_font(14, QFont.Weight.DemiBold))
        self.name.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.live = row_label("Scanning", "live"); self.live.hide()
        self.frames = row_label("", "stat")
        r1.addWidget(self.name); r1.addWidget(self.live); r1.addStretch(1); r1.addWidget(self.frames)
        v.addLayout(r1)
        r2 = QHBoxLayout(); r2.setSpacing(8)
        self.base = QLabel(); self.base.setFont(ui_font(11)); self.lost = row_label("", "lost")
        self.scan_mode = row_label("Small part", "stat")
        self.delete = button("Delete", "danger", "Delete this side and its frames (Del)"); self.delete.setFont(ui_font(11)); self.delete.hide()
        self.delete.clicked.connect(lambda: self.delete_asked.emit(self.no))
        r2.addWidget(self.scan_mode); r2.addWidget(self.base); r2.addWidget(self.lost); r2.addStretch(1); r2.addWidget(self.delete)
        v.addLayout(r2)
        self._state = None

    def set_info(self, c, images):
        self.no = c["no"]; self.name_key = c["name"]
        # a touch-up side says so when there is room (a narrow card while scanning keeps "Side N")
        self.name.set_texts(*(("Touch up",) if c.get("touch") and not c["frames"] else ("Side %d · touch up" % c["no"],"Touch up") if c.get("touch") else ("Side %d" % c["no"],)))
        self.frames.setText("%d frames" % c["frames"])
        self.scan_mode.setText({"far": "Large part", "near": "Small part"}.get(c.get("scan_mode"), "Object not saved"))
        self.lost.setText("%d lost" % c["lost"] if c["lost"] else ""); self.lost.setVisible(bool(c["lost"]))
        base = c.get("base")
        if base:
            self.base.setText("base from bed" if base == "bed" else "base: Auto")
            role = "badge-bed" if base == "bed" else "badge-auto"
            if self.base.property("role") != role: self.base.setProperty("role", role); repolish(self.base)
        self.base.setVisible(bool(base))
        if images is not None and (self.thumbs.rgb, self.thumbs.model) != images: self.thumbs.set_images(*images)
        self.set_state(c["picked"], c.get("scanning", False))

    def set_state(self, picked=False, live=False):
        st = (picked, live)
        if st == self._state: return
        self._state = st
        self.setProperty("picked", "true" if picked and not live else "false"); self.setProperty("live", "true" if live else "false")
        self.live.setVisible(live); self.delete.setVisible(not live); repolish(self)

    def set_narrow(self, narrow, width):
        self.thumbs.single = narrow
        tw = width - 16
        self.thumbs.setFixedHeight(max(40, int(tw * 0.5) if narrow else int((tw - 6) / 2 * 0.75))); self.thumbs.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.picked.emit(self.no)


class BuildBox(QWidget):
    """Build combined model: one model from every side (Stop build while it runs), and while a build runs a bar with
    the step, percent and time. The combined model is shown on the Build step (the window goes there when the build is
    done), never on the Scan step. The runner behind it (build_project.py today, the staged engine later) only
    reaches it through set_progress()."""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        r = QHBoxLayout(); r.setSpacing(8)
        self.build = button("Build combined model", "build",
                            "Build all sides with the current settings (Ctrl+B). YOLO Yoink in Build lets you choose a new recipe.")
        self.build.setMinimumHeight(38)
        self.stop = button("Stop build", "stop", "Stop the build"); self.stop.setMinimumHeight(38); self.stop.hide()
        r.addWidget(self.build, 1); r.addWidget(self.stop, 1)
        v.addLayout(r)
        self.progress=QWidget();pv=QVBoxLayout(self.progress);pv.setContentsMargins(0,0,0,0);pv.setSpacing(v.spacing())
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(8); self.bar.setRange(0, 1000)
        self.step = StableLabel("", 2); self.step.setProperty("role", "stat")
        sp = self.step.sizePolicy(); sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored); self.step.setSizePolicy(sp)
        self.pct = row_label("", "value")
        pr = QHBoxLayout(); pr.addWidget(self.step, 1); pr.addWidget(self.pct, 0, Qt.AlignmentFlag.AlignTop)
        pv.addWidget(self.bar);pv.addLayout(pr);v.addWidget(self.progress)
        for w in (self.bar, self.step, self.pct): keep_room(w,False)     # a build starting or ending never moves the cards above
        self.prog = None; self.set_progress(None)

    def set_progress(self, prog):
        """A BuildProgress while a build runs (None or a finished one: no bar)."""
        on = prog is not None and not prog.finished
        self.prog = prog if on else None
        self.progress.setVisible(on)
        for w in (self.bar, self.step, self.pct): w.setVisible(on)
        if on:
            self.bar.setValue(int(prog.frac * 1000)); self.step.setText(prog.step); self.pct.setText(prog.elapsed_text())

    def tick(self):
        """Once a second while building: the clock moves even when the build is quiet."""
        if self.prog is not None: self.pct.setText(self.prog.elapsed_text())

    def set_buttons(self, btns):
        building = btns["stop_build"][2]
        self.build.setVisible(not building); self.stop.setVisible(building)
        self.build.setEnabled(btns["build"][2])


class SidesPanel(QFrame):
    """The project's sides as cards (scrolling) under a pinned card (the page's Bed card), + New side, Learn bed (only
    with the bed adapter), and the BuildBox.
    Cards update in place: they are rebuilt only when sides come or go, and their pictures are converted only when a
    side's thumbs changed."""
    pick_asked = Signal(int)
    delete_asked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("SidesPanel"); self.setProperty("role", "panel")
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)
        self.tabs=QTabBar();self.tabs.setObjectName('ScanSettingsTabs');self.tabs.addTab('Sides');self.tabs.addTab('Settings');self.tabs.setExpanding(True);self.tabs.setDrawBase(False)
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus);v.addWidget(self.tabs)
        self.pages=QStackedWidget();v.addWidget(self.pages,1)
        self.side_page=QWidget();self.pages.addWidget(self.side_page)
        side_layout=QVBoxLayout(self.side_page);side_layout.setContentsMargins(0,0,0,0);side_layout.setSpacing(v.spacing())
        self.side_layout=side_layout
        self.tabs.currentChanged.connect(self.pages.setCurrentIndex)
        head = QHBoxLayout()
        t = QLabel("Sides"); t.setFont(ui_font(16, QFont.Weight.Bold)); self.count = row_label("", "stat")
        head.addWidget(t); head.addWidget(self.count); head.addStretch(1)
        side_layout.addLayout(head)
        self.hint = row_label("Scanning adds to the picked side. Flip the part, then add a new side.", "hint"); self.hint.setWordWrap(True)
        side_layout.addWidget(self.hint)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list = QWidget(); self.lv = QVBoxLayout(self.list); self.lv.setContentsMargins(0, 0, 4, 0); self.lv.setSpacing(8); self.lv.addStretch(1)
        self.scroll.setWidget(self.list); side_layout.addWidget(self.scroll, 1)
        self.empty = row_label("No sides yet. Press Start to scan the first one.", "hint"); self.empty.setWordWrap(True)
        self.lv.insertWidget(0, self.empty)
        r = QHBoxLayout(); r.setSpacing(8)
        self.new = button("+ New side", None, "Start a new side after you flip the part over (N)"); self.new.setMinimumHeight(34)
        self.learn = button("Learn bed", None, "Spin the empty bed once so the app learns it. After that it finds the bed right away on every scan. Optional: one dot on the bed's side edge, and two side by side opposite it, help tracking when the part hides the top dots.")
        self.learn.setMinimumHeight(34); self.learn.hide()
        r.addWidget(self.new, 1); r.addWidget(self.learn, 1)
        self.action_row = r
        v.addLayout(r)   # under both tabs: + New side and Learn bed are reachable from Settings too
        # Touch up: only once there is a built model (the page shows it then)
        self.touch = button("Touch up", None, "Scan the finished model again to fill holes or rough spots. "
                            "The new scan lines up with the model as you go.")
        self.touch.setMinimumHeight(34); self.touch.hide(); v.addWidget(self.touch)
        self.touch_reason = QLabel(""); self.touch_reason.setWordWrap(True); self.touch_reason.setProperty("role", "hint")
        self.touch_reason.hide(); v.addWidget(self.touch_reason)
        sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet("background: %s;" % T["line"]); v.addWidget(sep)
        self.buildbox = BuildBox(); v.addWidget(self.buildbox)
        self.cards = []; self.picked_card = None; self._cache = {}; self._width = 352; self._narrow = False
        self.pinned = None
        # Selection changes and window resizing scroll explicitly. Build progress
        # only changes the viewport height and must preserve the visible cards.

    def pin(self, w):
        """A card that stays above the list (the Bed card): it does not scroll away."""
        self.pinned = w; lay = self.side_layout; lay.insertWidget(lay.indexOf(self.scroll), w)
        if hasattr(w, "set_narrow"): w.set_narrow(self._narrow)

    def set_settings(self,widget):
        self.pages.addWidget(widget)

    def _images(self, c):
        key = (c["name"], c.get("version", 0))
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != c["name"]}
            self._cache[key] = (to_qimage(c["rgb_thumb"]), to_qimage(c["model_thumb"]))
        return self._cache[key]

    def set_sides(self, cards):
        """Returns True when the picked card changed or sides came or went (the page then scrolls it into view)."""
        names = [c["name"] for c in cards]
        rebuilt = names != [c.name_key for c in self.cards]
        if rebuilt:
            for c in self.cards: c.setParent(None); c.deleteLater()
            self.cards = []
            for c in cards:
                w = SideCard(); w.picked.connect(self.pick_asked); w.delete_asked.connect(self.delete_asked)
                w.set_narrow(self._narrow, self._width - 24 - 4 - 10)
                self.lv.insertWidget(self.lv.count() - 1, w); self.cards.append(w)
            live = set(names); self._cache = {k: v for k, v in self._cache.items() if k[0] in live}
        before = self.picked_card
        for w, c in zip(self.cards, cards): w.set_info(c, self._images(c))
        self.picked_card = next((w for w, c in zip(self.cards, cards) if c["picked"]), None)
        self.count.setText("%d" % len(cards)); self.empty.setVisible(not cards)
        return rebuilt or self.picked_card is not before

    def show_picked(self):
        """A side was picked, added or removed: scroll so the picked card is in view (past a few sides the list runs
        off the panel)."""
        if self.picked_card is not None: self.scroll.ensureWidgetVisible(self.picked_card, 0, 8)

    def fit(self, width, height):
        self._width = width; self._narrow = width < 330
        for c in self.cards: c.set_narrow(self._narrow, width - 24 - 4 - 10)
        if self.pinned is not None and hasattr(self.pinned, "set_narrow"): self.pinned.set_narrow(self._narrow)
        self.hint.setVisible(width >= 300 and height >= 700)


class LogPanel(QFrame):
    """What the tracker, Auto gain, the bed and the build have to say, in full, newest at the bottom (500 lines). It
    takes the camera column's spare height; on short windows it folds away and the status line's Log button shows the
    same lines in a popover."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "panel")
        v = QVBoxLayout(self); v.setContentsMargins(12, 8, 8, 8); v.setSpacing(4)
        v.addWidget(row_label("Log"))
        self.text = log_edit(500)
        self.text.setMinimumHeight(QFontMetrics(self.text.font()).lineSpacing() * 5 + 12)
        v.addWidget(self.text, 1)
        self.lines = collections.deque(maxlen=500); self.mirror = None

    def append(self, line):
        self.lines.append(line)
        for t in (self.text, self.mirror):
            if t is not None: t.appendPlainText(line)

    def toPlainText(self): return "\n".join(self.lines)

    def clear(self):
        self.lines.clear(); self.text.clear()
        if self.mirror is not None: self.mirror.clear()

    def min_height(self, lines=5):
        return int(QFontMetrics(self.text.font()).lineSpacing() * lines + 12 + 16 + 4 + QFontMetrics(ui_font(13)).height())


def log_edit(max_lines):
    t = QPlainTextEdit(); t.setObjectName("Log"); t.setReadOnly(True); t.setFont(ui_font(12))
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)          # Space must always reach Start / Pause
    t.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    t.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth); t.setMaximumBlockCount(max_lines)
    follow_end(t)
    return t


def follow_end(t):
    """Keep a log view on its newest line until the user scrolls up; back at the bottom, it follows again. Decided on
    the scroll bar's range, not on each append: when the bar first appears the text wraps narrower after the append,
    and a view that only scrolled on append stopped short of the bottom for good."""
    sb = t.verticalScrollBar(); st = dict(follow=True, auto=False)
    def moved(v):
        if not st["auto"]: st["follow"] = v >= sb.maximum() - 2
    def grew(lo, hi):
        if st["follow"]: st["auto"] = True; sb.setValue(hi); st["auto"] = False
    sb.valueChanged.connect(moved); sb.rangeChanged.connect(grew)
    t.follow_state = st


def fit_overlay_in_parent(panel):
    """Keep a movable child overlay reachable after its parent changes size."""
    parent = panel.parentWidget()
    if parent is None: return
    panel.resize(min(panel.width(), max(panel.minimumWidth(), parent.width() - 16)),
                 min(panel.height(), max(panel.minimumHeight(), parent.height() - 16)))
    panel.move(max(8, min(panel.x(), parent.width() - panel.width() - 8)),
               max(8, min(panel.y(), parent.height() - panel.height() - 8)))


class MovablePanelTitle(QLabel):
    """A log popup's title: drag it to move the popup without grabbing its text or Close button."""
    def __init__(self, title, panel):
        super().__init__(title, panel); self.panel = panel; self._grab = None
        self.setCursor(Qt.CursorShape.OpenHandCursor); self.setToolTip("Drag to move")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._grab = (e.globalPosition().toPoint(), self.panel.pos())
            self.setCursor(Qt.CursorShape.ClosedHandCursor); e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._grab is not None and e.buttons() & Qt.MouseButton.LeftButton:
            start, old = self._grab; delta = e.globalPosition().toPoint() - start
            self.panel.move(old + delta); fit_overlay_in_parent(self.panel)
            self.panel._user_placed = True; e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._grab = None; self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(e)


class PanelResizeGrip(QWidget):
    """Bottom-right handle that resizes a child popup inside its page."""
    def __init__(self, panel):
        super().__init__(panel); self.panel = panel; self._grab = None
        self.setFixedSize(20, 20); self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Drag to resize")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._grab = (e.globalPosition().toPoint(), self.panel.size())
            e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._grab is not None and e.buttons() & Qt.MouseButton.LeftButton:
            start, old = self._grab; delta = e.globalPosition().toPoint() - start
            parent = self.panel.parentWidget()
            if parent is not None:
                self.panel.resize(max(self.panel.minimumWidth(), min(old.width() + delta.x(), parent.width() - self.panel.x() - 8)),
                                  max(self.panel.minimumHeight(), min(old.height() + delta.y(), parent.height() - self.panel.y() - 8)))
                self.panel._user_placed = True
            e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._grab = None; super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        p = QPainter(self); p.setPen(QColor("#8294a8"))
        for offset in (0, 5, 10): p.drawLine(6 + offset, 18, 18, 6 + offset)


class LogPopover(QFrame):
    """Movable, resizable Scan log, opened by the status line's Log button."""
    def __init__(self, log_panel, parent=None):
        super().__init__(parent); self.setObjectName("LogPopover"); self.panel = log_panel; self._user_placed = False
        self.setMinimumSize(300, 180)
        v = QVBoxLayout(self); v.setContentsMargins(12, 8, 8, 10); v.setSpacing(4)
        h = QHBoxLayout(); self.drag_title = MovablePanelTitle("Scan log", self)
        self.drag_title.setProperty("role", "section"); self.drag_title.setFont(ui_font(13, QFont.Weight.DemiBold))
        h.addWidget(self.drag_title, 1)
        self.close_b = button("Close", "quiet", "Close the log (Esc)"); self.close_b.clicked.connect(self.close_popover); h.addWidget(self.close_b)
        v.addLayout(h); self.text = log_edit(500); v.addWidget(self.text, 1)
        footer = QHBoxLayout(); footer.addStretch(1); self.resize_grip = PanelResizeGrip(self); footer.addWidget(self.resize_grip)
        v.addLayout(footer); self.hide()

    def open_popover(self):
        self.text.follow_state["follow"] = True
        self.text.setPlainText("\n".join(self.panel.lines)); self.panel.mirror = self.text
        sb = self.text.verticalScrollBar(); sb.setValue(sb.maximum()); self.show(); self.raise_()

    def close_popover(self): self.panel.mirror = None; self.hide()


# ---------------------------------------------------------------- shell
class StepButton(QPushButton):
    """One step on the left rail: its number in a ring and its name (the name goes when the rail is narrow). Steps
    that are not built yet are dimmed."""
    def __init__(self, n, name, parent=None):
        super().__init__(parent); self.n = n; self.name = name; self.compact = False; self.ready = True
        self.setCheckable(True); self.setCursor(Qt.CursorShape.PointingHandCursor); self.setToolTip("%s (Ctrl+%d)" % (name, n))
        self.setFixedHeight(64); self.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.setAccessibleName(name)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(6, 2, -6, -2)
        if self.isChecked() or self.underMouse():
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(qc("raised")); p.drawRoundedRect(r, 8, 8)
        if self.isChecked():
            p.setBrush(qc("accent")); p.drawRoundedRect(QRectF(0, r.y() + 12, 3, r.height() - 24), 1.5, 1.5)
        R = 13; c = QPointF(self.width() / 2, r.y() + (r.height() / 2 if self.compact else 22))
        if self.isChecked():
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(qc("accent"))
        else:
            p.setPen(QPen(qc("ink3") if not self.ready else qc("ink2"), 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, R, R)
        f = ui_font(12, QFont.Weight.Bold); p.setFont(f); fm = QFontMetrics(f)
        p.setPen(qc("on_color") if self.isChecked() else (qc("ink2") if self.ready else qc("ink3")))
        t = str(self.n); p.drawText(QPointF(c.x() - fm.horizontalAdvance(t) / 2, c.y() + fm.ascent() / 2 - 1), t)
        if not self.compact:
            f = ui_font(12, QFont.Weight.DemiBold if self.isChecked() else QFont.Weight.Normal); p.setFont(f); fm = QFontMetrics(f)
            p.setPen(qc("ink") if self.isChecked() else (qc("ink2") if self.ready else qc("ink3")))
            p.drawText(QPointF(self.width() / 2 - fm.horizontalAdvance(self.name) / 2, c.y() + R + 6 + fm.ascent()), self.name)


class Mark(QWidget):
    """The existing PointYoink logo in the rail's reserved 40 px slot."""
    def __init__(self, parent=None, size=40):
        super().__init__(parent); self.setFixedSize(size, size)
        self.logo = QPixmap(str(Path(REPO) / "icon.png"))
        self.setAccessibleName("PointYoink logo")

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.logo.isNull(): p.drawPixmap(QRectF(2, 2, self.width() - 4, self.height() - 4), self.logo, QRectF(self.logo.rect()))


class Scrim(QWidget):
    """Dims the window behind a sheet (the Keys sheet, the log popover); a click closes it."""
    clicked = Signal()

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(6, 7, 9, 170))

    def mousePressEvent(self, e): self.clicked.emit()


class Rail(QFrame):
    """The six steps down the left. Scan and Build are live; the other four are dimmed and open their placeholder."""
    step_asked = Signal(int)

    def __init__(self, ready=(1, 2), parent=None):
        super().__init__(parent); self.setObjectName("Rail")
        v = QVBoxLayout(self); v.setContentsMargins(0, 12, 0, 12); v.setSpacing(2)
        self.logo = Mark(); self.logo.setToolTip(APP_TITLE)
        v.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter); v.addSpacing(10)
        self.buttons = []
        for i, name in enumerate(STEPS):
            b = StepButton(i + 1, name); b.ready = (i + 1) in ready
            b.clicked.connect(lambda _=False, k=i: self.step_asked.emit(k)); v.addWidget(b); self.buttons.append(b)
        v.addStretch(1)
        self.keys = button("Keys", "quiet", "Keyboard shortcuts (?)")
        v.addWidget(self.keys, 0, Qt.AlignmentFlag.AlignHCenter)
        self.keys.hide()
        self.show_step(0)

    def show_step(self, k):
        for i, b in enumerate(self.buttons): b.setChecked(i == k)

    def set_compact(self, compact):
        self.setFixedWidth(60 if compact else 84)
        for b in self.buttons: b.compact = compact; b.setFixedHeight(48 if compact else 64); b.update()
        self.keys.setText("?" if compact else "Keys")


class StatusLine(QFrame):
    """State pill (the same colour as the rings round the IR picture and the model), then the numbers (values bright,
    words dim), then what to do next."""
    def __init__(self, parent=None):
        # grows with the font (a fixed 36 px squeezed the pill against the window's edge on a high-DPI desktop)
        super().__init__(parent); self.setObjectName("StatusLine"); self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        h = QHBoxLayout(self); h.setContentsMargins(12, 6, 14, 6); h.setSpacing(18)
        self.pill = QLabel(); self.pill.setFont(ui_font(13, QFont.Weight.Bold))
        self.stats = QLabel(); self.stats.setProperty("role", "stat"); self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.hint = ElidedLabel(); self.hint.setProperty("role", "next")         # what to do next: readable, not dim
        self.hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.log_btn = button("Log", "quiet", "Show the log"); self.log_btn.hide()
        self.setMinimumHeight(max(40, self.log_btn.sizeHint().height() + 14))
        h.addWidget(self.pill); h.addWidget(self.stats); h.addWidget(self.hint, 1); h.addWidget(self.log_btn)
        self._shown = None
        self._context_hint = None
        self._scan_hint = ""

    def set_context(self, hint=None):
        """The scanner summary stays global; instructions follow the active workspace."""
        self._context_hint = hint
        self.pill.setVisible(hint is None)
        self.stats.setVisible(hint is None)
        text = self._scan_hint if hint is None else hint
        if not hasattr(self, "steady"):
            from app2.qt.steady import Steady
            self.steady = Steady(lambda t: (self.hint.setText(t), self.hint.setToolTip(t)), self)
        self.steady.offer(text, text, urgent=hint is not None)   # a workspace change is not a flicker

    def set_view(self, v, show_auto=False):
        stats = list(v.stats) + ([("{}", v.auto_msg[:1].upper() + v.auto_msg[1:], None)] if (show_auto and v.auto_msg) else [])
        key = (v.kind, v.pill, tuple(stats), v.hint)
        if key == self._shown: return
        self._shown = key
        c = T[KIND_COLOUR.get(v.kind, "ink2")]
        bg = QColor(c); bg.setAlphaF(0.16)
        self.pill.setText("●  " + v.pill)
        self.pill.setStyleSheet("color: %s; background: rgba(%d,%d,%d,%.2f); border-radius: 11px; padding: 3px 12px;"
                                % (c, bg.red(), bg.green(), bg.blue(), bg.alphaF()))
        cells = []
        for tmpl, val, colour in stats:
            val = "<span style='color:%s; font-weight:600'>%s</span>" % (T[colour] if colour else T["ink"], val)
            a, _, b = tmpl.partition("{}")
            cells.append("<span style='color:%s'>%s</span>%s<span style='color:%s'>%s</span>"
                         % (T["ink2"], a.replace(" ", "&nbsp;"), val, T["ink2"], b.replace(" ", "&nbsp;")))
        self.stats.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(cells))
        self._scan_hint = v.hint
        self.set_context(self._context_hint)


class Banner(QFrame):
    """A warning across the top of the live area (no frames, scanner gone, replay finished, tracking too slow).
    Hidden when all is well."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("Banner")
        h = QHBoxLayout(self); h.setContentsMargins(14, 8, 10, 8); h.setSpacing(12)
        self.icon = QLabel("!"); self.icon.setFixedSize(22, 22); self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setFont(ui_font(14, QFont.Weight.Black))
        self.text = QLabel(); self.text.setObjectName("BannerText"); self.text.setWordWrap(True)
        self.more = row_label("", "stat"); self.more.setWordWrap(True)
        tv = QVBoxLayout(); tv.setSpacing(0); tv.addWidget(self.text); tv.addWidget(self.more)
        h.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop); h.addLayout(tv, 1)
        self.shown = None; self.hide()

    def show_msg(self, banner):
        """banner: (level, text, more) or None."""
        if banner == self.shown: return
        self.shown = banner
        if banner is None: self.hide(); return
        level, text, more = banner
        self.setProperty("level", level); repolish(self)
        c = T[{"warn": "warn", "lost": "lost"}.get(level, "accent")]
        self.icon.setText("i" if level == "info" else "!")
        self.icon.setStyleSheet("background: %s; color: %s; border-radius: 11px;" % (c, T["on_color"]))
        self.text.setStyleSheet("color: %s;" % c)
        self.text.setText(text); self.more.setText(more); self.more.setVisible(bool(more))
        self.show()


class Header(QFrame):
    """The step, date, editable project name, source, and project buttons."""
    project_named = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("Header"); self.setFixedHeight(52)
        h = QHBoxLayout(self); h.setContentsMargins(18, 0, 14, 0); h.setSpacing(10)
        self.step = QLabel("Scan"); self.step.setFont(ui_font(18, QFont.Weight.Bold))
        self.step.setFixedWidth(max(QFontMetrics(self.step.font()).horizontalAdvance(name) for name in STEPS) + 4)
        self.project = ElidedLabel(); self.project.setObjectName("ProjectName")
        self.project.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        date_font = ui_font(14, QFont.Weight.DemiBold)
        self.project.setFont(date_font)
        self.project.setFixedWidth(QFontMetrics(date_font).horizontalAdvance("28 Sep 2026, 23:59") + 8)
        self.folder = QLineEdit(); self.folder.setProperty("role", "hint"); self.folder.setFixedHeight(32)
        self.folder.setMinimumWidth(90)
        self.folder.setMaxLength(100); self.folder.setPlaceholderText("Name this project")
        self.folder.editingFinished.connect(lambda: self.project_named.emit(self.folder.text().strip()) if self.folder.text().strip() else None)   # Enter or leaving the field
        self.original = ElidedLabel(); self.original.setProperty("role", "hint"); self.original.setFixedWidth(210); self.original.hide()   # the scanner's own name, beside a name you gave
        self.source = QLabel(); self.source.setObjectName("SourcePill"); self.source.setTextFormat(Qt.TextFormat.RichText); self.source.setFixedSize(200, 32)
        self.slot = QWidget(); QHBoxLayout(self.slot).setContentsMargins(0, 0, 0, 0); self.slot.hide()      # a page can put its own tabs here (Import)
        self.pill = QFrame(); self.pill.setObjectName("RebuildPill"); self.pill.setFixedHeight(32); self.pill.setStyleSheet("QFrame#RebuildPill { background: #1c2a3d; border: 1px solid #33527c; border-radius: 8px; }")         # Build: what is out of date, and the button
        ph = QHBoxLayout(self.pill); ph.setContentsMargins(12, 0, 3, 0); ph.setSpacing(8)
        self.pill_text = ElidedLabel(""); self.pill_text.setFont(ui_font(13, QFont.Weight.DemiBold)); self.pill_text.setStyleSheet("color: %s;" % T["accent_hi"]); ph.addWidget(self.pill_text, 1)
        self.pill_btn = button("Rebuild the rest", "build"); self.pill_btn.setFixedHeight(26); ph.addWidget(self.pill_btn); self.pill.hide()
        h.addWidget(self.step); h.addSpacing(14); h.addWidget(self.slot); h.addWidget(self.project); h.addWidget(self.folder, 1); h.addWidget(self.original); h.addWidget(self.pill); h.addWidget(self.source, 0, Qt.AlignmentFlag.AlignVCenter); h.addSpacing(10)
        for w in (self.project, self.folder, self.source):          # hidden on Import: the step name stays where it is
            sp = w.sizePolicy(); sp.setRetainSizeWhenHidden(True); w.setSizePolicy(sp)
        self.motion = button("Motion", "quiet", "Scanner motion over WiFi"); self.motion.setFixedSize(78, 32)
        self.new = button("New project", "quiet", "Save this project and start an empty one")
        self.open = button("Open project", "quiet", "Carry on with a saved project")
        self.finish = button("Finish", None, "Save every side and close (Ctrl+Q)")
        for b in (self.motion, self.new, self.open, self.finish): b.setFixedHeight(32); h.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
        self._source = None
        self._rebuild_text = ""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_header()

    def set_rebuild_text(self, text, tooltip=None):
        self._rebuild_text = text
        self.pill_text.setToolTip(tooltip or text)
        self.pill.setToolTip(tooltip or text)
        self._fit_header()

    def _fit_header(self):
        compact = self.width() < 1440
        date_width = self.project.fontMetrics().horizontalAdvance("28 Sep 2026, 23:59") + 8
        self.project.setFixedWidth(min(date_width, 165 if compact else date_width))
        full = self._rebuild_text
        short = full.split(" · ", 1)[0]
        text = short if compact else full
        self.pill_text.setText(text)
        self.pill_text.setMinimumWidth(min(150 if compact else 360, self.pill_text.fontMetrics().horizontalAdvance(text) + 4) if text else 0)
        self.pill_text.setMaximumWidth(150 if compact else 360)
        if self._source:
            text, colour = self._source
            shown = "Not connected" if compact and text == "Scanner not connected" else text
            self.source.setFixedWidth(min(265, self.source.fontMetrics().horizontalAdvance(shown) + 48))
            self.source.setText("<span style='color:%s'>●</span>&nbsp; %s" % (T[colour], shown))
            self.source.setToolTip(text)

    def set_project(self, when, folder, name=None):
        self.project.setText(when or folder); self.project.setToolTip(folder)
        if not self.folder.hasFocus(): self.folder.setText(name if name is not None else (folder if when else ""))
        named = bool(name) and name != folder                     # a name you gave: the scanner's own name stays visible beside it
        self.folder.setPlaceholderText((folder + "   (click to name it)") if folder else "Name this project")
        self.original.setText(("(%s)" % folder) if named else ""); self.original.setToolTip("Original name: " + folder)
        self.original.setVisible(False)                           # never beside the name: squeezed by the pills it showed as "(P"; the tooltips carry it
        self.folder.setToolTip("Rename this project. Enter or clicking away saves. The original name (%s) is kept.\n" % folder)

    def set_source(self, text, colour="kept"):
        if (text, colour) == self._source: return
        self._source = (text, colour)
        self._fit_header()


class Toast(QFrame):
    """A short message floating over the bottom of the window (Export's result, a refused Open or New), whatever page
    is shown. Goes by itself after a few seconds; a click closes it."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("Banner")
        h = QHBoxLayout(self); h.setContentsMargins(14, 8, 14, 8); h.setSpacing(10)
        self.icon = QLabel("!"); self.icon.setFixedSize(22, 22); self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setFont(ui_font(14, QFont.Weight.Black))
        self.text = QLabel(); self.text.setWordWrap(True); self.text.setFont(ui_font(13, QFont.Weight.DemiBold))
        h.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop); h.addWidget(self.text, 1)
        self.action = button("", "quiet"); self.action.setFixedHeight(28); self.action.hide(); self._act = None
        self.action.clicked.connect(self._acted); h.addWidget(self.action, 0, Qt.AlignmentFlag.AlignVCenter)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.hide)
        self.hide()

    def _acted(self):
        fn = self._act; self.hide()
        if fn is not None: fn()

    def say(self, text, level="warn", secs=8, action=None):
        """action: (button text, what it does), e.g. Open folder after an export."""
        self._act = action[1] if action else None; self.action.setText(action[0] if action else ""); self.action.setVisible(bool(action))
        if not text: self.hide(); return
        self.setProperty("level", level); repolish(self)
        c = T[{"warn": "warn", "lost": "lost"}.get(level, "accent")]
        self.icon.setText("i" if level == "info" else "!")
        self.icon.setStyleSheet("background: %s; color: %s; border-radius: 11px;" % (c, T["on_color"]))
        self.text.setStyleSheet("color: %s;" % c); self.text.setText(text)
        self.place(); self.show(); self.raise_(); self.timer.start(int(secs * 1000))

    def place(self):
        p = self.parentWidget()
        if p is None: return
        w = min(640, max(280, p.width() - 32)); h = self.layout().totalHeightForWidth(w) if self.layout().hasHeightForWidth() else self.sizeHint().height()
        h = max(h, self.sizeHint().height()); y = p.height() // 4             # upper part of the window, where the eye is
        for bar in p.findChildren(DistanceBar):                  # sit above the distance bar, never over it
            if bar.isVisible(): y = min(y, bar.mapTo(p, bar.rect().topLeft()).y() - h - 8)
        self.setGeometry((p.width() - w) // 2, y, w, h)

    def mousePressEvent(self, e): self.hide()


class KeysSheet(QFrame):
    """The ? sheet: every key of the Scan step in one place."""
    KEYS = [("Space", "Start, pause or continue the picked side"), ("N", "New side"), ("Up / Down", "Pick the side above or below"),
            ("Del", "Delete the picked side"), ("H", "Home view (or double-click the model)"), ("R", "Auto range on or off"),
            ("G", "Auto gain on or off"), ("L", "Fill light on or off"), ("Ctrl+B", "Build combined model"), ("Ctrl+1 to 3", "Scan, Build or Edit"),
            ("F11", "Full screen"), ("Ctrl+Q", "Finish: save every side and close"), ("?", "This sheet (Esc closes it)")]

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("KeysSheet")
        g = QGridLayout(self); g.setContentsMargins(20, 16, 20, 18); g.setHorizontalSpacing(16); g.setVerticalSpacing(8)
        t = QLabel("Keys on the Scan step"); t.setFont(ui_font(15, QFont.Weight.Bold)); g.addWidget(t, 0, 0, 1, 2)
        for i, (k, what) in enumerate(self.KEYS):
            kl = QLabel(k); kl.setProperty("role", "keycap-dark"); kl.setFont(ui_font(12, QFont.Weight.DemiBold))
            g.addWidget(kl, i + 1, 0, Qt.AlignmentFlag.AlignRight); g.addWidget(QLabel(what), i + 1, 1)
        self.hide()



def quiet_scrollbar(bar):
    """Reserve the gutter while suppressing a scrollbar that has no scroll range."""
    def update_range(low, high):
        empty = high <= low
        if bar.property("emptyRange") == empty: return
        bar.setProperty("emptyRange", empty)
        bar.style().unpolish(bar); bar.style().polish(bar); bar.update()
    bar.setStyleSheet('QScrollBar[emptyRange="true"]::handle {background:transparent;} QScrollBar[emptyRange="true"]::add-line, QScrollBar[emptyRange="true"]::sub-line {background:transparent;border:none;}')
    bar.rangeChanged.connect(update_range); update_range(bar.minimum(), bar.maximum())
