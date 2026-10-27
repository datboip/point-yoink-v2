# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Bed card on top of the Scan step's sides, and the bed drawn big over the model panel.

The card shows the learned table dots as a small picture (the bed's outline, round or a rectangle, its dots where the
bed map has them), "Bed" with a list of the bed profiles (and "Learn a new bed..."), and one line from
ui_model.bed_card: how many dots and when they were learned, that a provisional bed came from a side (this project
only), that none is learned yet, a flat bed with no dots, or how far Learn bed has got (with an arc round the rim).
Clicking it shows the same picture big over the model panel (BedView); the page decides when that may happen. Both
draw from the session's BedInfo (dots in the platter plane, mm) only."""
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor, QFont, QFontMetrics, QPolygonF
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QSizePolicy, QComboBox
from app2.qt.theme import qc, ui_font
from app2.qt.widgets import row_label, repolish, chip


LEARN_NEW = "Learn a new bed..."


def outline_points(key, k=4):
    """A bed outline key (bedhook.BedInfo.outline) as points (mm, bed frame): None for a circle about the centre (the
    disc is drawn), a circle's (centre, radius) off it, a rectangle's four corners."""
    if not key or key[0] == "circle" and not (key[1] or key[2]): return None
    if key[0] == "circle": return ("circle", (key[1], key[2]), key[3])
    import math
    cx, cy, w, d, a = key[1:6]; ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    return ("rect", [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))])


def draw_bed(p, r, info, deg=None, big=False):
    """The platter (a disc in r, as big as fits, or the bed's rectangle) with the bed's dots. info None: an empty
    dashed rim. deg (learning): an arc round the rim, as far as the table has turned."""
    side = min(r.width(), r.height()); pad = max(3.0, side * 0.06)
    R = side / 2 - pad; c = r.center()
    disc = QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R)
    p.setBrush(QColor(21, 24, 29)); rim = QPen(qc("platter", 0.55) if info is not None else qc("ink3"), 2.5 if big else 1.5)
    if info is None or deg is not None: rim.setStyle(Qt.PenStyle.DashLine); rim.setColor(qc("ink3"))
    p.setPen(rim)
    shape = outline_points(getattr(info, "outline", None)) if (info is not None and deg is None) else None
    if shape is None: p.drawEllipse(disc)
    else:
        s0 = R / max(1.0, info.radius)
        if shape[0] == "circle":
            (x, y), rr = shape[1], shape[2]
            p.drawEllipse(QPointF(c.x() + x * s0, c.y() - y * s0), rr * s0, rr * s0)
        else: p.drawPolygon(QPolygonF([QPointF(c.x() + x * s0, c.y() - y * s0) for x, y in shape[1]]))
    if deg is not None:
        arc = QPen(qc("accent"), 4 if big else 2.5); arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc); p.setBrush(Qt.BrushStyle.NoBrush); p.drawArc(disc, 90 * 16, -int(deg * 16))     # clockwise from the top
        return
    if info is None: return
    s = R / max(1.0, info.radius)
    rd = max(1.6 if not big else 3.0, (info.dot_mm or 8.0) / 2 * s)
    if big:                                                        # the platter's centre, faint
        p.setPen(QPen(qc("ink3"), 1)); k = 6
        p.drawLine(QPointF(c.x() - k, c.y()), QPointF(c.x() + k, c.y())); p.drawLine(QPointF(c.x(), c.y() - k), QPointF(c.x(), c.y() + k))
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(qc("platter"))
    for x, y in info.dots: p.drawEllipse(QPointF(c.x() + x * s, c.y() - y * s), rd, rd)


class BedThumb(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.info = None; self.deg = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_bed(self, info, deg):
        if (info, deg) != (self.info, self.deg): self.info, self.deg = info, deg; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()); path = QPainterPath(); path.addRoundedRect(r, 6, 6); p.fillPath(path, qc("well"))
        draw_bed(p, r, self.info, self.deg)


class BedCard(QFrame):
    """The Bed card (a side card's look): the dot picture, "Bed", one line under it. Blue outline while the bed is
    shown big, green while learning. clicked: only when a click may do something (set_view's clickable)."""
    clicked = Signal()
    profile_picked = Signal(str)            # a bed profile from the list
    learn_new = Signal()                    # "Learn a new bed..."

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "card"); self.setObjectName("BedCard")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        h = QHBoxLayout(self); h.setContentsMargins(8, 8, 10, 8); h.setSpacing(10)
        self.thumb = BedThumb(); self.thumb.setFixedSize(64, 64); h.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignVCenter)
        v = QVBoxLayout(); v.setSpacing(2); v.setContentsMargins(0, 0, 0, 0)
        r1 = QHBoxLayout(); r1.setSpacing(8)
        self.name = QLabel("Bed"); self.name.setFont(ui_font(14, QFont.Weight.DemiBold))
        self.live = row_label("Learning", "live"); self.live.hide()
        self.pick = QComboBox(); self.pick.setToolTip("Which bed is on the turntable")
        self.pick.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents); self.pick.hide()
        self.pick.activated.connect(self._picked); self._items = None; self._current = None
        r1.addWidget(self.name); r1.addWidget(self.pick, 1); r1.addWidget(self.live); r1.addStretch(0)
        self.sub = row_label("", "stat"); self.sub.setFont(ui_font(12)); self.sub.setWordWrap(True)
        self.sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addStretch(1); v.addLayout(r1); v.addWidget(self.sub); v.addStretch(1)
        h.addLayout(v, 1)
        self.view = None; self.clickable = False; self._look = None
        self.setFixedHeight(86)                     # the list and two lines under it fit; a fixed height never moves the list

    def set_profiles(self, names, current, enabled=True):
        """The bed profiles in the list, the one in use (None: none picked), and whether picking is allowed now."""
        items = tuple(names or ())
        if (items, current) != (self._items, self._current):
            self._items, self._current = items, current
            self.pick.blockSignals(True); self.pick.clear()
            for n in items: self.pick.addItem(n, n)
            if current is None or current not in items: self.pick.insertItem(0, "Pick a bed", None)
            self.pick.addItem(LEARN_NEW, LEARN_NEW)
            self.pick.setCurrentIndex(max(0, self.pick.findData(current)) if current in items else 0)
            self.pick.blockSignals(False)
        self.pick.show(); self.pick.setEnabled(bool(enabled))

    def _picked(self, i):
        data = self.pick.itemData(i)
        cur = self.pick.findData(self._current) if self._current in (self._items or ()) else 0
        if data == LEARN_NEW: self.pick.setCurrentIndex(max(0, cur)); self.learn_new.emit(); return
        if data is None or data == self._current: return
        self.profile_picked.emit(str(data))

    def set_view(self, view, info, shown=False):
        """view: ui_model.bed_card's dict; info: the BedInfo (None: none learned); shown: the bed is shown big."""
        self.view = view; self.clickable = bool(view["clickable"])
        if self.sub.text() != view["sub"]: self.sub.setText(view["sub"]); self.sub.setToolTip(view["sub"])
        learning = view["kind"] == "learning"
        self.thumb.set_bed(info, view["deg"])
        look = (bool(shown), learning)
        if look != self._look:
            self._look = look
            self.setProperty("picked", "true" if shown else "false"); self.setProperty("live", "true" if learning else "false")
            self.live.setVisible(learning); repolish(self)
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.clickable else Qt.CursorShape.ArrowCursor)
        self.setToolTip(("Back to the live camera" if shown else "Show the learned bed big") if self.clickable else "")

    def set_narrow(self, narrow):
        self.thumb.setFixedSize(52, 52) if narrow else self.thumb.setFixedSize(64, 64)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.clickable: self.clicked.emit()


class BedView(QWidget):
    """The bed big, over the model panel (the page places it on the panel's geometry): the platter and its dots, a
    title chip and how to get back."""
    def __init__(self, parent=None):
        super().__init__(parent); self.info = None; self.sub = ""; self.deg = None; self.hide()

    def set_bed(self, info, sub, deg=None):
        """deg: Learn bed's progress; the dots are the map so far and an arc shows how far the table has turned."""
        if (info, sub, deg) != (self.info, self.sub, self.deg): self.info, self.sub, self.deg = info, sub, deg; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath(); path.addRoundedRect(r, 12, 12); p.fillPath(path, qc("well"))
        f = ui_font(13); fm = QFontMetrics(f); top = 12 + fm.height() + 8 + 12 + 24; foot = fm.height() + 24
        area = QRectF(r.x() + 24, r.y() + top, r.width() - 48, r.height() - top - foot)
        if area.width() > 40 and area.height() > 40: draw_bed(p, area, self.info, self.deg, big=True)
        sub = fm.elidedText(self.sub, Qt.TextElideMode.ElideRight, max(0, int(r.width()) - 140))
        chip(p, r.x() + 12, r.y() + 12, [("Learning bed" if self.deg is not None else "Bed map", "platter"), (sub, None)], 13)
        note = "Dots appear as the table turns; the arc is how far it has gone." if self.deg is not None else "Saved top view. Does not follow the live camera."
        p.setFont(f); p.setPen(qc("ink2"))
        p.drawText(QPointF(r.x() + 16, r.y() + top - 10),
                   fm.elidedText(note, Qt.TextElideMode.ElideRight, max(0, int(r.width()) - 32)))
        left = r.x()
        if self.info is not None:
            o = getattr(self.info, "outline", None)
            t = ("Bed %d x %d mm" % (round(o[3]), round(o[4]))) if o and o[0] == "rect" else "Platter %d mm across" % round(2 * self.info.radius)
            left = chip(p, r.x() + 12, r.bottom() - 12 - fm.height() - 8, [(t, None)], 12).right()
        hint = "Stop learning brings the live camera back." if self.deg is not None else "Click the Bed card again for the live camera."
        x = r.right() - 16 - fm.horizontalAdvance(hint)
        if x > left + 16:                                          # a narrow panel: the hint gives way
            p.setFont(f); p.setPen(qc("ink3")); p.drawText(QPointF(x, r.bottom() - 16 - fm.descent()), hint)
        p.setPen(QPen(qc("line"), 1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(path)

