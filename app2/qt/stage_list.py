# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's left column: the info block ("12 sides, 4800 frames", "Camera: Near"), the
stage list (Raw frames, Side, Line-up, Fused, Meshed, Scan here, Textured) with each stage's state on the right, and the
side pager ("< 3 / 5 >") on Raw frames and Side."""
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QFont
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy
from app2.qt.theme import T, qc, ui_font
from app2.qt.widgets import button, repolish, ElidedLabel
from app2.qt import build_words as W


class InfoBlock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(4, 0, 0, 0); v.setSpacing(1); self.lines = []
        for i in range(4):                      # sides/frames, camera + size, counts, sizes on disk (2026-10-24: one line each, no eliding)
            l = ElidedLabel(); l.setFont(ui_font(13, QFont.Weight.DemiBold if i == 0 else QFont.Weight.Normal))
            l.setStyleSheet("color: %s;" % (T["ink"] if i == 0 else T["ink2"])); v.addWidget(l); self.lines.append(l)
        self.narrow = False

    def set_lines(self, lines):
        for k, l in enumerate(self.lines):
            t = lines[k] if k < len(lines) else ""
            l.setText(t); l.setVisible(k == 0 or not self.narrow)     # blank keeps its line: the stage list never shifts

    def set_narrow(self, narrow): self.narrow = narrow; self.set_lines([l.text() for l in self.lines])


class StageRow(QFrame):
    clicked = Signal(str)
    toggled = Signal(str)                       # the group arrow was clicked

    def __init__(self, sid, name, parent=None, disabled=False, tip=None, indent=False, group=False):
        super().__init__(parent); self.sid = sid; self.current = False; self.disabled = disabled
        self.setProperty("role", "stage"); self.setProperty("current", "false")
        self.setCursor(Qt.CursorShape.ArrowCursor if disabled else Qt.CursorShape.PointingHandCursor)
        g = QGridLayout(self); g.setContentsMargins(16 + (18 if indent else 0), 6 if indent else 9, 10, 6 if indent else 9); g.setHorizontalSpacing(6); g.setVerticalSpacing(0)
        self.name = QLabel(name); self.name.setProperty("role", "stagename"); self.name.setProperty("dim", "true" if disabled else "false")
        self.name.setFont(ui_font(13 if indent else 15, QFont.Weight.DemiBold))
        self.sub = QLabel("") if sid in ("lineup","meshed") else ElidedLabel("")
        if sid in ("lineup","meshed"):
            self.sub.setWordWrap(True); self.sub.setMinimumWidth(0)
            self.sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.sub.setProperty("role", "stagesub"); self.sub.setFont(ui_font(11))
        self.state = QLabel(""); self.state.setFont(ui_font(11, QFont.Weight.DemiBold))
        self.name.setMinimumHeight(self.name.sizeHint().height()); self.state.setMinimumHeight(self.name.sizeHint().height())
        g.addWidget(self.name, 0, 0); g.addWidget(self.state, 0, 1, Qt.AlignmentFlag.AlignRight); g.addWidget(self.sub, 1, 0, 1, 2)
        g.setColumnStretch(0, 1)
        self.arrow = None; self.open = True
        if group:                                       # a group head: its arrow folds and unfolds the steps under it
            self.arrow = QLabel("\u25be"); self.arrow.setFont(ui_font(14)); self.arrow.setFixedWidth(22); self.arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.arrow.setToolTip("Show or hide the steps"); g.addWidget(self.arrow, 0, 2)
        if tip: self.setToolTip(tip)
        self.state_key = None; self.indent = indent
        if indent: self.sub.hide()                       # step rows are one line; what the sub line said is the tooltip
        self.setMinimumHeight(self.sizeHint().height())     # a row never squashes into the next

    def set_current(self, on):
        if on == self.current: return
        self.current = on; self.setProperty("current", "true" if on else "false"); repolish(self)
        self.name.setStyleSheet("color: %s;" % T["accent_hi"] if on else ""); self.update()

    def set_info(self, sub, state_text="", colour=None, tip=None, dim=False):
        self._sub_text = sub or ""
        self._update_sub()
        if self.indent and not self.disabled: self.setToolTip(sub or "")
        key = (state_text, colour, dim)
        if key != self.state_key:
            self.state_key = key
            self.state.setText(("●  " if colour and state_text and colour not in ("ink3",) else "") + (state_text or ""))
            self.state.setStyleSheet("color: %s;" % T[colour or "ink3"])
            if not self.disabled:
                self.name.setProperty("dim", "true" if dim else "false"); repolish(self.name)
                if self.current: self.name.setStyleSheet("color: %s;" % T["accent_hi"])
        self.state.setToolTip(tip or "")

    def _update_sub(self):
        text = getattr(self, "_sub_text", self.sub.text())
        if getattr(self, "compact", False) and self.sid in ("lineup", "meshed"):
            self.sub.setText(self.sub.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, max(20, self.sub.width())))
            self.sub.setToolTip(text)
        else: self.sub.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event); self._update_sub()

    def set_open(self, on):
        self.open = bool(on)
        if self.arrow is not None: self.arrow.setText("\u25be" if on else "\u25b8")

    def mousePressEvent(self, e):
        if self.disabled or e.button() != Qt.MouseButton.LeftButton: return
        if self.arrow is not None and self.arrow.geometry().adjusted(-8, -8, 8, 8).contains(e.position().toPoint()): self.toggled.emit(self.sid); return
        self.clicked.emit(self.sid)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.current:                                   # a lit edge on the current row, in our accent
            p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("accent")); p.drawRoundedRect(QRectF(0, 6, 4, self.height() - 12), 2, 2)


class StageList(QFrame):
    picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "panel")
        v = QVBoxLayout(self); v.setContentsMargins(6, 8, 6, 8); v.setSpacing(6); self.rows = {}
        self.children_of = {}; self._user_open = {}
        for s in W.STAGES:
            r = StageRow(s["id"], s["row"], disabled=bool(s.get("disabled")), tip=s["sub"] if s.get("disabled") else None, indent=bool(s.get("indent")),
                         group=bool(s.get("group")))
            r.clicked.connect(self.picked.emit); r.toggled.connect(self._toggle); v.addWidget(r); self.rows[s["id"]] = r
            if s.get("in_group"): self.children_of.setdefault(s["in_group"], []).append(s["id"])
            if s.get("disabled"): r.set_info(s["sub"])
        for gid in self.children_of:
            self.rows[gid].set_open(False)
            for k in self.children_of[gid]: self.rows[k].setVisible(False)
        v.addStretch(1)

    def set_compact(self, compact):
        if getattr(self, "compact", None) == compact: return
        self.compact = compact
        self.layout().setSpacing(2 if compact else 6)
        self.layout().setContentsMargins(6, 4 if compact else 8, 6, 4 if compact else 8)
        for row in self.rows.values():
            row.compact = compact
            row.setMinimumHeight(0)
            row.layout().setContentsMargins(16 + (18 if row.indent else 0), 3 if compact else (6 if row.indent else 9),
                                            10, 3 if compact else (6 if row.indent else 9))
            if row.sid in ("lineup", "meshed"):
                row.sub.setWordWrap(not compact)
                row.sub.setMaximumHeight(row.sub.fontMetrics().height() if compact else 16777215)
                row._update_sub()
            row.setMinimumHeight(row.sizeHint().height())
        current = next((key for key, row in self.rows.items() if row.current), None)
        if current: self.set_current(current)
        self.updateGeometry()

    def _toggle(self, gid):
        head = self.rows.get(gid)
        if head is None: return
        on = not head.open; head.set_open(on); self._user_open[gid] = on
        for k in self.children_of.get(gid, []): self.rows[k].setVisible(on)

    def set_current(self, sid):
        for k, r in self.rows.items(): r.set_current(k == sid)
        group = next((s["in_group"] for s in W.STAGES if s["id"] == sid and s.get("in_group")), sid if sid in self.children_of else None)
        if group is not None and not self._user_open.get(group):          # the group you are in opens, the others fold (unless you opened one)
            for gid in self.children_of:
                on = gid == group or bool(self._user_open.get(gid)); self.rows[gid].set_open(on)
                for k in self.children_of[gid]: self.rows[k].setVisible(on)

        if getattr(self, "compact", False):
            for gid, children in self.children_of.items():
                if not self._user_open.get(gid):
                    self.rows[gid].set_open(False)
                    for key in children: self.rows[key].setVisible(key == sid)


class Pager(QWidget):
    """"Side / < 3 / 5 >" only when inspecting an individual captured side."""
    step = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        self.lab = QLabel(W.BOTTOM["pager_label"]); self.lab.setProperty("role", "hint"); self.lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h = QHBoxLayout(); h.setSpacing(4)
        self.prev = button("‹", "quiet", "Previous side (Left)"); self.next = button("›", "quiet", "Next side (Right)")
        for b in (self.prev, self.next): b.setFont(ui_font(20)); b.setFixedSize(40, 36)
        self.text = QLabel(""); self.text.setFont(ui_font(16, QFont.Weight.DemiBold)); self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text.setFixedWidth(150)                     # the arrows stay put as the words change ("6 / 12 (left out)"): clicking on keeps hitting them
        self.note = QLabel(""); self.note.setProperty("role", "hint"); self.note.setAlignment(Qt.AlignmentFlag.AlignCenter); self.note.setFixedHeight(14)
        h.addStretch(1); h.addWidget(self.prev); h.addWidget(self.text); h.addWidget(self.next); h.addStretch(1)
        self.all = QLabel(W.BOTTOM["pager_all"]); self.all.setProperty("role", "hint"); self.all.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source = QWidget(); source_layout = QVBoxLayout(self.source); source_layout.setContentsMargins(0, 0, 0, 0); source_layout.setSpacing(0)
        source_layout.addWidget(self.lab); source_layout.addLayout(h); source_layout.addWidget(self.note)
        from PySide6.QtWidgets import QStackedLayout
        self.pages = QStackedLayout(); self.pages.addWidget(self.source); self.pages.addWidget(self.all); v.addLayout(self.pages)
        self.prev.clicked.connect(lambda: self.step.emit(-1)); self.next.clicked.connect(lambda: self.step.emit(1))
        self.setFixedHeight(72); self.set_page(0, 0, False)

    def set_page(self, k, n, paged, note=""):
        paged = bool(paged) and n > 0                       # no sides (no project open): nothing to page through, never "1 / 0"
        self.setVisible(paged)
        self.pages.setCurrentWidget(self.source if paged else self.all)
        if paged:
            self.text.setText(W.BOTTOM["pager"].format(k=k, n=n)); self.note.setText(note or ""); self.prev.setEnabled(k > 1); self.next.setEnabled(k < n)
