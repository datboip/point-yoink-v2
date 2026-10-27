# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's popovers: History (every change, "now" at the cursor; a click goes back or forward to it) and
Versions (the stored results of a stage: when, what, how many; Compare with one, Keep it from the cache limit)."""
import time
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QVBoxLayout, QScrollArea, QWidget
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button, row_label, MovablePanelTitle, PanelResizeGrip, fit_overlay_in_parent
from app2.qt import build_words as W
from app2.qt.tool_panel import hint_label


def version_row(v):
    """(when, "Oct 22 20:05    Watertight 0.18 mm    250000 polygons") for one version entry; the Before menu uses it too."""
    when = time.strftime("%b %d %H:%M", time.localtime(v.get("created") or 0))
    st = v.get("stats") or {}
    count = ("%d polygons" % st["faces"]) if st.get("faces") else ("%d points" % st["points"]) if st.get("points") else ""
    prm = v.get("params") or {}
    m = v.get("method") or ""
    summary = "%s %s" % (METHOD_WORDS.get(m, m), ("%g mm" % prm["point_distance"]) if "point_distance" in prm else "")
    return when, W.VERSIONS["row"].format(when=when, summary=summary.strip(), count=count)


class _Popover(QFrame):
    closed = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent); self.setObjectName("Popover"); self.setFixedWidth(460); self._user_placed = False
        v = QVBoxLayout(self); v.setContentsMargins(16, 12, 12, 12); v.setSpacing(6); self.container_layout = v
        h = QHBoxLayout(); self.title = MovablePanelTitle(title, self); self.title.setFont(ui_font(15, QFont.Weight.Bold)); h.addWidget(self.title, 1)
        x = button("✕", "quiet", "Close (Esc)"); x.clicked.connect(self.close_popover); h.addWidget(x); v.addLayout(h)
        self.sa = QScrollArea(); self.sa.setWidgetResizable(True); self.inner = QWidget(); self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(0, 0, 4, 0); self.lay.setSpacing(1); self.sa.setWidget(self.inner); v.addWidget(self.sa, 1)
        self.hide()

    def _clear(self):
        while self.lay.count():
            it = self.lay.takeAt(0); w = it.widget()
            if w: w.setParent(None); w.deleteLater()

    def close_popover(self): self.hide(); self.closed.emit()

    def place(self, anchor_global, max_h):
        """Above its button, inside the window."""
        self.setFixedHeight(min(max_h, 80 + 34 * max(1, self.lay.count())))
        if not self._user_placed:
            p = self.parentWidget().mapFromGlobal(anchor_global)
            x = max(8, min(p.x() - self.width() + 40, self.parentWidget().width() - self.width() - 8))
            y = max(8, p.y() - self.height() - 8)
            self.move(x, y)
        fit_overlay_in_parent(self); self.show(); self.raise_()


METHOD_WORDS = {"tsdf": "Standard", "points": "Points", "as_fused": "As scanned", "poisson": "Watertight", "bpa": "Joined points"}


def plain_note(note):
    """An engine history note in the page's words: preset keys as their chip names, method names as the chips say."""
    import re
    names = {k: lab for k, lab, _h in W.ONETAP["presets"]}
    names.update(auto="Auto Yoink", custom="Custom")
    note = str(note or "")
    note = re.sub(r"^(Custom )?[Pp]reset: ([\w-]+)$", lambda m: ("Custom settings, started from " if m.group(1) else "Preset: ") + names.get(m.group(2), m.group(2)), note)
    note = re.sub(r"\bpreset ([\w-]+)", lambda m: 'preset "%s"' % names.get(m.group(1), m.group(1)), note)
    return re.sub(r"(method |-> |=)(tsdf|points|as_fused|poisson|bpa)(?![.\w])", lambda m: m.group(1) + METHOD_WORDS[m.group(2)], note)


class HistoryPopover(_Popover):
    go = Signal(int)            # steps: -3 = undo three times, +1 = redo once

    def __init__(self, parent=None): super().__init__(W.BOTTOM["history"], parent)

    def set_entries(self, entries, cursor):
        self._clear()
        for k in range(len(entries) - 1, -1, -1):
            e = entries[k]; now = k == cursor
            b = button(("%s   %s" % (plain_note(e.get("note")), "(" + W.BOTTOM["history_now"] + ")" if now else "")).strip(), "quiet",
                       "Go back to here" if k < cursor else "Go forward to here" if k > cursor else "")
            b.setStyleSheet("text-align: left; %s" % ("color: %s; font-weight: 700;" % T["accent_hi"] if now else
                                                     "color: %s;" % T["ink3"] if k > cursor else ""))
            b.clicked.connect(lambda _=False, s=k - cursor: self.go.emit(s) if s else None); self.lay.addWidget(b)
        self.lay.addStretch(1)


class BuildLogPopover(_Popover):
    """The last build's engine line, bar, result and full output (the Scan step's builds too): what the one-shot Build page
    showed, one click away ("Log" in the footer)."""
    def __init__(self, parent=None):
        super().__init__("Build log", parent)
        self.setMinimumSize(340, 220); self.setMaximumSize(16777215, 16777215); self.resize(620, 520)
        from PySide6.QtWidgets import QProgressBar
        from app2.qt.widgets import log_edit
        self.engine = row_label("", "stat"); self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(10)
        self.bar.setRange(0, 1000); self.pct = row_label("", "value"); self.step = row_label("", "stat"); self.result = row_label("", "value")
        self.result.setWordWrap(True); self.log = log_edit(2000)
        for w in (self.engine, self.bar, self.step, self.pct, self.result): self.lay.addWidget(w)
        self.lay.addWidget(self.log, 1)
        footer = QHBoxLayout(); footer.addStretch(1); self.resize_grip = PanelResizeGrip(self); footer.addWidget(self.resize_grip)
        self.container_layout.addLayout(footer)

    def place(self, anchor_global, max_h):
        if not self._user_placed:
            self.resize(self.width(), min(max_h, 520)); p = self.parentWidget().mapFromGlobal(anchor_global)
            self.move(max(8, min(p.x() - self.width() + 40, self.parentWidget().width() - self.width() - 8)), max(8, p.y() - self.height() - 8))
        fit_overlay_in_parent(self)
        self.show(); self.raise_()


class VersionsPopover(_Popover):
    compare = Signal(str)       # key16
    keep = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__("", parent); self.setFixedWidth(600)
        self.sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_versions(self, stage, versions, current_key):
        self.title.setText(W.VERSIONS["title"].format(stage=stage)); self._clear()
        self.lay.addWidget(hint_label(W.VERSIONS["help"]))
        if not versions: self.lay.addWidget(hint_label(W.NOTICE["no_versions"]))
        for v in versions:
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 2, 0, 2); h.setSpacing(6)
            l = QLabel(version_row(v)[1]); l.setFont(ui_font(12)); l.setWordWrap(True)
            h.addWidget(l, 1)
            cur = current_key and str(current_key).startswith(v["key16"])
            if cur:
                c = QLabel(W.VERSIONS["current"]); c.setProperty("role", "badge-auto"); c.setAlignment(Qt.AlignmentFlag.AlignCenter)
                c.setFixedWidth(92); h.addWidget(c)
            else:
                b = button(W.VERSIONS["compare"], "quiet", "Compare this version with the one in use"); b.clicked.connect(lambda _=False, k=v["key16"]: self.compare.emit(k))
                b.setFixedWidth(92); h.addWidget(b)
            pinned = bool(v.get("pinned"))
            k_ = button(W.VERSIONS["unkeep"] if pinned else W.VERSIONS["keep"], "quiet", W.VERSIONS["keep_tip"]); k_.setFixedWidth(104)
            k_.clicked.connect(lambda _=False, k=v["key16"], on=not pinned: self.keep.emit(k, on)); h.addWidget(k_)
            self.lay.addWidget(w)
        self.lay.addStretch(1)
