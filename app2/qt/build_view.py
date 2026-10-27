# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's centre column: the 3D view (EditView, or a picture when there is no OpenGL) with its chips ("Fused:
Standard 0.4 mm", "Red: would go"), the Compare and Home view buttons, the count note, the progress card
("Fusing the scan" / "Working" / "YOLO Yoink", not modal: the model can still be turned), Compare's
Before / After split, and under the view the Rebuild bar, the notice line and the Rectangle / Lasso buttons with
Through and the selection bar ("5120 polygons selected  Delete  Invert  Clear")."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QEvent
from PySide6.QtGui import QPainter, QFont, QFontMetrics, QColor, QAction
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QProgressBar, QTextBrowser, QSizePolicy, QMenu, QStackedLayout)
from app2.qt.theme import T, qc, ui_font
from app2.qt.widgets import ImageTile, button, chip, ElidedLabel, to_qimage, fit_parts, fit_width, keep_room, StableLabel
from app2.qt import build_words as W


class ChipLayer(QWidget):
    """Chips over the GL view (top left) and the count note (bottom right); transparent to the mouse."""
    def __init__(self, parent):
        super().__init__(parent); self.parts = []; self.note = None; self.center = None; self.max_w = None; self.note_lift = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); r = QRectF(self.rect())
        if self.parts: chip(p, 10, 10, fit_parts(self.parts, self.max_w or self.width() - 20, 12), 12)
        if self.note:
            f = ui_font(11); fm = QFontMetrics(f); w = fm.horizontalAdvance(self.note) + 16
            chip(p, r.right() - 10 - w, r.bottom() - 10 - fm.height() - 8 - self.note_lift, [(self.note, None)], 11)
        if self.center:
            f = ui_font(14); p.setFont(f); p.setPen(qc("ink3"))
            p.drawText(r.adjusted(24, 40, -24, -40), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.center)


class ViewPane(QWidget):
    """One view: EditView (gl) or an ImageTile, with its chips."""
    def __init__(self, gl, parent=None):
        super().__init__(parent); self.gl = gl
        lay = QStackedLayout(self); lay.setStackingMode(QStackedLayout.StackingMode.StackAll)
        if gl:
            from app2.qt.edit_view import EditView
            self.view = EditView(); lay.addWidget(self.view)
            self.chips = ChipLayer(self); lay.addWidget(self.chips); self.chips.raise_()
        else:
            self.view = None; self.tile = ImageTile([], radius=12); lay.addWidget(self.tile); self.chips = None
        self.setMinimumSize(120, 90); self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_note_lift(self, h):
        """The count note moves up by h (the float buttons sit along the bottom of a narrow view)."""
        t = self.chips if self.gl else self.tile
        if t.note_lift != h: t.note_lift = h; t.update()

    def set_label_room(self, w):
        """The label's room (it gives way to the float buttons over the view)."""
        if self.gl:
            if self.chips.max_w != w: self.chips.max_w = w; self.chips.update()
        elif self.tile.max_w != w: self.tile.max_w = w; self.tile.update()

    def set_chips(self, parts, note=None, center=None):
        if self.gl: self.chips.parts = parts; self.chips.note = note; self.chips.center = center; self.chips.update()
        else:
            self.tile.parts = list(parts); self.tile.set_note(note); self.tile.set_center(center); self.tile.update()

    def set_image(self, img):
        if not self.gl: self.tile.set_image(img)


CARD_W, CARD_GLOBE = 640, 260                      # one size for the progress card everywhere (Scan step, Build, YOLO Yoink); the globe shrinks to fit a short view


def globe_side(view_h):
    return int(min(CARD_GLOBE, max(100, view_h - 52 - 20 - 400)))                  # 400 = the card's text, log, bar and button


class ProgressCard(QFrame):
    stop = Signal()
    status = Signal(str, float)            # a one-line summary and the fraction (-1 = unknown) for the top bar's compact strip
    pause = Signal(bool)                   # Pause (True) / Resume (False) the build
    docked_changed = Signal(bool)          # True: the card went up to the top bar, False: it is back or the build ended

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("ProgressCard"); self.docked = False
        v = QVBoxLayout(self); v.setContentsMargins(18, 12, 18, 12); v.setSpacing(6)
        self.title = QLabel(""); self.title.setFont(ui_font(17, QFont.Weight.DemiBold)); self.title.setAlignment(Qt.AlignmentFlag.AlignCenter); self.title.setMinimumHeight(30)
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(8); self.bar.setRange(0, 1000)
        self.bar.setStyleSheet("QProgressBar {background: #2a3038; border: none; border-radius: 4px;} "
                               "QProgressBar::chunk {background: %s; border-radius: 4px;}" % T["accent"])
        self.l1 = QLabel(""); self.l1.setWordWrap(True); self.l1.setFont(ui_font(15, QFont.Weight.DemiBold)); self.l1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.l2 = ElidedLabel(""); self.l2.setFont(ui_font(12)); self.l2.setProperty("role", "hint"); self.l2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.words = None    # words(node id) -> the step in plain words (the page's)
        self.cached = ElidedLabel(""); self.cached.setProperty("role", "hint")
        self.overall = QLabel(""); self.overall.setProperty("role", "hint"); self.overall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.doing = ElidedLabel(""); self.doing.setProperty("role", "hint")
        h = QHBoxLayout(); self.elapsed = QLabel(""); self.elapsed.setProperty("role", "stat"); self.pct = QLabel(""); self.pct.setProperty("role", "value"); self.pct.setFixedWidth(110)
        self.stop_b = button(W.PROGRESS["stop"], "stop", "Stop: nothing half-built is kept"); self.holding = False; self._hold_why = None
        self.stop_b.clicked.connect(lambda: self.close_failed() if self.holding else self.stop.emit())
        fit_width(self.stop_b, [W.PROGRESS["stop"], W.PROGRESS["stopping"]])
        self.pause_b = button("Pause", "quiet", "Freeze the build here and go on later (it keeps its memory while paused)"); self.paused = False
        fit_width(self.pause_b, ["Pause", "Resume"]); self.pause_b.clicked.connect(self._toggle_pause)
        h.addWidget(self.elapsed); h.addStretch(1); h.addWidget(self.pause_b); h.addWidget(self.stop_b)
        self.pct.setMinimumWidth(0); self.pct.setMaximumWidth(16777215); self.pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        from app2.qt.build_animation import GlobeBuild
        self.anim = GlobeBuild(132)                       # the point globe that builds the logo; runs only while the card is shown
        self._onetap = False
        self.size_b = button("\u2013", "quiet", "Minimize to the top bar"); self.size_b.setFixedSize(30, 28); self.collapsed = False
        self.size_b.clicked.connect(self._toggle_size)
        trow = QHBoxLayout(); trow.setContentsMargins(0, 0, 0, 0); trow.addSpacing(30); trow.addWidget(self.title, 1); trow.addWidget(self.size_b)
        v.addLayout(trow); v.addWidget(self.anim, 0, Qt.AlignmentFlag.AlignHCenter)
        for w in (self.overall, self.cached): w.setFixedHeight(24)
        self.l1.setFixedHeight(28); self.l2.setFixedHeight(20)   # the headline (the step, plain) and under it what it is doing now ("frame 635 of 635")
        self.doing.hide()                                  # "Running: Side and Line-up": stage names a first-timer can't use
        self.log = QTextBrowser(); self.log.setFixedHeight(150); self.log.setFont(ui_font(12)); self.log.setFrameShape(QFrame.Shape.NoFrame)
        self.log.setOpenLinks(False); self.log.setToolTip("What the build has done so far"); self.log.document().setDocumentMargin(2)
        self.log.setStyleSheet("QTextBrowser {background: #1a1f25; border: 1px solid #2c333c; border-radius: 6px; padding: 4px 6px; color: %s;}" % T["ink2"])
        self._events = []; self._live = ""; self._seen = {}; self._rows = {}; self._shown = None
        v.addSpacing(4); v.addWidget(self.l1); v.addWidget(self.l2)
        v.addWidget(self.bar)                                    # the bar runs the full width; the % sits under it, centered
        line = QHBoxLayout(); line.setSpacing(8); line.addStretch(1); line.addWidget(self.pct); line.addWidget(self.overall); line.addStretch(1)
        v.addLayout(line)
        for w in (self.log, self.cached): v.addWidget(w)
        v.addLayout(h); self.hide()

    def _toggle_size(self):
        """Minimize: the card leaves the view and a compact strip in the top bar carries the progress (click it to bring the card back)."""
        self.docked = True; QWidget.hide(self); self.docked_changed.emit(True)

    def _toggle_pause(self):
        self.paused = not self.paused; self.pause_b.setText("Resume" if self.paused else "Pause"); self.pause.emit(self.paused)
        if self.paused: self.pct.setText("Paused"); self.status.emit("Paused  \u00b7  " + self.title.text(), -2.0)

    def set_docked(self, on):
        """Follow another card's minimize or restore (one build shows on the Scan step and the Build page: one state for both)."""
        if on == self.docked: return
        self.docked = on
        if on: QWidget.hide(self)
        elif self.isVisible() or self.title.text(): self.show(); self.raise_()

    def undock(self):
        self.docked = False; self.docked_changed.emit(False); self.show(); self.raise_()
        p = self.parentWidget()
        while p is not None and not hasattr(p, "_place"): p = p.parentWidget()
        if p is not None: p._place()

    def show_failed(self, why, elapsed):
        """A build that failed keeps its card up, with the reason in red at the end of the log and a Close button, so the log can be read
        (it used to vanish with the build, leaving one short notice)."""
        if getattr(self, "holding", False) and self._hold_why == why: return
        self.holding = True; self._hold_why = why
        self.add_failure(why, elapsed); self.pct.setText("Failed"); self.pause_b.setVisible(False)
        self.stop_b.setText("Close"); self.stop_b.setEnabled(True); self.status.emit("Failed  \u00b7  " + self.title.text(), -2.0)
        if not self.docked: self.show(); self.raise_()

    def close_failed(self):
        self.holding = False; self._hold_why = None; self.pause_b.setVisible(True); self.stop_b.setText(W.PROGRESS["stop"]); self.hide()

    def hide(self):
        """The build is over (or idle): a docked card's strip goes too, and the log starts over for the next run. A failed build's card
        stays up until its Close (holding)."""
        if getattr(self, "holding", False): return
        self._events = []; self._live = ""; self._seen = {}; self._rows = {}; self._shown = None; self.log.clear()
        if self.docked: self.docked = False; self.docked_changed.emit(False); self.status.emit("", -1.0)
        if getattr(self, "paused", False): self.paused = False; self.pause_b.setText("Pause")          # a build that ended is not paused any more
        super().hide()

    def _stamp(self, elapsed):
        m, sec = divmod(int(elapsed), 60); return "%d:%02d" % (m, sec)

    LOG_MARK = {"start": ("\u2026", "ink3"), "done": ("\u2713", "kept"), "warn": ("!", "warn"), "check": ("!", "warn"), "failed": ("\u2717", "lost"), "note": ("i", "ink3")}

    def _log_nodes(self, nodes, elapsed):
        """One log row per step (steps already built are not listed): (time, mark, title, note). A running step's row keeps
        its detail and percent up to date, so sides building side by side each show where they are."""
        if not nodes: return
        from app2.builder.stagehost import step_words
        for key, n in nodes.items():
            nid = n.get("id", key); st = n.get("state"); prev = self._seen.get(nid)
            if st == prev and st not in ("start", "warn", "check"): continue
            self._seen[nid] = st
            if prev is None and st in ("stale", "waiting", "cached", "off", "done", "warn", "check"): continue     # the plan's starting state: steps already built are not this run's
            if st not in self.LOG_MARK: continue
            title = (self.words(nid) if self.words else "") or step_words(n) or nid; note = ""
            if st == "done": note = "done in %d s" % round(n["secs"]) if n.get("secs") else "done"
            elif st == "start":
                note = n.get("text") or "started"
                lead = "side %s " % n.get("n")                  # "Side 4: Fusion  frame 210 of 635", not "... side 4 frame ..."
                if n.get("n") and note.lower().startswith(lead): note = note[len(lead):]
                if n.get("frac") is not None: note += " (%d%%)" % round(100 * float(n["frac"]))
            elif st in ("warn", "check"):
                note = "done, with a warning" if st == "warn" else "done, needs a look"
                for x in (n.get("warn") or []): note += "\n\u2022 " + W.plain_warning(x)[:300]          # one bullet per warning, plainly
            elif st == "failed": note = "FAILED\n" + str(n.get("error") or "no reason given")[:600]
            if st == "warn":
                from app2.builder.backend_notes import expected_cpu_warning
                warnings = n.get("warn") or []
                if warnings and all(expected_cpu_warning(x) for x in warnings):
                    st = "note"; note = "done on the CPU (CUDA is unavailable)"
            row = (self._stamp(elapsed), st, title, note)
            if nid in self._rows: self._events[self._rows[nid]] = row
            else: self._rows[nid] = len(self._events); self._events.append(row)

    def add_failure(self, why, elapsed):
        """The run ended on an error that no step reported (the engine's own message): one red row at the end."""
        row = (self._stamp(elapsed), "failed", "Build", "FAILED\n" + str(why)[:600])
        if not self._events or self._events[-1] != row: self._events.append(row)
        self._render_log()

    def _log_line(self, what, elapsed, nodes=None):
        """The last row is the live one: what runs now plus the engine's own detail ("lining up side 19 with side 3") and the step's percent.
        Steps that report their own rows already show that, so there is no extra live row."""
        if nodes: self._live = ""; return
        detail = next((n.get("text") for n in reversed(list((nodes or {}).values())) if n.get("state") == "start" and n.get("text")), "")
        if detail and (detail in what or what in detail): detail = detail if len(detail) > len(what) else ""     # the same words twice: say them once
        if detail and detail in what: detail = ""
        part = next((n.get("frac") for n in reversed(list((nodes or {}).values())) if n.get("state") == "start" and n.get("frac") is not None), None)
        self._live = ("%s%s%s" % (what, (": " + detail) if detail else "", (" (%d%%)" % round(100 * float(part))) if part is not None else "")) if what else ""

    def _render_log(self):
        from html import escape
        def row(stamp, mark, color, title, note, strong=False):
            lines = str(note).split("\n") if note else [""]
            head = "<tr><td width=34 style='color:%s'>%s</td><td width=14 style='color:%s'><b>%s</b></td><td><b>%s</b> %s</td></tr>" % (
                T["ink3"], escape(stamp), color, escape(mark), escape(title), escape(lines[0]))
            more = "".join("<tr><td></td><td></td><td style='color:%s'>%s</td></tr>" % (color if mark in ("!", "\u2717") else T["ink3"], escape(x)) for x in lines[1:] if x)
            return head + more
        rows = [row(st, *self.LOG_MARK[kind][:1], T[self.LOG_MARK[kind][1]], title, note) for st, kind, title, note in self._events[-150:]]
        if self._live: rows.append("<tr><td width=34></td><td width=14 style='color:%s'><b>&#9654;</b></td><td style='color:%s'>%s</td></tr>" % (T["accent_hi"], T["ink"], escape(self._live)))
        html = "<table cellspacing=0 cellpadding=1 width='100%%'>%s</table>" % "".join(rows)
        if html == self._shown: return
        self._shown = html; bar = self.log.verticalScrollBar(); was = bar.value(); follow = was >= bar.maximum() - 4
        self.log.setHtml(html)                                     # resets the scroll: put it back where it was, or follow the newest row
        bar.setValue(bar.maximum() if follow else min(was, bar.maximum()))

    def show_state(self, title, what, params, frac, elapsed, cached_n, stopping=False, percent_label="Build", overall="", doing="", nodes=None):
        if self.holding:                                   # a new build starts over the failed one's card
            self.holding = False; self._hold_why = None; self.pause_b.setVisible(True); self.stop_b.setText(W.PROGRESS["stop"])
            self._events = []; self._live = ""; self._seen = {}; self._rows = {}; self._shown = None; self.log.clear()
        if sum(n.get("state") == "start" for n in (nodes or {}).values()) > 1:          # several steps at once: one grouped headline
            from app2.builder.progress_words import headline
            what = headline(nodes, self.words)
        self._onetap = title == W.PROGRESS["onetap"]; self.big = False                # the same size for every build
        self.size_b.setVisible(True); self.title.setFont(ui_font(17, QFont.Weight.DemiBold))
        from app2.builder.progress_words import headline
        current_step = headline(nodes, self.words) or (what if what and not what.lower().startswith("starting") else "")
        self.title.setText(current_step or title)
        if current_step and (not what or what.lower().startswith("starting")): what = current_step
        if frac is None: self.bar.setRange(0, 0)
        else: self.bar.setRange(0, 1000); self.bar.setValue(int(1000 * max(0.0, min(1.0, frac))))
        self._log_nodes(nodes, elapsed); self._log_line(what, elapsed, nodes); self._render_log()
        self.l1.setText("" if what and what == (current_step or title) else (what or W.PROGRESS["starting"] + "..."))   # not the title twice
        self.l2.setText(params if params and params.lower() not in (what or "").lower() else "")   # not the headline twice
        self.cached.setText(W.PROGRESS["cached"].format(n=cached_n, s="" if cached_n == 1 else "s") if cached_n else "")
        self.overall.setText(overall)
        m, s = divmod(int(elapsed), 60); self.elapsed.setText(W.PROGRESS["elapsed"].format(m=m, s=s))
        self.pct.setText("Working..." if frac is None else "%d%%" % int(100 * frac))
        line = "%s%s" % (title, "  \u00b7  " + overall if overall else (("  \u00b7  " + what) if what else ""))
        if self.paused: self.pct.setText("Paused"); line = "Paused  \u00b7  " + line
        self.status.emit(line, -1.0 if frac is None else float(frac))
        self.stop_b.setText(W.PROGRESS["stopping"] if stopping else W.PROGRESS["stop"]); self.stop_b.setEnabled(not stopping)
        if not self.docked: self.show(); self.raise_()


class Bar(QFrame):
    """The Rebuild bar (info blue, with its button) and the notice line (amber). Both float over the bottom of the view
    (the view never resizes when one comes or goes; it did, and the model jumped)."""
    clicked = Signal()
    changed = Signal()

    def __init__(self, name, button_text=None, tip=None, parent=None):
        super().__init__(parent); self.setObjectName(name)
        h = QHBoxLayout(self); h.setContentsMargins(12, 5, 6, 5); h.setSpacing(10)
        self.text = QLabel(""); self.text.setWordWrap(True); self.text.setFont(ui_font(13, QFont.Weight.DemiBold))
        self.text.setStyleSheet("color: %s;" % (T["accent_hi"] if name == "RebuildBar" else T["warn"]))
        self.text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if name != "RebuildBar": self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)        # a notice reads centered in its pill
        h.addWidget(self.text, 1)
        self.btn = None
        if button_text:
            self.btn = button(button_text, "build", tip); self.btn.clicked.connect(self.clicked.emit); h.addWidget(self.btn)
        self.hide()

    def show_text(self, text):
        was = (self.isVisible(), self.text.text())
        if not text: self.hide()
        else: self.text.setText(text); self.show(); self.raise_()
        if was != (self.isVisible(), self.text.text()): self.changed.emit()

    def height_for(self, w):
        lay = self.layout()
        return lay.totalHeightForWidth(w) if lay.hasHeightForWidth() else self.sizeHint().height()


class RoundTool(QWidget):
    """A readable selection button in the existing tool strip's reserved slot."""
    def __init__(self, glyph, name, tip, parent=None):
        super().__init__(parent)
        # Keep the old circle plus caption height so the scene does not move.
        self.setFixedHeight(44 + 1 + QFontMetrics(ui_font(11)).height())
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        self.b = button(name, "selectionTool", tip); self.b.setFont(ui_font(13)); self.b.setCheckable(True)
        font = QFont(self.b.font()); font.setBold(True)
        self.b.setFixedSize(max(78, QFontMetrics(font).horizontalAdvance(name) + 24), 36)
        self.b.setAccessibleName(name); self.b.setAccessibleDescription(tip)
        self.b.setStyleSheet("QPushButton {padding: 4px 10px; border: 1px solid #424b57; border-radius: 6px; background: #22272e; color: #d8e0ea;} "
                             "QPushButton:hover {background: #303a48;} "
                             "QPushButton:checked {background: #274a78; border-color: #639feb; color: white; font-weight: 700;} "
                             "QPushButton:disabled {color: #737e8b; border-color: #303741; background: #1b2026;}")
        v.addStretch(1); v.addWidget(self.b, 0, Qt.AlignmentFlag.AlignHCenter); v.addStretch(1)


class BuildViewPanel(QWidget):
    compare_clicked = Signal()
    home_clicked = Signal()
    home_set_clicked = Signal()          # menu: use the view as it is now as the Home view
    home_reset_clicked = Signal()
    stop_clicked = Signal()
    rebuild_clicked = Signal()
    pause_clicked = Signal(bool)
    tool_picked = Signal(str)          # "orbit", "rect", "lasso"
    through_toggled = Signal(bool)
    delete_clicked = Signal()
    invert_clicked = Signal()
    clear_clicked = Signal()
    before_picked = Signal(str)        # "tool", "stage", "version"
    before_version_picked = Signal(str)   # key16 of a version listed in the Before menu
    marked_only_changed = Signal(bool)
    projection_changed = Signal(str)
    colour_picked = Signal(str)        # "shaded", "sides", "flat"

    def __init__(self, gl, parent=None):
        super().__init__(parent); self.gl = gl; self.comparing = False
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        self.area = QWidget(); self.area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.area.installEventFilter(self)                   # the view area also resizes when the rows under it change
        av = QVBoxLayout(self.area); av.setContentsMargins(0, 0, 0, 0); av.setSpacing(6)
        self.top_row = QWidget(); th = QHBoxLayout(self.top_row); th.setContentsMargins(0, 0, 0, 0); th.setSpacing(6)      # Line-up Points: the base and the moving side
        self.pa = ViewPane(gl); self.pb = ViewPane(gl); th.addWidget(self.pa, 1); th.addWidget(self.pb, 1); self.top_row.hide()
        self.bottom_row = QWidget(); ah = QHBoxLayout(self.bottom_row); ah.setContentsMargins(0, 0, 0, 0); ah.setSpacing(6)
        self.before = ViewPane(gl); self.before.hide(); self.main = ViewPane(gl)
        ah.addWidget(self.before, 1); ah.addWidget(self.main, 1)
        av.addWidget(self.top_row, 1); av.addWidget(self.bottom_row, 1)
        v.addWidget(self.area, 1)
        self.compare = button(W.VIEW["compare"], "float", "Before and after, side by side (C). Hold B to see before.")
        self.home = button(W.VIEW["home"], "float", "Look at the whole result (H)")
        self.before_btn = button(W.VIEW["before"] + "  ▾", "float", "What to compare with")
        from app2.qt.view_modes import ViewModes
        self.colour_btn = ViewModes(self); self.colour_menu = QMenu(self)
        if gl: self.main.view.marked_only_changed.connect(self._marked_only_changed)
        for b in (self.compare, self.home, self.before_btn, self.colour_btn): b.setParent(self.area)
        self.before_btn.hide()
        self.guide_controls = QWidget(self.area)
        row = QHBoxLayout(self.guide_controls); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        self.frame_part = button("Frame part", "float", "Fit the part in the view")
        self.frame_everything = button("Frame everything", "float", "Fit the part, scanners, and view cones")
        self.guide_ghosts = button("Ghosts and cones", "float", "Show suggested scanner spots and what each one would see")
        self.guide_ghosts.setCheckable(True); self.guide_ghosts.setChecked(True)
        for control in (self.frame_part, self.frame_everything, self.guide_ghosts): row.addWidget(control)
        self.guide_controls.hide()
        from app2.qt.scan_guide import frame_view
        self.frame_part.clicked.connect(lambda: gl and frame_view(self.view))
        self.frame_everything.clicked.connect(lambda: gl and frame_view(self.view, True))
        self.guide_ghosts.toggled.connect(self._show_guide_ghosts)
        self.card = ProgressCard(self.area); self.card.stop.connect(self.stop_clicked.emit); self.card.pause.connect(self.pause_clicked.emit)
        self.empty = QLabel("", self.area); self.empty.setWordWrap(True); self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.empty.setFont(ui_font(15))
        self.empty.setStyleSheet("color: #8b95a1;"); self.empty.hide()
        self.compare.clicked.connect(self.compare_clicked.emit); self.home.clicked.connect(self.home_clicked.emit)
        menu = self.before_menu = QMenu(self.before_btn)
        self.before_change_action = menu.addAction("Before this change")
        self.before_change_action.setEnabled(False)
        self.before_change_action.triggered.connect(lambda: self.before_picked.emit("change"))
        menu.addSeparator()
        self.before_actions = {}
        for k, text in W.VIEW["compare_menu"][:2]:
            a = menu.addAction(text); a.setCheckable(True); a.triggered.connect(lambda _=False, x=k: self.before_picked.emit(x)); self.before_actions[k] = a
        self.before_versions_sep = menu.addSeparator()
        self.before_version_actions = []                      # the last few versions, by date (set_before_versions)
        k, text = W.VIEW["compare_menu"][2]
        self.before_all_versions = menu.addAction(text); self.before_all_versions.triggered.connect(lambda _=False, x=k: self.before_picked.emit(x))
        self.before_btn.clicked.connect(lambda: menu.exec(self.before_btn.mapToGlobal(self.before_btn.rect().bottomLeft())))
        self.rebuild = Bar("RebuildBar", W.REBUILD_BAR["button"], W.REBUILD_BAR["model_note"]); self.rebuild.clicked.connect(self.rebuild_clicked.emit)
        self.notice = Bar("NoticeBar")
        for b in (self.rebuild, self.notice): b.setParent(self.area); b.hide(); b.changed.connect(self._place)
        # the float buttons whose text changes are as wide as their longest text (the others do not slide along)
        fit_width(self.compare, [W.VIEW["compare"] + " ✓"], 28)
        tools = QHBoxLayout(); tools.setSpacing(14); tools.setContentsMargins(0, 0, 0, 0)
        self.rect_t = RoundTool("□", W.BOTTOM["rect"], "Rectangle (R): drag over what to select; Ctrl takes away")
        self.lasso_t = RoundTool("◯", W.BOTTOM["lasso"], "Lasso (L): draw round what to select; Ctrl takes away")
        self.rect_t.b.clicked.connect(lambda on: self.tool_picked.emit("rect" if on else "orbit"))
        self.lasso_t.b.clicked.connect(lambda on: self.tool_picked.emit("lasso" if on else "orbit"))
        self.through = button(W.BOTTOM["through"], "chip", "Also select what is hidden behind (T)"); self.through.setCheckable(True)
        self.through.toggled.connect(self.through_toggled.emit)
        self.sel_label = QLabel(""); self.sel_label.setProperty("role", "value")
        bf = QFont(self.sel_label.font()); bf.setBold(True)       # room for a 7-digit count: Delete / Invert never slide
        self.sel_label.setMinimumWidth(QFontMetrics(bf).horizontalAdvance(W.BOTTOM["selected"].format(count=8888888, what="polygons")) + 4)
        self.sel_delete = button(W.BOTTOM["sel_delete"], "stop", "Delete selected faces (Del). Undo restores them.")
        self.sel_invert = button(W.BOTTOM["sel_invert"], None, "Select everything else")
        self.sel_clear = button(W.BOTTOM["sel_clear"], "quiet", "Clear selection (Esc). The model is not changed.")
        self.sel_delete.clicked.connect(self.delete_clicked.emit); self.sel_invert.clicked.connect(self.invert_clicked.emit)
        self.sel_clear.clicked.connect(self.clear_clicked.emit)
        # the selection bar sits next to the tools when it fits, else on its own line under them (1280 x 720)
        self.sel_w = QWidget(); sh = QHBoxLayout(self.sel_w); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(8)
        for w in (self.sel_label, self.sel_delete, self.sel_invert, self.sel_clear): sh.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        sh.addStretch(1)
        self.tools_note = StableLabel("", 2); self.tools_note.setProperty("role", "hint"); self.tools_note.setMinimumWidth(120)
        # the selection bar and the "tools are off because..." note share one slot (never both): the slot is as big as
        # the bigger of them whichever shows, so a select or a note never moves the tools or resizes the view
        self.slot = QWidget(); self.slot_lay = QStackedLayout(self.slot); self.slot_lay.addWidget(self.sel_w); self.slot_lay.addWidget(self.tools_note)
        # no minimum width of its own: when the row is too narrow for it, _flow puts it on the line under the tools
        self.slot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        tools.addStretch(1); tools.addWidget(self.rect_t); tools.addWidget(self.lasso_t); tools.addWidget(self.through, 0, Qt.AlignmentFlag.AlignVCenter)
        tools.addSpacing(10); self._sel_at = tools.count(); tools.addWidget(self.slot, 3, Qt.AlignmentFlag.AlignVCenter); tools.addStretch(1)
        self.tools_row = tools
        self.sel_row_w = QWidget(); self.sel_row = QHBoxLayout(self.sel_row_w); self.sel_row.setContentsMargins(0, 0, 0, 0)
        self.sel_row.addStretch(1); self.sel_row.addStretch(1); self.sel_row_w.hide()
        self.tools_w = QWidget(); tv = QVBoxLayout(self.tools_w); tv.setContentsMargins(0, 0, 0, 0); tv.setSpacing(4)
        tv.addLayout(tools); tv.addWidget(self.sel_row_w); v.addWidget(self.tools_w)
        # tools, their note and the selection bar keep their room when hidden (a stage without tools, no selection):
        # the view above them keeps its height
        for w in (self.rect_t, self.lasso_t, self.through, self.sel_label, self.sel_delete, self.sel_invert, self.sel_clear): keep_room(w)
        self.set_selection(0, "polygons")
        keep_room(self.slot); self._tools_show = True; self.set_edit_workspace(False)

    def set_edit_workspace(self, enabled):
        """Both workspaces use the full scene height; Edit owns selection in its palette."""
        self.edit_workspace = bool(enabled)
        self.tools_w.hide()
        for widget in (self.rect_t, self.lasso_t, self.through): widget.setVisible(self.edit_workspace and self._tools_show)
        self.slot.setVisible(self.edit_workspace)

    def _flow(self):
        """The selection bar beside Rectangle / Lasso / Through when the row has room, else on a line of its own."""
        # by the width alone (not whether something is selected): the second line is there or not before any select
        need = sum(w.sizeHint().width() for w in (self.rect_t, self.lasso_t, self.through, self.sel_w)) + 14 * 4 + 10
        below = need > self.width()
        in_row2 = self.sel_row.indexOf(self.slot) >= 0
        if below and not in_row2: self.tools_row.removeWidget(self.slot); self.sel_row.insertWidget(1, self.slot, 3)
        elif not below and in_row2:
            self.sel_row.removeWidget(self.slot); self.tools_row.insertWidget(self._sel_at, self.slot, 3, Qt.AlignmentFlag.AlignVCenter)
        self.sel_row_w.setVisible(below)

    # ---------------- state in ----------------
    @property
    def view(self): return self.main.view

    def set_tool(self, t):
        self.rect_t.b.setChecked(t == "rect"); self.lasso_t.b.setChecked(t == "lasso")

    def set_tools(self, on, note="", show=True):
        """Rectangle / Lasso / Through usable (on) or not (note says why); show=False hides the row's tools."""
        self._tools_show = show
        for w in (self.rect_t, self.lasso_t, self.through): w.setVisible(show and getattr(self, "edit_workspace", False)); w.setEnabled(on)
        self.tools_note.setText("" if on else (note or ""))
        self.slot_lay.setCurrentWidget(self.tools_note if (note and not on) else self.sel_w)

    def set_selection(self, count, what):
        on = count > 0
        self.sel_label.setText(W.BOTTOM["selected"].format(count=count, what=what) if on else "")
        for w in (self.sel_label, self.sel_delete, self.sel_invert, self.sel_clear): w.setVisible(on)   # their room stays
        self._flow()

    def _refresh_marked_action(self):
        action = getattr(self, '_marked_action', None)
        if action is None: return
        view = self.main.view
        action.setEnabled(view.would_go is not None and bool(np.any(view.would_go)))
        action.setChecked(view.marked_only)
        label = 'Marked only' if view.marked_only else getattr(self, '_colour_label', W.VIEW['colour'])
        self.colour_btn.setText(label)
        checkbox=self.colour_btn.marked
        checkbox.blockSignals(True);checkbox.setEnabled(action.isEnabled());checkbox.setChecked(view.marked_only);checkbox.blockSignals(False)

    def _marked_only_changed(self, enabled):
        self._refresh_marked_action(); self.marked_only_changed.emit(enabled)

    def set_projection_mode(self, mode):
        if not self.gl: return
        for pane in (self.main, self.before): pane.view.set_projection_mode(mode)
        for action in self.colour_menu.actions():
            if str(action.data()).startswith('projection:'):
                action.setChecked(action.data() == 'projection:' + mode)
        self.colour_btn.projection.blockSignals(True);self.colour_btn.projection.setCurrentIndex(self.colour_btn.projection.findData(mode));self.colour_btn.projection.blockSignals(False)
        self.projection_changed.emit(mode)

    def set_colours(self, modes, current):
        """modes: [(key, text)] for this stage (none: the button hides)."""
        options = tuple(modes)
        if options != getattr(self, '_colour_options', None):
            self.colour_menu.close()
            self.colour_menu.clear(); self._colour_options = options
            for key, text in options:
                action = self.colour_menu.addAction(text)
                action.setData(key); action.setCheckable(True)
                action.triggered.connect(lambda _=False, mode=key: self.colour_picked.emit(mode))
            if self.gl:
                self.colour_menu.addSeparator()
                for mode, text in [('perspective', 'Perspective'), ('orthographic', 'Orthographic')]:
                    action = self.colour_menu.addAction(text)
                    action.setData('projection:' + mode); action.setCheckable(True)
                    action.triggered.connect(lambda _=False, value=mode: self.set_projection_mode(value))
                self.colour_menu.addSeparator()
                self._marked_action = self.colour_menu.addAction('Show marked pieces only')
                self._marked_action.setData('marked-only'); self._marked_action.setCheckable(True)
                self._marked_action.triggered.connect(self.main.view.set_marked_only)
                self.colour_menu.addSeparator()
                set_home = self.colour_menu.addAction("Set this view as Home view"); set_home.triggered.connect(self.home_set_clicked.emit)
                reset_home = self.colour_menu.addAction("Reset Home view"); reset_home.triggered.connect(self.home_reset_clicked.emit)


        for action in self.colour_menu.actions():
            if str(action.data()).startswith('projection:'):
                action.setChecked(action.data() == 'projection:' + self.main.view.cam.projection_mode)
            elif action.data() != "marked-only": action.setChecked(action.data() == current)
        self.colour_btn.set_modes(modes,current)
        self.colour_btn.setVisible(bool(modes)); self.colour_btn.more.setVisible(bool(modes) and self.gl)
        self._colour_label = dict(modes).get(current, W.VIEW["colour"])
        self.colour_btn.setText(self._colour_label + "  \u25be"); self._refresh_marked_action(); self._place()

    def set_before_choice(self, kind, text):
        """The Before button names what it compares with ("Before: all tools ▾", "Before: Oct 22 20:05 ▾") and the menu
        ticks it."""
        self.before_btn.setText("%s: %s  ▾" % (W.VIEW["before"], text))
        for k, a in self.before_actions.items(): a.setChecked(k == kind)
        for a in self.before_version_actions: a.setChecked(kind == "version" and a.text().startswith(text))
        self.before_change_action.setCheckable(True); self.before_change_action.setChecked(kind == "change")
        if self.before_btn.isVisible(): self.before_btn.adjustSize(); self._place()

    def set_before_versions(self, rows):
        """rows: [(key16, text)] newest first, listed in the Before menu so an earlier version is one click away."""
        for a in self.before_version_actions: self.before_menu.removeAction(a)
        self.before_version_actions = []
        for k16, text in rows:
            a = QAction(text, self.before_menu); a.setCheckable(True); a.setData(text)
            a.triggered.connect(lambda _=False, k=k16: self.before_version_picked.emit(k))
            self.before_menu.insertAction(self.before_all_versions, a); self.before_version_actions.append(a)
        self.before_all_versions.setText(W.VIEW["compare_menu"][2][1] if not rows else W.VIEW["all_versions"])

    def set_points_mode(self, on):
        """Line-up Points: two small views on top (the base and the moving side, where you click), the lined-up result below (the main view)."""
        self.top_row.setVisible(bool(on)); self._place()

    def set_comparing(self, on):
        self.comparing = on; self.before.setVisible(on); self.before_btn.setVisible(on)
        self.compare.setText(W.VIEW["compare"] + (" ✓" if on else ""))
        self._place()

    def set_empty(self, text):
        """A plain line in the middle of the view while there is nothing to show (not built yet, or building)."""
        self.empty.setText(text or ""); self.empty.setVisible(bool(text)); self._place()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._place(); self._flow()

    def eventFilter(self, obj, e):
        if obj is self.area and e.type() == QEvent.Type.Resize: self._place()
        return False

    def showEvent(self, e):
        super().showEvent(e); QTimer.singleShot(0, self._place)

    def _place(self):
        """Float buttons along the bottom of the view like a slicer's view bar: Home
        view and the View drop-down bottom left above the nav cube, the Look row and Compare bottom right; the notice top
        right, the progress card top centre."""
        a = self.area
        view_b = self.colour_btn.more
        left_row = [b for b in (self.home, view_b) if not b.isHidden()]
        x = 10
        for b in left_row:
            b.adjustSize(); b.move(x, a.height() - b.height() - 10); x += b.width() + 8; b.raise_()
        left_edge_used = x
        room = a.width() - left_edge_used - self.compare.sizeHint().width() - 30
        self.colour_btn.fit_width(max(1, room))
        right_row = [b for b in (self.compare, self.colour_btn) if not b.isHidden()]
        x = a.width() - 10
        for b in right_row:
            b.adjustSize(); b.move(x - b.width(), a.height() - b.height() - 10); x = b.x() - 8; b.raise_()
        shown = left_row + right_row
        row_top = min([b.y() for b in shown] or [a.height()])
        top_right = 10                                       # Frame part / Frame everything / Ghosts: top right, the notice under them
        if not self.guide_controls.isHidden():
            self.guide_controls.adjustSize()
            self.guide_controls.move(max(10, a.width() - self.guide_controls.width() - 10), 10)
            self.guide_controls.raise_(); top_right = self.guide_controls.y() + self.guide_controls.height() + 8
        taken = max((max(220, min(420, a.width() - 24)) + 8) if not self.notice.isHidden() else 0,
                    (self.guide_controls.width() + 8) if not self.guide_controls.isHidden() else 0)        # top right, beside the label
        self.main.set_label_room(max(80, a.width() - self.main.x() - 22 - taken))
        self.before.set_label_room(max(80, self.before.width() - 20))
        lift = (a.height() - row_top + 8) if shown else 0    # the count note sits above everything along the bottom
        # the Rebuild bar lives in the page header now (a pill beside the project name); here it only keeps its state.
        self.rebuild.setGeometry(0, 0, 0, 0)
        # the notice floats top right, under the guide controls when those show
        bw = max(220, min(420, a.width() - 24))
        x0 = a.width() - bw - 10; y = top_right
        for b in (self.notice,):
            if b.isHidden(): continue
            h = b.height_for(bw); b.setGeometry(max(8, x0), y, bw, h); b.raise_(); y += h + 6
        bars = 0
        self.main.set_note_lift(lift); self.before.set_note_lift(0)             # the bars float over the bottom of the view; nothing is pushed up
        if self.gl:
            for pane in (self.main, self.before):
                cube = pane.view.nav_cube
                clearance = 0
                left_edge = pane.x() + pane.view.x()
                for control in shown:                                # the notice and rebuild bars overlap the view, they do not lift the cube
                    if control.isHidden() or control.y() < a.height() / 2: continue
                    if control.x() < left_edge + cube.width() and control.x() + control.width() > left_edge:
                        clearance = max(clearance, a.height() - control.y() + 6)
                # The key spans the pane, including the narrow view's bottom
                # controls. Lift it above that row and keep the cube above it.
                key_inset=0
                for control in shown:
                    if control.y()<a.height()/2:continue
                    if control.x()<left_edge+pane.view.width() and control.x()+control.width()>left_edge:
                        key_inset=max(key_inset,a.height()-control.y()+pane.view.fontMetrics().lineSpacing()//2)
                pane.view.capture_key_inset=key_inset
                if getattr(pane.view,'capture_colors',False):
                    from app2.qt.capture_key import layout
                    clearance=max(clearance,key_inset+int(layout(pane.view.width(),pane.view.height())[0].height())+pane.view.fontMetrics().lineSpacing())
                pane.view.overlay.update()
                cube.set_bottom_inset(clearance)

        if self.empty.isVisible():
            w_ = min(420, max(160, a.width() - 60)); self.empty.setFixedWidth(w_); self.empty.adjustSize()
            self.empty.move((a.width() - w_) // 2, (a.height() - self.empty.height()) // 2); self.empty.raise_()
        if self.before_btn.isVisible():
            self.before_btn.adjustSize(); self.before_btn.move(10, 44); self.before_btn.raise_()

        compact = self.window().height() <= 900
        w = min(420 if compact else CARD_W, max(240, a.width() - 40)); self.card.setFixedWidth(w)
        self.card.layout().setSpacing(4 if compact else 6)
        self.card.log.setFixedHeight(72 if compact else 150)
        self.card.cached.setVisible(not compact and bool(self.card.cached.text()))
        side = 56 if compact else globe_side(a.height())
        self.card.anim.setFixedSize(side, side); self.card.adjustSize()
        self.card.move(10 if compact else (a.width() - w) // 2, 52 if compact else max(52, (a.height() - self.card.height()) // 2))
        if self.card.isVisible(): self.card.raise_()

    def _show_guide_ghosts(self, visible):
        if self.gl:
            self.view.guide_ghosts = visible
            self.view.overlay.update(); self.view.update()
