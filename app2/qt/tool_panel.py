# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's control widgets: the slider block (label, reset arrow, track, value, hint, the small factory-value tick),
chips, the on/off switch, XYZ steppers, the tab bar with its "..." tools, the More settings disclosure and the
"Applied to" list. Looks only: each takes a control dict from build_words and plain values, and says what the user did
through signals. Nothing here talks to the engine. Every button is Qt.NoFocus (keys belong to the page)."""
import math
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QFont
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QMenu, QSizePolicy, QComboBox,
                               QStackedWidget, QDoubleSpinBox)
from app2.qt.theme import T, qc, ui_font, INPUT_QSS
from app2.qt.widgets import Track, button, row_label, repolish, StableLabel, ElidedLabel
from app2.qt import build_words as W
from app2.qt.help_text import HelpButton, help_for

EXTRA_QSS = """
QFrame[role="stage"] { background: transparent; border: none; border-radius: 8px; }
QFrame[role="stage"][current="true"] { background: #1f2a3a; }
QFrame[role="stage"]:hover { background: #22272e; }
QFrame[role="stage"][current="true"]:hover { background: #22304a; }
QLabel[role="stagename"] { color: %(ink)s; }
QLabel[role="stagename"][dim="true"] { color: %(ink3)s; }
QLabel[role="stagesub"] { color: %(ink3)s; }
QPushButton[kind="tab"] { background: transparent; border: none; border-radius: 8px; padding: 7px 3px; color: %(ink2)s; }
QPushButton[kind="tab"]:hover { background: %(raised)s; }
QPushButton[kind="tab"]:checked { color: %(ink)s; background: #274a78; font-weight: 700; }
QPushButton[kind="tab"]:disabled { color: %(ink3)s; }
QPushButton[kind="round"] { background: %(raised)s; border: 1px solid %(line)s; border-radius: 22px; color: %(ink)s; }
QPushButton[kind="round"]:checked { background: #1f3048; border-color: %(accent)s; color: %(accent_hi)s; }
QPushButton[kind="round"]:disabled { color: %(ink3)s; background: %(panel)s; }
QPushButton[kind="onetap"] { background: %(kept)s; color: %(on_color)s; border: none; border-radius: 10px; font-weight: 700; }
QPushButton[kind="onetap"]:hover { background: %(kept_hi)s; }
QPushButton[kind="onetap"]:disabled { background: %(raised)s; color: %(ink3)s; }
QPushButton[kind="apply"] { background: %(accent)s; color: %(on_color)s; border: none; font-weight: 700; border-radius: 9px; }
QPushButton[kind="apply"]:hover { background: %(accent_hi)s; }
QPushButton[kind="apply"]:disabled { background: %(panel)s; color: %(ink3)s; border-color: %(line)s; }
QPushButton[kind="apply"][quiet="true"] { background: #1f3048; color: %(accent_hi)s; border: 1px solid #33527c; }
QPushButton[kind="apply"][quiet="true"]:hover { background: #274063; }
QPushButton[kind="delete"] { background: #3b1c1b; color: %(lost)s; border: 1px solid #7a2e2a; font-weight: 700; border-radius: 9px; }
QPushButton[kind="delete"]:hover { background: #4a2220; }
QPushButton[kind="delete"]:disabled { background: %(panel)s; color: %(ink3)s; border-color: %(line)s; }
QPushButton[kind="more"] { background: transparent; border: none; color: %(ink2)s; text-align: left; padding: 4px 0; }
QPushButton[kind="more"]:hover { color: %(ink)s; }
QPushButton[kind="reset"] { background: transparent; border: none; color: %(ink3)s; padding: 0 4px; }
QPushButton[kind="reset"]:hover { color: %(ink)s; }
QComboBox { background: %(raised)s; border: 1px solid %(line)s; border-radius: 7px; padding: 4px 10px; }
QComboBox QAbstractItemView { background: %(raised)s; selection-background-color: %(accent)s; }
QFrame#Popover { background: %(panel)s; border: 1px solid %(line)s; border-radius: 12px; }
QFrame#ProgressCard { background: #1a1d22; border: 1px solid %(line)s; border-radius: 12px; }
QFrame#RebuildBar { background: #1c2a3d; border: 1px solid #33527c; border-radius: 8px; }
QFrame#RebuildPill { background: #1c2a3d; border: 1px solid #33527c; border-radius: 8px; }
QFrame#NoticeBar { background: #3a2f17; border: 1px solid #6a5528; border-radius: 8px; }
QFrame#NoticeBar[level="info"] { background: #1c2a3d; border: 1px solid #33527c; }
QMenu { background: %(raised)s; border: 1px solid %(line)s; }
QMenu::item { padding: 5px 18px; }
QMenu::item:selected { background: %(raised_hi)s; }
""" % T + INPUT_QSS


def fmt_value(c, v):
    """A control's value as the panel prints it ("0.4", "15%", "6.0")."""
    if v is None: return ""
    if c.get("fmt") == "k":                                  # a count: 250k, 1M
        v = float(v); return ("%gM" % (v / 1e6)) if v >= 1e6 else ("%gk" % (v / 1e3)) if v >= 1e3 else "%g" % v
    f = c.get("fmt") or "{:g}"
    try: s = f.format(v)
    except (ValueError, TypeError): s = str(v)
    return s + ("%" if c.get("unit") == "%" else "")


def sample_text(t):
    """A template hint with long numbers in it: the room it needs once the page fills it in."""
    if not t or "{" not in t: return ""
    import re
    t = t.replace("{now}", next(b for b in W.PANELS["meshed"]["blocks"] if b["name"] == "Finish for print")["now"])
    return re.sub(r"\{[^}]*\}", "1234567", t)


def big_button(text, kind="apply", h=44):
    b = button(text, kind); b.setMinimumHeight(h); b.setFont(ui_font(15, QFont.Weight.Bold)); return b


def hline():
    f = QFrame(); f.setFixedHeight(1); f.setStyleSheet("background: %s;" % T["line"]); return f


def hint_label(text="", px=12):
    h = QLabel(text); h.setProperty("role", "hint"); h.setWordWrap(True); h.setFont(ui_font(px)); return h


class CurrentStack(QWidget):
    """Pages in a column, only the current one shown: as tall as that page and no taller. (A QStackedWidget kept the
    height of the tallest page shown before, so a short page floated in empty space or pushed Apply below the fold.)
    Same calls as QStackedWidget: addWidget, setCurrentWidget, currentWidget, currentChanged."""
    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent); self._pages = []; self._cur = None
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def addWidget(self, w):
        self._pages.append(w); self.lay.addWidget(w, 0, Qt.AlignmentFlag.AlignTop)
        if self._cur is None: self._cur = w
        else: w.hide()
        return len(self._pages) - 1

    def setCurrentWidget(self, w):
        if w not in self._pages: return
        changed = w is not self._cur; self._cur = w
        for p in self._pages:
            if p is not w: p.hide()
        w.show(); self.updateGeometry()
        if changed: self.currentChanged.emit(self._pages.index(w))

    def currentWidget(self): return self._cur
    def count(self): return len(self._pages)
    def widget(self, i): return self._pages[i] if 0 <= i < len(self._pages) else None
    def indexOf(self, w): return self._pages.index(w) if w in self._pages else -1


class Dot(QWidget):
    """The small "changed, not applied yet" dot after a label."""
    def __init__(self, colour="accent", parent=None):
        super().__init__(parent); self.colour = colour; self.setFixedSize(10, 10); self.hide()
        self.setToolTip("Changed: press the panel's button to use it")

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(self.colour)); p.drawEllipse(QRectF(1, 1, 8, 8))


class Control(QWidget):
    """Common part of a control row: engine value vs pending value, the changed dot."""
    moved = Signal(object)

    def __init__(self, c, parent=None):
        super().__init__(parent); self.c = c; self.engine_value = None; self.dot = Dot()

    def changed(self):
        a, b = self.value(), self.engine_value
        if isinstance(a, float) or isinstance(b, float):
            try: return abs(float(a) - float(b)) > 1e-9
            except (TypeError, ValueError): return a != b
        return a != b

    on_dot = None                      # the tab's restyle (its Apply is quiet while nothing is changed)

    def _refresh_dot(self):
        self.dot.setVisible(self.engine_value is not None and self.changed())
        if self.on_dot is not None: self.on_dot()

    def _reset_value(self):
        """Back to what is applied now when it was moved off that; else back to the default (so the arrow always does something)."""
        applied = None if self.engine_value is None else float(self.engine_value)
        if applied is not None and abs(self.value() - applied) > 1e-9: self.set_value(applied * self.scale, emit=True)
        else: self.set_value(self.start, emit=True)


class SliderRow(Control):
    """The slider block: "Label (reset) (extra)" / track ... value / hint. A float slider is the int Track scaled by
    its step; side.keep.min_share (a share) is shown as a percent (x100)."""
    def __init__(self, c, kind="mesh", parent=None):
        super().__init__(c, parent)
        self.scale = 100.0 if c.get("path") == "side.keep.min_share" else 1.0
        self.lo, self.hi, self.step = float(c["lo"]), float(c["hi"]), float(c["step"])
        d = c["default"]; self.default = float(d[kind] if isinstance(d, dict) else d)       # the engine's default
        self.revo = c.get("revo")
        revo_num = isinstance(self.revo, (int, float)) and not isinstance(self.revo, bool)
        # a NEW tool step starts at the factory value (Isolation 15%, Smooth 10 x10, ...); recipe settings show the recipe
        self.start = float(self.revo) if (c.get("tool") and revo_num) else self.default
        v = QVBoxLayout(self); v.setContentsMargins(0, 4, 0, 4); v.setSpacing(1)
        top = QHBoxLayout(); top.setSpacing(4)
        self.label = QLabel(c["label"]); self.label.setFont(ui_font(14))
        rtip = W.RESET_TIP_REVO if (revo_num and abs(float(self.revo) - self.start) < 1e-9) else W.RESET_TIP
        self.reset = button("↺", "reset", rtip.format(default=self._txt(self.start)))
        self.reset.setFont(ui_font(15)); self.reset.clicked.connect(self._reset_value)
        # the extra ("(grid 0.10 mm)", "( about 12.5 MB STL )") fills the rest of the line and ends in "..." when long
        self.extra = ElidedLabel(""); self.extra.setProperty("role", "stat"); self.extra.setFont(ui_font(12))
        top.addWidget(self.label); top.addWidget(HelpButton(c["label"], help_for(c))); top.addWidget(self.reset); top.addWidget(self.dot); top.addWidget(self.extra, 1)
        v.addLayout(top)
        row = QHBoxLayout(); row.setSpacing(10)
        n = self._idx(self.hi)
        # the small tick: the factory value when we start elsewhere, or ours when we start at the factory value
        other = self.default if self.start != self.default else (float(self.revo) if revo_num else None)
        marks = [self._idx(other)] if other is not None and abs(other - self.start) > 1e-9 else []
        self.track = Track(0, n, [self._idx(self.start)], step=1, page=max(1, n // 10), fill="accent", marks=marks, name=c["label"])
        if marks: self.track.setToolTip((W.OURS_TICK_TIP if self.start != self.default else W.REVO_TICK_TIP).format(revo=self._txt(other)))
        self.numeric_factor = 1.0
        self.value_l = QDoubleSpinBox(); self.value_l.setFixedSize(122, 32)
        self.value_l.setFont(ui_font(13)); self.value_l.setKeyboardTracking(False)
        self.value_l.setRange(self.lo, self.hi); self.value_l.setSingleStep(self.step)
        self.value_l.setDecimals(max(0, min(6, len(str(self.step).split(".")[-1]) if not self.step.is_integer() else 0)))
        self.value_l.setSuffix("%" if c.get("unit") == "%" else "")
        self.value_l.setValue(self.start); self.value_l.setAccessibleName(c["label"])
        self.value_l.setToolTip("Type a value, or use the up and down arrows. Apply uses the change.")
        self.value_l.valueChanged.connect(self._number_changed)
        row.addWidget(self.track, 1); row.addWidget(self.value_l); v.addLayout(row)
        # a hint the page fills in ("Keeps this share of the polygons: 1000 to about 400.") keeps the room of its longest
        h = c.get("hint") or ""
        self.hint = StableLabel(h, 1 if h else 0, [h, sample_text(h), c.get("hint_plain")]); self.hint.setProperty("role", "hint")
        self.hint.setFont(ui_font(12)); self.hint.setVisible(bool(h)); v.addWidget(self.hint)
        self.warn = StableLabel("", 0); self.warn.setFont(ui_font(12)); self.warn.setStyleSheet("color: %s;" % T["warn"])
        self.warn.hide(); v.addWidget(self.warn)
        self.track.changed.connect(self._moved)
        self._v = self.start

    def _idx(self, x): return int(round((float(x) - self.lo) / self.step))
    def _of(self, i): return round(self.lo + i * self.step, 9)
    shown_as = None            # the page may show the value in other units (Fill holes: rim edges as mm across the part)

    def _txt(self, x): return self.shown_as(x) if self.shown_as else fmt_value(self.c, x)

    def set_shown_as(self, fn):
        self.shown_as = fn; self.value_l.setToolTip(self._txt(self._v))

    def set_numeric_units(self, factor=1.0, suffix="", decimals=None):
        self.numeric_factor = float(factor)
        box = self.value_l; box.blockSignals(True)
        if decimals is not None: box.setDecimals(decimals)
        box.setRange(self.lo * factor, self.hi * factor)
        box.setSingleStep(self.step * factor); box.setSuffix(suffix)
        box.setValue(self._v * factor); box.blockSignals(False)

    def _sync_number(self):
        self.value_l.blockSignals(True); self.value_l.setValue(self._v * self.numeric_factor); self.value_l.blockSignals(False)

    def _number_changed(self, value):
        raw = value / self.numeric_factor
        if self.step.is_integer() and self.lo.is_integer(): raw = round(raw)
        self.set_value(raw, emit=True)

    def value(self):
        if self.scale != 1.0: return self._v / self.scale
        if float(self.step).is_integer() and float(self.lo).is_integer(): return int(round(self._v))    # whole-number knobs
        return self._v

    def set_engine(self, v):
        """The value the engine has (recipe); the pending value follows it."""
        if v is None: return
        self.engine_value = float(v); self.set_value(float(v) * self.scale if self.scale != 1.0 else float(v)); self._refresh_dot()

    def set_value(self, shown, emit=False):
        shown = min(self.hi, max(self.lo, float(shown)))
        self._v = shown; self.track.set_values([self._idx(shown)]); self._sync_number(); self._refresh_dot()
        if emit: self.moved.emit(self.value())

    def _moved(self, vals):
        self._v = self._of(vals[0]); self._sync_number(); self._refresh_dot(); self.moved.emit(self.value())

    def set_extra(self, text): self.extra.setText(text or "")
    def set_hint(self, text): self.hint.setText(text or ""); self.hint.setVisible(bool(text) or bool(self.hint.alts))
    def set_warn(self, text): self.warn.setText(text or ""); self.warn.setVisible(bool(text) or bool(self.warn.alts))

    def reserve_warn(self, text):
        """The warning's room is kept even while there is none (it comes and goes as the slider moves)."""
        self.warn.set_alts([text]); self.warn.show()


class SegRow(Control):
    """Chips in a row (Fusion method, Mesh method, Cut method) with the picked chip's hint under them."""
    def __init__(self, c, extra=(), parent=None):
        """extra: [(value, text, hint)] for the "*" choice (every other registered method)."""
        super().__init__(c, parent)
        words = c.get("extra_words") or {}                    # {value: (chip text, hint)} over the engine's own label
        self.choices = [x for x in c["choices"] if x[0] != "*"]
        self.choices += [(val, *words.get(val, (text, hint))) for val, text, hint in extra if val not in [x[0] for x in self.choices]]
        v = QVBoxLayout(self); v.setContentsMargins(0, 4, 0, 4); v.setSpacing(5)
        top = QHBoxLayout(); top.setSpacing(4)
        lab = QLabel(c["label"]); lab.setFont(ui_font(14)); top.addWidget(lab); top.addWidget(self.dot); top.addStretch(1); v.addLayout(top)
        h = QHBoxLayout(); h.setSpacing(0); self.btns = {}
        for i, (val, text, hint) in enumerate(self.choices):
            b = button(text, "seg", hint or None); b.setCheckable(True); b.setMinimumHeight(32)
            b.setProperty("pos", "first" if i == 0 else "last" if i == len(self.choices) - 1 else "mid")
            b.clicked.connect(lambda _=False, x=val: self.pick(x, emit=True)); h.addWidget(b, 1); self.btns[val] = b
        v.addLayout(h)
        hints = [x[2] for x in self.choices if x[2]]           # the room of the longest hint: picking never moves what is under
        self.hint = StableLabel("", 0, hints); self.hint.setProperty("role", "hint"); self.hint.setFont(ui_font(12)); v.addWidget(self.hint)
        self.hint.setVisible(bool(hints))
        self._v = c.get("default"); self.pick(self._v)

    def value(self): return self._v

    def pick(self, val, emit=False):
        if val not in self.btns: return
        self._v = val
        for k, b in self.btns.items(): b.setChecked(k == val)
        self.hint.setText(next((x[2] for x in self.choices if x[0] == val), "") or "")
        self._refresh_dot()
        if emit: self.moved.emit(val)

    def set_engine(self, v):
        if v is None: return
        self.engine_value = v; self.pick(v); self._refresh_dot()

    def set_choice(self, val, visible=True, enabled=True, tip=None):
        b = self.btns.get(val)
        if b is None: return
        b.setVisible(visible); b.setEnabled(enabled)
        if tip is not None: b.setToolTip(tip)


class Switch(QWidget):
    toggled = Signal(bool)

    def __init__(self, on=False, parent=None):
        super().__init__(parent); self.on = on; self.setFixedSize(46, 26); self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_on(self, on): self.on = bool(on); self.update()

    def mousePressEvent(self, e):
        if self.isEnabled(): self.on = not self.on; self.update(); self.toggled.emit(self.on)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc("accent") if self.on else qc("raised_hi")); p.drawRoundedRect(QRectF(0, 0, 46, 26), 13, 13)
        p.setBrush(qc("ink")); p.drawEllipse(QPointF(33 if self.on else 13, 13), 10, 10)


class ToggleRow(Control):
    def __init__(self, c, parent=None):
        super().__init__(c, parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 6, 0, 6); v.setSpacing(2)
        h = QHBoxLayout(); l = QLabel(c["label"]); l.setFont(ui_font(14)); h.addWidget(l); h.addWidget(HelpButton(c["label"], help_for(c))); h.addWidget(self.dot); h.addStretch(1)
        self.sw = Switch(bool(c.get("default"))); h.addWidget(self.sw); v.addLayout(h)
        if c.get("hint"): v.addWidget(hint_label(c["hint"]))
        self._v = bool(c.get("default")); self.sw.toggled.connect(self._t)

    def _t(self, on): self._v = on; self._refresh_dot(); self.moved.emit(on)
    def value(self): return self._v

    def set_engine(self, v):
        if v is None: return
        self.engine_value = bool(v); self._v = bool(v); self.sw.set_on(v); self._refresh_dot()


class CutChoiceRow(Control):
    """Cut base: what the last build found (the status line), then two big choices, Cut it for me and Mark it by hand, a Keep the
    base switch, and under Cut it for me the small Where to look chips. One recipe value behind it all (side[].platter.method:
    auto / plate / search / manual / none), so the page reads and sets it like the chips it replaced."""
    FOR_ME = ("auto", "plate", "search")

    def __init__(self, c, parent=None):
        super().__init__(c, parent); w = W.CUT
        v = QVBoxLayout(self); v.setContentsMargins(0, 4, 0, 4); v.setSpacing(6)
        top = QHBoxLayout(); top.setSpacing(4)
        lab = QLabel(c["label"]); lab.setFont(ui_font(14)); top.addWidget(lab); top.addWidget(self.dot); top.addStretch(1); v.addLayout(top)
        self.status = StableLabel("", 2); self.status.setFont(ui_font(13)); v.addWidget(self.status)   # two lines always: the buttons never move
        h = QHBoxLayout(); h.setSpacing(0)
        self.b_me = button(w["forme"][0], "seg", w["forme"][1]); self.b_hand = button(w["hand"][0], "seg", w["hand"][1])
        for b, pos in ((self.b_me, "first"), (self.b_hand, "last")):
            b.setCheckable(True); b.setMinimumHeight(44); b.setFont(ui_font(14, QFont.Weight.DemiBold)); b.setProperty("pos", pos); h.addWidget(b, 1)
        v.addLayout(h)
        self.hint = StableLabel("", 0, [w["forme"][1], w["hand"][1], w["keep"][1]]); self.hint.setProperty("role", "hint"); self.hint.setFont(ui_font(12)); v.addWidget(self.hint)
        kr = QHBoxLayout(); kl = QLabel(w["keep"][0]); kl.setFont(ui_font(13)); kr.addWidget(kl); kr.addStretch(1); self.sw = Switch(False); kr.addWidget(self.sw); v.addLayout(kr)
        self.where_box = QWidget(); wv = QVBoxLayout(self.where_box); wv.setContentsMargins(0, 2, 0, 0); wv.setSpacing(3)
        wl = QLabel(w["where"]); wl.setFont(ui_font(12)); wl.setProperty("role", "hint"); wv.addWidget(wl)
        wh = QHBoxLayout(); wh.setSpacing(0); self.btns = {}
        for i, (val, text, tip) in enumerate(w["where_choices"]):
            b = button(text, "seg", tip); b.setCheckable(True); b.setMinimumHeight(26); b.setFont(ui_font(12))
            b.setProperty("pos", "first" if i == 0 else "last" if i == len(w["where_choices"]) - 1 else "mid")
            b.clicked.connect(lambda _=False, x=val: self.pick(x, emit=True)); wh.addWidget(b, 1); self.btns[val] = b
        wv.addLayout(wh); v.addWidget(self.where_box)
        self.where = c.get("default") if c.get("default") in self.FOR_ME else "auto"      # the last Cut it for me way: the switch and the big button come back to it
        self.b_me.clicked.connect(lambda _=False: self.pick(self.where, emit=True))
        self.b_hand.clicked.connect(lambda _=False: self.pick("manual", emit=True))
        self.sw.toggled.connect(lambda on: self.pick("none" if on else self.where, emit=True))
        self._v = c.get("default") or "auto"; self.pick(self._v)

    def value(self): return self._v

    def pick(self, val, emit=False):
        if val not in self.FOR_ME + ("manual", "none"): return
        self._v = val; me = val in self.FOR_ME
        if me: self.where = val
        self.b_me.setChecked(me); self.b_hand.setChecked(val == "manual"); self.sw.set_on(val == "none")
        for b in (self.b_me, self.b_hand): b.setEnabled(val != "none")
        self.where_box.setVisible(me)
        for k, b in self.btns.items(): b.setChecked(k == val)
        self.hint.setText(W.CUT["keep"][1] if val == "none" else W.CUT["hand"][1] if val == "manual" else W.CUT["forme"][1])
        self._refresh_dot()
        if emit: self.moved.emit(val)

    def set_engine(self, v):
        if v is None: return
        self.engine_value = v; self.pick(v); self._refresh_dot()

    def set_choice(self, val, visible=True, enabled=True, tip=None):
        """The chips' API (the page asks for By hand to be shown): every choice is always there."""

    def set_status(self, text, level=None):
        self.status.setText(text or ""); self.status.setStyleSheet("color: %s;" % T[level] if level else "")


class ComboRow(Control):
    """A dropdown (Line-up's base side: Auto or a side)."""
    def __init__(self, c, choices=(), parent=None):
        super().__init__(c, parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 4, 0, 4); v.setSpacing(4)
        h = QHBoxLayout(); l = QLabel(c["label"]); l.setFont(ui_font(14)); h.addWidget(l); h.addWidget(HelpButton(c["label"], help_for(c))); h.addWidget(self.dot); h.addStretch(1); v.addLayout(h)
        self.box = QComboBox(); self.box.setFocusPolicy(Qt.FocusPolicy.NoFocus); v.addWidget(self.box)
        self.set_choices(choices); self.box.currentIndexChanged.connect(lambda _i: (self._refresh_dot(), self.moved.emit(self.value())))

    def set_choices(self, choices):
        cur = self.value() if self.box.count() else None
        self.box.blockSignals(True); self.box.clear()
        for val, text in choices: self.box.addItem(text, val)
        if cur is not None:
            i = self.box.findData(cur); self.box.setCurrentIndex(max(0, i))
        self.box.blockSignals(False)

    def value(self): return self.box.currentData()

    def set_engine(self, v):
        self.engine_value = v; i = self.box.findData(v)
        self.box.blockSignals(True); self.box.setCurrentIndex(max(0, i)); self.box.blockSignals(False); self._refresh_dot()


class XYZStepper(Control):
    """Move (mm) / Turn (deg): X, Y, Z with - and + (Shift: the big step)."""
    def __init__(self, c, parent=None):
        super().__init__(c, parent)
        self.v = [0.0, 0.0, 0.0]; self.step = float(c["step"]); self.big = float(c["big"]); self.lo, self.hi = float(c["lo"]), float(c["hi"])
        g = QGridLayout(self); g.setContentsMargins(0, 4, 0, 4); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        l = QLabel(c["label"]); l.setFont(ui_font(14)); g.addWidget(l, 0, 0, 1, 4)
        from app2.qt import rotate_gizmo as RG
        self.vals = []
        for k, ax in enumerate("XYZ"):
            tag = QLabel(ax); tag.setFixedWidth(16); tag.setFont(ui_font(14, QFont.Weight.Bold)); tag.setStyleSheet("color: %s;" % RG.COLOURS[k])     # the ring's own color
            m = button("\u2212", None, "Shift: %g" % self.big); p = button("+", None, "Shift: %g" % self.big)
            for b in (m, p): b.setFixedSize(38, 32); b.setFont(ui_font(16))
            val = QLabel("0"); val.setFont(ui_font(14, QFont.Weight.DemiBold)); val.setMinimumWidth(60); val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet("background:#12161b;border:1px solid #2c343e;border-radius:6px;padding:4px 8px;")
            m.clicked.connect(lambda _=False, i=k: self._bump(i, -1)); p.clicked.connect(lambda _=False, i=k: self._bump(i, 1))
            g.addWidget(tag, k + 1, 0); g.addWidget(m, k + 1, 1); g.addWidget(val, k + 1, 2); g.addWidget(p, k + 1, 3); self.vals.append(val)
        g.setColumnStretch(2, 1); self.engine_value = [0.0, 0.0, 0.0]
        self.setMinimumWidth(0)

    def _bump(self, i, sign):
        from PySide6.QtWidgets import QApplication
        big = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.set_value([x + (sign * (self.big if big else self.step) if j == i else 0) for j, x in enumerate(self.v)], emit=True)

    def value(self): return list(self.v)

    def set_value(self, v, emit=False):
        self.v = [round(min(self.hi, max(self.lo, float(x))), 6) for x in v]
        for l, x in zip(self.vals, self.v): l.setText("%g" % x)
        self._refresh_dot()
        if emit: self.moved.emit(self.value())

    def set_engine(self, v): self.engine_value = [float(x) for x in v]; self.set_value(v)


class MoreSettings(QWidget):
    """"More settings" disclosure: closed by default; the advanced controls go in body."""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 2, 0, 0); v.setSpacing(0)
        self.btn = button("▸  " + W.MORE, "more"); self.btn.clicked.connect(lambda: self.set_open(not self.body.isVisible()))
        self.body = QWidget(); self.lay = QVBoxLayout(self.body); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(2)
        v.addWidget(self.btn); v.addWidget(self.body); self.body.hide()

    def set_open(self, on):
        self.body.setVisible(on); self.btn.setText(("▾  " if on else "▸  ") + W.MORE)


class TabBar(QWidget):
    """Fixed-height tool strip with clipped horizontal scrolling and two arrows. labels: {name: text shown} when a
    tab's text differs from the name the page uses for it (the Side panel's "Platter" tab reads "Cut base")."""
    picked = Signal(str)
    HEIGHT = 34
    ARROW = 24

    def __init__(self, names, parent=None, labels=None):
        from PySide6.QtWidgets import QScrollArea
        super().__init__(parent)
        self.labels = dict(labels or {})
        self.btns = {}; self.extra = {}; self._more_items = []; self.current = next(iter(names), None)
        self.setMinimumWidth(0); self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.scroll = QScrollArea(self); self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea {background:transparent;border:0;}")
        self.scroll.viewport().installEventFilter(self)
        self.content = QWidget(); self.scroll.setWidget(self.content)
        self.left = button("‹", "quiet", "Scroll tools left"); self.left.setParent(self)
        self.right = button("›", "quiet", "Scroll tools right"); self.right.setParent(self)
        self.more = self.right
        for arrow in (self.left, self.right):
            arrow.setFont(ui_font(20)); arrow.setFixedSize(self.ARROW, self.HEIGHT)
            arrow.setStyleSheet("padding:0;")
        self.left.clicked.connect(lambda:self._scroll_by(-1)); self.right.clicked.connect(lambda:self._scroll_by(1))
        bar = self.scroll.horizontalScrollBar()
        bar.valueChanged.connect(self._sync_arrows); bar.rangeChanged.connect(self._sync_arrows)
        for name in names: self._add_button(name)
        self._fit()

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(0, self.HEIGHT)

    def minimumSizeHint(self): return self.sizeHint()

    def _add_button(self, name):
        text = self.labels.get(name, name)
        b = button(text, "tab"); b.setParent(self.content); b.setCheckable(True); b.setFont(ui_font(14))
        b.setFixedHeight(self.HEIGHT); b.setToolTip(text)
        b.clicked.connect(lambda _=False, x=name: self.picked.emit(x)); self.btns[name] = b
        b.show(); return b

    @staticmethod
    def _width(button):
        from PySide6.QtGui import QFontMetrics
        font = QFont(button.font()); font.setBold(True)
        return QFontMetrics(font).horizontalAdvance(button.text()) + 24

    def _fit(self):
        x = 0
        for name, button in self.btns.items():
            width = self._width(button); button.setFixedWidth(width)
            button.move(x,0); button.setChecked(name == self.current); x += width+4
        self.content.setFixedSize(max(0,x-4),self.HEIGHT)
        fits = self.content.width() <= self.width()                    # every tab fits: no arrows, the strip takes the whole width
        self.left.setVisible(not fits); self.right.setVisible(not fits)
        self.left.move(0, 0); self.right.move(max(0,self.width()-self.ARROW), 0)
        if fits: self.scroll.setGeometry(0, 0, self.width(), self.HEIGHT)
        else: self.scroll.setGeometry(self.ARROW+2, 0, max(0,self.width()-2*self.ARROW-4), self.HEIGHT)
        self._sync_arrows()

    def _sync_arrows(self, *_args):
        bar = self.scroll.horizontalScrollBar()
        self.left.setEnabled(bar.value() > bar.minimum())
        self.right.setEnabled(bar.value() < bar.maximum())

    def _scroll_by(self, direction):
        bar = self.scroll.horizontalScrollBar()
        bar.setValue(bar.value()+direction*max(32,int(self.scroll.viewport().width()*.7)))

    def _ensure_current(self):
        if self.current in self.btns: self.scroll.ensureWidgetVisible(self.btns[self.current],4,0)
        self._sync_arrows()

    def resizeEvent(self, event):
        super().resizeEvent(event); self._fit(); self._ensure_current()

    def showEvent(self, event):
        super().showEvent(event); self._fit(); self._ensure_current()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.scroll.viewport() and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().x() or event.angleDelta().y()
            if delta: self._scroll_by(-1 if delta > 0 else 1)
            event.accept(); return True
        return super().eventFilter(obj,event)

    def set_more(self, items):
        items = list(items)
        if items == self._more_items: return
        self._more_items = items; available = {name for name,on in items}
        for name in list(self.extra):
            if name not in available: self._remove_extra(name)
        for name, enabled in items:
            self.add_extra(name); self.btns[name].setEnabled(enabled)
        self._fit(); self._ensure_current()

    def add_extra(self, name):
        if name in self.btns: return
        self.extra[name] = self._add_button(name); self._fit()

    def _remove_extra(self, name):
        button = self.extra.pop(name); self.btns.pop(name,None)
        button.hide(); button.deleteLater()

    def drop_extras(self):
        for name in list(self.extra): self._remove_extra(name)
        self._more_items = []; self._fit(); self._ensure_current()

    def show_tab(self, name):
        from PySide6.QtCore import QTimer
        self.current = name
        for key, button in self.btns.items(): button.setChecked(key == name)
        self._ensure_current(); QTimer.singleShot(0,self._ensure_current)

    def set_tab_enabled(self, name, on, tip=None):
        button = self.btns.get(name)
        if button is not None: button.setEnabled(on); button.setToolTip(tip or self.labels.get(name, name))


class AppliedPartsLabel(ElidedLabel):
    """Two short elided lines, breaking only between complete summary parts."""
    def __init__(self,text):
        super().__init__(text);self.setFont(ui_font(13,QFont.Weight.DemiBold))
        self.setFixedHeight(self.fontMetrics().height()*(2 if " · " in text else 1)+4)
    def paintEvent(self,event):
        from PySide6.QtGui import QPainter
        parts=self.text().split(" · ")
        lines=[parts[0]," · ".join(parts[1:])] if len(parts)>1 else parts
        painter=QPainter(self);painter.setFont(self.font());painter.setPen(self.palette().color(self.foregroundRole()))
        metrics=self.fontMetrics()
        for i,text in enumerate(lines):
            shown=metrics.elidedText(text,Qt.TextElideMode.ElideRight,max(1,self.width()))
            painter.drawText(0,2+metrics.ascent()+i*metrics.height(),shown)


class AppliedList(QFrame):
    """Applied operations with a draft of on/off changes rebuilt as one batch."""
    edit = Signal(int)
    toggle = Signal(int)
    remove = Signal(int)
    pending_changed = Signal(object)
    apply_pending = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "panel")
        self.v = QVBoxLayout(self); self.v.setContentsMargins(14, 10, 8, 10); self.v.setSpacing(3)
        self.head = row_label(""); self.v.addWidget(self.head)
        self.body = QVBoxLayout(); self.body.setSpacing(1); self.v.addLayout(self.body)
        self.v.addStretch(1)
        self.rows = []; self.enabled_tools = True; self.pending = {}; self.checks = {}; self.original = {}; self._fingerprint = None; self._rendered_rows = None; self.badges = {}; self.cached = {}
        self.footer = QWidget(); self.footer.setFixedHeight(62)
        fv = QVBoxLayout(self.footer); fv.setContentsMargins(0, 3, 0, 3); fv.setSpacing(2)
        self.plan_note = ElidedLabel(""); self.plan_note.setFixedHeight(20); fv.addWidget(self.plan_note)
        h = QHBoxLayout(); h.setSpacing(6); fv.addLayout(h)
        self.pending_count = ElidedLabel("No changes"); self.pending_count.setMinimumWidth(0)
        self.pending_count.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.apply_b = button("Apply changes", "build", "Apply the checked changes and rebuild once."); self.apply_b.setFixedSize(124, 32)
        self.reset_b = button("Reset", "quiet", "Discard pending changes."); self.reset_b.setFixedSize(56, 32); self.reset_b.setStyleSheet("padding: 0 3px;")
        h.addWidget(self.pending_count, 1); h.addWidget(self.apply_b); h.addWidget(self.reset_b); self.v.addWidget(self.footer)
        self.apply_b.clicked.connect(lambda: self.apply_pending.emit(dict(self.pending)) if self.pending and self.enabled_tools else None)
        self.reset_b.clicked.connect(self.clear_pending); self._update_pending()

    def _update_pending(self):
        count = len(self.pending)
        self.pending_count.setText("%d pending" % count if count else "No changes")
        if not count: self.set_pending_plan("")
        self.apply_b.setEnabled(bool(count) and self.enabled_tools); self.reset_b.setEnabled(bool(count) and self.enabled_tools)

    def set_pending_plan(self, text, cached=False):
        self.plan_note.setText(text)
        self.plan_note.setStyleSheet("color: %s;" % (T["kept"] if cached else T["ink2"]))

    def _update_badges(self):
        changed_upstream = False
        for key, badge in self.badges.items():
            changed_upstream = changed_upstream or key in self.pending
            badge.setText("Cached" if self.cached.get(key) and not changed_upstream else "")
            badge.setToolTip("A saved result is available for this edit. Changing an earlier edit may need a new result.")

    def clear_pending(self):
        changed = bool(self.pending); self.pending = {}
        for key, checkbox in self.checks.items():
            checkbox.blockSignals(True); checkbox.setChecked(self.original[key]); checkbox.blockSignals(False)
        self._update_pending(); self._update_badges()
        if changed: self.pending_changed.emit({})

    def _checked(self, key, checked):
        if checked == self.original[key]: self.pending.pop(key, None)
        else: self.pending[key] = bool(checked)
        self._update_pending(); self._update_badges()
        if self.pending: self.set_pending_plan("Checking changes...")
        self.pending_changed.emit(dict(self.pending))

    def set_rows(self, stage, rows, extra_lines=(), sections=()):
        """id identifies a saved operation; recipe_key may hold its full parameter signature."""
        from PySide6.QtWidgets import QCheckBox
        rows = list(rows); extra_lines = tuple(extra_lines); sections = tuple(sections)
        rendered = (stage, repr(rows), extra_lines, sections)
        if rendered == self._rendered_rows:
            return
        self._rendered_rows = rendered
        fingerprint = (stage, repr([(r.get("id"), bool(r.get("on", True)), r["label"], r.get("recipe_key")) for r in rows]))
        if self._fingerprint != fingerprint: self.clear_pending()
        self._fingerprint = fingerprint
        self.head.setText(W.APPLIED["head"].format(stage=stage))
        while self.body.count():
            item = self.body.takeAt(0); widget = item.widget()
            if widget is not None: widget.hide(); widget.setParent(None); widget.deleteLater()
        self.rows = []; self.checks = {}; self.original = {}; self.badges = {}; self.cached = {}
        if not rows:
            empty = StableLabel(W.APPLIED["empty"], 2); empty.setMinimumWidth(0); self.body.addWidget(empty)
        heads = dict(sections)
        for index, row in enumerate(rows):
            if index in heads: self.body.addWidget(StableLabel(heads[index], 1))
            widget = QWidget(); h = QHBoxLayout(widget); h.setContentsMargins(0, 1, 0, 1); h.setSpacing(6)
            key = row.get("id"); original = bool(row.get("on", True))
            checkbox = QCheckBox(); checkbox.setFixedSize(20, 20)
            checkbox.setToolTip("Choose changes, then Apply changes to rebuild once.")
            checkbox.setAccessibleName("Include " + row["label"])
            checkbox.setChecked(self.pending.get(key, original)); checkbox.setEnabled(self.enabled_tools and bool(key))
            if key:
                self.original[key] = original; self.checks[key] = checkbox
                checkbox.toggled.connect(lambda on, k=key: self._checked(k, on))
            label=AppliedPartsLabel("%d  %s" % (index+1,row["label"]));label.setObjectName("AppliedEntryText");label.setToolTip(row["label"])
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setFont(ui_font(13, QFont.Weight.DemiBold)); label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            if not original: label.setStyleSheet("color: %s;" % T["ink3"])
            elif row.get("editing"): label.setStyleSheet("color: %s;" % T["accent_hi"])
            editable = bool(row.get("editable", True))
            edit = button(W.APPLIED["edit"], "quiet", "Change it and press Update"); edit.setFixedSize(44, 30); edit.setStyleSheet("padding: 0 3px;")
            edit.setEnabled(self.enabled_tools and editable); edit.clicked.connect(lambda _=False, i=index: self.edit.emit(i))
            more = button("⋯", "quiet", "Turn off, Turn on, Remove"); more.setFixedSize(28, 30); more.setStyleSheet("padding: 0 2px;"); more.setEnabled(self.enabled_tools)
            menu = QMenu(more)
            action = menu.addAction(W.APPLIED["off"] if original else W.APPLIED["on"])
            action.triggered.connect(lambda _=False, i=index: self.toggle.emit(i))
            action = menu.addAction(W.APPLIED["remove"]); action.triggered.connect(lambda _=False, i=index: self.remove.emit(i))
            more.clicked.connect(lambda _=False, mm=menu, bb=more: mm.exec(bb.mapToGlobal(bb.rect().bottomLeft())))
            # one row: checkbox, name, "Cached", then Edit and ⋯ side by side (the badge under the name pushed the
            # checkbox below the label's first line)
            badge = QLabel(""); badge.setFixedSize(46, 18); badge.setFont(ui_font(10)); badge.setStyleSheet("color: %s;" % T["kept"])
            badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if key: self.badges[key] = badge; self.cached[key] = bool(row.get("cached"))
            actions = QHBoxLayout(); actions.setContentsMargins(0, 0, 0, 0); actions.setSpacing(2)
            actions.addWidget(edit); actions.addWidget(more)
            h.addWidget(checkbox, 0, Qt.AlignmentFlag.AlignVCenter); h.addWidget(label, 1, Qt.AlignmentFlag.AlignVCenter)
            h.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter); h.addLayout(actions)
            # Keep Edit's slot for noneditable deletes so labels and menus line up.
            self.body.addWidget(widget); self.rows.append((widget, edit, more, menu, editable))
            for text in (row.get("stats"), row.get("check")):
                if text:
                    note = StableLabel(text, 2); note.setMinimumWidth(0); self.body.addWidget(note)
        for text in extra_lines: self.body.addWidget(StableLabel(text, 2))
        self._update_pending(); self._update_badges()

    def set_tools_enabled(self, on):
        self.enabled_tools = bool(on)
        for _widget, edit, more, _menu, editable in self.rows: edit.setEnabled(on and editable); more.setEnabled(on)
        for checkbox in self.checks.values(): checkbox.setEnabled(on)
        self._update_pending()
