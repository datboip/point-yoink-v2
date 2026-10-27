# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's tool panels for Raw frames, Side, Fused ("Fusion") and Meshed ("Mesh"), built from build_words:
The title, tab bar, controls, More settings, and one full-width button (Apply, or Detect then Delete / Cancel).
Values come from the recipe (stages.params) and stay pending until the button is pressed (a changed dot shows it);
the page turns a press into engine calls. Nothing here talks to the engine.

  p = StagePanel("fused"); p.load(ctx)          # ctx: recipe, fused kind / method, the side, op being edited...
  p.button_clicked(tab) / p.second_clicked(tab) / p.extra_clicked(tab, index) / p.moved(tab)
  p.tab_values(tab) -> {"paths": {path: value}, "tool": name, "params": {...}, "changed": bool}"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QStackedWidget, QSizePolicy, QCheckBox
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button, StableLabel, fit_width, keep_room
from app2.qt import build_words as W
from app2.qt.tool_panel import (CurrentStack, SliderRow, SegRow, CutChoiceRow, ToggleRow, ComboRow, XYZStepper, MoreSettings, TabBar, big_button, hline,
                                hint_label, fmt_value)


def make_control(c, kind="mesh", extra=()):
    k = c.get("kind")
    if k == "slider": return SliderRow(c, kind)
    if k == "seg": return SegRow(c, extra=extra)
    if k == "cutchoice": return CutChoiceRow(c)
    if k == "toggle": return ToggleRow(c)
    if k == "combo": return ComboRow(c, choices=c.get("choices", ()))
    if k == "xyz": return XYZStepper(c)
    return None


def note_alts(stage, t):
    """Every note a tab may show under its controls (its room is the tallest of them); [] = the tab has no note."""
    alts = []
    if t.get("button") == W.DETECT: alts.append(W.NOTICE["detect_last"])
    for k in ("off_note", "poisson_off"):
        if t.get(k): alts.append(t[k])
    if stage == "side" and t.get("name") == "Platter": alts.append(W.CUT["all_sides"])
    if t.get("stl"): alts += [W.EXPORT["done"].format(name="ProjectLive09272026155225_print.stl", mb="12.5", faces=1000000), W.EXPORT["nothing"]]
    return alts


class ApplyAllRow(ToggleRow):
    """The Side panel's "Apply to all sides" switch. Off: the values are this side's own; on: shared by every side. Reads and sets
    as "this" / "all" so the page's recipe paths need not know it is a switch."""
    def __init__(self, parent=None):
        super().__init__(dict(label=W.APPLY_ALL[0], hint=W.APPLY_ALL[1], default=False), parent)

    def value(self): return "all" if self._v else "this"

    def pick(self, val, emit=False):
        on = val == "all"
        if on == self._v: return
        self._v = on; self.sw.set_on(on)
        if emit: self.moved.emit(self.value())

    def set_engine(self, v): pass                                   # not a recipe value: nothing to compare against


class TabBody(QWidget):
    """One tab: its controls (advanced ones under More settings), the message line, the button row."""
    delete_sides = Signal(list)                 # Raw frames: the side numbers ticked for deletion
    def __init__(self, stage, t, kind="mesh", method_extra=(), parent=None):
        super().__init__(parent); self.stage = stage; self.t = t; self.name = t.get("name") or ""; self.kind = kind
        self.mode = "idle"; self.editing = None; self.controls = []
        v = QVBoxLayout(self); v.setContentsMargins(0, 6, 0, 0); v.setSpacing(4); self.v = v
        self.apply_to = None
        if t.get("apply_to"):
            self.apply_to = ApplyAllRow(); v.addWidget(self.apply_to)
        self.more = MoreSettings(); first = True
        if stage == "meshed" and t.get("name") == "Mesh": keep_room(self.more)
        for c in t.get("controls", []):
            w = make_control(c, kind, extra=method_extra if c.get("path") == "fused.method" else ())
            if w is None: continue
            self.controls.append(w); w.on_dot = self.restyle
            if not c.get("advanced") and not first: w.sep = hline(); v.addWidget(w.sep)    # a hairline between blocks
            if not c.get("advanced"): first = False
            (self.more.lay if c.get("advanced") else v).addWidget(w)
        if any(c.get("advanced") for c in t.get("controls", [])): v.addWidget(self.more)
        else: self.more.hide()
        if t.get("hint"): v.addWidget(hint_label(t["hint"]))
        # the note (platter found, Detect only on the last step, Export saved) and the message (Detect's count, Editing
        # step k) keep their room whatever they say, and Cancel stays in its place (off when idle): nothing a press
        # shows or hides moves the buttons (they bounced as words came and went)
        alts = note_alts(stage, t)
        self.note = StableLabel("", 1 if alts else 0, alts, grow=stage == "raw"); self.note.setProperty("role", "hint"); self.note.setFont(ui_font(12))
        v.addWidget(self.note); self.note.setVisible(bool(alts) or stage == "raw")    # Raw frames' note is its side list
        row = QHBoxLayout(); row.setSpacing(8)
        self.main = big_button(t.get("button") or W.APPLY, "apply") if t.get("button") else None
        if self.main is not None: fit_width(self.main, [t.get("button") or W.APPLY, W.DELETE, W.UPDATE, W.APPLY], 24)
        self.second = big_button(W.CANCEL, None)
        # Cancel is for Detect (Delete / Cancel) and for editing an applied tool step (Update / Cancel)
        self.uses_second = self.main is not None and (t.get("button") == W.DETECT or any(c.get("tool") for c in t.get("controls", [])))
        self.second.setEnabled(False); self.second.setVisible(self.uses_second)
        if self.main is not None: row.addWidget(self.main, 2)
        row.addWidget(self.second, 1)
        v.addLayout(row)
        # Detect's count and "Editing step k" (3 lines: a count with the pieces that come back; longer ends in "...")
        self.msg = StableLabel("", 3 if self.uses_second else 0); self.msg.setFont(ui_font(13, QFont.Weight.DemiBold))
        v.addWidget(self.msg); self.msg.setVisible(self.uses_second)
        self.side_boxes = {}
        if stage == "raw":                       # one tick box per side, and one Delete for the ticked ones
            self.sides_v = QVBoxLayout(); self.sides_v.setSpacing(5); v.addLayout(self.sides_v)
            self.delete_b = button("Delete checked sides", "danger", W.PANELS["raw"]["tabs"][0]["delete_tip"]); self.delete_b.setMinimumHeight(34)
            self.delete_b.setEnabled(False); v.addWidget(self.delete_b)
            self.delete_b.clicked.connect(lambda: self.delete_sides.emit(sorted(n for n, c in self.side_boxes.items() if c.isChecked())))
        self.extras = []
        for text, tip in t.get("extra", []) or []:
            b = button(text, None, tip); b.setMinimumHeight(34); v.addWidget(b); self.extras.append(b)
        v.addStretch(1)

    def visible_controls(self): return [c for c in self.controls if not c.isHidden() or c.c.get("advanced")]

    def shown(self, c): return c.property("shown") is not False

    def set_msg(self, text, colour=None):
        """The line under the buttons: blank keeps its room."""
        self.msg.setText(text or ""); self.msg.setStyleSheet("color: %s;" % T[colour] if colour else "")

    def set_sides(self, rows):
        """Raw frames: rows = [(n, text, flagged)]. Keeps the ticks of sides still there; a flagged side (looks like it is from
        another scan) is shown in the warning color, never ticked for you."""
        keep = {n for n, c in self.side_boxes.items() if c.isChecked()}
        while self.sides_v.count():
            w = self.sides_v.takeAt(0).widget()
            if w is not None: w.deleteLater()
        self.side_boxes = {}
        import re
        for n, text, flagged in rows:
            parts = re.split(r" {2,}", text.strip()); title, detail = parts[0], "  \u00b7  ".join(parts[1:])
            card = QFrame(); card.setObjectName("SideCard"); cv = QVBoxLayout(card); cv.setContentsMargins(10, 6, 10, 7); cv.setSpacing(1)
            c = QCheckBox(title); c.setChecked(n in keep); c.setFont(ui_font(13, QFont.Weight.DemiBold)); c.setMinimumHeight(24)
            d = QLabel(detail); d.setWordWrap(True); d.setFont(ui_font(12)); d.setContentsMargins(24, 0, 0, 0)
            col = T["warn"] if flagged else "#8b95a1"
            if flagged: c.setStyleSheet("QCheckBox {color: %s;}" % T["warn"])
            d.setStyleSheet("color: %s;" % col)
            card.setStyleSheet("QFrame#SideCard {background: #1d2229; border: 1px solid #2b333d; border-radius: 8px;}")
            cv.addWidget(c); cv.addWidget(d)
            c.toggled.connect(self._sides_ticked); self.sides_v.addWidget(card); self.side_boxes[n] = c
        self._sides_ticked()

    def _sides_ticked(self, *_):
        k = sum(1 for c in self.side_boxes.values() if c.isChecked())
        self.delete_b.setEnabled(bool(k)); self.delete_b.setText("Delete checked sides (%d)" % k if k else "Delete checked sides")

    def set_note(self, text):
        self.note.setText(text or "")
        if self.stage == "raw":
            self.note.setVisible(bool(text)); self.note.updateGeometry()
            self.note.setStyleSheet("color: %s; font-weight: 600;" % T["warn"])      # a warning, not more help text

    def set_mode(self, mode):
        """idle: Apply / Detect (Cancel off); detected: Delete + Cancel; editing: Update + Cancel."""
        self.mode = mode
        if self.main is None: return
        base = self.t.get("button") or W.APPLY
        if mode in ("detected", "none"): self.main.setText(W.DELETE); self.main.setProperty("kind", "delete")
        elif mode == "editing": self.main.setText(W.UPDATE); self.main.setProperty("kind", "apply")
        else: self.main.setText(base); self.main.setProperty("kind", "apply")
        self.main.setEnabled(mode != "none" and getattr(self, "tools_on", True) and not self.main.property("off"))
        self.main.setProperty("quiet", mode == "idle" and self.settings_only() and not self.changed())    # nothing to apply yet
        self.second.setEnabled(mode != "idle" and getattr(self, "tools_on", True))
        if mode != "idle" and not self.uses_second: self.second.show()        # never expected: better seen than stuck
        self.main.style().unpolish(self.main); self.main.style().polish(self.main)

    def changed(self): return any(c.changed() for c in self.controls if self.shown(c))

    def settings_only(self):
        """A tab of recipe settings only (no tool step to add, no Detect): its Apply does something only after a change."""
        return self.t.get("button") != W.DETECT and bool(self.controls) and all(c.c.get("path") and not c.c.get("tool") for c in self.controls)

    def restyle(self):
        if getattr(self, "main", None) is not None and self.mode == "idle": self.set_mode("idle")


class StagePanel(QFrame):
    button_clicked = Signal(str)
    second_clicked = Signal(str)
    extra_clicked = Signal(str, int)
    sides_delete = Signal(list)
    moved = Signal(str)
    tab_changed = Signal(str)
    more_tool = Signal(str)

    def __init__(self, stage, kind="mesh", method_extra=(), parent=None):
        super().__init__(parent); self.setProperty("role", "panel"); self.stage = stage; self.kind = kind
        P = W.PANELS[stage]; self.P = P
        v = QVBoxLayout(self); v.setContentsMargins(14, 8, 14, 14); v.setSpacing(4)
        self.title = QLabel(P["title"]); self.title.setFont(ui_font(17, QFont.Weight.Bold)); v.addWidget(self.title)
        names = [t["name"] for t in P["tabs"] if t.get("name")]
        labels = {t["name"]: t["label"] for t in list(P["tabs"]) + list(W.MORE_TOOLS.values()) if t.get("name") and t.get("label")}
        self.tabs = TabBar(names, labels=labels); self.tabs.setVisible(bool(names)); v.addWidget(self.tabs)
        if stage in ("side", "fused", "meshed"): self.tabs.hide(); self.tabs.setVisible = lambda _on=True: None      # the rail's rows choose the step
        if P.get("more_tools_mesh"): keep_room(self.tabs.more)       # "..." comes and goes with the result's kind: tabs stay put
        if names: v.addWidget(hline())
        self.stack = CurrentStack(); v.addWidget(self.stack, 1)
        self.bodies = {}
        for t in P["tabs"]:
            b = TabBody(stage, t, kind, method_extra); self.stack.addWidget(b); self.bodies[t.get("name") or ""] = b
            self._wire(b)
        self.blocks = {}                                     # tools of our own under the tabs, always in sight (Finish for print)
        for t in P.get("blocks", []):
            v.addSpacing(12); v.addWidget(hline()); v.addSpacing(6)
            bt = QLabel(t["title"]); bt.setFont(ui_font(16, QFont.Weight.Bold)); v.addWidget(bt)
            if t.get("intro"): v.addWidget(hint_label(t["intro"]))
            b = TabBody(stage, t, kind); v.addWidget(b); self.bodies[t["name"]] = b; self.blocks[t["name"]] = b; self._wire(b)
        for tid, t in W.MORE_TOOLS.items():                 # the "..." tools, as tabs of their own when picked
            name = t["name"]
            if name in self.bodies: continue
            b = TabBody(stage, dict(t, run_to=None, tool_id=tid), "mesh"); self.stack.addWidget(b); self.bodies[name] = b; self._wire(b)
        self.tabs.picked.connect(self.show_tab)
        self.current = names[0] if names else ""
        self.show_tab(self.current)

    def _wire(self, b):
        n = b.name
        if b.main is not None: b.main.clicked.connect(lambda _=False, x=n: self.button_clicked.emit(x))
        b.second.clicked.connect(lambda _=False, x=n: self.second_clicked.emit(x))
        for i, e in enumerate(b.extras): e.clicked.connect(lambda _=False, x=n, k=i: self.extra_clicked.emit(x, k))
        b.delete_sides.connect(self.sides_delete)
        for c in b.controls: c.moved.connect(lambda _v, x=n: self.moved.emit(x)); c.moved.connect(lambda _v, bb=b: bb.restyle())
        if b.apply_to is not None: b.apply_to.moved.connect(lambda _v, x=n: self.moved.emit(x))

    def body(self, name=None): return self.bodies.get(self.current if name is None else name)

    def show_tab(self, name):
        name = self.tab_key(name)
        if name not in self.bodies or name in self.blocks: return        # a block is always shown
        if name in [t["name"] for t in W.MORE_TOOLS.values()] and name not in self.tabs.btns: self.tabs.add_extra(name)
        for n, b in self.bodies.items():            # only the shown tab sets the panel's height (no empty space under Apply)
            if n in self.blocks: continue
            pol = QSizePolicy.Policy.Preferred if n == name else QSizePolicy.Policy.Ignored
            b.setSizePolicy(pol, pol)
        self.current = name; self.stack.setCurrentWidget(self.bodies[name]); self.stack.updateGeometry()
        self.tabs.show_tab(name); self.tab_changed.emit(name)

    def tab_key(self, name):
        """The page's name for a tab, also when asked by the text it shows ("Cut base" -> "Platter")."""
        return next((t["name"] for t in list(self.P["tabs"]) + list(W.MORE_TOOLS.values()) if t.get("label") and t["label"] == name), name)

    def set_more_tools(self, names):
        """"..." tools for a mesh result (Fill holes, Simplify, Remove slivers), or none."""
        items = [(W.MORE_TOOLS[t]["name"], True) for t in names if W.MORE_TOOLS[t]["name"] not in [x["name"] for x in self.P["tabs"]]]
        self.tabs.set_more(items)
        if not items:
            self.tabs.drop_extras()
            if self.current not in [t.get("name") or "" for t in self.P["tabs"]]: self.show_tab([t.get("name") or "" for t in self.P["tabs"]][0])

    def set_enabled_tools(self, on):
        for b in self.bodies.values():
            b.tools_on = on
            if b.main is not None: b.main.setEnabled(on and b.mode != "none" and not b.main.property("off"))
            b.second.setEnabled(on and b.mode != "idle")
            for e in b.extras: e.setEnabled(on)

    def show_if(self, tags):
        """Show a control when its show_if tag is in tags (fused kind, fused method, mesh method)."""
        for b in self.bodies.values():
            for c in b.controls:
                s = c.c.get("show_if"); ok = s is None or s in tags
                # Basic Poisson settings stay in sight when As scanned is selected.
                # Hiding them before the caller restores them moves Finish for print.
                visible = ok or (self.stage == "meshed" and s == "poisson" and not c.c.get("advanced"))
                c.setProperty("shown", ok); c.setVisible(visible)
                if getattr(c, "sep", None) is not None: c.sep.setVisible(visible)
            if b.more is not None:
                adv = [c for c in b.controls if c.c.get("advanced") and c.property("shown") is not False]
                b.more.setVisible(bool(adv))

    def tab_values(self, name=None):
        """What the tab's button would send: path controls -> {path: value} of the CHANGED ones (all of them when
        all=True), tool controls -> (tool, {param: value}) of every shown control."""
        b = self.body(name); paths = {}; params = {}; tool = None; changed = False
        for c in b.controls:
            if c.property("shown") is False: continue
            if c.changed(): changed = True
            if c.c.get("tool"):
                tool = c.c["tool"]; params[c.c["param"]] = c.value()
            elif c.c.get("path") and c.changed():
                paths[c.c["path"]] = c.value()
        return {"paths": paths, "tool": tool, "params": params, "changed": changed, "tool_id": b.t.get("tool_id")}

