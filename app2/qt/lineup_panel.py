# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's Line-up panel: Sides (how well each side lands, Show, Use), Flip, Points, Nudge and Settings. Show and the
Flip / Nudge previews only change the view (the side's model matrix); Use, Flip, Points, Nudge and Settings are pending
until their button. Points holds the picked pairs (PairPicks); the page (build_step) feeds it clicks from the view.
Nudge holds the turn (rot, world axes about the side's centre) the view's rotate rings (ring_turn) and the Rotate boxes make.
No deletes on this stage."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QStackedWidget, QDoubleSpinBox, QCheckBox, QAbstractSpinBox, QComboBox, QSizePolicy
from app2.qt.theme import T, ui_font
from app2.qt import rotate_gizmo as RG
from app2.qt.widgets import button, row_label, StableLabel
from app2.qt import build_words as W
from app2.qt.tool_panel import CurrentStack, SegRow, ComboRow, XYZStepper, TabBar, big_button, hline, hint_label
from app2.qt.stage_panels import TabBody

TABS = {t["name"]: t for t in W.PANELS["lineup"]["tabs"]}


def score_colour(pct): return "kept" if pct >= 80 else "warn" if pct >= 50 else "lost"


class PairPicks:
    """The Points tab's picks: the first click on EITHER side is a pending
    half, a click on the same side again moves it, a click on the other side completes the pair. Pairs are stored
    [to_xyz, moving_xyz] (which 0 = the "Match to" side, 1 = the moving side), each in its own side's scan frame."""
    def __init__(self): self.clear()

    def clear(self): self.pairs = []; self.pending = None; self.redo = None

    def pick(self, which, xyz):
        """-> "redone" | "wrong_side" | "moved" | "half" | "pair"."""
        p = [float(x) for x in xyz]
        if self.redo is not None:
            i, w = self.redo
            if which != w: return "wrong_side"
            self.pairs[i][w] = p; self.redo = None; return "redone"
        if self.pending is not None and self.pending[0] == which: self.pending = (which, p); return "moved"
        if self.pending is None: self.pending = (which, p); return "half"
        a = self.pending[1]; self.pairs.append([a, p] if self.pending[0] == 0 else [p, a]); self.pending = None; return "pair"

    def undo(self):
        """The pending half first, then the last pair."""
        if self.redo is not None: self.redo = None
        elif self.pending is not None: self.pending = None
        elif self.pairs: self.pairs.pop()

    def delete(self, i):
        if 0 <= i < len(self.pairs): self.pairs.pop(i); self.pending = None; self.redo = None

    def start_redo(self, i, which):
        if 0 <= i < len(self.pairs): self.pending = None; self.redo = (i, which)

    def dots(self):
        """[(which, xyz, pair number)]: every placed point, the pending half last."""
        out = [(w, pr[w], i + 1) for i, pr in enumerate(self.pairs) for w in (0, 1) if self.redo != (i, w)]
        if self.pending is not None: out.append((self.pending[0], self.pending[1], len(self.pairs) + 1))
        return out

    def problem(self):
        """None when the pairs can be lined up, else a word key: need / duplicate / collinear."""
        if len(self.pairs) < 3: return "need"
        for k in (0, 1):
            P = np.array([pr[k] for pr in self.pairs], float)
            d = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(len(P)) * 1e9
            if d.min() < 0.5: return "duplicate"
            if np.linalg.svd(P - P.mean(0), compute_uv=False)[1] < 1.0: return "collinear"
        return None

    def matrix(self):
        """4x4 taking the moving side's own frame onto the base side's frame, from the pairs (the rigid fit the engine starts from), or None."""
        if self.problem() is not None: return None
        B = np.array([p[0] for p in self.pairs]); M = np.array([p[1] for p in self.pairs])
        cb, cm = B.mean(0), M.mean(0); U, _s, Vt = np.linalg.svd((M - cm).T @ (B - cb))
        d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ cm; return T

    def residuals(self):
        """mm per pair after a rigid fit of the moving points onto the "to" points (the Kabsch the engine starts from)."""
        if self.problem() is not None: return None
        B = np.array([p[0] for p in self.pairs]); M = np.array([p[1] for p in self.pairs])
        cb, cm = B.mean(0), M.mean(0); U, _s, Vt = np.linalg.svd((M - cm).T @ (B - cb))
        d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return list(np.linalg.norm((M - cm) @ R.T + cb - B, axis=1))


HINT_CHOICES = (("", "Auto"), ("same", "Same way up"), ("flipped", "Flipped over"), ("sideways", "On its side"))


class SideRow(QFrame):
    """One side as a card: name and score with Show / Use / Fine on the first line, how it was scanned on the second, what was done to it below."""
    def __init__(self, sid, n, colour, parent=None):
        super().__init__(parent); self.sid = sid; self.setObjectName("SideCard")
        self.setStyleSheet("QFrame#SideCard {background:#1a2027;border:1px solid #2a323c;border-radius:9px;}")
        v = QVBoxLayout(self); v.setContentsMargins(10, 8, 10, 8); v.setSpacing(6)
        h = QHBoxLayout(); h.setSpacing(8)
        sw = QLabel(); sw.setFixedSize(12, 12); sw.setStyleSheet("background: %s; border-radius: 6px;" % colour); h.addWidget(sw)
        nm = QLabel("Side %d" % n); nm.setFont(ui_font(14, QFont.Weight.DemiBold)); h.addWidget(nm)
        self.score = QLabel(""); h.addWidget(self.score)
        h.addStretch(1)
        self.show_c = button(TABS["Sides"]["row_eye"], "chip", "Show this side in the view (nothing changes in the build)"); self.show_c.setCheckable(True)
        self.use_c = button(TABS["Sides"]["row_use"], "chip", "Use this side in the model (off = left out)"); self.use_c.setCheckable(True)
        h.addWidget(self.show_c); h.addWidget(self.use_c); v.addLayout(h)
        # second line: how it was scanned, and the base badge
        bh = QHBoxLayout(); bh.setContentsMargins(0, 0, 0, 0); bh.setSpacing(8)
        self.up = QComboBox(); self.up.setMinimumWidth(0); self.up.setFixedHeight(28); self.up.setMaximumWidth(260)
        for key, text in HINT_CHOICES: self.up.addItem(text, key)
        self.up.setToolTip("How this side sat when you scanned it, compared with the base side. The line-up only tries fits that match. Auto tries everything.")
        self.base = QLabel(TABS["Sides"]["row_base"]); self.base.setProperty("role", "badge-auto"); self.base.hide()
        cap = QLabel("How scanned"); cap.setProperty("role", "hint"); cap.setFont(ui_font(12)); bh.addWidget(cap)
        bh.addWidget(self.up, 1); bh.addWidget(self.base); bh.addStretch(1); v.addLayout(bh)
        # third line, only when there is something to say: what was already done to this side (can be long, so it wraps)
        self.tag = QLabel(""); self.tag.setProperty("role", "hint"); self.tag.setFont(ui_font(12)); self.tag.setWordWrap(True); self.tag.hide()
        self.tag.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred); v.addWidget(self.tag)
        self.auto = QLabel(""); self.auto.setFont(ui_font(12)); self.auto.setWordWrap(True); self.auto.setStyleSheet("color: %s;" % T["warn"]); self.auto.hide()
        self.auto.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred); v.addWidget(self.auto)
        self.weak = QLabel("    " + TABS["Sides"]["weak"]); self.weak.setFont(ui_font(12)); self.weak.setStyleSheet("color: %s;" % T["lost"]); self.weak.hide()
        v.addWidget(self.weak)

    def set_auto_out(self, text):
        self.auto.setText(text or ""); self.auto.setVisible(bool(text))

    def set_info(self, score, base, used):
        if not used: self.score.setText("left out"); self.score.setStyleSheet("color: %s; font-weight: 600;" % T["ink3"])
        elif score is None: self.score.setText("")
        else:
            pct = int(round(100 * score)); self.score.setText("%d%%" % pct)
            self.score.setStyleSheet("color: %s; font-weight: 600;" % T[score_colour(pct)])
        self.base.setVisible(bool(base)); self.weak.setVisible(bool(used and score is not None and score < 0.5))


class LineupPanel(QFrame):
    apply_sides = Signal(object)             # [(sid, use)] in order
    show_side = Signal(str, bool)
    flip_preview = Signal(str, str)          # sid, axis ("" = none)
    flip_apply = Signal(str, str)
    nudge_preview = Signal(str, object, object)          # sid, move [x, y, z] mm, turn (3x3, world axes, about the side's centre)
    nudge_apply = Signal(str, object, object, bool)      # ... and "Keep exactly as placed"
    remove_op = Signal(str, int)             # list token (lineup.fix / lineup.adjust), index
    settings_apply = Signal()
    refit = Signal()
    tab_changed = Signal(str)
    picked_side = Signal(str)
    points_changed = Signal()                # picks, or the Move side / Match to choice, changed: redraw the view
    points_apply = Signal(str, str, object)  # sid, to, [[to_xyz, side_xyz]]
    auto_points = Signal(str, str, int)      # sid, to, how many: show the automatic fit's matches as editable pairs
    checked_changed = Signal(object)         # {side id: the side it was checked against}

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "panel")
        v = QVBoxLayout(self); v.setContentsMargins(14, 8, 14, 14); v.setSpacing(4)
        t = QLabel(W.PANELS["lineup"]["title"]); t.setFont(ui_font(17, QFont.Weight.Bold)); v.addWidget(t)
        self.tabs = TabBar(list(TABS)); v.addWidget(self.tabs); v.addWidget(hline())
        self.stack = CurrentStack(); v.addWidget(self.stack, 1); self.pages = {}
        # ---- Sides
        sp = QWidget(); sv = QVBoxLayout(sp); sv.setContentsMargins(0, 6, 0, 0); sv.setSpacing(4)
        sv.addWidget(row_label(TABS["Sides"]["head"], "stat"))
        self.rows_box = QVBoxLayout(); self.rows_box.setSpacing(8); sv.addLayout(self.rows_box)
        self.agree = StableLabel("", 1, [TABS["Sides"]["agree"].format(after="88.88", before="88.88")]); self.agree.setProperty("role", "hint")
        self.agree.setFont(ui_font(12)); sv.addWidget(self.agree)                 # blank keeps its room
        self.checked_l = QLabel(""); self.checked_l.setProperty("role", "hint"); self.checked_l.setFont(ui_font(12)); sv.addWidget(self.checked_l)
        self._tools_on = True
        self.sides_btn = big_button(TABS["Sides"]["apply_none"]); sv.addWidget(self.sides_btn); sv.addStretch(1)
        self.sides_btn.clicked.connect(self._apply_sides)
        self.stack.addWidget(sp); self.pages["Sides"] = sp; self.rows = {}; self.used = {}
        # ---- Flip
        fp = QWidget(); fv = QVBoxLayout(fp); fv.setContentsMargins(0, 6, 0, 0); fv.setSpacing(4)
        self.flip_side = ComboRow(dict(label=TABS["Flip"]["side_pick"])); fv.addWidget(self.flip_side)
        self.flip_axis = SegRow(TABS["Flip"]["controls"][0]); fv.addWidget(self.flip_axis)
        fv.addWidget(hint_label(TABS["Flip"]["hint"]))
        self.flip_btn = big_button(W.APPLY); fv.addWidget(self.flip_btn)
        fv.addWidget(row_label(TABS["Flip"]["list_head"])); self.flip_list = QVBoxLayout(); fv.addLayout(self.flip_list); fv.addStretch(1)
        self.flip_btn.clicked.connect(lambda: self.flip_apply.emit(self.flip_side.value() or "", self.flip_axis.value()))
        self.flip_axis.moved.connect(lambda a: self.flip_preview.emit(self.flip_side.value() or "", a))
        self.flip_side.moved.connect(lambda s: (self.picked_side.emit(s or ""), self.flip_preview.emit(s or "", "")))
        self.stack.addWidget(fp); self.pages["Flip"] = fp
        # ---- Points (align via points)
        pt = TABS["Points"]; self.picks = PairPicks()
        pp = QWidget(); pv = QVBoxLayout(pp); pv.setContentsMargins(0, 6, 0, 0); pv.setSpacing(4)
        self.pt_side = ComboRow(dict(label=pt["side_pick"])); self.pt_to = ComboRow(dict(label=pt["to_pick"]))
        gh = QHBoxLayout(); gh.setSpacing(6)
        self.pt_prev = button("\u2039 Previous side", None, "The side before this one in the list, weakest first"); self.pt_nextb = button("Next side \u203a", "build", "The next side to line up by hand, weakest first, matched to the base side")
        for b_ in (self.pt_prev, self.pt_nextb): b_.setMinimumHeight(30); gh.addWidget(b_, 1)
        pv.addLayout(gh); self.pt_guide = hint_label(""); pv.addWidget(self.pt_guide)
        self.pt_prev.clicked.connect(lambda: self._guide(-1)); self.pt_nextb.clicked.connect(lambda: self._guide(1))
        self.pt_order = []; self.base_id = None; self._advance = False
        ah = QHBoxLayout(); ah.setSpacing(6)
        self.pt_auto = button("Show auto dots", None, "Show the matches the automatic fit found for this pair as dots you can check, remove or redo, then Line up from points.")
        self.pt_auto.setMinimumHeight(30); self.pt_auto_n = QComboBox(); self.pt_auto_n.setFixedHeight(30)
        for n in (6, 12, 24): self.pt_auto_n.addItem("%d dots" % n, n)
        self.pt_auto_n.setCurrentIndex(1); ah.addWidget(self.pt_auto, 1); ah.addWidget(self.pt_auto_n); pv.addLayout(ah)
        self.pt_auto.clicked.connect(lambda: self.auto_points.emit(self.pt_side.value() or "", self.pt_to.value() or "", int(self.pt_auto_n.currentData())))
        pv.addWidget(self.pt_side); pv.addWidget(self.pt_to); pv.addWidget(hint_label(pt["hint"]))
        self.pt_next = QLabel(""); self.pt_next.setFont(ui_font(13, QFont.Weight.DemiBold)); self.pt_next.setWordWrap(True); pv.addWidget(self.pt_next)
        self.pt_rows = QVBoxLayout(); self.pt_rows.setSpacing(0); pv.addLayout(self.pt_rows)
        h = QHBoxLayout(); self.pt_undo = button(pt["undo"], None, "Take back the last point"); self.pt_clear = button(pt["clear"], None, "Remove every point")
        h.addWidget(self.pt_undo, 1); h.addWidget(self.pt_clear, 1); pv.addLayout(h)
        self.pt_btn = big_button(pt["button"]); pv.addWidget(self.pt_btn)
        self.checked = {}
        self.pt_fine = button("This one is fine", None, "Mark this pair as lined up fine (for you: it does not change the build), then go to the next side."); self.pt_fine.setCheckable(True); self.pt_fine.setMinimumHeight(32)
        pv.addWidget(self.pt_fine); self.pt_fine.clicked.connect(self._fine_clicked)
        self.pt_why = hint_label(""); pv.addWidget(self.pt_why)
        pv.addWidget(row_label(pt["list_head"])); self.pt_list = QVBoxLayout(); pv.addLayout(self.pt_list); pv.addStretch(1)
        self.pt_undo.clicked.connect(lambda: (self.picks.undo(), self._points_refresh()))
        self.pt_clear.clicked.connect(lambda: (self.picks.clear(), self._points_refresh()))
        self.pt_side.moved.connect(lambda s: (self.picked_side.emit(s or ""), self.picks.clear(), self._points_refresh(), self._fine_refresh(), self._guide_label()))
        self.pt_to.moved.connect(lambda _s: (self.picks.clear(), self._points_refresh(), self._fine_refresh(), self._guide_label()))
        self.pt_btn.clicked.connect(lambda: setattr(self, "_advance", True))      # after a match, go on to the next side
        self.pt_btn.clicked.connect(lambda: self.points_apply.emit(self.pt_side.value() or "", self.pt_to.value() or "",
                                                                   [list(map(list, p)) for p in self.picks.pairs]))
        self.stack.addWidget(pp); self.pages["Points"] = pp
        # ---- Nudge: Move steppers and Bambu's rotate rings (the view) with Rotate (relative / absolute) boxes
        nt = TABS["Nudge"]; self.rot = np.eye(3); self._live = None
        np_ = QWidget(); np_.setMinimumWidth(0); nv = QVBoxLayout(np_); nv.setContentsMargins(0, 6, 0, 0); nv.setSpacing(6)
        self.nudge_side = ComboRow(dict(label=nt["side_pick"])); nv.addWidget(self.nudge_side)
        self.move = XYZStepper(nt["controls"][0]); nv.addWidget(self.move)
        nv.addWidget(hint_label(nt["rings"]))
        self.rel, self.abs_ = [], []
        for head, tip, boxes in ((nt["rel_head"], nt["rel_tip"], self.rel), (nt["abs_head"], nt["abs_tip"], self.abs_)):
            nv.addWidget(row_label(head, "stat")); h = QHBoxLayout(); h.setSpacing(6)
            for k, ax in enumerate("XYZ"):
                l = QLabel(ax); l.setStyleSheet("color: %s; font-weight: 700;" % RG.COLOURS[k]); h.addWidget(l)
                b = QDoubleSpinBox(); b.setRange(-360.0, 360.0); b.setDecimals(1); b.setSingleStep(15.0); b.setToolTip(tip)
                b.setKeyboardTracking(False); b.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)       # no arrows: three boxes fit the panel
                b.setMinimumWidth(0); b.setAlignment(Qt.AlignmentFlag.AlignCenter); h.addWidget(b, 1); boxes.append(b)
            nv.addLayout(h)
        for k, b in enumerate(self.rel): b.editingFinished.connect(lambda k=k: self._rel_entered(k))
        for b in self.abs_: b.editingFinished.connect(self._abs_entered)
        self.turned = QLabel(nt["not_turned"]); self.turned.setFont(ui_font(12)); self.turned.setProperty("role", "hint"); nv.addWidget(self.turned)
        self.keep = QCheckBox(nt["keep"]); self.keep.setToolTip(nt["keep_tip"]); nv.addWidget(self.keep)
        nv.addWidget(hint_label(nt["hint"]))
        h = QHBoxLayout(); self.nudge_reset = big_button(nt["secondary"], None); self.nudge_btn = big_button(W.APPLY)
        h.addWidget(self.nudge_reset, 1); h.addWidget(self.nudge_btn, 2); nv.addLayout(h)
        nv.addWidget(row_label(nt["list_head"])); self.nudge_list = QVBoxLayout(); nv.addLayout(self.nudge_list); nv.addStretch(1)
        self.move.moved.connect(lambda *_: self._nudge_prev())
        self.nudge_side.moved.connect(lambda s: (self.picked_side.emit(s or ""), self.reset_nudge()))
        self.nudge_reset.clicked.connect(lambda: self.reset_nudge())
        self.nudge_btn.clicked.connect(lambda: self.nudge_apply.emit(self.nudge_side.value() or "", self.move.value(), self.rot.copy(), self.keep.isChecked()))
        self.stack.addWidget(np_); self.pages["Nudge"] = np_
        # ---- Settings
        self.settings = TabBody("lineup", TABS["Settings"]); self.stack.addWidget(self.settings); self.pages["Settings"] = self.settings
        self.settings.main.clicked.connect(self.settings_apply.emit)
        if self.settings.extras: self.settings.extras[0].clicked.connect(self.refit.emit)
        self.tabs.picked.connect(self.show_tab); self.current = "Sides"; self.show_tab("Sides")
        self.sides = []

    def show_tab(self, name):
        self.current = name; self.stack.setCurrentWidget(self.pages[name]); self.tabs.show_tab(name); self.tab_changed.emit(name)

    def body(self, name=None): return self.settings if (name or self.current) == "Settings" else None

    def set_auto_out(self, info):
        """{side id: {n, agree, alone}}: a card says when YOLO left that side out and why (until Use is turned back on)."""
        for sid, r in self.rows.items():
            v = (info or {}).get(sid)
            r.set_auto_out("Left out by YOLO: agrees only %d%% with the other sides" % v["agree"] if v and not r.use_c.isChecked() else "")

    def set_sides(self, sides, scores, base, agree, flips, nudges, colours):
        """sides: [{id, n, active}]; scores {id: 0..1}; agree (before, after) or None; flips / nudges: [(label)]."""
        if [s["id"] for s in sides] != [s["id"] for s in self.sides]:
            while self.rows_box.count():
                w = self.rows_box.takeAt(0).widget()
                if w: w.setParent(None); w.deleteLater()
            self.rows = {}
            for k, s in enumerate(sides):
                r = SideRow(s["id"], s["n"], colours[k % len(colours)]); r.show_c.setChecked(True)
                r.show_c.toggled.connect(lambda on, x=s["id"]: self.show_side.emit(x, on))
                r.use_c.toggled.connect(lambda _on: self._count())
                r.up.currentIndexChanged.connect(lambda _i: self._count())
                self.rows_box.addWidget(r); self.rows[s["id"]] = r
            ch = [(s["id"], "Side %d" % s["n"]) for s in sides]
            self.flip_side.set_choices(ch); self.nudge_side.set_choices(ch); self.pt_side.set_choices(ch); self.pt_to.set_choices(ch)
            if len(ch) > 1 and self.pt_to.value() == self.pt_side.value(): self.pt_to.box.setCurrentIndex(1)
            self.settings_base_choices(sides)
        self.sides = sides; self.base_id = base
        self.pt_order = [x["id"] for x in sorted((x for x in sides if x["active"] and x["id"] != base), key=lambda x: scores.get(x["id"], 0.0))]
        for s in sides:
            r = self.rows[s["id"]]
            if self.used.get(s["id"]) != s["active"]:        # the saved value changed (applied, undone, first time): show it; a pending click is kept
                r.use_c.blockSignals(True); r.use_c.setChecked(s["active"]); r.use_c.blockSignals(False)
            self.used[s["id"]] = s["active"]; r.set_info(scores.get(s["id"]), s["id"] == base, s["active"])
        a = TABS["Sides"]["agree"].format(after="%.2f" % agree[1], before="%.2f" % agree[0]) if agree else ""
        self.agree.setText(a); self._count()
        self._fill(self.flip_list, flips, "lineup.fix"); self._fill(self.nudge_list, nudges, "lineup.adjust")
        self._fill(self.pt_list, flips, "lineup.fix", key="lineup.fix#points"); self._points_refresh()
        self._guide_label()
        if self._advance: self._advance = False; self._guide(1)

    # ---- Nudge
    def nudge_rot(self):
        """The turn shown now: the applied-to-the-boxes turn, with a ring drag in progress on top (world axes)."""
        return RG.axis_rot(*self._live) @ self.rot if self._live is not None else self.rot.copy()

    def _nudge_prev(self):
        R = self.nudge_rot(); ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        self.turned.setText(TABS["Nudge"]["turned"].format(deg="%.1f" % ang) if ang >= 0.05 else TABS["Nudge"]["not_turned"])
        for b, x in zip(self.abs_, RG.euler_xyz(R)):
            if not b.hasFocus(): b.blockSignals(True); b.setValue(round(x, 1)); b.blockSignals(False)
        self.nudge_preview.emit(self.nudge_side.value() or "", self.move.value(), R)

    def arrow_move(self, k, mm, done):
        """The view's move arrows: the side slides along world axis k, live while dragging, kept on release."""
        base = self._move_base if getattr(self, "_move_base", None) is not None else np.asarray(self.move.value(), float)
        t = base + RG.AXES[k] * mm
        if done: self._move_base = None; self.move.set_value([float(x) for x in t])
        else:
            self._move_base = base; self.move.set_value([float(x) for x in t])
        self._nudge_prev()

    def ring_turn(self, k, deg, done):
        """The view's ring drag: live while dragging, kept on release."""
        if done: self._live = None; self.rot = RG.axis_rot(k, deg) @ self.rot
        else: self._live = (k, deg)
        self._nudge_prev()

    def rotate_by(self, k, deg): self._live = None; self.rot = RG.axis_rot(k, deg) @ self.rot; self._nudge_prev()

    def _rel_entered(self, k):
        b = self.rel[k]; v = b.value(); b.blockSignals(True); b.setValue(0.0); b.blockSignals(False)
        if abs(v) > 1e-9: self.rotate_by(k, v)

    def _abs_entered(self):
        self._live = None; self.rot = RG.from_euler(*[b.value() for b in self.abs_]); self._nudge_prev()

    def reset_nudge(self, emit=True):
        """Back to no move, no turn. emit False: after Apply, the view keeps the pose until the new line-up comes in."""
        self._live = None; self.rot = np.eye(3); self.move.set_value([0, 0, 0])
        for b in self.rel + ([] if emit else self.abs_): b.blockSignals(True); b.setValue(0.0); b.blockSignals(False)
        if emit: self._nudge_prev()
        else: self.turned.setText(TABS["Nudge"]["not_turned"])

    # ---- Points
    def side_name(self, sid):
        return next((str(s["n"]) for s in self.sides if s["id"] == sid), "?")

    def points_sides(self):
        """(moving sid, to sid) chosen on the Points tab."""
        return self.pt_side.value() or "", self.pt_to.value() or ""

    def add_pick(self, which, xyz):
        """A click on the view resolved to the "to" side (0) or the moving side (1), in that side's own frame."""
        r = self.picks.pick(which, xyz); self._points_refresh(r)
        return r

    def _points_refresh(self, last=None):
        pt = TABS["Points"]; sid, to = self.points_sides(); pk = self.picks
        side_of = lambda w: self.side_name(to if w == 0 else sid)
        if pk.redo is not None: txt = pt["next_redo"].format(n=pk.redo[0] + 1, side=side_of(pk.redo[1]))
        elif pk.pending is not None: txt = pt["next_other"].format(side=side_of(1 - pk.pending[0]))
        else: txt = pt["next_start"].format(n=len(pk.pairs) + 1)
        if last == "wrong_side": txt = pt["wrong_side"].format(n=pk.redo[0] + 1, side=side_of(pk.redo[1]))
        self.pt_next.setText(txt)
        while self.pt_rows.count():
            w = self.pt_rows.takeAt(0).widget()
            if w: w.setParent(None); w.deleteLater()
        res = pk.residuals()
        for i, _pr in enumerate(pk.pairs):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 1, 0, 1); h.setSpacing(6)
            sw = QLabel(str(i + 1)); sw.setFixedSize(18, 18); sw.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sw.setStyleSheet("background: %s; color: #000; border-radius: 9px; font-weight: 700;" % W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)])
            h.addWidget(sw); h.addWidget(QLabel(pt["pair_row"].format(n=i + 1)))
            if res is not None:
                mm = res[i]; r = QLabel("%.1f mm" % mm)
                r.setStyleSheet("color: %s; font-weight: 600;" % T["kept" if mm < W.PAIR_GOOD_MM else "lost" if mm > W.PAIR_BAD_MM else "warn"])
                h.addWidget(r)
            h.addStretch(1)
            for which in (0, 1):
                b = button(pt["redo"].format(side=side_of(which)), "quiet", pt["redo_tip"].format(side=side_of(which)))
                b.clicked.connect(lambda _=False, k=i, ww=which: (self.picks.start_redo(k, ww), self._points_refresh())); h.addWidget(b)
            d = button(pt["delete"], "quiet", "Delete this pair"); d.clicked.connect(lambda _=False, k=i: (self.picks.delete(k), self._points_refresh()))
            h.addWidget(d); self.pt_rows.addWidget(w)
        why = self.points_problem()
        self.pt_btn.setEnabled(why is None and self._tools_on)
        k = 3 - len(pk.pairs)
        if why is None: self.pt_why.setText(pt["fit"].format(mm="%.1f" % (sum(res) / len(res))) if res else "")
        elif why == "need": self.pt_why.setText(pt["need"].format(k=k, s="" if k == 1 else "s"))
        else: self.pt_why.setText(pt[why])
        self.pt_undo.setEnabled(bool(pk.pairs or pk.pending or pk.redo)); self.pt_clear.setEnabled(bool(pk.pairs or pk.pending))
        self.points_changed.emit()

    def points_problem(self):
        sid, to = self.points_sides()
        if not sid or not to or sid == to: return "same"
        return self.picks.problem()

    def settings_base_choices(self, sides):
        for c in self.settings.controls:
            if isinstance(c, ComboRow): c.set_choices([("auto", "Auto (fits best with the others)")] + [(s["id"], "Side %d" % s["n"]) for s in sides])

    def _fill(self, box, labels, token, key=None):
        rm = self.__dict__.setdefault("removes", {}); rm[key or token] = []
        while box.count():
            w = box.takeAt(0).widget()
            if w: w.setParent(None); w.deleteLater()
        if not labels: box.addWidget(hint_label(W.NOTICE["none_yet"]))
        for i, text in enumerate(labels):              # a label, or (label, list token, index in that list)
            tok, k0 = (text[1], text[2]) if isinstance(text, (tuple, list)) else (token, i)
            text = text[0] if isinstance(text, (tuple, list)) else text
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(text); l.setFont(ui_font(13)); h.addWidget(l, 1)
            b = button(TABS["Flip"]["list_remove"], "quiet", "Remove it (the line-up runs again)"); b.clicked.connect(lambda _=False, t=tok, k=k0: self.remove_op.emit(t, k))
            b.setEnabled(self._tools_on); rm[key or token].append(b)
            h.addWidget(b); box.addWidget(w)

    def _guide(self, step):
        """Step through the sides to line up by hand, weakest first; each is matched to the base side."""
        if not self.pt_order: return
        cur = self.pt_side.value(); k = self.pt_order.index(cur) if cur in self.pt_order else (-1 if step > 0 else 0)
        nxt = self.pt_order[(k + step) % len(self.pt_order)]
        self.pt_side.box.setCurrentIndex(max(0, self.pt_side.box.findData(nxt)))
        if self.base_id and self.pt_to.value() != self.base_id and self.base_id != nxt: self.pt_to.box.setCurrentIndex(max(0, self.pt_to.box.findData(self.base_id)))
        self._guide_label()

    def _guide_label(self):
        cur = self.pt_side.value(); n = {x["id"]: x["n"] for x in self.sides}
        if cur in self.pt_order:
            self.pt_guide.setText("Side %d to match to side %d. %d of %d, weakest first. Click 3 or more matching spots on each, then Line up from points." %
                                  (n.get(cur, 0), n.get(self.pt_to.value(), 0), self.pt_order.index(cur) + 1, len(self.pt_order)))
        else: self.pt_guide.setText("Press Next side to go through the sides one at a time, weakest first.")

    def set_tags(self, tags):
        """tags: {side id: [words]}: what was already done to a side by hand, so its row says so."""
        for sid, r in self.rows.items():
            words = tags.get(sid) or []
            r.tag.setText(" \u00b7 ".join(words)); r.tag.setVisible(bool(words))

    def set_checked(self, checked):
        """checked: {side id: the side it was checked against}, as saved with the project (a mark for you; it does not change the build)."""
        self.checked = {k: v for k, v in (checked or {}).items() if isinstance(k, str) and isinstance(v, str)}
        self._checked_label(); self._fine_refresh()

    def checked_map(self): return dict(self.checked)

    def _checked_label(self):
        n = len([k for k in self.checked if k in self.rows])
        self.checked_l.setText("Checked %d of %d (mark a pair that lined up fine on the Points tab)" % (n, len(self.rows)) if self.rows else "")

    def _fine_refresh(self):
        sid, to = self.points_sides(); on = bool(sid) and self.checked.get(sid) == to
        self.pt_fine.setChecked(on); self.pt_fine.setText("Fine against side %s (tap to undo)" % self.side_name(to).replace("Side ", "") if on else "This one is fine")

    def _fine_clicked(self):
        sid, to = self.points_sides()
        if not sid or not to or sid == to: return
        if self.checked.get(sid) == to: self.checked.pop(sid, None)
        else: self.checked[sid] = to
        self._checked_label(); self._fine_refresh(); self.checked_changed.emit(dict(self.checked))
        if sid in self.checked: self._guide(1)                       # fine: on to the next side

    def set_pairs(self, pairs):
        """pairs: [[base_xyz, moving_xyz]] in each side's own frame (the automatic fit's matches): they become ordinary editable pairs."""
        self.picks.pairs = [[list(map(float, a)), list(map(float, b))] for a, b in pairs]; self.picks.pending = None; self.picks.redo = None
        self._points_refresh()

    def set_hints(self, hints):
        """hints: {side id: "same" | "flipped" | "sideways"} as the engine has them now."""
        prev = getattr(self, "hints_now", {}); self.hints_now = dict(hints or {})
        for sid, r in self.rows.items():
            if getattr(r, "_hint_synced", False) and prev.get(sid, "") == self.hints_now.get(sid, ""): continue      # a pending pick is kept
            r.up.blockSignals(True); r.up.setCurrentIndex(max(0, r.up.findData(self.hints_now.get(sid, "")))); r.up.blockSignals(False)
            r._hint_synced = True
        self._count()

    def pending_hints(self):
        """The hints as the rows show them now: the string the engine stores ("id:flipped,id:same")."""
        return ",".join("%s:%s" % (sid, r.up.currentData()) for sid, r in self.rows.items() if r.up.currentData())

    def hints_dirty(self):
        now = ",".join("%s:%s" % (sid, how) for sid, how in getattr(self, "hints_now", {}).items() if how)
        return sorted(self.pending_hints().split(",")) != sorted(now.split(","))

    def pending_sides(self):
        return [(sid, r.use_c.isChecked()) for sid, r in self.rows.items() if r.use_c.isChecked() != self.used.get(sid)]

    def _count(self):
        n = len(self.pending_sides()) + (1 if self.hints_dirty() else 0)
        self.sides_btn.setText(TABS["Sides"]["apply_n"].format(n=n, s="" if n == 1 else "s") if n else TABS["Sides"]["apply_none"])
        self.sides_btn.setEnabled(n > 0 and self._tools_on)

    def _apply_sides(self): self.apply_sides.emit(self.pending_sides())

    def set_enabled_tools(self, on):
        """Every button that changes the build, the Flips / Nudges lists' Remove buttons too (off during a run)."""
        self._tools_on = on
        for b in (self.flip_btn, self.nudge_btn, self.settings.main): b.setEnabled(on)
        self.pt_btn.setEnabled(on and self.points_problem() is None)
        for e in self.settings.extras: e.setEnabled(on)
        for b in [x for bs in getattr(self, "removes", {}).values() for x in bs]: b.setEnabled(on)
        self._count()

