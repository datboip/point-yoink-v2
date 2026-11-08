# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ultra Yoink: build several recipes one after another, then flip between the results. Every step's result is stored
under its own settings, so the shared steps (sides, line-up) build once and switching back to a built recipe later
only re-shows stored results. The sheet warns first with the time and the disk space the extra results take."""
import os
import shutil
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QCheckBox, QWidget, QFrame, QScrollArea
from app2.qt import build_words as W
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button

U = W.ULTRA


def _mb(b): return "%d MB" % round(b / 2**20) if b < 2**30 else "%.1f GB" % (b / 2**30)


class PickCard(QFrame):
    """A whole card that is ticked by clicking it: blue when included, plain when not, with a clear Included / Not included chip."""
    toggled = Signal(bool)

    def __init__(self):
        super().__init__(); self.setObjectName("UltraCard"); self.setCursor(Qt.CursorShape.PointingHandCursor); self._on = False; self.badge = None

    def isChecked(self): return self._on

    def setChecked(self, on):
        on = bool(on)
        if on == self._on: return
        self._on = on; self._look(); self.toggled.emit(on)

    def _look(self):
        self.setProperty("on", self._on); self.style().unpolish(self); self.style().polish(self)
        if self.badge is not None:
            self.badge.setText("\u2713  Included" if self._on else "Not included")
            self.badge.setStyleSheet("color:%s;background:%s;border-radius:9px;padding:2px 10px;font-weight:600;" % (("#dceaff", "#2f5fa8") if self._on else ("#8b95a1", "#1d2229")))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.setChecked(not self._on)
        else: super().mousePressEvent(event)


def _name(k):
    names = {key: label for key, label, _h in W.ONETAP["presets"]}
    return names.get(k, k[6:] if str(k).startswith("saved:") else k)


def _min(s): return "under a minute" if s < 60 else "about %d min" % round(s / 60)


class UltraSheet(QDialog):
    _open = {}

    @classmethod
    def open_for(cls, page):
        d = cls._open.get(id(page))
        if d is None: d = cls._open[id(page)] = cls(page)
        if page.ultra_running():                                  # reopened while it works: say where it is
            u = page._ultra; cur = u.get("cur")
            d.show_running(len(u.get("results", [])), len(u.get("all", [])), _name(cur[0]) if cur else None); d.show(); d.raise_(); return d
        d.loading(); d.show(); d.raise_()
        items = [k for k, _l, _h in W.ONETAP["presets"]]
        try:                                                      # the user's own saved recipes, after the built-in ones
            from app2.builder import preset_store as store
            items += [dict(preset="saved:" + n, fragment=store.recipe_fragment(st)) for n, st in store.load_presets().items()]
        except (ValueError, OSError): pass
        page.call("h_ultra_plan", items, then=d.show_plan,
                  fail=lambda e: d.note.setText(U["plan_failed"].format(why=e)))
        return d

    def __init__(self, page):
        super().__init__(page.window()); self.page = page; self.plan = []
        self.setWindowTitle(U["title"]); self.setMinimumWidth(830)
        from PySide6.QtGui import QGuiApplication
        screen = (page.window().screen() if hasattr(page.window(), "screen") else None) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        self.resize(min(980, avail.width() - 40) if avail else 980, min(900, avail.height() - 60) if avail else 900)      # roomy: all six cards in two rows, no squeezing
        self.setStyleSheet("QDialog {background: %s; color: %s;}" % (T["panel"], T["ink"]))      # its own window: the app's dark ground
        v = QVBoxLayout(self); v.setContentsMargins(20, 18, 20, 16); v.setSpacing(10)
        t = QLabel(U["title"]); t.setFont(ui_font(18, QFont.Weight.Bold)); v.addWidget(t)
        self.steps = QLabel(""); self.steps.setTextFormat(Qt.TextFormat.RichText); v.addWidget(self.steps)
        self.intro = QLabel(U["intro"]); self.intro.setWordWrap(True); self.intro.setProperty("role", "hint"); v.addWidget(self.intro)
        self.rows_w = QWidget(); self.grid = QGridLayout(self.rows_w); self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(12); self.grid.setVerticalSpacing(12)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(self.rows_w)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); scroll.setMinimumHeight(300); scroll.setStyleSheet("QScrollArea {background: transparent;}"); scroll.viewport().setStyleSheet("background: transparent;"); self.rows_w.setStyleSheet("background: transparent;"); v.addWidget(scroll, 1)
        self.note = QLabel(""); self.note.setWordWrap(True); self.note.setMinimumHeight(44)
        self.note.setStyleSheet("color: %s; font-weight: 600; background:#2a2418; border:1px solid #5a4a22; border-radius:6px; padding:8px 12px;" % T["warn"]); v.addWidget(self.note)
        from app2.qt.tool_panel import big_button
        h = QHBoxLayout(); h.setSpacing(10)
        self.all_b = button("Tick all", "quiet"); self.none_b = button("Tick none", "quiet")
        self.all_b.clicked.connect(lambda: self._tick(True)); self.none_b.clicked.connect(lambda: self._tick(False))
        self.back_b = button("Back", "quiet"); self.back_b.clicked.connect(self._back); self.back_b.hide()
        h.addWidget(self.all_b); h.addWidget(self.none_b); h.addWidget(self.back_b); h.addStretch(1)
        self.cancel_b = big_button(U["cancel"], "quiet"); self.cancel_b.setFixedHeight(44); self.cancel_b.setMinimumWidth(190); self.cancel_b.clicked.connect(self._leave); h.addWidget(self.cancel_b)
        self.go_b = big_button(U["go"], "build"); self.go_b.setFixedHeight(44); self.go_b.setMinimumWidth(240); self.go_b.clicked.connect(self._next); h.addWidget(self.go_b)
        v.addLayout(h)
        self.checks = {}; self.step = 1; self._picked = []

    STEPS = ("Pick", "Review", "Building", "Compare")

    def _set_step(self, n, intro):
        """The four-step strip at the top says where you are, and each step's own words say what happens next."""
        self.step = n
        parts = ["<b style='color:#7db4ff'>%d  %s</b>" % (i, name) if i == n else "<span style='color:%s'>%s  %s%s</span>" % ("#8b95a1" if i > n else "#5fbf7a", "\u2713" if i < n else i, name, "")
                 for i, name in enumerate(self.STEPS, 1)]
        self.steps.setText("  <span style='color:#5b6573'>\u203a</span>  ".join(parts)); self.intro.setText(intro)
        self.all_b.setVisible(n == 1); self.none_b.setVisible(n == 1); self.back_b.setVisible(n == 2)
        self.cancel_b.setText({1: "\u2039  Back to YOLO Yoink", 2: "Close", 3: "Hide", 4: "Done"}[n])

    def _leave(self):
        """Step 1 goes back to the YOLO Yoink window it was opened from; the other steps just close this one."""
        back = self.step == 1 and hasattr(self.page, "open_onetap")
        self.close()
        if back: self.page.open_onetap()

    def _next(self):
        {1: self.review, 2: self.go, 4: self._again}.get(self.step, lambda: None)()

    def _back(self):
        if self.plan: self.show_plan(self.plan, keep=self._picked)

    def _tick(self, on):
        for c, _o in self.checks.values(): c.setChecked(on)

    def _clear(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None: w.hide(); w.setParent(None); w.deleteLater()     # gone now, not at the next event loop (rows overlapped)
        for r in range(self.grid.rowCount() + 1): self.grid.setRowStretch(r, 0)
        self.checks = {}

    def loading(self):
        self._clear(); self._set_step(1, U["intro"]); self.note.setText(U["loading"]); self.go_b.setEnabled(False); self.go_b.setText("Next: review")

    # ---------------- before: pick, see the cost
    CARD = ("QFrame#UltraCard {background:#282e36;border:1px solid #414b58;border-radius:8px;} "
            "QFrame#UltraCard:hover {border-color:#639feb;} QFrame#UltraCard[on=\"true\"] {background:#274a78;border-color:#639feb;} "
            "QFrame#UltraCard QLabel {background:transparent;color:#e6ebf2;} "
            "QFrame#UltraCard QLabel[role=\"hint\"] {color:#b3bdc9;} QFrame#UltraCard[on=\"true\"] QLabel[role=\"hint\"] {color:#e3ecf8;} "
            "QFrame#UltraFacts {background:rgba(0,0,0,0.22);border:none;border-radius:6px;} "
            "QFrame#UltraFacts QLabel {color:#aab4c0;} QFrame#UltraCard[on=\"true\"] QFrame#UltraFacts QLabel {color:#d3e2f6;}")

    def _picture(self, key):
        """(pixmap, 'Your part' | 'Example'): this project's own picture for the recipe, else the demo clip."""
        from app2 import recipe_pictures
        own = recipe_pictures.existing(self.page.host_root or "").get("today" if key == "auto" else key)
        path, tag = (own[0], "Your part") if own else (os.path.join(os.path.dirname(__file__), "assets", "presets", key + ".png"), "Example")
        pix = QPixmap(path)
        if not pix.isNull(): pix = pix.scaled(222, 126, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation).copy(0, 0, 222, 126)
        return pix, tag

    def _card(self, key, name, lines, pick=False):
        """A recipe card: its picture, its name (and the Included chip when it can be ticked), then small lines.
        Returns (frame, layout) so a result card can add its Show button."""
        f = PickCard() if pick else QFrame()
        if not pick: f.setObjectName("UltraCard")
        f.setStyleSheet(self.CARD); f.setFixedWidth(244)
        v = QVBoxLayout(f); v.setContentsMargins(10, 10, 10, 10); v.setSpacing(5)
        pix, tag = self._picture(key)
        pic = QLabel(); pic.setFixedSize(222, 126); pic.setPixmap(pix); pic.setStyleSheet("border:1px solid #414b58;border-radius:5px;background:#12161b;")
        v.addWidget(pic)
        head = QHBoxLayout(); title = QLabel(name); title.setFont(ui_font(13, QFont.Weight.DemiBold)); head.addWidget(title, 1)
        if pick:
            f.badge = QLabel(); head.addWidget(f.badge); f._look()
        v.addLayout(head)
        from app2.qt.onetap import FACTS, facts_line
        facts = [facts_line(FACTS[key])] if key in FACTS else []         # the numbers it builds with: method, detail, cleanup
        lines = [t for t in lines if t]
        if lines:                                                         # what it is, in plain text
            d = QLabel(lines[0]); d.setWordWrap(True); d.setProperty("role", "hint"); v.addWidget(d)
        box = QFrame(); box.setObjectName("UltraFacts"); bv = QVBoxLayout(box); bv.setContentsMargins(8, 6, 8, 6); bv.setSpacing(3)
        for text in [*lines[1:], *facts, W.RECIPE["card_facts"].get(key, ""), tag]:   # cost, numbers, size, picture source
            if not text: continue
            l = QLabel(text); l.setWordWrap(True); l.setFont(ui_font(11)); bv.addWidget(l)
        v.addWidget(box)
        return f, v

    def _place(self, i, frame):
        self.grid.addWidget(frame, i // 3, i % 3, Qt.AlignmentFlag.AlignTop); self._equalize()

    def _equalize(self):
        """Every card the same height (the tallest one's), so the grid lines up whatever the text wraps to."""
        cards = [self.grid.itemAt(n).widget() for n in range(self.grid.count())]
        cards = [c for c in cards if c is not None]
        if not cards: return
        for c in cards: c.setMinimumHeight(0); c.setMaximumHeight(16777215)
        h = max(c.layout().totalHeightForWidth(c.width() or 244) for c in cards)
        for c in cards: c.setFixedHeight(h)

    def show_plan(self, plan, keep=None):
        self._clear(); self.plan = plan
        self._set_step(1, "Step 1 of 4. Click the recipes to try on this part. Each one builds its own version of the model, so you can compare them side by side afterward.")
        self.go_b.setText("Next: review")
        for i, o in enumerate(plan):
            cost = "%s, %s more" % (_min(o["est_s"]), _mb(o["est_bytes"])) if o["to_run"] else U["built"]
            hint = W.RECIPE["menu_hint"].get(o["preset"], "Your saved settings." if str(o["preset"]).startswith("saved:") else "")
            f, _v = self._card(o["preset"], _name(o["preset"]), [hint, cost], pick=True)
            f.setChecked((o["preset"] in keep) if keep else o["preset"] not in ("quick",) and not str(o["preset"]).startswith("saved:"))
            f.toggled.connect(self._total); self.checks[o["preset"]] = (f, o)
            self._place(i, f)
        self._total()

    def _total(self, *_):
        picked = [o for c, o in self.checks.values() if c.isChecked()]
        secs = sum(o["est_s"] for o in picked if o["to_run"]); size = sum(o["est_bytes"] for o in picked)
        try: free = shutil.disk_usage(self.page.host_root or "/").free
        except OSError: free = None
        self.note.setText(U["warn"].format(n=len(picked), time=_min(secs), disk=_mb(size), free=_mb(free) if free else "?") if picked else U["none"])
        self.go_b.setEnabled(bool(picked) and (free is None or free > size * 1.5))

    def review(self):
        """Step 2: the list of what will run, the total cost, and what happens while it works. Nothing starts until Start."""
        picked = [(k, o) for k, (c, o) in self.checks.items() if c.isChecked()]
        if not picked: return
        self._picked = [k for k, _o in picked]
        self._clear(); self._set_step(2, "Step 2 of 4. Check this list. Nothing has started yet.")
        for r, (k, o) in enumerate(picked):
            f = QFrame(); f.setObjectName("UltraCard"); f.setStyleSheet(self.CARD); h = QHBoxLayout(f); h.setContentsMargins(12, 8, 12, 8)
            pix, tag = self._picture(k); pic = QLabel(); pic.setPixmap(pix.scaled(88, 50, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation).copy(0, 0, 88, 50)); pic.setFixedSize(88, 50); h.addWidget(pic)
            name = QLabel(_name(k)); name.setFont(ui_font(13, QFont.Weight.DemiBold)); h.addWidget(name, 1)
            cost = "%s  \u00b7  %s more" % (_min(o["est_s"]), _mb(o["est_bytes"])) if o["to_run"] else "already built: shows in seconds"
            h.addWidget(QLabel(cost)); self.grid.addWidget(f, r, 0)
        self.grid.setRowStretch(len(picked), 1)                 # rows stay compact; the spare height goes below them
        secs = sum(o["est_s"] for _k, o in picked if o["to_run"]); size = sum(o["est_bytes"] for _k, o in picked)
        try: free = _mb(shutil.disk_usage(self.page.host_root or "/").free)
        except OSError: free = "?"
        self.note.setText("%d recipe%s, %s, about %s more on disk (free: %s).\n\n"
                          "What happens: the sides and line-up build once, then each recipe builds its own Fused and Meshed steps, one after another. "
                          "You can hide this window while it works and open Ultra Yoink again to check. Every result is kept in Versions, "
                          "and your current model is not changed until you Show one." % (len(picked), "" if len(picked) == 1 else "s", _min(secs), _mb(size), free))
        self.go_b.setText("Start Ultra Yoink"); self.go_b.setEnabled(True)

    def go(self):
        keys = list(self._picked) or [k for k, (c, _o) in self.checks.items() if c.isChecked()]
        if not keys: return
        self.page.start_ultra(keys, self); self.show_running()

    # ---------------- during
    def show_running(self, done=0, total=None, current=None):
        """Step 3: every chosen recipe as a row (done, building now, waiting) and the overall count."""
        keys = list(getattr(self.page, "_ultra", {}).get("all", [])) or list(self._picked)
        n = total or len(keys) or 1
        self._clear(); self._set_step(3, "Step 3 of 4. Building. You can hide this window: it keeps going, and Ultra Yoink shows where it is when you open it again.")
        for r, k in enumerate(keys):
            state = "done" if r < done else "now" if r == done else "wait"
            f = QFrame(); f.setObjectName("UltraCard"); f.setStyleSheet(self.CARD); h = QHBoxLayout(f); h.setContentsMargins(12, 10, 12, 10)
            mark = QLabel({"done": "\u2713", "now": "\u25cf", "wait": "\u25cb"}[state]); mark.setFixedWidth(22)
            mark.setStyleSheet("color:%s;font-size:16px;" % {"done": "#5fbf7a", "now": "#7db4ff", "wait": "#5b6573"}[state]); h.addWidget(mark)
            name = QLabel(_name(k)); name.setFont(ui_font(13, QFont.Weight.DemiBold)); h.addWidget(name, 1)
            h.addWidget(QLabel({"done": "built", "now": "building now...", "wait": "waiting"}[state])); self.grid.addWidget(f, r, 0)
        self.grid.setRowStretch(len(keys), 1)
        self.go_b.setEnabled(False); self.go_b.setText(U["running_button"])
        self.note.setText(U["running"].format(done=done, n=n, current=current or ""))

    # ---------------- after: flip between the results
    def show_results(self, results, stopped=False):
        self._clear()
        self._set_step(4, "Step 4 of 4. " + U["results_intro"])
        for i, res in enumerate(results):
            facts = ["%s  ·  %s triangles  ·  %s" % (_min(res["secs"]), "{:,}".format(res["faces"]) if res.get("faces") else "-", _mb(res["bytes"]) if res.get("bytes") else "-")] if res["ok"] else [U["failed"]]
            f, v = self._card(res["preset"], _name(res["preset"]), facts)
            b = button(U["show"], "quiet"); b.setEnabled(res["ok"]); b.clicked.connect(lambda _c=False, k=res["preset"]: self.page.show_recipe(k))
            v.addWidget(b); self._place(i, f)
        self.note.setText(U["stopped"] if stopped else U["done"])
        self.go_b.setEnabled(True); self.go_b.setText(U["again"])

    def _again(self):
        UltraSheet.open_for(self.page)
