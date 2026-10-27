# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ultra Yoink: build several recipes one after another, then flip between the results. Every step's result is stored
under its own settings, so the shared steps (sides, line-up) build once and switching back to a built recipe later
only re-shows stored results. The sheet warns first with the time and the disk space the extra results take."""
import shutil
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QCheckBox, QWidget
from app2.qt import build_words as W
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button

U = W.ULTRA


def _mb(b): return "%d MB" % round(b / 2**20) if b < 2**30 else "%.1f GB" % (b / 2**30)


def _min(s): return "under a minute" if s < 60 else "about %d min" % round(s / 60)


class UltraSheet(QDialog):
    _open = {}

    @classmethod
    def open_for(cls, page):
        d = cls._open.get(id(page))
        if d is None: d = cls._open[id(page)] = cls(page)
        if page.ultra_running(): d.show_running(); d.show(); d.raise_(); return d
        d.loading(); d.show(); d.raise_()
        page.call("h_ultra_plan", [k for k, _l, _h in W.ONETAP["presets"]], then=d.show_plan,
                  fail=lambda e: d.note.setText(U["plan_failed"].format(why=e)))
        return d

    def __init__(self, page):
        super().__init__(page.window()); self.page = page; self.plan = []
        self.setWindowTitle(U["title"]); self.setMinimumWidth(640)
        self.setStyleSheet("QDialog {background: %s; color: %s;}" % (T["panel"], T["ink"]))      # its own window: the app's dark ground
        v = QVBoxLayout(self); v.setContentsMargins(20, 18, 20, 16); v.setSpacing(10)
        t = QLabel(U["title"]); t.setFont(ui_font(18, QFont.Weight.Bold)); v.addWidget(t)
        self.intro = QLabel(U["intro"]); self.intro.setWordWrap(True); self.intro.setProperty("role", "hint"); v.addWidget(self.intro)
        self.rows_w = QWidget(); self.grid = QGridLayout(self.rows_w); self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(14); self.grid.setVerticalSpacing(6); v.addWidget(self.rows_w)
        self.note = QLabel(""); self.note.setWordWrap(True); self.note.setStyleSheet("color: %s; font-weight: 600;" % T["warn"]); v.addWidget(self.note)
        h = QHBoxLayout(); h.addStretch(1)
        self.cancel_b = button(U["cancel"], None); self.cancel_b.clicked.connect(self.close); h.addWidget(self.cancel_b)
        self.go_b = button(U["go"], "primary"); self.go_b.clicked.connect(self.go); h.addWidget(self.go_b)
        v.addLayout(h)
        self.checks = {}

    def _clear(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None: w.hide(); w.setParent(None); w.deleteLater()     # gone now, not at the next event loop (rows overlapped)
        self.checks = {}

    def loading(self):
        self._clear(); self.note.setText(U["loading"]); self.go_b.setEnabled(False); self.go_b.setText(U["go"]); self.intro.setText(U["intro"])

    # ---------------- before: pick, see the cost
    def show_plan(self, plan):
        self._clear(); self.plan = plan; names = {k: l for k, l, _h in W.ONETAP["presets"]}
        for r, h in enumerate((U["col_recipe"], U["col_time"], U["col_disk"])):
            lab = QLabel(h); lab.setProperty("role", "hint"); self.grid.addWidget(lab, 0, r)
        for i, o in enumerate(plan, 1):
            c = QCheckBox("%s  ·  %s" % (names.get(o["preset"], o["preset"]), W.RECIPE["menu_hint"].get(o["preset"], "")))
            c.setChecked(o["preset"] != "quick"); c.toggled.connect(self._total); self.checks[o["preset"]] = (c, o)
            self.grid.addWidget(c, i, 0)
            self.grid.addWidget(QLabel(_min(o["est_s"]) if o["to_run"] else U["built"]), i, 1)
            self.grid.addWidget(QLabel(_mb(o["est_bytes"]) if o["to_run"] else "0 MB"), i, 2)
        self._total()

    def _total(self, *_):
        picked = [o for c, o in self.checks.values() if c.isChecked()]
        secs = sum(o["est_s"] for o in picked if o["to_run"]); size = sum(o["est_bytes"] for o in picked)
        try: free = shutil.disk_usage(self.page.host_root or "/").free
        except OSError: free = None
        self.note.setText(U["warn"].format(n=len(picked), time=_min(secs), disk=_mb(size), free=_mb(free) if free else "?") if picked else U["none"])
        self.go_b.setEnabled(bool(picked) and (free is None or free > size * 1.5))

    def go(self):
        keys = [k for k, (c, _o) in self.checks.items() if c.isChecked()]
        if not keys: return
        self.page.start_ultra(keys, self); self.show_running()

    # ---------------- during
    def show_running(self, done=0, total=None, current=None):
        self.go_b.setEnabled(False); self.go_b.setText(U["running_button"])
        n = total or len(getattr(self.page, "_ultra", {}).get("all", [])) or 1
        self.note.setText(U["running"].format(done=done, n=n, current=current or ""))

    # ---------------- after: flip between the results
    def show_results(self, results, stopped=False):
        self._clear(); names = {k: l for k, l, _h in W.ONETAP["presets"]}
        self.intro.setText(U["results_intro"])
        for r, h in enumerate((U["col_recipe"], U["col_took"], U["col_faces"], U["col_size"], "")):
            lab = QLabel(h); lab.setProperty("role", "hint"); self.grid.addWidget(lab, 0, r)
        for i, res in enumerate(results, 1):
            self.grid.addWidget(QLabel(names.get(res["preset"], res["preset"])), i, 0)
            self.grid.addWidget(QLabel(_min(res["secs"]) if res["ok"] else U["failed"]), i, 1)
            self.grid.addWidget(QLabel("{:,}".format(res["faces"]) if res.get("faces") else "-"), i, 2)
            self.grid.addWidget(QLabel(_mb(res["bytes"]) if res.get("bytes") else "-"), i, 3)
            b = button(U["show"], "quiet"); b.setEnabled(res["ok"]); b.clicked.connect(lambda _c=False, k=res["preset"]: self.page.show_recipe(k))
            self.grid.addWidget(b, i, 4)
        self.note.setText(U["stopped"] if stopped else U["done"])
        self.go_b.setEnabled(True); self.go_b.setText(U["again"]); self.go_b.clicked.disconnect(); self.go_b.clicked.connect(self._again)

    def _again(self):
        self.go_b.clicked.disconnect(); self.go_b.clicked.connect(self.go); UltraSheet.open_for(self.page)
