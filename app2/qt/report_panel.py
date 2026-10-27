# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's Scan here panel: Map (open edge, pinholes, the legend, the places to scan; a click turns the view
to one) and Settings (the distances the map is drawn with; Apply rebuilds the map and publishes the model)."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QStackedWidget
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button, row_label
from app2.qt import build_words as W
from app2.qt.tool_panel import CurrentStack, TabBar, hline, hint_label
from app2.qt.stage_panels import TabBody

TABS = {t["name"]: t for t in W.PANELS["report"]["tabs"]}


class ReportPanel(QFrame):
    place_clicked = Signal(int)
    scan_new_side = Signal()
    settings_apply = Signal()
    tab_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent); self.setProperty("role", "panel")
        v = QVBoxLayout(self); v.setContentsMargins(14, 8, 14, 14); v.setSpacing(4)
        t = QLabel(W.PANELS["report"]["title"]); t.setFont(ui_font(17, QFont.Weight.Bold)); v.addWidget(t)
        self.tabs = TabBar(list(TABS)); v.addWidget(self.tabs); v.addWidget(hline())
        self.stack = CurrentStack(); v.addWidget(self.stack, 1)
        mp = QWidget(); mv = QVBoxLayout(mp); mv.setContentsMargins(0, 6, 0, 0); mv.setSpacing(5)
        self.head = QLabel(""); self.head.setFont(ui_font(15, QFont.Weight.DemiBold)); self.head.setWordWrap(True)
        self.sub = hint_label(""); mv.addWidget(self.head); mv.addWidget(self.sub)
        self.hole_metrics=hint_label("");mv.addWidget(self.hole_metrics)
        self._guide=None
        lg = QVBoxLayout(); lg.setSpacing(4)
        self.legend=[]
        for colour,text in (("#ff4632","Not scanned yet (holes)"),("#b45ae6","Rough or doubled, scan again"),("#46d778","Done")):
            l=QLabel("<span style='color:%s'>●</span> %s" % (colour,text));l.setTextFormat(Qt.TextFormat.RichText);lg.addWidget(l);self.legend.append(l)
        lg.addStretch(1); mv.addLayout(lg); mv.addWidget(hline())
        self.places = QVBoxLayout(); self.places.setSpacing(0); mv.addLayout(self.places)
        self.new_side = button(TABS["Map"]["extra"][0][0], "build", TABS["Map"]["extra"][0][1]); self.new_side.setMinimumHeight(36)
        self.new_side.clicked.connect(self.scan_new_side.emit); mv.addWidget(self.new_side); mv.addStretch(1)
        self.stack.addWidget(mp); self.map = mp
        self.settings = TabBody("report", TABS["Settings"]); self.settings.set_note("These settings change only the hole details. The missing and rough map uses the built surface."); self.stack.addWidget(self.settings)
        self.settings.main.clicked.connect(self.settings_apply.emit)
        self.tabs.picked.connect(self.show_tab); self.current = "Map"; self.show_tab("Map")

    def show_tab(self, name):
        self.current = name; self.stack.setCurrentWidget(self.map if name == "Map" else self.settings); self.tabs.show_tab(name)
        self.tab_changed.emit(name)

    def body(self, name=None): return self.settings if (name or self.current) == "Settings" else None

    def set_report(self, stats, min_mm):
        st=stats or {}
        if "open_edge_mm" in st:
            self.hole_metrics.setText("Hole details: %d mm of open edge, %d runs, %d pinholes under %g mm." % (round(st["open_edge_mm"]),st.get("runs",len(st.get("places") or [])),st.get("pinholes",0),min_mm))
        else:self.hole_metrics.setText("")
        self.hole_metrics.setVisible(bool(self.hole_metrics.text()))
        self.set_guide(self._guide)

    def set_guide(self, guide):
        """The same missing/rough spots and ranked positions that color the 3D view."""
        self._guide=guide
        self.head.setText("%d missing or rough spots" % guide["spots"] if guide else "Choose What is left to find missing and rough spots")
        self.sub.setText((guide.get("reason") or "Put the scanner at a suggested position. Red is missing; purple is rough.") if guide else "")
        while self.places.count():
            w=self.places.takeAt(0).widget()
            if w:w.setParent(None);w.deleteLater()
        for pose in (guide or {}).get("poses",[]):
            text=QLabel("%d. %s" % (pose["rank"],pose["text"]));text.setWordWrap(True)
            text.setStyleSheet("color:%s;padding:5px 0;" % pose["color"])
            text.setAccessibleName("Scanner suggestion %d" % pose["rank"]);self.places.addWidget(text)
        if guide and guide.get("needs_flip"):
            self.places.addWidget(hint_label("Bed side needs the part flipped"))

    def set_enabled_tools(self, on):
        self.settings.main.setEnabled(on); self.new_side.setEnabled(True)

