# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Export sheet: pick a file format (STL by default, then the last one used) and, for a backup or to share, a ZIP
with the scan data. Shows the model's size in mm once the engine host has measured it. A sheet over the whole window
(like YOLO Yoink's), so File > Export... and the ribbon's Export open it from any page."""
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox
from app2.qt.theme import ui_font
from app2.qt.widgets import button, StableLabel, Scrim
from app2.qt.tool_panel import big_button
from app2.qt import build_words as W

WIDTH, HEIGHT = 520, 450


class ExportDialog(QFrame):
    ok = Signal(str, bool)                 # format key, with the scan data (ZIP)
    cancel = Signal()
    timelapse = Signal()
    watertight = Signal()                  # Make watertight and rebuild: Meshed to Watertight, then the build

    def __init__(self, parent):
        super().__init__(parent); self.setObjectName("KeysSheet"); E = W.EXPORT_SHEET
        self.scrim = Scrim(parent); self.scrim.hide(); self.scrim.clicked.connect(self.cancel.emit)
        parent.installEventFilter(self)
        v = QVBoxLayout(self); v.setContentsMargins(26, 20, 26, 16); v.setSpacing(8)
        title = QLabel(E["title"]); title.setFont(ui_font(20, QFont.Weight.DemiBold)); title.setFixedHeight(34)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(title)
        v.addWidget(QLabel(E["format"]))
        row = QHBoxLayout(); row.setSpacing(6); self.fmt_b = {}
        for key, label in E["formats"]:
            b = button(label, "seg"); b.setCheckable(True); b.setFixedHeight(32)
            b.clicked.connect(lambda checked=False, k=key: self.pick(k)); row.addWidget(b); self.fmt_b[key] = b
        row.addStretch(1); v.addLayout(row)
        self.has_color=False
        self.hint = StableLabel("", 2, alts=list(E["hints"].values()), grow=True); v.addWidget(self.hint)
        self.zip = QCheckBox(E["zip"]); self.zip.setFixedHeight(30)
        self.zip.setStyleSheet("QCheckBox {spacing:8px;} QCheckBox::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;background:#151a21;} QCheckBox::indicator:checked {background:#5b9df0;border:2px solid #dceaff;}")
        v.addWidget(self.zip)
        self.zip_hint = StableLabel(E["zip_hint"], 2); self.zip_hint.setProperty("role", "hint"); v.addWidget(self.zip_hint)
        self.size = StableLabel("", 1); self.size.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(self.size)
        self.filesize = StableLabel("", 1); self.filesize.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(self.filesize)
        closed = QHBoxLayout(); closed.setSpacing(8)
        self.closed = StableLabel("", 2, alts=[E["closed_no"].format(n=888888), E["closed_ok"]], grow=True); self.closed.setFont(ui_font(13, QFont.Weight.DemiBold))
        self.watertight_b = button(E["make_watertight"], "build"); self.watertight_b.setFixedHeight(30); self.watertight_b.hide()
        self.watertight_b.clicked.connect(self.watertight.emit)
        closed.addWidget(self.closed, 1); closed.addWidget(self.watertight_b, 0, Qt.AlignmentFlag.AlignTop); v.addLayout(closed)
        self.est = None; self.scan_bytes = None
        self.zip.toggled.connect(lambda _on: self._refresh_file())
        v.addStretch(1)
        row = QHBoxLayout(); row.addStretch(1)
        self.timelapse_b = button(W.TIMELAPSE_SHEET["open_timelapse"], "quiet")
        self.timelapse_b.setFixedHeight(30); self.timelapse_b.clicked.connect(self.timelapse.emit)
        row.addWidget(self.timelapse_b); v.addLayout(row)
        foot = QHBoxLayout(); self.cancel_b = big_button(E["cancel"], "quiet"); self.ok_b = big_button(E["ok"], "build")
        self.cancel_b.setFixedHeight(44); self.ok_b.setFixedHeight(44); foot.addWidget(self.cancel_b); foot.addWidget(self.ok_b)
        v.addLayout(foot)
        self.cancel_b.clicked.connect(self.cancel.emit); self.ok_b.clicked.connect(self.run)
        self.current = None; self.hide()

    def present(self, formats, fmt, can_zip=True):
        """Show the sheet: formats = the keys this install writes (the others hidden), fmt the one picked."""
        for k, b in self.fmt_b.items(): b.setVisible(k in formats)
        self.pick(fmt if fmt in formats else formats[0])
        self.zip.setChecked(False); self.zip.setEnabled(can_zip); self.zip_hint.setText(W.EXPORT_SHEET["zip_hint" if can_zip else "zip_none"])
        self.set_dims(None); self.set_closed(None); self.est = None; self.scan_bytes = None; self._refresh_file(); self._place()
        self.scrim.show(); self.scrim.raise_(); self.show(); self.raise_()

    def close_sheet(self):
        self.hide(); self.scrim.hide()

    def pick(self, key):
        self.current = key
        for k, b in self.fmt_b.items(): b.setChecked(k == key)
        hint=W.EXPORT_SHEET["hints"].get(key, "")
        if self.has_color:hint+=(" STL does not save scan color." if key=="stl" else " Scan color is included.")
        self.hint.setText(hint); self._refresh_file()

    def set_has_color(self, on):
        self.has_color = bool(on)
        if self.current: self.pick(self.current)

    def set_files(self, est, scan_bytes=None):
        """est: {format key: bytes} for this model; scan_bytes: the size of the scan folders (for the ZIP), or None."""
        self.est = est; self.scan_bytes = scan_bytes; self._refresh_file()

    def _refresh_file(self):
        from app2.builder.export_model import human_bytes
        E = W.EXPORT_SHEET
        if not getattr(self, "est", None): self.filesize.setText(E["file_measuring"] if self.isVisible() else ""); return
        one = self.est.get(self.current or "stl", 0)
        if self.zip.isChecked() and self.zip.isEnabled():
            tot = one + (self.scan_bytes or 0)
            self.filesize.setText(E["zip_file"].format(size=human_bytes(tot)))
        else: self.filesize.setText(E["file"].format(size=human_bytes(one)))

    def set_dims(self, text):
        """The model's "W x D x H mm", or None while it is being measured ("" when it could not be)."""
        E = W.EXPORT_SHEET
        self.size.setText(E["measuring"] if text is None else E["size"].format(dims=text) if text else "")

    def set_closed(self, n, can_fix=True):
        """n open edges (0: closed), None while counting, -1 when it could not be counted. The button shows when the
        model is not closed and the page can switch Meshed to Watertight."""
        E = W.EXPORT_SHEET
        if n is None: text = E["closed_checking"]
        elif n < 0: text = E["closed_unknown"]
        elif n == 0: text = E["closed_ok"]
        else: text = E["closed_no"].format(n=format(n, ","))
        self.closed.setText(text); self.closed.setStyleSheet("color: %s;" % ("#f0b35c" if (n or 0) > 0 else "#c6d4e4"))
        self.watertight_b.setVisible(bool(n and n > 0 and can_fix))

    def choice(self): return self.current, self.zip.isChecked() and self.zip.isEnabled()

    def run(self):
        if self.current: self.ok.emit(*self.choice())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape: self.cancel.emit(); return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter): self.run(); return
        super().keyPressEvent(e)

    def _place(self):
        p = self.parentWidget(); w = min(WIDTH, max(320, p.width() - 24)); h = min(HEIGHT, max(300, p.height() - 24))
        self.setFixedSize(w, h); self.move((p.width() - w) // 2, max(10, (p.height() - h) // 2))
        self.scrim.setGeometry(p.rect())

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.Resize and self.isVisible(): self._place()
        return False
