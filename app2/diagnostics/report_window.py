# SPDX-License-Identifier: AGPL-3.0-or-later
"""Help > Report a problem: a window that shows a copy of the report, with what to include and an Anonymize switch."""
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from app2.diagnostics import report as R

ISSUES_URL = "https://github.com/datboip/point-yoink-v2/issues/new"


class ReportWindow(QDialog):
    def __init__(self, collect, parent=None):
        """collect() -> (data, scrub_context): called each time the window opens, so it is current."""
        super().__init__(parent); self.setWindowTitle("Report a problem"); self.resize(820, 760); self.collect = collect
        self.data, self.ctx = collect(); self.edited = False
        self.setStyleSheet("QDialog {background:#20252c;} QLabel, QCheckBox {color:#e6edf6;} QLabel[role='hint'] {color:#9fb0c3;} "
                           "QCheckBox::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;background:#151a21;} QCheckBox::indicator:checked {background:#5b9df0;border-color:#5b9df0;} "
                           "QPlainTextEdit {background:#12161b;color:#e6edf6;border:1px solid #2c343e;border-radius:6px;} "
                           "QPushButton {background:#2a313a;color:#e6edf6;border:1px solid #3b4654;border-radius:8px;padding:7px 14px;} QPushButton:hover {background:#344254;}")
        v = QVBoxLayout(self); v.setSpacing(8)
        self.warning = QLabel(R.WARNING); self.warning.setWordWrap(True); self.warning.setObjectName("ReportWarning")
        self.warning.setStyleSheet("QLabel#ReportWarning {background:#3a2f17;border:1px solid #6a5528;border-radius:8px;padding:10px;color:#f2d28a;}")
        v.addWidget(self.warning)
        top = QHBoxLayout(); self.anon = QCheckBox("Anonymize (hide names, folders, addresses)"); self.anon.setChecked(True); top.addWidget(self.anon); top.addStretch(1); v.addLayout(top)
        box = QFrame(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(2)
        self.boxes = {}
        for key, title, desc in R.SECTIONS:
            c = QCheckBox(title); c.setChecked(True); c.setToolTip(desc); row = QHBoxLayout(); row.addWidget(c)
            d = QLabel(desc); d.setProperty("role", "hint"); row.addWidget(d, 1); w = QWidget(); w.setLayout(row); bl.addWidget(w); self.boxes[key] = c
        v.addWidget(box)
        v.addWidget(QLabel("Your note (what you were doing, what went wrong):"))
        self.note = QPlainTextEdit(); self.note.setFixedHeight(60); v.addWidget(self.note)
        self.never = QLabel(R.NEVER); self.never.setProperty("role", "hint"); v.addWidget(self.never)
        v.addWidget(QLabel("The copy (you can edit it; Copy and Save use exactly this text):"))
        self.preview = QPlainTextEdit(); self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap); v.addWidget(self.preview, 1)
        self.size = QLabel(""); self.size.setProperty("role", "hint"); v.addWidget(self.size)
        row = QHBoxLayout()
        self.copy_b = QPushButton("Copy to clipboard"); self.save_b = QPushButton("Save as file..."); self.issue_b = QPushButton("Open the issues page"); self.close_b = QPushButton("Close")
        self.issue_b.setToolTip("Opens the GitHub issues page in your browser. Nothing is sent from here.")
        for b in (self.copy_b, self.save_b, self.issue_b): row.addWidget(b)
        row.addStretch(1); row.addWidget(self.close_b); v.addLayout(row)
        self.anon.toggled.connect(self.refresh); self.note.textChanged.connect(self.refresh)
        for c in self.boxes.values(): c.toggled.connect(self.refresh)
        self.preview.textChanged.connect(self._size)
        self.copy_b.clicked.connect(self.copy); self.save_b.clicked.connect(self.save); self.issue_b.clicked.connect(lambda: webbrowser.open(ISSUES_URL)); self.close_b.clicked.connect(self.close)
        self.refresh()

    def sections(self): return {k for k, c in self.boxes.items() if c.isChecked()}

    def text(self): return self.preview.toPlainText()

    def refresh(self, *_):
        """Rebuild the copy from the boxes (this drops edits made in the preview since the last rebuild)."""
        self.preview.setPlainText(R.build_report(self.data, self.sections(), self.anon.isChecked(), self.ctx, self.note.toPlainText()))

    def _size(self):
        n = len(self.text().encode("utf-8")); self.size.setText("%d lines, %s" % (self.text().count("\n"), ("%.1f KB" % (n / 1024)) if n >= 1024 else "%d bytes" % n))

    def copy(self): QGuiApplication.clipboard().setText(self.text())

    def save(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save the report", "pointyoink-report.txt", "Text files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f: f.write(self.text())
