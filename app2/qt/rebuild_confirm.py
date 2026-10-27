# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explain an edit-chain change before rebuilding, inside the workspace."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QPushButton

class RebuildConfirm(QFrame):
    accepted = Signal()
    cancelled = Signal()
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('KeysSheet'); self.setFixedSize(490,242)
        box = QVBoxLayout(self); box.setContentsMargins(20,18,20,18); box.setSpacing(10)
        self.title = QLabel(); self.title.setFixedHeight(28); box.addWidget(self.title)
        self.text = QLabel(); self.text.setWordWrap(True); self.text.setFixedHeight(112); box.addWidget(self.text)
        row = QHBoxLayout(); self.cancel = QPushButton('Cancel'); self.go = QPushButton()
        for button in (self.cancel,self.go): button.setFixedHeight(40); row.addWidget(button)
        box.addLayout(row)
        self.cancel.clicked.connect(self.cancelled); self.go.clicked.connect(self.accepted)
        self.hide()
    def describe(self, action, label, later):
        self.title.setText(action + ' this edit?')
        self.text.setText(label + '\n\nThis changes the mesh and rebuilds %d later edit%s. Earlier stages are reused when possible. It may take a while. Undo can restore this edit.' % (later, '' if later == 1 else 's'))
        self.go.setText(action + ' and rebuild')
