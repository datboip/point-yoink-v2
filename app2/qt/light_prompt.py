# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fixed child card so the fill-light timer never opens a second window."""
from PySide6.QtWidgets import QFrame,QVBoxLayout,QHBoxLayout,QLabel
from PySide6.QtCore import Signal
from app2.qt.widgets import button

class LightPrompt(QFrame):
    extend=Signal()
    turn_off=Signal()
    def __init__(self,parent):
        super().__init__(parent);self.setObjectName('KeysSheet');self.setFixedSize(440,140)
        layout=QVBoxLayout(self);layout.setContentsMargins(16,12,16,12)
        layout.addWidget(QLabel('Still using the fill light?'))
        self.note=QLabel();self.note.setFixedHeight(24);layout.addWidget(self.note)
        row=QHBoxLayout();yes=button('Yes, add 2 minutes','build');off=button('Turn off','quiet')
        row.addWidget(yes);row.addWidget(off);layout.addLayout(row)
        yes.clicked.connect(self.extend.emit);off.clicked.connect(self.turn_off.emit);self.hide()
    def update_time(self,remaining):
        self.setVisible(remaining is not None)
        if remaining is not None:
            self.note.setText('Turns off by itself in %d seconds.' % remaining)
            self.move((self.parentWidget().width()-self.width())//2,70);self.raise_()
