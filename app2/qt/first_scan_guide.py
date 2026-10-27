# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first-scan guide card at the top of Sides, and a ring around the button it asks for."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from app2.qt.theme import T


class GuideCard(QFrame):
    hidden_by_user = Signal()

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName('GuideCard')
        line = self.fontMetrics().lineSpacing()
        self.setStyleSheet('QFrame#GuideCard { background: %s; border: 1px solid %s; border-radius: %dpx; }' % (T['panel'], T['accent'], line // 2))
        v = QVBoxLayout(self); v.setContentsMargins(line // 2, line // 3, line // 2, line // 2); v.setSpacing(line // 4)
        top = QHBoxLayout(); self.count = QLabel(); self.count.setProperty('role', 'hint'); top.addWidget(self.count, 1)
        self.close = QPushButton('Hide guide'); self.close.setFlat(True); self.close.setToolTip('Hide the step-by-step guide. The Guide button at the top brings it back.')
        self.close.clicked.connect(self.hidden_by_user); top.addWidget(self.close); v.addLayout(top)
        self.title = QLabel(); font = self.title.font(); font.setBold(True); font.setPointSizeF(font.pointSizeF() * 1.25); self.title.setFont(font)
        self.text = QLabel(); self.text.setWordWrap(True)
        v.addWidget(self.title); v.addWidget(self.text)

    def set_step(self, step, total):
        n, title, text, _ = step
        self.count.setText('First scan guide: step %d of %d' % (n, total)); self.title.setText(title); self.text.setText(text)

    def tallest(self, width, steps, total):
        """The height of the longest step at this width, so a new step or Hide guide never moves what sits below."""
        shown = (self.count.text(), self.title.text(), self.text.text())
        tall = 0
        for step in steps:
            self.set_step(step, total); self.layout().invalidate(); tall = max(tall, self.heightForWidth(width))
        self.count.setText(shown[0]); self.title.setText(shown[1]); self.text.setText(shown[2]); self.layout().invalidate()
        return tall


class Ring(QWidget):
    """An accent ring drawn around the button the guide asks for. It never takes a click."""
    def __init__(self, parent):
        super().__init__(parent); self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.target = None; self.hide()

    def around(self, target):
        self.target = target if target is not None and target.isVisible() else None
        if self.target is None: self.hide(); return
        pad = self.fontMetrics().lineSpacing() // 4
        box = self.target.rect().translated(self.target.mapTo(self.parentWidget(), self.target.rect().topLeft()))
        self.setGeometry(box.adjusted(-pad, -pad, pad, pad)); self.show(); self.raise_()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen
        from app2.qt.theme import qc
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(2, self.fontMetrics().lineSpacing() // 8); p.setPen(QPen(qc('accent'), width)); p.setBrush(Qt.BrushStyle.NoBrush)
        r = self.rect().adjusted(width, width, -width, -width); p.drawRoundedRect(r, width * 3, width * 3)
