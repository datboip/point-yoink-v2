# SPDX-License-Identifier: AGPL-3.0-or-later
"""Visible, keyboard reachable choices for short lists."""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QButtonGroup


class Choices(QWidget):
    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)
    activated = Signal(int)

    def __init__(self, parent=None, columns=None, compact=False):
        super().__init__(parent)
        self.items = []
        self.buttons = []
        self.columns = columns
        self.compact = compact
        self._index = -1
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(max(1, self.fontMetrics().lineSpacing() // 6))
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.setStyleSheet('QPushButton {background:#20262e;color:#c6d4e4;border:1px solid #46515e;border-radius:5px;padding:0.18em 0.4em;} QPushButton:checked {background:#234f64;color:#ffffff;border-color:#67c5e8;} QPushButton:focus {border:2px solid #e5eef8;} QPushButton:disabled {color:#76818f;}')

    def addItem(self, text, data=None):
        index = len(self.items)
        self.items.append((str(text), data))
        b = QPushButton(str(text), self)
        b.setCheckable(True)
        b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        b.setToolTip(str(text))
        b.setAccessibleName(str(text))
        self.group.addButton(b, index)
        b.clicked.connect(lambda checked=False, i=index: self._choose(i))
        self.buttons.append(b)
        columns = self.columns or len(self.items)
        for i, button in enumerate(self.buttons):
            self.grid.addWidget(button, i // columns, i % columns)
        b.show()
        self.grid.invalidate()
        if self._index < 0:
            self.setCurrentIndex(0)
        self.updateGeometry()

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def setColumns(self, columns):
        self.columns=max(1,columns)
        for i,b in enumerate(self.buttons):self.grid.addWidget(b,i//self.columns,i%self.columns)
        self.updateGeometry()

    def clear(self):
        for b in self.buttons:
            self.group.removeButton(b)
            self.grid.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self.items.clear(); self.buttons.clear(); self._index=-1
        self.updateGeometry()

    def _choose(self, index):
        self.setCurrentIndex(index)
        self.activated.emit(index)

    def setCurrentIndex(self, index):
        """-1: nothing picked (the Look row while an overlay is shown)."""
        if index == self._index: return
        if index < 0:
            if 0 <= self._index < len(self.buttons):
                self.group.setExclusive(False); self.buttons[self._index].setChecked(False); self.group.setExclusive(True)
            self._index = -1; return
        if not 0 <= index < len(self.items):
            return
        self._index = index
        self.buttons[index].setChecked(True)
        self.currentIndexChanged.emit(index)
        self.currentTextChanged.emit(self.currentText())

    def setCurrentText(self, text):
        self.setCurrentIndex(self.findText(text))

    def currentIndex(self): return self._index
    def currentText(self): return self.itemText(self._index)
    def currentData(self): return self.itemData(self._index)
    def count(self): return len(self.items)
    def itemText(self, index): return self.items[index][0] if 0 <= index < len(self.items) else ''
    def itemData(self, index): return self.items[index][1] if 0 <= index < len(self.items) else None
    def findText(self, text): return next((i for i, item in enumerate(self.items) if item[0] == text), -1)
    def findData(self, data): return next((i for i, item in enumerate(self.items) if item[1] == data), -1)
    def setItemEnabled(self, index, enabled): self.buttons[index].setEnabled(enabled)

    def abbreviate(self, short):
        for b, (text, _) in zip(self.buttons, self.items):
            b.setText(text[:1] if short else text)
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(b.text())+b.fontMetrics().lineSpacing()//3)
            b.setStyleSheet('padding:0.15em 0.1em;' if short else '')
        self.updateGeometry()

    def minimumSizeHint(self):
        return self.grid.minimumSize()

    def sizeHint(self):
        return self.grid.sizeHint()
