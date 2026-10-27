# SPDX-License-Identifier: AGPL-3.0-or-later
"""A group of big tile buttons that stands in for a dropdown: pick one, it stays picked. Same calls and signals as QComboBox
(addItem, currentData, setCurrentIndex, findData, currentIndexChanged...) so the code around it did not have to change."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy


class ChoiceGroup(QWidget):
    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)

    def __init__(self, columns=2, parent=None):
        super().__init__(parent); self._cols = max(1, columns); self._buttons = []; self._data = []; self._index = -1
        self._grid = QGridLayout(self); self._grid.setContentsMargins(0, 0, 0, 0)
        gap = self.fontMetrics().lineSpacing() // 2; self._grid.setHorizontalSpacing(gap); self._grid.setVerticalSpacing(gap)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def addItem(self, text, data=None, tip=None):
        b = QPushButton(text, self); b.setCheckable(True); b.setFocusPolicy(Qt.FocusPolicy.NoFocus); b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setProperty('kind', 'choice'); b.setMinimumHeight(self.fontMetrics().lineSpacing() * 3)
        b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if tip: b.setToolTip(tip)
        i = len(self._buttons); b.clicked.connect(lambda _=False, i=i: self.setCurrentIndex(i))
        self._buttons.append(b); self._data.append(data if data is not None else text)
        self._grid.addWidget(b, i // self._cols, i % self._cols)
        if self._index < 0: self._set(0, emit=False)

    def addItems(self, texts):
        for t in texts: self.addItem(t)

    def setItemTip(self, i, tip):
        if 0 <= i < len(self._buttons): self._buttons[i].setToolTip(tip)

    def count(self): return len(self._buttons)
    def currentIndex(self): return self._index
    def currentText(self): return self._buttons[self._index].text() if self._index >= 0 else ''
    def currentData(self): return self._data[self._index] if self._index >= 0 else None
    def itemData(self, i): return self._data[i] if 0 <= i < len(self._data) else None
    def itemText(self, i): return self._buttons[i].text() if 0 <= i < len(self._buttons) else ''
    def findData(self, data):
        for i, d in enumerate(self._data):
            if d == data: return i
        return -1
    def findText(self, text):
        for i, b in enumerate(self._buttons):
            if b.text() == text: return i
        return -1
    def setCurrentText(self, text):
        i = self.findText(text)
        if i >= 0: self.setCurrentIndex(i)

    def setCurrentIndex(self, i):
        if 0 <= i < len(self._buttons): self._set(i, emit=True)
        else: self._sync()

    def _sync(self):
        for k, b in enumerate(self._buttons): b.setChecked(k == self._index)

    def _set(self, i, emit):
        changed = i != self._index; self._index = i; self._sync()
        if emit and changed: self.currentIndexChanged.emit(i); self.currentTextChanged.emit(self.currentText())
