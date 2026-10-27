# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pages for steps that are not built yet (Review, Clean, Measure, Export): one plain page each saying what the step
will do. The shell shows them in its page stack; there is no step rail any more (the Workspace ribbon leads there)."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from app2.qt.theme import ui_font

SENTENCES = {
    "Review": "Look over the built model and how well each side lined up. Not built yet.",
    "Clean": "Cut away stray bits and the base by hand. Not built yet.",
    "Measure": "Measure distances and diameters on the model. Not built yet.",
    "Export": "Save your model with File > Export STL. More export formats are not built yet.",
}


class PlaceholderPage(QWidget):
    def __init__(self, step, parent=None):
        super().__init__(parent); self.step = step
        v = QVBoxLayout(self); v.addStretch(1)
        t = QLabel(step); t.setFont(ui_font(22, QFont.Weight.Bold)); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s = QLabel(SENTENCES[step]); s.setProperty("role", "hint"); s.setFont(ui_font(15)); s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setWordWrap(True)
        v.addWidget(t); v.addSpacing(8); v.addWidget(s); v.addStretch(2)

