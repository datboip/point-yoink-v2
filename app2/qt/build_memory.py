# SPDX-License-Identifier: AGPL-3.0-or-later
"""A nonblocking choice before memory-constrained builds."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QPushButton


class BuildMemoryDialog(QDialog):
    choice = Signal(bool)

    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.setObjectName('BuildMemoryDialog'); self.setWindowTitle('RAM is low for this build')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        layout=QVBoxLayout(self)
        if plan.get('unknown'):
            text='Available RAM could not be read.'
        else:
            gib=1024**3
            text=('Available RAM: %.1f GiB. The largest planned step asks for %.1f GiB. '
                  'The desktop reserve is %.1f GiB.') % (plan['available']/gib,plan['required']/gib,plan['reserve']/gib)
            text+='\nPlanned jobs ask for up to %.1f GiB at once.' % (plan.get('peak',plan['required'])/gib)
        text+='\n\nFewer jobs runs one build step at a time and keeps the current RAM limits. A large step can still run out of RAM.'
        self.message=QLabel(text); self.message.setWordWrap(True); layout.addWidget(self.message)
        buttons=QDialogButtonBox(); layout.addWidget(buttons)
        self.continue_button=buttons.addButton('Build with fewer jobs',QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button=buttons.addButton('Cancel',QDialogButtonBox.ButtonRole.RejectRole)
        self.continue_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        self.finished.connect(lambda result:self.choice.emit(result==QDialog.DialogCode.Accepted))
        self.message.ensurePolished();buttons.ensurePolished()
        self.message.setMinimumHeight(self.message.heightForWidth(446))
        self.resize(470,self.sizeHint().height())
