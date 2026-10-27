# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixed-size child panel for the optional WiFi motion stream."""
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from app2.scan.telemetry import TelemetryClient, configured_ip


class TelemetryPanel(QFrame):
    dismissed = Signal()

    def __init__(self, parent=None, client=None):
        super().__init__(parent)
        self.client = client or TelemetryClient()
        self.setObjectName('TelemetryPanel')
        self.setFixedSize(380, 462)
        self.setStyleSheet('#TelemetryPanel {background:#20252c;border:1px solid #3b424b;border-radius:8px;}')
        layout = QVBoxLayout(self); layout.setContentsMargins(14,14,14,14); layout.setSpacing(8)
        title = QLabel('Scanner motion over WiFi'); title.setFixedHeight(26); layout.addWidget(title)
        row = QHBoxLayout()
        self.address = QLineEdit(configured_ip()); self.address.setPlaceholderText('Scanner IP address'); self.address.setFixedSize(222,32)
        self.connect_button = QPushButton('Connect'); self.connect_button.setFixedSize(120,32)
        row.addWidget(self.address); row.addWidget(self.connect_button); layout.addLayout(row)
        self.status = QLabel('Disconnected'); self.status.setFixedSize(352,24); layout.addWidget(self.status)
        note = QLabel('Relative angles, not magnetic north.\nDevice axes and units are not verified.'); note.setFixedSize(352,40); layout.addWidget(note)
        grid = QGridLayout(); grid.setSpacing(6); self.fields = {}
        for r, label in enumerate(('Heading / Pitch / Roll (degrees)', 'Position X / Y / Z', 'Gyro X / Y / Z', 'Accel X / Y / Z')):
            caption = QLabel(label); caption.setFixedHeight(20); grid.addWidget(caption,r*2,0,1,3)
            cells = []
            for c in range(3):
                field = QLabel('--'); field.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                field.setFixedSize(112,24); grid.addWidget(field,r*2+1,c); cells.append(field)
            self.fields[label] = cells
        layout.addLayout(grid)
        self.age = QLabel('No packets yet'); self.age.setFixedSize(352,22); layout.addWidget(self.age)
        footer = QHBoxLayout(); footer.addStretch()
        self.done = QPushButton('Close'); self.done.setFixedSize(100,30); footer.addWidget(self.done); layout.addLayout(footer)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.address.returnPressed.connect(self.toggle_connection)
        self.done.clicked.connect(self.dismiss)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh); self.timer.start()
        self.destroyed.connect(lambda *_: self.client.close())

    def toggle_connection(self):
        state = self.client.snapshot()[0]
        if state in ('Connecting','Waiting for data','Receiving','Stale'):
            self.client.disconnect()
        else:
            try:
                self.client.connect(self.address.text())
            except ValueError:
                self.status.setText('Enter a valid IP address')
                return
        self.refresh()

    def refresh(self):
        state, sample, count, age = self.client.snapshot()
        active = state in ('Connecting','Waiting for data','Receiving','Stale')
        self.connect_button.setText('Disconnect' if active else 'Connect')
        self.address.setEnabled(not active); self.status.setText(state)
        rows = (sample.angles, sample.position, sample.gyro, sample.acceleration) if sample else (None,)*4
        for cells, values in zip(self.fields.values(), rows):
            for i, field in enumerate(cells):
                value = None if values is None else values[i]
                field.setText('--' if value is None else f'{value:+.3g}')
                field.setEnabled(state == 'Receiving')
        self.age.setText('No packets yet' if age is None else f'{count:.3g} packets · Last update {min(age,9999):.1f} s ago')

    def dismiss(self):
        self.client.disconnect(); self.hide(); self.dismissed.emit()

    def shutdown(self):
        self.timer.stop(); self.client.close()
