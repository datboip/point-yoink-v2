# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Motion tab of the scanner panel: the optional WiFi motion stream, and the address it connects to."""
import math
import time
from collections import deque
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from app2.scan.telemetry import (TelemetryClient, axes_from, body_angles, configured_axes, configured_ip, configured_zero, mean_pose,
                                 save_axes, save_zero)

ACTIVE = ('Connecting', 'Waiting for data', 'Receiving', 'Stale')


TRACE_SECONDS = 10.
TRACE_COLORS = (('Heading', '#f2b43a'), ('Pitch', '#4a9dff'), ('Roll', '#3fd16f'))


class PoseDrawing(QWidget):
    """A small side-on picture of one calibration pose: the scanner, its camera face, the floor and an arrow for where the camera looks."""
    def __init__(self, down, parent=None):
        super().__init__(parent); self.down = down; self.setFixedSize(92, 78)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height(); floor = h - 8
        p.setPen(QPen(QColor('#5a636e'), 2)); p.drawLine(6, floor, w - 6, floor)
        p.save()
        if self.down: p.translate(w / 2 - 8, floor - 13); p.rotate(90)                       # camera face turned to the floor
        else: p.translate(w / 2 - 8, floor - 28)
        body = QRectF(-11, -28, 22, 56)
        p.setPen(QPen(QColor('#9aa5b1'), 2)); p.setBrush(QColor('#2a3038')); p.drawRoundedRect(body, 5, 5)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor('#f2b43a')); p.drawRoundedRect(QRectF(7, -22, 6, 26), 2, 2)   # the camera face
        p.restore()
        p.setPen(QPen(QColor('#4a9dff'), 2)); p.setBrush(QColor('#4a9dff'))
        if self.down: x, y0, y1 = w - 16, 18, floor - 6; p.drawLine(QPointF(x, y0), QPointF(x, y1)); tip = [QPointF(x, y1 + 2), QPointF(x - 4, y1 - 5), QPointF(x + 4, y1 - 5)]
        else: y, x0, x1 = floor - 44, w / 2 + 6, w - 8; p.drawLine(QPointF(x0, y), QPointF(x1, y)); tip = [QPointF(x1 + 2, y), QPointF(x1 - 5, y - 4), QPointF(x1 - 5, y + 4)]
        p.drawPolygon(tip); p.end()


class MotionView(QWidget):
    """How the scanner is turned: an artificial horizon (pitch and roll, with the heading as a notch on its rim) and
    the last few seconds of the three angles beside it. Relative angles only, like the numbers under it."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150); self.setAccessibleName('Scanner motion view')
        self.angles = None; self.trace = deque()

    def set_angles(self, angles, now=None):
        now = time.monotonic() if now is None else now
        self.angles = None if angles is None else tuple(angles)
        if angles is not None: self.trace.append((now, *angles))
        while self.trace and self.trace[0][0] < now - TRACE_SECONDS: self.trace.popleft()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(), QColor('#161a1f'))
        side = min(self.height() - 16, 140); ball = QRectF(10, (self.height() - side) / 2, side, side)
        self._horizon(p, ball)
        self._trace(p, QRectF(ball.right() + 14, ball.top(), max(10, self.width() - ball.right() - 24), side))

    def _horizon(self, p, box):
        c = box.center(); r = box.width() / 2
        clip = QPainterPath(); clip.addEllipse(box)
        if self.angles is None:
            p.setPen(QPen(QColor('#3b424b'), 2)); p.drawEllipse(box); p.setPen(QColor('#6b7480')); p.drawText(box, Qt.AlignmentFlag.AlignCenter, 'no data'); return
        heading, pitch, roll = self.angles
        p.save(); p.setClipPath(clip); p.translate(c); p.rotate(-roll)
        shift = max(-r, min(r, pitch / 45. * r))             # 45 degrees of pitch moves the horizon to the rim
        p.fillRect(QRectF(-2 * r, -2 * r + shift, 4 * r, 2 * r), QColor('#2c5d8f'))        # sky
        p.fillRect(QRectF(-2 * r, shift, 4 * r, 2 * r), QColor('#6b4a2b'))                 # ground
        p.setPen(QPen(QColor('#e6edf6'), 2)); p.drawLine(QPointF(-2 * r, shift), QPointF(2 * r, shift))
        p.restore()
        p.setPen(QPen(QColor('#e6edf6'), 2)); p.drawLine(QPointF(c.x() - r * .45, c.y()), QPointF(c.x() - r * .15, c.y()))   # the scanner, fixed in the middle
        p.drawLine(QPointF(c.x() + r * .15, c.y()), QPointF(c.x() + r * .45, c.y())); p.drawEllipse(c, 3, 3)
        p.setPen(QPen(QColor('#3b424b'), 2)); p.drawEllipse(box)
        p.save(); p.translate(c); p.rotate(heading); p.setPen(QPen(QColor('#f2b43a'), 3))     # heading: a notch that turns around the rim
        p.drawLine(QPointF(0, -r), QPointF(0, -r + 10)); p.restore()

    def _trace(self, p, box):
        p.setPen(QPen(QColor('#2a3038'), 1)); p.drawRect(box); p.drawLine(QPointF(box.left(), box.center().y()), QPointF(box.right(), box.center().y()))
        p.setPen(QColor('#6b7480')); p.drawText(QPointF(box.left() + 4, box.top() + 12), '+180'); p.drawText(QPointF(box.left() + 4, box.bottom() - 4), '-180')
        if len(self.trace) < 2: return
        now = self.trace[-1][0]
        for k, (name, color) in enumerate(TRACE_COLORS):
            path = QPainterPath(); last = None
            for row in self.trace:
                x = box.right() - (now - row[0]) / TRACE_SECONDS * box.width(); y = box.center().y() - row[1 + k] / 180. * box.height() / 2
                if last is None or abs(row[1 + k] - last) > 180: path.moveTo(x, y)         # a turn through 180 is not a line across the graph
                else: path.lineTo(x, y)
                last = row[1 + k]
            p.setPen(QPen(QColor(color), 2)); p.drawPath(path)


class TelemetryPanel(QWidget):
    dismissed = Signal()

    def __init__(self, parent=None, client=None):
        super().__init__(parent)
        self.client = client or TelemetryClient()
        self.manual_off = False              # the user pressed Disconnect: the app stops connecting by itself
        self.searching = False               # the app is looking for the scanner on the network
        self.zero = configured_zero()        # the pose that counts as level (heading, pitch and roll are 0 there)
        self.axes = configured_axes()        # the scanner's own up, forward and right in the sensor's axes, from Calibrate
        self.step = 0; self._flat = None; self._error = ''     # Calibrate: 0 idle, 1 waiting for the flat pose, 2 waiting for the pointed-down pose
        self.setObjectName('TelemetryPanel')
        layout = QVBoxLayout(self); layout.setContentsMargins(14,14,14,14); layout.setSpacing(8)
        title = QLabel('Scanner motion over Wi-Fi'); title.setFixedHeight(26); layout.addWidget(title)
        row = QHBoxLayout()
        self.address = QLineEdit(configured_ip()); self.address.setPlaceholderText('Scanner IP address'); self.address.setMinimumWidth(180)
        self.connect_button = QPushButton('Connect'); self.connect_button.setFixedWidth(120)
        row.addWidget(self.address, 1); row.addWidget(self.connect_button); layout.addLayout(row)
        self.status = QLabel('Disconnected'); layout.addWidget(self.status)
        self.notice = QLabel(''); self.notice.setWordWrap(True); self.notice.setStyleSheet('color:#f2b43a;'); self.notice.hide(); layout.addWidget(self.notice)
        self.view = MotionView(); layout.addWidget(self.view)
        legend = QLabel('  '.join('<span style="color:%s">&#9632;</span> %s' % (color, name) for name, color in TRACE_COLORS) + '  (last 10 s)')
        legend.setTextFormat(Qt.TextFormat.RichText); layout.addWidget(legend)
        # Calibration: two steps, each with its own words and button
        box = QFrame(); box.setObjectName('CalibrationBox'); box.setStyleSheet('#CalibrationBox {border:1px solid #3b424b;border-radius:8px;}')
        bv = QVBoxLayout(box); bv.setContentsMargins(14,12,14,12); bv.setSpacing(12)
        heading = QLabel('<b>Calibrate the angles</b><br>Two quick steps so the app knows which way is up and where the camera points.')
        heading.setTextFormat(Qt.TextFormat.RichText); heading.setWordWrap(True); bv.addWidget(heading)
        self.badges = []; self.step_buttons = []
        for number, title, hint, caption in ((1, 'Stand it up on its base', 'Like it sits on a desk, camera facing forward. Hold it still, then press the button.', 'Capture standing'),
                                             (2, 'Lay it on its front face', 'Camera side down on the table (or hold it pointing at the floor). Hold it still, then press the button.', 'Capture face down')):
            row = QHBoxLayout(); row.setSpacing(12)
            drawing = PoseDrawing(number == 2)
            badge = QLabel(str(number)); badge.setFixedSize(28,28); badge.setAlignment(Qt.AlignmentFlag.AlignCenter); row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
            words = QVBoxLayout(); words.setSpacing(2)
            words.addWidget(QLabel('<b>%s</b>' % title)); small = QLabel(hint); small.setWordWrap(True); small.setStyleSheet('color:#9aa5b1;'); words.addWidget(small)
            row.addLayout(words, 1); row.addWidget(drawing, 0, Qt.AlignmentFlag.AlignVCenter)
            button = QPushButton(caption); button.setMinimumWidth(170); row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            bv.addLayout(row); self.badges.append(badge); self.step_buttons.append(button)
        self.set_level, self.capture_down = self.step_buttons
        self.zero_note = QLabel(''); self.zero_note.setWordWrap(True); bv.addWidget(self.zero_note)
        foot = QHBoxLayout(); self.clear_level = QPushButton('Clear calibration'); foot.addWidget(self.clear_level); self.zero_heading = QPushButton('Set heading to here'); self.zero_heading.setToolTip('Call the way the scanner points right now "0". Pitch and roll are not changed.'); foot.addWidget(self.zero_heading); foot.addStretch(1); bv.addLayout(foot)
        layout.addWidget(box)
        layout.addSpacing(4)
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6); self.fields = {}
        for r, label in enumerate(('Heading / Pitch / Roll (degrees)', 'Compass sensor X / Y / Z (raw)', 'Gyro X / Y / Z', 'Accel X / Y / Z')):
            caption = QLabel(label); grid.addWidget(caption,r,0)
            cells = []
            for c in range(3):
                field = QLabel('--'); field.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                field.setMinimumWidth(78); grid.addWidget(field,r,c+1); cells.append(field)
            self.fields[label] = cells
        grid.setColumnStretch(0, 1)
        layout.addLayout(grid)
        self.age = QLabel('No packets yet'); layout.addWidget(self.age)
        note = QLabel('Connects by itself when the scanner is on the same Wi-Fi as this PC. If it cannot find the scanner, type the address shown in the scanner\'s Wi-Fi settings. '
                      'Angles are relative to where you calibrated, not magnetic north. Device axes and units are not verified.')
        note.setWordWrap(True); note.setStyleSheet('color:#9aa5b1;'); layout.addWidget(note)
        layout.addStretch(1)
        self.set_level.clicked.connect(self.capture_flat_pose); self.capture_down.clicked.connect(self.capture_down_pose); self.clear_level.clicked.connect(self.forget_level); self.zero_heading.clicked.connect(self.set_heading_here)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.address.returnPressed.connect(self.toggle_connection)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh); self.timer.start()

    def _read_still(self):
        """The average pose and acceleration over the last second, or None (with the reason shown) when it is not usable."""
        self._error = ''
        state = self.client.snapshot()[0]; samples = self.client.recent(1.) if state == 'Receiving' else []
        if len(samples) < 3: self._error = 'No motion data yet. Connect first, then hold the scanner as the step says.'; self.refresh(); return None
        pose, acceleration, spread = mean_pose(samples)
        if spread > 1.5: self._error = 'It moved while I was reading it. Hold the scanner still and press the button again.'; self.refresh(); return None
        return pose, acceleration

    def capture_flat_pose(self):
        """Step 1: flat on its base. This is level, and which way is up."""
        got = self._read_still()
        if got is None: return
        self._flat = got; self.step = 2; self.refresh()

    def capture_down_pose(self):
        """Step 2: pointed straight down. This is which way is forward."""
        if self._flat is None: return
        got = self._read_still()
        if got is None: return
        axes = axes_from(self._flat[1], got[1])
        if axes is None:
            self._error = 'That does not look like pointing straight down from flat. Point the camera at the floor, hold still, and press the button again.'; self.refresh(); return
        self.zero = self._flat[0]; self.axes = axes; save_zero(self.zero); save_axes(axes); self.step = 0; self._flat = None
        self.view.trace.clear(); self.refresh()

    def calibrate(self):
        """The next step in order (flat first, then pointed down)."""
        self.capture_down_pose() if self.step == 2 else self.capture_flat_pose()

    def set_heading_here(self):
        """Heading 0 is the way the scanner points right now. Pitch and roll come from gravity, so they stay as they are."""
        got = self._read_still()
        if got is None: return
        self.zero = got[0]; save_zero(self.zero); self.view.trace.clear(); self.refresh()

    def forget_level(self):
        self._error = ''; self.zero = None; self.axes = None; self.step = 0; self._flat = None; save_zero(None); save_axes(None); self.view.trace.clear(); self.refresh()

    def set_notice(self, text):
        self.notice.setText(text); self.notice.setVisible(bool(text))

    def connect_to(self, address):
        """Connect to this address (found or typed). Raises ValueError for a bad one."""
        self.client.connect(address); self.address.setText(address.strip()); self.manual_off = False
        self.refresh()

    def toggle_connection(self):
        state = self.client.snapshot()[0]
        if state in ACTIVE:
            self.manual_off = True
            self.client.disconnect()
        else:
            try:
                self.connect_to(self.address.text())
            except ValueError:
                self.status.setText('Enter a valid IP address')
                return
            from app2.scan.scanner_ip import remember
            remember(self.address.text()); self.set_notice('')
        self.refresh()

    def refresh(self):
        state, sample, count, age = self.client.snapshot()
        active = state in ACTIVE
        self.connect_button.setText('Disconnect' if active else 'Connect')
        self.address.setEnabled(not active)
        self.status.setText('Looking for the scanner on Wi-Fi...' if self.searching and not active else state)
        angles = (body_angles(self.axes, sample.acceleration, sample.quaternion, self.zero) if self.axes is not None and sample else
                  sample.angles_from(self.zero) if sample else None)
        self.view.set_angles(angles if state == 'Receiving' else None)
        self.clear_level.setEnabled(self.zero is not None or self.step != 0)
        self.zero_heading.setEnabled(self.axes is not None and self.step == 0 and state == 'Receiving')
        done = self.axes is not None and self.step == 0
        self.set_level.setText('Capture standing again' if done else 'Capture standing')
        self.capture_down.setEnabled(self.step == 2)
        for k, badge in enumerate(self.badges):
            finished = done or (k == 0 and self.step == 2); current = (k == 0 and self.step == 0 and not done) or (k == 1 and self.step == 2)
            badge.setText('\u2713' if finished else str(k + 1))
            badge.setStyleSheet('background:%s;color:%s;border-radius:14px;font-weight:bold;' % (('#3fd16f', '#10151a') if finished else ('#4a9dff', '#10151a') if current else ('#2a3038', '#9aa5b1')))
        if self.step == 2: note = 'Step 1 done. Now step 2: lay the scanner on its front face, camera side down, then press Capture face down.'
        elif done: note = 'Calibrated. Standing on its base reads 0. Pitch is tilting up or down, roll is tilting sideways, heading is turning left or right.'
        elif self.zero is not None: note = 'Calibrated the old way. Do both steps again for correct pitch and roll.'
        else: note = 'Not calibrated yet. Start with step 1.'
        self.zero_note.setText(self._error or note)
        rows = (angles, sample.position, sample.gyro, sample.acceleration) if sample else (None,)*4
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
