# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera delivery rates in a reserved Build footer."""
from collections import deque
import time
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel


class LiveRates(QLabel):
    WINDOW_S = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('role', 'hint')
        self.setToolTip('Camera frames delivered to the window. Color and depth have different frame rates.')
        self.stamps = {'Color': deque(), 'Depth': deque()}
        self.last_color = None
        self.last_token = 0
        self.active = False
        self.hide()
        self.timer = QTimer(self); self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh); self.timer.start()

    def add(self, frame, now=None):
        now = time.monotonic() if now is None else now
        token=getattr(frame,'color_token',None)
        current=token is None or token>=self.last_token
        color=getattr(frame,'colour',None)
        if color is not None and color is not self.last_color and current:
            self.last_color = color; self.stamps['Color'].append(now)
            if token is not None:self.last_token=token
        if hasattr(frame,'colour_only') and not frame.colour_only: self.stamps['Depth'].append(now)
        for stamps in self.stamps.values():
            while stamps and now - stamps[0] > self.WINDOW_S: stamps.popleft()

    def refresh(self, now=None):
        now = time.monotonic() if now is None else now
        values = []
        fresh = False
        for name, stamps in self.stamps.items():
            recent = bool(stamps and now - stamps[-1] <= self.WINDOW_S)
            fresh |= recent
            rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if recent and len(stamps) > 1 and stamps[-1] > stamps[0] else 0.
            values.append('%s %.1f fps' % (name, rate))
        self.setText('   '.join(values))
        self.setVisible(self.active and fresh)
