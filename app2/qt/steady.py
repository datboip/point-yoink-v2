# SPDX-License-Identifier: AGPL-3.0-or-later
"""Words that stay up long enough to read. A different message waits until the shown one has had its turn, so two
messages taking turns every frame read as the first one, not a flicker. A problem does not wait."""
from PySide6.QtCore import QElapsedTimer, QTimer

HOLD_MS = 2000   # a short line read at a glance


class Steady:
    def __init__(self, show, parent=None):
        self.show = show; self.clock = QElapsedTimer(); self.shown = object(); self.pending = None
        self.timer = QTimer(parent); self.timer.setSingleShot(True); self.timer.timeout.connect(self._flush)

    def offer(self, key, *args, urgent=False):
        """key: what the reader reads (the words); args go to show. The same key shows its new args at once."""
        if key == self.shown: self.pending = None; self.timer.stop(); self.show(*args); return
        left = HOLD_MS - self.clock.elapsed() if self.clock.isValid() else 0
        if urgent or left <= 0: self._show(key, args)
        else: self.pending = (key, args); self.timer.start(int(left))

    def _show(self, key, args):
        self.shown = key; self.pending = None; self.timer.stop(); self.clock.start(); self.show(*args)

    def _flush(self):
        if self.pending is not None: self._show(*self.pending)
