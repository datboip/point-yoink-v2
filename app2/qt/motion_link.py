# SPDX-License-Identifier: AGPL-3.0-or-later
"""Connects the motion stream by itself once the scanner is live over USB: the remembered address first, then a look
around the Wi-Fi network. When nothing answers, it asks for the address once (the panel's Motion tab) and keeps
looking quietly in the background."""
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

from app2.qt.telemetry_panel import ACTIVE
from app2.scan import scanner_ip
from app2.scan.telemetry import configured_ip

RETRY_S = (20., 60., 300.)           # waits between looks when nothing answers: a sweep of the network should not run every few seconds


class MotionLink(QObject):
    _result = Signal(str, int)
    need_address = Signal()                      # nothing answered: ask for the address (once per run)

    def __init__(self, panel, is_live, parent=None, find=None):
        super().__init__(parent)
        self.panel = panel; self.is_live = is_live; self._find = find or scanner_ip.find
        self._token = 0; self._searching = False; self._next = 0.; self._asked = False; self._misses = 0
        self._result.connect(self._done)
        self.timer = QTimer(self); self.timer.setInterval(2000); self.timer.timeout.connect(self.tick); self.timer.start()

    def tick(self):
        if self.panel.manual_off or self._searching or not self.is_live(): return
        if self.panel.client.snapshot()[0] in ACTIVE: return
        if time.monotonic() < self._next: return
        self.search()

    def search(self):
        self._searching = True; self._token += 1; token = self._token
        saved = configured_ip(); self.panel.set_notice('')
        self.panel.searching = True

        def work():
            try: ip = self._find(saved) or ''
            except Exception: ip = ''
            self._result.emit(ip, token)
        threading.Thread(target=work, name='motion-find', daemon=True).start()

    def _done(self, ip, token):
        if token != self._token: return
        self._searching = False; self.panel.searching = False
        if ip and not self.panel.manual_off:
            try:
                self.panel.connect_to(ip); scanner_ip.remember(ip); self.panel.set_notice(''); self._misses = 0
                return
            except ValueError: pass
        self._next = time.monotonic() + RETRY_S[min(self._misses, len(RETRY_S) - 1)]; self._misses += 1
        self.panel.set_notice('Could not find the scanner on Wi-Fi. Make sure it is on the same network as this PC, '
                              'then type the address shown in its Wi-Fi settings and press Connect.')
        if not self._asked: self._asked = True; self.need_address.emit()

    def shutdown(self):
        self.timer.stop(); self._token += 1
