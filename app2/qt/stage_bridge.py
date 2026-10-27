# SPDX-License-Identifier: AGPL-3.0-or-later
"""From the Build step's worker threads (the "stages-edit" engine thread, the "stages-load" view loader) to the page:
every emit(kind, payload) becomes a Qt signal on the window's thread, the same way SessionBridge does it.
  queued       stages_result: every result, in order
  latest wins  stages_status, stages_busy, stage_view: one mailbox per kind (a page that stalls for a moment gets the
               newest status or the newest loaded result, never a backlog)"""
import collections, threading
from PySide6.QtCore import QObject, Signal, Qt, QCoreApplication, QEvent

QUEUED = ("stages_result",)
LATEST = ("stages_status", "stages_busy", "stage_view")
_EMPTY = object()


class StageBridge(QObject):
    stages_status = Signal(object)
    stages_busy = Signal(object)
    stages_result = Signal(object)
    stage_view = Signal(object)
    _poke = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._closed = False
        self._sig = {k: getattr(self, k) for k in QUEUED + LATEST}
        self._box = {}; self._fifo = collections.deque(); self._lock = threading.Lock(); self.counts = {}
        self._poke.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def __call__(self, kind, payload):
        # Serialize the final poke with retirement: no native signal can race close.
        with self._lock:
            if self._closed: return
            if kind not in self._sig: raise KeyError(kind)
            if kind in LATEST: waiting = kind in self._box; self._box[kind] = payload
            else: self._fifo.append((kind, payload)); waiting = False
            if not waiting: self._poke.emit(kind)

    def close(self):
        """Retire on the GUI thread before dropping a worker or deleting its owner."""
        with self._lock:
            if self._closed: return
            self._closed = True
            self._box.clear(); self._fifo.clear(); self._sig.clear()
        self._poke.disconnect(self._deliver)
        QCoreApplication.removePostedEvents(self, QEvent.Type.MetaCall)

    def _deliver(self, kind):
        with self._lock:
            if self._closed: return
            if kind in LATEST: payload = self._box.pop(kind, _EMPTY)
            else: kind, payload = self._fifo.popleft() if self._fifo else (kind, _EMPTY)
        if payload is not _EMPTY:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self._sig[kind].emit(payload)

