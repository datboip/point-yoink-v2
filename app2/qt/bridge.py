# SPDX-License-Identifier: AGPL-3.0-or-later
"""From the scan session's threads to the window: every emit(kind, payload) becomes a Qt signal of that kind, delivered
on the window's thread (queued); no widget is ever touched off it.

Two ways through:
  queued       state, range, gain, log, build_line, closed, error: every payload is delivered, in order.
  latest wins  status, sides, live, pose, cloud, model_image, side_mesh, turn, build: one mailbox per kind. A new payload replaces
               one the window has not taken yet, so a window that stalls for a moment never works through a backlog
               of pictures or clouds; it just gets the newest. (cloud and model_image carry a sequence number, so the
               model panel can tell which of the two came last even when their deliveries cross.)"""
import collections, threading
from PySide6.QtCore import QObject, Signal, Qt

QUEUED = ("state", "range", "gain", "log", "build_line", "closed", "error")
LATEST = ("status", "sides", "live", "pose", "cloud", "model_image", "side_mesh", "turn", "build")
_EMPTY = object()


class SessionBridge(QObject):
    state = Signal(object)
    range_moved = Signal(object)
    gain_moved = Signal(object)
    log = Signal(object)
    build_line = Signal(object)
    closed = Signal(object)
    error = Signal(object)
    status = Signal(object)
    sides = Signal(object)
    live = Signal(object)
    pose = Signal(object)
    cloud = Signal(object)
    model_image = Signal(object)
    side_mesh = Signal(object)
    turn = Signal(object)
    build = Signal(object)
    _poke = Signal(str)

    def __init__(self):
        super().__init__()
        self._sig = {"state": self.state, "range": self.range_moved, "gain": self.gain_moved, "log": self.log,
                     "build_line": self.build_line, "closed": self.closed, "error": self.error, "status": self.status,
                     "sides": self.sides, "live": self.live, "pose": self.pose, "cloud": self.cloud,
                     "model_image": self.model_image, "side_mesh": self.side_mesh, "turn": self.turn, "build": self.build}
        self._box = {}; self._fifo = collections.deque(); self._lock = threading.Lock(); self.counts = {}
        self._poke.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def __call__(self, kind, payload):
        """The session's emit: any thread. Everything reaches the window through one queued poke per payload (or per
        empty mailbox), so the signals below always fire on the window's thread, whatever they are connected to."""
        if kind not in self._sig: raise KeyError(kind)
        with self._lock:
            if kind in LATEST: waiting = kind in self._box; self._box[kind] = payload
            else: self._fifo.append((kind, payload)); waiting = False
        if not waiting: self._poke.emit(kind)

    def _deliver(self, kind):
        with self._lock:
            if kind in LATEST: payload = self._box.pop(kind, _EMPTY)
            else: kind, payload = self._fifo.popleft() if self._fifo else (kind, _EMPTY)
        if payload is not _EMPTY:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self._sig[kind].emit(payload)
