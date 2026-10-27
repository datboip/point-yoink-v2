# SPDX-License-Identifier: AGPL-3.0-or-later
"""How the app is linked to the scanner right now, for the ribbon's Scanner tile: set by the Import page (Wi-Fi receiver,
transfers) and read by the tile, which also asks the USB ids for the mode. One small shared record; no device access."""
import time
from collections import deque

state = dict(wifi=None, code=None, rate=0.0, t=0.0)      # wifi: None | 'waiting' | 'receiving'; rate: bytes/s of a transfer
rates = deque(maxlen=60)                                  # transfer speed history for the tile's graph (MB/s)


def set_wifi(mode, code=None):
    state.update(wifi=mode, code=code if mode else None, t=time.time())
    if not mode: state['rate'] = 0.0


def set_rate(bytes_per_s):
    state.update(rate=float(bytes_per_s or 0), t=time.time())


def sample():
    """Called by the tile every 2 s: the speed graph drops to 0 when no transfer has reported for a while."""
    if time.time() - state['t'] > 5: state['rate'] = 0.0
    rates.append(state['rate'] / (1 << 20))
