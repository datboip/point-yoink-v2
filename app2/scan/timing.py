# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in live timing and raw color capture, for finding where color frames are lost.

POINTYOINK_TIMING=<file>: one line per event, "<monotonic s> <stage> <thread>".
POINTYOINK_RAW_COLOR=<folder>: each color JPEG as it arrives, named by its arrival time, up to
POINTYOINK_RAW_COLOR_FRAMES (default 1800).
Both are off unless set; the hooks then cost one attribute check.
"""
import os
import threading
import time

PATH = os.environ.get('POINTYOINK_TIMING')
RAW = os.environ.get('POINTYOINK_RAW_COLOR')
RAW_LIMIT = int(os.environ.get('POINTYOINK_RAW_COLOR_FRAMES', 1800))   # about a minute at 30 fps, then saving stops
_saved = 0
_lock = threading.Lock()
_file = None


def mark(stage):
    global _file
    if not PATH:
        return
    line = '%.6f %s %s\n' % (time.monotonic(), stage, threading.current_thread().name)
    with _lock:
        if _file is None:
            _file = open(PATH, 'a', buffering=1 << 16)
        _file.write(line)


def save_color(jpeg, t):
    global _saved
    if not RAW or _saved >= RAW_LIMIT:
        return
    _saved += 1
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, '%.6f.jpg' % t), 'wb') as f:
        f.write(jpeg)
