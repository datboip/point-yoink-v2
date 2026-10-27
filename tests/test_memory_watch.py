"""The memory watchdog: reads this process and its children, warns on the app's share or on low RAM, once a minute."""
import os
from app2.memory_watch import usage, warning, Watch, GIB


def test_usage_counts_this_process_and_children():
    u = usage()
    if u is None: return                                         # no /proc here
    ours, total, available = u
    assert 0 < ours < total and 0 <= available <= total
    assert usage(os.getppid())[0] >= ours                        # the parent's tree holds ours


def test_warning_on_share_or_low_ram():
    total = 62 * GIB
    assert warning(2 * GIB, total, 40 * GIB) is None
    big = warning(36 * GIB, total, 8 * GIB); assert big and "holds 36 of the PC's 62 GB" in big
    low = warning(5 * GIB, total, 3 * GIB); assert low and "3.0 GB of RAM left" in low
    assert "Stop the build or coverage measure" in big


def test_watch_repeats_once_a_minute(monkeypatch):
    import app2.memory_watch as mw
    reading = [(36 * GIB, 62 * GIB, 8 * GIB)]
    monkeypatch.setattr(mw, "usage", lambda root_pid=None: reading[0])
    w = Watch()
    assert w.check(0.0); assert w.check(10.0) is None; assert w.check(61.0)
    reading[0] = (2 * GIB, 62 * GIB, 40 * GIB)
    assert w.check(62.0) is None and w.text is None
    reading[0] = (36 * GIB, 62 * GIB, 8 * GIB)
    assert w.check(63.0)                                         # cleared and back: said at once
