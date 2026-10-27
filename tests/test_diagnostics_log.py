"""The 2.0 log file: what it records, that the hooks pass errors on, and the crash marker."""
import os
import sys
import tempfile
import threading

import pytest

from app2.diagnostics import log as L


@pytest.fixture
def logdir(monkeypatch):
    d = tempfile.mkdtemp(prefix="pointyoink_log_")
    monkeypatch.setenv("POINTYOINK_LOG_DIR", d)
    yield d
    L.teardown()


def text(d): return open(os.path.join(d, "app2.log"), encoding="utf-8").read()


def test_header_prints_and_stderr_reach_the_file(logdir):
    L.setup("9.9.9", "(test)")
    print("project folder: /home/someone/scans/ProjectLive01012026000000", flush=True)
    sys.stderr.write("a warning\n"); sys.stderr.flush()
    t = text(logdir)
    assert "PointYoink 9.9.9 starting (test)" in t and "system:" in t
    assert "project folder: /home/someone/scans/ProjectLive01012026000000" in t and "WARNING a warning" in t


def test_uncaught_errors_are_logged_and_still_reach_the_old_hook(logdir):
    seen = []
    old = sys.excepthook; sys.excepthook = lambda *a: seen.append(a[1])
    try:
        L.setup("1"); err = ValueError("boom")
        sys.excepthook(ValueError, err, err.__traceback__)
        assert seen == [err] and "uncaught error (main thread): boom" in text(logdir)
        L.teardown()
    finally:
        sys.excepthook = old


def test_thread_errors_are_logged(logdir):
    L.setup("1")
    t = threading.Thread(target=lambda: 1 / 0, name="worker-x"); t.start(); t.join()
    assert "uncaught error (thread worker-x)" in text(logdir) and "ZeroDivisionError" in text(logdir)


def test_crash_marker_survives_until_a_clean_exit(logdir):
    assert L.crashed_last_time() is False
    L.setup("1"); L.teardown()                               # never wrote the clean-exit mark
    assert L.crashed_last_time() is True
    L.setup("1"); assert L.was_crash() is True and "did not end cleanly" in text(logdir)
    L._clean_exit(); L.teardown()
    assert L.crashed_last_time() is False


def test_the_file_rolls_over_and_the_tail_is_read_back(logdir):
    lg = L.setup("1"); h = L._state["handler"]
    assert h.maxBytes == 1_000_000 and h.backupCount == 3
    for i in range(500): lg.info("line %d", i)
    tail = L.tail(10)
    assert len(tail) == 10 and tail[-1].endswith("line 499")
