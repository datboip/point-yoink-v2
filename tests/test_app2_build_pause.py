"""Pause freezes a build and everything it started; Resume lets it go on; Stop works even while paused."""
import os, signal, subprocess, time
from app2.builder.runner import BuildRunner


def state(pid):
    try:
        with open("/proc/%d/stat" % pid) as f: return f.read().rsplit(")", 1)[1].split()[0]
    except OSError: return "gone"


def make():
    r = BuildRunner.__new__(BuildRunner); r.cancelled = False; r.backend = type("B", (), {"cancel_grace": 1})()
    r.p = subprocess.Popen(["sh", "-c", "sleep 30 & sleep 30 & wait"], start_new_session=True); time.sleep(0.4)
    return r


def test_pause_stops_the_whole_tree_and_resume_goes_on():
    r = make()
    try:
        tree = r._tree(r.p.pid); assert len(tree) >= 3                          # the shell and its two sleeps
        assert r.pause() and r.paused
        time.sleep(0.2); assert all(state(q) == "T" for q in tree)
        assert not r.pause()                                                     # already paused
        assert r.resume() and not r.paused
        time.sleep(0.2); assert all(state(q) in ("S", "R") for q in tree)
    finally:
        os.killpg(r.p.pid, signal.SIGKILL); r.p.wait()


def test_stop_while_paused_ends_the_build():
    r = make()
    try:
        r.pause(); r.cancel(); r.p.wait(timeout=5)
        assert r.cancelled and not r.paused and r.p.poll() is not None
    finally:
        try: os.killpg(r.p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
