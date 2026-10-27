"""A side's frame directory is scanned once and the scan is kept while the directory's mtime stands: a frame added
(or removed) changes it and the next ask rescans."""
import os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scanner"))
from stages.project import Project


def _project(n_frames):
    root = tempfile.mkdtemp(prefix="scan-cache-")
    cache = os.path.join(root, "data", "10010101010101", "cache"); os.makedirs(cache); os.makedirs(os.path.join(root, "stages"))
    with open(os.path.join(cache, "pointyoink_track.json"), "w") as f: json.dump({}, f)
    for i in range(n_frames):
        with open(os.path.join(cache, "frame_%03d.dph" % i), "wb") as f: f.write(b"\0" * 8)
        with open(os.path.join(cache, "frame_%03d.irl" % i), "wb") as f: f.write(b"\0" * 8)
    return root, cache


def test_the_scan_is_reused_until_the_directory_changes(monkeypatch):
    root, cache = _project(2); p = Project(root); sid = "10010101010101"
    n = {"scans": 0}; real = os.scandir
    def counted(d):
        if str(d).endswith("cache"): n["scans"] += 1                 # glob() of the side folders scans too; only the frame directory counts
        return real(d)
    monkeypatch.setattr("stages.project.os.scandir", counted)
    assert len(p.sides()) == 1 and p.sides()[0].frames == 2 and p.dph_stats(sid)[0] == 2 and len(p._scan(sid)["irl"]) == 2
    assert n["scans"] == 1                                  # four asks, one scan
    p.refresh(); p.frames_fp(sid)
    assert n["scans"] == 1                                  # a refresh (new plan / status) does not rescan an unchanged directory
    with open(os.path.join(cache, "frame_002.dph"), "wb") as f: f.write(b"\0" * 8)
    os.utime(cache, ns=(os.stat(cache).st_mtime_ns + 1_000_000, os.stat(cache).st_mtime_ns + 1_000_000))   # a coarse clock still moves it
    assert p.sides()[0].frames == 3 and n["scans"] == 2    # the new frame shows up


def test_frames_fp_follows_the_frame_count():
    root, cache = _project(1); p = Project(root); sid = "10010101010101"
    a = p.frames_fp(sid)
    with open(os.path.join(cache, "frame_001.dph"), "wb") as f: f.write(b"\0" * 8)
    os.utime(cache, ns=(os.stat(cache).st_mtime_ns + 1_000_000, os.stat(cache).st_mtime_ns + 1_000_000))
    p.refresh()
    assert p.frames_fp(sid) != a
