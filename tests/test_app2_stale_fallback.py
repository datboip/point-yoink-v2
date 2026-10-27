"""A stale step still shows its last stored result (the view is not empty after a cache-key change)."""
import json, os, tempfile, time, types
from app2.builder.stagehost import _newest_stored

def mk(d, key, when):
    k = os.path.join(d, "side.fuse@a", key); os.makedirs(k)
    open(os.path.join(k, "raw.ply"), "w").write("x")
    m = os.path.join(k, "meta.json"); json.dump({"outputs": {"mesh": {"file": "raw.ply"}}}, open(m, "w")); os.utime(m, (when, when))

def test_newest_wins_and_missing_is_none():
    d = tempfile.mkdtemp(); e = types.SimpleNamespace(store=types.SimpleNamespace(nodes=d)); st = types.SimpleNamespace(store="side.fuse@a")
    assert _newest_stored(e, st) is None
    mk(d, "old", time.time() - 100); mk(d, "new", time.time())
    f = _newest_stored(e, st)
    assert f["mesh"].endswith(os.path.join("new", "raw.ply"))

def test_missing_file_is_none():
    d = tempfile.mkdtemp(); e = types.SimpleNamespace(store=types.SimpleNamespace(nodes=d)); st = types.SimpleNamespace(store="side.fuse@a")
    mk(d, "k", time.time()); os.remove(os.path.join(d, "side.fuse@a", "k", "raw.ply"))
    assert _newest_stored(e, st) is None
