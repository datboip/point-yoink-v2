"""Pause: the voxel cache and the live model go to disk in the background (2026-10-24). The cache write is split into a
CPU snapshot and a later write; a session's _defer_disk runs them in order on one thread and _flush_disk waits."""
import json, threading, time
import numpy as np


def test_snapshot_written_later_reads_back(tmp_path):
    from app2.scan import live_cache
    snap = dict(path=tmp_path / live_cache.NAME, metadata=json.dumps(dict(version=live_cache.VERSION, key='k')),
                keys=np.zeros((2, 3), np.int32), tsdf=np.ones(2, np.float32), weight=np.ones(2, np.float32),
                field_keys=np.zeros((1, 3), np.int32), field_best=np.zeros(1, np.float32), field_support=np.zeros(1, np.float32))
    assert live_cache.write_snapshot(snap) and snap['path'].exists() and not snap['path'].with_suffix('.tmp').exists()
    with np.load(snap['path']) as d: assert d['tsdf'].tolist() == [1.0, 1.0] and json.loads(str(d['metadata']))['key'] == 'k'


def test_defer_disk_runs_in_order_and_flush_waits():
    from app2.scan.session import ScanSession
    s = ScanSession.__new__(ScanSession); s._log = lambda *a: None
    done = []
    s._defer_disk(lambda: (time.sleep(0.1), done.append(1)))
    s._defer_disk(lambda: done.append(2))
    assert done == []                                  # Pause came back before the writes
    s._flush_disk(); assert done == [1, 2]
    s._defer_disk(lambda: 1 / 0); s._flush_disk()      # a failing write is logged, the writer lives on
    s._defer_disk(lambda: done.append(3)); s._flush_disk(); assert done == [1, 2, 3]
