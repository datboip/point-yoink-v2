"""An error inside the plus integration loop ends the process with a message instead of hanging: the frame feeder thread used to
wait forever on its semaphore while the pool's exit joined it (a 0.3 mm build sat idle for an hour on 2026-10-06)."""
import threading, types
import numpy as np
import pytest
import fuse


def test_an_error_in_the_loop_aborts_quickly_with_the_frame_named(monkeypatch):
    msgs = []; monkeypatch.setattr(fuse, "emit", lambda stage, **kw: msgs.append((stage, kw)))
    calls = {"n": 0}

    class Grid:                                              # stands in for Open3D's VoxelBlockGrid: the 3rd frame blows up
        def hashmap(self): return types.SimpleNamespace(capacity=lambda: 10, size=lambda: 1)
    fake_o3d = types.SimpleNamespace(t=types.SimpleNamespace(geometry=types.SimpleNamespace(VoxelBlockGrid=lambda **k: Grid())))
    fake_o3c = types.SimpleNamespace(float32=0, uint8=1, Tensor=lambda *a, **k: None, Device=lambda s: s)
    import sys
    monkeypatch.setitem(sys.modules, "open3d", fake_o3d); monkeypatch.setitem(sys.modules, "open3d.core", fake_o3c)
    fake_o3d.core = fake_o3c

    def load(job):
        import time
        calls["n"] += 1; time.sleep(0.02)                     # slow enough that the feeder is blocked on its semaphore when the error hits
        if calls["n"] == 3: raise RuntimeError("CUDA out of memory")
        return None                                          # a skipped frame: the loop goes on
    jobs = [("/x/f%d.dph" % i, {}, None, (1, 1, 0, 0), None, (8, 6), False, None) for i in range(40)]
    a = types.SimpleNamespace(voxel=0.3, plus_trunc=1.0, plus_block_count=0, plus_workers=2, depth_scale=0.1)
    done = threading.Event(); err = []

    def run():
        try: fuse._plus_integrate(a, jobs, load, "CPU:0", 8)
        except Exception as e: err.append(e)
        done.set()
    threading.Thread(target=run, daemon=True).start()
    assert done.wait(20), "the integration hung instead of failing"
    assert err and "out of memory" in str(err[0])
    assert any(st == "error" and "frame" in kw["msg"] and "0.4 mm" in kw["msg"] for st, kw in msgs)
