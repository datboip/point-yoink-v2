"""Fallbacks: builds still run without a systemd user session (uncapped, and the log says so); live scanning
without CUDA fails with a plain message instead of an Open3D error."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
import pytest
from stages import runner as R, resources as RS


def budget(monkeypatch):
    monkeypatch.setattr(RS, "budget", lambda memory_gb=None: RS.Budget(8 * RS.GIB, 8.0, 4, 400))


def test_no_systemd_runs_uncapped_and_logs_it(tmp_path, monkeypatch):
    budget(monkeypatch); monkeypatch.setenv("POINTYOINK_NO_SCOPES", "1")
    r = R.Runner(); r.processing_lock_path = str(tmp_path / "p.lock"); r.gpu_lock_path = str(tmp_path / "g.lock")
    assert r.argv(["python3", "x.py"], 4, False)[:3] == ["nice", "-n", str(r.nice)]
    log = tmp_path / "step.log"
    text = r.run([sys.executable, "-c", "print('hi')"], log=str(log))[0]
    assert "hi" in text and R.NO_SCOPES in log.read_text() and "systemd-run" not in log.read_text()


def test_systemd_present_keeps_the_scope(monkeypatch):
    budget(monkeypatch); monkeypatch.delenv("POINTYOINK_NO_SCOPES", raising=False); monkeypatch.setattr(R, "_scopes", True)
    assert R.Runner().argv(["python3", "x.py"], 4, False)[:2] == ["systemd-run", "--user"]


def test_scope_probe_failure_means_no_scopes(monkeypatch):
    monkeypatch.delenv("POINTYOINK_NO_SCOPES", raising=False); monkeypatch.setattr(R, "_scopes", None)
    def boom(*a, **k): raise FileNotFoundError("systemd-run")
    monkeypatch.setattr(R.subprocess, "run", boom)
    assert R.scopes_available() is False


def test_live_scan_without_cuda_says_so(monkeypatch):
    o3c = pytest.importorskip("open3d.core")
    from app2.scan import tracking as T
    monkeypatch.setattr(o3c.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA GPU"): T.cuda_device()
    with pytest.raises(RuntimeError, match="NVIDIA GPU"): T.cuda_tracker(None, 400, 300)
