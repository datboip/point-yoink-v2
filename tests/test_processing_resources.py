"""Resource budgets use small synthetic jobs and disposable config folders."""
import fcntl
import json
from pathlib import Path
import sys
import threading
import time

import pytest
from scanner.stages import resources as R
from scanner.stages.runner import Cancelled, Runner, StepError


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("POINTYOINK_STAGES_WRAP", raising=False)
    monkeypatch.setattr(R, "memory_info", lambda: (64 * R.GIB, 40 * R.GIB))
    monkeypatch.setattr(R, "cpu_count", lambda: 24)


def test_default_and_saved_limits(tmp_path):
    assert R.budget().memory_bytes == R.default_memory_gb() * R.GIB
    R.save(20, 50)
    assert R.load() == {"memory_gb": 20, "cpu_percent": 50}
    b = R.budget()
    assert b.memory_bytes == 20 * R.GIB
    assert (b.cpu_threads, b.cpu_quota) == (12, 1200)
    assert R.settings_path().parent == tmp_path
    assert not list(tmp_path.glob("*.tmp"))


def test_available_memory_clamps_even_a_large_saved_budget(monkeypatch):
    R.save(40, 75)
    monkeypatch.setattr(R, "memory_info", lambda: (64 * R.GIB, 18 * R.GIB))
    assert R.budget().memory_bytes == 2 * R.GIB
    monkeypatch.setattr(R, "memory_info", lambda: (64 * R.GIB, 15 * R.GIB))
    with pytest.raises(ValueError, match="desktop reserve"):
        R.budget()


def test_missing_memory_information_fails_closed(monkeypatch):
    monkeypatch.setattr(R, "memory_info", lambda: None)
    with pytest.raises(ValueError, match="Cannot read"):
        R.budget()
    assert R.memory_max_gb() == 0


@pytest.mark.parametrize("value", [0, -1, 1000, float("nan"), float("inf"), True, "wrong"])
def test_invalid_memory_save_does_not_change_settings(value):
    R.save(3, 75)
    before = R.settings_path().read_bytes()
    with pytest.raises(ValueError):
        R.save(value, 75)
    assert R.settings_path().read_bytes() == before


@pytest.mark.parametrize("value", [0, 100, float("nan"), float("inf"), True])
def test_invalid_cpu_limit(value):
    with pytest.raises(ValueError):
        R.save(3, value)


def test_corrupt_settings_are_conservative():
    R.settings_path().write_text('{"memory_gb": NaN, "cpu_percent": 100}')
    assert R.budget().memory_bytes == R.default_memory_gb() * R.GIB


def test_cli_can_only_reduce_and_fractional_limit_rounds_down():
    R.save(3, 75)
    assert R.budget(12).memory_bytes == 3 * R.GIB
    assert R.budget(1.1).memory_bytes <= 1.1 * R.GIB
    assert R.budget(1.1).memory_bytes > 1.09 * R.GIB


@pytest.mark.parametrize("stage_request", [None, 3, 4, 6, 10, 12, 100])
def test_every_stage_uses_user_budget_including_poisson_and_previews(stage_request):
    args = Runner().argv(["python", "job.py"], stage_request, False)
    assert "MemoryMax=%d" % (R.default_memory_gb() * R.GIB) in args
    assert "MemorySwapMax=0" in args
    assert "OOMPolicy=kill" in args
    assert "CPUQuota=1800%" in args


def test_user_can_raise_budget_past_old_stage_cap():
    R.save(20, 75)
    assert "MemoryMax=%d" % (20 * R.GIB) in Runner().argv(["python"], 10, False)


def test_custom_wrapper_cannot_bypass_caps():
    with pytest.raises(StepError, match="wrappers cannot enforce"):
        Runner(wrap="sh wrapper.sh {gb}").argv(["python"], 10, True)


def test_isolated_locks_and_environment(tmp_path):
    r = Runner()
    assert Path(r.gpu_lock_path).parent == tmp_path
    assert Path(r.processing_lock_path).parent == tmp_path
    e = r.env({"OMP_NUM_THREADS": "999", "SOME_OTHER_VALUE": 5})
    assert e["OMP_NUM_THREADS"] == "18"
    assert e["SOME_OTHER_VALUE"] == "5"


def test_waiting_for_processing_slot_is_cancellable():
    r = Runner()
    with open(r.processing_lock_path, "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = []
        def work():
            try:
                with r._processing_slot():
                    result.append("entered")
            except Cancelled:
                result.append("cancelled")
        t = threading.Thread(target=work)
        t.start()
        time.sleep(.05)
        r.cancel()
        t.join(3)
        assert not t.is_alive()
        assert result == ["cancelled"]


def test_memory_rechecked_after_wait_and_error_releases_slot(monkeypatch):
    r = Runner()
    monkeypatch.setattr(R, "memory_info", lambda: (64 * R.GIB, 1 * R.GIB))
    with pytest.raises(StepError, match="desktop reserve"):
        r.run([sys.executable, "-c", "raise AssertionError('must not start')"])
    with open(r.processing_lock_path, "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_configured_mesh_backend_reads_only_isolated_config(tmp_path, monkeypatch):
    from scanner.stages.methods import mesh
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    (tmp_path / "stages.json").write_text(json.dumps({"pml_python": str(fake_python)}))
    monkeypatch.delenv("POINTYOINK_PML_PY", raising=False)
    assert mesh.pml_python() == str(fake_python)


def test_child_virtual_limit_does_not_override_resident_limit():
    assert Runner().env()["POINTYOINK_MEM_CAP_GB"] == "100000"


def test_default_limit_scales_with_the_machine_and_is_never_a_build_killing_3_gib_on_a_big_one():
    G = R.GIB
    assert R.default_memory_gb((62 * G, 50 * G)) == 16                  # a big PC: enough for Fused (jobs ask for up to 8 GiB, several workers)
    assert R.default_memory_gb((32 * G, 25 * G)) >= 8
    assert R.default_memory_gb((8 * G, 6 * G)) == 3                     # a small PC stays small
    assert R.default_memory_gb(None) in (3, 16)


@pytest.mark.parametrize("total,available,reserve,maximum,budget", [
    (4, 3, 2, 2, 1), (8, 6, 2, 6, 3), (16, 12, 4, 12, 6),
    (32, 24, 8, 24, 12), (64, 40, 16, 48, 16),
])
def test_budgets_scale_to_small_and_large_pcs(monkeypatch, total, available, reserve, maximum, budget):
    monkeypatch.setattr(R, "memory_info", lambda: (total * R.GIB, available * R.GIB))
    assert R.desktop_reserve_bytes(total * R.GIB) == reserve * R.GIB
    assert R.memory_max_gb() == maximum
    assert R.budget().memory_bytes == budget * R.GIB                      # the default scales with the machine
    R.save(maximum, 75)
    assert R.budget().memory_bytes == (available - reserve) * R.GIB


def test_this_desktop_still_reserves_more_than_15_gib(monkeypatch):
    total = 67325026304
    monkeypatch.setattr(R, "memory_info", lambda: (total, 40 * R.GIB))
    assert R.desktop_reserve_bytes(total) >= 15 * R.GIB
    R.save(R.memory_max_gb(), 80)
    assert R.budget().memory_bytes <= 40 * R.GIB - 15 * R.GIB


def test_small_pc_still_refuses_when_reserve_cannot_be_kept(monkeypatch):
    monkeypatch.setattr(R, "memory_info", lambda: (8 * R.GIB, 2 * R.GIB))
    with pytest.raises(ValueError, match="desktop reserve"):
        R.budget()
