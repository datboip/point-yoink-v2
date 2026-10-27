"""Admission, service confinement, and recorded results for the optional CPU check."""
import json
from pathlib import Path

import pytest
from app2.builder import hardware_benchmark as benchmark
from scanner.stages import resources

GIB = 1024 ** 3


@pytest.fixture
def safe_host(monkeypatch, tmp_path):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(resources, "memory_info", lambda: (64 * GIB, 40 * GIB))
    monkeypatch.setattr(resources, "cpu_count", lambda: 24)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/usr/bin/" + name)


def result():
    return {"benchmark": benchmark.BENCHMARK, "recorded_at": "2026-10-04T00:00:00+00:00",
            "points": 100000, "queries": 150000, "neighbors": 8, "repeats": 3,
            "elapsed_seconds": .9, "queries_per_second": 150000 / .9, "checksum": 123.4,
            "hardware": {"logical_cpus": 24, "ram_total_bytes": 64 * GIB,
                         "ram_available_bytes": 40 * GIB, "cpu": "Test CPU", "platform": "Linux"}}


def test_command_is_capped_service_with_no_uncapped_fallback(safe_host):
    unit = benchmark.new_unit()
    args = benchmark.command(unit)
    for required in ("--user", "--wait", "--pipe", "--property=MemoryMax=1G",
                     "--property=MemorySwapMax=0", "--property=CPUQuota=200%",
                     "--property=RuntimeMaxSec=30", "--property=KillMode=control-group",
                     "--property=ProtectControlGroups=yes", "--property=RestrictNamespaces=yes"):
        assert required in args
    assert args[-2:] == ["--worker", unit]
    assert benchmark.stop_command(unit)[-2:] == ["stop", unit]


@pytest.mark.parametrize("cores,quota", [(1, 75), (2, 150), (24, 200)])
def test_small_cpu_is_not_fully_consumed(safe_host, monkeypatch, cores, quota):
    monkeypatch.setattr(resources, "cpu_count", lambda: cores)
    assert "--property=CPUQuota=%d%%" % quota in benchmark.command(benchmark.new_unit())


def test_missing_systemd_and_insufficient_memory_refuse_launch(safe_host, monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="systemd"):
        benchmark.command(benchmark.new_unit())
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(resources, "memory_info", lambda: (64 * GIB, 16 * GIB))
    with pytest.raises(ValueError, match="desktop reserve"):
        benchmark.command(benchmark.new_unit())
    monkeypatch.setattr(resources, "memory_info", lambda: None)
    with pytest.raises(ValueError, match="available RAM"):
        benchmark.command(benchmark.new_unit())


@pytest.mark.parametrize("name", ["user.service", "pointyoink-benchmark-*.service", "../other.service", ""])
def test_cannot_stop_unowned_unit(name):
    with pytest.raises(ValueError):
        benchmark.stop_command(name)
    with pytest.raises(ValueError):
        benchmark.command(name)


def test_worker_checks_kernel_caps_before_work(monkeypatch):
    unit = benchmark.new_unit()
    data = {"/proc/self/cgroup": "0::/user.slice/" + unit,
            "memory.max": str(GIB), "memory.swap.max": "0", "cpu.max": "200000 100000"}
    monkeypatch.setattr(Path, "read_text", lambda path: data.get(str(path), data.get(path.name)))
    benchmark._verify_cgroup(unit)
    data["memory.max"] = str(2 * GIB)
    with pytest.raises(ValueError, match="not safe"):
        benchmark._verify_cgroup(unit)
    data["memory.max"] = str(GIB)
    data["cpu.max"] = "max 100000"
    with pytest.raises(ValueError, match="Cannot verify"):
        benchmark._verify_cgroup(unit)
    data["cpu.max"] = "200000 100000"
    data["memory.swap.max"] = "1"
    with pytest.raises(ValueError, match="not safe"):
        benchmark._verify_cgroup(unit)
    data["/proc/self/cgroup"] = "0::/user.slice/unrelated.service"
    with pytest.raises(ValueError, match="own capped service"):
        benchmark._verify_cgroup(unit)


def test_recommendation_keeps_headroom():
    assert benchmark.recommendation(64 * GIB, 40 * GIB, 24) == {"memory_gb": 8, "cpu_percent": 75}
    assert benchmark.recommendation(64 * GIB, 18 * GIB, 4) == {"memory_gb": 2, "cpu_percent": 60}
    assert benchmark.recommendation(64 * GIB, 16 * GIB, 2) is None


def test_gpu_memory_is_parsed_independently():
    rows = benchmark.gpu_rows('"NVIDIA GPU, Example", 10240, 8192\nOther GPU, 4096, 1024\n')
    assert rows == [{"name": "NVIDIA GPU, Example", "total_gb": 10, "free_gb": 8},
                    {"name": "Other GPU", "total_gb": 4, "free_gb": 1}]


@pytest.mark.parametrize("text", ["GPU, N/A, N/A", "GPU, nan, 100", "GPU, 10, 11", "", "GPU, 10"])
def test_gpu_unavailable_is_readable(text):
    with pytest.raises(ValueError):
        benchmark.gpu_rows(text)


def test_record_does_not_apply_limits(safe_host, tmp_path):
    benchmark.save_result(result())
    saved = benchmark.load_result()
    assert saved["queries"] == 150000
    assert saved["recommendation"] == {"memory_gb": 8, "cpu_percent": 75}
    assert not resources.settings_path().exists()
    assert (tmp_path / "hardware-benchmark.json").is_file()
    (tmp_path / "hardware-benchmark.json").write_text("not JSON")
    assert benchmark.load_result() is None


@pytest.mark.parametrize("key,value", [("elapsed_seconds", 0), ("queries_per_second", float("nan")),
                                       ("checksum", True), ("queries", 1)])
def test_invalid_measurements_are_not_saved(safe_host, key, value):
    data = result()
    data[key] = value
    with pytest.raises(ValueError, match="invalid result"):
        benchmark.decode_result(json.dumps(data))


@pytest.mark.parametrize("total,available,expected", [(8, 6, 1), (16, 12, 3), (32, 24, 6), (64, 40, 8)])
def test_recommendations_scale_without_changing_quality(total, available, expected):
    assert benchmark.recommendation(total * GIB, available * GIB, 4) == {"memory_gb": expected, "cpu_percent": 60}


def service_status(unit, **changes):
    values = {"Id": unit, "LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead",
              "MainPID": "0", "ControlPID": "0", "Job": ""}
    values.update(changes)
    return "\n".join(key + "=" + value for key, value in values.items()) + "\n"


def test_status_command_reads_only_the_owned_service(safe_host):
    unit = benchmark.new_unit()
    command = benchmark.status_command(unit)
    assert command == ["/usr/bin/systemctl", "--user", "show", "--all", "--no-pager",
                       "--property=Id,LoadState,ActiveState,SubState,MainPID,ControlPID,Job", unit]
    with pytest.raises(ValueError):
        benchmark.status_command("unrelated.service")


@pytest.mark.parametrize("load", ["loaded", "not-found"])
def test_explicit_inactive_or_absent_service_can_release_lock(load):
    unit = benchmark.new_unit()
    assert benchmark.service_stopped(unit, service_status(unit, LoadState=load))


@pytest.mark.parametrize("changes", [
    {"ActiveState": "active", "SubState": "running", "MainPID": "123"},
    {"ActiveState": "activating", "SubState": "start"},
    {"ActiveState": "deactivating", "SubState": "stop-sigterm"},
    {"ActiveState": "failed", "SubState": "failed"},
    {"LoadState": "error"}, {"MainPID": "123"}, {"ControlPID": "123"}, {"Job": "123"},
    {"Id": "unrelated.service"},
])
def test_active_or_uncertain_service_cannot_release_lock(changes):
    unit = benchmark.new_unit()
    assert not benchmark.service_stopped(unit, service_status(unit, **changes))


@pytest.mark.parametrize("output", ["", "Failed to connect to bus", "LoadState=not-found\nActiveState=inactive\n", None])
def test_incomplete_service_status_cannot_release_lock(output):
    assert not benchmark.service_stopped(benchmark.new_unit(), output)


def test_conflicting_status_fields_cannot_release_lock():
    unit = benchmark.new_unit()
    assert not benchmark.service_stopped(unit, service_status(unit) + "MainPID=5\n")
