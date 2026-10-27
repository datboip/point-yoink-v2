# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small, opt-in CPU benchmark. The numerical worker only runs inside a capped service."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import uuid

from scanner.stages import resources

MEMORY_BYTES = 1024 ** 3
MAX_SECONDS = 30
BENCHMARK = "nearest_neighbors_v1"
_UNIT = re.compile(r"pointyoink-benchmark-[0-9a-f]{32}\.service\Z")


def new_unit():
    return "pointyoink-benchmark-%s.service" % uuid.uuid4().hex


def command(unit):
    """Return a service command. Never fall back to an uncapped Python process."""
    if not _UNIT.fullmatch(unit):
        raise ValueError("Invalid benchmark service name.")
    executable = shutil.which("systemd-run")
    if not executable or not shutil.which("systemctl"):
        raise ValueError("This benchmark needs systemd user services to enforce its limits.")
    info = resources.memory_info()
    if info is None:
        raise ValueError("Cannot read available RAM. The benchmark was not started.")
    total, available = info
    reserve = resources.desktop_reserve_bytes(total)
    if available - reserve < MEMORY_BYTES:
        raise ValueError("Not enough free RAM to run the benchmark and keep the desktop reserve.")
    quota = max(1, min(200, resources.cpu_count() * 75))
    source = Path(__file__).resolve().parents[2]
    return [executable, "--user", "--quiet", "--wait", "--pipe", "--collect",
            "--unit=" + unit, "--working-directory=" + str(source),
            "--property=MemoryMax=1G", "--property=MemorySwapMax=0",
            "--property=CPUQuota=%s%%" % quota, "--property=TasksMax=32",
            "--property=RuntimeMaxSec=%s" % MAX_SECONDS, "--property=TimeoutStopSec=2",
            "--property=KillMode=control-group", "--property=NoNewPrivileges=yes",
            "--property=ProtectControlGroups=yes", "--property=RestrictNamespaces=yes",
            "--setenv=PYTHONDONTWRITEBYTECODE=1",
            "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
            "--setenv=MKL_NUM_THREADS=1", "--setenv=BLIS_NUM_THREADS=1",
            "--setenv=NUMEXPR_NUM_THREADS=1", sys.executable,
            "-m", "app2.builder.hardware_benchmark", "--worker", unit]


def stop_command(unit):
    if not _UNIT.fullmatch(unit):
        raise ValueError("Invalid benchmark service name.")
    return [shutil.which("systemctl") or "systemctl", "--user", "stop", unit]


def status_command(unit):
    """Inspect only our service, including pending jobs that could start it later."""
    if not _UNIT.fullmatch(unit):
        raise ValueError("Invalid benchmark service name.")
    return [shutil.which("systemctl") or "systemctl", "--user", "show", "--all", "--no-pager",
            "--property=Id,LoadState,ActiveState,SubState,MainPID,ControlPID,Job", unit]


def service_stopped(unit, output):
    """A complete, successful status response must prove that no work remains."""
    if not isinstance(unit, str) or not _UNIT.fullmatch(unit) or not isinstance(output, str):
        return False
    state = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or key in state:
            return False
        state[key] = value.strip()
    required = {"Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlPID", "Job"}
    if not required.issubset(state) or state["Id"] != unit:
        return False
    if state["LoadState"] not in ("loaded", "not-found"):
        return False
    stopped = state["ActiveState"] == "inactive" and state["SubState"] == "dead"
    return stopped and state["MainPID"] == "0" and state["ControlPID"] == "0" and state["Job"] == ""


def _verify_cgroup(unit):
    """Refuse numerical imports until the live kernel controls are verified."""
    if not _UNIT.fullmatch(unit):
        raise ValueError("Invalid benchmark service name.")
    entries = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line[3:] for line in entries if line.startswith("0::")), "")
    if not relative.startswith("/") or Path(relative).name != unit:
        raise ValueError("The benchmark is not inside its own capped service.")
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    try:
        memory = int((group / "memory.max").read_text().strip())
        swap = int((group / "memory.swap.max").read_text().strip())
        quota, period = (int(n) for n in (group / "cpu.max").read_text().split())
    except (OSError, ValueError):
        raise ValueError("Cannot verify the benchmark's memory and CPU limits.") from None
    if not 0 < memory <= MEMORY_BYTES or swap != 0 or not 0 < quota <= 2 * period:
        raise ValueError("The benchmark's memory or CPU limit is not safe.")


def recommendation(total_bytes, available_bytes, logical_cpus):
    """Conservative starting limits, not predictions of scan speed or quality."""
    reserve = resources.desktop_reserve_bytes(total_bytes)
    allowed = math.floor(min(total_bytes * .20, available_bytes - reserve,
                             total_bytes - reserve, 8 * resources.GIB) / resources.GIB)
    if allowed < 1:
        return None
    cpu = 75 if logical_cpus >= 8 else 60 if logical_cpus >= 4 else 50
    return {"memory_gb": allowed, "cpu_percent": cpu}


def gpu_rows(output):
    """nvidia-smi reports MiB; keep GPU memory separate from system RAM."""
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 3:
            raise ValueError("GPU memory information was not available.")
        name, total, free = row
        try:
            total, free = float(total), float(free)
        except ValueError:
            raise ValueError("GPU memory information was not available.") from None
        if not name.strip() or not math.isfinite(total) or not math.isfinite(free) or not 0 <= free <= total or total <= 0:
            raise ValueError("GPU memory information was not available.")
        rows.append({"name": name.strip(), "total_gb": total / 1024, "free_gb": free / 1024})
    if not rows:
        raise ValueError("No NVIDIA GPU was reported.")
    return rows


def _hardware():
    model = platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = resources.memory_info()
    if info is None:
        raise ValueError("Cannot read available RAM.")
    return {"cpu": model, "logical_cpus": resources.cpu_count(),
            "ram_total_bytes": info[0], "ram_available_bytes": info[1],
            "platform": platform.system()}


def _worker(unit):
    _verify_cgroup(unit)
    info = resources.memory_info()
    if info is None or info[1] - resources.desktop_reserve_bytes(info[0]) < MEMORY_BYTES:
        raise ValueError("Available RAM changed. The benchmark cannot keep the desktop reserve.")
    # This is intentionally a single-threaded, fixed-size nearest-neighbor workload.
    import numpy as np
    from scipy.spatial import cKDTree
    hardware = _hardware()
    rng = np.random.default_rng(20261004)
    points = rng.random((100000, 3))
    queries = rng.random((50000, 3))
    elapsed = []
    checksum = 0.0
    for _ in range(3):
        started = time.perf_counter()
        tree = cKDTree(points)
        distances, _indices = tree.query(queries, k=8, workers=1)
        elapsed.append(time.perf_counter() - started)
        checksum += float(distances.sum())
    duration = sum(elapsed)
    return {"benchmark": BENCHMARK, "recorded_at": datetime.now(timezone.utc).isoformat(),
            "points": len(points), "queries": len(queries) * len(elapsed), "neighbors": 8,
            "repeats": len(elapsed), "elapsed_seconds": duration,
            "queries_per_second": len(queries) * len(elapsed) / duration,
            "checksum": checksum, "hardware": hardware,
            "recommendation": recommendation(hardware["ram_total_bytes"],
                                               hardware["ram_available_bytes"], hardware["logical_cpus"])}


def decode_result(output):
    try:
        data = json.loads(output)
        if data["benchmark"] != BENCHMARK or not isinstance(data["hardware"], dict):
            raise ValueError
        for key in ("elapsed_seconds", "queries_per_second", "checksum"):
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError
        if data["queries"] != 150000 or data["points"] != 100000 or data["neighbors"] != 8 or data["repeats"] != 3:
            raise ValueError
        hardware = data["hardware"]
        for key in ("ram_total_bytes", "ram_available_bytes", "logical_cpus"):
            if isinstance(hardware[key], bool) or not isinstance(hardware[key], int) or hardware[key] <= 0:
                raise ValueError
        if hardware["ram_available_bytes"] > hardware["ram_total_bytes"]:
            raise ValueError
        data["recommendation"] = recommendation(hardware["ram_total_bytes"],
                                                hardware["ram_available_bytes"], hardware["logical_cpus"])
        return data
    except (TypeError, ValueError, KeyError):
        raise ValueError("The benchmark returned an invalid result.") from None


def save_result(result):
    data = decode_result(json.dumps(result))
    path = resources.config_dir() / "hardware-benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="benchmark-", suffix=".tmp", delete=False) as f:
            name = f.name
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_result():
    path = resources.config_dir() / "hardware-benchmark.json"
    try:
        if path.stat().st_size > 65536:
            return None
        return decode_result(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(_worker(args.worker)))
        return 0
    except ImportError:
        print("This benchmark needs NumPy and SciPy in the app's Python environment.", file=sys.stderr)
    except Exception as error:
        print("Benchmark stopped: %s" % error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
