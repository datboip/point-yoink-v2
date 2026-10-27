# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processing preferences and conservative admission limits. No Qt or numerical imports."""
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3
DEFAULT_MEMORY_GB = 3            # the floor of the default (see default_memory_gb)
DEFAULT_MEMORY_CAP_GB = 16      # a build gains little past this
DEFAULT_CPU_PERCENT = 75
MAX_PERCENT = 80
MIN_RESERVE_GB = 2


def config_dir():
    config = os.environ.get("POINTYOINK_CONFIG")
    return Path(config).expanduser().absolute().parent if config else Path.home() / ".config/pointyoink"


def settings_path():
    return config_dir() / "processing.json"


def memory_info():
    """Total and available physical RAM in bytes, or None when unavailable."""
    try:
        entries = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            entries[key] = int(value.split()[0]) * 1024
        total, available = entries["MemTotal"], entries["MemAvailable"]
        if total > 0 and 0 <= available <= total:
            return total, available
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return None


def cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def desktop_reserve_bytes(total):
    """Keep a quarter of physical RAM, with a 2 GiB floor on smaller PCs."""
    return max(MIN_RESERVE_GB * GIB, math.ceil(total * .25))


def memory_max_gb(info=None):
    info = memory_info() if info is None else info
    if info is None:
        return 0
    total, _available = info
    # Keep the same headroom in the UI, benchmark, and worker admission checks.
    return max(0, math.floor((total - desktop_reserve_bytes(total)) / GIB))


def default_memory_gb(info=None):
    """The memory limit used until the user saves one: half of what the desktop reserve leaves, between 3 and 16 GiB. A fixed 3 GiB
    killed real Fused builds (jobs ask for up to 8 GiB and run several workers); on a small PC this stays small."""
    maximum = memory_max_gb(info)
    return max(DEFAULT_MEMORY_GB, min(DEFAULT_MEMORY_CAP_GB, maximum // 2)) if maximum else DEFAULT_MEMORY_GB


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError("%s must be a number." % name)
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a number." % name) from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("%s must be between %g and %g." % (name, low, high))
    return value


def load():
    defaults = {"memory_gb": default_memory_gb(), "cpu_percent": DEFAULT_CPU_PERCENT}
    try:
        data = json.loads(settings_path().read_text())
        memory = _number(data["memory_gb"], "Memory limit", 1, 1048576)
        cpu = _number(data["cpu_percent"], "CPU limit", 10, MAX_PERCENT)
        return {"memory_gb": memory, "cpu_percent": int(cpu)}
    except (OSError, ValueError, KeyError, TypeError):
        return defaults


def save(memory_gb, cpu_percent, *, limits=None):
    maximum = memory_max_gb()
    if maximum < 1:
        raise ValueError("Cannot set a processing budget while RAM information is unavailable or the desktop reserve cannot be kept.")
    memory = _number(memory_gb, "Memory limit", 1, maximum)
    cpu = _number(cpu_percent, "CPU limit", 10, MAX_PERCENT)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="processing-", suffix=".tmp", delete=False) as f:
            name = f.name
            payload = {"memory_gb": memory, "cpu_percent": int(cpu)}
            try:
                old = json.loads(path.read_text())
                if old.get('gpu_jobs') in (1, 2): payload['gpu_jobs'] = old['gpu_jobs']
            except (OSError, ValueError, AttributeError): pass
            if limits is not None: payload.update(limits)
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class Budget:
    memory_bytes: int
    requested_gb: float
    cpu_threads: int
    cpu_quota: int

    @property
    def memory_gb(self):
        return self.memory_bytes / GIB


def budget(memory_gb=None):
    """Refresh at launch, after queue waits. A CLI limit may only reduce the saved budget."""
    settings = load()
    options = limit_options()
    if options['mode'] == 'auto':
        automatic = recommended_limits()
        if automatic is None: raise ValueError('Not enough available RAM to keep the desktop reserve.')
        settings.update(memory_gb=automatic['memory_gb'], cpu_percent=automatic['cpu_percent'])
    requested = settings["memory_gb"]
    if memory_gb is not None:
        requested = min(requested, _number(memory_gb, "Memory limit", .5, 1048576))
    info = memory_info()
    if info is None:
        raise ValueError("Cannot read available RAM. Processing was not started.")
    total, available = info
    reserve = desktop_reserve_bytes(total)
    limit = min(int(requested * GIB), total - reserve, available - reserve)
    # MiB precision rounds down. A fractional user limit must never be rounded up.
    limit = max(0, int(limit) // (1024 ** 2) * (1024 ** 2))
    if limit < GIB // 2:
        raise ValueError("Not enough available RAM to keep the desktop reserve. Close other programs and try again.")
    cores = cpu_count()
    quota = max(1, math.floor(cores * settings["cpu_percent"]))
    threads = max(1, math.floor(quota / 100))
    if options['mode'] == 'manual' and options.get('cpu_threads') is not None:
        threads = min(threads, options['cpu_threads'])
        quota = min(quota, threads * 100)
    return Budget(limit, requested, threads, quota)


def limit_options():
    """Additional controls; old two-field processing preferences remain supported."""
    try: data = json.loads(settings_path().read_text())
    except (OSError, ValueError): data = {}
    if not isinstance(data, dict): data = {}
    mode = data.get('limits_mode', 'manual')
    mode = mode if mode in ('manual', 'auto') else 'manual'
    value = data.get('cpu_threads')
    threads = value if type(value) is int and 1 <= value <= max(1, cpu_count()) else None
    jobs = data.get('gpu_jobs', 2)
    return dict(mode=mode, cpu_threads=threads, gpu_jobs=jobs if type(jobs) is int and jobs in (1, 2) else 2)


def recommended_limits():
    from app2.builder.hardware_benchmark import recommendation
    info = memory_info()
    if info is None: return None
    result = recommendation(*info, cpu_count())
    if result is None: return None
    return dict(result, cpu_threads=max(1, math.floor(cpu_count()*result['cpu_percent']/100)), gpu_jobs=2)


def save_limits(memory_gb, threads, gpu_jobs=2, mode='manual'):
    if mode not in ('auto', 'manual'): raise ValueError('Choose Auto or Manual.')
    if type(gpu_jobs) is not int or gpu_jobs not in (1, 2): raise ValueError('Choose one or two GPU jobs.')
    if mode == 'auto':
        result = recommended_limits()
        if result is None: raise ValueError('Not enough available RAM to keep the desktop reserve.')
        memory_gb, threads, cpu = result['memory_gb'], result['cpu_threads'], result['cpu_percent']
        gpu_jobs = result['gpu_jobs']
    else:
        maximum = max(1, math.floor(cpu_count()*MAX_PERCENT/100))
        if type(threads) is not int or not 1 <= threads <= maximum: raise ValueError('CPU threads must be between 1 and %d.' % maximum)
        cpu = max(10, min(MAX_PERCENT, math.ceil(threads*100/cpu_count())))
    save(memory_gb, cpu, limits=dict(limits_mode=mode, cpu_threads=threads, gpu_jobs=gpu_jobs))


def gpu_job_limit():
    return limit_options()['gpu_jobs']
