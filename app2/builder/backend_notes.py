# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU-only builds are expected when CUDA is unavailable."""
import shutil
import threading

_available = None
_started = False
_lock = threading.Lock()


def cuda_available():
    global _started, _available
    if _available is not None:
        return _available
    if not shutil.which("nvidia-smi"):
        _available = False
        return False
    with _lock:
        if not _started:
            _started = True
            def check():
                global _available
                try:
                    import open3d.core as core
                    _available = bool(core.cuda.is_available())
                except Exception:
                    pass  # Unknown capability keeps a real fallback warning visible.
            threading.Thread(target=check, daemon=True, name="cuda-capability").start()
    return _available


def expected_cpu_warning(text):
    return str(text).startswith("GPU was requested, but fusion ran on the CPU") and cuda_available() is False
