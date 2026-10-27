"""Every test run gets its own config folder unless a test sets one: step timings and settings written by tests
must never land in the real ~/.config/pointyoink (they did: cache hits saved as 0 s runs broke real estimates)."""
import os, tempfile

if not os.environ.get("POINTYOINK_CONFIG"):
    os.environ["POINTYOINK_CONFIG"] = os.path.join(tempfile.mkdtemp(prefix="pyk-test-config-"), "config.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "real_worker: saved-frame smoke using real engine and worker subprocesses")
    config.addinivalue_line("markers", "simple_level: start the Scan page in Simple")


import pytest


@pytest.fixture(autouse=True)
def advanced_controls(request, monkeypatch):
    """Most page tests drive every control, so pages open in Advanced (whatever config a test points at); a test of
    Simple itself marks simple_level."""
    if request.node.get_closest_marker("simple_level") is None:
        from app2.qt import ui_level
        monkeypatch.setattr(ui_level, "saved", lambda: "Advanced")
    yield
