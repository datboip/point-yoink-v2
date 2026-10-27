from types import SimpleNamespace as NS
from app2.qt.shell import MainWindow


def test_an_uncaught_error_toasts_once_a_minute():
    said = []
    w = NS(_error_said=0.0, say=lambda *a, **k: said.append((a, k)), _menu_action=lambda a: None)
    MainWindow._uncaught_toast(w, "boom"); MainWindow._uncaught_toast(w, "again")
    assert len(said) == 1 and said[0][1]["action"][0] == "Report a problem"


def test_low_disk_toasts_then_waits(tmp_path, monkeypatch):
    import shutil
    said = []
    w = NS(projects_dir=str(tmp_path), say=lambda *a, **k: said.append(a), DISK_LOW_GB=5.0)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: NS(free=2 * 1024 ** 3))
    MainWindow._check_disk(w, 100.0); MainWindow._check_disk(w, 200.0); assert len(said) == 1 and "2.0 GB" in said[0][0]
    MainWindow._check_disk(w, 800.0); assert len(said) == 2
