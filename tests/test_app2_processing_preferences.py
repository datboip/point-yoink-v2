"""Preferences remain usable by name and cannot start an uncapped benchmark."""
import fcntl

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QScrollArea, QPushButton
from app2.builder import hardware_benchmark as benchmark
from app2.qt.preferences_dialog import PreferencesDialog
from app2.qt import theme
from scanner.stages import resources

GIB = 1024 ** 3


@pytest.fixture
def dialog(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(resources, "memory_info", lambda: (64 * GIB, 40 * GIB))
    monkeypatch.setattr(resources, "cpu_count", lambda: 24)
    monkeypatch.setattr(PreferencesDialog, "_start_gpu_query", lambda self: None)
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    window = PreferencesDialog()
    window.show()
    app.processEvents()
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def processing_tab(dialog):
    names = {dialog.tabs.tabText(i): i for i in range(dialog.tabs.count())}
    assert "Save folders" in names and "Processing" in names
    dialog.tabs.setCurrentIndex(names["Processing"])
    QApplication.processEvents()
    return dialog.tabs.currentWidget()


def test_folder_controls_and_processing_fit_small_window(dialog):
    tab = processing_tab(dialog)
    assert isinstance(tab, QScrollArea)
    assert set(dialog.labels) == {"projects", "exports", "screenshots", "recordings"}
    assert dialog.width() <= 1280 and dialog.height() <= 800
    for key in dialog.labels:
        assert dialog.findChild(QPushButton, "change_" + key + "_folder") is not None
    assert dialog.cpu_limit.maximum() == 80
    assert dialog.cpu_limit.value() == 75
    assert "64.0 GiB total" in dialog.memory_info.text()
    assert "16.0 GiB" in dialog.reserve_info.text()
    assert dialog._benchmark is None
    assert not resources.settings_path().exists()


def test_limits_change_only_when_explicitly_saved(dialog):
    processing_tab(dialog)
    dialog.memory_limit.setValue(6.5)
    dialog.cpu_limit.setValue(60)
    assert not resources.settings_path().exists()
    dialog.save_button.click()
    assert resources.load() == {"memory_gb": 6.5, "cpu_percent": 60}
    assert "Saved" in dialog.settings_status.text()


def test_recommendation_is_explicit_and_rechecks_ram(dialog, monkeypatch):
    processing_tab(dialog)
    assert not resources.settings_path().exists()
    monkeypatch.setattr(resources, "memory_info", lambda: (64 * GIB, 18 * GIB))
    dialog.apply_button.click()
    assert resources.load() == {"memory_gb": 2, "cpu_percent": 75}
    monkeypatch.setattr(resources, "memory_info", lambda: (64 * GIB, 16 * GIB))
    dialog._apply_recommendation()
    assert "not available" in dialog.settings_status.text()
    assert resources.load()["memory_gb"] == 2


def test_missing_systemd_shows_safe_failure(dialog, monkeypatch):
    processing_tab(dialog)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: None)
    dialog.benchmark_button.click()
    assert "systemd" in dialog.benchmark_status.text()
    assert dialog._benchmark is None
    assert dialog._benchmark_lock is None
    assert not resources.settings_path().exists()


def test_benchmark_refuses_to_overlap_processing_job(dialog, monkeypatch):
    processing_tab(dialog)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/usr/bin/" + name)
    with (resources.config_dir() / "processing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dialog.benchmark_button.click()
        assert "already running" in dialog.benchmark_status.text()
        assert dialog._benchmark is None
        assert dialog._benchmark_lock is None


def test_finished_result_does_not_apply_recommendation(dialog):
    processing_tab(dialog)
    dialog._show_result({"elapsed_seconds": .75, "queries_per_second": 200000})
    assert "200,000 queries/s" in dialog.benchmark_status.text()
    assert "synthetic" in dialog.benchmark_status.text()
    assert not resources.settings_path().exists()


def test_close_defers_until_owned_benchmark_stops(dialog, monkeypatch):
    called = []
    process = QProcess(dialog)
    dialog._benchmark = process
    dialog._benchmark_unit = benchmark.new_unit()
    lock = (resources.config_dir() / "processing.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dialog._benchmark_lock = lock
    monkeypatch.setattr(dialog, "_cancel_benchmark", lambda: called.append(dialog._benchmark_unit))
    dialog.accept()
    assert called == [dialog._benchmark_unit]
    assert dialog._benchmark_lock is lock and not lock.closed
    assert dialog.isVisible()
    dialog._finish_benchmark()
    assert lock.closed
    assert not dialog.isVisible()


def test_wait_connection_crash_keeps_lock_until_service_stops(dialog, monkeypatch):
    process = QProcess(dialog)
    dialog._benchmark = process
    dialog._benchmark_unit = benchmark.new_unit()
    lock = (resources.config_dir() / "processing.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dialog._benchmark_lock = lock
    called = []
    monkeypatch.setattr(dialog, "_cancel_benchmark", lambda: called.append(True))
    dialog._benchmark_finished(-9, QProcess.ExitStatus.CrashExit)
    assert called and not lock.closed
    dialog._stop_finished(0, QProcess.ExitStatus.NormalExit)
    assert lock.closed and dialog._benchmark is None


@pytest.mark.parametrize("total,available,reserve,maximum", [(8, 6, 2, 6), (16, 12, 4, 12)])
def test_small_pc_displays_and_saves_available_budget(dialog, monkeypatch, total, available, reserve, maximum):
    monkeypatch.setattr(resources, "memory_info", lambda: (total * GIB, available * GIB))
    small = PreferencesDialog()
    try:
        processing_tab(small)
        assert "%.1f GiB" % reserve in small.reserve_info.text()
        assert small.memory_limit.maximum() == maximum
        small.memory_limit.setValue(2)
        small.save_button.click()
        assert resources.load()["memory_gb"] == 2
    finally:
        small.close()
        small.deleteLater()
        QApplication.processEvents()


@pytest.mark.parametrize("load_state", ["not-found", "loaded"])
def test_crashed_waiter_closes_after_failed_stop_but_confirmed_inactive_service(dialog, monkeypatch, load_state):
    _check_failed_stop_status(dialog, monkeypatch, load_state=load_state, stopped=True)


@pytest.mark.parametrize("state,job,code,crashed", [
    ("active", "", 0, False),
    ("inactive", "7", 0, False),
    ("unknown", "", 0, False),
    ("inactive", "", 1, False),
    ("inactive", "", 0, True),
])
def test_crashed_waiter_keeps_lock_when_service_liveness_is_unknown(dialog, monkeypatch, state, job, code, crashed):
    _check_failed_stop_status(dialog, monkeypatch, state=state, job=job, code=code, crashed=crashed, stopped=False)


def _check_failed_stop_status(dialog, monkeypatch, load_state="loaded", state="inactive", job="", code=0, crashed=False, stopped=False):
    process = QProcess(dialog)
    dialog._benchmark = process
    unit = dialog._benchmark_unit = benchmark.new_unit()
    lock = (resources.config_dir() / "processing.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dialog._benchmark_lock = lock
    monkeypatch.setattr(dialog, "_cancel_benchmark", lambda: None)
    dialog._benchmark_finished(-9, QProcess.ExitStatus.CrashExit)
    dialog.accept()
    calls = []
    output = ("Id=%s\nLoadState=%s\nActiveState=%s\nSubState=%s\nMainPID=%s\nControlPID=0\nJob=%s\n" % (
        unit, load_state, state, "dead" if state == "inactive" else "running",
        "0" if state == "inactive" else "123", job))
    if state == "unknown":
        output = ""

    class StatusReply:
        def readAllStandardOutput(self):
            return output.encode()

        def deleteLater(self):
            pass

    def check_service(args, checking=False):
        calls.append(args)
        assert checking
        dialog._checking_service = True
        dialog._stopper = StatusReply()

    monkeypatch.setattr(dialog, "_run_service_control", check_service)
    try:
        dialog._stop_finished(5, QProcess.ExitStatus.NormalExit)
        assert calls == [benchmark.status_command(unit)]
        assert dialog._benchmark_lock is lock and not lock.closed
        assert dialog.isVisible()
        dialog._stop_finished(code, QProcess.ExitStatus.CrashExit if crashed else QProcess.ExitStatus.NormalExit)
        if stopped:
            assert lock.closed and dialog._benchmark is None
            assert not dialog.isVisible()
        else:
            assert not lock.closed and dialog._benchmark is process
            assert dialog.isVisible()
            assert dialog._benchmark_timer.isActive()
            assert "Processing remains paused" in dialog.benchmark_status.text()
    finally:
        if dialog._benchmark is not None:
            dialog._finish_benchmark()
