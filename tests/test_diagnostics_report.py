"""The report: sections add and remove exactly their own text, anonymize hides what it must, and the window is honest about being a copy."""
import os
import socket
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XMODIFIERS"] = "@im=none"
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink_report_")) / "config.json")

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from app2.diagnostics import report as R
from app2.diagnostics.report_window import ReportWindow
from app2.qt import theme

DATA = dict(system=["2026-10-02 INFO === PointYoink 2.0.0-pre.7 starting", "OpenGL 3.3, NVIDIA GeForce RTX 3080"],
            settings={"cut margin mm": 2.8, "preset": "today", "line-up hints": "09282026222119:flipped", "empty": None},
            project={"sides": 12, "left out": 1, "frames": 13967},
            popover="Open project: /home/alice/revopoint-scans-models/ProjectLive09282026192332 has 12 sides",
            applog=["line %d /home/alice/x" % i for i in range(400)],
            error="2026 CRITICAL uncaught error (main thread): boom\nTraceback...\nValueError: boom",
            steplog="side 09282026222119 failed: cut removed every triangle")
CTX = dict(home="/home/alice", user="alice", host="testbox", roots=["/home/alice/revopoint-scans-models"],
           projects=["/home/alice/revopoint-scans-models/ProjectLive09282026192332"])
MARK = {"system": "RTX 3080", "settings": "cut margin mm: 2.8", "project": "frames: 13967", "popover": "has 12 sides", "applog": "line 399",
        "error": "ValueError: boom", "steplog": "cut removed every triangle", "note": "my own words"}


def test_each_section_adds_and_removes_only_its_own_text():
    for key in R.KEYS:
        only = R.build_report(DATA, {key}, anonymize=False, note="my own words")
        assert MARK[key] in only
        for other, mark in MARK.items():
            if other != key: assert mark not in only, (key, other)
    assert R.build_report(DATA, set(), anonymize=False) == "PointYoink problem report\n"


def test_the_log_is_cut_to_the_last_300_lines_and_empty_settings_are_skipped():
    t = R.build_report(DATA, {"applog", "settings"}, anonymize=False)
    assert "line 99 " not in t and "line 100 " in t and "empty" not in t


def test_anonymize_hides_paths_names_and_ids_but_keeps_the_facts():
    t = R.build_report(DATA, set(R.KEYS), anonymize=True, scrub_context=CTX, note="I am alice")
    for gone in ("alice", "testbox", "09282026192332", "09282026222119", "/home/"): assert gone not in t, gone
    for kept in ("RTX 3080", "cut margin mm: 2.8", "frames: 13967", "ValueError: boom", "cut removed every triangle"): assert kept in t


@pytest.fixture
def window(monkeypatch):
    app = QApplication.instance() or QApplication([]); theme.install(app)
    def no_network(*a, **k): raise AssertionError("the report window must not use the network")
    monkeypatch.setattr(socket, "socket", no_network)
    w = ReportWindow(lambda: (DATA, CTX)); w.show(); app.processEvents()
    yield w
    w.deleteLater(); app.processEvents()


def test_the_window_says_it_is_a_copy_and_starts_anonymized_with_every_section_on(window):
    assert "copy of your report" in window.warning.text() and "read it" in window.warning.text() and "Nothing has been sent" in window.warning.text()
    assert window.anon.isChecked() and set(window.boxes) == set(R.KEYS) and all(c.isChecked() for c in window.boxes.values())
    assert "alice" not in window.text()


def test_unticking_a_section_removes_it_and_anonymize_off_shows_the_raw_text(window):
    window.boxes["popover"].setChecked(False)
    assert "has 12 sides" not in window.text() and "RTX 3080" in window.text()
    window.anon.setChecked(False)
    assert "/home/alice/x" in window.text()


def test_copy_uses_exactly_what_is_in_the_box_after_editing(window):
    window.preview.setPlainText("only this")
    window.copy()
    assert QGuiApplication.clipboard().text() == "only this"
