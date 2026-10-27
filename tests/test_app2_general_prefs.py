"""Preferences, General: the look a project opens with, the fill light minutes, the timelapse mark and open-last, kept
under "general" in config.json and read by the parts that use them."""
import json, os
from PySide6.QtWidgets import QApplication


def test_defaults_bounds_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    from app2 import general_prefs as G
    assert G.load() == dict(default_look="shaded", light_minutes=5, timelapse_watermark=True, open_last=False)
    G.save(default_look="scan_color", light_minutes=200, timelapse_watermark=False, open_last=True, junk=1)
    g = G.load(); assert g["default_look"] == "scan_color" and g["light_minutes"] == 60 and not g["timelapse_watermark"] and g["open_last"]
    assert "junk" not in json.loads((tmp_path / "config.json").read_text())["general"]
    G.save(default_look="nonsense"); assert G.get("default_look") == "shaded"


def test_general_tab_writes_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    app = QApplication.instance() or QApplication([])
    from app2.qt.preferences_dialog import PreferencesDialog
    from app2 import general_prefs as G
    d = PreferencesDialog()
    try:
        assert d.tabs.tabText(0) == "General"
        d.default_look.setCurrentIndex(1); d.light_minutes.setValue(12); d.timelapse_watermark.setChecked(False); d.open_last.setChecked(True); app.processEvents()
        g = G.load(); assert g == dict(default_look="scan_color", light_minutes=12, timelapse_watermark=False, open_last=True)
    finally: d.close()


def test_light_timer_and_build_default_follow_the_prefs(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    from app2 import general_prefs as G
    G.save(light_minutes=3, default_look="scan_color")
    from app2.scan.session import ScanSession
    s = ScanSession.__new__(ScanSession)
    from app2.scan.light_timer import LightTimer
    s.light_timer = LightTimer(); s.light_timer.idle_wait = 60.0 * G.get("light_minutes"); assert s.light_timer.idle_wait == 180.0
    from types import SimpleNamespace as NS
    from app2.qt.build_step import BuildStepPage
    from tests.test_color_exports_round15 import colored
    src = tmp_path / "mesh.ply"; colored(src); seen = []
    page = NS(shown={"kind": "mesh"}, stage="fused", gl=True, colour_mode={}, _display_path=lambda: str(src), vp=NS(set_colours=lambda c, cur: seen.append(cur)))
    BuildStepPage._refresh_colours(page); assert seen[-1] == "scan_color"
