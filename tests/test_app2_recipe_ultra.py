"""Recipe chip: names the recipe Build uses and notices edits. Ultra Yoink: builds the picked recipes in turn, notes
each result, stops on a failure, and warns with the disk space before it starts."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication, QWidget
from stages import params as PR
from app2.qt.build_step import BuildStepPage
from app2.qt.ultra import UltraSheet


def page_with(recipe):
    p = SimpleNamespace(snap=SimpleNamespace(recipe=recipe))
    return p


def test_recipe_name_and_edited():
    r = PR.apply_preset({}, "revo"); r["preset"] = "revo"
    key, name, edited = BuildStepPage._recipe_info(page_with(r))
    assert (key, name, edited) == ("revo", "Detail Yoink", False)
    r2 = PR.apply_preset({}, "revo"); r2["preset"] = "revo"; r2.setdefault("params", {}).setdefault("fused", {})["point_distance"] = 0.3
    assert BuildStepPage._recipe_info(page_with(r2))[2] is True
    assert BuildStepPage._recipe_info(page_with({}))[:2] == ("today", "Classic Yoink")


class FakeSheet:
    def __init__(self): self.results = None; self.stopped = None; self.running = []
    def show_running(self, done=0, total=None, current=None): self.running.append((done, total, current))
    def show_results(self, results, stopped=False): self.results = results; self.stopped = stopped


def fake_page(tmp_path):
    calls = []
    p = SimpleNamespace(host_root=str(tmp_path), busy_call=None, calls=calls)
    p.running = lambda: False
    p.ultra_running = lambda: bool(getattr(p, "_ultra", None))
    p.call = lambda name, *a, then=None, fail=None: (calls.append((name, a)), then and then(None))
    p.start_run = lambda run: calls.append(("run", run.what))
    for n in ("start_ultra", "_ultra_next", "_ultra_done"): setattr(p, n, getattr(BuildStepPage, n).__get__(p))
    return p


def test_ultra_builds_each_in_turn_and_notes_results(tmp_path, monkeypatch):
    import app2.qt.build_step as BS
    monkeypatch.setattr(BS.QTimer, "singleShot", staticmethod(lambda _ms, fn: fn()))
    p = fake_page(tmp_path); sheet = FakeSheet()
    (tmp_path / "model.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nelement face 1234\nend_header\n")
    p.start_ultra(["today", "revo"], sheet)
    assert p.calls[:2] == [("preset", ("today",)), ("run", "onetap")]
    p._ultra_done(True)                                       # today finished
    assert p.calls[2:4] == [("preset", ("revo",)), ("run", "onetap")]
    p._ultra_done(True)
    assert sheet.stopped is False and [r["preset"] for r in sheet.results] == ["today", "revo"] and sheet.results[0]["faces"] == 1234
    assert not p.ultra_running()


def test_ultra_stops_on_failure(tmp_path):
    p = fake_page(tmp_path); sheet = FakeSheet()
    p.start_ultra(["today", "revo", "print"], sheet); p._ultra_done(False)
    assert sheet.stopped is True and len(sheet.results) == 1 and not p.ultra_running()
    assert ("preset", ("revo",)) not in p.calls


def test_sheet_warns_with_disk_and_blocks_when_too_little_free(monkeypatch):
    app = QApplication.instance() or QApplication([])
    pg = QWidget(); pg.host_root = "/"; pg.ultra_running = lambda: False; pg.call = lambda *a, **k: None
    d = UltraSheet(pg)
    plan = [dict(preset="today", to_run=3, est_s=40, est_bytes=90 * 2**20), dict(preset="revo", to_run=3, est_s=160, est_bytes=300 * 2**20)]
    d.show_plan(plan)
    assert "390 MB" in d.note.text() and d.go_b.isEnabled()
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 2**20))
    d._total(); assert not d.go_b.isEnabled()
    d.close(); d.deleteLater(); pg.deleteLater(); app.processEvents()     # no dialog left alive at exit (it crashed the interpreter's teardown)
