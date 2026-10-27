import pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from PySide6.QtWidgets import QApplication
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge
from app2.qt import ui_level
from app2.scan import settings
from test_round25_ui import scene_state, settle


@pytest.mark.simple_level
def test_simple_hides_setup_controls_and_advanced_brings_them_back(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings.save(ui_level=None)
    p = ScanPage(SessionBridge(), False); p.resize(1920, 1000); p.show(); p.on_state(scene_state(tmp_path)); settle(app, p)
    assert p.level == 'Simple'
    p.sides.tabs.setCurrentIndex(1); settle(app, p)
    c = p.controls
    for kept in (c.scan_mode, c.surface, c.range_row, p.go, c.led): assert kept.isVisible()
    for hidden in ui_level.advanced_widgets(p): assert not hidden.isVisible()
    assert not p.live_strip.color_group.isVisible()
    p.set_level('Advanced'); settle(app, p)
    assert settings.load()['ui_level'] == 'Advanced'
    for shown in (c.capture_mode, c.gain_row, c.save_color, c.capture_tools, p.camera_options.ir): assert shown.isVisible()
    assert p.live_strip.color_group.isVisible()


@pytest.mark.simple_level
def test_guide_steps_and_hide_guide_never_move_the_rows_below(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings.save(ui_level=None, first_scan_guide=True)
    p = ScanPage(SessionBridge(), False); p.resize(1920, 1000); p.show(); s = scene_state(tmp_path); p.on_state(s); settle(app, p)
    row = p.controls.range_row
    def where(): settle(app, p); return p.ir.geometry(), p.live_strip.geometry(), row.mapTo(p, row.rect().topLeft())
    first = where(); assert p.guide.isVisible()
    p._show_guide(False); assert where() == first and not p.guide.isVisible()
    p._show_guide(True); s.n_sides = 1; s.picked_has_frames = True; p.on_state(s)
    assert where() == first and p.guide.title.text() == 'Flip the part'


@pytest.mark.simple_level
@pytest.mark.parametrize('size', [(1280, 800), (1688, 1044), (2000, 820)])
def test_switching_simple_and_advanced_moves_nothing(tmp_path, size):
    from PySide6.QtCore import QPoint
    from app2.qt.shell import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow(lambda *a, **kw: None, gl=False, projects_dir=str(tmp_path), defer_new_project=True)
    w.resize(*size); w.show(); w.show_step(0); w.startup.hide(); p = w.scan
    try:
        p.on_state(scene_state(tmp_path, has_led=True)); p.sides.tabs.setCurrentIndex(1)
        def where():
            settle(app, p)
            return [(x.mapTo(p, QPoint()), x.size()) for x in (p.ir, p.depth, p.live_strip, p.model, p.sides, p.go)]
        p.set_level('Simple'); simple = where(); p.set_level('Advanced')
        assert where() == simple
    finally:
        w.close()
