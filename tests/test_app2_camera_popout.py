"""The detachable camera window keeps all three live feeds as it resizes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication

from app2.qt.camera_popout import CameraPopout
from app2.qt.scan_page import CAMERA_PARTS
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge


def test_three_feeds_stay_live_in_resizable_window():
    app = QApplication.instance() or QApplication([])
    window = CameraPopout(CAMERA_PARTS)
    frame = SimpleNamespace(ir=np.full((24, 32, 3), 18, np.uint8),
                            depth=np.full((24, 32, 3), 70, np.uint8),
                            colour=np.full((24, 32, 3), 130, np.uint8),
                            colour_only=False, zref_mm=220)
    window.show(); app.processEvents()
    window.set_live(frame, CAMERA_PARTS[0]); app.processEvents()
    assert all(tile.img is not None for tile in (window.ir, window.depth, window.colour))
    assert window.ir.note == 'Center region 220 mm'
    assert window.grid.itemAtPosition(0,0).widget() is window.ir
    window.resize(700, 650); app.processEvents()
    assert window.grid.itemAtPosition(1,0).widget() is window.depth
    frame.colour_only = True; frame.colour = np.full((24, 32, 3), 190, np.uint8)
    window.set_live(frame, CAMERA_PARTS[0])
    assert window.colour.img.pixelColor(0, 0).red() == 190
    window.close()


def test_scan_hides_main_cameras_until_popout_closes():
    app = QApplication.instance() or QApplication([])
    page = ScanPage(SessionBridge(), gl=False)
    page.resize(1280, 800); page.show(); app.processEvents()
    frame = SimpleNamespace(ir=np.full((24, 32, 3), 20, np.uint8),
                            depth=np.full((24, 32, 3), 50, np.uint8),
                            colour=np.full((24, 32, 3), 90, np.uint8),
                            colour_only=False, zref_mm=210, platter=False, cut=0)
    page.on_live(frame)
    from PySide6.QtCore import QPoint
    anchors=(page.left,page.model,page.sides,page.go,page.camera_bar,page.camera_scroll)
    geometry=[(w.mapTo(page,QPoint()),w.size()) for w in anchors]
    page.popout_b.click(); app.processEvents()
    assert page.camera_window.isVisible() and page.camera_detached.isVisible()
    assert page.camera_scroll.isVisible()
    assert [(w.mapTo(page,QPoint()),w.size()) for w in anchors]==geometry
    assert all(tile.img is None for tile in (page.ir, page.depth, page.colour))
    assert page.model.live_img is None
    assert all(tile.img is not None for tile in (page.camera_window.ir, page.camera_window.depth, page.camera_window.colour))
    frame.ir = np.full((24, 32, 3), 40, np.uint8)
    page.on_live(frame)
    assert page.ir.img is None and page.camera_window.ir.img.pixelColor(0, 0).red() == 40
    page.camera_window.close(); app.processEvents()
    assert page.camera_scroll.isVisible() and not page.camera_detached.isVisible()
    assert all(tile.img is not None for tile in (page.ir, page.depth))
    assert page.colour.img is None and page.colour.center   # the big view already shows live color
    assert page.model.live_img is not None
    page.close()
