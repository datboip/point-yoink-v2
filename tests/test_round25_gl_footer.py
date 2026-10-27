"""Run under a private X display to check the real OpenGL viewport footer."""
import os
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from app2.qt.scan_page import ScanPage
from app2.qt.bridge import SessionBridge
from test_round25_ui import scene_state


@pytest.mark.parametrize('size',[(1280,800),(1920,1080)])
def test_real_gl_viewport_stays_above_distance_footer(tmp_path,size):
    app=QApplication.instance() or QApplication([])
    if app.platformName()=='offscreen':pytest.skip('Needs a private X display with an OpenGL context.')
    p=ScanPage(SessionBridge(),True);p.resize(*size);p.show()
    try:
        p.on_state(scene_state(tmp_path));view=p.model.view
        view.set_points(np.array([[0,0,300],[10,0,300],[0,10,300]],np.float32));p.model._pick_page()
        for _ in range(8):app.processEvents()
        p.relayout(*size);p.place_distance();app.processEvents()
        assert view.isValid() and view.isVisible()
        assert view.mapTo(p,QPoint(0,view.height())).y()<=p.distance_bar.y()
        assert p.model.overlay.geometry()==p.model.viewport.geometry()
        p.grab() # Exercise composition of the GL widget and sibling footer.
    finally:p.close();app.processEvents()
