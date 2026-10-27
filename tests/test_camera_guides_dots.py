import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QRectF


def test_dots_and_aim_draw_together_and_dots_land_where_asked():
    app = QApplication.instance() or QApplication([])
    from app2.qt import camera_guides as g
    xy = np.array([[120.0, 90.0], [600.0, 300.0]]); uv = np.zeros((2, 2), int)
    data = dict(dots_xy=xy, dots_uv=uv, dots_levels=np.array([2, 1]), dots_removed=np.zeros(2, bool), size=[800, 600], aim=[400, 300])
    img = QImage(800, 600, QImage.Format.Format_ARGB32); img.fill(0xff000000); p = QPainter(img)
    g.paint(p, QRectF(0, 0, 800, 600), data, {'dots': True, 'box': False, 'depth_edge': False}, 16); p.end()
    assert img.pixelColor(120, 90).green() > 100 and img.pixelColor(600, 300).red() > 100   # Best green, OK yellow
