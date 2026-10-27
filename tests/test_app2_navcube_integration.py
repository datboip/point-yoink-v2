"""Recovered GL views keep their corner cubes clear of the fixed lower controls."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp()) / 'config.json')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import app2
from PySide6.QtWidgets import QApplication
from app2.qt.navcube import SIZE, CANVAS, OFFX, OFFY
import viewnav


def test_all_gl_views_own_cubes_without_unused_clearance():
    from app2.qt.edit_view import EditView
    from app2.qt.view3d import ModelView
    app = QApplication.instance() or QApplication([])
    views = [EditView(), EditView(), ModelView()]
    for view, inset in zip(views, [0, 0, 0]):
        view.resize(700, 600); view.nav_cube._place()
        assert view.nav_cube.parent() is view
        assert view.nav_cube.bottom_inset == inset
        assert view.nav_cube.geometry().bottom() + 1 == 600 - inset
        assert not view.nav_cube.isHidden()
        before = view.nav_cube.geometry()
        view.cam.orbit(20, 15); view.nav_cube.sync()
        assert view.nav_cube.geometry() == before
        assert np.allclose(view.nav_cube.rot(), view.cam.view[:3, :3])
        view.deleteLater()
    app.processEvents()


def test_cube_face_stops_scanner_following():
    from app2.qt.view3d import ModelView
    app = QApplication.instance() or QApplication([])
    view = ModelView(); assert view.following
    view.nav_cube.go_face('front')
    assert not view.following
    rotation = view.cam.view.copy()
    view.follow_scanner(np.eye(4), np.zeros(3))
    assert np.allclose(view.cam.view, rotation)
    view.deleteLater(); app.processEvents()


def test_rotating_axis_labels_stay_inside_reserved_widget():
    for az in range(0, 360, 15):
        for el in range(-90, 91, 15):
            rotation = viewnav.rot_from_az_el(az, el)
            points = [q for region in viewnav.regions(rotation, CANVAS, CANVAS)[1] for q in region['poly']]
            for a, b in viewnav.axis_segments(rotation, CANVAS, CANVAS)[1].values():
                vector = np.asarray(b) - a; length = np.linalg.norm(vector)
                if length < 2:
                    points.extend([np.asarray(a) + [-18, 4], np.asarray(a) + [-6, 16]])
                    continue
                center = np.asarray(b) + vector / length * 7
                points.extend([center - 6, center + 6])
            points = np.asarray(points) - [OFFX, OFFY]
            assert points.min() >= 0 and points.max() <= SIZE
