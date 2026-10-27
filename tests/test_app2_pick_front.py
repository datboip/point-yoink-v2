"""A pick takes the surface nearest the camera near the click, not the far side seen through a gap."""
import numpy as np
from app2.qt.edit_view import EditView

def test_front_beats_far_side_through_gap():
    img = np.ones((20, 20), float); img[10, 10] = 0.9                    # far point under the click
    img[10, 12] = 0.4                                                    # near surface two pixels away
    assert EditView.front_pixel(img, 10, 10, 3) == (10, 12, 0.4)

def test_outside_window_ignored_and_empty_unchanged():
    img = np.ones((20, 20), float); img[10, 10] = 0.9; img[2, 2] = 0.1
    assert EditView.front_pixel(img, 10, 10, 3) == (10, 10, 0.9)
    assert EditView.front_pixel(np.ones((5, 5)), 2, 2, 3)[2] == 1.0


def test_front_point_picks_the_front_layer_and_never_empty_space():
    from app2.gl_core import Camera
    cam = Camera(); cam.view = np.eye(4); cam.view[2, 3] = -100.0; cam.bound = None; cam.near, cam.far = 1.0, 500.0
    w, h = 200, 200; pj = np.asarray(cam.projection(w / h), np.float64); V = np.asarray(cam.view, np.float64)
    X = np.array([[0, 0, 0], [0, 0, 20], [30, 30, 0]], float)             # the second is 20 mm closer to the camera at the same pixel
    got = EditView.front_point(X, V, pj, w / 2, h / 2, w, h, 6)
    assert got.tolist() == [0, 0, 20]
    assert EditView.front_point(X, V, pj, 5, 5, w, h, 6) is None           # empty space
