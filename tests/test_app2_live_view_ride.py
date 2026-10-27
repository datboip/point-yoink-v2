"""Live view turned away from Home keeps riding with the scanner from your angle (2026-10-24), so a turntable part keeps
spinning on screen; ride=False keeps the old world-fixed view. Home view still snaps back to the scanner's view."""
import numpy as np
from PySide6.QtWidgets import QApplication


def _T(yaw_deg, t):
    a = np.radians(yaw_deg); T = np.eye(4); T[0, 0] = T[2, 2] = np.cos(a); T[0, 2] = np.sin(a); T[2, 0] = -np.sin(a); T[:3, 3] = t; return T


def test_turned_view_rides_with_the_scanner():
    from app2.qt.view3d import ModelView
    QApplication.instance() or QApplication([])
    v = ModelView(); T0 = _T(0, (0, 0, -300)); piv = np.zeros(3)
    v.set_home(T0, piv); v._set_following(False); v.cam.orbit(40.0, 0.0)
    mine = v.cam.view.copy(); T1 = _T(30, (150, 0, -260))
    v.follow_scanner(T1, piv)
    expect = mine @ T0 @ np.linalg.inv(T1)
    assert np.allclose(v.cam.view, expect, atol=1e-9) and not v.following
    v.ride = False; v.follow_scanner(_T(60, (260, 0, -150)), piv)
    assert np.allclose(v.cam.view, expect, atol=1e-9)              # world-fixed: nothing moved
    v.go_home(); assert v.following and np.allclose(v.cam.view, v.cam.CV2GL @ np.linalg.inv(_T(60, (260, 0, -150))), atol=1e-9)
