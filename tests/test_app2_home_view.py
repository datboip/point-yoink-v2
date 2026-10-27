"""The user's own Home view: an orientation saved with the project and used by every framing."""
import json, os, tempfile, types
import numpy as np
from app2.gl_core import Camera
from app2.qt import build_step as B


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_frame_uses_the_saved_orientation_and_still_fits_the_result():
    cam = Camera(); lo, hi = np.array([-10., -10, -10]), np.array([10., 10, 10])
    cam.frame(lo, hi); d0 = -cam.view[2, 3]
    cam.home_rotation = rot_y(40); cam.frame(lo, hi)
    assert np.allclose(cam.view[:3, :3], rot_y(40)) and np.isclose(-(cam.view[2, :3] @ cam.pivot + cam.view[2, 3]), d0)   # same distance, turned
    big = Camera(); big.home_rotation = rot_y(40); big.frame(lo * 5, hi * 5)
    assert -(big.view[2, :3] @ big.pivot + big.view[2, 3]) > d0 * 4.5                                                     # a bigger result: farther


def test_default_framing_is_unchanged_without_a_saved_home():
    cam = Camera(); cam.frame(np.array([-5., -5, -5]), np.array([5., 5, 5]))
    assert np.allclose(cam.view[:3, :3], np.eye(3))


class Probe:
    _home_path = B.BuildStepPage._home_path; _home_cams = B.BuildStepPage._home_cams; _apply_home_view = B.BuildStepPage._apply_home_view
    set_home_view = B.BuildStepPage.set_home_view; reset_home_view = B.BuildStepPage.reset_home_view
    def __init__(self):
        self.host_root = tempfile.mkdtemp(); self.gl = True; self.notes = []; self.went = 0
        mk = lambda: types.SimpleNamespace(view=types.SimpleNamespace(cam=Camera()))
        self.vp = types.SimpleNamespace(main=mk(), before=mk(), pa=mk(), pb=mk()); self.vp.view = self.vp.main.view
    def notice(self, t, **k): self.notes.append(t)
    def home(self): self.went += 1


def test_set_saves_per_project_and_reaches_every_pane_then_reset_clears_it():
    p = Probe(); p.vp.main.view.cam.view = np.eye(4); p.vp.main.view.cam.view[:3, :3] = rot_y(30)
    p.set_home_view()
    saved = np.array(json.load(open(p._home_path()))["R"]); assert np.allclose(saved, rot_y(30))
    for pane in (p.vp.main, p.vp.before, p.vp.pa, p.vp.pb): assert np.allclose(pane.view.cam.home_rotation, rot_y(30))
    p.reset_home_view()
    assert not os.path.exists(p._home_path()) and p.went == 1
    for pane in (p.vp.main, p.vp.before, p.vp.pa, p.vp.pb): assert pane.view.cam.home_rotation is None
