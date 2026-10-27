"""The orientation cube (app2/qt/navcube.py, viewnav math): a click on each face turns the camera to that face's
preset about the same pivot (the pivot stays where it was on screen), a press that drags turns the view instead of
snapping, and the cube turns with the camera. Offscreen, no GL: a plain widget with a gl_core.Camera stands in for the
view."""
import os, sys, unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np                                                    # noqa: E402


def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakeView:
    pass


class NavCubeTest(unittest.TestCase):
    def setUp(self):
        app()
        from PySide6.QtWidgets import QWidget
        from PySide6.QtCore import Signal
        from app2 import gl_core
        from app2.qt.navcube import NavCube

        class View(QWidget):
            camera_moved = Signal()
        self.view = View(); self.view.resize(700, 500); self.view.cam = gl_core.Camera()
        self.view.cam.frame(np.zeros(3), np.array([100.0, 80.0, 60.0]))
        self.moved = []; self.view.camera_moved.connect(lambda: self.moved.append(1))
        self.cube = NavCube(self.view); self.view.show(); app().processEvents()

    def tearDown(self):
        self.view.close(); self.view.deleteLater(); app().processEvents()

    def click(self, x, y, dx=0, dy=0):
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtTest import QTest
        p = QPoint(int(round(x)), int(round(y)))
        QTest.mousePress(self.cube, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
        if dx or dy: QTest.mouseMove(self.cube, p + QPoint(dx, dy))
        QTest.mouseRelease(self.cube, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p + QPoint(dx, dy))
        app().processEvents()

    def face_centre(self, key):
        """The middle of face `key` as the cube draws it now (widget coordinates), or None when it faces away."""
        from app2.qt import navcube as N
        import viewnav as VN
        r = next((r for r in VN.regions(self.cube.rot(), N.CANVAS, N.CANVAS)[1] if r["key"] == key), None)
        if r is None: return None
        c = np.mean(np.asarray(r["poly"], float), axis=0); return c[0] - N.OFFX, c[1] - N.OFFY

    def test_sits_in_the_bottom_left_corner_at_a_fixed_size(self):
        from app2.qt.navcube import SIZE
        self.assertEqual((self.cube.width(), self.cube.height()), (SIZE, SIZE))
        self.assertEqual((self.cube.x(), self.cube.y() + SIZE), (0, self.view.height()))
        self.view.resize(900, 640); app().processEvents()
        self.assertEqual(self.cube.y() + SIZE, 640, "the cube did not follow the view's bottom edge")

    def test_each_face_click_sets_its_preset(self):
        from app2.qt.navcube import preset, look_at_face
        import viewnav as VN
        cam = self.view.cam
        for key in [f[0] for f in VN.FACES]:
            look_at_face(cam, key); cam.orbit(25, -20)                # a slanted view where the face still shows
            pe0 = cam.view[:3, :3] @ cam.pivot + cam.view[:3, 3]
            c = self.face_centre(key); self.assertIsNotNone(c, key)
            self.assertEqual(self.cube.face_at(*c), key, "hit-test at the face's middle: %s" % key)
            n = len(self.moved); self.click(*c)
            self.assertTrue(np.allclose(cam.view[:3, :3], preset(key), atol=1e-9), "%s: the camera is not at its preset" % key)
            pe1 = cam.view[:3, :3] @ cam.pivot + cam.view[:3, 3]
            self.assertTrue(np.allclose(pe0, pe1), "%s: the pivot moved on screen" % key)
            self.assertEqual(len(self.moved), n + 1, "%s: camera_moved not sent (Compare's other view follows it)" % key)

    def test_front_is_what_the_front_preset_looks_at(self):
        """The face under the cube's middle after a click is the one clicked (the cube turns with the camera)."""
        from app2.qt.navcube import look_at_face
        look_at_face(self.view.cam, "top"); self.cube.update()
        regs = self.cube._regions(); front = max(regs, key=lambda r: r["normal_z"])
        self.assertEqual(front["key"], "top")

    def test_a_drag_turns_instead_of_snapping(self):
        from app2.qt.navcube import preset, look_at_face
        cam = self.view.cam; look_at_face(cam, "front"); cam.orbit(30, 10)
        R0 = cam.view[:3, :3].copy(); c = self.face_centre("front")
        self.click(*c, dx=30, dy=0)
        self.assertFalse(np.allclose(cam.view[:3, :3], preset("front")), "a drag snapped to the face")
        self.assertFalse(np.allclose(cam.view[:3, :3], R0), "a drag did not turn the view")


if __name__ == "__main__":
    unittest.main()
