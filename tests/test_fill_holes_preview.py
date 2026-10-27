"""Fill holes shows what it will close: boundary loops, the size rule, and the outlines queued to the view."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from app2.builder import holes as H


def test_loops_sizes_and_split():
    # a square with a triangular notch left open: two triangles and one 4-edge outer rim + the whole thing is one loop
    V = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]], float); F = np.array([[0, 1, 2], [2, 1, 3]])
    loops = H.boundary_loops(V, F)
    assert len(loops) == 1 and len(loops[0][0]) == 5 and abs(loops[0][1] - np.hypot(10, 10) / 2) < 1e-6
    filled, kept = H.split(loops, 5.0); assert not filled and len(kept) == 1
    filled, kept = H.split(loops, 8.0); assert len(filled) == 1 and not kept
    assert H.segments(loops).shape == (4, 2, 3) and H.segments([]).shape == (0, 2, 3)
    assert abs(H.auto_mm(V) - 0.04 * np.hypot(10, 10)) < 1e-6
    assert H.boundary_loops(V, np.zeros((0, 3), int)) == []


def test_two_holes_in_a_strip():
    # two separate open triangles far apart: two loops
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [50, 0, 0], [60, 0, 0], [50, 10, 0]], float)
    F = np.array([[0, 1, 2], [3, 4, 5]])
    loops = H.boundary_loops(V, F); assert len(loops) == 2
    small, big = sorted(loops, key=lambda l: l[1])
    assert small[1] < 1 and big[1] > 7
    filled, kept = H.split(loops, 3.0); assert len(filled) == 1 and len(kept) == 1


def test_view_queues_lines():
    from PySide6.QtWidgets import QApplication
    from app2.qt.edit_view import EditView
    app = QApplication.instance() or QApplication([])
    v = EditView(); segs = np.zeros((2, 2, 3), np.float32)
    v.set_lines([(segs, (1, 0, 0, 1))]); assert v._todo["lines"] and v._lines[0][1] == (1, 0, 0, 1)
    v.set_lines([]); assert v._todo["lines"] == []
    v._restore_scene(); assert v._todo["lines"] == []
    v.close()


def test_words():
    from app2.qt.build_words import HOLES
    assert "Blue outlines close" in HOLES["count"].format(n=2, m=5, mm="7", auto=" (Auto)")
