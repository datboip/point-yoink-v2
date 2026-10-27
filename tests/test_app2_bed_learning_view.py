"""Learn bed shows its progress big over the model panel: the dots placed so far and an arc (2026-10-25)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scanner'))


def test_learning_map_lists_the_clusters_seen_twice():
    from app2.scan.bed_adapter import ScannerBed
    a = ScannerBed.__new__(ScannerBed); a.learner = None; a.bed = None; a.pending = None; a.plane = None
    assert a.learning_map() is None
    a.learner = NS(_clusters=lambda n: (np.array([[10., 0.], [0., 30.]]), np.array([3, 2])))
    m = a.learning_map(); assert m.dots == ((10.0, 0.0), (0.0, 30.0)) and m.radius == 40.0 and m.provisional


def test_bed_view_shows_the_map_while_learning(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app2.qt.bed_card import BedView
    from app2.scan.bedhook import BedInfo
    QApplication.instance() or QApplication([])
    v = BedView(); info = BedInfo(None, ((10.0, 0.0),), 40.0, None, '', True, None, None, 'dots', None)
    v.set_bed(info, 'Learning: 120 of 360°, 1 dots', 120); v.resize(400, 300); v.show(); v.repaint()
    assert v.deg == 120 and v.info is info
    v.set_bed(info, 'x'); assert v.deg is None
