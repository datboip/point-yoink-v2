# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO Yoink's Compare tab: the user's own part per recipe, side by side, with Use this recipe."""
import os, tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp()) / 'config.json')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QLabel, QPushButton


def test_compare_tab_shows_pictures_and_picks(tmp_path):
    app = QApplication.instance() or QApplication([])
    from PySide6.QtGui import QImage
    from app2.qt.onetap import OneTapDialog
    from app2.recipe_pictures import DIR
    d = tmp_path / DIR; d.mkdir()
    for name in ('today', 'quick'): img = QImage(60, 40, QImage.Format_RGB888); img.fill(0x777777); img.save(str(d / (name + '.png')))
    dlg = OneTapDialog(); picked = []; dlg.pick = lambda k, ask=False: picked.append(k)
    dlg.show_pictures(str(tmp_path)); dlg.set_mode('compare')
    assert dlg.pages.currentIndex() == 2 and dlg.compare_b.isChecked() and not dlg.preview.isVisible()
    uses = [w for w in dlg.compare_page.widget().findChildren(QPushButton) if w.text() == 'Use this recipe']
    assert len(uses) == 2
    uses[0].click(); assert picked and dlg.pages.currentIndex() == 0


def test_compare_tab_empty_says_what_to_do(tmp_path):
    app = QApplication.instance() or QApplication([])
    from app2.qt.onetap import OneTapDialog
    dlg = OneTapDialog(); dlg.show_pictures(str(tmp_path)); dlg.set_mode('compare')
    texts = [w.text() for w in dlg.compare_page.widget().findChildren(QLabel)]
    assert any('Nothing to compare yet' in t for t in texts)
