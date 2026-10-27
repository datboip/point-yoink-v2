"""Raw frames: each side is a card, title on top and the details wrapped under it; ticks survive a refresh."""
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel
from app2.qt import build_words as W
from app2.qt.stage_panels import TabBody

def test_cards_and_ticks():
    app = QApplication.instance() or QApplication([])
    b = TabBody("raw", W.PANELS["raw"]["tabs"][0]); b.resize(330, 700); b.show()
    rows = [(1, "Side 1    Sep 28 22:21    2727 frames    0 lost", False), (2, "Side 2    Sep 29 0:57    1302 frames    0 lost    platter saved", True)]
    b.set_sides(rows); app.processEvents()
    assert [c.text() for c in b.side_boxes.values()] == ["Side 1", "Side 2"]
    texts = [l.text() for l in b.findChildren(QLabel)]
    assert any("Sep 28 22:21" in t and "2727 frames" in t for t in texts) and any("platter saved" in t for t in texts)
    b.side_boxes[2].setChecked(True); b.set_sides(rows); app.processEvents()
    assert b.side_boxes[2].isChecked() and not b.side_boxes[1].isChecked()
    assert b.delete_b.text() == "Delete checked sides (1)"
    for c in b.findChildren(QCheckBox): assert c.width() <= 330
