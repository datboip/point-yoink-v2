import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_compact_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication,QPushButton
from app2.qt.edit_page import EditPage
from app2.qt.theme import install

@pytest.mark.parametrize('width',[200,248])
def test_palette_compact_text_and_icons_fit(width):
    app=QApplication.instance() or QApplication([]);install(app)
    page=EditPage();page.palette.resize(width,520);page.palette.show();app.processEvents()
    assert not page.palette.horizontalScrollBar().isVisible()
    assert page.palette.widget().width()<=page.palette.viewport().width()
    for button in page.palette.widget().findChildren(QPushButton):
        if not button.isVisible() or button.text()=='?':continue
        needed=button.fontMetrics().horizontalAdvance(button.text())+6
        if not button.icon().isNull():needed+=button.iconSize().width()+4
        assert needed<=button.width(),(button.text(),needed,button.width())
    page.palette.close();page.close()
