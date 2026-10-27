"""Beginner help describes current controls without changing fixed help geometry."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-help-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt.help_text import HELP, HelpButton, help_for


@pytest.mark.parametrize('title',['Loose piece size','Connected piece','YOLO Yoink','Perspective','Orthographic'])
def test_plain_help_fits_reserved_card(title):
    app=QApplication.instance() or QApplication([])
    parent=QWidget(); parent.resize(1280,800); button=HelpButton(title,parent=parent)
    button.open_help(); card=parent._help_card
    assert card.size().width()==340 and card.size().height()==154
    bounds=QFontMetrics(card.text.font()).boundingRect(QRect(0,0,316,1000),Qt.TextFlag.TextWordWrap,card.text.text())
    assert bounds.height()<=card.text.height()
    assert '\u2014' not in card.text.text()
    parent.close()


def test_isolation_help_warns_about_inner_surfaces():
    text=help_for({'label':'Loose piece size'})
    assert 'connected pieces' in text and 'inner cavity' in text and 'before Delete' in text
