"""Help cards stay inside the app and do not move scan or recipe controls."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_help_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt.widgets import Controls
from app2.qt.onetap import OneTapDialog
from app2.qt.help_text import HelpButton


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_scan_help_inside_window_without_moving_controls(size):
    app=QApplication.instance() or QApplication([])
    window=QWidget(); window.resize(*size)
    controls=Controls(window); controls.setGeometry(12,300,360,260); controls.fit_width(360)
    window.show(); app.processEvents()
    watched=[controls.range,controls.gain,controls.capture_mode,controls.led]
    before=[w.geometry() for w in watched]
    for help_button in controls.findChildren(HelpButton):
        help_button.click(); app.processEvents(); card=window._help_card
        assert not card.isWindow()
        assert window.rect().adjusted(8,8,-8,-8).contains(card.geometry())
        assert [w.geometry() for w in watched]==before
        assert card.text.text()
    window.close()


def test_preset_help_preserves_fixed_sheet_and_single_card():
    app=QApplication.instance() or QApplication([])
    window=QWidget(); window.resize(1280,800)
    sheet=OneTapDialog(window); sheet.move(250,60); sheet.show(); window.show(); app.processEvents()
    before=sheet.geometry(); controls=sheet.controls.geometry()
    for help_button in [sheet.preset_help,sheet.custom_help]:
        help_button.click(); app.processEvents()
        assert sheet.geometry()==before and sheet.controls.geometry()==controls
        assert window.rect().adjusted(8,8,-8,-8).contains(window._help_card.geometry())
        card=window._help_card
        text_bounds=card.text.fontMetrics().boundingRect(QRect(0,0,card.text.width(),1000),Qt.TextFlag.TextWordWrap,card.text.text())
        assert text_bounds.height() <= card.text.height()
    window.close()
