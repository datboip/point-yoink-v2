"""Visible display buttons survive model refreshes and select the right mode."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-style-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from app2.qt.build_view import BuildViewPanel


def test_display_button_click_survives_state_refresh():
    app=QApplication.instance() or QApplication([])
    panel=BuildViewPanel(False);panel.resize(800,600)
    modes=[('shaded','Shaded'),('flat','Flat facets')]
    panel.set_colours(modes,'shaded');panel.show();app.processEvents()
    buttons=list(panel.colour_btn.look_buttons());chosen=[]
    panel.colour_picked.connect(chosen.append)
    panel.set_colours(modes,'shaded')
    assert panel.colour_btn.look_buttons()==buttons
    QTest.mouseClick(buttons[1],Qt.LeftButton)
    assert chosen==['flat'] and not panel.colour_menu.isVisible()
    panel.set_colours(modes,'flat')
    assert buttons[1].isChecked() and panel.colour_btn.text().startswith('Flat facets')
    panel.close();panel.deleteLater();app.processEvents()
