"""Simple and Advanced share one recipe and fixed sheet bounds."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-modes-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication,QWidget
from app2.qt import theme
from app2.qt.onetap import OneTapDialog

@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_simple_advanced_keep_recipe_and_geometry(size):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    parent=QWidget();parent.resize(*size);sheet=OneTapDialog(parent);sheet.pick('auto')
    sheet.move((size[0]-sheet.width())//2,(size[1]-sheet.height())//2)
    parent.show();sheet.show();app.processEvents()
    try:
        assert sheet.pages.currentIndex()==0 and not sheet.controls.isVisible()
        geometry=sheet.geometry();footer=sheet.ok_b.geometry();original=sheet.payload();changed=[]
        sheet.custom_changed.connect(changed.append)
        for advanced in (True,False,True):
            sheet.set_mode(advanced);app.processEvents()
            assert sheet.width()==760
            assert sheet.height()==720
            assert sheet.profile_cards["auto"].isChecked()
            assert sheet.payload()==original and not changed
            assert parent.rect().contains(sheet.geometry())
        sheet.pick('custom',True);sheet.detail.setValue(.7)
        edited=sheet.payload();sheet.set_mode(False)
        assert sheet.payload()==edited
        result=[];sheet.custom_ok.connect(result.append);sheet.run();assert result==[edited]
        sheet.show_error('Could not read this project.')
        assert sheet.foot.text()=='Could not read this project.' and not sheet.ok_b.isEnabled()
    finally:parent.close()
