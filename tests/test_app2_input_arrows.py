"""Styled number fields keep visible arrow glyphs at compact fixed sizes."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_arrow_test_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication,QSpinBox,QDoubleSpinBox,QComboBox
from app2.qt.theme import install


@pytest.mark.parametrize('enabled',[True,False])
def test_compact_controls_render_visible_arrow_glyphs(enabled):
    app=QApplication.instance() or QApplication([]);install(app)
    for kind,width,height in [(QSpinBox,70,30),(QDoubleSpinBox,122,32),(QComboBox,120,32)]:
        control=kind();control.setFixedSize(width,height);control.setEnabled(enabled)
        if isinstance(control,QComboBox):control.addItem('Replace')
        else:control.setRange(0,100);control.setValue(24)
        control.show();app.processEvents();image=control.grab().toImage()
        assert image.width()==width and image.height()==height
        # Sample the arrow interior, excluding subcontrol edges and text.
        bright=sum(image.pixelColor(x,y).red()>110 for x in range(width-16,width-4) for y in range(4,height-4))
        assert bright>=8
        control.close()
