"""Applied checkboxes draft one batch without rebuilding on each click."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_applied_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QScrollArea
from app2.qt.tool_panel import AppliedList,EXTRA_QSS


def rows():
    return [dict(id='op-'+str(i),label='Finish for print with a long saved setting name '+str(i),on=True,recipe_key='original',cached=True) for i in range(8)]


def test_pending_batch_preserved_reset_and_external_change():
    app=QApplication.instance() or QApplication([])
    panel=AppliedList();changes=[];applied=[];old=[]
    panel.pending_changed.connect(changes.append);panel.apply_pending.connect(applied.append);panel.toggle.connect(old.append)
    data=rows();panel.set_rows('Meshed',data)
    panel.checks['op-0'].click();assert panel.pending=={'op-0':False} and not old
    assert panel.badges['op-0'].text()==''
    panel.set_rows('Meshed',data);assert panel.pending=={'op-0':False} and not panel.checks['op-0'].isChecked()
    panel.checks['op-0'].click();assert not panel.pending
    panel.checks['op-1'].click();panel.apply_b.click();assert applied==[{'op-1':False}] and not old
    panel.reset_b.click();assert not panel.pending and panel.checks['op-1'].isChecked()
    panel.checks['op-0'].click();data[0]['recipe_key']='new setting';panel.set_rows('Meshed',data)
    assert not panel.pending and changes[-1]=={}
    panel.close()


@pytest.mark.parametrize('width',[330,400])
def test_long_applied_rows_stay_inside_scroll_view(width):
    app=QApplication.instance() or QApplication([])
    window=QWidget();window.resize(width,600);window.setStyleSheet(EXTRA_QSS)
    layout=QVBoxLayout(window);layout.setContentsMargins(0,0,0,0)
    scroll=QScrollArea();scroll.setWidgetResizable(True);layout.addWidget(scroll)
    panel=AppliedList();panel.set_rows('Meshed',rows());scroll.setWidget(panel);window.show();app.processEvents()
    height=panel.footer.height();geometry=panel.footer.geometry()
    for key,check in panel.checks.items():
        assert check.size().width()==20
        assert panel.badges[key].size().width()==46
    for widget,edit,more,menu,editable in panel.rows:
        assert widget.width()<=scroll.viewport().width()
        assert widget.rect().contains(edit.geometry()) and widget.rect().contains(more.geometry())
    assert panel.width()<=scroll.viewport().width()
    panel.checks['op-0'].click();panel.set_pending_plan('All steps cached',True);app.processEvents()
    assert panel.footer.height()==height and panel.footer.geometry()==geometry
    panel.set_tools_enabled(False);assert not panel.apply_b.isEnabled() and not panel.checks['op-0'].isEnabled()
    window.close()
