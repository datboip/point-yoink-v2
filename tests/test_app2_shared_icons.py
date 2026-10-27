"""Reuse the local icon pack and keep idle scrollbar gutters stable."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QScrollArea,QWidget
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from app2.qt.icons import icon
from app2.qt.widgets import quiet_scrollbar


def test_shared_edit_icons_exist_for_normal_and_disabled():
    app=QApplication.instance() or QApplication([])
    for name in ('lasso','box','brush','magic','undo','redo','delete','history','save-version','select-through','visible-only'):
        asset=icon(name)
        assert not asset.isNull(),name
        assert not asset.pixmap(16,16,QIcon.Mode.Disabled).isNull(),name


def test_idle_scrollbar_keeps_gutter_without_visible_handle():
    app=QApplication.instance() or QApplication([])
    area=QScrollArea();area.setWidgetResizable(True);area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    body=QWidget();area.setWidget(body);quiet_scrollbar(area.verticalScrollBar())
    area.resize(300,200);area.show();app.processEvents()
    width=area.viewport().width()
    assert area.verticalScrollBar().property('emptyRange')
    body.setMinimumHeight(600);app.processEvents()
    assert not area.verticalScrollBar().property('emptyRange')
    assert area.viewport().width()==width
    area.close()
