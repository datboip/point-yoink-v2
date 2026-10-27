"""Measured bar alignment and no camera scrolling in the real shell."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication,QLabel
from PySide6.QtTest import QTest


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('size',[(1280,800),(1688,1044),(1920,1080)])
def test_bar_alignment_compact_placement_and_no_camera_scroll(app,tmp_path,size):
    from app2.qt.shell import MainWindow
    from test_round25_ui import scene_state
    w=MainWindow(lambda *a,**kw:None,gl=False,projects_dir=str(tmp_path),defer_new_project=True)
    w.resize(*size);w.show();w.show_step(0);w.startup.hide();p=w.scan
    try:
        p.on_state(scene_state(tmp_path,has_led=True))
        for tab in (0,1):
            p.sides.tabs.setCurrentIndex(tab);p._size=None
            for _ in range(10):app.processEvents()
            QTest.qWait(50)
            assert (w.width(),w.height())==size
            assert p.camera_scroll.verticalScrollBar().maximum()==0
            rows=(p.controls.gain_row,p.controls.range_row)
            assert rows[0].name.mapTo(p,QPoint()).x()==rows[1].name.mapTo(p,QPoint()).x()
            assert rows[0].auto.mapTo(p,QPoint()).x()==rows[1].auto.mapTo(p,QPoint()).x()
            for row in rows:
                gap=row.auto.mapTo(p,QPoint()).x()-row.value.mapTo(p,QPoint()).x()-row.value.width()
                assert 0<=gap<=p.fontMetrics().lineSpacing()
                assert row.mapTo(p,QPoint()).y()>=p.ir.mapTo(p,QPoint()).y()+p.ir.height()
            buttons=(p.controls.auto_gain,p.controls.auto_range,*p.model.distance_mode.buttons)
            assert len({b.height() for b in buttons})==1
            compact=p.height()<680
            assert p.live_strip.compact==compact
            assert p.sides.isAncestorOf(p.controls.led)==compact
            assert p.model.external_distance_mode!=compact
            if not compact:
                label=next(x for x in p.live_strip.color_group.findChildren(QLabel) if x.text()=='3D view')
                assert label.mapTo(p,QPoint()).x()==rows[0].name.mapTo(p,QPoint()).x()
            old=(p.left.geometry(),p.model.geometry(),p.sides.geometry())
            p.controls.auto_gain.click();app.processEvents()
            assert old==(p.left.geometry(),p.model.geometry(),p.sides.geometry())
    finally:w._really=True;w.build.shutdown();w.close();app.processEvents()
