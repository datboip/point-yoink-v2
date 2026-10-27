"""Header bounds and resource tile targets at laptop and Deck sizes."""
from types import SimpleNamespace as NS
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PySide6.QtTest import QTest


@pytest.fixture
def app():
    from app2.qt import theme
    app=QApplication.instance() or QApplication([]); theme.install(app)
    return app


@pytest.mark.parametrize('width,height',[(1280,800),(1366,768),(1920,1080)])
def test_header_has_no_overlap_and_shortens_before_clipping(app,width,height):
    from app2.qt.widgets import Header
    window=QWidget(); layout=QVBoxLayout(window); layout.setContentsMargins(0,0,0,0)
    header=Header(); layout.addWidget(header); layout.addStretch()
    for button in (header.motion,header.new,header.open,header.finish):button.hide()
    header.set_project('13 Oct 2026, 11:22','Run1','Run 1 Standard General color with a long readable name')
    header.set_source('Scanner not connected','ink3')
    if hasattr(header,'set_rebuild_text'):header.set_rebuild_text('5 to rebuild · about 3 min')
    else:header.pill_text.setText('5 to rebuild · about 3 min')
    header.pill.show();window.resize(width,height);window.show();QTest.qWait(30)
    try:
        assert window.size().width()==width
        controls=[header.step,header.project,header.folder,header.pill,header.source]
        rectangles=[QRect(control.mapTo(header,QPoint()),control.size()) for control in controls]
        assert all(header.rect().contains(rectangle) for rectangle in rectangles)
        assert not any(a.intersects(b) for i,a in enumerate(rectangles) for b in rectangles[i+1:])
        assert header.pill_text.width() >= header.pill_text.fontMetrics().horizontalAdvance('5 to rebuild')
        scanner_text='Not connected' if width<1440 else 'Scanner not connected'
        assert header.source.width() >= header.source.fontMetrics().horizontalAdvance(scanner_text)+40
        if width<1440:
            assert header.pill_text.text()=='5 to rebuild'
            assert 'Not connected' in header.source.text() and 'Scanner not connected'==header.source.toolTip()
        else:
            assert header.pill_text.text()=='5 to rebuild · about 3 min'
            assert header.pill_text.width()>=header.pill_text.fontMetrics().horizontalAdvance(header.pill_text.text())
    finally:window.close()


def test_leaving_import_never_restores_stray_original_name(app):
    from app2.qt.widgets import Header
    from app2.qt.shell import MainWindow
    header=Header();header.set_project('Run1','Run1','Run 1 Standard General color')
    page=NS(header=header,device_projects=NS(tabs=QWidget()))
    MainWindow._header_for_import(page,False)
    assert header.original.isHidden() and header.original.text()=='(Run1)'
    assert 'Run1' in header.folder.toolTip()
    header.close()


def test_driver_ram_click_hits_resource_tile_not_raw_frames(app,tmp_path,monkeypatch):
    from app2.qt import usage_meter as module
    from app2.diagnostics.driver import Driver
    fake=NS(last={},hist={'cpu':[],'gpu':[],'ram':[]},cpu_name='CPU',gpu_name='GPU')
    monkeypatch.setattr(module,'sampler',lambda:fake)
    window=QWidget();layout=QVBoxLayout(window);meter=module.UsageMeter();layout.addWidget(meter)
    raw=QPushButton('Raw frames');layout.addWidget(raw);window.resize(1280,800);window.show();QTest.qWait(30)
    meter.tile_requested.disconnect();clicks=[];meter.tile_requested.connect(clicks.append)
    driver=Driver(window,str(tmp_path/'header-private.sock'))
    try:
        driver.cmd_click({'text':'RAM','wait':0})
        assert clicks==['RAM']
        target,_=driver._find({'name':'ResourceTile_RAM'})
        assert driver.cmd_tree({'contains':'ResourceTile_RAM'})['count']==1
        rect=QRect(target.mapTo(meter,QPoint()),target.size())
        assert rect==QRect(3*(meter.TILE_W+8),0,meter.TILE_W,meter.H)
        assert meter.rect().contains(rect)
        assert all(driver._find({'name':'ResourceTile_'+tile})[0] is meter.tile_buttons[tile] for tile in ('SCANNER','CPU','GPU','RAM'))
    finally:driver.server.close();window.close();meter._timer.stop()


@pytest.mark.parametrize('tile',['CPU','GPU','RAM','SCANNER'])
def test_resource_details_fit_small_parent_with_close_always_visible(app,monkeypatch,tile):
    from app2.qt import resource_panel as module
    monkeypatch.setattr(module.runner,'scopes_available',lambda:False)
    monkeypatch.setattr(module.resources,'recommended_limits',lambda:dict(memory_gb=8,cpu_threads=4,gpu_jobs=1))
    window=QWidget();window.resize(1280,800);window.show();QTest.qWait(20)
    panel=module.ResourcePanel(tile,window)
    panel.note.setText('Running jobs keep their current limits. '*15)
    panel.show();QTest.qWait(40)
    try:
        available=QRect(window.mapToGlobal(QPoint()),window.size())
        assert available.contains(panel.frameGeometry())
        close=QRect(panel.close_button.mapTo(panel,QPoint()),panel.close_button.size())
        assert panel.rect().contains(close)
        assert panel.body_scroll.widgetResizable()
        assert panel.body_scroll.horizontalScrollBar().maximum()==0
    finally:panel.close();window.close()
