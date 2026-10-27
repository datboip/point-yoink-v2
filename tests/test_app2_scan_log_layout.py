"""Scan keeps a readable log, or exposes the full log on short windows."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-log-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication
from app2.qt import theme
from app2.qt.shell import MainWindow

@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_scan_log_readable_and_stable(size,tmp_path):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    def offline(*args,**kwargs): raise RuntimeError('No scanner in log test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path));win.resize(*size);win.show();win.show_step(0);app.processEvents()
    page=win.scan
    QTest.qWait(150);app.processEvents()
    try:
        before=page.log.geometry()
        for i in range(40):page.log.append('Building side %d: checking tracked frames and saved model.'%i)
        app.processEvents()
        assert page.log.geometry()==before
        assert win.size().width()==size[0] and win.size().height()==size[1]
        assert page.go.geometry().bottom()<page.left.height()
        assert not page.log.isVisible() and page.status.log_btn.isVisible()
        page.toggle_log();app.processEvents()
        assert page.popover.isVisible() and not page.popover.isWindow()
        assert page.rect().contains(page.popover.geometry())
        title=page.popover.drag_title; old=page.popover.pos()
        QTest.mousePress(title,Qt.MouseButton.LeftButton,pos=QPoint(20,10))
        QTest.mouseMove(title,QPoint(80,40))
        QTest.mouseRelease(title,Qt.MouseButton.LeftButton,pos=QPoint(80,40));app.processEvents()
        assert page.popover.pos()!=old
        grip=page.popover.resize_grip; old_size=page.popover.size()
        QTest.mousePress(grip,Qt.MouseButton.LeftButton,pos=QPoint(8,8))
        QTest.mouseMove(grip,QPoint(38,28))
        QTest.mouseRelease(grip,Qt.MouseButton.LeftButton,pos=QPoint(38,28));app.processEvents()
        assert page.popover.size()!=old_size
        placed=page.popover.geometry();page.toggle_log();page.toggle_log();app.processEvents()
        assert page.popover.geometry()==placed
    finally:
        win._really=True;win.build.shutdown();win.close();win.deleteLater();app.processEvents()


def test_build_log_can_move_resize_and_reopen():
    from PySide6.QtWidgets import QWidget
    from app2.qt.history import BuildLogPopover
    app=QApplication.instance() or QApplication([]);theme.install(app)
    parent=QWidget();parent.resize(1000,700);parent.show()
    log=BuildLogPopover(parent);log.place(parent.mapToGlobal(QPoint(500,600)),600);app.processEvents()
    try:
        title=log.title;old=log.pos()
        QTest.mousePress(title,Qt.MouseButton.LeftButton,pos=QPoint(20,10))
        QTest.mouseMove(title,QPoint(80,40))
        QTest.mouseRelease(title,Qt.MouseButton.LeftButton,pos=QPoint(80,40));app.processEvents()
        assert log.pos()!=old
        grip=log.resize_grip;old_size=log.size()
        QTest.mousePress(grip,Qt.MouseButton.LeftButton,pos=QPoint(8,8))
        QTest.mouseMove(grip,QPoint(38,28))
        QTest.mouseRelease(grip,Qt.MouseButton.LeftButton,pos=QPoint(38,28));app.processEvents()
        assert log.size()!=old_size
        placed=log.geometry();log.close_popover();log.place(parent.mapToGlobal(QPoint(500,600)),600);app.processEvents()
        assert log.geometry()==placed
    finally:
        log.close();parent.close()


def test_long_build_details_do_not_replace_scan_action():
    from types import SimpleNamespace as NS
    from app2.qt.scan_page import ScanPage
    labels=[]
    page=NS(sides=NS(buildbox=NS(prog=NS(step='Side fuse, side 5 frame 919 of 999999'))),
            state=NS(building=True),ended=False,
            go=NS(set_action=lambda *args:labels.append(args)))
    ScanPage._build_step(page)
    assert labels==[('Building model...', 'primary', False)]
