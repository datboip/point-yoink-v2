"""Timelapse notices keep project and app paths private."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QWidget
from app2.qt.timelapse_sheet import TimelapseSheet, REPO
from app2.qt import theme


def test_failed_notice_hides_project_and_app_paths(tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    parent = QWidget(); parent.resize(1280, 800)
    sheet = TimelapseSheet(parent)
    sheet.project = '/private/library/ScanExample'
    sheet._why = 'Cannot read '+sheet.project+'/data/frame.dph from '+REPO
    class FailedProcess:
        def deleteLater(self): pass
    sheet.proc = FailedProcess()
    sheet._finished(-1)
    assert sheet.project not in sheet.status.text()
    assert REPO not in sheet.status.text()
    assert 'frame.dph' in sheet.status.text()
    parent.close()


def test_worker_uses_egl_independently_of_window_platform(tmp_path,monkeypatch):
    from test_app2_timelapse import make_project
    from test_app2_timelapse_sheet import pump
    monkeypatch.setenv('PYOPENGL_PLATFORM','glx')
    app=QApplication.instance() or QApplication([]);theme.install(app)
    parent=QWidget();parent.resize(1280,800)
    sheet=TimelapseSheet(parent);sheet.present(make_project(str(tmp_path)),1)
    try:
        sheet.run()
        assert sheet.proc.processEnvironment().value('PYOPENGL_PLATFORM')=='egl'
    finally:
        if sheet.busy:
            sheet.stop();assert pump(lambda:not sheet.busy,30)
        parent.close()
