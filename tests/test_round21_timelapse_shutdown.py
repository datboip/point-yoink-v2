"""Closing the window waits for its own timelapse child to clean up."""
import os,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
from PySide6.QtWidgets import QApplication
from app2.qt import theme
from test_app2_timelapse import make_project
from test_app2_timelapse_sheet import pump
from test_app2_workflow_fixes import FakeSession,state
from app2.qt.shell import MainWindow


def test_window_close_waits_for_actual_export_child(tmp_path):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    project=make_project(str(tmp_path))
    made=[]
    win=MainWindow(lambda bridge,**over:made.append(FakeSession(bridge,over.get('project'),str(tmp_path))) or made[-1],gl=False,projects_dir=str(tmp_path))
    try:
        win.resize(1280,800);win.show();pump()
        win.bridge('state',state(project,1));assert pump(lambda:getattr(win.state,'project',None)==project,3)
        win.open_timelapse();sheet=win.timelapse_dlg
        sheet.pick('format','gif');sheet.pick('size','1080p');sheet.pick('length',30)
        sheet.run();assert pump(lambda:sheet.bar.value()>0,120)
        output=sheet.out
        win._really=True
        win.close()
        assert sheet.busy and win.isVisible(), 'Window disappeared while its child was still running'
        assert pump(lambda:not sheet.busy and not win.isVisible(),30)
        assert not os.path.exists(output) and not os.path.exists(output+'.part')
    finally:
        if getattr(win,'timelapse_dlg',None) is not None and win.timelapse_dlg.busy:
            win.timelapse_dlg.stop();pump(lambda:not win.timelapse_dlg.busy,30)
        win._really=True;win.build.shutdown(.5);win.close();win.deleteLater();app.processEvents()


def test_no_session_close_shuts_global_build_filter(tmp_path):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    win=MainWindow(lambda bridge,**over:FakeSession(bridge,None,str(tmp_path)),gl=False,projects_dir=str(tmp_path))
    try:
        win.show();pump();win.session=None
        win.close()
        assert win.build._shut, 'Accepted no-session close left Build timers and global filter active'
    finally:
        win._really=True;win.build.shutdown(.5);win.close();win.deleteLater();app.processEvents()


def test_closed_window_stops_global_focus_callbacks(tmp_path):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    win=MainWindow(lambda bridge,**over:FakeSession(bridge,None,str(tmp_path)),gl=False,projects_dir=str(tmp_path))
    calls=[]
    class Shortcut:
        _allow_edit=True
        def setEnabled(self,value):calls.append(value)
    try:
        win.show();pump();win._keys=[Shortcut()];win._really=True
        win.close();calls.clear()
        app.focusChanged.emit(None,None)
        assert calls==[], 'Closed window still changed shortcuts on global focus events'
    finally:
        win.build.shutdown(.5);win.close();win.deleteLater();app.processEvents()


def test_build_shutdown_disconnects_global_focus_callback(tmp_path):
    app=QApplication.instance() or QApplication([]);theme.install(app)
    win=MainWindow(lambda bridge,**over:FakeSession(bridge,None,str(tmp_path)),gl=False,projects_dir=str(tmp_path))
    calls=[]
    class Shortcut:
        def setEnabled(self,value):calls.append(value)
    try:
        win.build._build_shortcuts=[Shortcut()]
        win.build.shutdown(.5)
        app.focusChanged.emit(None,None)
        assert calls==[], "Stopped Build still received global focus events"
    finally:
        win._really=True;win.close();win.deleteLater();app.processEvents()
