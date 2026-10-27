import os, threading, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["POINTYOINK_NO_DEVICE"]="1"
from test_app2_workflow_fixes import state
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from app2.qt.stage_bridge import StageBridge
from app2.qt.bridge import SessionBridge
from app2.qt.build_step import BuildStepPage
from app2.builder.stagehost import Snapshot


def pump():
    QApplication.processEvents()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_retired_bridge_drops_queued_and_late_worker_payloads():
    app=QApplication.instance() or QApplication([])
    b=StageBridge();seen=[];b.stages_result.connect(seen.append)
    b("stages_result",{"queued":True})
    b.close()
    t=threading.Thread(target=lambda:b("stages_result",{"late":True}))
    t.start();t.join();pump()
    assert seen==[]
    assert not b._fifo and not b._box
    b.deleteLater();pump()


def test_dropped_host_bridge_survives_worker_then_deletes_on_gui(tmp_path):
    app=QApplication.instance() or QApplication([])
    page=BuildStepPage(SessionBridge(),False)
    started=threading.Event();release=threading.Event()
    class Engine:
        def snapshot(self):
            started.set();release.wait(3)
            return Snapshot()
    page.engine_factory=lambda root:Engine()
    page.state=state(str(tmp_path))
    page._ensure_host(force=True)
    host=page.host;bridge=page.sb;destroyed=[]
    bridge.destroyed.connect(lambda:destroyed.append(threading.get_ident()))
    try:
        assert started.wait(3)
        assert bridge.parent() is page, "Worker bridge has no GUI owner"
        page._drop_host()
        assert page.host is None and page.sb is None
        bridge("stages_result",{"late":True})
        release.set();host.thread.join(3)
        assert not host.thread.is_alive()
        pump()
        assert destroyed==[threading.get_ident()]
    finally:
        release.set();host.close(.5);page.shutdown(.5);page.deleteLater();pump()
