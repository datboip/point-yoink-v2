"""Small-screen Build geometry and honest CPU progress presentation."""
import pytest
from types import SimpleNamespace as NS
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtTest import QTest
from app2.qt.theme import install

@pytest.fixture(scope="module")
def app():
    instance=QApplication.instance() or QApplication([]); install(instance); return instance

@pytest.mark.parametrize("size", [(1280,800),(1366,768),(1920,1080)])
def test_yolo_two_card_rows_fit(app,size):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    window=QWidget(); window.setFixedSize(*size); layout=QVBoxLayout(window)
    page=BuildStepPage(SessionBridge(),False); layout.addWidget(page)
    try:
        window.show(); app.processEvents()
        page.open_onetap=lambda:None
        dialog=page.onetap_dlg; dialog.pick("today"); dialog.show(); page._center_onetap(); app.processEvents(); QTest.qWait(30)
        viewport=dialog.simple_page.viewport(); first=list(dialog.profile_cards.values())[:4]
        assert all(viewport.rect().contains(c.mapTo(viewport,c.rect().bottomRight())) for c in first)
        assert dialog.preview_pic.height()==(84 if size[1]<=900 else 180)
    finally:
        app.removeEventFilter(page); page.shutdown(); window.close(); app.processEvents()


def test_progress_names_step_and_is_compact(app):
    from app2.qt.build_view import BuildViewPanel
    window=QWidget();window.setFixedSize(1280,800);layout=QVBoxLayout(window);vp=BuildViewPanel(False);layout.addWidget(vp)
    window.show(); app.processEvents()
    nodes={"side.platter@s":{"state":"start","n":1,"id":"side.platter@s"}}
    vp.card.show_state("Working","Starting","",.12,24,0,nodes=nodes)
    vp._place();app.processEvents()
    assert vp.card.title.text()=="Side 1: Cut base"
    assert vp.card.l1.text()==""                       # the step is the title; not written again under the globe
    assert vp.card.anim.height()<=64 and vp.card.log.height()<=80
    assert vp.card.height()<vp.main.height()*.65
    window.close()


def test_cpu_only_note_and_real_gpu_warning(app,monkeypatch):
    from app2.qt.build_view import ProgressCard
    from app2.builder import backend_notes
    text="GPU was requested, but fusion ran on the CPU. Surface coverage can differ. Review the result."
    monkeypatch.setattr(backend_notes,"cuda_available",lambda:False)
    card=ProgressCard();card._log_nodes({"fused":{"state":"start"}},0);card._log_nodes({"fused":{"state":"warn","warn":[text]}},4)
    assert card._events[0][1]=="note" and "CUDA is unavailable" in card._events[0][3]
    monkeypatch.setattr(backend_notes,"cuda_available",lambda:True)
    card._log_nodes({"fused":{"state":"warn","warn":[text]}},5)
    assert card._events[0][1]=="warn" and "graphics card" in card._events[0][3]
    monkeypatch.setattr(backend_notes,"cuda_available",lambda:False)
    card._log_nodes({"fused":{"state":"warn","warn":[text,"GPU ran out of memory"]}},6)
    assert card._events[0][1]=="warn"
    card.close()


def test_stage_rail_compact_rows_fit(app):
    from app2.qt.stage_list import StageList
    rail=StageList();rail.resize(248,390)
    rail.rows["meshed"].set_info("Same as Fused (As scanned). Pick Watertight for a closed surface.")
    rail.set_current("ms_mesh");rail.set_compact(True);rail.show();app.processEvents()
    visible=[r for r in rail.rows.values() if r.isVisible()]
    assert all(a.geometry().bottom()<b.geometry().top() for a,b in zip(visible,visible[1:]))
    assert rail.rows["ms_mesh"].isVisible()
    assert rail.rows["ms_mesh"].geometry().bottom()<390
    assert rail.sizeHint().height()<390
    rail.close()


def test_raw_content_starts_at_top_without_completed_coverage(app):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    from PySide6.QtCore import QPoint
    page=BuildStepPage(SessionBridge(),False);page.setFixedSize(1280,600)
    try:
        page.stage="raw";page.coverage_state.setText("Coverage measured.");page.coverage_state.show()
        page._show_panel("raw");page.show();app.processEvents()
        assert not page.coverage_state.isVisible()
        assert page.panels.mapTo(page.scroll.widget(),QPoint(0,0)).y()<=8
    finally:
        app.removeEventFilter(page); page.shutdown(); page.close(); app.processEvents()


def test_shutdown_unregisters_global_event_filter(app):
    from PySide6.QtCore import QObject, QEvent
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge

    class ObservedPage(BuildStepPage):
        def __init__(self):
            self.sentinel = QObject()
            self.delivered = 0
            super().__init__(SessionBridge(), False)

        def eventFilter(self, obj, event):
            if obj is self.sentinel and event.type() == QEvent.Type.User:
                self.delivered += 1
            return super().eventFilter(obj, event)

    page = ObservedPage()
    try:
        app.sendEvent(page.sentinel, QEvent(QEvent.Type.User))
        assert page.delivered == 1  # The app-wide filter was registered.
        page.shutdown()
        page.delivered = 0
        app.sendEvent(page.sentinel, QEvent(QEvent.Type.User))
        assert page.delivered == 0
    finally:
        app.removeEventFilter(page)
        page.shutdown()
        page.close()
        app.processEvents()
