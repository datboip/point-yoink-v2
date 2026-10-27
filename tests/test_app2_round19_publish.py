"""Foreground publication and immutable follow-on measurements."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.fixture
def built():
    from test_stages_plus import _PlusRun
    case = _PlusRun(); case.setUp()
    try:
        case.e.clear_ops("fused.plus")
        yield case
    finally: case.tearDown()


def test_mesh_published_before_report_and_follow_on_has_no_build_lock(built):
    engine = built.e
    result = engine.run(defer_report=True)
    assert result.ok and result.model
    assert "report" not in result.ran + result.cached
    root = Path(built.tmp); publication = json.loads((root / "stages/published.json").read_text())
    assert publication["coverage_pending"] and (root / "model.ply").exists()
    assert not (root / "stages/lock.json").exists()
    def check_lock(st, key, outs, *args, **kwargs):
        assert not (root / "stages/lock.json").exists()
        return execute(st, key, outs, *args, **kwargs)
    execute = engine._execute; engine._execute = check_lock
    measurement = engine.measure_published(publication["generation"])
    assert not measurement["discarded"]
    assert not json.loads((root / "stages/published.json").read_text())["coverage_pending"]


def test_new_publication_discards_old_measurements_even_for_identical_mesh(built):
    engine = built.e; root = Path(built.tmp)
    engine.run(defer_report=True)
    old = json.loads((root / "stages/published.json").read_text())
    engine.run(defer_report=True)
    new = json.loads((root / "stages/published.json").read_text())
    assert old["digest"] == new["digest"] and old["generation"] != new["generation"]
    assert engine.measure_published(old["generation"])["discarded"]
    assert json.loads((root / "stages/published.json").read_text()) == new


def test_forced_report_runs_again_after_mesh_publication(built):
    engine = built.e; root = Path(built.tmp)
    engine.run()
    calls = []; execute = engine._execute
    def count(st, *args, **kwargs):
        calls.append(st.id); return execute(st, *args, **kwargs)
    engine._execute = count
    engine.run(defer_report=True, force=("report",))
    publication = json.loads((root / "stages/published.json").read_text())
    assert publication["force_report"]
    assert not engine.measure_published(publication["generation"])["discarded"]
    assert calls == ["report"]


def test_transient_report_is_retried_and_failure_is_not_success(built, monkeypatch):
    engine = built.e; root = Path(built.tmp)
    method = type(engine.graph().by_id["report"].method); original = method.execute
    failed = [True]
    def report(self, context):
        result = original(self, context)
        if failed[0]: result.update(transient=True, fail=dict(kind="launch", sig="launch", rc=1))
        return result
    monkeypatch.setattr(method, "execute", report)
    engine.run(defer_report=True)
    publication = json.loads((root / "stages/published.json").read_text())
    assert engine.measure_published(publication["generation"])["error"]
    failed[0] = False
    engine.run(defer_report=True)
    publication = json.loads((root / "stages/published.json").read_text())
    assert not engine.measure_published(publication["generation"])["error"]


def test_same_process_store_temporaries_and_state_updates_are_independent(built):
    from concurrent.futures import ThreadPoolExecutor
    from stages.store import Store
    stores = [Store(built.p, built.e.memo) for _ in range(4)]
    folders = [store.tmp_dir("report", "same-key") for store in stores]
    assert len(set(folders)) == 4 and all(Path(folder).is_dir() for folder in folders)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: pair[1].note_state("node-%d" % pair[0], key="key", status="done"), enumerate(stores)))
    assert all("node-%d" % index in stores[0].state()["nodes"] for index in range(4))


def test_publication_changes_during_report_are_discarded(built):
    engine = built.e; root = Path(built.tmp)
    engine.run(defer_report=True)
    publication = json.loads((root / "stages/published.json").read_text())
    execute = engine._execute
    def replace_during_report(st, key, outs, *args, **kwargs):
        result = execute(st, key, outs, *args, **kwargs)
        updated = dict(publication, generation="newer-publication")
        (root / "stages/published.json").write_text(json.dumps(updated))
        return result
    engine._execute = replace_during_report
    assert engine.measure_published(publication["generation"])["discarded"]
    assert not (root / "model_needs.json").exists()


def test_quality_cache_includes_lineup_and_full_surface_bytes(tmp_path):
    from app2.builder.quality_cache import identity
    mesh = tmp_path / "model.ply"; sides = tmp_path / "sides.json"
    mesh.write_bytes(b"same-length-A"); sides.write_bytes(b"placement-A")
    old = identity(mesh, sides)
    sides.write_bytes(b"placement-B"); assert identity(mesh, sides) != old
    old = identity(mesh, sides)
    mesh.write_bytes(b"same-length-B"); assert identity(mesh, sides) != old


def test_cancel_coverage_returns_without_waiting_for_worker_cleanup():
    import threading, time
    from app2.qt.build_step import BuildStepPage
    entered = threading.Event(); release = threading.Event()
    def blocking_cancel():
        entered.set(); release.wait(2)
    job = dict(cancel=threading.Event(), engine=NS(cancel=blocking_cancel))
    page = NS(_coverage_job=job)
    started = time.monotonic()
    BuildStepPage._cancel_measurements(page)
    try:
        assert time.monotonic() - started < 0.1
        assert job['cancel'].is_set() and entered.wait(0.5)
    finally: release.set()


def test_pending_publication_resumes_once_after_reopen(tmp_path, monkeypatch):
    import threading
    from PySide6.QtWidgets import QApplication
    from app2.qt.bridge import SessionBridge
    from app2.qt.build_step import BuildStepPage
    app = QApplication.instance() or QApplication([])
    stage = tmp_path/'stages'; stage.mkdir()
    (stage/'published.json').write_text(json.dumps(dict(generation='pending', coverage_pending=True)))
    starts=[]; monkeypatch.setattr(threading.Thread, 'start', lambda worker: starts.append(worker))
    page=BuildStepPage(SessionBridge(), gl=False); page.host_root=str(tmp_path)
    try:
        page._resume_measurements(); page._resume_measurements()
        assert len(starts)==1 and page.coverage_state.text()=='Measuring coverage...'
    finally: page.shutdown(); page.close()


def test_prefetch_hash_runs_off_gui_thread(tmp_path, monkeypatch):
    import threading
    from app2.qt.build_step import BuildStepPage
    from app2.builder import quality_cache
    entered=threading.Event(); main=threading.get_ident()
    mesh=tmp_path/'mesh.ply'; sides=tmp_path/'sides.json'
    mesh.write_bytes(b'mesh'); sides.write_bytes(b'sides')
    def identity(*args):
        assert threading.get_ident()!=main
        entered.set(); raise RuntimeError('Stop after checking thread identity')
    monkeypatch.setattr(quality_cache,'identity',identity)
    snap=NS(final=lambda stage:stage, out=lambda node, kind=None:str(sides if kind=='sides' else mesh))
    page=NS(snap=snap, running=lambda:False, host_root=str(tmp_path))
    BuildStepPage._prefetch_quality(page)
    assert entered.wait(1)


def test_native_fin_deferred_undo_redo_and_reopen_preserve_publication():
    from test_app2_round16_undo_ui import NativeFixture
    from stages.engine import Engine
    from stages.project import Project
    case=NativeFixture(); case.setUp()
    try:
        case.e.clear_ops('fused.plus'); case.run_()
        model=Path(case.tmp)/'model.ply'; initial=model.read_bytes()
        anchors=[[[12,0,2],[0,0,1]],[[2,0,12],[1,0,0]],[[20,0,2],[0,0,1]]]
        case.e.add_op('meshed',dict(tool='fin',params=dict(anchors=json.dumps(anchors),thickness=2,reach=8)))
        assert case.e.run(defer_report=True).ok
        edited=model.read_bytes(); assert edited != initial
        reopened=Engine(Project.open(case.tmp))
        assert len(reopened.recipe()['ops']['meshed'])==1
        reopened.undo(); assert reopened.run(defer_report=True).ok
        assert model.read_bytes()==initial
        reopened.redo(); assert reopened.run(defer_report=True).ok
        assert model.read_bytes()==edited
        assert json.loads((Path(case.tmp)/'stages/published.json').read_text())['coverage_pending']
    finally: case.tearDown()


def test_ui_backend_requests_early_publication_only_for_publishing_runs():
    from app2.builder.stagehost import StagesRun
    assert "--defer-report" in StagesRun().argv("owned-project")
    assert "--defer-report" not in StagesRun(publish=False).argv("owned-project")
    assert "--defer-report" not in StagesRun(defer_report=False).argv("owned-project")


def test_guide_default_frames_part_and_keeps_everything_separate():
    from app2.qt.scan_guide import show, frame_view
    framed = []; depth = []; part = (np.zeros(3), np.ones(3) * 10)
    view = NS(_box=part, items={}, cam=NS(frame=lambda *box: framed.append(box), fit_depth=lambda *box: depth.append(box)),
              overlay=NS(update=lambda: None), update=lambda: None)
    guide = {"poses": [{"T": np.eye(4).tolist(), "distance": 300}],
             "lens": dict(width=640, height=480, cx=320, cy=240, fx=500, fy=500, far=400)}
    show(view, guide, frame=True)
    assert np.array_equal(framed[-1][1], part[1])
    assert np.array_equal(view._box[1], part[1])
    frame_view(view, True)
    assert framed[-1][1][2] >= 350
    frame_view(view)
    assert np.array_equal(framed[-1][1], part[1])


def test_follow_on_result_does_not_replace_active_foreground_job_label(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app2.qt.bridge import SessionBridge
    from app2.qt.build_step import BuildStepPage
    app = QApplication.instance() or QApplication([])
    page = BuildStepPage(SessionBridge(), gl=False)
    page.host_root = str(tmp_path)
    job = dict(root=str(tmp_path), generation="owned-generation")
    page._coverage_job = job
    page.job_state.setText("Making the patch preview."); page.job_progress.show()
    try:
        page._coverage_finished(dict(job=job, error="coverage failed"))
        assert page.job_state.text() == "Making the patch preview."
        assert not page.job_progress.isHidden()
        assert "retry" in page.coverage_state.text()
    finally: page.shutdown(); page.close()


def test_guide_controls_and_cube_projection_routes():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import BuildViewPanel
    from app2.qt.scan_guide import show
    app = QApplication.instance() or QApplication([])
    panel = BuildViewPanel(True)
    panel.view._box = (np.zeros(3), np.ones(3) * 10)
    guide = {"poses": [{"T": np.eye(4).tolist(), "distance": 300}],
             "lens": dict(width=640, height=480, cx=320, cy=240, fx=500, fy=500, far=400)}
    show(panel.view, guide, frame=True)
    part = panel.view.cam.view.copy()
    panel.frame_everything.click(); assert not np.array_equal(panel.view.cam.view, part)
    panel.frame_part.click(); assert np.allclose(panel.view.cam.view, part)
    panel.guide_ghosts.click(); assert not panel.view.guide_ghosts
    panel.guide_ghosts.click(); assert panel.view.guide_ghosts
    for mode in ("orthographic", "perspective"):
        panel.view.nav_cube._projection(mode); assert panel.view.cam.projection_mode == mode
    panel.close()
