"""Frame paths use completed plane results while the existing finder owns fitting."""
import queue
import threading
from types import SimpleNamespace
import numpy as np
from scanner.table_plane import SidePlane
from app2.scan.bed_adapter import ScannerBed, SideState
from app2.scan.session import ScanSession


def setup():
    table = SidePlane()
    table.pts = [np.zeros((1500,3)) for _ in range(8)]
    angle = np.linspace(0,2*np.pi,8,endpoint=False)
    table.centres = list(np.column_stack((300*np.cos(angle),300*np.sin(angle),np.full(8,300.))))
    table._sampling_mm = 1.
    table._cache = (4,(np.array([0.,0.,1.]),0.,100,0.))
    table._top_support = {'verified':True}
    table._disc_cache = (np.zeros(3),100.)
    bed = ScannerBed()
    side = type('Side',(),{})()
    side.table,side.ps,side.idx = table,0,10
    side.bed = SideState(bed._new_sidebed())
    side.frames,side.dirty = [],True
    session = ScanSession.__new__(ScanSession)
    session.bed,session.recording,session.picked,session.table = bed,True,side,None
    return table,bed,side,session


def test_frame_reads_never_run_the_plane_fitter(monkeypatch):
    table,bed,side,session = setup()
    def forbidden(self,min_frames=8,max_tilt_deg=8.,share=.3,sharp=4.):
        raise AssertionError('frame path called the plane fitter')
    monkeypatch.setattr(SidePlane,'plane',forbidden)
    assert session._platter(side) is not None
    assert session._base_plane(side) is not None
    side.bed.sb.state = 'found'
    side.bed.sb.fixed = {'n':np.array([0.,0.,1.]),'centre':np.zeros(3)}
    side.bed.sb.check(table,cached=True)


def test_existing_finder_runs_one_snapshot_while_frames_read_cache(monkeypatch):
    table,bed,side,session = setup()
    entered,release = threading.Event(),threading.Event()
    def fitting(self,min_frames=8,max_tilt_deg=8.,share=.3,sharp=4.):
        assert threading.current_thread() is not threading.main_thread()
        entered.set()
        assert release.wait(3.)
        self._cache = (len(self.pts),(np.array([0.,0.,1.]),0.,100,0.))
        self._top_support = {'verified':True}
        return self._cache[1]
    monkeypatch.setattr(SidePlane,'plane',fitting)
    bed.running = True
    worker = threading.Thread(target=bed._work,daemon=True)
    worker.start()
    try:
        bed.request_auto(side)
        assert entered.wait(1.)
        for _ in range(20):
            bed.request_auto(side)
            assert session._platter(side) is not None
            assert session._base_plane(side) is not None
        assert bed.jobs.empty()
        release.set()
        result = bed.res.get(timeout=2.)
        bed.res.put(result)
        lines,changed = bed.drain([side],side,lambda sd:1)
        assert changed
        assert table._cache[0] == len(table.pts)
        assert table.cached_certified_plane() is not None
    finally:
        release.set()
        bed.running = False
        worker.join(2.)


def test_late_auto_result_from_another_pass_does_not_replace_cache():
    table,bed,side,session = setup()
    snapshot = table.snapshot()
    snapshot._cache = (8,(np.array([0.,1.,0.]),123.,100,0.))
    snapshot._top_support = {'verified':True}
    original = table._cache
    bed.res.put(dict(kind='auto',side=side,table=table,ps=0,snapshot=snapshot))
    side.ps = 1
    assert bed.drain([side],side,lambda sd:1) == ([],False)
    assert table._cache is original


def test_newer_synchronous_result_and_worker_errors_keep_previous_cache():
    table,bed,side,session = setup()
    snapshot = table.snapshot()
    snapshot._cache = (8,(np.array([0.,1.,0.]),123.,100,0.))
    table.pts.append(np.zeros((1500,3)))
    table._cache = (9,table._cache[1])
    original = table._cache
    assert not table.install_snapshot(snapshot)
    bed.res.put(dict(kind='auto',side=side,table=table,ps=0,snapshot=snapshot,error='fixture failure'))
    assert bed.drain([side],side,lambda sd:1) == ([],False)
    assert table._cache is original


def test_dropped_auto_job_releases_its_busy_marker():
    table,bed,side,session = setup()
    bed.running = True
    bed.request_auto(side)
    job = bed.jobs.get_nowait()
    assert bed._auto_pending
    bed._job_done(job)
    assert not bed._auto_pending
    assert bed._pending == 0
    bed.running = False
