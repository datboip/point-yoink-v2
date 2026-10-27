"""Pause, Back 5 / 10 / 30, Forward and Continue on a scanning side: the side's files, its pose table, its counts and
the live model agree on the frames kept, and a build only sees those. A synthetic replayed side (no scanner, no GPU:
PlaybackTracker stands in for the CUDA tracker and remembers which frames its model holds)."""
import glob, json, os, struct, tempfile, threading, time, zlib
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink-rewind-')) / 'config.json')
os.environ['POINTYOINK_NO_DEVICE'] = '1'
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('XMODIFIERS', '@im=none')
import numpy as np
import pytest
from app2.scan import fmt
from app2.scan.project import Project, Side
from app2.scan.tracking import PlaybackTracker, TrackerHost
from app2.scan import ui_model

W, H = fmt.W, fmt.H
CAL = (500.0, 500.0, 400.0, 300.0)


def param_dir(root):
    d = os.path.join(root, 'param'); os.makedirs(d, exist_ok=True)
    fx, fy, cx, cy = CAL
    Path(d, 'Pl.bin').write_bytes(struct.pack('<HH9f', W, H, fx, 0, cx, 0, fy, cy, 0, 0, 1))
    return d


def depth(k):
    """A flat wall at 300 mm with frame k's own mark, so every frame's bytes differ."""
    d = np.full((H, W), 3000, np.uint16); d[10, 10 + k] = 3100; d[20:40, 20:40] = 2900 + k; return d


def pose(k):
    a = np.radians(2.0 * k); T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]; T[:3, 3] = (0.01 * np.sin(a), 0, 0.001 * k)
    return T


def crc(d): return zlib.crc32(np.ascontiguousarray(d).tobytes())


def replay_project(root, n):
    """A saved one-side project with n frames to play back (as the Tk scanner / 2.0 lay it out)."""
    proj = os.path.join(root, 'src', 'ProjectLive01012026000000'); os.makedirs(os.path.join(proj, 'data'))
    folder = fmt.new_side_folder(proj, param_dir(root)); cache = os.path.join(folder, 'cache')
    stamps, entries = [], []
    for k in range(n):
        name = fmt.frame_name(0, k); M = pose(k).copy(); M[:3, 3] *= 1000.0
        fmt.write_frame(cache, name, depth(k), np.zeros((H, W), np.uint8), M, irr=np.full((H,W),123,np.uint8))
        stamps.append(dict(frame=name, t=float(k), ok=True, fit=0.9, rmse=0.5, reloc=False, near=100, far=450, led=False, gain=4, hinted=False))
        entries.append((0, k, M))
    fmt.write_side_report(cache, entries, stamps)
    return proj


# ---------------------------------------------------------------------------------------------- the side's files alone
def saved_side(root, oks):
    """A side with frames whose tracked flags are oks, saved like the session saves it."""
    proj = os.path.join(root, 'ProjectLive01012026000001')
    pr = Project(proj, param_dir(root)); sd = pr.make_side()
    for k, ok in enumerate(oks):
        name = fmt.frame_name(0, k); M = np.eye(4); M[0, 3] = k
        fmt.write_frame(sd.cache, name, depth(k), np.zeros((H, W), np.uint8), M, irr=np.full((H,W),123,np.uint8))
        from PIL import Image
        Image.new("RGB",(2,2),(k,20,30)).save(os.path.join(sd.cache,name+".img"),format="JPEG")
        sd.stamps.append(dict(frame=name, t=float(k), ok=ok, fit=0.9, rmse=0.5, reloc=False, near=100, far=450, led=False, gain=4, hinted=False))
        if ok: sd.entries.append((0, k, M)); sd.table.pts.append(np.zeros((1, 3))); sd.table.centres.append(np.zeros(3))
    sd.ps = 0; sd.idx = len(oks); sd.lost = oks.count(False); sd.dirty = True
    pr.save_side(sd, log=lambda *a: None)
    return pr, sd


def names_in(folder): return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(folder, '*.dph')))


def on_disk(sd):
    """What a build reads: the frames in cache/, the pose table, the track json."""
    from fuse import read_global_poses
    gp = read_global_poses(os.path.join(sd.cache, fmt.POSE_TABLE))
    with open(os.path.join(sd.cache, fmt.TRACK_JSON)) as f: tj = json.load(f)
    return names_in(sd.cache), sorted(fmt.frame_name(*k) for k in gp), tj


def test_back_forward_and_forget_keep_the_side_consistent(tmp_path):
    oks = [False, True, True, True, False, True, True, True, True, False, True, True]
    pr, sd = saved_side(str(tmp_path), oks)
    assert sd.rewind_room == 10                                  # the lost first frame and the first tracked one stay
    got = pr.rewind(sd, 5); pr.save_side(sd, log=lambda *a: None)
    st, en = got
    assert len(st) == 5 and len(en) == 4 and sd.frames == 7 and len(sd.entries) == 5 and sd.lost == 2
    assert len(sd.table.pts) == 5                                # Auto remove base: the taken frames' samples gone too
    cache_names, table, tj = on_disk(sd)
    kept = [fmt.frame_name(0, k) for k in range(7)]
    assert cache_names == kept and tj['frames'] == 7 and tj['tracked'] == 5 and [x['frame'] for x in tj['stamps']] == kept
    assert table == [fmt.frame_name(0, k) for k in range(7) if oks[k]]
    assert names_in(os.path.join(sd.folder, fmt.DISCARDED)) == [fmt.frame_name(0, k) for k in range(7, 12)]
    assert all(os.path.exists(os.path.join(sd.folder, fmt.DISCARDED, fmt.frame_name(0, 8) + e)) for e in fmt.FRAME_EXTS)
    pr.rewind(sd, 30); pr.save_side(sd, log=lambda *a: None)     # clamped: back to the first tracked frame
    assert sd.frames == 2 and len(sd.entries) == 1 and sd.rewind_room == 0 and pr.rewind(sd, 5) is None
    assert sd.rewound_frames == 10 and len(sd.rewound) == 2
    pr.forward(sd); pr.save_side(sd, log=lambda *a: None)        # undo the newest Back only
    assert sd.frames == 7 and len(sd.entries) == 5 and on_disk(sd)[0] == kept and sd.rewound_frames == 5
    assert pr.forget_rewound(sd) == 5 and pr.forward(sd) is None  # Continue: final, the files stay aside
    assert names_in(os.path.join(sd.folder, fmt.DISCARDED)) == [fmt.frame_name(0, k) for k in range(7, 12)]
    again = Side(sd.folder)                                       # reopened: the counts from disk, no reused pass
    assert again.frames == 7 and len(again.entries) == 5 and again.lost == 2 and again.rewound == []
    Path(sd.folder, fmt.DISCARDED, 'frame_004_0000.dph').write_bytes(b'')    # a later pass taken back, then reopened
    assert Side(sd.folder).ps == 4


def test_host_rewind_from_checkpoint_matches_a_fresh_fuse(tmp_path):
    """TrackerHost.rewind: newest checkpoint plus the frames after it = the same model as fusing the kept frames again."""
    pr, sd = saved_side(str(tmp_path), [True] * 75)
    frames = {(0, k): depth(k) for k in range(75)}
    rd = lambda side, ps, idx: frames.get((ps, idx))
    host = TrackerHost(CAL, W, H, factory=lambda cal, w, h: PlaybackTracker([], cal, w, h), release=lambda: None)
    host.reseed(sd, rd)
    assert [c[0] for c in host.ckpts] == [30, 60] and host.trk.fused == 75
    pr.rewind(sd, 20)                                            # 55 kept: checkpoint 30, then 25 frames fused again
    assert host.rewind(sd, rd) == 'checkpoint'
    assert host.trk.frames == [crc(depth(k)) for k in range(55)] and host.trk.fused == 55
    from track import F
    assert np.allclose(host.T[:3, 3] * 1000.0, (sd.entries[-1][2] @ F)[:3, 3])
    assert [c[0] for c in host.ckpts] == [30]
    st, en = pr.forward(sd); host.extend(sd, en, rd)
    assert host.trk.frames == [crc(depth(k)) for k in range(75)] and [c[0] for c in host.ckpts] == [30, 60]
    pr.rewind(sd, 50)                                            # 25 kept: before every checkpoint, fused from the start
    assert host.rewind(sd, rd) == 'full' and host.trk.frames == [crc(depth(k)) for k in range(25)]


# ---------------------------------------------------------------------------------------------- the buttons' words
def state(**kw):
    base = dict(picked=2, picked_has_frames=True, recording=False, busy=None, building=False, learning=False, n_sides=2,
                rewind_room=40, rewound=0, forward_n=0, capture_mode='continuous', near=100, far=450, gain=4,
                auto_range=True, auto_gain=False, has_led=False, below_bed=True, remove_base=True, source_kind='replay')
    base.update(kw); return ui_model.ScanState(**base)


def test_buttons_show_back_and_forward_only_while_paused():
    b = ui_model.buttons(state())
    assert b['rewind_visible'] and b['back5'][2] and b['back30'][2] and not b['forward'][2] and b['go'][0] == 'Continue side 2'
    b = ui_model.buttons(state(rewound=15, forward_n=5))
    assert b['forward'] == ('Forward 5', 'plain', True) and b['rewind_note'] == '15 frames taken back'
    assert b['hint'].startswith('Took back 15 frames.')
    assert not ui_model.buttons(state(recording=True))['rewind_visible']
    assert not ui_model.buttons(state(picked=None))['rewind_visible']
    b = ui_model.buttons(state(busy='Going back 5 frames on side 2...'))
    assert b['rewind_visible'] and not b['back5'][2] and not b['forward'][2]
    assert not ui_model.buttons(state(rewind_room=0))['back10'][2]
    for t in (b['rewind_note'], ui_model.HINT_REWOUND, ui_model.REWIND_TIP): assert '—' not in t


def test_scan_page_rewind_bar(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app2.qt import theme
    from app2.qt.shell import MainWindow
    app = QApplication.instance() or QApplication([]); theme.install(app)
    def offline(*a, **k): raise RuntimeError('no scanner in this test')
    win = MainWindow(offline, gl=False, projects_dir=str(tmp_path)); win.resize(1688, 1044); win.show(); win.show_step(0)
    app.processEvents(); page = win.scan
    calls = []

    class Fake:
        def __getattr__(self, name): return lambda *a: calls.append((name,) + a)
    try:
        page.session = Fake(); page.ended = False
        page.on_state(state(rewound=10, forward_n=10)); app.processEvents()
        bar = page.rewind_bar
        assert bar.isVisible() and bar.note.text() == '10 frames taken back' and bar.fwd.text() == 'Forward 10'
        assert page.model.geometry().contains(bar.geometry())
        bar.backs[10].click(); bar.fwd.click(); page.key_space()
        assert calls == [('rewind', 10), ('forward',), ('toggle_scan',)]
        page.on_state(state(recording=True)); app.processEvents()
        assert not bar.isVisible()
    finally:
        page.session = None; win.close(); app.processEvents()


# ---------------------------------------------------------------------------------------------- a replayed side, live
class Run:
    """A ScanSession on a replayed side, frame by frame (lockstep). hold_at: frame numbers where the player stops
    after pressing Start / Pause (the test lets it go on)."""
    def __init__(self, root, n, press, hold):
        from app2.scan.session import ScanSession, ScanConfig
        from app2.scan.sources import ReplaySource
        self.src_proj = replay_project(root, n); self.press, self.hold = press, hold; self.go = threading.Event()
        self.events = []
        self.src = ReplaySource(self.src_proj, lockstep=True, on_before=self._before)
        cfg = ScanConfig(os.path.join(root, 'projects'), param_dir(root), auto_range=False, below_bed=False,
                         tracker_factory=lambda cal, w, h: PlaybackTracker([pose(k) for k in range(n)], cal, w, h),
                         release=lambda: None, view_launcher=lambda argv: None)
        self.s = ScanSession(cfg, self.src, lambda kind, payload: self.events.append((kind, payload)))
        self.s.start()

    def _before(self, meta):
        k = meta['k']
        if k in self.press: self.s.toggle_scan()
        if k in self.hold: self.sync(); self.go.wait(30); self.go.clear()

    def sync(self, timeout=30):
        """Every command sent so far has run."""
        ev = threading.Event(); self.s._post(ev.set); assert ev.wait(timeout)

    def until(self, cond, timeout=30):
        t = time.monotonic() + timeout
        while not cond():
            assert time.monotonic() < t, 'timed out'
            time.sleep(0.02)

    def close(self):
        self.go.set(); self.s.close(); assert self.s.closed.wait(30)


def test_pause_back_forward_on_a_replayed_side(tmp_path):
    r = Run(str(tmp_path), 60, press={0, 35}, hold={35})
    try:
        r.until(lambda: r.src.sent >= 35 and r.s.picked is not None and not r.s.recording and r.s.busy is None)
        r.sync(); s = r.s; sd = s.picked; model = lambda: s.host.trk.frames
        # paused: nothing more recorded (the player waits at frame 35)
        assert sd.frames == 35 and len(sd.entries) == 35 and names_in(sd.cache) == [fmt.frame_name(0, k) for k in range(35)]
        assert [c[0] for c in s.host.ckpts] == [30] and model() == [crc(depth(k)) for k in range(35)]
        st = s._state(); assert st.rewind_room == 34 and st.rewound == 0
        s.rewind(10); r.sync()
        assert sd.frames == 25 and len(sd.entries) == 25 and model() == [crc(depth(k)) for k in range(25)]
        cache_names, table, tj = on_disk(sd)
        assert cache_names == [fmt.frame_name(0, k) for k in range(25)] and table == cache_names
        assert tj['frames'] == 25 and tj['tracked'] == 25
        card = [p for k, p in r.events if k == 'sides'][-1][0]
        assert card['frames'] == 25 and card['kept'] == 25
        cloud = [p for k, p in r.events if k == 'cloud'][-1]; assert cloud['side_no'] == 1 and len(cloud['P'])
        assert np.allclose(s.host.T[:3, 3], pose(24)[:3, 3])        # tracking goes on from the last kept frame
        s.forward(); r.sync()
        assert sd.frames == 35 and model() == [crc(depth(k)) for k in range(35)]
        assert on_disk(sd)[0] == on_disk(sd)[1] == [fmt.frame_name(0, k) for k in range(35)]
        s.rewind(5); r.sync()                                        # from the checkpoint at 30, nothing to fuse again
        assert sd.frames == 30 and model() == [crc(depth(k)) for k in range(30)] and s._state().forward_n == 5
        assert names_in(os.path.join(sd.folder, fmt.DISCARDED)) == [fmt.frame_name(0, k) for k in range(30, 35)]
        assert any(k == 'log' and 'took back 5 frames' in p for k, p in r.events)
    finally:
        r.close()


def test_continue_after_back_records_on_and_builds_from_kept_frames(tmp_path):
    root = str(tmp_path)
    r = Run(root, 60, press={0, 30, 40, 50}, hold={30, 50})
    try:
        r.until(lambda: r.src.sent >= 30 and r.s.picked is not None and not r.s.recording and r.s.busy is None)
        r.sync(); s = r.s; sd = s.picked
        assert sd.frames == 30
        s.rewind(5); r.sync()
        assert sd.frames == 25 and sd.rewound_frames == 5
        r.go.set()                                                    # frames 30..39 arrive paused, Continue at 40
        r.until(lambda: r.src.sent >= 50 and not r.s.recording and r.s.busy is None); r.sync()
        assert sd.rewound == [] and sd.ps == 1                       # Continue made them final; a new pass
        assert sd.frames == 35 and len(sd.entries) == 35             # 25 kept + frames 40..49
        new = [fmt.frame_name(1, k) for k in range(10)]
        cache_names, table, tj = on_disk(sd)
        assert cache_names == sorted([fmt.frame_name(0, k) for k in range(25)] + new) and table == cache_names
        assert names_in(os.path.join(sd.folder, fmt.DISCARDED)) == [fmt.frame_name(0, k) for k in range(25, 30)]
        # the model holds the kept frames and the new pass
        assert s.host.trk.frames == [crc(depth(k)) for k in range(25)] + [crc(depth(k)) for k in range(40, 50)]
        # a build reads the pose table's frames from cache/ only: exactly the kept ones
        from app2.scan.sources import ReplaySource
        plan = ReplaySource(s.project.path).plan
        assert [p[2] for p in plan] == [x['frame'] for x in tj['stamps']] and len(plan) == 35
    finally:
        r.close()
