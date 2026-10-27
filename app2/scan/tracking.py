# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tracker's home: one track.Tracker at a time, used from one thread only (the session's worker). The Tracker holds
an Open3D CUDA voxel grid; only one model may live on the GPU at a time (a second grid in the same process made tracking
crawl: 35 ms per frame alone, minutes with an older grid still alive), so switching sides frees the old one first.
Every tracker call goes through TrackerHost, so the tracker can move into a child process later without the session
changing (the GIL is shared with the window today)."""
import gc, threading
import numpy as np
from track import F


NO_CUDA = ("Live scanning needs an NVIDIA GPU and an Open3D build with CUDA, and this one has none. Scanning on the "
           "MIRACO and importing its projects still works, and so does building.")


def cuda_device():
    """Open3D's CUDA:0, or a plain-language error when the GPU or a CUDA build of Open3D is missing."""
    import open3d.core as o3c
    try: ok = o3c.cuda.is_available()
    except Exception: ok = False
    if not ok: raise RuntimeError(NO_CUDA)
    return o3c.Device("CUDA:0")


LIVE_MAX_DEPTH_MM=1000.0   # Existing live depth cap, shared by the software range control.

def cuda_tracker(cal, w, h):
    """The live scanner's tracker: the range cut does the cutting; the slow global search runs every 3rd lost frame so
    the view stays live; barely seen spots never seen again are dropped after 8 looks."""
    from track import Tracker
    return Tracker(cal, w, h, cuda_device(), max_depth_mm=LIVE_MAX_DEPTH_MM, reloc_every=3, floater_looks=8)


def cuda_cache_release():
    """Hand Open3D's cached CUDA memory back (what live tensors use stays)."""
    try:
        import open3d.core as o3c; o3c.cuda.release_cache()
    except Exception: pass


def free_gpu():
    """Drop what Python still holds of the old model and hand the GPU memory back."""
    gc.collect(); cuda_cache_release()


def rot_deg(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def pose_gap(A, B):
    """How far apart two camera poses are: (mm, degrees). Poses in tracker metres."""
    D = np.linalg.inv(A) @ B
    return float(np.linalg.norm(D[:3, 3]) * 1000.0), rot_deg(D[:3, :3])


def source_cloud(trk, raw):
    """A depth frame as Tracker.step() matches it: the same depth cut, cleaning and pixel stride (track.py, step())."""
    import open3d as o3d, open3d.core as o3c
    from track import clean_frame
    z = raw.copy(); z[z * trk.depth_scale > trk.DMAX * 1000.0] = 0
    z = clean_frame(z, trk.cal, trk.grid, depth_scale=trk.depth_scale)
    return o3d.t.geometry.PointCloud.create_from_depth_image(
        o3d.t.geometry.Image(o3c.Tensor(z)).to(trk.dev), trk.K, o3c.Tensor(np.eye(4)), trk.DS, trk.DMAX, stride=trk.stride)


# The first frame of a Continue is not taken on ICP alone. ICP from where the camera was at the Pause fitted a wrong spot
# well enough to pass (replay of ProjectLive09262026203159 side 12: the table turned about 50 deg during a 5 s pause;
# ICP from the old pose took fit 0.78, rmse 0.96 mm at 41 deg / 125 mm off and the whole pass fused as a ghost copy).
# The global search (the tracker's own relocalisation, which does not start from the old pose) is the second opinion,
# but one search alone is not enough either: on that near-round part a single search also landed 110 to 179 deg off
# with a fit better than ICP's. Measured on replays (continue_probe.py): right answers fit 0.95 to 0.999, wrong ones
# 0.61 to 0.92.
AGREE_MM, AGREE_DEG = 10.0, 5.0          # ICP and the search agree: the part did not move
STRONG_FIT = 0.95                        # a search counts only above this fit (0.917 was 51 mm off in one replay)
STRONG_RMSE = 0.7                        # ... and below this residual, mm (fit 0.996 at 0.75 mm put 3 frames 12 mm off)
STRONG_ICP = 0.95                        # ICP from the pause counts alone above this fit, when no strong search disagrees
STEP_MM, STEP_DEG = 35.0, 8.0            # how far the camera may move per frame between two searches (3 fps on a turning
                                         # table: 17 to 28 mm and 3 to 6.5 deg seen)
SEARCH_EVERY_AFTER = 6                   # after this many searches in a pass, search only every 3rd frame (about 1 s
                                         # each), unless a strong search in the last 3 frames waits for its second


class GapPlacer:
    """Places the first frame(s) of a Continue on a model that already has frames. Per frame, two answers: ICP from
    the last good pose and the global search. A frame is taken when
      "agree"    both fit and lie within AGREE_MM / AGREE_DEG: ICP's pose
      "search"   a strong search (fit >= STRONG_FIT, rmse <= STRONG_RMSE) matches the strong search of one of the last
                 3 frames, within the camera's move between them (the part moved while paused, and two separate
                 searches say where to)
      "icp"      ICP fits at STRONG_ICP / STRONG_RMSE or better and no strong search says otherwise
    else it stays lost ("unsure"; a strong search is kept as the candidate for the next frames) or had too little depth
    ("few"). place() returns (pose or None, why, info); info has both fits (fit, rmse mm), their gap (mm, deg) and the
    seconds the search took."""
    def __init__(self): self.key = None; self.cand = None; self.tries = 0

    def _source(self, trk, raw): return source_cloud(trk, raw)

    def _icp_answer(self, trk, src):
        """ICP from the last good pose: (pose, fit, rmse) or None."""
        try: return trk._icp(src, trk.model, trk.last_good)
        except RuntimeError: return None

    def place(self, trk, raw, ps, idx):
        import time
        if self.key != ps: self.key = ps; self.cand = None; self.tries = 0      # a new pass: start over
        if self.cand is not None and idx - self.cand[0] > 3: self.cand = None   # too old to be matched
        if self.tries >= SEARCH_EVERY_AFTER and idx % 3 and self.cand is None: return None, "unsure", dict(skipped=True, secs=0.0)
        src = self._source(trk, raw)
        if len(src.point.positions) < 500: return None, "few", dict(secs=0.0)
        self.tries += 1
        trk.model = trk._model_cloud()                    # fresh: the model as the pause left it
        a = self._icp_answer(trk, src)
        t0 = time.monotonic(); b = trk._relocalize(src, trk.model); dt = time.monotonic() - t0
        # the poses are numpy views of Open3D tensors: copies, so a kept candidate never holds an Open3D buffer (one
        # still alive at exit made Open3D report a leak and the process exit with status 1)
        if a is not None: a = (np.array(a[0]), a[1], a[2])
        if b is not None: b = (np.array(b[0]), b[1], b[2])
        a_ok = a is not None and a[1] > trk.min_fit and a[2] < trk.max_rmse
        b_ok = b is not None and b[1] > trk.min_fit and b[2] < trk.max_rmse * 0.8      # the tracker's own bar for a search
        gap = pose_gap(a[0], b[0]) if (a is not None and b is not None) else None
        info = dict(icp=None if a is None else (round(a[1], 3), round(a[2], 3)),
                    search=None if b is None else (round(b[1], 3), round(b[2], 3)), gap=gap, secs=dt)
        if a_ok and b_ok and gap[0] <= AGREE_MM and gap[1] <= AGREE_DEG: return a[0], "agree", info
        strong = b_ok and b[1] >= STRONG_FIT and b[2] <= STRONG_RMSE
        if strong:
            if self.cand is not None:
                k = idx - self.cand[0]; moved = pose_gap(self.cand[1], b[0]); info["moved"] = (k, moved)
                if 1 <= k <= 3 and moved[0] <= STEP_MM * k and moved[1] <= STEP_DEG * k:
                    self.cand = (idx, b[0])               # the newest one: should ICP still refuse it, the next compares to it
                    return b[0], "search", info
            self.cand = (idx, b[0])
            return None, "unsure", info
        if a_ok and a[1] >= STRONG_ICP and a[2] <= STRONG_RMSE: return a[0], "icp", info
        return None, "unsure", info


def cloud_of(trk):
    """The model so far as arrays for the 3D view (worker thread only): positions in mm (float32, n x 3), unit normals
    (float32), coverage level per point (uint8: 0 barely seen, 1 getting there, 2 well covered) and the well-covered
    share, which is exactly what Tracker.render(coverage=True) reports. Fresh arrays: safe to hand to another thread."""
    if hasattr(trk, "cloud_arrays"): return trk.cloud_arrays()           # stand-in trackers (tests)
    if trk.model is None: trk.model = trk._model_cloud()
    Pm = trk.model.point.positions.cpu().numpy()
    if not len(Pm): return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), 0.0
    P = (Pm * 1000.0).astype(np.float32)
    N = np.array(trk.model.point.normals.cpu().numpy(), np.float32)
    L = trk.levels if (trk.levels is not None and len(trk.levels) == len(Pm)) else trk.coverage_levels(Pm)
    L = np.array(L, np.uint8)
    return P, N, L, float((L == 2).mean())


def grid_snapshot(trk):
    """The tracker's model as it is now, for going back to it (Back 5 / 10 / 30): the TSDF grid's active blocks copied
    to the CPU (keys, tsdf, weight). About 30 ms and 56 MB for a 300-frame side (RTX 3080). Stand-in trackers bring
    their own snapshot(); None when the tracker has neither."""
    if hasattr(trk, "snapshot"): return trk.snapshot()
    vbg = getattr(trk, "vbg", None)
    if vbg is None: return None
    import open3d.core as o3c
    h = vbg.hashmap(); idx = h.active_buf_indices().to(o3c.int64)
    return h.key_tensor()[idx].cpu(), [v[idx].cpu() for v in h.value_tensors()]


def grid_restore(trk, snap):
    """Put a grid_snapshot back into the tracker's own grid (every block now in it is erased first). About 40 ms."""
    if hasattr(trk, "restore"): trk.restore(snap); return
    import open3d.core as o3c
    h = trk.vbg.hashmap(); keys, vals = snap
    live = h.active_buf_indices().to(o3c.int64)
    if live.shape[0]: h.erase(h.key_tensor()[live])
    # a block the next frames add takes a freed buffer as it is: zero, as in a new grid (stale values there made the
    # model differ from a fresh fuse of the same frames)
    for v in h.value_tensors(): v[:] = 0
    if keys.shape[0]:
        buf, ok = h.activate(keys.to(trk.dev)); buf = buf[ok].to(o3c.int64)
        for dst, src in zip(h.value_tensors(), vals): dst[buf] = src.to(trk.dev)
    trk.model = None; trk.levels = None
    if hasattr(trk, "_miss_k"):                           # floater looks start over, as on a side loaded from disk
        trk._miss_k = np.zeros(0, np.int64); trk._miss_n = np.zeros(0, np.int16); trk._dead = np.zeros(0, np.int64)


def settle_pose(trk, T, fused):
    """After going back: tracking carries on from the last kept frame's pose T (metres) with that many frames fused."""
    trk.T = np.asarray(T, float).copy(); trk.fused = fused
    if hasattr(trk, "last_good"): trk.last_good = trk.T.copy()
    if hasattr(trk, "vel"): trk.vel = np.eye(4)
    if hasattr(trk, "streak"): trk.streak = 0
    trk.model = None


class TrackerHost:
    """Holds the tracker of the side being scanned (owner = that side). Every call checks it runs on the thread that
    made the host: the tracker is never touched from the window or a pump thread."""
    # Open3D keeps the CUDA memory of every step's temporaries in its cache: 6 to 8.5 MB more per tracked frame (1.1 GB
    # at the start of a side, 7.6 GB at 761 frames, measured; a 10 GB card runs out at about 1000). Handing the cache
    # back every CACHE_EVERY steps holds a side at 1.1 to 1.3 GB and costs no step time.
    CACHE_EVERY = 30
    # Back 5 / 10 / 30: the model is kept every CKPT_EVERY tracked frames (grid_snapshot, CPU memory), the newest
    # CKPT_KEEP of them. Going back restores the newest one at or before the kept frames and fuses the few after it
    # again (at most CKPT_EVERY - 1: a frame's clean-up is 7 to 16 ms, the fuse itself under 1 ms). Fusing a whole
    # 300-frame side again took 7 s, so that is only done when going back past the oldest one kept.
    CKPT_EVERY, CKPT_KEEP = 30, 6

    def __init__(self, cal, w, h, factory=cuda_tracker, release=free_gpu):
        self.cal, self.w, self.h, self.factory, self._release = cal, w, h, factory, release
        self.trk = None; self.owner = None; self.thread = threading.get_ident(); self.steps = 0; self.placer = GapPlacer()
        self.seeder=None
        self.marker_required=False
        self._relock = None; self.ckpts = []               # [(tracked frames in the model, grid_snapshot)], oldest first

    def _check(self):
        assert threading.get_ident() == self.thread, "tracker used off its thread"

    def release(self):
        """Free the model from the GPU (New side, pick, delete, build, close)."""
        self._check()
        self.seeder=None
        if self.trk is None: return
        self.trk = None; self.owner = None; self.placer = GapPlacer(); self.ckpts = []; self._release()

    def new(self, owner, factory=None):
        """A fresh tracker for owner. factory: another maker than the host's own (Touch up's ModelTracker); a tracker
        may name its own Continue placer (make_placer)."""
        self._check(); self.release()
        self.marker_required=False
        selected=factory or self.factory
        profile=getattr(owner,'scan_profile',{})
        if selected is cuda_tracker and (profile.get('mode')=='far' or profile.get('live_build')):
            from track import Tracker
            import inspect
            defaults=inspect.signature(Tracker).parameters;base=float(defaults['voxel_mm'].default);rmse=float(defaults['max_rmse'].default)
            spacing=float(profile['live_build']['voxel_mm']) if profile.get('live_build') else max(base,float(profile.get('voxel_mm',base)))
            self.trk=Tracker(self.cal,self.w,self.h,cuda_device(),voxel_mm=spacing,max_depth_mm=LIVE_MAX_DEPTH_MM,reloc_every=3,floater_looks=8,max_rmse=rmse)
            self.trk.dense_sparse_registration=True
        else:self.trk=selected(self.cal,self.w,self.h)
        surface=profile.get('surface_start')
        if surface is not None:self.trk.depth_jump_mm=surface['depth_jump_mm']
        self.trk.keep_integration_depth=bool(profile.get('live_build'))
        self.owner=owner
        from app2.scan.motion_seed import PoseSeeder
        self.seeder=PoseSeeder(getattr(owner,"motion_calibration",None),history=self.CKPT_KEEP)
        self.placer = getattr(self.trk, "make_placer", GapPlacer)(); return self.trk

    def has(self, side): return self.trk is not None and self.owner is side

    def reseed(self, side, read_depth, progress=None, factory=None):
        """Rebuild a saved side's model from its frames at their saved poses (no matching), so scanning can carry on.
        Also refills the side's Auto remove base samples. read_depth(side, ps, idx) -> HxW u16 or None."""
        from table_plane import SidePlane
        self.new(side, factory); side.table = SidePlane()
        self._seed(side, side.entries, 0, read_depth, progress)

    def _seed(self, side, entries, n0, read_depth, progress=None, table=True):
        """Fuse entries at their saved poses on top of a model holding n0 tracked frames (checkpoints on the way)."""
        trk = self.trk
        for k, (ps, idx, M) in enumerate(entries):
            raw = read_depth(side, ps, idx)
            if raw is not None:
                T = M @ F; T = T.copy(); T[:3, 3] /= 1000.0
                trk.seed(raw, T)
                if table:
                    try: side.table.add(raw, self.cal, M @ F)
                    except Exception as e: print("table:", e, flush=True)
            self.tracked(n0 + k + 1)
            if progress and k % 25 == 0: progress(k, len(entries))

    def tracked(self, n):
        """A tracked frame went into the model, which now holds n: every CKPT_EVERY-th, the model is kept."""
        if n <= 0 or n % self.CKPT_EVERY or self.trk is None: return
        if self.ckpts and self.ckpts[-1][0] >= n: self.ckpts = [c for c in self.ckpts if c[0] < n]
        try: s = grid_snapshot(self.trk)
        except Exception as e: print("checkpoint:", e, flush=True); return
        if s is None: return
        self.ckpts.append((n, s)); del self.ckpts[:-self.CKPT_KEEP]

    def rewind(self, side, read_depth, progress=None):
        """The model again for side.entries, which Back just made shorter: from the newest checkpoint at or before them
        plus the frames after it, else fused again from the first frame. Tracking then carries on from the last kept
        frame's pose. Returns "checkpoint" or "full". Auto remove base's samples were trimmed by the caller."""
        self._check(); n = len(side.entries)
        self.ckpts = [c for c in self.ckpts if c[0] <= n]
        if self.trk is None or not self.ckpts or not side.entries:
            self.reseed(side, read_depth, progress); return "full"
        c, snap = self.ckpts[-1]
        grid_restore(self.trk, snap); self._release()          # the dropped blocks' memory back to the card
        self._seed(side, side.entries[c:], c, read_depth, progress, table=False)
        M = side.entries[-1][2]; T = M @ F; T = T.copy(); T[:3, 3] /= 1000.0
        settle_pose(self.trk, T, n); self.placer = getattr(self.trk, "make_placer", GapPlacer)()
        from app2.scan.motion_seed import PoseSeeder
        self.seeder=PoseSeeder(getattr(side,"motion_calibration",None),history=self.CKPT_KEEP)
        if hasattr(self.trk,"pose_constraint"):self.trk.pose_constraint=None
        return "checkpoint"

    def extend(self, side, entries, read_depth):
        """Forward: entries that Back took come back on top of the model (and Auto remove base's samples)."""
        self._check(); n0 = len(side.entries) - len(entries)
        self._seed(side, entries, n0, read_depth)
        if entries:
            self.placer = getattr(self.trk, "make_placer", GapPlacer)()
            from app2.scan.motion_seed import PoseSeeder
            self.seeder=PoseSeeder(getattr(side,"motion_calibration",None),history=self.CKPT_KEEP)
            if hasattr(self.trk,"pose_constraint"):self.trk.pose_constraint=None

    # ---- the tracker's calls, for the side it belongs to
    def step(self, raw, after_gap=None, timestamp=None, motion=None, board=None):
        """Track one frame. after_gap: (pass, frame) while no frame of a pass has been taken yet: on a model that has
        frames already (a Continue), GapPlacer places the frame first (a hint from the bed hook wins; stand-in trackers
        skip it). A frame it can't place comes back lost with "unsure" set and never reaches the tracker."""
        self._check(); trk = self.trk; placed = None
        self.steps += 1
        required,self.marker_required=self.marker_required,False
        if required and trk.fused and getattr(trk,'_hint',None) is None:
            trk.streak+=1;trk.vel=np.eye(4)
            return dict(ok=False,T=trk.last_good.copy(),fit=float('nan'),rmse=float('nan'),reloc=False,hinted=False,
                        refused=False,refused_T=None,unsure=True,registration_points=0,
                        refusal_reason='Flat bed needs visible dots or a recent color match.')
        # a tracker that asks for it (Touch up's ModelTracker) is found again after a lost stretch the careful way too:
        # its own quick global search took a flipped pose on a near-round part in a replay and kept it (good fits there)
        if after_gap is None and getattr(trk, "relock", False) and trk.streak >= 2:
            if self._relock is None: self._relock = self.steps
            after_gap = (("relock", self._relock), self.steps)       # a frame the placer refuses never reaches the tracker
        elif getattr(trk, "streak", 1) == 0: self._relock = None
        if self.steps % self.CACHE_EVERY == 0 and hasattr(trk, "vbg"): cuda_cache_release()     # a real (CUDA) tracker
        if after_gap is not None and trk.fused and getattr(trk, "_hint", None) is None and hasattr(trk, "_relocalize"):
            T, why, info = self.placer.place(trk, raw, *after_gap)
            placed = dict(why=why, **info)
            if T is None:
                fit, rmse = info.get("icp") or (float("nan"), float("nan"))
                return dict(ok=False, T=trk.last_good.copy(), fit=fit, rmse=rmse, reloc=False, hinted=False, refused=False,
                            refused_T=None, unsure=True, placed=placed)
            trk.hint(T, guard_mm=AGREE_MM, guard_deg=AGREE_DEG)    # ICP starts there, and may not wander off
        sources=[]
        sparse=getattr(self.owner,'scan_profile',{}).get('mode')=='far' and timestamp is not None
        if sparse and self.seeder is not None and trk.fused and not getattr(trk,'streak',0) and getattr(trk,'_hint',None) is None:
            predicted,sources=self.seeder.predict(timestamp,motion,board)
            if predicted is not None:trk.hint(predicted,guard_mm=AGREE_MM,guard_deg=AGREE_DEG)
        if hasattr(trk,'pose_constraint') and trk.pose_constraint is None:
            # 2026-10-25: Small part mode too. The orbit certifies itself from 8+ poses on one axis (a turntable), snaps
            # each fit back onto that orbit when the fit is no worse (a smooth skin lets ICP slide along the curve; the
            # stripes and lost turns on the hood), and drops itself the moment a pose leaves the orbit (a hand-held move).
            from app2.scan.motion_seed import TurntableOrbit
            trk.pose_constraint=TurntableOrbit(trk.max_rmse)
        if sparse and hasattr(trk,'registration_plane'):
            supported=getattr(getattr(self.owner,'table',None),'certified_plane',None)
            table=getattr(self.owner,'table',None)
            cache=getattr(table,'_cache',(None,None))
            cached=getattr(table,'cached_certified_plane',None)
            fresh=cached is not None or (supported is not None and cache[0]==len(getattr(table,'pts',[])))
            plane=cached() if cached is not None else (supported() if fresh else None)
            if plane is not None:
                # The same saved plane's existing one-millimeter margin, in tracker meters.
                trk.registration_plane=(np.asarray(plane[0]),float(plane[1])/1000,.001)
            elif fresh:trk.registration_plane=None
        r = trk.step(raw)
        if sparse and self.seeder is not None:
            if r['ok']:self.seeder.accepted(r['T'],timestamp,motion,board)
            r['seed_sources']=sources;r['depth_time']=timestamp
        if placed is not None:
            r["placed"] = placed
            if placed["why"] == "search" and r["ok"]: r["reloc"] = True           # the stamp says it was found again
            if not r["ok"]: r["unsure"] = True                                     # placed, but ICP refused it there
        return r

    def marker_hint(self,T,quality):
        self._check()
        import bed
        count=quality.get('inliers',0);runner=quality.get('runner_up',count);rms=quality.get('rms_mm');span=quality.get('span_mm',0)
        valid=rms is not None and np.isfinite([rms,span]).all() and 0<=rms<=bed.HIT_RMS and count>=bed.HIT_MIN and count>=bed.HIT_RATIO*runner and span>0 and span>=3*rms
        if valid and hasattr(self.trk,'_marker_anchor'):
            mm=min(bed.HINT_GUARD_MM,max(.1,3*float(rms)))
            deg=min(bed.HINT_GUARD_DEG,float(np.degrees(np.arctan2(mm,span))))
            self.trk.hint(T,guard_mm=mm,guard_deg=deg,marker_anchor=True)
        else:self.trk.hint(T,guard_mm=bed.HINT_GUARD_MM,guard_deg=bed.HINT_GUARD_DEG)

    def hint(self, T, **kw): self._check(); self.trk.hint(T, **kw)

    def resume_hint(self,T):
        self._check()
        import bed
        options=dict(guard_mm=bed.HINT_GUARD_MM,guard_deg=bed.HINT_GUARD_DEG)
        if hasattr(self.trk,'_hint_local_only'):options['local_only']=True
        self.trk.hint(T,**options)

    def render(self, T, **kw): self._check(); return self.trk.render(T, **kw)

    def cloud(self): self._check(); return cloud_of(self.trk)

    def surface(self):
        """The model as a lit surface for the live view: vertices and normals in mm (float32) and triangles (int32), or
        None (a stand-in tracker, or nothing fused yet). About 15 ms for a 300 frame side on the GPU."""
        self._check(); vbg = getattr(self.trk, 'vbg', None)
        if vbg is None: return None
        try: m = vbg.extract_triangle_mesh(weight_threshold=2.0)        # the Build uses 3.0; 0.5 drew a sawtooth border of half-weighted voxels
        except RuntimeError: return None
        if not len(m.triangle.indices): return None
        m.compute_vertex_normals()
        return ((m.vertex.positions.cpu().numpy() * 1000.0).astype(np.float32), m.vertex.normals.cpu().numpy().astype(np.float32),
                m.triangle.indices.cpu().numpy().astype(np.int32))

    @property
    def fused(self): return 0 if self.trk is None else self.trk.fused

    @property
    def T(self): return self.trk.T

    def model_centre(self):
        """The model's middle (tracker metres), for turning the image-mode view about it."""
        self._check(); trk = self.trk
        if getattr(trk, "model", None) is None and hasattr(trk, "_model_cloud"): trk.model = trk._model_cloud()
        if getattr(trk, "model", None) is not None: return trk.model.point.positions.cpu().numpy().mean(0)
        return trk.T[:3, 3] + trk.T[:3, 2] * 0.25

    def raw(self):
        """The tracker itself, for helpers that read its model (Auto gain's well_covered). Worker thread only."""
        self._check(); return self.trk


class PlaybackTracker:
    """Stands in for track.Tracker in tests without a GPU: hands back the poses a real run saved, frame after frame, in
    the order the replay sends them (poses: list of 4x4 OpenCV camera -> scan, metres, or None for a lost frame). Same
    step() / seed() / render() / hint() surface; the model picture is the frame's depth, shaded, and the model cloud
    is the last frame's points (every 4th pixel) placed at its pose."""
    def __init__(self, poses, cal, w, h):
        self.poses = list(poses); self.k = 0; self.cal, self.W, self.H = cal, w, h
        self.T = np.eye(4); self.fused = 0; self.n = 0; self.model = None; self.levels = None; self.relocs = 0; self._last = None
        self.frames = []                                  # crc32 of every frame fused, in order: what the model holds

    def hint(self, T, **kw): pass

    def snapshot(self): return list(self.frames), self._last

    def restore(self, snap): self.frames, self._last = list(snap[0]), snap[1]

    def _fuse(self, raw):
        import zlib
        self.fused += 1; self._last = raw; self.frames.append(zlib.crc32(np.ascontiguousarray(raw).tobytes()))

    def step(self, raw):
        P = self.poses[self.k] if self.k < len(self.poses) else None; self.k += 1; self.n += 1
        ok = P is not None
        if ok: self.T = np.asarray(P, float).copy(); self._fuse(raw)
        fit, rmse = (float("nan"), float("nan")) if self.fused <= 1 else (0.9, 0.5)
        return dict(ok=ok, T=self.T.copy(), fit=fit, rmse=rmse, reloc=False, hinted=False, refused=False)

    def seed(self, raw, T):
        self.T = np.asarray(T, float).copy(); self.n += 1; self._fuse(raw)

    def render(self, T, scale=0.5, weight_threshold=None, coverage=False):
        w, h = int(self.W * scale), int(self.H * scale); img = np.full((h, w, 3), 24, np.uint8)
        if self._last is not None:
            d = self._last[::int(round(1 / scale)), ::int(round(1 / scale))][:h, :w].astype(np.float32); nz = d > 0
            if nz.any():
                lo, hi = np.percentile(d[nz], [2, 98]); v = np.clip(255 - (d - lo) / max(1, hi - lo) * 200, 40, 255)
                img[:d.shape[0], :d.shape[1]][nz] = (v[nz, None] * np.array([90, 170, 255]) / 255).astype(np.uint8)
        return (img, 0.0) if coverage else img

    def coverage_levels(self, P): return np.zeros(len(P), np.int8)

    def cloud_arrays(self):
        if self._last is None: return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), 0.0
        v, u = np.nonzero(self._last[::4, ::4]); v = v * 4; u = u * 4
        fx, fy, cx, cy = self.cal; z = self._last[v, u].astype(np.float64) * 0.1
        Pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
        T = self.T.copy(); T[:3, 3] *= 1000.0
        P = (Pc @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        N = np.tile((-T[:3, 2]).astype(np.float32), (len(P), 1))
        L = np.full(len(P), 2, np.uint8)
        return P, N, L, 1.0 if len(P) else 0.0
