# SPDX-License-Identifier: AGPL-3.0-or-later
"""Touch up: scan a built model again only where it still needs it. The new side is tracked in the built model's own
frame from its first frame: its tracker starts out holding the model (ModelTracker), so the first frames are found on
the model (TouchPlacer: ICP from where earlier sides started, and the global search, as a Continue is placed), and
every frame after that is matched against the model plus what the side has added. The side is saved with its poses in
the model's frame and a touchup.json (model_frames.py), so the build places it as it is, without any pair fitting.

What needs touching up comes from scanner/quality_map.py: holes (red) and rough, pitted or doubled patches (purple),
grouped into spots. The model panel shows the model dimmed gray, the spots, where the camera looks now (lighter gray),
and each spot turning green as new frames see it: a spot is done when most of it was seen COVER_GOOD times (holding
the scanner still on a spot averages more frames into it)."""
import os, time, copy, re, threading
import numpy as np
from track import Tracker, F
from app2.scan.tracking import GapPlacer, source_cloud

# the model panel's point levels (gl_core.PALETTE): 0 red hole, 4 dim gray fine, 5 light gray fine in view now,
# 6 purple rough, 7 green done
L_HOLE, L_FINE, L_VIEW, L_ROUGH, L_DONE = 0, 4, 5, 6, 7
COVER_GOOD = 12            # frames that saw a spot's point (Tracker.COVER_GOOD: "well covered")
DONE_SHARE = 0.7           # a spot is done when this share of its points is well covered
SEEN_MM = 2.5              # a frame sees a model point when its depth there is within this of the point
DISPLAY_MM = 1.0           # the model's points for tracking and showing: one per 1 mm cell
CACHE = "model_quality.npz"
MIN_PART_PX = 12000        # a frame with less of the part than this (pixels, platter left out) is not tracked
NEW_MM = 2.0               # new surface this close to the model's is left out of what the tracker matches against


def thin(V, mm):
    """Indices of one vertex per mm-cell (the first)."""
    k = np.floor(V / mm).astype(np.int64)
    return np.sort(np.unique(k, axis=0, return_index=True)[1])


class TouchModel:
    """The built model as the touch-up side sees it: thinned points (mm), normals, what each is (quality map), the
    spots and their coverage. Worker thread only."""
    def __init__(self, P, N, kind, spot_of, spots, path=None):
        self.P = np.ascontiguousarray(P, np.float32); self.N = np.ascontiguousarray(N, np.float32)
        self.kind = kind; self.spot_of = spot_of; self.spots = spots; self.path = path
        self.seen = np.zeros(len(P), np.int16); self.view = np.zeros(len(P), bool)
        self.members = [np.nonzero(spot_of == i)[0] for i in range(len(spots))]
        self.done = np.zeros(len(spots), bool)
        self._guide_lock = threading.Lock()
        self._guide_revision = 0
        self._guide_refresh_running = False
        self._guide_source = None
        self._live_data = None
        self.on_guide_changed = None
        self.scan_guide = None
        self.pivot = self.P[::max(1, len(self.P) // 50000)].mean(0).astype(np.float32) if len(self.P) else np.zeros(3, np.float32)

    @classmethod
    def load(cls, model_path, log=print):
        """Read model.ply and its quality map (cached next to it as model_quality.npz while model.ply is unchanged)."""
        import open3d as o3d
        from quality_map import quality_map, vertex_normals, HOLE, ROUGH
        m = o3d.io.read_triangle_mesh(model_path); V = np.asarray(m.vertices); T = np.asarray(m.triangles)
        if not len(V): raise ValueError("the built model is empty")
        cache = os.path.join(os.path.dirname(model_path), CACHE); st = os.stat(model_path)
        key = np.array([st.st_size, st.st_mtime_ns, len(V)], np.int64)
        qm = None
        try:
            z = np.load(cache, allow_pickle=False)
            if np.array_equal(z["key"], key): qm = dict(kind=z["kind"], lab=z["lab"], spots=z["spots"])
        except Exception: pass
        if qm is None:
            t0 = time.time(); q = quality_map(V, T)
            lab = np.full(len(V), -1, np.int32); rows = []
            for i, s in enumerate(q["spots"]):
                lab[s["idx"]] = i
                rows.append([1.0 if s["kind"] == "hole" else 2.0, *s["centre"], *s["normal"], s["area_mm2"], s["radius_mm"]])
            qm = dict(kind=q["kind"], lab=lab, spots=np.array(rows, float).reshape(-1, 9))
            try: np.savez(cache + ".tmp.npz", key=key, **qm); os.replace(cache + ".tmp.npz", cache)
            except OSError as e: print("quality cache:", e, flush=True)
            log("quality map: %d spots (%d holes, %d rough), %.1f%% of the surface rough (%.0f s)" % (
                len(rows), sum(r[0] == 1 for r in rows), sum(r[0] == 2 for r in rows), 100 * q["rough_share"], time.time() - t0))
        N = vertex_normals(V, T); keep = thin(V, DISPLAY_MM)
        # every spot keeps all its vertices' cells in the thinned set: a spot with no points could never turn green
        keep = np.union1d(keep, np.nonzero(qm["lab"] >= 0)[0][thin(V[qm["lab"] >= 0], DISPLAY_MM)]) if (qm["lab"] >= 0).any() else keep
        spots = [dict(kind="hole" if r[0] == 1 else "rough", centre=r[1:4], normal=r[4:7], area_mm2=r[7], radius_mm=r[8])
                 for r in qm["spots"]]
        model=cls(V[keep], N[keep], qm['kind'][keep], qm['lab'][keep], spots, model_path)
        from app2.scan.capture_distance import project_levels
        model.capture_distance=project_levels(os.path.dirname(model_path),model.P)
        # the built surface itself, for the 3D view's Surface style (the dots are the thinned points): every vertex
        # takes the color of its nearest shown point (v_index), so the surface recolors as the dots do
        try:
            from scipy.spatial import cKDTree
            model.mesh = (np.ascontiguousarray(V, np.float32), np.ascontiguousarray(N, np.float32), np.ascontiguousarray(T, np.int32))
            model.v_index = cKDTree(model.P).query(V, workers=-1)[1].astype(np.int64)
        except Exception as e:
            print("touch surface:", e, flush=True); model.mesh = None; model.v_index = None
        return model

    # ---- coverage
    def update(self, d, T_m, cal, depth_scale=0.1):
        """A tracked frame: which model points it saw (depth there within SEEN_MM, facing the camera). d: HxW uint16 depth
        (0.1 mm) as tracked, T_m: its pose (OpenCV camera -> model, metres). Returns the spots that just got done."""
        fx, fy, cx, cy = cal; H, W = d.shape
        R = T_m[:3, :3]; t = T_m[:3, 3] * 1000.0
        Pc = (self.P - t) @ R; z = Pc[:, 2]                               # model mm -> camera mm
        ok = z > 20.0
        u = np.full(len(z), -1, np.int64); v = u.copy()
        u[ok] = np.rint(Pc[ok, 0] * fx / z[ok] + cx); v[ok] = np.rint(Pc[ok, 1] * fy / z[ok] + cy)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        i = np.nonzero(ok)[0]
        facing = (self.N[i] @ R * -Pc[i]).sum(1) > 0.3 * np.linalg.norm(Pc[i], axis=1)
        i = i[facing]
        dz = d[v[i], u[i]].astype(np.float32) * depth_scale
        i = i[(dz > 0) & (np.abs(dz - z[i]) <= SEEN_MM)]
        self.view[:] = False; self.view[i] = True
        self.seen[i] = np.minimum(self.seen[i] + 1, 30000)
        newly = []
        for k, mb in enumerate(self.members):
            if self.done[k] or not len(mb): continue
            if (self.seen[mb] >= COVER_GOOD).mean() >= DONE_SHARE: self.done[k] = True; newly.append(k)
        if newly: self._request_guide_refresh()
        return newly

    def levels(self):
        """Per point level for the model panel (see L_*)."""
        L = np.full(len(self.P), L_FINE, np.uint8); L[self.view] = L_VIEW
        L[self.kind == 2] = L_ROUGH; L[self.kind == 1] = L_HOLE
        spot = self.spot_of >= 0
        L[spot & (self.seen >= COVER_GOOD)] = L_DONE
        if self.done.any(): L[spot & self.done[np.maximum(self.spot_of, 0)]] = L_DONE
        return L

    @property
    def scan_guide(self):
        return self._scan_guide

    @scan_guide.setter
    def scan_guide(self, value):
        live = value.get('live_samples') if value else None
        data = None
        if live and self.spots:
            from scipy.spatial import cKDTree
            centers = np.asarray([s['centre'] for s in self.spots], float)
            source_centers = np.asarray(live['spot_centers'], float).reshape(-1, 3)
            distance, mapping = cKDTree(centers).query(source_centers)
            tolerance = np.finfo(np.float32).eps * max(1., float(np.linalg.norm(centers, axis=1).max()))
            mapping = np.asarray(mapping, np.int64)
            matches = distance <= tolerance
            for i, k in enumerate(mapping):
                matches[i] &= self.spots[k]['kind'] == live['spot_kinds'][i]
            mapping[~matches] = -1
            groups = np.asarray(live['spot_ids'], np.int64)
            associated = np.full(len(groups), -1, np.int64)
            known = (groups >= 0) & (groups < len(mapping))
            associated[known] = mapping[groups[known]]
            ids = np.asarray(live['vertex_ids'], np.int64)
            lookup = {int(v): i for i, v in enumerate(ids)}
            rows = [(dict(p), np.asarray([lookup[i] for i in p.get('indices', []) if i in lookup], np.int64))
                    for p in value.get('poses', [])]
            data = (associated, ids, np.asarray(live['areas'], float), rows)
        with self._guide_lock:
            self._guide_revision += 1
            self._guide_source = copy.deepcopy(value)
            self._live_data = data
            self._scan_guide = value
        if data is not None and self.done.any(): self._request_guide_refresh()

    def _rank_live_guidance(self, done):
        source = self._guide_source
        if self._live_data is None or source is None: return source
        groups, ids, weights, rows = self._live_data
        covered = np.zeros(len(groups), bool)
        known = groups >= 0
        covered[known] = done[groups[known]]
        total = float(weights[~covered].sum())
        ranked = []
        for order, (pose, samples) in enumerate(rows):
            remaining = samples[~covered[samples]]
            area = float(weights[remaining].sum())
            if area <= 0: continue
            p = dict(pose)
            p.update(indices=ids[remaining].tolist(), area_mm2=area, share=area / total if total else 0.)
            p['text'] = re.sub(r'fills about \d+ percent', 'fills about %d percent' % round(100*p['share']), p.get('text', ''))
            ranked.append((area, order, p))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        from scanner.scan_guide import RANK_COLORS
        poses = []
        for rank, (_, _, pose) in enumerate(ranked, 1):
            pose.update(rank=rank, color=RANK_COLORS[(rank-1) % len(RANK_COLORS)])
            poses.append(pose)
        result = dict(source, poses=poses, covered_spots=int(done.sum()), surface_mm2=total,
                      spots=max(0, int(source.get('spots', len(self.spots))) - int(done.sum())))
        lookup = {int(v): i for i, v in enumerate(ids)}
        for key in ('remaining', 'needs_flip', 'unreachable'):
            result[key] = [v for v in source.get(key, []) if v not in lookup or not covered[lookup[v]]]
        live = source['live_samples']
        if 'counts' in live and 'kinds' in live:
            counts = np.asarray(live['counts']); kinds = np.asarray(live['kinds'])
            result['missing_vertices'] = int(counts[(kinds == 1) & ~covered].sum())
            result['rough_vertices'] = int(counts[(kinds == 2) & ~covered].sum())
        return result

    def _request_guide_refresh(self):
        if self._live_data is None: return
        with self._guide_lock:
            self._guide_revision += 1
            if self._guide_refresh_running: return
            self._guide_refresh_running = True
        def refresh():
            try:
                while True:
                    with self._guide_lock:
                        revision = self._guide_revision
                        done = self.done.copy()
                    ranked = self._rank_live_guidance(done)
                    with self._guide_lock:
                        if revision != self._guide_revision: continue
                        self._scan_guide = dict(ranked, live_revision=revision) if ranked else ranked
                        self._guide_refresh_running = False
                    callback = self.on_guide_changed
                    if callback is not None: callback()
                    return
            except Exception as error:
                with self._guide_lock: self._guide_refresh_running = False
                print('Touch up suggestions:', error, flush=True)
        threading.Thread(target=refresh, name='touch-guide-refresh', daemon=True).start()

    def remaining_counts(self):
        holes=sum(1 for i,s in enumerate(self.spots) if not self.done[i] and s["kind"]=="hole")
        rough=sum(1 for i,s in enumerate(self.spots) if not self.done[i] and s["kind"]=="rough")
        return dict(holes=holes,rough=rough,left=holes+rough)

    def counts(self): return int(self.done.sum()),len(self.spots)


class ModelTracker(Tracker):
    """A Tracker that starts out holding the built model (base: points in metres with normals), so it tracks in the
    model's frame from the first frame. Frames it takes are fused as usual; the cloud it matches against is the model
    plus what the new frames added (holes filled). fused starts at 1: the model counts as the first frame, so the first
    real frame is matched, never taken as it comes. cands: camera poses (OpenCV camera -> model, metres) where earlier
    sides started, tried first when looking for the part."""
    def __init__(self, cal, W, H, dev, P_m, N, cands=(), platters=(), **kw):
        import open3d as o3d, open3d.core as o3c
        kw.pop("floater_looks", None)
        super().__init__(cal, W, H, dev, T0=(cands[0] if len(cands) else None), floater_looks=0, **kw)
        self.base = o3d.t.geometry.PointCloud(dev)
        self.base.point.positions = o3c.Tensor(np.ascontiguousarray(P_m, np.float32)).to(dev)
        self.base.point.normals = o3c.Tensor(np.ascontiguousarray(N, np.float32)).to(dev)
        self.cands = [np.asarray(c, float) for c in cands]; self.platters = list(platters); self._platter_last = None
        from scipy.spatial import cKDTree
        self._base_tree = cKDTree(np.asarray(P_m, np.float64))
        self.fused = 1
        self.relock = True             # TrackerHost: after 2 lost frames, find the part again with the TouchPlacer

    def _model_cloud(self):
        self.levels = None
        if self.fused <= 1: return self.base
        try: pc = self.vbg.extract_point_cloud(weight_threshold=3.0)
        except RuntimeError: return self.base
        if len(pc.point.positions) == 0: return self.base
        pc = pc.voxel_down_sample(0.001); pc.estimate_normals(max_nn=20, radius=0.003)
        # only what the model does not have (the platter, surface in its holes): new surface right on the model's would
        # be matched first, and the side would then follow its own frames instead of the model (seen: 1 to 2 mm off
        # on replays; the model alone kept 0.2 mm but lost every frame with much platter in it)
        Q = pc.point.positions.cpu().numpy()
        far = self._base_tree.query(Q, distance_upper_bound=NEW_MM / 1000.0)[0] > NEW_MM / 1000.0
        if not far.any(): return self.base
        pc = pc.select_by_mask(self.o3c.Tensor(far).to(pc.device))
        o3d = self.o3d
        both = o3d.t.geometry.PointCloud(self.dev)
        both.point.positions = self.o3c.concatenate([self.base.point.positions, pc.point.positions.to(self.o3c.float32)])
        both.point.normals = self.o3c.concatenate([self.base.point.normals, pc.point.normals.to(self.o3c.float32)])
        return both

    def strip(self, raw):
        """The frame without the platter (find_platter / strip_platter): the model has no platter, and a platter
        fused at a first pose a little off held every later pose that far off (3-4 deg on a replay), outvoting the part."""
        last = [self._platter_last] if self._platter_last is not None else []
        pl = find_platter(raw, self.cal, last + self.platters)
        if pl is not None: self._platter_last = pl
        return strip_platter(raw, self.cal, pl)

    def step(self, raw):
        """As Tracker.step, on the frame without its platter. A frame with too little of the part left (MIN_PART_PX
        pixels) is not matched at all: it can't pin a pose down (such frames landed 3-4 mm off in replays), so it is
        lost without counting towards a relock, and the next frame starts from the last good pose."""
        z = self.strip(raw)
        if np.count_nonzero(z) < MIN_PART_PX:
            self.n += 1
            return dict(ok=False, T=self.last_good.copy(), fit=float("nan"), rmse=float("nan"), reloc=False, hinted=False,
                        refused=False, refused_T=None, few=True)
        self._in_step = True
        try: return super().step(z)
        finally: self._in_step = False

    def _relocalize(self, src, model):
        """The global search, for the placer only: Tracker.step's own quick one (taken on one answer) never runs here."""
        if getattr(self, "_in_step", False): return None
        return super()._relocalize(src, model)

    def seed(self, raw, T):
        """Reloading a saved touch-up side: its frames at their poses on top of the model."""
        super().seed(self.strip(raw), T)


class TouchPlacer(GapPlacer):
    """GapPlacer for a tracker that holds a model, before its first frame is taken. Two changes: the platter is left
    out of the frame while looking (the model has no platter: a third of a frame on it held every fit under the bar),
    and ICP starts from the best of every place an earlier side started and the last pose, not the last pose alone.
    (ModelTracker.strip leaves the platter out of every frame it tracks, too.)
    Taking a frame still needs the usual two answers (ICP and the global search agree, or two searches in a row do)."""
    def _source(self, trk, raw):
        return source_cloud(trk, trk.strip(raw) if hasattr(trk, "strip") else raw)

    def _icp_answer(self, trk, src):
        import open3d as o3d
        best = None
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):   # far starts: "0 correspondences"
            for C in list(getattr(trk, "cands", ()) or ()) + [trk.last_good]:
                try: T, fit, rmse = trk._icp(src, trk.model, C)
                except RuntimeError: continue
                if best is None or (fit, -rmse) > (best[1], -best[2]): best = (np.array(T), fit, rmse)
        return best


def _refine_plane(P, n, off, bands=(8.0, 4.0, 2.0), min_pts=150):
    """Least-squares plane through the points of P (mm) within each band of the plane in turn (n . x + off = 0).
    Returns (n, off, points in the last band) or None."""
    n = np.asarray(n, float); off = float(off); k = 0
    for b in bands:
        Q = P[np.abs(P @ n + off) < b]; k = len(Q)
        if k < min_pts: return None
        c = Q.mean(0); m = np.linalg.svd(Q - c, full_matrices=False)[2][2]
        if m @ n < 0: m = -m
        n, off = m, -float(m @ c)
    return n, off, k


def find_platter(d, cal, priors=(), min_share=0.15, depth_scale=0.1, stride=4):
    """The platter in a depth frame (camera frame, mm): (n towards the camera, off) or None. Found again near one of
    the priors (the platters earlier sides saw, and the one found in the last frame: in the camera's view the platter
    stays put while the table turns, and the scanner rarely moves far between sides), by least squares in shrinking
    bands (deterministic: RANSAC found it on one call and missed it on the next); else the frame's biggest plane when
    that holds min_share of the points."""
    fx, fy, cx, cy = cal
    v, u = np.nonzero(d[::stride, ::stride]); z = d[::stride, ::stride][v, u].astype(np.float64) * depth_scale
    if len(z) < 300: return None
    u = u * stride; v = v * stride
    P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
    best = None
    for n0, off0 in priors or ():
        f = _refine_plane(P, n0, off0)
        if f is None or abs(f[0] @ np.asarray(n0, float)) < 0.95: continue
        if best is None or f[2] > best[2]: best = f
    if best is None:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            plane, inl = pc.segment_plane(1.5, 3, 300)
        if len(inl) < min_share * len(P): return None
        n = np.asarray(plane[:3], float); s_ = np.linalg.norm(n); best = (n / s_, float(plane[3]) / s_, len(inl))
    n, off = best[0], best[1]
    return (n, off) if off > 0 else (-n, -off)            # the camera (origin) on the positive side


def strip_platter(d, cal, plane, keep_mm=2.0, depth_scale=0.1):
    """d with every pixel up to keep_mm above the platter plane (and below it) zeroed: the dots on the platter and
    its rim (a dark platter shows little else). A copy; d itself when there is no plane."""
    if plane is None: return d
    fx, fy, cx, cy = cal; n, off = plane; H, W = d.shape
    rx = (np.arange(W, dtype=np.float32) - cx) / fx; ry = (np.arange(H, dtype=np.float32) - cy) / fy
    z = d.astype(np.float32) * depth_scale
    h = z * (n[0] * rx[None, :] + n[1] * ry[:, None] + n[2]) + off
    out = d.copy(); out[(d > 0) & (h <= keep_mm)] = 0
    return out


def platter_priors(project, frames=None):
    """The platter each saved side cut (table_plane.json) in the camera frame at that side's first tracked frame:
    [(n, off mm)], for finding the platter again in a touch-up side's first frames."""
    import json, glob
    from fuse import read_global_poses
    from table_plane import to_camera
    out = []
    for s in sorted(glob.glob(os.path.join(project, "data", "*"))):
        if frames is not None and os.path.basename(s) not in frames: continue
        try:
            tp = json.load(open(os.path.join(s, "table_plane.json")))
            gp = read_global_poses(os.path.join(s, "cache", "pointyoink_register_pose.pose"))
            nc, oc = to_camera(np.array(tp["n"], float), -float(tp["d"]), gp[sorted(gp)[0]] @ F)   # the json's d is -off
            out.append((np.asarray(nc, float), float(oc)))
        except Exception: continue
    return out


ModelTracker.make_placer = TouchPlacer


def start_poses(project, frames):
    """Where earlier sides started, in the model's frame (tracker metres): each placed side's first tracked pose."""
    from fuse import read_global_poses
    out = []
    for folder, M in sorted(frames.items()):
        pp = os.path.join(project, "data", folder, "cache", "pointyoink_register_pose.pose")
        if not os.path.exists(pp): continue
        try: gp = read_global_poses(pp)
        except Exception: continue
        if not gp: continue
        first = gp[sorted(gp)[0]]
        T = np.asarray(M, float) @ first @ F; T = T.copy(); T[:3, 3] /= 1000.0
        out.append(T)
    return out


def model_tracker_factory(model, cands, platters=(), device=None):
    """A factory(cal, w, h) for TrackerHost that makes a ModelTracker for this model (CUDA like the live tracker)."""
    def make(cal, w, h):
        import open3d.core as o3c
        from app2.scan.tracking import cuda_device
        dev = o3c.Device(device) if device else cuda_device()
        return ModelTracker(cal, w, h, dev, model.P / 1000.0, model.N, cands=cands, platters=platters, max_depth_mm=1000.0)
    return make
