#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Re-track a raw MIRACO scan from its depth frames alone, instead of trusting the scanner's live tracking.
# When the scanner loses tracking (dark parts, fast moves) its per-frame .inf poses slip and every later frame
# lands in the wrong place: doubled walls and striped surfaces that no fragment clean-up can undo. Here each
# frame is matched against the model fused so far (multi-scale point-to-plane ICP with a robust Tukey kernel,
# constant-velocity prediction), frames that do not fit are left out, and after a jump (Resume Scan) the frame
# is found again on the model by feature matching. Replayed against the scanner's own poses this lands within
# about 0.2 mm on scans it tracked well, and stays clean on scans where the scanner lost tracking.
#   python3 track.py --frames <cache dir> --calib <param/Pl.bin> [--gpu]
# Output: <cache dir>/pointyoink_register_pose.pose (the scanner's pose-table layout; fuse.py picks it up and skips
# frames missing from it) and <cache dir>/pointyoink_track.json (per-run report). The scanner's own
# global_register_pose.pose is never touched.
import sys, os, argparse, json, glob, time, resource
MEM_CAP_GB = float(os.environ.get("POINTYOINK_MEM_CAP_GB", "12"))
def apply_mem_cap():
    try:
        cap = int(MEM_CAP_GB * 1024**3); resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception: pass
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse import read_calib, read_pose, frame_key
from register import frame_size, write_pose_table

F = np.diag([1.0, -1.0, -1.0, 1.0])       # OpenCV camera <-> scanner camera frame (its own inverse)

def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)

def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def clean_frame(d, cal, grid, jump_mm=3.0, grazing=80.0, depth_scale=0.1, cosang_out=None):
    """Same per-frame filter as fuse.py: drop depth jumps (edges, flying pixels) and surfaces seen edge-on.
    cosang_out: a list to receive |cos(view angle)| per pixel, for a second, stricter cut without recomputing it."""
    fx, fy, cx, cy = cal
    z = d.astype(np.float32) * depth_scale; valid = z > 0
    jump = np.zeros_like(valid)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        zs = np.roll(z, (dy, dx), axis=(0, 1)); vs = np.roll(valid, (dy, dx), axis=(0, 1))
        jump |= vs & (np.abs(zs - z) > jump_mm)
    edge = jump | ~valid
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, -1, 0) | np.roll(edge, 1, 1) | np.roll(edge, -1, 1)
    u, v = grid; X = (u - cx) * z / fx; Y = (v - cy) * z / fy
    dXu = np.roll(X, -1, 1) - np.roll(X, 1, 1); dYu = np.roll(Y, -1, 1) - np.roll(Y, 1, 1); dZu = np.roll(z, -1, 1) - np.roll(z, 1, 1)
    dXv = np.roll(X, -1, 0) - np.roll(X, 1, 0); dYv = np.roll(Y, -1, 0) - np.roll(Y, 1, 0); dZv = np.roll(z, -1, 0) - np.roll(z, 1, 0)
    nx = dYu * dZv - dZu * dYv; ny = dZu * dXv - dXu * dZv; nz = dXu * dYv - dYu * dXv
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    rl = np.sqrt(X * X + Y * Y + z * z) + 1e-9
    cosang = np.abs((nx * X + ny * Y + nz * z) / (nl * rl))
    d = d.copy(); d[edge | (valid & (cosang < np.cos(np.radians(grazing))))] = 0
    if cosang_out is not None: cosang_out.append(cosang)
    return d

REGISTRATION_MIN_POINTS=500  # Existing minimum, unchanged.

def registration_stride(depth,stride,dense_sparse=False):
    """Retain measured samples when an independent marker pose can place a sparse frame."""
    if dense_sparse and stride>1 and np.count_nonzero(depth[::stride,::stride])<REGISTRATION_MIN_POINTS and np.count_nonzero(depth)>=REGISTRATION_MIN_POINTS:return 1
    return stride

class Tracker:
    """Frame-to-model tracking, one depth frame at a time; shared by the batch re-track below and the live scanner.
    step() returns whether the frame fitted, and fuses it into the model when it did. Poses are OpenCV camera->world, metres."""
    def __init__(self, cal, W, H, dev, T0=None, voxel_mm=0.5, depth_scale=0.1, max_depth_mm=600.0, refresh=3,
                 min_fit=0.5, max_rmse=1.5, reloc_every=1, floater_looks=0):
        import open3d as o3d, open3d.core as o3c
        import open3d.t.pipelines.registration as treg
        from open3d.t.pipelines.registration.robust_kernel import RobustKernel, RobustKernelMethod
        self.o3d, self.o3c, self.treg = o3d, o3c, treg
        self.cal, self.W, self.H, self.dev = cal, W, H, dev
        fx, fy, cx, cy = cal
        self.grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
        self.depth_scale = depth_scale; self.DS = 1000.0 / depth_scale; self.DMAX = max_depth_mm / 1000.0
        self.refresh, self.min_fit, self.max_rmse = refresh, min_fit, max_rmse
        self.reloc_every = max(1, reloc_every)            # live: try the (slow) global search only every Nth lost frame
        self.vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3c.float32, o3c.float32), ((1), (1)),
                                                 voxel_mm / 1000.0, 16, 25000, dev)
        self.scales = [(0.004, 0.020, 15), (0.002, 0.008, 15), (0.001, 0.003, 25)]   # coarse -> fine: voxel m, max match m, iterations
        self.est = treg.TransformationEstimationPointToPlane(RobustKernel(RobustKernelMethod.TukeyLoss, 0.002))
        self.stride = 1 if W <= 400 else 2
        self.T = np.eye(4) if T0 is None else np.asarray(T0, float).copy()
        self.vel = np.eye(4); self.last_good = self.T.copy(); self.streak = 0
        self.pose_constraint = None; self.registration_plane = None
        self.model = None; self.n = 0; self.fused = 0; self.relocs = 0; self._hint = None; self._guard = None; self._marker_anchor = False
        self._hint_local_only=False
        # floaters: a barely seen spot that was in plain view floater_looks more times (counted when the model cloud is
        # rebuilt) and never seen there again is left out of the model cloud, so it is neither matched against nor
        # shown. 0 = off. The voxels stay: should the spot be confirmed after all, it comes back by itself.
        self.floater_looks = floater_looks; self.levels = None; self.dropped = 0; self._fresh = False
        self._miss_k = np.zeros(0, np.int64); self._miss_n = np.zeros(0, np.int16); self._dead = np.zeros(0, np.int64)

    max_weight = None            # cap on a voxel's weight (frames seen). None: uncapped, the surface freezes where it was seen most
    ORBIT_FIT_SLACK, ORBIT_RMSE_SLACK_MM = 0.0, 0.0   # how much worse the orbit-projected fit may be and still be taken (0: no worse)

    def _integrate(self, z, T):
        o3d, o3c = self.o3d, self.o3c
        img = o3d.t.geometry.Image(o3c.Tensor(z)).to(self.dev); ext = o3c.Tensor(np.linalg.inv(T))
        bc = self.vbg.compute_unique_block_coordinates(img, self.K, ext, self.DS, self.DMAX, 3.0)
        self.vbg.integrate(bc, img, self.K, ext, self.DS, self.DMAX, 3.0)
        if self.max_weight and self.fused % self.refresh == 0: self._cap_weights()

    def _cap_weights(self):
        """Clip the weight of the active blocks to max_weight, so a revisit can still move the skin. Only the active
        blocks (a copy in, a copy back): a mask over the whole grid buffer ran the card out of memory."""
        o3c = self.o3c; h = self.vbg.hashmap(); idx = h.active_buf_indices().to(o3c.int64)
        if len(idx) == 0: return
        w = self.vbg.attribute("weight"); w2 = w.reshape((w.shape[0], -1))        # 2-D view: Open3D's indexer takes at most 5 operands
        for s in range(0, len(idx), 1024):                                          # in slices: one copy of every block at once ran the card dry
            ii = idx[s:s + 1024]; sub = w2[ii].contiguous(); sub.clip_(0.0, float(self.max_weight)); w2[ii] = sub

    def _model_cloud(self):
        pc = self.vbg.extract_point_cloud(weight_threshold=0.5).voxel_down_sample(0.001)
        pc.estimate_normals(max_nn=20, radius=0.003)
        self.levels = None
        if self.floater_looks:
            P = pc.point.positions.cpu().numpy(); lv = self.coverage_levels(P)
            if len(self._dead):
                gone = (lv == 0) & np.isin(self._cells(P), self._dead)
                if gone.any():
                    pc = pc.select_by_mask(self.o3c.Tensor(~gone).to(pc.device)); lv = lv[~gone]
                self.dropped = int(gone.sum())
            self.levels = lv; self._fresh = True
        return pc

    @staticmethod
    def _cells(Q, mm=1.0):
        """One number per point: the cell (mm wide) it lies in. Q in metres."""
        Q = np.floor(Q * (1000.0 / mm)).astype(np.int64); return (Q[:, 0] * 73856093) ^ (Q[:, 1] * 19349663) ^ (Q[:, 2] * 83492791)

    def _look(self, z, T):
        """Count, for every barely seen point of the model in plain view of this frame (in the picture, facing the
        camera, nothing in front of it), whether the frame saw it again. Seen again: its count starts over (and its
        weight grows by itself). Not there (no depth, or depth from further back): one more miss; at floater_looks
        misses the spot is dead. z: the cleaned depth frame, T: its pose (OpenCV camera -> world, metres)."""
        P = self.model.point.positions.cpu().numpy(); red = self.levels == 0
        if len(P) != len(red) or not red.any(): return
        P = P[red]; N = self.model.point.normals.cpu().numpy()[red]; keys = self._cells(P)
        Ti = np.linalg.inv(T); Pc = P @ Ti[:3, :3].T + Ti[:3, 3]; Nc = N @ Ti[:3, :3].T
        fx, fy, cx, cy = self.cal; zc = np.maximum(Pc[:, 2], 1e-6)
        u = np.rint(Pc[:, 0] * fx / zc + cx).astype(np.int64); v = np.rint(Pc[:, 1] * fy / zc + cy).astype(np.int64)
        facing = np.abs((Nc * Pc).sum(1)) > 0.5 * np.linalg.norm(Pc, axis=1)          # within 60 deg of head-on
        i = np.nonzero((Pc[:, 2] > 0.02) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & facing)[0]
        dz = z[v[i], u[i]].astype(np.float64) * self.depth_scale / 1000.0
        seen = (dz > 0) & (np.abs(dz - zc[i]) <= 0.003)
        missed = (dz == 0) | (dz > zc[i] + 0.003)
        now = np.unique(keys); k = np.isin(self._miss_k, now) & ~np.isin(self._miss_k, keys[i][seen])
        old_k, old_n = self._miss_k[k], self._miss_n[k]
        mk = np.unique(keys[i][missed]); allk = np.union1d(old_k, mk); n = np.zeros(len(allk), np.int16)
        n[np.searchsorted(allk, old_k)] = old_n; n[np.searchsorted(allk, mk)] += 1
        dead = n >= self.floater_looks
        if dead.any(): self._dead = np.union1d(self._dead, allk[dead])
        self._miss_k, self._miss_n = allk[~dead], n[~dead]

    def _icp(self, src, model, init):
        o3d, o3c, treg = self.o3d, self.o3c, self.treg
        r = treg.multi_scale_icp(src, model, o3d.utility.DoubleVector([s[0] for s in self.scales]),
                                 [treg.ICPConvergenceCriteria(1e-6, 1e-6, s[2]) for s in self.scales],
                                 o3d.utility.DoubleVector([s[1] for s in self.scales]), o3c.Tensor(init), self.est)
        return r.transformation.cpu().numpy(), float(r.fitness), float(r.inlier_rmse) * 1000.0

    def _relocalize(self, src, model):
        """Find the frame on the model from scratch (after Resume Scan or a lost stretch): FPFH + RANSAC, then fine ICP."""
        o3d = self.o3d; L = o3d.pipelines.registration; vox = 0.003
        def down(pc_t):
            pc = pc_t.to_legacy().voxel_down_sample(vox)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 3, max_nn=30))
            return pc, L.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))
        (s, fs), (m, fm) = down(src), down(model)
        best = None
        for _ in range(3):
            g = L.registration_ransac_based_on_feature_matching(
                s, m, fs, fm, True, vox * 1.5, L.TransformationEstimationPointToPoint(False), 3,
                [L.CorrespondenceCheckerBasedOnEdgeLength(0.9), L.CorrespondenceCheckerBasedOnDistance(vox * 1.5)],
                L.RANSACConvergenceCriteria(200000, 0.999))
            try: r = self._icp(src, model, g.transformation)
            except RuntimeError: continue
            if best is None or r[1] > best[1]: best = r
        return best

    def hint(self, T, guard_mm=None, guard_deg=None, marker_anchor=False, local_only=False):
        """Start the next step() from pose T (OpenCV camera -> world, metres), found some other way (the table's dots).
        ICP must still fit, so a wrong hint costs one try and nothing else. With guard_mm (and guard_deg) that frame is
        only taken within that distance of T: a relocalisation that lands elsewhere (a near-symmetric part matched the
        wrong way round, seen at 58-178 deg off in replays) is refused and the frame stays lost."""
        self._hint = np.asarray(T, float).copy(); self._marker_anchor=bool(marker_anchor)
        self._hint_local_only=bool(local_only)
        self._guard = None if guard_mm is None else (self._hint, float(guard_mm), float(guard_deg or 180.0))

    def step(self, raw):
        """raw: HxW uint16 depth. Returns dict(ok, T, fit, rmse, reloc, hinted, refused, refused_T: the refused fit's pose)."""
        o3d, o3c = self.o3d, self.o3c
        z = raw.copy(); z[z * self.depth_scale > self.DMAX * 1000.0] = 0
        ca = []; z = clean_frame(z, self.cal, self.grid, jump_mm=getattr(self,'depth_jump_mm',3.0), depth_scale=self.depth_scale, cosang_out=ca); self._cosang = ca[0] if ca else None
        registration_points=0;used_stride=self.stride
        ok = True; fit = rmse = float("nan"); reloc = False; registration_cut = False
        hint, self._hint = self._hint, None; hinted = False
        local_only,self._hint_local_only=self._hint_local_only,False
        marker_anchor, self._marker_anchor = self._marker_anchor, False; marker_anchored=False
        guard, self._guard = self._guard, None; refused = False; refused_T = None; constrained = False
        if self.fused:
            if self.model is None or self.n % self.refresh == 0 or self.streak: self.model = self._model_cloud()
            registration_z = z
            if self.registration_plane is not None:
                normal, offset, margin = self.registration_plane
                pose = hint if hint is not None else self.last_good
                nc = pose[:3,:3].T@normal; oc = offset + normal@pose[:3,3]
                u,v=self.grid;zm=z.astype(float)*self.depth_scale/1000
                above=nc[0]*(u-self.cal[2])*zm/self.cal[0]+nc[1]*(v-self.cal[3])*zm/self.cal[1]+nc[2]*zm+oc>margin
                candidate=z.copy();candidate[~above]=0
                if np.count_nonzero(candidate)>=REGISTRATION_MIN_POINTS:registration_z=candidate;registration_cut=True
            used_stride=registration_stride(registration_z,self.stride,getattr(self,"dense_sparse_registration",False) and marker_anchor)
            src = o3d.t.geometry.PointCloud.create_from_depth_image(
                o3d.t.geometry.Image(o3c.Tensor(registration_z)).to(self.dev), self.K, o3c.Tensor(np.eye(4)), self.DS, self.DMAX, stride=used_stride)
            hinted = hint is not None
            pred = hint if hinted else (self.last_good if self.streak else self.T @ self.vel)
            ok = False
            registration_points=len(src.point.positions)
            if registration_points >= REGISTRATION_MIN_POINTS:
                if hinted and marker_anchor:
                    checked=self.treg.evaluate_registration(src,self.model,self.scales[-1][1],o3c.Tensor(hint))
                    fit=float(checked.fitness);rmse=float(checked.inlier_rmse)*1000
                    if fit>self.min_fit and rmse<self.max_rmse:new=hint;ok=True;marker_anchored=True
                if not ok:
                    try:
                        new, fit, rmse = self._icp(src, self.model, pred)
                        ok = fit > self.min_fit and rmse < self.max_rmse
                    except RuntimeError: pass
                if not ok and not local_only and self.streak >= 1 and (self.streak - 1) % self.reloc_every == 0:
                    r = self._relocalize(src, self.model)
                    if r is not None and r[1] > self.min_fit and r[2] < self.max_rmse * 0.8:
                        new, fit, rmse = r; ok = True; reloc = True; self.relocs += 1
            if ok and guard is not None:
                D = np.linalg.inv(guard[0]) @ new
                if np.linalg.norm(D[:3, 3]) * 1000.0 > guard[1] or rot_deg(D[:3, :3]) > guard[2]:
                    ok = False; refused = True; refused_T = new.copy()     # where the refused fit put the camera
                    if reloc: self.relocs -= 1; reloc = False
            if ok and self.pose_constraint is not None and not marker_anchored:
                # Learn from the unprojected fit only; never certify our own projection.
                self.pose_constraint.accepted(new)
                projected = self.pose_constraint.project(new)
                if projected is not None:
                    checked = self.treg.evaluate_registration(src, self.model, self.scales[-1][1], o3c.Tensor(projected))
                    projected_fit = float(checked.fitness); projected_rmse = float(checked.inlier_rmse)*1000
                    if projected_fit >= fit - self.ORBIT_FIT_SLACK and projected_rmse <= rmse + self.ORBIT_RMSE_SLACK_MM and projected_fit > self.min_fit and projected_rmse < self.max_rmse:
                        new, fit, rmse = projected, projected_fit, projected_rmse
                        constrained = True; self.pose_constraint.used += 1
            if ok and guard is not None:
                D = np.linalg.inv(guard[0]) @ new
                if np.linalg.norm(D[:3, 3]) * 1000.0 > guard[1] or rot_deg(D[:3, :3]) > guard[2]:
                    ok = False; refused = True; refused_T = new.copy()     # where the refused fit put the camera
                    if reloc: self.relocs -= 1; reloc = False
            if ok:
                self.vel = np.eye(4) if (self.streak or hinted) else np.linalg.inv(self.T) @ new
                self.T = new; self.last_good = new; self.streak = 0
                if self.floater_looks and self._fresh and self.levels is not None:
                    self._fresh = False; self._look(z, new)
            else:
                # lost: never extrapolate and never fuse a frame that does not fit; wait at the last good pose
                self.streak += 1; self.vel = np.eye(4); self.T = self.last_good
        elif (z > 0).sum() < 500:
            ok = False                                            # nothing to start the model from yet
        if ok:
            # fuse only what was seen reasonably head-on: a surface seen past FUSE_GRAZING degrees edge-on gives noisy depth, and
            # fusing it made rough patches that still counted as well covered (another side sees that face straight on)
            zi = z
            if self._cosang is not None and self.FUSE_GRAZING < 89.0:
                zi = z.copy(); zi[self._cosang < np.cos(np.radians(self.FUSE_GRAZING))] = 0
            if getattr(self,'keep_integration_depth',False):self.last_integration_depth=zi
            self._integrate(zi, self.T); self.fused += 1
        self.n += 1
        return dict(ok=ok, T=self.T.copy(), fit=fit, rmse=rmse, reloc=reloc, hinted=hinted, refused=refused, refused_T=refused_T, constrained=constrained, registration_cut=registration_cut,marker_anchored=marker_anchored,registration_points=registration_points,registration_stride=used_stride,
                    refusal_reason=("too little cleaned depth" if registration_points<REGISTRATION_MIN_POINTS else "pose guard" if refused else "fit or motion check") if not ok else None)

    def seed(self, raw, T):
        """Fuse a frame at a pose that is already known (reloading a saved side); no matching. Tracking then carries on from T."""
        z = raw.copy(); z[z * self.depth_scale > self.DMAX * 1000.0] = 0
        z = clean_frame(z, self.cal, self.grid, depth_scale=self.depth_scale)
        self._integrate(z, T); self.fused += 1; self.n += 1
        self.T = np.asarray(T, float).copy(); self.last_good = self.T.copy(); self.vel = np.eye(4); self.streak = 0
        self.model = None

    # coverage: how many frames have seen a surface spot (TSDF weight). Below THIN it's barely there, from GOOD up it's solid
    COVER_THIN, COVER_GOOD = 4.0, 12.0
    FUSE_GRAZING = 65.0        # degrees off head-on past which a pixel is tracked against but not fused (tracking keeps the 80 degree cut)
    _cosang = None

    def coverage_levels(self, P):
        """For model points P (metres): 0 = seen fewer than COVER_THIN times, 1 = in between, 2 = COVER_GOOD or more.
        The surface extracted at a higher weight threshold is a subset of the lower one, so a point's level is the highest
        threshold whose surface still has a point in its 2 mm cell. (With 1 mm cells the colours speckled: the same
        surface comes out a fraction of a mm apart at two thresholds, often on either side of a cell wall.)"""
        def keys(Q): return self._cells(Q, 2.0)
        lv = np.zeros(len(P), np.int8); kp = keys(P)
        for level, thr in ((1, self.COVER_THIN), (2, self.COVER_GOOD)):
            try: S = self.vbg.extract_point_cloud(weight_threshold=thr).point.positions.cpu().numpy()
            except RuntimeError: break                  # nothing seen that often yet: Open3D refuses an empty surface
            if len(S): lv[np.isin(kp, keys(S))] = level
        return lv

    def render(self, T, scale=0.5, weight_threshold=None, coverage=False):
        """Shaded view of the model built so far, seen from camera pose T (OpenCV camera->world, metres). Returns HxWx3 uint8
        (and, with coverage=True, the share of the surface that is well covered: red = barely seen, yellow = getting there,
        blue = well covered). Splats the tracking model's points (1 mm apart, with normals) through a depth test. Not
        VoxelBlockGrid.ray_cast: on a freshly made grid it spun the GPU forever (seen live, twice)."""
        w, h = int(self.W * scale), int(self.H * scale)
        img = np.full((h, w, 3), 24, np.uint8)
        if self.fused == 0: return (img, 0.0) if coverage else img
        if self.model is None: self.model = self._model_cloud()
        P = self.model.point.positions.cpu().numpy(); N = self.model.point.normals.cpu().numpy()
        if not len(P): return (img, 0.0) if coverage else img
        palette = np.array([[255, 70, 50], [255, 200, 40], [90, 170, 255]], float)
        if coverage:
            lv = self.levels if (self.levels is not None and len(self.levels) == len(P)) else self.coverage_levels(P)
            good = float((lv == 2).mean())
            base = palette[lv]
        else:
            base = np.broadcast_to(palette[2], P.shape)
        Ti = np.linalg.inv(T); Pc = P @ Ti[:3, :3].T + Ti[:3, 3]; Nc = N @ Ti[:3, :3].T
        fx, fy, cx, cy = (c * scale for c in self.cal)
        z = Pc[:, 2]; ok = z > 0.02
        u = (Pc[ok, 0] * fx / z[ok] + cx).astype(np.int32); v = (Pc[ok, 1] * fy / z[ok] + cy).astype(np.int32)
        zz = z[ok]; shade = np.clip(0.15 + 0.85 * np.abs(Nc[ok, 2]), 0, 1)
        bc = base[ok]
        inside = (u >= 0) & (u < w - 2) & (v >= 0) & (v < h - 2)
        u, v, zz, shade, bc = u[inside], v[inside], zz[inside], shade[inside], bc[inside]
        order = np.argsort(-zz)                                   # far first, so nearer points overwrite them
        u, v, shade, bc = u[order], v[order], shade[order], bc[order]
        col = (shade[:, None] * bc).astype(np.uint8)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):   # 3x3 splats close the gaps
            img[v + dv, u + du] = col
        return (img, good) if coverage else img

    def mesh(self, weight_threshold=3.0):
        m = self.vbg.extract_triangle_mesh(weight_threshold=weight_threshold).to_legacy()
        m.scale(1000.0, center=(0, 0, 0)); m.compute_vertex_normals()
        return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.5, help="mm, the tracking model's resolution")
    ap.add_argument("--depth-scale", type=float, default=0.1); ap.add_argument("--max-depth", type=float, default=600.0)
    ap.add_argument("--refresh", type=int, default=3, help="rebuild the model cloud every N frames")
    ap.add_argument("--min-fit", type=float, default=0.5, help="share of a frame's points that must match the model")
    ap.add_argument("--max-rmse", type=float, default=1.5, help="mm, largest accepted fit residual")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import open3d.core as o3c
    use_gpu = a.gpu and o3c.cuda.is_available()
    if not use_gpu: apply_mem_cap()
    dev = o3c.Device("CUDA:0") if use_gpu else o3c.Device("CPU:0")

    dphs = sorted(glob.glob(os.path.join(a.frames, "*.dph")), key=lambda p: frame_key(p) or (0, 0))
    dphs = [p for p in dphs if frame_key(p) and os.path.exists(p[:-4] + ".inf")][::max(1, a.every)]
    if len(dphs) < 2: emit("error", msg="too few frames to track (%d)" % len(dphs)); return 2
    W, H = frame_size(a.frames, dphs)
    cal = read_calib(a.calib)
    if W != 800: cal = tuple(c * W / 800.0 for c in cal)        # Pl.bin is for the 800x600 frames
    emit("frames", count=len(dphs), size="%dx%d" % (W, H), device=str(dev))

    # start where the scanner started, so the result sits in the scan's own coordinate frame
    T0 = read_pose(dphs[0][:-4] + ".inf") @ F; T0[:3, 3] /= 1000.0
    tr = Tracker(cal, W, H, dev, T0, voxel_mm=a.voxel, depth_scale=a.depth_scale, max_depth_mm=a.max_depth,
                 refresh=a.refresh, min_fit=a.min_fit, max_rmse=a.max_rmse)
    entries = []; lost = []; t0 = time.time()
    for i, p in enumerate(dphs):
        raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint16)
        if raw.size != W * H: lost.append("%d_%d" % frame_key(p)); continue
        r = tr.step(raw.reshape(H, W))
        if r["ok"]:
            M = r["T"] @ F; M[:3, 3] *= 1000.0                   # back to the scanner's camera frame, mm
            entries.append((frame_key(p)[0], frame_key(p)[1], M))
        else: lost.append("%d_%d" % frame_key(p))
        if i % 10 == 0 or i == len(dphs) - 1:
            emit("track", done=i + 1, total=len(dphs), lost=len(lost), relocs=tr.relocs)
    relocs = tr.relocs

    out = a.out or os.path.join(a.frames, "pointyoink_register_pose.pose")
    tmp = out + ".tmp.%d" % os.getpid()
    write_pose_table(tmp, entries); os.replace(tmp, out)
    # how far the scanner's own tracking was from ours: large numbers mean it slipped
    moved = []
    for ps, fr, M in entries:
        inf = os.path.join(a.frames, "frame_%03d_%04d.inf" % (ps, fr))
        if os.path.exists(inf): moved.append(float(np.linalg.norm(read_pose(inf)[:3, 3] - M[:3, 3])))
    report = dict(version=1, frames=len(dphs), tracked=len(entries), lost=lost, relocalized=relocs,
                  secs=round(time.time() - t0, 1), size="%dx%d" % (W, H),
                  scanner_offset_median_mm=round(float(np.median(moved)), 2) if moved else None,
                  scanner_offset_max_mm=round(float(np.max(moved)), 2) if moved else None)
    json.dump(report, open(os.path.join(os.path.dirname(out), "pointyoink_track.json"), "w"), indent=1)
    emit("done", out=os.path.basename(out), frames=len(dphs), tracked=len(entries), lost=len(lost), relocs=relocs,
         moved_median_mm=report["scanner_offset_median_mm"] or 0, moved_max_mm=report["scanner_offset_max_mm"] or 0,
         secs=report["secs"])
    return 0 if entries else 1

if __name__ == "__main__":
    try: sys.exit(main())
    except MemoryError: emit("error", msg="out of memory"); sys.exit(3)
    except Exception as e: emit("error", msg=str(e)); sys.exit(1)
