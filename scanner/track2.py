#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frame-to-model tracker for MIRACO depth: multi-scale robust point-to-plane ICP against the fused TSDF.

Replays a native scan and scores our trajectory against the device's own poses.
usage: track2.py <scan data dir> [--pass N] [--every N] [--limit N] [--out mesh.ply]
"""
import argparse, glob, os, sys, time, json
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.t.pipelines.registration as treg
from open3d.t.pipelines.registration.robust_kernel import RobustKernel, RobustKernelMethod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_track import F, read_calib, read_pose, read_global_poses, key, clean_frame, rot_err_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--pass", dest="pas", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--voxel", type=float, default=0.5, help="TSDF voxel mm")
    ap.add_argument("--max-depth", type=float, default=600.0)
    ap.add_argument("--refresh", type=int, default=3, help="re-extract the model cloud every N frames")
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--no-reloc", action="store_true")
    ap.add_argument("--min-fit", type=float, default=0.5)
    ap.add_argument("--ref", action="store_true", help="also fuse at the device's poses and compare surfaces")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    sp = json.load(open(os.path.join(a.scan, "property.rvproj"))).get("scan_param", {})
    W, H = int(sp["depth_width"]), int(sp["depth_height"])
    cal = read_calib(os.path.join(a.scan, "param", "Pl.bin"))
    if W != 800: cal = tuple(c * W / 800.0 for c in cal)
    fx, fy, cx, cy = cal
    cache = os.path.join(a.scan, "cache")
    gpp = os.path.join(cache, "global_register_pose.pose")
    gp = read_global_poses(gpp) if os.path.exists(gpp) else {}
    dphs = [p for p in sorted(glob.glob(os.path.join(cache, "*.dph")), key=key) if os.path.exists(p[:-4] + ".inf")]
    if a.pas is not None: dphs = [p for p in dphs if key(p)[0] == a.pas]
    dphs = dphs[::a.every]
    if a.limit: dphs = dphs[:a.limit]

    def gt(p):
        P = gp.get(key(p)); P = read_pose(p[:-4] + ".inf") if P is None else P
        P = P @ F; P = P.copy(); P[:3, 3] /= 1000.0
        return P

    dev = o3c.Device("CUDA:0")
    K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
    DS, DMAX = 10000.0, a.max_depth / 1000.0
    vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3c.float32, o3c.float32), ((1), (1)),
                                        a.voxel / 1000.0, 16, 25000, dev)
    # coarse -> fine: voxel (m), max correspondence (m), iterations
    scales = [(0.004, 0.020, 15), (0.002, 0.008, 15), (0.001, 0.003, 25)]
    kern = RobustKernel(RobustKernelMethod.TukeyLoss, 0.002)
    est = treg.TransformationEstimationPointToPlane(kern)

    def integrate(z, T):
        img = o3d.t.geometry.Image(o3c.Tensor(z)).to(dev); ext = o3c.Tensor(np.linalg.inv(T))
        bc = vbg.compute_unique_block_coordinates(img, K, ext, DS, DMAX, 3.0)
        vbg.integrate(bc, img, K, ext, DS, DMAX, 3.0)

    def model_cloud():
        pc = vbg.extract_point_cloud(weight_threshold=0.5)
        pc = pc.voxel_down_sample(0.001)
        pc.estimate_normals(max_nn=20, radius=0.003)
        return pc

    ref = None
    if a.ref:   # same fusion at the device's poses: the yardstick for our mesh
        ref = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3c.float32, o3c.float32), ((1), (1)),
                                            a.voxel / 1000.0, 16, 25000, dev)

    def integrate_into(g, z, T):
        img = o3d.t.geometry.Image(o3c.Tensor(z)).to(dev); ext = o3c.Tensor(np.linalg.inv(T))
        bc = g.compute_unique_block_coordinates(img, K, ext, DS, DMAX, 3.0)
        g.integrate(bc, img, K, ext, DS, DMAX, 3.0)

    def legacy_down(pc_t, vox):
        pc = pc_t.to_legacy().voxel_down_sample(vox)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 3, max_nn=30))
        return pc

    def relocalize(src_t, model_t):
        """Global registration of the current frame against the model: FPFH + RANSAC, then fine ICP."""
        L = o3d.pipelines.registration
        vox = 0.003
        s = legacy_down(src_t, vox); m = legacy_down(model_t, vox)
        fs = L.compute_fpfh_feature(s, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))
        fm = L.compute_fpfh_feature(m, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))
        best = None
        for _ in range(3):
            r = L.registration_ransac_based_on_feature_matching(
                s, m, fs, fm, True, vox * 1.5, L.TransformationEstimationPointToPoint(False), 3,
                [L.CorrespondenceCheckerBasedOnEdgeLength(0.9), L.CorrespondenceCheckerBasedOnDistance(vox * 1.5)],
                L.RANSACConvergenceCriteria(200000, 0.999))
            try:
                f = treg.multi_scale_icp(src_t, model_t, o3d.utility.DoubleVector([0.002, 0.001]),
                                         [treg.ICPConvergenceCriteria(1e-6, 1e-6, 30)] * 2,
                                         o3d.utility.DoubleVector([0.006, 0.003]), o3c.Tensor(r.transformation), est)
            except RuntimeError:
                continue
            if best is None or f.fitness > best.fitness: best = f
        return best

    print(f"{os.path.basename(a.scan)}: {len(dphs)} frames {W}x{H}")
    T0 = gt(dphs[0]); Tn = T0.copy(); vel = np.eye(4); last_good = T0.copy()
    rows = []; model = None; t0 = time.time(); fails = 0; streak = 0; relocs = 0
    for i, p in enumerate(dphs):
        z = np.frombuffer(open(p, "rb").read(), np.uint16).reshape(H, W).copy()
        z[z * 0.1 > a.max_depth] = 0
        z = clean_frame(z, cal)
        fit = rmse = float("nan"); how = ""
        if i > 0:
            if model is None or i % a.refresh == 0 or streak: model = model_cloud()
            src = o3d.t.geometry.PointCloud.create_from_depth_image(
                o3d.t.geometry.Image(o3c.Tensor(z)).to(dev), K, o3c.Tensor(np.eye(4)), DS, DMAX, stride=1 if W <= 400 else 2)
            pred = last_good if (streak or a.no_velocity) else Tn @ vel
            ok = False; new = pred
            try:
                r = treg.multi_scale_icp(src, model, o3d.utility.DoubleVector([s[0] for s in scales]),
                                         [treg.ICPConvergenceCriteria(1e-6, 1e-6, s[2]) for s in scales],
                                         o3d.utility.DoubleVector([s[1] for s in scales]), o3c.Tensor(pred), est)
                fit, rmse = float(r.fitness), float(r.inlier_rmse) * 1000
                ok = fit > a.min_fit and rmse < 1.5; new = r.transformation.cpu().numpy()
            except RuntimeError:
                pass
            if not ok and streak >= 1 and not a.no_reloc:
                r = relocalize(src, model)
                if r is not None and r.fitness > a.min_fit and r.inlier_rmse * 1000 < 1.2:
                    ok = True; new = r.transformation.cpu().numpy(); relocs += 1; how = " RELOC"
                    fit, rmse = float(r.fitness), float(r.inlier_rmse) * 1000
            if ok:
                vel = np.eye(4) if streak else np.linalg.inv(Tn) @ new
                Tn = new; last_good = new; streak = 0
            else:
                # lost: never extrapolate or fuse; wait at the last good pose and try to relocalize
                fails += 1; streak += 1; vel = np.eye(4); Tn = last_good
        else:
            ok = True
        if ok: integrate_into(vbg, z, Tn)
        Tg = gt(p)
        if ref is not None:
            # the device's own trajectory has glitches (e.g. a 178 deg flip); keep them out of the yardstick
            jump = i > 0 and (np.linalg.norm(Tg[:3, 3] - Tg_prev[:3, 3]) > 0.03 or rot_err_deg(Tg[:3, :3], Tg_prev[:3, :3]) > 10)
            if not jump: integrate_into(ref, z, Tg)
        Tg_prev = Tg
        et = float(np.linalg.norm(Tn[:3, 3] - Tg[:3, 3]) * 1000); er = rot_err_deg(Tn[:3, :3], Tg[:3, :3])
        rows.append(dict(i=i, frame="%d_%d" % key(p), fit=fit, rmse=rmse, et=et, er=er, ok=ok, T=Tn.tolist()))
        if i % 50 == 0 or i == len(dphs) - 1 or how:
            print(f"  {i:4d} {rows[-1]['frame']:>9} fit={fit:4.2f} rmse={rmse:5.2f}mm  vs device: {et:6.2f} mm {er:5.2f} deg{how}")
    dt = time.time() - t0
    et = np.array([r["et"] for r in rows]); er = np.array([r["er"] for r in rows])
    X = np.array([np.array(r["T"])[:3, 3] for r in rows if r["ok"]]) * 1000
    jit = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=1) if len(X) > 2 else np.zeros(1)
    print(f"{len(dphs)} frames in {dt:.1f}s ({len(dphs)/dt:.1f} fps), lost {fails}, relocalized {relocs}, jitter median {np.median(jit):.2f} mm")
    print(f"vs device: median {np.median(et):.2f} mm / {np.median(er):.2f} deg | p95 {np.percentile(et,95):.2f} mm / "
          f"{np.percentile(er,95):.2f} deg")
    if a.json: json.dump(rows, open(a.json, "w"))
    if a.out or ref is not None:
        m = vbg.extract_triangle_mesh(weight_threshold=3.0).to_legacy()
        m.scale(1000.0, center=(0, 0, 0)); m.compute_vertex_normals()
        if a.out: o3d.io.write_triangle_mesh(a.out, m); print("mesh ->", a.out, len(m.triangles), "tris")
    if ref is not None:
        rm = ref.extract_triangle_mesh(weight_threshold=3.0).to_legacy()
        rm.scale(1000.0, center=(0, 0, 0)); rm.compute_vertex_normals()
        if a.out: o3d.io.write_triangle_mesh(a.out.replace(".ply", "_device.ply"), rm)
        # surface agreement, both ways: ours->device finds ghosts/doubled walls, device->ours finds missing surface
        def dists(A, B):
            sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(B))
            pts = A.sample_points_uniformly(200000).points
            return sc.compute_distance(o3c.Tensor(np.asarray(pts), o3c.float32)).numpy()
        # rigidly align ours onto the device mesh first: a whole-model offset is harmless, shape error is not
        pa = m.sample_points_uniformly(100000); pb = rm.sample_points_uniformly(100000)
        pb.estimate_normals()
        al = o3d.pipelines.registration.registration_icp(pa, pb, 1.0, np.eye(4),
                 o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        A = al.transformation
        print("rigid offset ours->device: %.2f mm %.2f deg" % (np.linalg.norm(A[:3, 3]), rot_err_deg(A[:3, :3], np.eye(3))))
        m.transform(A)
        d1 = dists(m, rm); d2 = dists(rm, m)
        print(f"surface vs device-pose mesh: ours->dev median {np.median(d1):.3f} p95 {np.percentile(d1,95):.3f} mm | "
              f"dev->ours median {np.median(d2):.3f} p95 {np.percentile(d2,95):.3f} mm | "
              f"ours within 0.3mm: {100*np.mean(d1<0.3):.1f}%")


if __name__ == "__main__":
    main()
