#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay a native MIRACO scan's raw depth through our own tracker and compare to the device's poses.

The device's .inf / global_register_pose poses are ground truth. If our tracker reproduces them from
depth alone, live scanning on Linux is feasible; if not, we learn where and why it breaks.

usage: replay_track.py <scan data dir> [--voxel mm] [--diff mm] [--every N] [--pass N] [--filter] [--out mesh.ply]
"""
import argparse, glob, os, re, struct, sys, time, json
import numpy as np
import open3d as o3d
import open3d.core as o3c
from open3d.t.pipelines import slam

F = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV camera -> scanner camera frame (same as fuse.py)


def read_calib(path):
    v = struct.unpack("<10f", open(path, "rb").read())
    return v[1], v[5], v[3], v[6]


def read_pose(path):
    return np.array(struct.unpack_from("<16d", open(path, "rb").read(), 16)).reshape(4, 4)


def read_global_poses(path):
    b = open(path, "rb").read(); n = struct.unpack_from("<i", b, 0)[0]; out = {}
    for i in range(n):
        off = 4 + i * 136
        if off + 136 > len(b): break
        seq, frame = struct.unpack_from("<ii", b, off)
        out[(seq, frame)] = np.array(struct.unpack_from("<16d", b, off + 8)).reshape(4, 4)
    return out


def key(p):
    m = re.search(r"frame_(\d+)_(\d+)\.dph$", p)
    return int(m.group(1)), int(m.group(2))


def clean_frame(d, cal, jump_mm=3.0, grazing=80.0):
    """fuse.py's per-frame filter: drop depth jumps and grazing pixels."""
    fx, fy, cx, cy = cal
    z = d.astype(np.float32) * 0.1; valid = z > 0
    jump = np.zeros_like(valid)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        zs = np.roll(z, (dy, dx), axis=(0, 1)); vs = np.roll(valid, (dy, dx), axis=(0, 1))
        jump |= vs & (np.abs(zs - z) > jump_mm)
    edge = jump | ~valid
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, -1, 0) | np.roll(edge, 1, 1) | np.roll(edge, -1, 1)
    u, v = np.meshgrid(np.arange(d.shape[1], dtype=np.float32), np.arange(d.shape[0], dtype=np.float32))
    X = (u - cx) * z / fx; Y = (v - cy) * z / fy
    dXu = np.roll(X, -1, 1) - np.roll(X, 1, 1); dYu = np.roll(Y, -1, 1) - np.roll(Y, 1, 1); dZu = np.roll(z, -1, 1) - np.roll(z, 1, 1)
    dXv = np.roll(X, -1, 0) - np.roll(X, 1, 0); dYv = np.roll(Y, -1, 0) - np.roll(Y, 1, 0); dZv = np.roll(z, -1, 0) - np.roll(z, 1, 0)
    nx = dYu * dZv - dZu * dYv; ny = dZu * dXv - dXu * dZv; nz = dXu * dYv - dYu * dXv
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    rl = np.sqrt(X * X + Y * Y + z * z) + 1e-9
    cosang = np.abs((nx * X + ny * Y + nz * z) / (nl * rl))
    d = d.copy(); d[edge | (valid & (cosang < np.cos(np.radians(grazing))))] = 0
    return d


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--voxel", type=float, default=1.0, help="tracking model voxel, mm")
    ap.add_argument("--diff", type=float, default=10.0, help="max ICP correspondence distance, mm")
    ap.add_argument("--max-depth", type=float, default=600.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--pass", dest="pas", type=int, default=None, help="only this scan pass (seq number)")
    ap.add_argument("--filter", action="store_true", help="apply fuse.py's edge/grazing filter before tracking")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trunc", type=float, default=8.0, help="TSDF truncation, voxels")
    ap.add_argument("--oracle", action="store_true", help="start each frame at the device pose; report ICP's correction")
    ap.add_argument("--velocity", action="store_true", help="constant-velocity motion prediction")
    ap.add_argument("--out", default=None, help="write our fused mesh here (mm)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    sp = json.load(open(os.path.join(a.scan, "property.rvproj"))).get("scan_param", {})
    W, H = int(sp["depth_width"]), int(sp["depth_height"])
    cal = read_calib(os.path.join(a.scan, "param", "Pl.bin"))
    if W != 800: cal = tuple(c * W / 800.0 for c in cal)
    fx, fy, cx, cy = cal
    cache = os.path.join(a.scan, "cache")
    gp_path = os.path.join(cache, "global_register_pose.pose")
    gp = read_global_poses(gp_path) if os.path.exists(gp_path) else {}
    dphs = sorted(glob.glob(os.path.join(cache, "*.dph")), key=key)
    dphs = [p for p in dphs if os.path.exists(p[:-4] + ".inf")]
    if a.pas is not None: dphs = [p for p in dphs if key(p)[0] == a.pas]
    dphs = dphs[::a.every]
    if a.limit: dphs = dphs[:a.limit]

    def gt_pose(p):
        P = gp.get(key(p))
        P = read_pose(p[:-4] + ".inf") if P is None else P
        P = P @ F; P = P.copy(); P[:3, 3] /= 1000.0   # OpenCV cam -> world, metres
        return P

    dev = o3c.Device("CUDA:0")
    K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
    depth_scale = 10000.0            # 0.1 mm/count -> metres
    depth_max = a.max_depth / 1000.0
    T0 = gt_pose(dphs[0])
    T = o3c.Tensor(T0, o3c.float64)
    model = slam.Model(a.voxel / 1000.0, 16, 60000, T, dev)
    inp = slam.Frame(H, W, K, dev); ray = slam.Frame(H, W, K, dev)
    color = o3d.t.geometry.Image(o3c.Tensor(np.zeros((H, W, 3), np.uint8))).to(dev)

    print(f"{os.path.basename(a.scan)}  {len(dphs)} frames  {W}x{H}  f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f})  "
          f"global_poses={len(gp)}  voxel={a.voxel}mm diff={a.diff}mm filter={a.filter}")
    rows = []; t_start = time.time(); lost = 0
    Tn = T0.copy(); vel = np.eye(4); corr = []
    trunc = a.trunc
    for i, p in enumerate(dphs):
        d = np.frombuffer(open(p, "rb").read(), np.uint16).reshape(H, W).copy()
        d[d * 0.1 > a.max_depth] = 0
        if a.filter: d = clean_frame(d, cal)
        inp.set_data_from_image("depth", o3d.t.geometry.Image(o3c.Tensor(d)).to(dev))
        inp.set_data_from_image("color", color)
        fit = rmse = float("nan")
        if i > 0:
            # predict: oracle = device pose (convention check), else constant velocity
            pred = gt_pose(p) if a.oracle else (Tn @ vel if a.velocity else Tn)
            model.update_frame_pose(i, o3c.Tensor(pred, o3c.float64))
            model.synthesize_model_frame(ray, depth_scale, 0.05, depth_max, trunc, False)
            try:
                r = model.track_frame_to_model(inp, ray, depth_scale, depth_max, a.diff / 1000.0)
                fit, rmse = float(r.fitness), float(r.inlier_rmse) * 1000.0
                dT = r.transformation.cpu().numpy()
                corr.append((np.linalg.norm(dT[:3, 3]) * 1000, rot_err_deg(dT[:3, :3], np.eye(3))))
                new = pred @ dT
            except RuntimeError:
                lost += 1; new = pred
            if a.oracle: new = gt_pose(p)          # keep the model clean; we only measure the correction
            vel = np.linalg.inv(Tn) @ new; Tn = new
        model.update_frame_pose(i, o3c.Tensor(Tn, o3c.float64))
        model.integrate(inp, depth_scale, depth_max, trunc)
        Te = Tn; Tg = gt_pose(p)
        # error expressed relative to the first frame (both trajectories start at the same pose)
        et = float(np.linalg.norm(Te[:3, 3] - Tg[:3, 3]) * 1000.0)
        er = rot_err_deg(Te[:3, :3], Tg[:3, :3])
        rows.append(dict(i=i, frame="%d_%d" % key(p), fit=fit, rmse=rmse, et=et, er=er))
        if i % 25 == 0 or i == len(dphs) - 1:
            print(f"  {i:4d} {rows[-1]['frame']:>9}  fit={fit:5.2f} rmse={rmse:5.2f}mm   drift: {et:7.2f} mm  {er:6.2f} deg")
    dt = time.time() - t_start
    et = np.array([r["et"] for r in rows]); er = np.array([r["er"] for r in rows])
    # relative pose error over 10-frame windows: local tracking quality independent of accumulated drift
    if corr:
        c = np.array(corr)
        print(f"per-frame ICP correction: median {np.median(c[:,0]):.2f} mm / {np.median(c[:,1]):.2f} deg, p95 {np.percentile(c[:,0],95):.2f} mm / {np.percentile(c[:,1],95):.2f} deg")
    print(f"done {len(dphs)} frames in {dt:.1f}s ({len(dphs)/dt:.1f} fps)  tracker exceptions={lost}")
    print(f"drift vs device: median {np.median(et):.2f} mm / {np.median(er):.2f} deg, "
          f"p95 {np.percentile(et,95):.2f} mm / {np.percentile(er,95):.2f} deg, final {et[-1]:.2f} mm / {er[-1]:.2f} deg")
    if a.json: json.dump(rows, open(a.json, "w"))
    if a.out:
        m = model.extract_trianglemesh(weight_threshold=3.0).to_legacy()
        m.scale(1000.0, center=(0, 0, 0)); m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(a.out, m)
        print("mesh ->", a.out, len(m.triangles), "tris")


if __name__ == "__main__":
    main()
