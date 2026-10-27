"""strip_table.py refuses a plane only when part of the side sits well below it, not when the plate fills most of the view
(sides 1 and 4 of the 2026-10-06 scan: the right turntable plane removed 53% and 46%, all of it within 1.4 mm of the plane)."""
import json, os, struct, subprocess, sys
import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _scan(tmp_path):
    """A 200 x 200 mm plate at z = 0 (most of the vertices) with a 60 x 40 x 50 mm box on it, and a camera circle above."""
    scan = tmp_path / "scan"; (scan / "cache").mkdir(parents=True)
    plate = o3d.geometry.TriangleMesh.create_box(200, 200, 0.5).subdivide_midpoint(5); plate.translate((-100, -100, -0.5))
    box = o3d.geometry.TriangleMesh.create_box(60, 40, 50).subdivide_midpoint(3); box.translate((-30, -20, 0))
    m = plate + box; mesh = tmp_path / "raw.ply"; o3d.io.write_triangle_mesh(str(mesh), m)
    poses = []
    for k in range(24):
        a = 2 * np.pi * k / 24; M = np.eye(4); M[:3, 3] = [300 * np.cos(a), 300 * np.sin(a), 250]
        poses.append(struct.pack("<ii16d", 0, k, *M.ravel()))
    (scan / "cache" / "pointyoink_register_pose.pose").write_bytes(struct.pack("<i", len(poses)) + b"".join(poses))
    return scan, mesh, m


def _cut(tmp_path, scan, mesh, d):
    plane = tmp_path / ("plane_%g.json" % d); plane.write_text(json.dumps({"n": [0, 0, 1], "d": d}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scanner", "strip_table.py"), str(scan), str(tmp_path / "cut.ply"),
                        "--mesh", str(mesh), "--plane", str(plane)], capture_output=True, text=True, timeout=120)
    return r.stdout + r.stderr


def test_the_right_plane_on_a_plate_heavy_side_is_not_refused(tmp_path):
    scan, mesh, m = _scan(tmp_path)
    V = np.asarray(m.vertices); assert np.mean(V[:, 2] <= 1.0) > 0.4          # the plate is most of the side, as on sides 1 and 4
    out = _cut(tmp_path, scan, mesh, 0.0)
    assert "wrong_plane" not in out, out


def test_a_plane_through_the_part_is_refused(tmp_path):
    scan, mesh, _ = _scan(tmp_path)
    out = _cut(tmp_path, scan, mesh, 25.0)
    assert "wrong_plane" in out and "deep_share" in out, out
