# SPDX-License-Identifier: AGPL-3.0-or-later
import sys, numpy as np, open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
out = sys.argv[1]; files = sys.argv[2:]
ref = o3d.io.read_triangle_mesh(files[0]); v = np.asarray(ref.vertices); c = (v.min(0) + v.max(0)) / 2; r = np.linalg.norm(v.max(0) - v.min(0)) / 2
ax = np.argmax(v.max(0) - v.min(0)); tip = c.copy(); tip[ax] = v[:, ax].min() + 25      # one end of the part: rim and flange close up
rng = np.random.default_rng(0); refp = np.asarray(ref.sample_points_uniformly(200000).points); tree = cKDTree(refp)
W, H = 640, 480; sheet = Image.new("RGB", (W * len(files), (H + 20) * 3)); dr = ImageDraw.Draw(sheet)
for j, f in enumerate(files):
    m = ref if j == 0 else o3d.io.read_triangle_mesh(f)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    if j: 
        d = tree.query(np.asarray(m.sample_points_uniformly(200000).points))[0]
        note = "vs built: median %.2f mm, 95%% %.2f mm, max %.1f mm" % (np.median(d), np.percentile(d, 95), d.max())
    else: note = "as built"
    print(f.split("/")[-1], note)
    for k, (ctr, e, dist) in enumerate(((c, [0.3, -0.6, -1], 2.0), (tip, [-1, -0.7, -0.6], 0.75), (tip, [-0.6, 0.9, 0.5], 0.75))):
        eye = ctr + np.asarray(e, float) / np.linalg.norm(e) * r * dist
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=40, center=ctr.tolist(), eye=eye.tolist(), up=[0, -1, 0], width_px=W, height_px=H)
        a = sc.cast_rays(rays); hit = np.isfinite(a["t_hit"].numpy()); n = a["primitive_normals"].numpy()
        dd = rays.numpy()[..., 3:]; dd /= np.linalg.norm(dd, axis=-1, keepdims=True); sh = np.abs((n * dd).sum(-1))
        img = np.full((H, W, 3), 24, np.uint8); img[hit] = (np.clip(0.1 + 0.9 * sh[hit], 0, 1)[:, None] * np.array([150, 160, 175.0])).astype(np.uint8)
        sheet.paste(Image.fromarray(img), (W * j, (H + 20) * k + 20))
    dr.text((W * j + 6, 4), "%s  %dk tris  %s" % (f.split("/")[-1], len(m.triangles) // 1000, note), fill=(255, 255, 255))
sheet.save(out)
