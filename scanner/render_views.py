#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Six-angle shaded preview of a mesh, rendered offscreen (raycast on the CPU; no window, no screen capture).
usage: render_views.py model.ply out.png"""
import sys, time, json, numpy as np, open3d as o3d
started = time.perf_counter(); timings = {}
from PIL import Image, ImageDraw

m = o3d.io.read_triangle_mesh(sys.argv[1])
v = np.asarray(m.vertices); c = (v.min(0) + v.max(0)) / 2; r = np.linalg.norm(v.max(0) - v.min(0)) / 2
timings["mesh_read"] = time.perf_counter() - started; phase = time.perf_counter()
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

timings["scene"] = time.perf_counter() - phase; phase = time.perf_counter()

def render(eye_dir, W=560, H=420):
    eye = c + np.asarray(eye_dir, float) / np.linalg.norm(eye_dir) * r * 2.3
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=45, center=c.tolist(), eye=eye.tolist(), up=[0, -1, 0], width_px=W, height_px=H)
    ans = sc.cast_rays(rays); hit = np.isfinite(ans["t_hit"].numpy()); n = ans["primitive_normals"].numpy()
    d = rays.numpy()[..., 3:]; d /= np.linalg.norm(d, axis=-1, keepdims=True); sh = np.abs((n * d).sum(-1))
    img = np.full((H, W, 3), 24, np.uint8)
    img[hit] = (np.clip(0.15 + 0.85 * sh[hit], 0, 1)[:, None] * np.array([90, 170, 255.0])).astype(np.uint8)
    if m.has_vertex_colors():
        ids = ans["primitive_ids"].numpy()[hit]; uv = ans["primitive_uvs"].numpy()[hit]
        weights = np.column_stack((1-uv.sum(1), uv))
        rgb = (np.asarray(m.vertex_colors)[np.asarray(m.triangles)[ids]] * weights[:,:,None]).sum(1)
        img[hit] = np.uint8(np.clip(rgb * (0.15 + 0.85 * sh[hit])[:,None] * 255, 0, 255))
    return Image.fromarray(img)


out = Image.new("RGB", (560 * 3, 440 * 2)); dr = ImageDraw.Draw(out)
for k, e in enumerate(([0, -0.4, -1], [0, 0.4, 1], [1, -0.5, 0], [-1, 0.3, 0.2], [0, -1, -0.1], [0, 1, 0.1])):
    out.paste(render(e), ((k % 3) * 560, (k // 3) * 440 + 20))
size = v.max(0) - v.min(0)
dr.text((6, 2), "%s   %d triangles   bounding box %.1f x %.1f x %.1f mm" % (sys.argv[1].split("/")[-1], len(m.triangles), *size), fill=(255, 255, 255))
timings["six_renders"] = time.perf_counter() - phase; phase = time.perf_counter()
out.save(sys.argv[2])
timings["png_write"] = time.perf_counter() - phase
timings["total"] = time.perf_counter() - started
if len(sys.argv) > 3: json.dump(timings, open(sys.argv[3], "w"), indent=2)
