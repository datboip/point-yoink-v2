# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bambu Studio style move arrows: three arrows along the world X / Y / Z axes from a centre, red / green / blue;
hovering lights one, dragging it slides the thing along that axis only (Shift: 1 mm steps). Pure math here, tested
without GL; edit_view draws the arrows on its overlay next to the rotate rings and routes the mouse.

  k = arrow_hit(cam, c, length, (x, y), w, h)                  # 0 / 1 / 2 (X / Y / Z) or None
  d = ArrowDrag(cam, c, k, (x, y), w, h); mm = d.moved((x, y), snap=shift)   # along +axis since the press

cam: anything with project(P, w, h) -> (x, y, eye depth) and unproject(x, y, depth, w, h) (gl_core.Camera)."""
import numpy as np
from app2.qt.rotate_gizmo import AXES, COLOURS, HOVER, view_ray

HIT_PX = 9.0                   # px from an arrow's shaft or head that still grabs it
SNAP_MM = 1.0                  # Shift
HEAD = 0.18                    # the cone head's length as a share of the arrow
HEAD_W = 0.07                  # its half width


def arrow_points(c, length, k):
    """(tail, tip, head base) of arrow k in world mm."""
    c = np.asarray(c, float); a = AXES[k]
    return c, c + a * length, c + a * length * (1.0 - HEAD)


def project_arrow(cam, c, length, k, w, h):
    """Screen (3,2) of tail, tip, head base and their eye depths."""
    P = np.array(arrow_points(c, length, k)); x, y, z = cam.project(P, w, h)
    return np.stack([x, y], 1), z


def _seg_dist(p, A, B):
    d = B - A; L2 = float(d @ d)
    t = 0.0 if L2 < 1e-12 else float(np.clip((p - A) @ d / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (A + d * t)))


def arrow_hit(cam, c, length, pos, w, h, tol=HIT_PX):
    """The arrow under the cursor (nearest shaft within tol px), or None."""
    p = np.asarray(pos, float); best = None
    for k in range(3):
        xy, z = project_arrow(cam, c, length, k, w, h)
        if not (z > 0).all() or not np.isfinite(xy).all(): continue
        d = _seg_dist(p, xy[0], xy[1])
        if d <= tol and (best is None or d < best[0]): best = (d, k)
    return None if best is None else best[1]


def axis_param(cam, c, k, pos, w, h):
    """Where along axis k (mm from c) the point of the axis line nearest the cursor's ray is; None when the ray runs
    along the axis."""
    o, d = view_ray(cam, pos, w, h); a = AXES[k]; c = np.asarray(c, float)
    # closest points of two lines: c + a*s and o + d*t
    ad = float(a @ d); den = 1.0 - ad * ad
    if den < 1e-9: return None
    r = o - c
    s = (float(r @ a) - ad * float(r @ d)) / den
    return float(s)


class ArrowDrag:
    """One drag on arrow k from pos0. moved(pos) -> mm slid along +axis since the press. The cursor's ray is dropped
    onto the axis line each move, so the arrow stays under the cursor whatever the view angle; an axis seen nearly end
    on falls back to screen travel along the arrow's screen direction."""
    def __init__(self, cam, c, k, pos0, w, h, length=10.0):
        self.cam, self.c, self.k, self.w, self.h = cam, np.asarray(c, float), k, w, h
        self.p0 = np.asarray(pos0, float); self.s0 = axis_param(cam, c, k, pos0, w, h)
        xy, _z = project_arrow(cam, c, length, k, w, h)
        t = xy[1] - xy[0]; n = float(np.linalg.norm(t))
        self.tan = t / n if n > 1e-9 else np.array([1.0, 0.0])
        self.px_per_mm = max(n / max(length, 1e-9), 1e-6)
        self.end_on = self.s0 is None or n < 12.0

    def moved(self, pos, snap_on=False):
        p = np.asarray(pos, float)
        if self.end_on: mm = float((p - self.p0) @ self.tan) / self.px_per_mm
        else:
            s = axis_param(self.cam, self.c, self.k, p, self.w, self.h)
            mm = 0.0 if s is None else s - self.s0
        return float(np.round(mm / SNAP_MM) * SNAP_MM) if snap_on else mm


def label(k, mm):
    """The readout by the arrow while dragging."""
    return "%s: %+.1f mm" % ("XYZ"[k], mm)
