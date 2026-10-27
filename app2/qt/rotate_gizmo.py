# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bambu Studio style rotate rings for Line-up Nudge: three circles about the world X / Y / Z axes through the picked
side's centre, red / green / blue; hovering lights a ring, dragging along it turns the side about that axis (Shift: 15 deg
steps). Pure math here, tested without GL; edit_view draws the rings on its overlay and routes the mouse.

  k = ring_hit(cam, c, radius, (x, y), w, h)                    # 0 / 1 / 2 (X / Y / Z) or None; w, h: widget pixels
  d = RingDrag(cam, c, radius, k, (x, y), w, h); deg = d.angle((x, y), snap=shift)   # right hand about +axis
  R = axis_rot(k, deg); euler_xyz(R) / from_euler(rx, ry, rz)   # the Rotate (absolute) boxes (R = Rz Ry Rx, as nudges)

cam: anything with project(P, w, h) -> (x, y, eye depth) and unproject(x, y, depth, w, h) (gl_core.Camera)."""
import numpy as np

AXES = np.eye(3)
COLOURS = ("#ff4d4d", "#4dd060", "#4d8dff")      # X red, Y green, Z blue (Bambu)
HOVER = "#ffd24d"
HIT_PX = 8.0                                     # px from a ring that still grabs it
SNAP = 15.0                                      # deg, Shift
RING_N = 96
EDGE_ON = 0.2                                    # |cos| between the view ray and the ring's axis below which the ring is
                                                 # nearly edge-on: the drag follows the ring's screen tangent instead
# Bambu's protractor, as fractions of the ring radius (the cursor's distance from the centre in the ring's plane):
FREE_IN = 0.8                                    # at or beyond: on the ring or outside it, free (the white circle)
SPOKE_OUT = 0.6                                  # FREE_IN > frac >= SPOKE_OUT: the tick band, 5 deg; below: spokes, 30 deg
SPOKE = 30.0                                     # deg between spokes (12), the spoke zone's snap
TICKS = ((30, 0.6), (15, 0.68), (5, 0.74))       # (every deg, inner end): longest at the spokes, then 15, short every 5


def basis(k):
    """(u, v) spanning ring k's plane, u x v = +axis (so positive angles turn right-handed about the axis)."""
    return AXES[(k + 1) % 3], AXES[(k + 2) % 3]


def axis_rot(k, deg):
    a = AXES[k]; t = np.radians(deg); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def from_euler(rx, ry, rz):
    return axis_rot(2, rz) @ axis_rot(1, ry) @ axis_rot(0, rx)


def euler_xyz(R):
    """(rx, ry, rz) deg with R = Rz Ry Rx (stages.tools.edits.r_xyz); at ry = +-90 rx is 0."""
    R = np.asarray(R, float); s = -np.clip(R[2, 0], -1, 1)
    if abs(s) > 1 - 1e-9: return 0.0, float(np.degrees(np.arcsin(s))), float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
    return (float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))), float(np.degrees(np.arcsin(s))),
            float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))


def snap(deg, step=SNAP): return float(np.round(deg / step) * step) if step else float(deg)


def snap_step(frac, shift=False):
    """deg a drag snaps to from where the cursor is (frac: its distance from the centre / ring radius; None = unknown,
    as on an edge-on ring): on the ring or anywhere outside it free, the tick band 5, the spokes 30; Shift at least 15."""
    step = 0.0 if frac is None or frac >= FREE_IN else 5.0 if frac >= SPOKE_OUT else SPOKE
    return max(step, SNAP) if shift else step


def protractor(c, radius, k, a0=0.0):
    """World segments [(A, B, kind)] of the protractor in ring k's plane, 0 at a0 deg: the white circle ("circle"), ticks
    every 5 deg ("tick", longer at 15, longest at 30) and spokes from the centre every 30 deg ("spoke")."""
    c = np.asarray(c, float); u, v = basis(k); at = lambda deg, f: c + radius * f * (np.cos(np.radians(deg)) * u + np.sin(np.radians(deg)) * v)
    segs = [(at(a0 + a, FREE_IN), at(a0 + a + 5, FREE_IN), "circle") for a in range(0, 360, 5)]
    for a in range(0, 360, 5):
        inner = next(f for step, f in TICKS if a % step == 0)
        segs.append((at(a0 + a, inner), at(a0 + a, FREE_IN), "tick"))
        if a % int(SPOKE) == 0: segs.append((c + 0.0, at(a0 + a, SPOKE_OUT), "spoke"))
    return segs


def ring_points(c, radius, k, n=RING_N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False); u, v = basis(k)
    return np.asarray(c, float) + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def project_ring(cam, c, radius, k, w, h, n=RING_N):
    """-> (screen xy (n, 2), eye depth (n,), world points (n, 3)) of ring k."""
    P = ring_points(c, radius, k, n); x, y, ze = cam.project(P, w, h)
    return np.stack([x, y], 1), ze, P


def seg_dist(p, A, B):
    """Distance from the point p to each segment A[i] -> B[i] (screen)."""
    d = B - A; L = np.maximum((d * d).sum(1), 1e-12); t = np.clip(((p - A) * d).sum(1) / L, 0, 1)
    return np.linalg.norm(A + t[:, None] * d - p, axis=1)


def ring_hit(cam, c, radius, pos, w, h, tol=HIT_PX):
    """The ring under the cursor: the nearest within tol px; the half of a ring nearer the camera wins a close call
    (3 px), as the front of a ring is drawn over the back."""
    p = np.asarray(pos, float); ce = cam.project(np.asarray(c, float)[None], w, h)[2][0]; best = None
    for k in range(3):
        xy, ze, _P = project_ring(cam, c, radius, k, w, h)
        ok = (ze > 0) & np.isfinite(xy).all(1); j = np.roll(np.arange(len(xy)), -1); ok = ok & ok[j]
        if not ok.any(): continue
        d = np.where(ok, seg_dist(p, xy, xy[j]), np.inf); i = int(np.argmin(d))
        if d[i] > tol: continue
        score = d[i] + (0.0 if (ze[i] + ze[j[i]]) / 2 <= ce else 3.0)
        if best is None or score < best[0]: best = (score, k)
    return None if best is None else best[1]


def view_ray(cam, pos, w, h):
    """World origin and unit direction of the ray through widget pixel pos."""
    x, y = float(pos[0]) - 0.5, float(pos[1]) - 0.5             # unproject takes pixel centres
    a, b = np.asarray(cam.unproject(x, y, 0.0, w, h), float), np.asarray(cam.unproject(x, y, 1.0, w, h), float)
    d = b - a; return a, d / max(np.linalg.norm(d), 1e-12)


def plane_hit(cam, c, k, pos, w, h):
    """(deg of the cursor about axis k in ring k's plane, its distance from the centre there, |cos| of the ray to the
    axis); deg and distance None when the ray misses the plane."""
    o, d = view_ray(cam, pos, w, h); n = AXES[k]; c = np.asarray(c, float); dn = float(d @ n)
    if abs(dn) < 1e-9: return None, None, 0.0
    X = o + d * float((c - o) @ n) / dn - c; u, v = basis(k)
    return float(np.degrees(np.arctan2(X @ v, X @ u))), float(np.linalg.norm(X)), abs(dn)


def plane_angle(cam, c, k, pos, w, h):
    a, _r, cos = plane_hit(cam, c, k, pos, w, h); return a, cos


class RingDrag:
    """One drag on ring k from pos0. angle(pos) -> deg turned since the press (unwrapped past +-180), snapped by where
    the cursor is (snap_step: free on or outside the ring, 5 deg on the tick band, 30 on the spokes; Shift at least 15).
    a0: the grabbed angle (the protractor's 0); frac: the cursor's last distance from the centre / radius.
    A ring seen face on: the cursor's angle about the centre in the ring's plane (Bambu). Nearly edge-on: the cursor's
    travel along the ring's screen tangent at the grabbed point, one ring radius of travel = 1 radian."""
    def __init__(self, cam, c, radius, k, pos0, w, h):
        self.cam, self.c, self.r, self.k, self.w, self.h = cam, np.asarray(c, float), float(radius), k, w, h
        self.p0 = np.asarray(pos0, float); a0, cos = plane_angle(cam, c, k, pos0, w, h)
        self.edge = a0 is None or cos < EDGE_ON; self.total = 0.0; self.last = a0; self.a0 = a0 or 0.0; self.frac = None
        if self.edge:
            xy, _ze, P = project_ring(cam, c, radius, k, w, h); i = int(np.argmin(np.linalg.norm(xy - self.p0, axis=1)))
            self.a0 = 360.0 * i / len(xy)
            tw = np.cross(AXES[k], P[i] - self.c); tw /= max(np.linalg.norm(tw), 1e-12)
            e = 0.05; q = cam.project(np.array([P[i], P[i] + tw * self.r * e]), w, h)
            ts = np.array([q[0][1] - q[0][0], q[1][1] - q[1][0]])     # screen px for e radians of turn
            self.px_per_deg = max(np.linalg.norm(ts), 1e-6) / np.degrees(e); self.tan = ts / max(np.linalg.norm(ts), 1e-12)

    def angle(self, pos, snap_on=False):
        p = np.asarray(pos, float)
        if self.edge: deg = float((p - self.p0) @ self.tan) / self.px_per_deg; self.frac = None     # no radius: free
        else:
            a, r, _cos = plane_hit(self.cam, self.c, self.k, p, self.w, self.h)
            if a is not None:
                self.total += (a - self.last + 180.0) % 360.0 - 180.0; self.last = a; self.frac = r / max(self.r, 1e-12)
            deg = self.total
        return snap(deg, snap_step(self.frac, snap_on))


def label(k, deg):
    """The readout by the ring while dragging, Bambu's "Y: 355.00" (0 to 360)."""
    return "%s: %.2f" % ("XYZ"[k], deg % 360.0)
