#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto remove base: find the turntable platter in depth frames while scanning, so the build never has to hunt for it.

On a turntable the camera circles the spin axis (in the scan's frame), and the platter is the flat layer square to that
axis with the part standing on it. Every tracked frame adds a sample of its points; the platter is the lowest sharp,
well-filled layer along the axis. A dark platter gives depth only on its dots and that is enough, because all frames
add up. (The first version looked for the biggest plane in each frame on its own: on a dark platter that was a flat
face of the part far more often than the platter. Seen live.) Tracking still sees the platter (its dots help on flat
parts); only the build cuts it, through the same plane.json that strip_table.py writes for fuse.py.
usage: table_plane.py <scan dir> [--out plane.json] [--every N]     (replays a saved side and prints its plane)"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

F = np.diag([1.0, -1.0, -1.0, 1.0])       # OpenCV camera <-> scanner camera frame (its own inverse)


def fit_plane(P):
    """Least-squares plane through points P (N x 3): unit normal n and centre c. Works from the 3 x 3 scatter matrix, so
    the cost does not grow with N squared (np.linalg.svd on the points themselves builds an N x N matrix by default)."""
    c = P.mean(0); Q = P - c
    return np.linalg.eigh(Q.T @ Q)[1][:, 0], c


def frame_points(d, cal, T_mm, count=1500, depth_scale=0.1, rng=None):
    """A random sample of one depth frame's points in the scan's frame (mm). T_mm: OpenCV camera -> scan frame."""
    v, u = np.nonzero(d)
    if len(u) < 200: return np.zeros((0, 3), np.float32)
    k = (rng or np.random.default_rng(len(u))).choice(len(u), min(count, len(u)), replace=False)
    u, v = u[k], v[k]; z = d[v, u].astype(np.float64) * depth_scale
    P = np.stack([(u - cal[2]) * z / cal[0], (v - cal[3]) * z / cal[1], z], 1)
    return (P @ T_mm[:3, :3].T + T_mm[:3, 3]).astype(np.float32)


def to_camera(n, off, T_mm):
    """A scan-frame plane (n . x + off = 0) as seen from a camera at pose T_mm: (n_c, off_c) for camera-frame points."""
    return T_mm[:3, :3].T @ n, float(off + n @ T_mm[:3, 3])


class SidePlane:
    """Collects a side's tracked frames; plane() is the platter under the part, or None when there is none to see."""
    def __init__(self): self.pts = []; self.centres = []; self._cache = (None, None); self._sampling_mm = None; self._top_support = None; self._disc_cache = None

    def add(self, d, cal, T_mm):
        """One tracked frame. d: H x W uint16 depth, T_mm: OpenCV camera -> scan frame with translation in mm."""
        self.centres.append(T_mm[:3, 3].copy()); self.pts.append(frame_points(d, cal, T_mm))
        from scanner.base_geometry import depth_spacing
        spacing = depth_spacing(d, cal)
        if spacing is not None:
            self._sampling_mm = max(spacing, self._sampling_mm or spacing)

    def spin_axis(self, min_arc_mm=15.0):
        """The turntable's axis: the camera centres circle it, so it is the normal of the plane they move in, pointing
        from the part to the camera. None until the part has turned far enough for the centres to trace an arc rather
        than a dot or a line."""
        if len(self.centres) < 8: return None
        C = np.array(self.centres); Q = C - C.mean(0)
        w, v = np.linalg.eigh(Q.T @ Q / len(C))
        if np.sqrt(max(w[1], 0)) < min_arc_mm: return None
        axis = v[:, 0]; P = np.concatenate(self.pts)
        if len(P) and (C.mean(0) - P.mean(0)) @ axis < 0: axis = -axis
        return axis

    def turned_deg(self):
        """How far the part has turned in front of the camera since the side began, in degrees (a full turn = 360), or
        None until the spin axis is known. The camera centres lie on a circle about the axis; this is the angle between
        the first and the newest one along it."""
        axis = self.spin_axis()
        if axis is None: return None
        C = np.array(self.centres); Q = C - C.mean(0)
        u = np.cross(axis, [1.0, 0, 0])
        if np.linalg.norm(u) < 0.1: u = np.cross(axis, [0, 1.0, 0])
        u /= np.linalg.norm(u); v = np.cross(axis, u)
        x, y = Q @ u, Q @ v
        cx, cy, _ = np.linalg.lstsq(np.stack([x, y, np.ones(len(x))], 1), x * x + y * y, rcond=None)[0]      # circle fit
        ang = np.unwrap(np.arctan2(y - cy / 2, x - cx / 2))
        return float(np.degrees(abs(ang[-1] - ang[0])))

    def plane(self, min_frames=8, max_tilt_deg=8.0, share=0.3, sharp=4.0):
        """(n, off, points on the platter, tilt from the spin axis in degrees) in the scan's frame, n towards the
        camera; or None. The platter is the lowest layer along the spin axis that is well filled (at least `share` of
        the fullest layer) and sharp (`sharp` times fuller than the layers just above and below it: a flat table, not
        a slice through a curved part), with next to nothing beneath it."""
        if self._cache[0] == len(self.pts): return self._cache[1]
        result = None
        self._disc_cache = None
        self._top_support = None
        if len(self.pts) >= min_frames and self.spin_axis() is not None:
            from scanner.base_geometry import find_top, OBSERVED_FRAME_BUDGET
            indices = np.linspace(0, len(self.pts)-1, min(OBSERVED_FRAME_BUDGET, len(self.pts))).round().astype(int)
            observed = np.concatenate([self.pts[i] for i in indices])
            support = find_top(observed, self.centres, self._sampling_mm, max_tilt=max_tilt_deg)
            if support is not None:
                n, off, count, evidence = support
                self._top_support = evidence
                result = n, off, count, evidence['tilt_deg']
        if result is None:
            result = self._plane(min_frames, max_tilt_deg, share, sharp)
        self._cache = (len(self.pts), result)
        return self._cache[1]

    def snapshot(self):
        """Immutable tracked-point blocks for the existing background finder."""
        copy = SidePlane()
        copy.pts = list(self.pts)
        copy.centres = list(self.centres)
        copy._sampling_mm = self._sampling_mm
        return copy

    def install_snapshot(self, snapshot):
        """Accept a completed finder result unless newer input was already measured."""
        count = snapshot._cache[0]
        if count is None or count > len(self.pts) or (self._cache[0] or 0) > count:
            return False
        self._cache = snapshot._cache
        self._top_support = snapshot._top_support
        self._disc_cache = snapshot._disc_cache
        return True

    def cached_plane(self):
        return self._cache[1] if (self._cache[0] or 0) <= len(self.pts) else None

    def cached_disc(self):
        return self._disc_cache if self.cached_plane() is not None else None

    def cached_certified_plane(self):
        evidence = getattr(self, '_top_support', None)
        return self.cached_plane() if evidence and evidence.get('verified') else None

    def legacy_plane(self):
        """The old inexpensive fallback when the finder adapter is unavailable."""
        if self._cache[0] != len(self.pts):
            self._top_support = None
            self._disc_cache = None
            self._cache = (len(self.pts), self._plane(*SidePlane.plane.__defaults__))
        return self._cache[1]

    def certified_plane(self):
        """The supported turntable top only; legacy floor estimates never gain this trust."""
        plane = self.plane()
        evidence = getattr(self, '_top_support', None)
        return plane if evidence and evidence.get('verified') else None

    def _plane(self, min_frames, max_tilt_deg, share, sharp):
        axis = self.spin_axis()
        if axis is None or len(self.pts) < min_frames: return None
        P = np.concatenate(self.pts).astype(np.float64)
        if len(P) < 5000: return None
        h = P @ axis; lo = np.floor(h.min()) - 6
        hist = np.histogram(h, bins=np.arange(lo, h.max() + 7, 1.0))[0].astype(float)
        s = hist + np.roll(hist, -1)                             # 2 mm layers: the platter may straddle two 1 mm bins
        for i in np.nonzero(s >= share * s.max())[0]:            # lowest first
            if i < 5 or i + 7 > len(s): continue
            # right above the platter there is a gap before the part's own surface fills up again (a part lying flat
            # on it starts a few mm higher), so the emptiest of the next layers is what it must stand out from
            around = max(hist[i + 2:i + 5].min(), hist[i - 5:i - 2].mean(), 1.0)
            if s[i] / 2 < sharp * around: continue
            if hist[:i - 2].sum() > 0.05 * hist.sum(): return None      # a lot lies beneath it: not the table
            sel = P[np.abs(h - (lo + i + 1.0)) < 2.0]
            n, c = fit_plane(sel)
            if n @ axis < 0: n = -n
            tilt = float(np.degrees(np.arccos(min(1.0, n @ axis))))
            if tilt > max_tilt_deg: return None
            return n, float(-n @ c), len(sel), tilt
        return None

    def disc(self, min_arc_deg=60.0):
        """The platter's centre (where the spin axis meets it) and radius (as far as depth was seen on it), or None.
        The centre is the middle of the circle the camera centres trace in the platter's plane (a circle fit, as in
        turned_deg), so None until the table has turned min_arc_deg. It used to be the mean of the camera centres, which
        is the circle's middle only after a full turn: part way round it sat off to the side and the dashed rim was drawn
        across the part (seen live in the Qt window)."""
        p = self.plane()
        if p is None or len(self.centres) < 8: return None
        n, off = p[0], p[1]; C = np.array(self.centres); m = C.mean(0); Q = C - m
        u = np.cross(n, [1.0, 0, 0])
        if np.linalg.norm(u) < 0.1: u = np.cross(n, [0, 1.0, 0])
        u /= np.linalg.norm(u); v = np.cross(n, u)
        x, y = Q @ u, Q @ v
        a, b, _ = np.linalg.lstsq(np.stack([x, y, np.ones(len(x))], 1), x * x + y * y, rcond=None)[0]     # circle fit
        ang = np.unwrap(np.arctan2(y - b / 2, x - a / 2))
        if np.degrees(ang.max() - ang.min()) < min_arc_deg: return None
        C0 = m + (a / 2) * u + (b / 2) * v
        centre = C0 - (C0 @ n + off) * n
        P = np.concatenate(self.pts).astype(np.float64); on = P[np.abs(P @ n + off) < 2.0]
        self._disc_cache = centre, float(np.percentile(np.linalg.norm(on - centre, axis=1), 98))
        return self._disc_cache

    def save(self, path, margin=1.0):
        """Write fuse.py's plane.json (it keeps points with x . n >= d + margin). Returns whether a plane was found."""
        p = self.plane()
        if p is None: return False
        tmp = path + ".tmp.%d" % os.getpid()
        json.dump({"n": p[0].tolist(), "d": float(-p[1]), "margin": margin, "keep_above": True,
                   "points": p[2], "tilt_deg": round(p[3], 3)}, open(tmp, "w"))
        os.replace(tmp, path); return True


def main():
    from fuse import read_calib, read_global_poses, frame_key
    ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=600)
    a = ap.parse_args()
    cache = os.path.join(a.scan, "cache")
    cal = read_calib(os.path.join(a.scan, "param", "Pl.bin"))
    if a.width != 800: cal = tuple(c * a.width / 800.0 for c in cal)
    gp = read_global_poses(os.path.join(cache, "pointyoink_register_pose.pose"))
    sp = SidePlane()
    for p in sorted(glob.glob(os.path.join(cache, "*.dph")))[::max(1, a.every)]:
        M = gp.get(frame_key(p))
        if M is None: continue
        d = np.frombuffer(open(p, "rb").read(), np.uint16)
        if d.size != a.width * a.height: continue
        sp.add(d.reshape(a.height, a.width), cal, M @ F)        # scanner camera frame -> OpenCV camera frame, still mm
    r = sp.plane()
    if r is None: print("no platter found in %d frames" % len(sp.pts)); return 1
    print("platter: n = %s, d = %.2f, %d points on it from %d frames, %.2f deg off the spin axis, radius %.0f mm"
          % (np.round(r[0], 4).tolist(), -r[1], r[2], len(sp.pts), r[3], (sp.disc() or (None, float("nan")))[1]))
    if a.out: sp.save(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
