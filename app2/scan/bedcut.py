# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut below the bed: while a side records, the depth under the turntable's platter (the floor, the turntable's own
base) is zeroed before the tracker sees the frame, and before the frame is saved (a range cut, like Near / Far: the
saved frame is what was tracked). That stuff stands still while the part and the platter turn, and it dragged the
frame-to-model tracker round with it (the user: "it picks up under the bed and just takes it for a spin"). The build
cuts all of it anyway (fuse.py cull_below_plane / strip_table.py drop everything up to 1 mm above the platter plane),
so nothing the model keeps is lost; the platter's own surface and its dots stay in (they turn with the part and help
tracking).

The cut is bounded by the bed's outline: under the platter (its outline grown by RIM_MM, seen along its normal)
everything more than BELOW_MM down goes; beside it only what lies more than OUTSIDE_BELOW_MM down (the table top, the
floor), so a part hanging down past the bed's edge is kept. With no outline known (Auto remove base before the table
turned far enough) everything BELOW_MM under the plane goes, as before.

The platter is the one the session tints violet: in the camera frame, n . x + off = 0 with n towards the camera (mm),
its centre (or None) and its size (or None): a radius (a circle about the centre) or a placed outline (bed.py's
PlacedOutline: sdf(P, n, centre) > 0 outside)."""
import numpy as np

BELOW_MM = 3.0       # more than this under the platter plane, over the bed: never the part (the build cuts at +1 mm)
RIM_MM = 10.0        # the bed's outline grown by this: still over the bed
LOW_MM = 1.0         # the build's own margin over the plane (the session's model view drops that too)
OUTSIDE_BELOW_MM = 25.0     # beside the bed only this far down is cut: the table and the floor lie further down, a part
                            # hanging over the bed's edge seldom does


def _size_key(rad):
    if rad is None or not hasattr(rad, "sdf"): return None if rad is None else float(rad)
    return ("outline", rad.outline.key(), tuple(np.round(np.asarray(rad.x_axis, float), 9)))


class BedCut:
    """Which pixels of a depth frame (H x W uint16, 0.1 mm) lie below the bed. Per platter, each pixel's ray gets the
    depths past which its point lies under -BELOW_MM and under -OUTSIDE_BELOW_MM (uint16 pictures), and whether the
    ray meets the platter plane well inside the outline, well outside it, or near its rim. A frame then costs a few
    uint16 compares over the image, and exact maths only on the pixels near the rim."""
    def __init__(self, cal, w, h, depth_scale=0.1, outside_below_mm=OUTSIDE_BELOW_MM):
        fx, fy, cx, cy = cal
        self.w, self.h, self.scale = int(w), int(h), float(depth_scale)
        self.outside_below_mm=float(outside_below_mm)
        rx = (np.arange(self.w, dtype=np.float64) - cx) / fx; ry = (np.arange(self.h, dtype=np.float64) - cy) / fy
        self.RX = np.tile(rx, self.h).astype(np.float32); self.RY = np.repeat(ry, self.w).astype(np.float32)   # flat
        self.R2 = (self.RX * self.RX + self.RY * self.RY + 1.0).astype(np.float32)                             # |ray|^2
        self._geo = None; self._pre = None

    @staticmethod
    def _sd(P, nc, cen, rad):
        """Signed distance (mm, > 0 outside) of camera-frame points P to the platter's outline, along the normal."""
        if hasattr(rad, "sdf"): return rad.sdf(P, nc, cen)
        q = P - cen; qn = q @ nc
        return np.sqrt(np.maximum((q * q).sum(1) - qn * qn, 0)) - float(rad)

    def _prepare(self, nc, oc, cen, rad):
        g = (nc, oc, cen, _size_key(rad), self.outside_below_mm)
        if self._geo is not None and all(np.array_equal(a, b) if (a is not None and b is not None) else a is b
                                         for a, b in zip(self._geo[:3], g[:3])) and self._geo[3:] == g[3:]:
            return self._pre
        dn = (self.RX * nc[0] + self.RY * nc[1] + nc[2]).astype(np.float32)          # n . ray
        den = dn * np.float32(self.scale)                                            # per depth unit
        def limit(h):                                   # depth unit past which a pixel's point lies under height h
            with np.errstate(divide="ignore", invalid="ignore"):
                thr = np.where(den < 0, (h - oc) / den, np.inf)
            return np.floor(np.clip(thr, 0, 65535)).astype(np.uint16)               # d > floor(thr) <=> d > thr
        lim_below = limit(-BELOW_MM)
        lim_far = limit(-self.outside_below_mm) if self.outside_below_mm > 0 else np.full_like(lim_below,np.iinfo(np.uint16).max)
        in_f = out_f = None
        if cen is not None:
            # where the ray meets the plane, its signed distance to the outline; from -BELOW_MM to -OUTSIDE_BELOW_MM a
            # point moves sideways by at most that depth * tan(angle to the normal), so rays meeting the plane clearly
            # inside (outside) the grown outline are inside (outside) for the whole band; the rest get exact maths
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(dn < 0, -oc / dn, np.nan)
                X = np.stack([t * self.RX, t * self.RY, t], 1).astype(np.float64)
                sd = np.full(len(t), np.inf); ok = np.isfinite(t)
                sd[ok] = self._sd(X[ok], nc, cen, rad)
                band_depth=max(BELOW_MM,self.outside_below_mm) if self.outside_below_mm else BELOW_MM
                slack = band_depth * np.sqrt(np.maximum(self.R2 - dn * dn, 0)) / np.abs(dn) + 1.0
            in_f = (sd + slack < RIM_MM) & (dn < 0)
            out_f = ~((sd - slack <= RIM_MM) & (dn < 0))
        self._geo = (nc.copy(), oc, None if cen is None else cen.copy(), _size_key(rad), self.outside_below_mm)
        self._pre = (lim_below, lim_far, in_f, out_f)
        return self._pre

    def cut(self, d, geo):
        """Flat indices (into d.ravel()) of the pixels to zero, sorted. geo: (n, off, centre or None, size or None) in
        the camera frame. Nothing is cut when the platter is not well below the camera (a wrong platter never eats the
        part)."""
        nc, oc, cen, rad = geo
        nc = np.asarray(nc, np.float64); oc = float(oc)
        cen = None if (cen is None or rad is None) else np.asarray(cen, np.float64)
        if d.shape != (self.h, self.w) or oc <= 20.0 or not np.isfinite(nc).all(): return np.zeros(0, np.intp)
        lim_below, lim_far, in_f, out_f = self._prepare(nc, oc, cen, None if cen is None else rad)
        flat = d.reshape(-1)
        cut = flat > lim_below                                               # under the platter's plane
        if cen is not None:
            far = flat > lim_far                                             # far under it: the table, the floor
            band = cut & ~far & ~in_f & ~out_f                               # near the rim: where does the point lie?
            cut = far | (cut & in_f)
            bi = np.flatnonzero(band)
            if len(bi):
                z = flat[bi].astype(np.float64) * self.scale
                P = np.stack([z * self.RX[bi], z * self.RY[bi], z], 1)
                cut[bi[self._sd(P, nc, cen, rad) < RIM_MM]] = True
        return np.flatnonzero(cut)

    def apply(self, d, geo):
        """Zero the pixels below the bed in d (in place). Returns their flat indices."""
        idx = self.cut(d, geo)
        if len(idx):
            if d.flags.c_contiguous: d.reshape(-1)[idx] = 0
            else: d[np.unravel_index(idx, d.shape)] = 0
        return idx
