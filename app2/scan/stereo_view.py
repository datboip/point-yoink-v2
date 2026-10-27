# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rectified IR geometry from this scanner's Q and rectification maps.

The RGB transform is never used as the stereo baseline. Unsupported maps leave
the raw right picture available, with its geometric marks omitted.
"""
from pathlib import Path
import numpy as np
import cv2
from collections import deque


class Stereo:
    def __init__(self, cal, baseline, right_cx, refine=False):
        self.cal = tuple(cal)
        self.baseline = float(baseline)
        self.right_cx = float(right_cx)
        self.shifts = deque()
        self.alignment_stats = None
        self.refine_enabled = bool(refine)

    @classmethod
    def load(cls, folder, cal):
        try:
            q = np.fromfile(Path(folder) / 'Q.bin', '<f4').reshape(4, 4)
            for name in ('mapparamL.bin', 'mapparamR.bin'):
                raw = (Path(folder) / name).read_bytes()
                if len(raw) < 48 or not np.allclose(np.frombuffer(raw[12:48], '<f4').reshape(3, 3), np.eye(3), atol=.01):
                    return None
            if not np.isfinite(q).all() or not np.allclose(q[:2, :2], np.eye(2), atol=1e-5) or abs(q[3, 2]) < 1e-8:
                return None
            from fuse import read_calib
            native = read_calib(str(Path(folder) / 'Pl.bin'))
            baseline = 1. / float(q[3, 2])
            # Q uses native pixels; the live calibration may be scaled.
            right_cx = cal[2] + float(q[3, 3] / q[3, 2]) * cal[0] / native[0]
            if not np.isfinite([baseline, right_cx]).all():
                return None
            from app2.scan import settings
            return cls(cal, baseline, right_cx,refine=bool(settings.load().get('ir_refine',False)))
        except (OSError, ValueError, IndexError):
            return None

    def right(self, xy, z):
        """Left pixels -> right pixels at measured depth, preserving disparity."""
        out = np.array(xy, dtype=float, copy=True)
        out[..., 0] += self.right_cx - self.cal[2] - self.cal[0] * self.baseline / np.asarray(z)
        return out

    def left(self, xy, z):
        out = np.array(xy, dtype=float, copy=True)
        out[..., 0] -= self.right_cx - self.cal[2] - self.cal[0] * self.baseline / np.asarray(z)
        return out

    def blend(self, left, right, depth, z):
        """Two tints only where measured support and appearance agree.

        Small holes borrow local measured depth. A bounded, robust dot shift
        can correct a residual image offset after three agreeing observations.
        Unsupported, occluded, and unmatched areas remain the left image.
        """
        yy, xx = np.indices(left.shape, dtype=np.float32)
        distances = np.where(depth > 0, depth * .1, fill_holes(depth, np.nan))
        supported = np.isfinite(distances) & (distances > 0)
        with np.errstate(divide='ignore',invalid='ignore'):
            xr = xx + self.right_cx - self.cal[2] - self.cal[0] * self.baseline / distances
        rr = cv2.remap(right, xr.astype(np.float32), yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        shift = self.refine(left, rr, supported) if self.refine_enabled else None
        if shift is not None:
            # Shift the sampling coordinates, preserving per-pixel disparity.
            rr = cv2.remap(right, (xr+shift[0]).astype(np.float32), yy+np.float32(shift[1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = supported & (xr >= 0) & (xr < right.shape[1]-1)
        # A one-camera-only edge cannot be aligned by any warp. Reject it
        # rather than producing a second colored contour. This robust
        # photometric check also rejects unequal illumination and disocclusion.
        delta = np.abs(left.astype(float)-rr.astype(float))
        q = delta[valid]
        if len(q):
            median = np.median(q)
            sigma = 1.482602218505602*np.median(np.abs(q-median))
            tolerance = max(1., median+3*sigma)
            agree = delta <= tolerance
            # Include each mismatched edge's interpolation neighborhood.
            agree = ~cv2.dilate((~agree).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        else:agree = np.zeros(left.shape, bool)
        # Matched pixels would look exactly like Left, so Both colors what it found instead: blue where the two
        # cameras agree, orange where both see the spot but disagree (glare, an edge one camera sees past).
        # Plain gray: only the left camera can be placed there.
        out = cv2.cvtColor(left,cv2.COLOR_GRAY2RGB)
        mean = ((left.astype(np.uint16)+rr)//2)
        for mask,(r,g,b) in ((valid&agree,BOTH_AGREE),(valid&~agree,BOTH_DIFFER)):
            m = mean[mask][:,None]*np.array((6,6,6),np.uint16)//10
            out[mask] = (m+np.array((r,g,b),np.uint16)).clip(0,255).astype(np.uint8)
        return out

    def refine(self, left, warped, supported):
        """A residual translation bounded by the observed dot radius."""
        from scanner.bed import detect_dots, HIT_MIN, ACCEPT_HITS
        a=detect_dots(left);b=detect_dots(warped)
        self.alignment_stats=None
        if len(a)<HIT_MIN or len(b)<HIT_MIN:self.shifts.clear();return None
        dist=np.linalg.norm(a[:,None,:2]-b[None,:,:2],axis=2)
        j=dist.argmin(1);mutual=dist.argmin(0)[j]==np.arange(len(a))
        pix=np.rint(a[:,:2]).astype(int)
        inside=(pix[:,0]>=0)&(pix[:,0]<left.shape[1])&(pix[:,1]>=0)&(pix[:,1]<left.shape[0])
        support=np.zeros(len(a),bool);support[inside]=supported[pix[inside,1],pix[inside,0]]
        keep=mutual&support&(dist[np.arange(len(a)),j]<=a[:,2])
        if keep.sum()<HIT_MIN:self.shifts.clear();return None
        delta=b[j[keep],:2]-a[keep,:2]
        median=np.median(delta,axis=0)
        residual=np.linalg.norm(delta-median,axis=1)
        bound=float(np.median(a[keep,2]))
        sigma=1.482602218505602*np.median(np.abs(residual-np.median(residual)))
        use=residual<=max(1.,np.median(residual)+3*sigma)
        if use.sum()<HIT_MIN or np.linalg.norm(median)>bound:self.shifts.clear();return None
        delta=delta[use];median=np.median(delta,axis=0)
        if self.shifts and np.linalg.norm(median-np.median(self.shifts,axis=0))>bound:self.shifts.clear()
        self.shifts.append(median)
        while len(self.shifts)>ACCEPT_HITS:self.shifts.popleft()
        if len(self.shifts)<ACCEPT_HITS:return None
        shift=np.median(self.shifts,axis=0)
        self.alignment_stats=dict(matches=len(delta),before_px=np.linalg.norm(delta,axis=1).tolist(),after_px=np.linalg.norm(delta-shift,axis=1).tolist(),shift_px=shift.tolist(),bound_px=bound)
        return shift


BOTH_AGREE = (0, 40, 110)    # added to 60% of the mean: blue, the two IR cameras agree
BOTH_DIFFER = (120, 50, 0)   # orange: both see it but the pictures differ

HOLE_SPAN = 1 / 40  # share of image width; a bit wider than a bed marker dot
HOLE_SUPPORT = .6  # share of that window that must have depth


def right_depth_mask(stereo, depth):
    """Pixels of the RIGHT picture that have depth: each left depth pixel moved by its own disparity."""
    v, u = np.nonzero(depth > 0)
    out = np.zeros(depth.shape, bool)
    if not len(v): return out
    x = np.rint(stereo.right(np.column_stack([u, v]).astype(float), depth[v, u] * .1)[:, 0]).astype(int)
    ok = (x >= 0) & (x < depth.shape[1])
    out[v[ok], x[ok]] = True
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def fill_holes(depth, z):
    """Distance for pixels without depth, from nearby measured depth.

    Shiny marker dots return no depth; warping them at the part distance splits
    each dot in two. Far from any depth, fall back to the part distance.
    """
    valid = (depth > 0).astype(np.float32)
    k = max(3, int(depth.shape[1] * HOLE_SPAN) | 1)
    total = cv2.boxFilter(depth.astype(np.float32) * .1, -1, (k, k), normalize=False)
    count = cv2.boxFilter(valid, -1, (k, k), normalize=False)
    with np.errstate(invalid='ignore', divide='ignore'):
        near = total / count
    # only holes mostly ringed by depth (a marker dot); wide gaps and part edges keep the part distance, a mean
    # across them smears the right picture
    return np.where(count >= k * k * HOLE_SUPPORT, near, z)


def select_ir(frame, mode):
    """Return image and marks for a selected camera; never substitute left for right."""
    if mode == 'Left':
        return frame.ir, getattr(frame, 'ir_guides', None)
    image = getattr(frame, 'ir_right' if mode == 'Right' else 'ir_both', None)
    guides = getattr(frame, 'right_guides' if mode == 'Right' else 'ir_guides', None)
    return image, guides
