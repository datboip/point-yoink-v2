# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conservative turntable-top evidence shared by saved cuts and live base removal."""
import glob
import json
import os
import numpy as np

# The saved no-base observer already uses this view budget. It limits fitting work,
# not the physical size or distance of accepted geometry.
OBSERVED_FRAME_BUDGET = 32


def depth_spacing(depth, cal):
    """Projected pixel spacing at the observed distance, in millimeters."""
    good = np.asarray(depth)[np.asarray(depth) > 0]
    fx, fy = np.asarray(cal, float)[:2]
    if not len(good) or not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
        return None
    return float(np.median(good) * .1 / np.sqrt(fx * fy))


def saved_spacing(scan):
    from fuse import read_calib
    import struct
    files = sorted(glob.glob(os.path.join(scan, 'cache', '*.dph')))
    if not files: return None
    try:
        depth = np.fromfile(files[0], np.uint16)
        sp = json.load(open(os.path.join(scan, 'property.rvproj'))).get('scan_param', {})
        width = int(sp.get('depth_width') or 0)
        if not width: return None
        cal = np.asarray(read_calib(os.path.join(scan, 'param', 'Pl.bin'))) * width / 800
        return depth_spacing(depth, cal)
    except (OSError, ValueError, struct.error):
        return None


def _context(points, centers, normal=None):
    P = np.asarray(points, float); C = np.asarray(centers, float)
    if C.ndim != 2 or C.shape[1] != 3 or len(C) < 3 or not np.isfinite(C).all(): return None
    axis = np.linalg.eigh((C-C.mean(0)).T @ (C-C.mean(0)))[1][:, 0]
    if (C.mean(0)-P.mean(0)) @ axis < 0: axis = -axis
    n = axis if normal is None else np.asarray(normal, float)
    if not np.isfinite(n).all() or np.linalg.norm(n) == 0: return None
    n = n / np.linalg.norm(n)
    u = np.cross(n, np.eye(3)[np.argmin(np.abs(n))]); u /= np.linalg.norm(u)
    v = np.cross(n, u); xy = np.column_stack((C@u, C@v))
    fit, _, rank, _ = np.linalg.lstsq(np.column_stack((xy, np.ones(len(C)))), (xy*xy).sum(1), rcond=None)
    if rank != 3: return None
    center = fit[:2]/2; rc = np.linalg.norm(xy-center, axis=1)
    angles = np.unwrap(np.arctan2(xy[:,1]-center[1], xy[:,0]-center[0]))
    return n, axis, u, v, center, float(np.std(rc)), float(np.degrees(np.ptp(angles)))


def verify_top(points, centers, normal, d, spacing, margin=1., max_tilt=8.):
    """Certify a coaxial round rim with stand below and a distinct nonround part above.

    A circular flange alone is insufficient. Uncertain evidence keeps the existing cut guard.
    Sampling and camera-orbit residuals determine the geometry error allowance.
    """
    from scanner.table_plane import SidePlane
    arc_needed = SidePlane.disc.__defaults__[0]
    P = np.asarray(points, float)
    if spacing is None or not np.isfinite(spacing) or spacing <= 0 or len(P) == 0:
        return dict(verified=False, reason='observed sampling is unknown')
    context = _context(P, centers, normal)
    if context is None: return dict(verified=False, reason='spin center is unknown')
    n, axis, u, v, center, orbit_error, arc = context
    tilt = float(np.degrees(np.arccos(np.clip(n@axis, -1, 1))))
    if arc < arc_needed or tilt > max_tilt:
        return dict(verified=False, reason='spin support is insufficient', arc_deg=arc, tilt_deg=tilt)
    xy = np.column_stack((P@u, P@v))-center; radial = np.linalg.norm(xy, axis=1)
    height = P@n-float(d); band = float(margin+spacing)
    near = np.abs(height) <= band; above = height > band; below = height < -band
    if not near.any() or not above.any() or not below.any():
        return dict(verified=False, reason='rim, upper part and lower stand are not separated')
    radius_guess = float(np.percentile(radial[near], 98))
    if radius_guess <= spacing: return dict(verified=False, reason='rim is unresolved')
    sectors = max(3, int(np.ceil(2*np.pi*radius_guess/spacing)))
    angle = np.arctan2(xy[:,1], xy[:,0]); cells = np.floor((angle+np.pi)*sectors/(2*np.pi)).astype(int)%sectors
    def profile(mask):
        outer = np.full(sectors, -np.inf); np.maximum.at(outer, cells[mask], radial[mask])
        good = np.isfinite(outer); angles = (np.flatnonzero(good)+.5)*2*np.pi/sectors-np.pi
        return np.column_stack((np.cos(angles), np.sin(angles)))*outer[good,None]
    from scipy.optimize import least_squares
    def models(xy):
        if len(xy) <= 5: return None  # Five rectangle parameters need at least one residual.
        r = float(np.median(np.linalg.norm(xy, axis=1)))
        def circle_res(p): return np.linalg.norm(xy-p[:2],axis=1)-p[2]
        circle = least_squares(circle_res, [0.,0.,r],loss='soft_l1',f_scale=spacing)
        def rectangle_res(p):
            c = np.cos(p[4]); z = np.sin(p[4]); rotation = np.array([[c,-z],[z,c]])
            q = np.abs((xy-p[:2])@rotation)-np.abs(p[2:4])
            return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
        covariance = np.linalg.eigh((xy-xy.mean(0)).T@(xy-xy.mean(0)))[1][:,-1]
        phi = float(np.arctan2(covariance[1],covariance[0]))
        c,z=np.cos(phi),np.sin(phi); rotation=np.array([[c,-z],[z,c]])
        width=np.percentile(np.abs(xy@rotation),98,axis=0)
        rectangle=least_squares(rectangle_res,[0.,0.,*width,phi],loss='soft_l1',f_scale=spacing)
        def score(fit,k): return len(xy)*np.log(max(2*fit.cost/len(xy),np.finfo(float).tiny))+k*np.log(len(xy))
        residual=circle_res(circle.x);sigma=float(1.4826*np.median(np.abs(residual-np.median(residual))))
        return dict(round=bool(score(circle,3)<score(rectangle,5)),radius_mm=float(circle.x[2]),
                    center_error_mm=float(np.linalg.norm(circle.x[:2])),circle_error_mm=sigma,
                    circle_score=float(score(circle,3)),rectangle_score=float(score(rectangle,5)))
    rim_xy=profile(near);upper_xy=profile(above);lower_xy=profile(below)
    rim=models(rim_xy);upper=models(upper_xy);lower=models(lower_xy)
    if rim is None or upper is None or lower is None:return dict(verified=False,reason='surface support is missing')
    # A 95% empirical error envelope includes observed rim noise and projected sampling.
    from statistics import NormalDist
    confidence=NormalDist().inv_cdf(.975)
    rim_allowance=band+orbit_error+confidence*rim['circle_error_mm']
    lower_allowance=band+orbit_error+confidence*lower['circle_error_mm']
    support_arc=float(len(rim_xy)*360/sectors)
    verified=(rim['round'] and lower['round'] and not upper['round'] and support_arc>=arc_needed
              and rim['center_error_mm']<=rim_allowance and lower['center_error_mm']<=lower_allowance)
    return dict(verified=bool(verified),reason='coaxial rim, stand and distinct upper part' if verified else 'round support is ambiguous',
                radius_mm=rim['radius_mm'],rim_error_mm=rim['circle_error_mm'],error_allowance_mm=rim_allowance,
                center_error_mm=rim['center_error_mm'],rim_model=rim,upper_model=upper,lower_model=lower,
                support_arc_deg=support_arc,arc_deg=arc,tilt_deg=tilt,sample_spacing_mm=float(spacing),band_mm=band,
                rim_points=int(near.sum()),upper_points=int(above.sum()),lower_points=int(below.sum()))


def find_top(points, centers, spacing, margin=1., max_tilt=8.):
    """Search observed density peaks for the highest supported turntable top."""
    from scanner.table_plane import SidePlane, fit_plane
    P = np.asarray(points, float)
    context = _context(P, centers)
    if context is None or spacing is None or spacing <= 0: return None
    axis = context[1]; h = P@axis
    edges = np.arange(h.min()-spacing, h.max()+2*spacing, spacing)
    hist, _ = np.histogram(h, edges)
    share = SidePlane.plane.__defaults__[2]
    peaks = np.flatnonzero((hist >= np.r_[0,hist[:-1]]) & (hist >= np.r_[hist[1:],0]) & (hist >= share*hist.max()))
    for i in peaks[::-1]:
        level = (edges[i]+edges[i+1])/2; near = P[np.abs(h-level) <= margin+spacing]
        if not len(near): continue
        n, c = fit_plane(near)
        if n@axis < 0: n = -n
        d = float(n@c)
        evidence = verify_top(P, centers, n, d, spacing, margin, max_tilt)
        if evidence['verified']: return n, -d, len(near), evidence
    return None
