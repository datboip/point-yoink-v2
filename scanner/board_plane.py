# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recover a board top from saved ringed markers with independent frame validation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

SOURCE = "saved-ringed-board-depth"
VERSION = 1


def supported(plane):
    """Trust measured support, never a source label alone."""
    try:
        p = plane['provenance']; fit = p['fit']; hold = p['heldout']
        n = np.asarray(plane['n'], float)
        return bool(plane.get('source') == SOURCE and p['version'] == VERSION and
                    n.shape == (3,) and np.isfinite(n).all() and abs(np.linalg.norm(n)-1) < 1e-5 and
                    np.isfinite(plane['d']) and fit['frame_fits'] >= 10 and
                    fit['stable_frame_share'] >= .8 and fit['inlier_share'] >= .75 and fit['rms_mm'] <= .75 and
                    -1 <= fit['dot_median_height_mm'] <= 2.5 and hold['frames'] >= 10 and
                    hold['stable_frame_share'] >= .8 and hold['supported_frame_share'] >= .8)
    except (KeyError, TypeError, ValueError):
        return False


def generic(plane):
    # Legacy Auto planes have point/tilt support, but no live frame/bed provenance.
    return not plane or ('tilt_deg' in plane and not any(k in plane for k in ('source', 'frames', 'bed_id', 'B_scan')))


def recover(side):
    import bed
    from table_plane import F
    started = time.perf_counter(); side = Path(side)
    saved = side/'table_plane.json'; old = json.loads(saved.read_text()) if saved.exists() else {}
    if supported(old): return old, {'reused': True}
    if not generic(old): return None, {'reason': 'saved plane has live or explicit provenance'}
    if not (side/'cache/pointyoink_register_pose.pose').exists() or not (side/'param/Pl.bin').exists():
        return None, {'reason': 'missing poses or calibration'}
    cal, poses = bed._side_io(str(side)); keys = sorted(poses)
    pooled=[]; dots=[]; frames=[]; raw_count=0
    def observe(key):
        d, ir = bed.read_frame(str(side), key)
        if ir is None: return None
        marks, flags = bed.detect_marks(ir, rings=True); marks = marks[flags]
        points = bed.ring_depth_points(d, ir, cal, marks, max_pts=2000)
        dp = bed.dot_depth_points(d, cal, marks); dp = dp[np.isfinite(dp).all(1)]
        T = poses[key] @ F
        hit = bed.ransac_plane(points, 1., 200, 60)
        if hit is None: return (None, None, dp @ T[:3,:3].T + T[:3,3], len(points))
        n, off, mask = hit; nw = T[:3,:3] @ n; distance = float(nw @ T[:3,3]-off)
        return (points[mask] @ T[:3,:3].T + T[:3,3], (nw, distance),
                dp @ T[:3,:3].T + T[:3,3], len(points))
    for key in keys[:30]:
        obs = observe(key)
        if obs is None: continue
        P, frame, D, count = obs; raw_count += count; dots.append(D)
        if frame is not None: pooled.append(P); frames.append(frame)
    if len(frames) < 10: return None, {'reason': 'fewer than ten ring-supported frames', 'frame_fits': len(frames)}
    P = np.concatenate(pooled); D = np.concatenate(dots)
    hit = bed.ransac_plane(P, 1., 500, 200)
    if hit is None: return None, {'reason': 'no common board plane'}
    n, off, mask = hit; distance = -float(off)
    orient = np.asarray(old.get('n', np.mean([v[:3,3] for v in poses.values()],axis=0)-P.mean(0)))
    if n @ orient < 0: n, distance = -n, -distance
    def agrees(frame):
        fn, fd = frame
        if fn @ n < 0: fn, fd = -fn, -fd
        return float(np.degrees(np.arccos(np.clip(fn @ n, -1, 1)))) < 3 and abs(fd-distance) < 2
    heights = D @ n-distance
    fit = dict(frame_fits=len(frames), stable_frame_share=float(np.mean([agrees(f) for f in frames])),
               inlier_share=float(mask.mean()), rms_mm=float(np.sqrt(np.mean((P[mask] @ n-distance)**2))),
               dot_median_height_mm=float(np.median(heights)) if len(D) else 999.)
    validation=[]; near=[]; marked=0; supported_frames=0
    for key in keys[30:60]:
        obs = observe(key)
        if obs is None: continue
        _P, frame, D, _count = obs; marked += 1
        h = D @ n-distance; near.extend(np.abs(h)<2.5); supported_frames += int((np.abs(h)<2.5).sum() >= 4)
        if frame is not None: validation.append(agrees(frame))
    hold = dict(frames=len(validation), marker_frames=marked, dots=len(near),
                stable_frame_share=float(np.mean(validation)) if validation else 0.,
                supported_frame_share=supported_frames/max(marked,1), frames_with_four_dot_depths=supported_frames,
                dot_near_plane_share=float(np.mean(near)) if near else 0.)
    provenance = dict(version=VERSION, fit=fit, heldout=hold, raw_ring_points=raw_count,
                      frame_keys=[list(k) for k in keys[:60]])
    plane=dict(n=n.tolist(), d=distance, margin=1., keep_above=True, source=SOURCE, points=int(mask.sum()), provenance=provenance)
    audit = dict(reliable=supported(plane), seconds=time.perf_counter()-started, **provenance)
    return (plane if audit['reliable'] else None), audit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('side'); ap.add_argument('out'); a=ap.parse_args()
    try: plane, audit = recover(a.side)
    except (OSError, ValueError, KeyError) as e: plane, audit = None, {'reason': str(e)}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
    if plane:
        temp=out.with_suffix('.tmp'); temp.write_text(json.dumps(plane,indent=2)); temp.replace(out)
    print('STAGE board_plane '+json.dumps(audit),flush=True)

if __name__ == '__main__': main()
