# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kept views around a learned bed, rebuilt from retained poses after Back/Forward."""
import hashlib
import json
from pathlib import Path
import numpy as np
from app2.scan import fmt
from app2.scan.range_guide import distance_levels, measured_profile


def _bed(side):
    sb = getattr(getattr(side, 'bed', None), 'sb', None)
    if sb is not None and callable(getattr(sb,'to_json',None)):
        return sb.to_json()
    try:
        return json.loads((Path(side.folder) / 'bed_pose.json').read_text())
    except (OSError, ValueError):
        return {}


def _distance(side, stamp, M, bed, cal):
    z = stamp.get('part_distance_mm')
    if z is not None:
        return z
    path=Path(side.cache)/(stamp['frame']+'.dph')
    try: key=(stamp['frame'],path.stat().st_mtime_ns,path.stat().st_size,tuple(np.asarray(M).ravel()),tuple(bed['n']),bed['off'])
    except OSError:return None
    cache=side.__dict__.setdefault('_turn_distance_cache',{})
    if key in cache:return cache[key]
    if cal is None:return None
    d = fmt.read_depth(side.cache, stamp['frame'])
    if d is None:
        return None
    from track import F
    from app2.scan.bedcut import LOW_MM
    T = M @ F
    n = T[:3, :3].T @ np.asarray(bed['n'])
    off = float(bed['off'] + np.asarray(bed['n']) @ T[:3, 3])
    yy, xx = np.nonzero(d)
    z = d[yy, xx] * .1
    height = z * ((xx - cal[2]) / cal[0] * n[0] + (yy - cal[3]) / cal[1] * n[1] + n[2]) + off
    z = z[height > LOW_MM]
    value=float(np.median(z)) if len(z) else None
    cache[key]=value
    return value


def snapshot(side):
    """No interpolation across skipped/lost frames. A sector needs a kept pose.

    The existing bed matcher's independent-view angle sets slice width. A ring
    means angular capture coverage, not proof that every surface was scanned.
    """
    from scanner.bed import HINT_APART_DEG
    from track import F
    bed = _bed(side)
    token = hashlib.sha256(json.dumps(dict(bed=bed, mode=getattr(side,"scan_profile",{}).get("mode"),stamps=getattr(side,"stamps",[]),
        poses=[(a, b, np.asarray(M).tolist()) for a, b, M in getattr(side,"entries",[])]), sort_keys=True).encode()).hexdigest()
    cached = getattr(side, '_turn_coverage_cache', None)
    if cached is not None and cached[0] == token:
        return cached[1]
    count = int(np.ceil(360 / HINT_APART_DEG))
    levels = np.full(count, -1, int)
    counts = np.zeros(count, int)
    out = dict(version=1, source='learned bed and kept poses', slice_deg=360 / count,
               levels=levels.tolist(), counts=counts.tolist(), degrees=None, frames=0,
               note='Show learned bed dots to measure the turn')
    if bed.get('state') != 'found' or bed.get('B_scan') is None or bed.get('n') is None:
        return out
    stamps = {s['frame']: s for s in side.stamps if s.get('ok')}
    kept = [(a, b, M) for a, b, M in side.entries if fmt.frame_name(a, b) in stamps]
    if not kept:
        return out
    from fuse import read_calib
    try:cal = read_calib(str(Path(side.folder) / 'param/Pl.bin'))
    except (OSError,ValueError):cal=None
    B = np.asarray(bed['B_scan'], float)
    phases = []
    first = None
    accepted = []
    for a, b, M in kept:
        R = (np.asarray(M) @ F)[:3, :3]
        n, x = R.T @ B[:3, 2], R.T @ B[:3, 0]
        if first is None:
            first = (n.copy(), x.copy())
        # Camera tilt does not count as a table turn. Project onto the current
        # bed plane before comparing directions.
        ref = first[1] - n * (first[1] @ n)
        norm = np.linalg.norm(ref)
        if norm <= np.finfo(float).eps:
            continue
        ref /= norm
        phase = np.degrees(np.arctan2(n @ np.cross(ref, x), ref @ x))
        phases.append(phase)
        accepted.append((a, b, M))
    if phases:
        angles = np.degrees(np.unwrap(np.radians(phases)))
        for angle, (a, b, M) in zip(angles, accepted):
            stamp = stamps[fmt.frame_name(a, b)]
            mode=stamp.get('scan_mode') or getattr(side,'scan_profile',{}).get('mode')
            profile=dict(distance_bands=stamp['capture_bands']) if stamp.get('capture_bands') else measured_profile(mode) if mode in ('near','far') else None
            z=_distance(side,stamp,M,bed,cal) if profile is not None else None
            level=-2 if z is None else int(distance_levels(np.asarray([z]),profile)[0])
            i=int(np.floor((angle % 360)/out['slice_deg'])) % count
            if counts[i]==0:levels[i]=level
            elif level>=0:levels[i]=max(levels[i],level)
            counts[i]+=1
        out.update(levels=levels.tolist(), counts=counts.tolist(), degrees=float(abs(angles[-1] - angles[0])),
                   frames=int(counts.sum()), note='Gaps need kept frames; color is the best distance band reached')
    side._turn_coverage_cache = (token, out)
    return out


def save(side):
    out = snapshot(side)
    fmt._replace(str(Path(side.folder) / 'turn-coverage.json'), json.dumps(out, indent=2) + '\n', 'w')
    return out
