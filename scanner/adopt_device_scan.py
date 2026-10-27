#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare copied scanner depth frames for PointYoink Build without re-tracking them.

The scanner's per-frame .inf poses are checked for every depth frame. When a
global registration table exists, its poses are used for Build because resumed
passes can be offset from the per-frame odometry. A side with an absent or bad
pose is left untouched until the user chooses how to handle those frames.
"""
import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

import numpy as np


FRAME = re.compile(r"frame_(\d+)_(\d+)\.dph$")
TRACK = "pointyoink_track.json"
POSES = "pointyoink_register_pose.pose"
MODELS_ONLY = ".pointyoink-models-only"


def _matrix(values):
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-5) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.05) or
            not 0.8 < np.linalg.det(rotation) < 1.2 or
            np.max(np.abs(matrix[:3, 3])) > 1e7):
        raise ValueError("invalid pose matrix")
    return matrix


def _inf_pose(path):
    with open(path, "rb") as file:
        raw = file.read(16 + 16 * 8)
    if len(raw) < 16 + 16 * 8:
        raise ValueError("short .inf pose")
    return _matrix(struct.unpack_from("<16d", raw, 16))


def _global_poses(path):
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError("short global pose table")
    count = struct.unpack_from("<i", raw)[0]
    if count < 0 or count > 200_000 or len(raw) < 4 + count * 136:
        raise ValueError("incomplete global pose table")
    poses = {}
    for index in range(count):
        offset = 4 + index * 136
        key = struct.unpack_from("<ii", raw, offset)
        if key in poses:
            raise ValueError("duplicate frame in global pose table")
        poses[key] = _matrix(struct.unpack_from("<16d", raw, offset + 8))
    return poses


def needs_preparation(project):
    """A full imported project has raw sides not yet adopted by PointYoink."""
    project = Path(project)
    if (project / MODELS_ONLY).exists():
        return False
    data = project / "data"
    try:
        return any(not (side / "cache" / TRACK).exists() and any((side / "cache").glob("*.dph"))
                   for side in data.iterdir() if side.is_dir() and not side.is_symlink())
    except OSError:
        return False


def issue_summary(report):
    """A short user-facing explanation when adoption needs a pose decision."""
    affected = [side["name"] for side in report["sides"] if side["lost"]]
    names = ", ".join(affected[:3]) + (" and others" if len(affected) > 3 else "")
    count = report["lost"]
    return ("%d frame%s in %s %s %s. Preparation paused; raw frames were not changed."
            % (count, "s" if count != 1 else "", names, "have" if count != 1 else "has",
               "missing or invalid device poses" if count != 1 else "a missing or invalid device pose"))


def _side_plan(side):
    cache = side / "cache"
    depth = sorted(cache.glob("*.dph"))
    if not depth:
        return None
    issues = []
    global_path = cache / "global_register_pose.pose"
    global_poses = None
    global_error = None
    if global_path.is_symlink():
        global_error = "global pose table is a link"
    elif global_path.is_file():
        try:
            global_poses = _global_poses(global_path)
        except (OSError, ValueError, struct.error) as exc:
            global_error = str(exc)
    entries = []
    stamps = []
    seen = set()
    for file in depth:
        match = FRAME.fullmatch(file.name)
        name = file.stem
        if not match:
            issues.append(dict(side=side.name, frame=name, reason="unrecognized depth frame name"))
            continue
        key = tuple(map(int, match.groups()))
        if key in seen:
            issues.append(dict(side=side.name, frame=name, reason="duplicate frame number"))
            continue
        seen.add(key)
        try:
            if file.is_symlink() or file.with_suffix(".inf").is_symlink():
                raise ValueError("frame file is a link")
            pose = _inf_pose(file.with_suffix(".inf"))
            if global_error:
                raise ValueError(global_error)
            if global_poses is not None:
                if key not in global_poses:
                    raise ValueError("no global pose for this frame")
                pose = global_poses[key]
        except FileNotFoundError:
            issues.append(dict(side=side.name, frame=name, reason="missing .inf pose"))
            continue
        except (OSError, ValueError, struct.error) as exc:
            issues.append(dict(side=side.name, frame=name, reason=str(exc)))
            continue
        entries.append((key[0], key[1], pose))
        stamps.append(dict(frame=name, t=file.stat().st_mtime, ok=True, fit=None, rmse=None, reloc=False,
                           near=0, far=6554, led=False, gain=0, hinted=False))
    return dict(side=side.name, frames=len(depth), entries=entries, stamps=stamps, issues=issues, cache=cache)


def _replace(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".pointyoink-adopt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
            file.flush(); os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def adopt_project(project):
    """Return a report. Any pose issue prevents writes to the whole project."""
    project = Path(project)
    if (project / MODELS_ONLY).exists():
        return dict(status="models_only", frames=0, tracked=0, lost=0, sides=[], issues=[])
    data = project / "data"
    if not data.is_dir() or data.is_symlink():
        raise ValueError("Choose a project with a data folder.")
    plans = []
    for side in sorted(data.iterdir()):
        if not side.is_dir():
            continue
        if side.is_symlink() or (side / "cache").is_symlink():
            raise ValueError("An imported scan side contains a link.")
        cache = side / "cache"
        if not cache.is_dir() or (cache / TRACK).exists():
            continue
        plan = _side_plan(side)
        if plan is not None:
            plans.append(plan)
    issues = [issue for plan in plans for issue in plan["issues"]]
    report = dict(status="needs_decision" if issues else "prepared" if plans else "nothing_to_prepare",
                  frames=sum(plan["frames"] for plan in plans), tracked=sum(len(plan["entries"]) for plan in plans),
                  lost=len(issues), sides=[dict(name=plan["side"], frames=plan["frames"],
                                         tracked=len(plan["entries"]), lost=len(plan["issues"])) for plan in plans],
                  issues=issues)
    if issues:
        return report
    from app2.scan.fmt import pose_table_bytes
    for plan in plans:
        cache = plan["cache"]
        if (cache / TRACK).exists():
            continue
        _replace(cache / POSES, pose_table_bytes(plan["entries"]))
        record = dict(version=1, live=False, frames=plan["frames"], tracked=len(plan["entries"]),
                      lost=[], stamps=plan["stamps"])
        _replace(cache / TRACK, json.dumps(record, indent=1).encode("utf-8"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="a copied scanner project on this PC")
    args = parser.parse_args(argv)
    try:
        report = adopt_project(args.project)
    except (OSError, ValueError) as exc:
        parser.exit(2, "Could not prepare imported scan: %s\n" % exc)
    print(json.dumps(report, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
