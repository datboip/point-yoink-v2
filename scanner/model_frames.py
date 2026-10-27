# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which frame the built model is in: for every side folder the build used, its placement side -> model (4x4, mm).
The build writes build/frames.json once model.ply is in place; for a model built before that file existed the same
answer is pieced together from build/placement.json, the refined placements and the folder names in build/beds.json.

Touch-up sides (scanned against the built model, Scan step's Touch up) are tracked in the model's own frame and keep
these placements in their touchup.json ("ref"), so a later build can place them without any pair fitting: whatever
frame the new model is in, T_touch = T_new[folder] @ inv(ref[folder]) for a side folder both builds placed."""
import glob, json, os
import numpy as np

TOUCH_JSON = "touchup.json"


def _load(p):
    with open(p) as f: return json.load(f)


def read(project):
    """{folder name: 4x4 side -> model (mm)} for the project's model.ply, or {} when it can't be told."""
    work = os.path.join(project, "build")
    try:
        j = _load(os.path.join(work, "frames.json"))
        return {k: np.array(v, float) for k, v in j["sides"].items()}
    except (OSError, ValueError, KeyError, TypeError): pass
    try: placed = _load(os.path.join(work, "placement.json")); order = placed["order"]
    except (OSError, ValueError, KeyError): return {}
    folders = {}
    try:
        for n, e in (_load(os.path.join(work, "beds.json")).get("sides") or {}).items(): folders[int(n) - 1] = e["folder"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError): pass
    if len(folders) < len(order):
        try:
            for n, v in _load(os.path.join(work, "sides_cache.json")).items():
                if n.startswith("side"): folders.setdefault(int(n[4:]) - 1, str(v).split("|")[1])
        except (OSError, ValueError, IndexError, AttributeError): pass
    out = {}
    refined = len(order) > 2 and all(os.path.exists(os.path.join(work, "T_refined_%d.json" % (k + 1))) for k in range(len(order)))
    for k, i in enumerate(order):
        if i not in folders: continue
        try:
            if refined: M = _load(os.path.join(work, "T_refined_%d.json" % (k + 1)))["matrix"]
            else: M = placed["T"][str(i)]
            out[folders[i]] = np.array(M, float)
        except (OSError, ValueError, KeyError): pass
    return out


def write(work, placements):
    """build/frames.next.json now (placements: {folder: 4x4}); commit() makes it frames.json once the model is written,
    so a build that fails half way never leaves a frames.json that does not match model.ply."""
    p = os.path.join(work, "frames.next.json"); tmp = p + ".tmp"
    json.dump({"version": 1, "sides": {k: np.asarray(v, float).tolist() for k, v in placements.items()}}, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def commit(work):
    p = os.path.join(work, "frames.next.json")
    if os.path.exists(p): os.replace(p, os.path.join(work, "frames.json"))


def touch_info(side_folder):
    """A touch-up side's touchup.json (dict with "ref": {folder: 4x4}), or None for an ordinary side."""
    try: j = _load(os.path.join(side_folder, TOUCH_JSON))
    except (OSError, ValueError): return None
    if not isinstance(j, dict) or j.get("frame") != "model" or not j.get("ref"): return None
    j["ref"] = {k: np.array(v, float) for k, v in j["ref"].items()}
    return j


def write_touch(side_folder, ref, model_path=None):
    """Mark a side as tracked in the model's frame (ref: the model's placements {folder: 4x4})."""
    j = {"version": 1, "frame": "model", "ref": {k: np.asarray(v, float).tolist() for k, v in ref.items()}}
    if model_path and os.path.exists(model_path):
        j["model"] = dict(name=os.path.basename(model_path), mtime=os.path.getmtime(model_path), size=os.path.getsize(model_path))
    p = os.path.join(side_folder, TOUCH_JSON); tmp = p + ".tmp"
    json.dump(j, open(tmp, "w"), indent=1); os.replace(tmp, p)


def touch_placement(info, placed, prefer=None):
    """The touch-up side -> new model placement, from its ref and the new build's placements {folder: 4x4}. prefer: the
    folder to go through first (the new base). Returns (4x4, folder used) or (None, None) when no side is in both."""
    ref = info["ref"]
    common = [f for f in ([prefer] if prefer else []) + sorted(placed) if f in ref and f in placed]
    if not common: return None, None
    f = common[0]
    return np.asarray(placed[f], float) @ np.linalg.inv(ref[f]), f
