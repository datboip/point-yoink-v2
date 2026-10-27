# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live-scan project on disk, byte for byte as the Tk scanner (scanner/pointyoink_scanner.py) writes it, so
build_project.py, fuse.py, table_plane.py and the Tk scanner all read projects made by 2.0 unchanged.

  <projects dir>/ProjectLive<%m%d%Y%H%M%S>/
    data/<%m%d%Y%H%M%S>/                    one folder per side; sorted by name = side order (the build sorts the same way)
      param/                                copy of the calibration folder (Pl.bin is what everything reads)
      property.rvproj                       {"scan_param": {...}}, json with indent 1
      cache/frame_PPP_NNNN.dph              depth, u16 800x600, 0.1 mm, the Near/Far cut already applied
      cache/frame_PPP_NNNN.irl              IR left, u8 800x600
      cache/frame_PPP_NNNN.irr              IR right, u8 800x600, when supplied by the source
      cache/frame_PPP_NNNN.inf              16 zero bytes, pose 16 <d, identity 16 <d (272 bytes); every recorded frame,
                                            a lost frame gets the last good pose. Pose: scanner camera -> scan frame, mm
      cache/pointyoink_register_pose.pose   tracked frames only: <i count, then per frame <ii pass, frame + 16 <d (mm)
      cache/pointyoink_track.json           version, live, frames, tracked, lost (frame names), stamps (one per frame)
      thumb_rgb.jpg                         colour camera at the side's first recorded frame, 320x240, JPEG quality 85
      thumb_model.png                       the side's model from its last tracked pose, 400x300
      table_plane.json                      the platter for the build to cut; absent when none was found
      bed_pose.json                         what the table dots saw (only through the bed hook)
      discarded/frame_PPP_NNNN.*            frames taken back while paused (Back 5 / 10 / 30); in no table, never built
    bed.json                                the bed this project was scanned with (only through the bed hook)
    model.ply, model.png, model_needs.*, build/        written by the build

PPP counts the passes of a side (every Start or Continue is a new pass, from 000), NNNN the frames in that pass.
JSON, the pose table and the thumbs go through a temp file and os.replace: the bytes are the Tk scanner's, a crash just
can't leave half a file. Frames are plain writes, as in the Tk scanner."""
import json, os, shutil, struct, time
import numpy as np

W, H = 800, 600
TRACK_JSON = "pointyoink_track.json"
POSE_TABLE = "pointyoink_register_pose.pose"
SOURCE = "pointyoink-scanner (PC mode)"          # property.rvproj's "source": the Tk scanner's text, so tools see no difference
STAMP_KEYS = ("frame", "t", "ok", "fit", "rmse", "reloc", "near", "far", "led", "gain", "hinted")
COLOR_STAMP_KEYS = ("capture_bands","color_saved", "color_t", "color_offset_ms", "color_seq", "color_frame_id", "color_sync")
PROJECT_PREFIX = "ProjectLive"


def stamp_now(): return time.strftime("%m%d%Y%H%M%S")


def frame_name(ps, idx): return "frame_%03d_%04d" % (ps, idx)


def _replace(path, data, mode="wb"):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, mode) as f: f.write(data)
    os.replace(tmp, path)


def inf_bytes(M):
    """The .inf of one frame: 16 zero bytes, the pose (scanner camera frame, mm), an identity."""
    return b"\0" * 16 + struct.pack("<16d", *np.asarray(M, np.float64).reshape(-1)) + struct.pack("<16d", *np.eye(4).reshape(-1))


def write_frame(cache, name, d, irl, M, irr=None):
    """One recorded frame: depth (after the range cut), IR left, pose. Plain writes like the Tk scanner (three small
    files per frame at 3 fps; a temp file each would only add work)."""
    with open(os.path.join(cache, name + ".dph"), "wb") as f: f.write(np.ascontiguousarray(d, np.uint16).tobytes())
    with open(os.path.join(cache, name + ".irl"), "wb") as f: f.write(np.ascontiguousarray(irl, np.uint8).tobytes())
    if irr is not None:
        if np.shape(irr)!=np.shape(irl):raise ValueError("Right IR size differs from left IR")
        with open(os.path.join(cache,name+".irr"),"wb") as f:f.write(np.ascontiguousarray(irr,np.uint8).tobytes())
    with open(os.path.join(cache, name + ".inf"), "wb") as f: f.write(inf_bytes(M))


DISCARDED = "discarded"                          # <side>/discarded/: frames taken back while paused (Back 5 ...), never built
FRAME_EXTS = (".dph", ".irl", ".irr", ".inf", ".img")


def move_frames(src, dst, names):
    """Move recorded frames' files (every extension there is) from folder src to dst; a file already in dst is replaced."""
    os.makedirs(dst, exist_ok=True)
    for n in names:
        for ext in FRAME_EXTS:
            p = os.path.join(src, n + ext)
            if os.path.exists(p): os.replace(p, os.path.join(dst, n + ext))


def discarded_passes(folder):
    """The pass numbers of the frames in <side>/discarded/ (a new pass never reuses one, so no name repeats)."""
    try: return [int(x[6:9]) for x in os.listdir(os.path.join(folder, DISCARDED)) if x.startswith("frame_") and x[6:9].isdigit()]
    except OSError: return []


def read_depth(cache, name, w=W, h=H):
    """A saved frame's depth (HxW u16, 0.1 mm), or None when it is missing or the wrong size."""
    p = os.path.join(cache, name + ".dph")
    if not os.path.exists(p): return None
    d = np.fromfile(p, np.uint16)
    return d.reshape(h, w) if d.size == w * h else None


def pose_table_bytes(entries):
    """register.write_pose_table's layout: <i count; per entry <ii pass, frame and 16 <d."""
    out = [struct.pack("<i", len(entries))]
    for ps, fr, M in entries:
        out.append(struct.pack("<ii", ps, fr)); out.append(struct.pack("<16d", *np.asarray(M, np.float64).reshape(-1)))
    return b"".join(out)


def track_json_text(frames, tracked, stamps):
    """pointyoink_track.json exactly as json.dump(..., indent=1) writes it in the Tk scanner (NaN stays NaN)."""
    return json.dumps(dict(version=1, live=True, frames=frames, tracked=tracked,
                           lost=[x["frame"] for x in stamps if not x["ok"]], stamps=stamps), indent=1)


def property_text(w=W, h=H):
    return json.dumps({"scan_param": {"depth_width": w, "depth_height": h, "depth_scale": 0.1, "ir_width": w, "ir_height": h,
                                      "source": SOURCE}}, indent=1)


def new_side_folder(project, param_dir, w=W, h=H):
    """data/<stamp>/ with cache/, a copy of the calibration and property.rvproj. Two sides in one second: wait for the
    next second (the folder name is the side order)."""
    stamp = stamp_now()
    while os.path.exists(os.path.join(project, "data", stamp)): time.sleep(1); stamp = stamp_now()
    folder = os.path.join(project, "data", stamp); os.makedirs(os.path.join(folder, "cache"))
    shutil.copytree(param_dir, os.path.join(folder, "param"))
    _replace(os.path.join(folder, "property.rvproj"), property_text(w, h), "w")
    return folder


def write_side_report(cache, entries, stamps):
    """The pose table and pointyoink_track.json of a side."""
    _replace(os.path.join(cache, POSE_TABLE), pose_table_bytes(entries))
    _replace(os.path.join(cache, TRACK_JSON), track_json_text(len(stamps), len(entries), stamps), "w")


def write_thumbs(folder, rgb_thumb, model_thumb):
    """thumb_model.png (PNG) and thumb_rgb.jpg (JPEG quality 85), written with PIL as the Tk scanner does."""
    from PIL import Image
    if model_thumb is not None:
        tmp = os.path.join(folder, "thumb_model.tmp.%d.png" % os.getpid())
        Image.fromarray(model_thumb).save(tmp); os.replace(tmp, os.path.join(folder, "thumb_model.png"))
    if rgb_thumb is not None:
        tmp = os.path.join(folder, "thumb_rgb.tmp.%d.jpg" % os.getpid())
        Image.fromarray(rgb_thumb).save(tmp, quality=85); os.replace(tmp, os.path.join(folder, "thumb_rgb.jpg"))


def write_auto_plane(sideplane, path, source=False):
    """Auto remove base's table_plane.json: table_plane.SidePlane.save's bytes, plus "source": "auto" when source is set
    (the Tk scanner marks it only when a bed is learned; with no bed the file stays exactly as SidePlane writes it).
    Returns whether a plane was found; without one an older table_plane.json is removed, as in the Tk scanner."""
    if not sideplane.save(path):
        if os.path.exists(path): os.remove(path)
        return False
    if source:
        with open(path) as f: j = json.load(f)
        j["source"] = "auto"; _replace(path, json.dumps(j), "w")
    return True


def plane_source(path):
    """table_plane.json's "source" ("bed", "auto"), or None (no file, unreadable, or not marked)."""
    try:
        with open(path) as f: return json.load(f).get("source")
    except Exception: return None


def read_side(folder):
    """What a saved side holds, for reopening it: (entries [(pass, frame, 4x4 mm)], stamps, rgb thumb, model thumb).
    A folder without pointyoink_track.json is not a saved side (None)."""
    from fuse import read_global_poses
    cache = os.path.join(folder, "cache"); tj = os.path.join(cache, TRACK_JSON)
    if not os.path.exists(tj): return None
    with open(tj) as f: stamps = json.load(f).get("stamps", [])
    pp = os.path.join(cache, POSE_TABLE)
    gp = read_global_poses(pp) if os.path.exists(pp) else {}
    entries = [(k[0], k[1], M) for k, M in sorted(gp.items())]
    thumbs = []
    for name in ("thumb_rgb.jpg", "thumb_model.png"):
        p = os.path.join(folder, name)
        if not os.path.exists(p): p = _scanner_thumb(folder, cache, name)       # a side the scanner made: its own pictures
        if p:
            from PIL import Image
            try:
                im = Image.open(p); im.draft("RGB", (320, 240)); im = im.convert("RGB")
                if im.width > 400: im.thumbnail((320, 240))
                thumbs.append(np.array(im))
            except OSError: thumbs.append(None)
        else: thumbs.append(None)
    return entries, stamps, thumbs[0], thumbs[1]


def _scanner_thumb(folder, cache, name):
    """For a scanner-made side: its first color frame stands in for thumb_rgb.jpg, its preview.png for thumb_model.png."""
    if name == "thumb_model.png":
        p = os.path.join(folder, "preview.png"); return p if os.path.exists(p) else None
    try: first = min(x for x in os.listdir(cache) if x.startswith("frame_") and x.endswith(".img"))
    except (OSError, ValueError): return None
    return os.path.join(cache, first)


def saved_sides(project):
    """The side folders of a project in side order: those with a pointyoink_track.json (as build_project.py finds them)."""
    data = os.path.join(project, "data")
    if not os.path.isdir(data): return []
    return [os.path.join(data, x) for x in sorted(os.listdir(data)) if os.path.exists(os.path.join(data, x, "cache", TRACK_JSON))]


def project_when(path):
    """"27 Sep 2026, 00:20" from a ProjectLive<%m%d%Y%H%M%S> (or the scanner's Project<...>) folder name, or "" for any other name."""
    name = os.path.basename(os.path.normpath(path))
    for prefix in (PROJECT_PREFIX, "Project"):                       # PointYoink's own scans, and the scanner's own project names
        if name.startswith(prefix) and name[len(prefix):].isdigit(): name = name[len(prefix):]; break
    try: t = time.strptime(name, "%m%d%Y%H%M%S")
    except ValueError: return ""
    return "%d %s" % (t.tm_mday, time.strftime("%b %Y, %H:%M", t))
