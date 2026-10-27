#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live MIRACO scanning on Linux over PC mode (plain v4l2, no vendor SDK), tracked with PointYoink's track.Tracker.

Saves a scan in the scanner's own project layout, so PointYoink's build step (fuse.py) can use it:
  <root>/Project<stamp>/data/<stamp>/{cache/frame_PPP_NNNN.dph|.inf|.irl, param/, property.rvproj}
.inf holds OUR live pose (scanner convention, mm); pointyoink_register_pose.pose lists only frames that tracked.

The window has a Sides panel: every side of the part is its own scan with a picture (colour camera at its start, its
model so far). Click a side to pick it; the big button scans into the picked side (a stop and start is just a new pass).
+ New side adds one. Finish saves and closes. SPACE = the big button.
usage: pointyoink-scanner [--last | --project <ProjectLive folder to continue>] [--param <scan>/param] [--root ~/revopoint-scans-models]
"""
import argparse, json, os, queue, shutil, struct, subprocess, sys, threading, time
os.environ["XMODIFIERS"] = "@im=none"          # Tk + ibus input method stalls the window; always off here
import numpy as np, cv2
import faulthandler, signal
faulthandler.register(signal.SIGUSR1)        # kill -USR1 <pid> prints where every thread is, without stopping the window

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from track import Tracker, F
from fuse import read_calib
from register import write_pose_table
from table_plane import SidePlane, to_camera
import bed as bedlib                            # Learn bed: the turntable's dot map, found in every frame
from bed import BedModel, SideBed, BedLearner

W, H = 800, 600
BED_CUTS = True             # the table dots' plane goes into table_plane.json when the bed was found (plan gate G2: T7 passed)
FS = W * H * 4                     # Y16 800x1200: depth u16 | IR left u8 | IR right u8
RGB_CAL = None       # (K, R, T) of the connected scanner's color camera, read from its own param folder (set_rgb_calib)


def set_rgb_calib(param_dir):
    """Read this scanner's color camera calibration; without it the color view shows no depth guides."""
    global RGB_CAL
    from fuse import read_rgb_calib
    RGB_CAL = read_rgb_calib(param_dir)
    return RGB_CAL is not None


def depth_px_to_rgb(u, v, z_mm, cal):
    """Depth-image pixels at depth z (mm) -> pixel positions in the 800x600 colour view."""
    fx, fy, cx, cy = cal
    P = np.stack([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm * np.ones_like(u, dtype=float)], -1)
    K, R, T = RGB_CAL
    Q = P @ R.T + T
    q = Q @ K.T
    return q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]


def label(img, text, org, scale=1.0, color=(255, 255, 255)):
    """Text that reads on any background: bold white on a dark box, like subtitles. org = bottom-left of the text."""
    th = max(2, int(round(2.4 * scale)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1]); pad = int(8 * scale)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - tht - pad), min(img.shape[1], x + tw + pad), min(img.shape[0], y + base + pad)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.uint16) * 3 // 10).astype(np.uint8)   # darken behind the text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def outlined_line(img, a, b, color, w=3):
    cv2.line(img, a, b, (0, 0, 0), w + 4, cv2.LINE_AA); cv2.line(img, a, b, color, w, cv2.LINE_AA)


def draw_depth_guides(rgb, d, cal, z_ref, dot=(60, 255, 60)):
    """On the colour view: where the kept depth points land (dots: green, red while tracking is lost), the depth camera's
    view outline at the part's distance z_ref (dotted) and its centre (+). The two cameras sit apart, so where the depth
    view lands in the colour picture depends on distance."""
    if RGB_CAL is None: return rgb
    keep = d > 0
    vv, uu = np.nonzero(keep[::6, ::6]); vv = vv * 6; uu = uu * 6
    if len(uu):
        x, y = depth_px_to_rgb(uu.astype(float), vv.astype(float), d[vv, uu].astype(float) * 0.1, cal)
        ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
        xi, yi = x[ok].astype(int), y[ok].astype(int)
        rgb[yi, xi] = dot; rgb[yi + 1, xi] = dot
    # outline of the depth view at the part's distance: walk the depth image border
    n = 60
    edge = np.concatenate([np.stack([np.linspace(0, W - 1, n), np.zeros(n)], 1), np.stack([np.full(n, W - 1), np.linspace(0, H - 1, n)], 1),
                           np.stack([np.linspace(W - 1, 0, n), np.full(n, H - 1)], 1), np.stack([np.zeros(n), np.linspace(H - 1, 0, n)], 1)])
    x, y = depth_px_to_rgb(edge[:, 0], edge[:, 1], np.full(len(edge), z_ref), cal)
    pts = np.stack([x, y], 1).astype(int)
    for i in range(0, len(pts) - 1, 2):                 # every other segment: a dotted line
        outlined_line(rgb, tuple(pts[i]), tuple(pts[i + 1]), (255, 220, 0))
    cx_, cy_ = depth_px_to_rgb(np.array([W / 2]), np.array([H / 2]), np.array([z_ref]), cal)
    c = (int(cx_[0]), int(cy_[0]))
    outlined_line(rgb, (c[0] - 22, c[1]), (c[0] + 22, c[1]), (255, 220, 0)); outlined_line(rgb, (c[0], c[1] - 22), (c[0], c[1] + 22), (255, 220, 0))
    label(rgb, "depth view @ %.0f mm" % z_ref, (max(12, pts[0][0] + 10), max(90, pts[0][1] - 14)), 1.1)
    return rgb


# The fill LED needs a local add-on (dev/scanner_led.py, not part of the repo). Without it the LED button is off.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev"))
    from scanner_led import set_fill_led
except Exception:
    set_fill_led = None


def set_ir_gain(g):
    """IR sensor gain (standard UVC control on the depth camera, 1..15). The scanner keeps exposure time on its own auto
    and refuses manual exposure, so gain is the brightness knob. Measured on a dark part: gain 4 gave about 2x the depth
    of the default 1; 15 clips the IR and loses depth again."""
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=gain=%d" % int(g)], capture_output=True)


def write_inf(path, M):
    with open(path, "wb") as f:
        f.write(b"\0" * 16); f.write(struct.pack("<16d", *M.reshape(-1))); f.write(struct.pack("<16d", *np.eye(4).reshape(-1)))


def restart_argv(argv):
    """New project: the same command line (--root, --param, --near, --far...) minus the project it continued (--last,
    --project <dir>, also as --project=<dir> or an abbreviation argparse would take)."""
    out, skip = [], False
    for x in argv:
        if skip: skip = False; continue
        k = x.split("=", 1)[0]
        if len(k) >= 4 and "--project".startswith(k):
            skip = "=" not in x; continue
        if len(k) >= 3 and "--last".startswith(k): continue
        out.append(x)
    return out


RAWQ_IDLE, RAWQ_REC = 4, 300       # raw frames kept waiting: not recording only the newest few; recording, 300 (~580 MB) at most


def put_frame(q, item, recording):
    """Queue a raw frame (1.9 MB) for the window. When the window stops taking them they must not fill the memory: not
    recording, only the newest RAWQ_IDLE matter; recording, a backlog of RAWQ_REC means the window hangs, and dropping the
    oldest beats running the PC out of memory. Returns how many frames were dropped."""
    q.put(item); dropped = 0
    while q.qsize() > (RAWQ_REC if recording else RAWQ_IDLE):
        try: q.get_nowait(); dropped += 1
        except queue.Empty: break
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default=os.path.expanduser(
        "~/revopoint-scans-models/Project09222026020625/data/09252026224708/param"))
    ap.add_argument("--root", default=os.path.expanduser("~/revopoint-scans-models"))
    ap.add_argument("--near", type=float, default=100.0, help="mm, closer than this is cut")
    ap.add_argument("--far", type=float, default=450.0, help="mm, further than this is cut")
    ap.add_argument("--project", default=None, help="continue this ProjectLive folder (its sides are listed and can be continued)")
    ap.add_argument("--last", action="store_true", help="continue the newest ProjectLive folder")
    a = ap.parse_args()
    if a.last and not a.project:
        import glob as _g
        old = sorted(_g.glob(os.path.join(a.root, "ProjectLive*")), key=os.path.getmtime)
        if old: a.project = old[-1]
    import open3d.core as o3c
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
    import io, traceback
    from fuse import read_global_poses
    # one BLAS / OpenCV thread for this process (the table finder's small matrices; the pools spun at 9x the CPU). Here, before
    # any other thread starts: threadpoolctl walks the loaded libraries, and doing that from the finder's thread deadlocked
    # against the window loading a library (seen in a replay: the window never came up)
    bedlib.quiet_threads()                     # (the limit holds for the process; nothing to keep)

    proj = a.project or os.path.join(a.root, "ProjectLive" + time.strftime("%m%d%Y%H%M%S"))
    os.makedirs(os.path.join(proj, "data"), exist_ok=True)
    cal = read_calib(os.path.join(a.param, "Pl.bin")); set_rgb_calib(a.param)
    def new_tracker():   # the range sliders do the cutting; the slow global search runs every 3rd lost frame so the window stays live
        return Tracker(cal, W, H, o3c.Device("CUDA:0"), max_depth_mm=1000.0, reloc_every=3, floater_looks=8)

    # ---------------- Learn bed (bed.py): the turntable's dots, learned once, place the platter from the first frames ----
    # The project's own copy wins (builds and replays repeat exactly); else the one in the config dir, shared by every
    # project on this turntable. No bed: everything works as before (Auto remove base).
    bedst = dict(bed=None, src=None, prior=None, miss=0, miss_said=False)
    for p_, src_ in ((os.path.join(proj, "bed.json"), "project"), (BedModel.config_path(), "config")):
        b_ = BedModel.load(p_) if os.path.exists(p_) else None
        if b_ is not None and len(b_) >= bedlib.HIT_MIN:
            bedst.update(bed=b_, src=src_, prior=b_.camera); break
    if bedst["bed"] is not None:
        print("bed: %d dots (%s, from the %s)" % (len(bedst["bed"]), bedst["bed"].id, bedst["src"]), flush=True)
        if bedst["bed"].calib_md5 and bedst["bed"].calib_md5 != bedlib.calib_md5(a.param):
            print("bed: learned with another calibration file; dots and depth share it, so it still fits", flush=True)
    def new_sidebed(): return SideBed(bedst["bed"], bedst["prior"])

    # ---------------- sides: each one is a scan folder in the project ----------------
    class Side:
        def __init__(self, folder):
            self.folder = folder; self.cache = os.path.join(folder, "cache")
            self.entries = []; self.stamps = []; self.trk = None; self.ps = -1; self.idx = 0; self.lost = 0
            self.rgb_thumb = None; self.model_thumb = None; self.dirty = False
            self.table = SidePlane()                          # Auto remove base: the platter as seen in this side's frames
            bp = os.path.join(folder, "bed_pose.json")        # the table dots' platter (bed.py), when a bed is learned
            self.bed = SideBed.load(bp, bedst["bed"], bedst["prior"]) if os.path.exists(bp) else new_sidebed()
            self.pass_ok = 0; self.dots_seen = 0; self.bed_off_pass = None; self.seed_tried = False
            self.moved_pass = None                            # the pass in which the part moved on the table (no finder jobs after)
            tj = os.path.join(self.cache, "pointyoink_track.json")
            if os.path.exists(tj):                            # a saved side: its frames, poses and pictures
                self.stamps = json.load(open(tj)).get("stamps", [])
                gp = read_global_poses(os.path.join(self.cache, "pointyoink_register_pose.pose"))
                self.entries = [(k[0], k[1], M) for k, M in sorted(gp.items())]
                self.lost = sum(1 for x in self.stamps if not x.get("ok"))
                self.ps = max([k[0] for k in gp] or [-1])
                for name, attr in (("thumb_rgb.jpg", "rgb_thumb"), ("thumb_model.png", "model_thumb")):
                    p = os.path.join(folder, name)
                    if os.path.exists(p): setattr(self, attr, np.array(Image.open(p).convert("RGB")))
        @property
        def frames(self): return len(self.stamps)

    def make_side():
        stamp = time.strftime("%m%d%Y%H%M%S")
        while os.path.exists(os.path.join(proj, "data", stamp)): time.sleep(1); stamp = time.strftime("%m%d%Y%H%M%S")
        folder = os.path.join(proj, "data", stamp); os.makedirs(os.path.join(folder, "cache"))
        shutil.copytree(a.param, os.path.join(folder, "param"))
        json.dump({"scan_param": {"depth_width": W, "depth_height": H, "depth_scale": 0.1, "ir_width": W, "ir_height": H,
                                  "source": "pointyoink-scanner (PC mode)"}}, open(os.path.join(folder, "property.rvproj"), "w"), indent=1)
        for other in sides: release(other)
        sd = Side(folder); sd.trk = new_tracker(); return sd

    def release(sd):
        """Free a side's model from the GPU. Only one model may live there at a time: a second Open3D voxel grid in the
        same process made tracking crawl (measured: 35 ms per frame alone, minutes with an older grid still alive)."""
        if sd is None or sd.trk is None: return
        sd.trk = None
        import gc; gc.collect(); o3c.cuda.release_cache()

    def load_tracker(sd):
        """Rebuild a saved side's model from its frames at their saved poses (no matching), so scanning can carry on."""
        for other in sides:
            if other is not sd: release(other)
        sd.trk = new_tracker(); sd.table = SidePlane()
        bp = os.path.join(sd.folder, "bed_pose.json")         # the bed as saved (no looking for it again over the frames)
        sd.bed = SideBed.load(bp, bedst["bed"], bedst["prior"]) if os.path.exists(bp) else new_sidebed()
        for ps, idx, M in sd.entries:
            p = os.path.join(sd.cache, "frame_%03d_%04d.dph" % (ps, idx))
            if not os.path.exists(p): continue
            T = M @ F; T = T.copy(); T[:3, 3] /= 1000.0
            raw = np.frombuffer(open(p, "rb").read(), np.uint16).reshape(H, W)
            sd.trk.seed(raw, T)
            try: sd.table.add(raw, cal, M @ F)
            except Exception as e: print("table:", e, flush=True)

    def save_side(sd):
        """Write the side's pose table, report and pictures. A side that never tracked a frame is removed. Saving never
        waits for the table finder: a result of it that comes in later brings the saved platter up to date by itself
        (drain_bed -> write_platter)."""
        if not sd.dirty: return
        if not sd.entries:
            shutil.rmtree(sd.folder, ignore_errors=True); sides.remove(sd)
            if sel[0] is sd: sel[0] = None          # it was picked: the buttons must not ask for its number any more
            return
        write_pose_table(os.path.join(sd.cache, "pointyoink_register_pose.pose"), sd.entries)
        json.dump(dict(version=1, live=True, frames=sd.frames, tracked=len(sd.entries),
                       lost=[x["frame"] for x in sd.stamps if not x["ok"]], stamps=sd.stamps),
                  open(os.path.join(sd.cache, "pointyoink_track.json"), "w"), indent=1)
        if sd.trk is not None and sd.entries:
            T = sd.entries[-1][2] @ F; T = T.copy(); T[:3, 3] /= 1000.0
            try: sd.model_thumb = sd.trk.render(T, scale=0.5, weight_threshold=2.0)
            except Exception as e: print("render:", e, flush=True)
        if sd.model_thumb is not None: Image.fromarray(sd.model_thumb).save(os.path.join(sd.folder, "thumb_model.png"))
        if sd.rgb_thumb is not None: Image.fromarray(sd.rgb_thumb).save(os.path.join(sd.folder, "thumb_rgb.jpg"), quality=85)
        tp = os.path.join(sd.folder, "table_plane.json")      # the platter, for the build to cut (build_project.py)
        try: save_platter(sd, tp)
        except Exception as e: print("table:", e, flush=True)
        sd.dirty = False
        print("saved side %s: %d frames, %d tracked, %d lost" % (os.path.basename(sd.folder), sd.frames, len(sd.entries), sd.lost), flush=True)

    def write_platter(sd, tp):
        """table_plane.json for the build: the table dots' plane when the bed was found ("source": "bed"), else Auto
        remove base's ("source": "auto"), else none; and bed_pose.json beside it (what the bed saw). Atomic writes only
        (the build may be reading them). The build reads only n, d and keep_above, so it needs no change. Returns the
        source written ("bed", "auto" or None)."""
        if BED_CUTS and sd.bed.found and sd.bed.fixed is not None:
            j = sd.bed.table_plane_json(1.0, sd.table.spin_axis())
            tmp = tp + ".tmp.%d" % os.getpid(); json.dump(j, open(tmp, "w")); os.replace(tmp, tp); src = "bed"
        elif sd.table.save(tp):
            src = "auto"
            if sd.bed.bed is not None:                                  # with no bed the file stays exactly as before
                try:
                    j = json.load(open(tp)); j["source"] = "auto"; tmp = tp + ".tmp.%d" % os.getpid(); json.dump(j, open(tmp, "w")); os.replace(tmp, tp)
                except Exception as e: print("table:", e, flush=True)
        else:
            src = None
            if os.path.exists(tp): os.remove(tp)
        if sd.bed.bed is not None: sd.bed.save(os.path.join(sd.folder, "bed_pose.json"))
        return src

    def resave_platter(sd):
        """A table-finder result that came in after the side was saved (a pause, a Finish, picking another side): its
        table_plane.json and bed_pose.json again, nothing else (a few ms on the Tk thread; saving no longer waits for it)."""
        try:
            src = write_platter(sd, os.path.join(sd.folder, "table_plane.json"))
            print("side %s: platter saved again (%s): a table-finder result came in after the save" % (os.path.basename(sd.folder), src), flush=True)
        except Exception as e: print("table:", e, flush=True)

    def save_platter(sd, tp):
        """write_platter, and what a save says: where the base cut comes from, the project's copy of the bed it was
        scanned with, "bed not recognised", and with no bed at all a provisional one learned from this side."""
        n_ = side_no(sd) if sd in sides else 0
        auto = sd.table.plane()
        src = write_platter(sd, tp)
        if src == "bed" and auto is not None and sd.bed.auto_check:
            ac = sd.bed.auto_check; log("side %d: base from the table dots (%d dots); Auto remove base: %s, %+.1f mm, %.1f deg"
                                        % (n_, sd.bed.dots_matched, ac.get("verdict"), ac.get("mm", 0), ac.get("deg", 0)))
        elif src == "auto" and sd.bed.bed is not None and not sd.bed.found: log("side %d: table not seen, Auto remove base" % n_)
        if sd.bed.bed is not None:
            sd.bed.bed.copy_to(proj)
            if not sd.bed.found and sd.dots_seen >= bedlib.HIT_MIN:      # dots in view, none matched: the table changed?
                bedst["miss"] += 1
                if bedst["miss"] >= 3 and not bedst["miss_said"]: bedst["miss_said"] = True; log("bed not recognised: Learn bed again?")
            elif sd.bed.found: bedst["miss"] = 0
        elif bedst["bed"] is None and auto is not None and not sd.seed_tried:
            # no bed at all: learn a provisional one from this side's own dots (plan P2), so the next side has it. On the
            # table finder's thread (2-4 s, it froze the window here), once per side, into this project only (the
            # config's bed, shared by every project, is written by Learn bed alone)
            sd.seed_tried = True
            bed_seeds.put(dict(kind="seed", side=sd, folder=sd.folder, plane=(np.array(auto[0], float), float(auto[1])),
                               source="side:%s/%s" % (os.path.basename(proj), os.path.basename(sd.folder))))

    def platter(sd):
        """The platter the build will cut, (n, off, centre, radius, source) in the side's scan frame, or None: the table
        dots' when the bed was found, else Auto remove base's. The tint, Auto gain, the bar and the save all use this."""
        if sd.bed.found and sd.bed.fixed is not None:
            f = sd.bed.fixed; return f["n"], f["off"], f["centre"], sd.bed.bed.radius_mm, "bed"
        p = sd.table.plane()
        if p is None: return None
        dc = sd.table.disc()
        return p[0], p[1], None if dc is None else dc[0], None if dc is None else dc[1], "auto"

    def base_txt(sd):
        if sd is None: return ""
        if sd.bed.bed is None: return "base: Auto (no bed learned)"
        if sd.bed.found: return "base: bed (%d dots)" % sd.bed.dots_matched
        if sd.bed.state.startswith("vetoed"): return "base: Auto (bed disagreed)"
        return "base: looking for the bed" if (st["recording"] and sd.idx - sd.bed.look_from(sd.ps) <= bedlib.POOL_TO) else "base: Auto"

    sides = [Side(os.path.join(proj, "data", x)) for x in sorted(os.listdir(os.path.join(proj, "data")))
             if os.path.exists(os.path.join(proj, "data", x, "cache", "pointyoink_track.json"))]
    sel = [None]                                               # the picked side

    # ---------------- the log box's lines: from any thread (the build, the frame pump); tick() shows them ----------------
    log_q = queue.Queue()
    def log(msg):
        log_q.put(time.strftime("%H:%M:%S ") + msg)
        try: print(msg, flush=True)
        except Exception: pass                  # a closed terminal must not take the window (or its error handler) down

    # ---------------- cameras ----------------
    # The streams can die or stall under us (the scanner dropped off USB mid-side, seen live): the window keeps running,
    # pauses the side and reconnects when /dev/video0 is back. cam["gen"] tells a pump whether it still belongs to the
    # current connection.
    q = queue.Queue(); running = [True]; rgb_latest = [None]; rec_now = [False]      # rec_now: tick() mirrors st["recording"]
    dropped = dict(n=0, said_n=0, said=0.0)         # raw frames the pump dropped while recording (the window hung); tick() says so
    cam = dict(proc=None, rgb=None, gen=0, last=time.monotonic(), dead=False, tried=0.0)
    def pump(p, gen):               # recording, no raw frame is dropped (the tracker is much faster than 3 fps) unless the window hangs
        out = p.stdout
        while running[0] and cam["gen"] == gen:
            fr = bytearray()
            while len(fr) < FS:
                c = out.read(FS - len(fr))
                if not c:
                    if cam["gen"] == gen: cam["dead"] = True
                    return
                fr += c
            cam["last"] = time.monotonic()
            k = put_frame(q, (cam["last"], bytes(fr)), rec_now[0])
            if k and rec_now[0]: dropped["n"] += k  # recording: every dropped frame is a hole in the side (tick() tells the log box)
    def pump_rgb(p, gen):           # colour is only for looking at: keep the newest frame, drop the rest
        out = p.stdout; buf = b""
        while running[0] and cam["gen"] == gen:
            chunk = out.read(262144)
            if not chunk: return
            buf += chunk
            while True:
                a0 = buf.find(b"\xff\xd8")
                if a0 < 0: buf = b""; break
                b0 = buf.find(b"\xff\xd9", a0 + 2)
                if b0 < 0: buf = buf[a0:]; break
                rgb_latest[0] = buf[a0:b0 + 2]; buf = buf[b0 + 2:]
    def reap(p):                    # a v4l2-ctl stuck on a vanished device may ignore terminate
        try: p.wait(timeout=3)
        except subprocess.TimeoutExpired: p.kill()
    def stop_streams():
        for k in ("proc", "rgb"):
            p = cam[k]
            if p is not None and p.poll() is None:
                p.terminate(); threading.Thread(target=reap, args=(p,), daemon=True).start()
    def start_streams():
        stop_streams(); cam["gen"] += 1; gen = cam["gen"]
        cam["proc"] = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video0", "--set-fmt-video", "width=800,height=1200,pixelformat=Y16 ",
                                        "--stream-mmap", "--stream-to=-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        cam["rgb"] = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video2", "--set-fmt-video", "width=2000,height=1500,pixelformat=MJPG",
                                       "--stream-mmap", "--stream-to=-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        cam["last"] = time.monotonic(); cam["dead"] = False
        threading.Thread(target=pump, args=(cam["proc"], gen), daemon=True).start()
        threading.Thread(target=pump_rgb, args=(cam["rgb"], gen), daemon=True).start()
    start_streams()

    # ---------------- the table finder runs beside the window: find / pool / learn jobs in, results out ----------------
    # Only tick() applies results (on the Tk thread). The worker sees copies only (d is a copy, IR views immutable bytes).
    # When it falls behind, the oldest waiting job goes: the newest frame is what matters.
    # Seeds (a provisional bed learned from a saved side, seconds) wait in their own queue: frame jobs must not push them out.
    bed_jobs = queue.Queue(maxsize=2); bed_res = queue.Queue(); bed_seeds = queue.Queue()
    def bed_job(job):
        try: bed_jobs.put_nowait(job); return
        except queue.Full: pass
        try: bed_jobs.get_nowait()
        except queue.Empty: pass
        try: bed_jobs.put_nowait(job)
        except queue.Full: pass
    def run_job(job):
        """One table-finder job -> its result (for drain_bed). On the worker thread; at Finish, on the main thread."""
        t0 = time.monotonic(); out = dict(job, res=None)
        try:
            if job["kind"] == "learn":
                lr = job["learner"]; out["res"] = lr.add(job["d"], job["ir"], t=job["t"], gain=job["gain"])
                if out["res"]["state"] == "done": out["bed"] = lr.result(a.param)
            elif job["kind"] == "find":
                dl = bedlib.detect_dots(job["ir"]); out["n_dots"] = len(dl)
                out["res"] = bedlib.find_bed(job["d"], job["ir"], cal, job["bed"], prior=job["prior"], dots=dl)
            elif job["kind"] == "pool":
                f = bedlib.pool_features(job["d"], job["ir"], cal); out["feats"] = f
                out["res"] = bedlib.solve_pool(job["snap"] + [(f, job["T"])], job["bed"], cal, job["extra"])
            elif job["kind"] == "seed":
                out["new_bed"], out["why"] = bedlib.provisional_bed(job["folder"], job["plane"], job["source"])
                if out["new_bed"] is not None: out["new_bed"].calib_md5 = bedlib.calib_md5(a.param)
        except Exception as e:
            out["error"] = "%s: %s" % (type(e).__name__, e)
        out["secs"] = time.monotonic() - t0
        out.pop("d", None); out.pop("ir", None)
        return out
    def bed_worker():
        skip = 0; slow = {}
        while running[0]:
            try: job = bed_seeds.get_nowait()
            except queue.Empty:
                try: job = bed_jobs.get(timeout=0.5)
                except queue.Empty: continue
            if skip > 0 and job["kind"] not in ("learn", "seed"): skip -= 1; continue     # a slow call: let the next few frames go
            out = run_job(job)
            if out["secs"] > 1.0 and job["kind"] not in ("learn", "seed"):
                skip = 5; key = (id(job.get("side")), job.get("ps")); slow[key] = slow.get(key, 0) + 1; out["slow"] = slow[key]
            bed_res.put(out)
    bed_thread = threading.Thread(target=bed_worker, daemon=True, name="bed_worker"); bed_thread.start()

    # ---------------- window ----------------
    st = dict(recording=False, last=None, arrivals=[], quit=False, led=False, model_img=None, rgb_img=None, busy=None,
              orbit=None, drag=None, orbit_dirty=False)
    root = tk.Tk(); root.title("PointYoink Scanner"); root.configure(bg="#111")
    BIG = ("DejaVu Sans", 16, "bold"); MID = ("DejaVu Sans", 12, "bold")
    main_f = tk.Frame(root, bg="#111"); main_f.pack(side="left", fill="both")
    side_f = tk.Frame(root, bg="#181818", width=350); side_f.pack(side="left", fill="y"); side_f.pack_propagate(False)
    lab = tk.Label(main_f, bg="#111"); lab.pack()
    bar = tk.Label(main_f, font=("DejaVu Sans", 14, "bold"), bg="#111", fg="#ccc", anchor="w"); bar.pack(fill="x", padx=10, pady=(6, 0))
    hint = tk.Label(main_f, font=("DejaVu Sans", 12), bg="#111", fg="#aaa", anchor="w"); hint.pack(fill="x", padx=10, pady=(0, 4))
    ctl = tk.Frame(main_f, bg="#111"); ctl.pack(fill="x", padx=10)
    near = tk.IntVar(value=int(a.near)); far = tk.IntVar(value=int(a.far))
    sl = dict(from_=50, to=800, orient="horizontal", length=300, resolution=5, bg="#111", fg="#ddd",
              troughcolor="#333", highlightthickness=0, font=("DejaVu Sans", 11))
    tk.Scale(ctl, label="Near cut (mm)", variable=near, **sl).pack(side="left", padx=(0, 12))
    tk.Scale(ctl, label="Far cut (mm)", variable=far, **sl).pack(side="left", padx=(0, 12))
    gain = tk.IntVar(value=4); cycle = tk.BooleanVar(value=False)
    tk.Scale(ctl, label="IR gain", variable=gain, from_=1, to=15, orient="horizontal", length=160, resolution=1, bg="#111",
             fg="#ddd", troughcolor="#333", highlightthickness=0, font=("DejaVu Sans", 11)).pack(side="left", padx=(0, 6))
    tk.Checkbutton(ctl, text="Auto gain", variable=cycle, font=MID, bg="#111", fg="#ddd", selectcolor="#2e5d2e",
                   activebackground="#111", activeforeground="#fff").pack(side="left", padx=(0, 6), pady=(18, 0))
    st["gain_set"] = None; st["turn"] = None; st["cycle_msg"] = ""
    def apply_gain():
        """Once per frame: the slider's gain (Auto gain moves the slider). Only talks to the camera on a change."""
        g = gain.get()
        if g != st["gain_set"]:
            st["gain_set"] = g; threading.Thread(target=set_ir_gain, args=(g,), daemon=True).start()
        return g

    def gain_order(g0):
        """The gains Auto gain tries, one per full turn: up first, then down, further out each time (dark plastic wants
        a high gain, shiny spots a low one). 12 is the top: 15 clips the IR and loses depth again."""
        out = []
        for g in (g0, g0 + 3, g0 - 2, g0 + 6, g0 - 3, g0 + 8):
            g = min(12, max(1, g))
            if g not in out: out.append(g)
        return out

    def well_covered(sd):
        """How many of the PART's points have been seen often enough (the blue ones in the model view). Once the side's
        platter plane is known, only points 2 mm or more above it and inside the platter's circle count: otherwise a gain
        that lights up more of the platter or the background looked like progress, and Auto gain kept the wrong gain."""
        trk = sd.trk
        if trk is None or trk.model is None: return 0
        Pm = trk.model.point.positions.cpu().numpy(); keep = trk.coverage_levels(Pm) == 2
        pl = platter(sd)                                            # the table dots' platter from the first turn, else Auto's
        if pl is not None:
            P = Pm * 1000.0; n, off = pl[0], pl[1]                  # tracker metres -> the side's scan frame in mm
            keep &= P @ n + off > 2.0
            if pl[2] is not None:
                h = P - pl[2]; keep &= np.linalg.norm(h - np.outer(h @ n, n), axis=1) < pl[3] + 10.0
        return int(keep.sum())

    def auto_gain(sd):
        """Auto gain: after every full turn of the table, see how much well-covered surface that turn added. While turns
        still add some, the next turn runs at the next gain, so each turn picks up what the others could not. When a
        turn adds next to nothing the gain stays put and the bar says so."""
        ang = sd.table.turned_deg()
        if ang is None: return
        t = st["turn"]
        if t is None or t["side"] is not sd:
            t = st["turn"] = dict(side=sd, mark=ang, count=well_covered(sd), order=gain_order(gain.get()), i=0, flat=0, done=False, n=0)
            st["cycle_msg"] = "auto gain: turn 1 at gain %d" % gain.get(); return
        if t["done"] or abs(ang - t["mark"]) < 360.0: return
        now = well_covered(sd); added = now - t["count"]; t["n"] += 1
        t["mark"] = ang; t["count"] = now
        t["flat"] = t["flat"] + 1 if added < 0.03 * max(now, 1) else 0
        log("auto gain: turn %d at gain %d added %d well-covered points (now %d)" % (t["n"], gain.get(), added, now))
        if t["flat"] >= 2 or t["i"] + 1 >= len(t["order"]):
            t["done"] = True; st["cycle_msg"] = "auto gain: more turns add nothing new, this side is done"; return
        t["i"] += 1; gain.set(t["order"][t["i"]])
        st["cycle_msg"] = "auto gain: turn %d at gain %d (last turn added %d%%)" % (t["n"] + 1, gain.get(), round(100 * added / max(now, 1)))
    auto_rng = tk.BooleanVar(value=True)
    tk.Checkbutton(ctl, text="Auto range", variable=auto_rng, font=MID, bg="#111", fg="#ddd", selectcolor="#2e5d2e",
                   activebackground="#111", activeforeground="#fff").pack(side="left", padx=6, pady=(18, 0))
    ledb = tk.Button(ctl, text="Fill LED: off", width=12, bg="#333", fg="#ddd", activebackground="#555", font=MID)
    ledb.pack(side="left", padx=6, pady=(18, 0))
    home_b = tk.Button(ctl, text="Home: live view", width=14, bg="#333", fg="#ddd", activebackground="#555", font=MID)
    home_b.pack(side="left", padx=6, pady=(18, 0))

    # --- the model panel can be turned and zoomed by hand; Home (or double-click) snaps it back to the scanner's view
    def rot(axis, ang):
        axis = axis / np.linalg.norm(axis); x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    def view_pose():
        o = st["orbit"]; T0 = o["T0"]; c = o["c"]
        R = rot(T0[:3, 1], o["yaw"]) @ rot(T0[:3, 0], o["pitch"])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c - R @ c
        T = M @ T0; Z = np.eye(4); Z[2, 3] = o["zoom"]; return T @ Z
    def in_model_panel(e): return e.x >= 600 and e.y >= 450          # bottom-right quarter of the 1200x900 view
    def orbit_start(e):
        sd = sel[0]
        if not in_model_panel(e) or sd is None or sd.trk is None or sd.trk.fused == 0: return
        if st.get("orbit") is None:
            if sd.trk.model is None: sd.trk.model = sd.trk._model_cloud()
            c = sd.trk.model.point.positions.cpu().numpy().mean(0)
            st["orbit"] = dict(yaw=0.0, pitch=0.0, zoom=0.0, T0=sd.trk.T.copy(), c=c)
        st["drag"] = (e.x, e.y)
    def orbit_drag(e):
        if st.get("orbit") is None or st.get("drag") is None: return
        dx, dy = e.x - st["drag"][0], e.y - st["drag"][1]; st["drag"] = (e.x, e.y)
        st["orbit"]["yaw"] -= dx * 0.01; st["orbit"]["pitch"] += dy * 0.01; st["orbit_dirty"] = True
    def orbit_zoom(e, step):
        if not in_model_panel(e): return
        orbit_start(e)
        if st.get("orbit") is None: return
        st["orbit"]["zoom"] += step; st["orbit_dirty"] = True
    def go_home(_=None):
        # back to the scanner's own view. While paused nothing else redraws the model, so ask for one redraw here
        # (without it the button looked dead: the picture stayed on the turned view)
        st["orbit"] = None; st["drag"] = None; st["orbit_dirty"] = False; st["home_dirty"] = True
    lab.bind("<ButtonPress-1>", orbit_start); lab.bind("<B1-Motion>", orbit_drag)
    lab.bind("<ButtonRelease-1>", lambda e: st.update(drag=None))
    lab.bind("<Button-4>", lambda e: orbit_zoom(e, 0.01)); lab.bind("<Button-5>", lambda e: orbit_zoom(e, -0.01))
    lab.bind("<Double-Button-1>", lambda e: go_home() if in_model_panel(e) else None)
    home_b.configure(command=go_home)
    def toggle_led():
        st["led"] = not st["led"]
        ledb.configure(text="Fill LED: ON" if st["led"] else "Fill LED: off", bg="#8a7a20" if st["led"] else "#333")
        def go():
            if set_fill_led is None: return
            try: set_fill_led(st["led"])
            except Exception as e: print("LED:", e, flush=True)
        threading.Thread(target=go, daemon=True).start()
    ledb.configure(command=toggle_led)
    if set_fill_led is None: ledb.configure(state="disabled", text="Fill LED: n/a")
    try: set_fill_led and set_fill_led(False)       # the scanner can't report the LED state back, so start from a known state (off)
    except Exception as e: print("LED:", e, flush=True)
    actions = tk.Frame(main_f, bg="#111"); actions.pack(fill="x", padx=10, pady=8)
    go_b = tk.Button(actions, font=BIG, width=26, height=2, relief="flat", wraplength=400); go_b.pack(side="left")   # long messages wrap onto the second line instead of being cut off
    fin_b = tk.Button(actions, text="Finish (save all and close)", font=MID, bg="#2d4a6b", fg="#fff", activebackground="#3d6a9b",
                      relief="flat", height=2); fin_b.pack(side="right")
    newp_b = tk.Button(actions, text="New project", font=MID, bg="#333", fg="#ddd", activebackground="#555",
                       relief="flat", height=2); newp_b.pack(side="right", padx=(0, 8))
    def new_project(_=None):
        """Save this project and open the window again on a new, empty one (a new part, or a fresh try at the same)."""
        if st["busy"] or st.get("building"): return
        from tkinter import messagebox
        if sides and not messagebox.askyesno("New project", "Save this project and start a new one?"): return
        st["restart_new"] = True; st["quit"] = True
    newp_b.configure(command=new_project)

    tk.Label(side_f, text="Sides", font=BIG, bg="#181818", fg="#eee").pack(pady=(10, 2))
    tk.Label(side_f, text="click one to pick it", font=("DejaVu Sans", 10), bg="#181818", fg="#888").pack()
    side_btns = tk.Frame(side_f, bg="#181818"); side_btns.pack(side="bottom", fill="x", padx=8, pady=8)   # always in view
    learn_b = tk.Button(side_btns, text="Learn bed", font=MID, bg="#333", fg="#ddd", activebackground="#555", relief="flat", height=2)
    learn_b.pack(fill="x", pady=(0, 6))
    # the cards scroll (bar or mouse wheel): past about seven sides the list ran off the window and took the buttons with it
    cards_wrap = tk.Frame(side_f, bg="#181818"); cards_wrap.pack(fill="both", expand=True, padx=(8, 2), pady=6)
    cards_cv = tk.Canvas(cards_wrap, bg="#181818", highlightthickness=0)
    cards_sb = tk.Scrollbar(cards_wrap, orient="vertical", command=cards_cv.yview, bg="#333", troughcolor="#181818")
    cards_cv.configure(yscrollcommand=cards_sb.set)
    cards_sb.pack(side="right", fill="y"); cards_cv.pack(side="left", fill="both", expand=True)
    cards_f = tk.Frame(cards_cv, bg="#181818")
    cards_win = cards_cv.create_window((0, 0), window=cards_f, anchor="nw")
    cards_f.bind("<Configure>", lambda e: cards_cv.configure(scrollregion=cards_cv.bbox("all")))
    cards_cv.bind("<Configure>", lambda e: cards_cv.itemconfigure(cards_win, width=e.width))
    def wheel(w):
        w.bind("<Button-4>", lambda e: cards_cv.yview_scroll(-2, "units")); w.bind("<Button-5>", lambda e: cards_cv.yview_scroll(2, "units"))
    wheel(cards_cv); wheel(cards_f)
    new_b = tk.Button(side_btns, text="+ New side", font=MID, bg="#2e5d2e", fg="#fff", activebackground="#3e7d3e", relief="flat", height=2)
    new_b.pack(fill="x", pady=(0, 6))
    build_b = tk.Button(side_btns, text="Build model from all sides", font=MID, bg="#5a3d8a", fg="#fff", activebackground="#7a5daa",
                        relief="flat", height=2)
    build_b.pack(fill="x", pady=(0, 6))
    view_b = tk.Button(side_btns, text="View model in 3D", font=MID, bg="#2d4a6b", fg="#fff", activebackground="#3d6a9b",
                       relief="flat", height=2)
    view_b.pack(fill="x", pady=(0, 6))
    def view_model(_=None):
        """The built model in its own 3D window (PointYoink's viewer, a separate process: drag, scroll, right-drag)."""
        m = os.path.join(proj, "model.ply"); need = os.path.join(proj, "model_needs.ply")
        if os.path.exists(need) and os.path.getmtime(need) >= os.path.getmtime(m): m = need      # red = scan here
        if os.path.exists(m):
            subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer.py"), m, "built model (red = scan here)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    view_b.configure(command=view_model)
    del_b = tk.Button(side_btns, text="Delete picked side", font=("DejaVu Sans", 11), bg="#4a2222", fg="#ddd", activebackground="#6a3232", relief="flat")
    del_b.pack(fill="x")
    # what the build and Auto gain have to say, in full: the big button only has room for a short step name
    log_t = tk.Text(side_btns, height=7, width=1, wrap="word", font=("DejaVu Sans", 9), bg="#0c0c0c", fg="#9a9a9a",
                    relief="flat", highlightthickness=0, state="disabled", cursor="arrow")
    log_t.pack(fill="x", pady=(8, 0))
    pbar = tk.Canvas(side_btns, height=16, bg="#0c0c0c", highlightthickness=0)       # build progress, shown only while building
    # where each build step sits on the bar (rough shares of the time: every side, every pair, then the rest)
    SPAN = {"side": (0.0, 0.35), "table": (0.0, 0.35), "align": (0.35, 0.70), "aligned": (0.35, 0.70), "base": (0.70, 0.72),
            "refine": (0.72, 0.80), "fuse": (0.80, 0.95), "done": (1.0, 1.0)}
    def draw_bar():
        f = st.get("build_frac")
        if f is None:
            if pbar.winfo_ismapped(): pbar.pack_forget()
            return
        if not pbar.winfo_ismapped(): pbar.pack(fill="x", pady=(6, 0), before=log_t)
        w = max(1, pbar.winfo_width()); pbar.delete("all")
        pbar.create_rectangle(0, 0, int(w * f), 16, fill="#7a5daa", width=0)
        pbar.create_text(w // 2, 8, text="%d%%   %d:%02d" % (100 * f, *divmod(int(time.time() - st["build_t0"]), 60)), fill="#fff", font=("DejaVu Sans", 9))
    def drain_log():
        lines = []
        while True:
            try: lines.append(log_q.get_nowait())
            except queue.Empty: break
        if not lines: return
        log_t.configure(state="normal"); log_t.insert("end", "\n".join(lines) + "\n")
        if int(log_t.index("end-1c").split(".")[0]) > 500: log_t.delete("1.0", "100.0")
        log_t.see("end"); log_t.configure(state="disabled")
    card_imgs = []

    def side_no(sd): return sides.index(sd) + 1

    def refresh_cards():
        for w in cards_f.winfo_children(): w.destroy()
        card_imgs.clear()
        for sd in sides:
            picked = sd is sel[0]
            f = tk.Frame(cards_f, bg="#2f6f2f" if picked else "#262626", padx=4, pady=4, cursor="hand2"); f.pack(fill="x", pady=4)
            a_ = sd.rgb_thumb if sd.rgb_thumb is not None else np.full((120, 160, 3), 40, np.uint8)
            b_ = sd.model_thumb if sd.model_thumb is not None else np.full((120, 160, 3), 24, np.uint8)
            pic = np.hstack([cv2.resize(a_, (150, 112)), cv2.resize(b_, (150, 112))])
            im = ImageTk.PhotoImage(Image.fromarray(pic)); card_imgs.append(im)
            l1 = tk.Label(f, image=im, bg=f["bg"]); l1.pack()
            txt = "Side %d   %d frames%s" % (side_no(sd), sd.frames, ("   %d lost" % sd.lost) if sd.lost else "")
            if picked: txt += "   (picked)"
            l2 = tk.Label(f, text=txt, font=("DejaVu Sans", 11, "bold"), bg=f["bg"], fg="#fff"); l2.pack(anchor="w")
            ws = [f, l1, l2]
            if sd.frames:                                            # where its base cut comes from
                ws.append(tk.Label(f, text="   base: bed" if sd.bed.found else "   base: Auto", font=("DejaVu Sans", 10),
                                   bg=f["bg"], fg="#c9a8ff" if sd.bed.found else "#999"))
                ws[-1].pack(anchor="w")
            for w in ws: w.bind("<Button-1>", lambda e, s_=sd: pick(s_)); wheel(w)
        if sel[0] in sides and len(sides) != st.get("cards_n"):          # a side was added or removed: bring the picked one into view
            st["cards_n"] = len(sides); cards_f.update_idletasks(); cards_cv.configure(scrollregion=cards_cv.bbox("all"))
            cards_cv.yview_moveto(max(0.0, (sides.index(sel[0]) - 1) / max(1, len(sides))))
        refresh_buttons()

    def refresh_learn():
        if st.get("learn") is not None: learn_b.configure(text="Stop learning", bg="#6b4a2d", state="normal"); return
        b = bedst["bed"]
        txt = "Learn bed" if b is None else ("Learn bed (now %d dots from a side)" % len(b) if b.provisional else "Bed: %d dots (learn again)" % len(b))
        learn_b.configure(text=txt, bg="#333",
                          state="disabled" if (st["recording"] or st["busy"] or st.get("building")) else "normal")

    def refresh_buttons():
        sd = sel[0]
        refresh_learn()
        if st["busy"]: go_b.configure(text=st["busy"], bg="#444", fg="#fff", state="disabled"); return
        if st.get("learn") is not None:
            s_ = st.get("learn_status") or {}
            go_b.configure(text="Learning the bed... %d of 360 deg, %d dots" % (min(360, s_.get("deg", 0)), s_.get("dots", 0)),
                           bg="#444", fg="#fff", state="disabled")
            hint.configure(text="Take the part off the turntable and let it turn. Learning stops by itself after one full turn (about 40 s).")
            for b_ in (new_b, del_b, build_b): b_.configure(state="disabled")
            return
        if st.get("building"):
            go_b.configure(text=st["building"], bg="#5a3d8a", fg="#fff", state="disabled")
            for b_ in (new_b, del_b, build_b): b_.configure(state="disabled")
            hint.configure(text="Building the model from all sides. This takes a minute or two; scanning is paused.")
            return
        build_b.configure(state="normal" if (not st["recording"] and any(sd.frames for sd in sides)) else "disabled")
        view_b.configure(state="normal" if os.path.exists(os.path.join(proj, "model.ply")) else "disabled")
        if st.get("build_msg"): hint.configure(text=st["build_msg"])
        if sd is None:
            go_b.configure(text="+ Start side %d" % (len(sides) + 1), bg="#2e7d32", fg="#fff", activebackground="#3e9d42", state="normal")
            hint.configure(text="Put the part on the table, set Near/Far so it's tinted green, then press the big button.")
        elif st["recording"]:
            go_b.configure(text="Pause side %d" % side_no(sd), bg="#b8860b", fg="#000", activebackground="#d8a62b", state="normal")
            hint.configure(text="Turn the table slowly. Drag the model panel to look around it (Home snaps back). Pause to move the scanner, or pause and + New side to flip.")
        else:
            verb = "Start" if sd.frames == 0 else "Continue"
            go_b.configure(text="%s scanning side %d" % (verb, side_no(sd)), bg="#2e7d32", fg="#fff", activebackground="#3e9d42", state="normal")
            hint.configure(text=("Side %d is picked. Continue to fill gaps in it, or press + New side after flipping the part." % side_no(sd))
                           if sd.frames else "Side %d is ready. Press the big button to start scanning it." % side_no(sd))
        del_b.configure(state="normal" if (sd is not None and not st["recording"]) else "disabled")
        new_b.configure(state="disabled" if st["recording"] else "normal")

    def busy(msg, fn):
        """Run something slow with the buttons locked and a message on the big button."""
        st["busy"] = msg; refresh_buttons(); root.update()
        try: fn()
        finally: st["busy"] = None; refresh_cards()

    def pause_side(sd, why=None):
        """Stop recording and save the side: the Pause button, the scanner gone quiet, a frame that took seconds, and a
        window error that keeps coming all take this one path, so what was recorded is always saved and the big button
        offers Continue."""
        st["recording"] = False; rec_now[0] = False
        if why: log("side %d paused: %s" % (side_no(sd), why))
        busy("Saving side %d..." % side_no(sd), lambda: save_side(sd))

    def pick(sd):
        if st["recording"] or st["busy"] or st.get("building"): return
        if sel[0] is not None and sel[0] is not sd: save_side(sel[0]); release(sel[0])
        sel[0] = sd if sd in sides else None
        st["model_img"] = sd.model_thumb if sd is not None else None; st["last"] = None; st["covered"] = None; go_home()
        refresh_cards()

    def new_side(_=None):
        if st["recording"] or st["busy"] or st.get("building") or st.get("learn") is not None: return
        if sel[0] is not None: save_side(sel[0])
        sd = make_side(); sides.append(sd); sel[0] = sd; st["model_img"] = None; st["last"] = None; st["covered"] = None; go_home()
        refresh_cards()

    def go(_=None):
        if st["busy"] or st.get("building") or st.get("learn") is not None: return
        st["build_msg"] = None
        if sel[0] is None: new_side()
        sd = sel[0]
        if st["recording"]: pause_side(sd); return
        if sd.trk is None:
            busy("Loading side %d..." % side_no(sd), lambda: load_tracker(sd))
        if sd.bed.bed is not bedst["bed"] and not sd.bed.found: sd.bed = new_sidebed()      # a bed learned since: use it
        sd.ps += 1; sd.idx = 0; st["recording"] = True; st["last"] = None; st["slow"] = False; sd.dirty = True
        sd.pass_ok = 0; st["bed_dots"] = None; sd.bed.new_pass(sd.ps)          # the table dots get a new chance every pass
        st["table"] = platter(sd) if sd.bed.found else None; st["table_pose"] = None
        refresh_cards()

    def learn_bed(_=None):
        """Learn bed: empty table, one turn, no tracking. Pressed again while learning: stop, nothing saved."""
        if st.get("learn") is not None:
            st["learn"] = None; st["learn_status"] = None; log("Learn bed stopped; nothing saved"); refresh_buttons(); return
        if st["recording"] or st["busy"] or st.get("building"): return
        if bedst["bed"] is not None and not messagebox.askyesno("Learn bed", "Learn the bed again? The old one is kept as bed_prev.json."): return
        st["learn"] = BedLearner(cal, gain=gain.get()); st["learn_status"] = None; st["build_msg"] = None
        log("Learn bed: take the part off the turntable and let it turn (one full turn)")
        refresh_buttons()

    def install_bed(b):
        """A freshly learned bed goes to the config dir (every project on this turntable) and this project; the old one
        stays as bed_prev.json. Sides not scanned yet look for the new one."""
        for path in (BedModel.config_path(), os.path.join(proj, "bed.json")):
            if os.path.exists(path): shutil.copyfile(path, os.path.join(os.path.dirname(path), "bed_prev.json"))
            b.save(path)
        bedst.update(bed=b, src="learn", prior=b.camera, miss=0, miss_said=False)
        for o in sides:
            if not o.frames: o.bed = new_sidebed()

    def learn_result(r):
        if r.get("learner") is not st.get("learn") or st.get("closing"): return     # stopped (or started again) meanwhile
        if r.get("error"): print("learn bed:", r["error"], flush=True)
        s_ = r.get("res")
        if s_ is None: return
        st["learn_status"] = s_
        if s_["state"] == "done":
            st["learn"] = None; b = r.get("bed")
            if b is None or len(b) < bedlib.LEARN_MIN_DOTS:
                msg = "Only %d dots seen. Try IR gain 3 to 6 and Learn bed again." % (0 if b is None else len(b))
            else:
                try:
                    install_bed(b)
                    msg = "Bed learned: %d dots, platter radius %.0f mm. Put the part on and start a side." % (len(b), b.radius_mm)
                    if b.self_ambiguity is not None and b.self_ambiguity > 0.6: msg += " (Its dots repeat: matching may be unsure.)"
                except Exception as e: msg = "Could not save the bed: %s" % e
            st["build_msg"] = msg; log(msg)
        elif s_["state"] == "failed":
            st["learn"] = None; st["build_msg"] = s_["msg"]; log("Learn bed: " + s_["msg"])
        refresh_buttons()

    def drain_bed(limit=20):
        """Apply the table finder's results (Tk thread only). A result that changes what a side saves and comes in
        after the side was saved (a pause: saving does not wait for the finder) writes that side's table_plane.json and
        bed_pose.json again at once, so they never stay behind what the window showed."""
        ui = not st.get("closing")                                  # after the window closed: no widgets to refresh
        for _ in range(limit):
            try: r = bed_res.get_nowait()
            except queue.Empty: return
            if r["kind"] == "learn": learn_result(r); continue
            if r["kind"] == "seed": seed_result(r); continue
            if r.get("error"): print("bed:", r["error"], flush=True)
            sd = r["side"]
            if sd not in sides or r["bed"] is not sd.bed.bed: continue
            sig = sd.bed.signature()
            if r.get("slow") == 3:
                sd.bed_off_pass = r["ps"]; log("side %d: the table finder is too slow on this PC; it stops for this pass" % side_no(sd))
            if r["kind"] == "find":
                sd.dots_seen = max(sd.dots_seen, r.get("n_dots", 0)); h = r["res"]
                if h is not None: st["bed_dots"] = (sd, r["ps"], r["idx"], h.dots_px, h.matched)
                had = sd.bed.camera_bed(r["ps"]) is not None
                for e in sd.bed.add_hit(r["ps"], r["idx"], h, r["T"]): log("side %d: %s" % (side_no(sd), e))
                if not had and sd.bed.camera_bed(r["ps"]) is not None:                  # accepted: tint it now
                    bedst["prior"] = sd.bed.camera_bed(r["ps"])
                    if sd is sel[0] and ui: st["table"] = platter(sd); refresh_cards()
            elif r["kind"] == "pool":
                if r.get("feats") is not None: sd.bed.add_pool(r["ps"], r["idx"], r["feats"], r["T"])
                for e in sd.bed.apply_pool(r["res"], r["ps"], r["idx"]):
                    log("side %d: %s" % (side_no(sd), e))
                    if sd is sel[0] and ui: st["table"] = platter(sd); refresh_cards()
            if sd.frames and sd.bed.signature() != sig:
                if sd.dirty: pass                     # recording, or not saved yet: its next save takes it along
                else: resave_platter(sd)              # already saved: bring its table_plane.json / bed_pose.json up to date now

    def seed_result(r):
        """A provisional bed learned from a side's own dots: it goes into this project only, marked provisional."""
        b = r.get("new_bed")
        if b is None:
            print("bed: no provisional bed from side %s (%s)" % (os.path.basename(r["folder"]), r.get("error") or r.get("why")), flush=True)
            return
        if bedst["bed"] is not None or r["side"] not in sides: return          # Learn bed finished meanwhile, or the side is gone
        try: b.save(os.path.join(proj, "bed.json"))
        except Exception as e: print("bed:", e, flush=True); return
        bedst.update(bed=b, src="side", prior=b.camera)
        for o in sides:
            if not o.frames: o.bed = new_sidebed()
        log("bed learned from side %d's own dots (%d dots), for this project only. Learn bed (empty table, one turn) "
            "makes one for every project." % (side_no(r["side"]), len(b)))
        if not st.get("closing"): refresh_learn()

    def finish_bed():
        """At Finish, after the window closed (blocking is fine now): stop the table finder once it is done with the job
        in hand, run the frame jobs it still had here, apply every result (they go into the saves that follow)."""
        running[0] = False; bed_thread.join(timeout=60.0)
        while True:
            try: job = bed_jobs.get_nowait()
            except queue.Empty: break
            if job["kind"] in ("find", "pool"): bed_res.put(run_job(job))       # (learning stops with the window)
        drain_bed(limit=10 ** 6)

    def finish_seeds():
        """At Finish, after the last saves: a provisional bed from a side saved just now (Finish while recording queues
        it, and the finder has stopped), here and now; the first one that works is kept, the rest are not needed."""
        while bedst["bed"] is None:
            try: job = bed_seeds.get_nowait()
            except queue.Empty: return
            seed_result(run_job(job))

    def bed_hint(sd, d, irl):
        """Before tracking a frame: on the first frame of a Continue, or after a lost frame, place the camera from the
        table dots (B_scan @ inv(B_cam)) and let ICP start there. Much faster than the feature search; ICP must still fit.
        When refused fits agree that the part moved on the table, SideBed.hint_outcome takes that move and the hints of
        this pass come from where the part now sits (guard still on)."""
        if sd.bed_off_pass == sd.ps: return
        last = st.get("last")
        if not ((sd.ps > 0 and sd.idx == 0) or (last is not None and not last["ok"])): return
        try:
            Tn = sd.bed.hint_pose(d, irl, cal, sd.ps)
            if Tn is not None: sd.trk.hint(Tn, guard_mm=bedlib.HINT_GUARD_MM, guard_deg=bedlib.HINT_GUARD_DEG)   # a far-off fit is refused
        except Exception as e: print("bed hint:", e, flush=True)

    def bed_after(sd, d, irl, r, T_mm):
        """After a frame: whether the dots still place the part (hints), whether the scanner moved (camera bed), and what
        the table finder should look at next (find while the pass is not placed, then every 10th tracked frame to check
        it; pool tracked frames 10-90 while the side has no bed)."""
        sb = sd.bed
        msg = sb.hint_outcome(r, sd.ps)
        if msg:
            sd.moved_pass = sd.ps
            # safer than guessing where the part went: pause once this frame is stored (tick's pause_why path)
            if getattr(sb, "move_pause", None) == sd.ps: st["pause_why"] = msg
            else: log("side %d: %s" % (side_no(sd), msg))
        if r["ok"]:
            for e in sb.check_camera(sd.ps, sd.idx, T_mm): log("side %d: %s" % (side_no(sd), e))
        # the part moved on the table: the dots' hits no longer match the side's placement (they would veto it), so no
        # finder jobs for the rest of the pass; the hints (hint_pose) carry on from where the part now sits
        if sb.bed is None or sb.state == "off" or sb.state.startswith("vetoed") or sd.bed_off_pass == sd.ps or sd.moved_pass == sd.ps: return
        T = T_mm if r["ok"] else None; n_ok = sd.pass_ok - (1 if r["ok"] else 0)       # tracked frames before this one
        base = dict(side=sd, ps=sd.ps, idx=sd.idx, d=d, ir=irl, bed=sb.bed)
        if sb.camera_bed(sd.ps) is None:
            if sd.idx - sb.look_from(sd.ps) <= bedlib.POOL_TO: bed_job(dict(base, kind="find", T=T, prior=sb.prior_for(sd.ps)))
        elif r["ok"] and n_ok % 10 == 0 and sb.found: bed_job(dict(base, kind="find", T=T, prior=sb.prior_for(sd.ps)))
        if sb.state == "looking" and r["ok"] and bedlib.POOL_FROM <= n_ok <= bedlib.POOL_TO and n_ok % bedlib.POOL_EVERY == 0:
            pl = sd.table.plane()
            bed_job(dict(base, kind="pool", T=T_mm, snap=sb.pool_snapshot(), extra=[(pl[0], pl[1])] if pl is not None else []))

    def delete_side(_=None):
        sd = sel[0]
        if sd is None or st["recording"]: return
        if not messagebox.askyesno("Delete side", "Delete side %d (%d frames)? This can't be undone." % (side_no(sd), sd.frames)): return
        shutil.rmtree(sd.folder, ignore_errors=True); sides.remove(sd); sel[0] = None; st["model_img"] = None
        refresh_cards()

    def build_model(_=None):
        """Save the current side, free the GPU for the build, and run build_project.py in the background.
        Its progress shows on the big button; the finished preview appears in the model panel."""
        if st["recording"] or st["busy"] or st.get("building") or st.get("learn") is not None: return
        if not [sd for sd in sides if sd.frames]: return
        for sd in list(sides): save_side(sd)          # the picked side, and any a late table-finder result changed
        for sd in sides: release(sd)
        st["building"] = "Building: starting..."; st["build_frac"] = 0.0; st["build_t0"] = time.time(); refresh_buttons()
        def worker():
            p = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_project.py"), proj],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            done = None
            for ln in p.stdout:
                if ln.startswith(("median of pair medians", "kept ")): log(ln.strip())      # refine and keep-part summaries
                if not ln.startswith("STAGE "): continue
                parts = ln.split(" ", 2)
                try: info = json.loads(parts[2])
                except Exception: info = {}
                if parts[1] == "frames":                                        # fuse's frame count: the bar creeps, the button counts
                    k, n = info.get("done", 0), max(1, info.get("total", 1))
                    if info.get("step") == "all":
                        st["build_frac"] = max(st["build_frac"], 0.80 + 0.15 * k / n); st["building"] = "Building: fusing all sides, frame %d of %d" % (k, n)
                    else:
                        sn, ns = info.get("side", 1), max(1, info.get("sides", 1))
                        st["build_frac"] = max(st["build_frac"], 0.35 * (sn - 1 + k / n) / ns)
                        st["building"] = "Building: side %d of %d, frame %d of %d" % (sn, ns, k, n)
                    continue
                if parts[1] in SPAN:                                            # move the bar: never backwards
                    lo, hi = SPAN[parts[1]]; k = info.get("done", info.get("side", 1) - 1 if "side" in info else 0); n = info.get("total", info.get("count", 1))
                    if parts[1] == "side": st["sides_n"] = n
                    if parts[1] == "table": n = st.get("sides_n", 1); k = info.get("side", 1)
                    st["build_frac"] = max(st["build_frac"], lo + (hi - lo) * min(1.0, k / max(1, n)))
                if parts[1] == "table": log("side %s: %s" % (info.get("side", "?"), info.get("msg", "")))
                elif parts[1] == "aligned": log("sides %s line up: fit %s, %s mm" % (info.get("sides"), info.get("fit"), info.get("rmse_mm")))
                elif parts[1] in ("warn", "error", "base", "done"): log("%s: %s" % (parts[1], info.get("msg") or info.get("model", "")))
                elif "msg" in info and parts[1] not in ("align",): log(info["msg"])
                if parts[1] == "table": info["msg"] = "platter cut on side %s" % info.get("side", "?")   # the cutter's report is a paragraph
                if "msg" in info:
                    m = info["msg"] if len(info["msg"]) <= 60 else info["msg"][:57].rsplit(" ", 1)[0] + "..."   # the button has 2 lines
                    st["building"] = "Building: " + m
                if parts[1] == "done": done = info
                if parts[1] == "error": st["build_msg"] = "Build failed: " + info.get("msg", "")
            p.wait()
            if done:
                try:
                    need_png = os.path.join(proj, "model_needs.png")      # the same views with the scan-here colours, when the build made them
                    if os.path.exists(need_png) and os.path.getmtime(need_png) >= os.path.getmtime(done["preview"]): done["preview"] = need_png
                    im = np.array(Image.open(done["preview"]).convert("RGB")); st["model_img"] = im; st["covered"] = None
                except Exception: pass
                st["build_msg"] = "Model built: %s" % done["model"]
            elif not st.get("build_msg"): st["build_msg"] = "Build failed (exit %d)" % p.returncode
            log(st["build_msg"])
            st["building"] = None; st["build_frac"] = None
        threading.Thread(target=worker, daemon=True).start()

    def finish(_=None): st["quit"] = True
    build_b.configure(command=build_model)
    go_b.configure(command=go); new_b.configure(command=new_side); del_b.configure(command=delete_side); fin_b.configure(command=finish)
    learn_b.configure(command=learn_bed)
    # smoke tests press buttons on a timer (no key injection, no screenshots): POINTYOINK_SCANNER_AUTOPILOT="go@8,go@40,quit@50"
    for step_ in filter(None, os.environ.get("POINTYOINK_SCANNER_AUTOPILOT", "").split(",")):
        what_, at_ = step_.split("@")
        root.after(int(float(at_) * 1000), {"go": go, "new": new_side, "learn": learn_bed, "build": build_model, "quit": finish}[what_])
    if bedst["bed"] is None:
        log("Tip: Learn bed once (empty table, one turn) and the base is found from the first second of every side.")
    root.bind("<space>", go); root.protocol("WM_DELETE_WINDOW", finish)
    print("project folder:", proj, flush=True)

    def auto_range(d):
        """Keep the Near/Far window wrapped around the part: the nearest big mass of depth in the middle of the view
        (the part and the platter under it), with room in front and behind. Moves smoothly, like the scanner's own Auto."""
        c = d[H // 5: H * 4 // 5, W // 5: W * 4 // 5]
        z = c[c > 0].astype(np.float32) * 0.1
        if z.size < 2000: return
        hist, edges = np.histogram(z, bins=np.arange(50, 1000, 10))
        big = np.nonzero(hist > 0.05 * hist.max())[0]
        if not len(big): return
        z_front = edges[big[0]]                                   # the nearest well-populated 10 mm layer
        body = z[(z >= z_front) & (z <= z_front + 250)]            # the part and its platter lie within 25 cm of that
        lo = max(50.0, float(np.percentile(body, 1)) - 40); hi = float(np.percentile(body, 99)) + 40
        # stay put unless the part has really left the window: widening waits 3 frames (the part is being cut off),
        # narrowing waits 30 and needs 30 mm to gain. Moving every frame made the cut edge, and the saved frames, flicker
        cur = st.get("rng")
        if cur is None: st["rng"] = cur = (lo, hi); st["rng_wide"] = st["rng_tight"] = 0
        else:
            wide = lo < cur[0] - 10 or hi > cur[1] + 10; tight = not wide and (lo > cur[0] + 30 or hi < cur[1] - 30)
            st["rng_wide"] = st.get("rng_wide", 0) + 1 if wide else 0
            st["rng_tight"] = st.get("rng_tight", 0) + 1 if tight else 0
            if st["rng_wide"] >= 3: st["rng"] = (min(lo, cur[0]), max(hi, cur[1])); st["rng_wide"] = 0
            elif st["rng_tight"] >= 30: st["rng"] = (lo, hi); st["rng_tight"] = 0
            else: return
        near.set(int(round(st["rng"][0] / 5) * 5)); far.set(int(round(st["rng"][1] / 5) * 5))

    def watch_stream():
        """No frames for 4 s: pause the side the way the Pause button does (it is saved) and say so, in the log box and
        in the bar (tick() redraws the bar even with no frames). Then retry the streams every 3 s once /dev/video0 is back."""
        quiet = time.monotonic() - cam["last"]
        if quiet < 4.0 and not cam["dead"]: return
        sd = sel[0]
        if st["recording"] and sd in sides and not st["busy"]:      # recording with no frames (also a Continue pressed meanwhile)
            log("no frames from the scanner for %d s: side %d paused and saved%s" % (
                quiet, side_no(sd), " (USB cable? still in PC mode?)" if not st.get("cam_msg") else ""))
            st["cam_paused"] = side_no(sd); st["cam_msg"] = "NO FRAMES from the scanner"
            pause_side(sd)
        elif not st.get("cam_msg"):
            st["cam_paused"] = None; st["cam_msg"] = "NO FRAMES from the scanner"
            log("no frames from the scanner for %d s (USB cable? still in PC mode?)" % quiet)
        there = os.path.exists("/dev/video0")
        what = ": side %d paused" % st["cam_paused"] if st.get("cam_paused") else ""
        st["cam_msg"] = ("NO FRAMES from the scanner for %d s%s, reconnecting..." % (quiet, what)) if there else \
                        ("SCANNER GONE: /dev/video0 is missing%s. Check the USB cable and PC mode; it reconnects by itself" % what)
        if there and time.monotonic() - cam["tried"] > 3.0 and (cam["dead"] or quiet > 6.0):
            cam["tried"] = time.monotonic(); log("reconnecting to the scanner")
            try: start_streams(); st["gain_set"] = None           # gain is sent again on the next frame
            except Exception as e: log("reconnect failed: %s" % e)

    TICK_ERRS_PAUSE = 3                 # rounds that failed within 10 s while recording: pause (and save) the side
    def tick():
        """One round of the window every 30 ms. Whatever goes wrong in it is logged (terminal and log box), and the next
        round is scheduled whatever happens (finally; the handler itself cannot raise): a window that stopped ticking
        stopped taking frames, and they piled up in memory. An error that keeps coming while recording pauses the side
        through the Pause button's path, so what was recorded is saved."""
        try:
            tick_once()
        except Exception:
            try: tick_failed()
            except Exception: pass
        finally:
            if st["quit"] or not running[0]:
                try: root.destroy()
                except Exception: pass
            else:
                try: root.after(30, tick)
                except Exception: pass

    def tick_failed():
        tb_ = traceback.format_exc(); key = tb_.strip().splitlines()[-1]; now_ = time.monotonic()
        # (rounds with no new frame draw nothing and pass, so the failures are counted over time, not in a row)
        st["tick_errs"] = [x for x in st.get("tick_errs", []) if now_ - x < 10.0] + [now_] if st["recording"] else []
        if key != st.get("tick_err") or now_ - st.get("tick_err_t", 0.0) > 10.0:     # the same error every 30 ms: every 10 s
            st["tick_err"] = key; st["tick_err_t"] = now_
            log("window error (it carries on):\n" + "\n".join(l for l in tb_.rstrip().splitlines() if l.strip().strip("^~")))
        sd = sel[0]
        if st["recording"] and len(st["tick_errs"]) >= TICK_ERRS_PAUSE and sd in sides and not st["busy"]:
            st["tick_errs"] = []
            pause_side(sd, "the window failed %d times in 10 s (%s); what was recorded is saved. Continue to try again, "
                           "or Finish and report it" % (TICK_ERRS_PAUSE, key))

    def say_dropped():
        """Recorded frames the pump had to drop (the window hung): into the log box, with the total, at most every 5 s
        (and once more when the drops stop). Here on the Tk thread: while it hangs nothing could be shown anyway."""
        n = dropped["n"]
        if n > dropped["said_n"] and time.monotonic() - dropped["said"] > 5.0:
            dropped["said"] = time.monotonic(); dropped["said_n"] = n
            sd = sel[0]
            log("the window did not keep up: %d recorded frames dropped so far%s (%d waiting)"
                % (n, " (holes in side %d)" % side_no(sd) if sd in sides else "", q.qsize()))

    def tick_once():
        latest = None
        rec_now[0] = st["recording"]                  # the frame pump keeps only the newest few frames while not recording
        say_dropped()
        drain_bed(); drain_log(); draw_bar()
        if st.get("orbit_dirty") and st.get("orbit") and sel[0] is not None and sel[0].trk is not None:
            st["orbit_dirty"] = False
            try: st["model_img"], st["covered"] = sel[0].trk.render(view_pose(), scale=0.8, coverage=True)
            except Exception as e: print("render:", e, flush=True)
            if latest is None and st.get("last_view") is not None: latest = st["last_view"]
        if st.get("home_dirty"):
            st["home_dirty"] = False
            if sel[0] is not None and sel[0].trk is not None and sel[0].trk.fused:
                try: st["model_img"], st["covered"] = sel[0].trk.render(sel[0].trk.T, scale=0.8, coverage=True)
                except Exception as e: print("render:", e, flush=True)
                if latest is None and st.get("last_view") is not None: latest = st["last_view"]
        for _ in range(4):                            # handle the frames that arrived (a few per redraw); show the newest
            try: t, fr = q.get_nowait()
            except queue.Empty: break
            if q.qsize() > 30 and st["recording"] and time.monotonic() - st.get("behind_t", 0.0) > 5.0:   # far behind: say so
                st["behind_t"] = time.monotonic(); log("falling behind: %d frames waiting" % q.qsize())
            st["arrivals"] = [x for x in st["arrivals"] if t - x < 5.0] + [t]
            if st.get("cam_msg"): st["cam_msg"] = None; st["cam_paused"] = None; log("frames from the scanner again")
            d = np.frombuffer(fr, np.uint16, W * H).reshape(H, W).copy()
            g_now = apply_gain()
            if auto_rng.get(): auto_range(d)
            lo_mm, hi_mm = sorted((near.get(), far.get()))
            d[(d < lo_mm * 10) | (d > hi_mm * 10)] = 0          # outside the range: gone from tracking, the view and the saved frame
            irl = np.frombuffer(fr, np.uint8, W * H, W * H * 2).reshape(H, W)
            sd = sel[0]
            if st.get("learn") is not None and not st["recording"]:            # Learn bed: every frame to the learner
                bed_job(dict(kind="learn", learner=st["learn"], d=d, ir=irl, t=t, gain=g_now))
            if st["recording"] and sd is not None and not st["busy"]:
                t_step = time.monotonic()
                bed_hint(sd, d, irl)                             # a Continue or a lost frame: start from the table dots
                r = sd.trk.step(d); st["last"] = r
                if r["ok"]: sd.pass_ok += 1
                elif r.get("refused"): print("frame %d: a fit far from where the table dots put the camera was refused" % sd.idx, flush=True)
                if time.monotonic() - t_step > 3.0:          # something is badly wrong: stop instead of freezing the window
                    st["slow"] = True; st["pause_why"] = "tracking a frame took %.1f s" % (time.monotonic() - t_step)
                name = "frame_%03d_%04d" % (sd.ps, sd.idx)
                open(os.path.join(sd.cache, name + ".dph"), "wb").write(d.tobytes())
                open(os.path.join(sd.cache, name + ".irl"), "wb").write(irl.tobytes())
                M = r["T"] @ F; M[:3, 3] *= 1000.0
                write_inf(os.path.join(sd.cache, name + ".inf"), M)
                if r["ok"]:
                    sd.entries.append((sd.ps, sd.idx, M))
                    try:
                        sd.table.add(d, cal, M @ F)
                        if sd.idx % 10 == 0:
                            for e in sd.bed.check(sd.table): log("side %d: %s" % (side_no(sd), e))     # V2, V3
                            st["table"] = platter(sd)                          # cached between new frames
                            if cycle.get(): auto_gain(sd)
                            else: st["turn"] = None; st["cycle_msg"] = ""
                        st["table_pose"] = M @ F
                    except Exception as e: st["table"] = None; print("table:", e, flush=True)
                else: sd.lost += 1; st["table_pose"] = None
                try: bed_after(sd, d, irl, r, M @ F)
                except Exception as e: print("bed:", e, flush=True)
                sd.stamps.append(dict(frame=name, t=t, ok=r["ok"], fit=r["fit"], rmse=r["rmse"], reloc=r["reloc"],
                                      near=lo_mm, far=hi_mm, led=st["led"], gain=g_now, hinted=bool(r.get("hinted"))))
                if sd.rgb_thumb is None and st["rgb_img"] is not None: sd.rgb_thumb = cv2.resize(st["rgb_img"], (320, 240))
                sd.idx += 1
                if r["ok"] and sd.idx % 3 == 1:
                    try: st["model_img"], st["covered"] = sd.trk.render(view_pose() if st["orbit"] else r["T"], scale=0.8, coverage=True)
                    except Exception as e: print("render:", e, flush=True)
                if sd.frames % 10 == 1: refresh_cards()
            latest = (d, irl); st["last_view"] = latest
            if st.get("pause_why"): break               # this frame is kept; the side is paused (and saved) below
        if st.get("pause_why"):
            why = st.pop("pause_why")
            if st["recording"] and sel[0] in sides: pause_side(sel[0], why)
        if latest is None and bool(st.get("cam_msg")) != st.get("cam_msg_drawn") and st.get("last_view") is not None:
            latest = st["last_view"]                      # no frames: redraw the last view once, framed in the warning colour
        st["cam_msg_drawn"] = bool(st.get("cam_msg"))
        if latest is not None:
            d, irl = latest; last = st["last"]; sd = sel[0]
            ir = cv2.cvtColor(irl, cv2.COLOR_GRAY2RGB)
            keep = d > 0                                   # what survives the range cut, tinted green on the IR view
            lost_now = st["recording"] and last is not None and not last["ok"]
            tint = ([10, 6, 6], [90, 0, 0]) if lost_now else ([6, 10, 6], [0, 70, 0])     # red while lost, green otherwise
            ir[keep] = (ir[keep].astype(np.uint16) * tint[0] // 10 + tint[1]).clip(0, 255).astype(np.uint8)
            tb = st.get("table") if st["recording"] else None
            # the platter the build will cut, purple. With this pass's camera bed (the table dots, camera frame) no pose is
            # needed and it is right on lost frames too; otherwise the side's one plane seen from where the camera is now
            camb = sd.bed.camera_bed(sd.ps) if (st["recording"] and sd is not None and sd.bed.found) else None
            geo = None
            if camb is not None: geo = (camb["n"], camb["off"], camb["centre"], sd.bed.bed.radius_mm)
            elif tb is not None and st.get("table_pose") is not None:
                Tm = st["table_pose"]; nc, oc = to_camera(tb[0], tb[1], Tm)
                geo = (nc, oc, None if tb[2] is None else Tm[:3, :3].T @ (tb[2] - Tm[:3, 3]), tb[3])
            if geo is not None:
                # Only pixels with depth at or under it are tinted, and the platter's rim is drawn as a ring. (Filling the
                # whole disc where there is no depth also painted every part of the part that gave no depth: seen live.)
                nc, oc, cen, rad_ = geo
                rx = (np.arange(W, dtype=np.float32) - cal[2]) / cal[0]; ry = (np.arange(H, dtype=np.float32) - cal[3]) / cal[1]
                den = rx[None, :] * nc[0] + ry[:, None] * nc[1] + nc[2]             # n . ray, ray = (rx, ry, 1)
                zf = d.astype(np.float32) * 0.1
                on = keep & (zf * den + oc < 2.0)
                ir[on] = (ir[on].astype(np.uint16) // 2 + np.array([80, 0, 110], np.uint16)).clip(0, 255).astype(np.uint8)
                if cen is not None and rad_ is not None:                          # the platter's centre, camera frame
                    e1 = np.cross(nc, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nc, e1)
                    ang = np.linspace(0, 2 * np.pi, 121)
                    ring = cen + rad_ * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
                    px = np.stack([ring[:, 0] / np.maximum(ring[:, 2], 1e-6) * cal[0] + cal[2], ring[:, 1] / np.maximum(ring[:, 2], 1e-6) * cal[1] + cal[3]], 1)
                    for k_ in range(0, 120, 2):                                   # dashed, and only what is in front of the camera
                        if ring[k_, 2] > 20 and ring[k_ + 1, 2] > 20 and np.abs(px[k_]).max() < 4000 and np.abs(px[k_ + 1]).max() < 4000:
                            cv2.line(ir, tuple(int(v_) for v_ in px[k_]), tuple(int(v_) for v_ in px[k_ + 1]), (200, 90, 255), 2, cv2.LINE_AA)
            # the table dots: green = matched to the bed, grey = not (learning: grey = new)
            show = None
            lr_ = st.get("learn")
            if lr_ is not None and lr_.last is not None: show = lr_.last[0][:, :3], lr_.last[1]
            elif st["recording"] and st.get("bed_dots") and st["bed_dots"][0] is sd and st["bed_dots"][1] == sd.ps and sd.idx - st["bed_dots"][2] <= 4:
                show = st["bed_dots"][3], st["bed_dots"][4]
            if show is not None:
                for (cx_, cy_, r_), m_ in zip(show[0], show[1]):
                    cv2.circle(ir, (int(cx_), int(cy_)), int(1.6 * r_ + 4), (60, 230, 60) if m_ else (150, 150, 150), 2, cv2.LINE_AA)
            if keep.sum() > 500:                           # the part's distance, smoothed so the outline and + glide instead of jump
                z_now = float(np.median(d[keep])) * 0.1
                st["zref"] = z_now if st.get("zref") is None else 0.85 * st["zref"] + 0.15 * z_now
            outlined_line(ir, (W // 2 - 22, H // 2), (W // 2 + 22, H // 2), (255, 220, 0)); outlined_line(ir, (W // 2, H // 2 - 22), (W // 2, H // 2 + 22), (255, 220, 0))
            dv = np.zeros((H, W, 3), np.uint8); nz = d > 0
            if nz.any():
                lo, hi = np.percentile(d[nz], [2, 98])
                v = np.clip(255 - (d[nz].astype(np.float32) - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
                dv[nz] = cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
            if rgb_latest[0] is not None:
                try:
                    im = Image.open(io.BytesIO(rgb_latest[0])); im.draft("RGB", (500, 375))   # JPEG decodes at 1/4 size: cheap
                    st["rgb_img"] = np.array(im.convert("RGB").resize((W, H)))
                except Exception: pass
            rgbv = st["rgb_img"].copy() if st["rgb_img"] is not None else np.zeros((H, W, 3), np.uint8)
            if st["rgb_img"] is not None:
                rgbv = draw_depth_guides(rgbv, d, cal, st.get("zref") or 250.0, (255, 60, 60) if lost_now else (60, 255, 60))
                if not st.get("overlay_saved"):            # one picture for checking the colour/depth alignment
                    Image.fromarray(rgbv).save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay_check.png"))
                    st["overlay_saved"] = True
            mv = st["model_img"]
            mv = cv2.resize(mv, (W, H), interpolation=cv2.INTER_NEAREST) if mv is not None else np.full((H, W, 3), 24, np.uint8)
            cov = st.get("covered")
            mname = "model: %d%% well covered" % round(100 * cov) if (cov is not None and st["model_img"] is not None) else "model so far"
            if st.get("orbit"): mname += "  (your view)"
            for img_, name_ in ((ir, "IR (green = kept, red = lost)"), (dv, "depth"), (rgbv, "color camera"), (mv, mname)):
                label(img_, name_, (16, 44), 1.3)
            if st["model_img"] is not None and cov is not None:            # key for the coverage colors
                for k_, (t_, c_) in enumerate((("barely seen: scan here", (255, 70, 50)), ("getting there", (255, 200, 40)), ("well covered", (90, 170, 255)))):
                    label(mv, t_, (16, H - 24 - 40 * (2 - k_)), 0.8, c_)
            col = bar_status()[0]
            view = np.vstack([np.hstack([ir, dv]), np.hstack([rgbv, mv])])
            c = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5)); cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), c, 10)
            img = ImageTk.PhotoImage(Image.fromarray(view).resize((1200, 900))); lab.configure(image=img); lab.image = img
        if st.get("building") != st.get("_shown_build"):
            st["_shown_build"] = st.get("building"); refresh_buttons()
        watch_stream()
        update_bar()                                  # every round, frames or not: "NO FRAMES" must show when they stop

    def bar_status():
        """(colour, message) of the status bar and the frame round the view."""
        sd = sel[0]; last = st["last"]
        if st.get("cam_msg"): return "#ff9040", st["cam_msg"]
        if st.get("slow") and not st["recording"]: return "#ff9040", "PAUSED: tracking got too slow. Press Finish and report it"
        if not st["recording"] or sd is None: return "#cccccc", ("NOT SCANNING" if sd is None else "SIDE %d PAUSED" % side_no(sd))
        if last is None or last["ok"]:
            return "#50dc50", "SIDE %d  TRACKING" % side_no(sd) + ("  (found again by the table dots)" if last and last.get("hinted")
                                                                    else "  (found again)" if last and last["reloc"] else "")
        return "#ff4040", "SIDE %d  LOST: turn back to where it was green" % side_no(sd)

    def update_bar():
        """The status bar, redrawn only when its text changes (cheap enough for every round)."""
        col, msg = bar_status(); sd = sel[0]; last = st["last"]
        now_ = time.monotonic(); arr = [x for x in st["arrivals"] if now_ - x < 5.0]
        fps = (len(arr) - 1) / max(1e-3, arr[-1] - arr[0]) if len(arr) > 1 else 0.0
        info = msg
        if sd is not None: info += "     frames %d   kept %d   lost %d" % (sd.frames, len(sd.entries), sd.lost)
        info += "   %.1f fps" % fps
        if last and last["ok"] and st["recording"]: info += "   fit %.2f   err %.2f mm" % (last["fit"], last["rmse"])
        if st["recording"] and sd is not None: info += "     " + base_txt(sd)
        if st.get("cycle_msg") and st["recording"]: info += "     " + st["cycle_msg"]
        if (info, col) != st.get("bar_shown"):
            st["bar_shown"] = (info, col); bar.configure(text=info, fg=col)

    refresh_cards()
    root.after(30, tick); root.mainloop()
    if set_fill_led is not None:                      # don't leave the light on after the window closes (no add-on: no LED)
        try: set_fill_led(False)
        except Exception as e: print("LED:", e, flush=True)
    set_ir_gain(1)                                    # leave the scanner at its default gain
    st["closing"] = True; st["recording"] = False; rec_now[0] = False
    finish_bed()                                      # the table finder's last jobs and results go into the saves below
    stop_streams()
    for sd in list(sides): save_side(sd)
    finish_seeds()                                    # no bed yet: a provisional one from a side saved just now
    for sd in list(sides): release(sd)             # free the GPU before Python tears down (a live model made Open3D abort at exit)
    if not sides: shutil.rmtree(proj, ignore_errors=True); print("nothing saved")
    else:
        print("project %s: %d side(s)" % (proj, len(sides)))
        for i, sd in enumerate(sides): print("  side %d: %d frames, %d tracked -> %s" % (i + 1, sd.frames, len(sd.entries), sd.folder))
    if st.get("restart_new"):                         # New project: the same window and options, with no project to continue
        os.execv(sys.executable, [sys.executable, "-u", os.path.abspath(__file__)] + restart_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
