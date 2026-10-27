# SPDX-License-Identifier: AGPL-3.0-or-later
"""A live-scan project and its sides, in the Tk scanner's layout (fmt.py). Pure data and files: no window, no tracker
(the session hands in the model picture when a side is saved). Owned by the session's worker thread."""
import glob, os, shutil, time
from app2.scan import fmt
from table_plane import SidePlane


class Side:
    """One side of the part: a scan folder in the project. entries: tracked frames [(pass, frame, 4x4 mm)]; stamps: one
    per recorded frame (tracked or lost); ps: the current pass (-1 before the first); idx: next frame in the pass.
    version counts changes to the thumbs, so the window only redraws a card's pictures when they changed."""
    def __init__(self, folder):
        from app2.scan.scan_mode import read
        self.scan_profile=read(folder);self.motion_calibration=None
        self.folder = folder; self.cache = os.path.join(folder, "cache")
        self.entries = []; self.stamps = []; self.ps = -1; self.idx = 0; self.lost = 0
        self.rgb_thumb = None; self.model_thumb = None; self.dirty = False; self.version = 0
        self.table = SidePlane()                          # Auto remove base: the platter as seen in this side's frames
        self.pass_ok = 0; self.bed = None                 # bed: the bed hook's state for this side
        self.bed_plane_said = False                       # "its base cut came from the bed" was logged
        # Touch up: a side tracked in the built model's frame (touchup.json, model_frames.py) and, while it is loaded,
        # the model it is tracked on (touchup.TouchModel)
        import model_frames
        self.touch_info = model_frames.touch_info(folder); self.touch_model = None
        saved = fmt.read_side(folder)
        if saved is not None:                             # a saved side: its frames, poses and pictures
            self.entries, self.stamps, self.rgb_thumb, self.model_thumb = saved
            self.lost = sum(1 for x in self.stamps if not x.get("ok"))
            # the last pass from every recorded frame, lost ones too: a pass that tracked nothing still wrote its frames,
            # and the next Continue must not write over them (the Tk scanner looked at the tracked ones only)
            self.ps = max([e[0] for e in self.entries] + [int(x["frame"][6:9]) for x in self.stamps
                                                          if str(x.get("frame", "")).startswith("frame_")]
                          + fmt.discarded_passes(folder) or [-1])
        # Back 5 / 10 / 30 while paused: each step's (stamps, entries) taken back, newest last. Their files wait in
        # discarded/ so Forward can put them back; the next Start (or picking another side, a build, closing) forgets
        # the steps and the files stay there for good, in no table
        self.rewound = []

    @property
    def frames(self): return len(self.stamps)

    @property
    def rewind_room(self):
        """How many frames Back may take: all but those up to and including the first tracked one (it holds the side's
        scan frame; a side with no tracked frame left would be removed)."""
        first = next((k for k, x in enumerate(self.stamps) if x.get("ok")), None)
        return 0 if first is None or not self.entries else len(self.stamps) - first - 1

    @property
    def rewound_frames(self): return sum(len(s) for s, _ in self.rewound)

    @property
    def name(self): return os.path.basename(self.folder)


class Project:
    def __init__(self, path, param_dir, w=fmt.W, h=fmt.H):
        self.path = path; self.param_dir = param_dir; self.w, self.h = w, h
        self.made_here = not os.path.isdir(path) or not os.listdir(path)    # only a folder this session made may be removed
        self.made_sides = []
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        self.sides = [Side(f) for f in fmt.saved_sides(path)]

    @staticmethod
    def new_path(projects_dir):
        """A new ProjectLive<stamp> folder name. New project in the same second as the last one: wait for the next
        second (else the "new" project would be the old one, sides and all)."""
        p = os.path.join(projects_dir, fmt.PROJECT_PREFIX + fmt.stamp_now())
        while os.path.exists(p): time.sleep(0.2); p = os.path.join(projects_dir, fmt.PROJECT_PREFIX + fmt.stamp_now())
        return p

    @staticmethod
    def newest(projects_dir):
        """--last: the newest ProjectLive folder (by change time, as the Tk scanner picks it)."""
        old = sorted(glob.glob(os.path.join(projects_dir, fmt.PROJECT_PREFIX + "*")), key=os.path.getmtime)
        return old[-1] if old else None

    def side_no(self, sd): return self.sides.index(sd) + 1

    def make_side(self):
        sd = Side(fmt.new_side_folder(self.path, self.param_dir, self.w, self.h)); self.sides.append(sd); self.made_sides.append(sd); return sd

    def delete_side(self, sd):
        shutil.rmtree(sd.folder, ignore_errors=True)
        if sd in self.sides: self.sides.remove(sd)

    def save_side(self, sd, model_thumb=None, bed=None, log=print):
        """Write the side's pose table, report and pictures, and its platter. A side that never tracked a frame is
        removed. model_thumb: the side's model rendered now (the session renders it on the tracker's thread)."""
        if not sd.dirty: return False
        if not sd.entries:
            self.delete_side(sd); return False
        fmt.write_side_report(sd.cache, sd.entries, sd.stamps)
        if getattr(sd,"scan_profile",{}).get("mode") in ("near","far"):
            from app2.scan.scan_mode import write
            write(sd.folder,sd.scan_profile)
        if model_thumb is not None: sd.model_thumb = model_thumb; sd.version += 1
        fmt.write_thumbs(sd.folder, sd.rgb_thumb, sd.model_thumb)
        try: self.save_platter(sd, bed, log)
        except Exception as e: log("table: %s" % e)
        from app2.scan.turn_coverage import save as save_turn
        save_turn(sd)
        sd.dirty = False
        log("saved side %s: %d frames, %d tracked, %d lost" % (sd.name, sd.frames, len(sd.entries), sd.lost))
        return True

    def rewind(self, sd, n):
        """Back n frames (at most rewind_room): the newest recorded frames, tracked or lost, leave the side's stamps and
        pose entries, and their files move to discarded/. Returns (stamps, entries) taken, or None. The caller saves
        the side (pose table, track json) and fixes the model."""
        n = min(int(n), sd.rewind_room)
        if n <= 0: return None
        st = sd.stamps[-n:]; names = {x["frame"] for x in st}
        m = 0
        while m < len(sd.entries) and fmt.frame_name(*sd.entries[-1 - m][:2]) in names: m += 1
        en = sd.entries[len(sd.entries) - m:]
        fmt.move_frames(sd.cache, os.path.join(sd.folder, fmt.DISCARDED), [x["frame"] for x in st])
        del sd.stamps[-n:]; del sd.entries[len(sd.entries) - m:]
        tb = sd.table                                     # Auto remove base: one sample per tracked frame, in order
        if tb is not None and m and len(tb.pts) >= m: del tb.pts[-m:]; del tb.centres[-m:]; tb._cache = (None, None)
        sd.lost = sum(1 for x in sd.stamps if not x.get("ok")); sd.dirty = True
        sd.rewound.append((st, en)); return st, en

    def forward(self, sd):
        """Undo the newest Back: its frames come back from discarded/ into the side. Returns (stamps, entries) or None.
        The caller adds the entries to the model (and Auto remove base) and saves the side."""
        if not sd.rewound: return None
        st, en = sd.rewound.pop()
        fmt.move_frames(os.path.join(sd.folder, fmt.DISCARDED), sd.cache, [x["frame"] for x in st])
        sd.stamps.extend(st); sd.entries.extend(en)
        sd.lost = sum(1 for x in sd.stamps if not x.get("ok")); sd.dirty = True
        return st, en

    def forget_rewound(self, sd):
        """The Back steps are final (a Start, another side picked, a build, closing): Forward can no longer bring them
        back; their files stay in discarded/. Returns how many frames that was."""
        n = sd.rewound_frames; sd.rewound = []; return n

    def save_platter(self, sd, bed, log):
        """table_plane.json for the build: the table dots' plane when the bed hook writes one, else Auto remove base's
        (marked "source": "auto" only when a bed is learned, as in the Tk scanner), else none. The build reads only n, d
        and keep_above. A plane that came from the bed (a side scanned by the Tk scanner with Learn bed) is left as it
        is when this session has no bed to write one: the side's scan frame does not change on a Continue, so it still
        fits, and Auto remove base must not replace it."""
        tp = os.path.join(sd.folder, "table_plane.json"); n_ = self.side_no(sd) if sd in self.sides else 0
        auto = sd.table.plane()
        if bed is None or not bed.save(sd, tp, auto, n_):
            if (bed is None or not bed.available) and fmt.plane_source(tp) == "bed":
                if not sd.bed_plane_said:
                    sd.bed_plane_said = True; log("side %d: its base cut came from the bed (Learn bed) and is kept as it was" % n_)
            else:
                fmt.write_auto_plane(sd.table, tp, source=bed is not None and bed.size is not None)
        if bed is not None:
            for e in bed.after_save(sd, auto, n_, self.sides): log(e)

    def model_path(self):
        """The built model to show: model_needs.ply (red = scan here) when it is at least as new as model.ply."""
        m = os.path.join(self.path, "model.ply"); need = os.path.join(self.path, "model_needs.ply")
        if os.path.exists(need) and os.path.exists(m) and os.path.getmtime(need) >= os.path.getmtime(m): return need
        return m if os.path.exists(m) else None

    def side_mesh(self, sd):
        """The side's own model from the last Build model (build/side<k>.ply: the part after the platter cut, in the
        side's scan frame, mm), or None when there is none or the side was scanned again since. k is the build's own
        number for the side's folder (side numbers shift when a side is deleted): build/sides_cache.json names the
        folder each side<k>.ply was made from ("version|folder|..."), build/beds.json too."""
        import json
        work = os.path.join(self.path, "build"); k = None
        try:
            with open(os.path.join(work, "sides_cache.json")) as f: cache = json.load(f)
            k = next((int(n[4:]) for n, v in cache.items() if n.startswith("side") and str(v).split("|")[1:2] == [sd.name]), None)
        except (OSError, ValueError, TypeError, AttributeError): pass
        if k is None:
            try:
                with open(os.path.join(work, "beds.json")) as f: beds = json.load(f).get("sides") or {}
                k = next((int(n) for n, e in beds.items() if (e or {}).get("folder") == sd.name), None)
            except (OSError, ValueError, TypeError, AttributeError): k = None
        p = os.path.join(work, "side%d.ply" % k) if k is not None else self._stages_side_mesh(sd)
        if not p: return None
        try:
            if os.path.getmtime(p) < os.path.getmtime(os.path.join(sd.cache, fmt.TRACK_JSON)): return None
        except OSError: return None
        return p

    def _stages_side_mesh(self, sd):
        """The staged build's own result for this side (the
        old build/side<k>.ply is gone, so a saved side fell back to its thumbnail): the newest stored side.keep@<folder>
        (the part after the platter cut), else side.fuse@<folder> (the fused side). Both are in the side's scan frame."""
        nodes = os.path.join(self.path, "stages", "nodes")
        for step, name in (("side.keep@", "side.ply"), ("side.fuse@", "raw.ply")):
            d = os.path.join(nodes, step + sd.name); best = None
            try: keys = os.listdir(d)
            except OSError: continue
            for key in keys:
                p = os.path.join(d, key, name); m = os.path.join(d, key, "meta.json")
                try:
                    if not os.path.isfile(m): continue
                    t = os.path.getmtime(p)
                except OSError: continue
                if best is None or t > best[0]: best = (t, p)
            if best: return best[1]
        return None

    def finish(self):
        """Closing: sides that never recorded a frame (a + New side never started) are removed, then a project with no
        sides left (else --last would open an empty project). Returns its sides' summary lines (the Tk scanner's)."""
        for sd in [s for s in self.made_sides if s in self.sides and not s.stamps and not glob.glob(os.path.join(s.cache, "*.dph"))]: self.delete_side(sd)
        if not self.sides and self.made_here:
            shutil.rmtree(self.path, ignore_errors=True); return ["nothing saved"]
        if not self.sides: return ["nothing saved"]       # someone else's folder (a scanner project, a picked folder): never removed
        return ["project %s: %d side(s)" % (self.path, len(self.sides))] + [
            "  side %d: %d frames, %d tracked -> %s" % (i + 1, sd.frames, len(sd.entries), sd.folder) for i, sd in enumerate(self.sides)]
