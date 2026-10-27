# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where Learn bed (scanner/bed.py: the empty platter learned once, found again in every side for base removal from the
first frame) plugs into a scan session. The session calls these on its worker thread only, at the same points the Tk
scanner does; the hook keeps its per-side state in side.bed. NullBed is the scanner without a bed: Auto remove base
only. load_bed_hook() returns the bed adapter (app2/scan/bed_adapter.py, scanner/bed.py behind it) when it imports,
else NullBed, so the app runs the same with or without it. app2 never imports scanner/bed.py itself; only the adapter
does, and the adapter keeps every config write under POINTYOINK_CONFIG.

Call points (Tk scanner of 2026-09-27 -> hook):
  startup bed lookup (project bed.json, then the config dir's)     open(project, param_dir, cal, log)
  Side(): SideBed.load(bed_pose.json) / new_sidebed()              side_opened(side, saved)
  load_tracker(): the bed as saved, when a side is re-seeded       side_loaded(side)
  go(): a bed learned since -> new SideBed                          pass_started(side)
  bed_hint() before tracker.step()                                  before_step(side, d, irl, tracker, last)
  bed_after() after the step (find / pool jobs; the part moved on
    the table: pause the pass)                                      after_step(side, d, irl, r, T_mm) -> (lines, why or None)
  every 10th tracked frame: sd.bed.check(sd.table)                  every10(side) -> log lines
  platter(): the table dots' platter first                          platter(side), found(side)
  save_platter(): table_plane.json "source": "bed", bed_pose.json,
    bed copy into the project, "bed not recognised", provisional
    bed learned from a side                                         save(side, path, auto_plane, no), after_save(...)
  drain_bed(): apply finder results                                 drain(sides, picked, no) -> (log lines, changed)
  before a save: the finder's queued jobs (briefly), its results
    applied; closing (finish_bed): the finder stops, what it still
    had runs here                                                   settle(sides, picked, no, final=False) -> (lines, changed)
  Learn bed button, frames to the learner, learn_result()           learn_toggle(gain, busy), learn_frame(d, irl, t, gain),
                                                                    take_message() -> Learn bed's result for the hint line
  purple tint from the camera bed, dot circles                      camera_geo(side, recording), dots(side, recording)
  bar "base: ..." text, card "base: bed / Auto"                     base_text(side, recording), card_base(side)
  "Learn bed (now N dots from a side)"                              provisional
  after the last saves (finish_seeds), then the finder ends         close() -> log lines
  the Bed card and its big picture (2.0 only)                       summary() -> BedInfo or None
  the Bed card's list of bed profiles, picking one (2.0 only)       profiles(), profile_name, pick_profile(name, sides)
Before any thread starts (the Tk scanner's main()): quiet_threads()."""
from collections import namedtuple

# What the window shows of the bed (the Bed card, the big bed picture): immutable and comparable, so it rides in the
# ScanState. dots: ((x, y) mm, ...) in the platter plane, the platter centre at 0; radius, dot_mm: mm (dot_mm may be
# None); learned: "YYYY-MM-DDTHH:MM:SS"; provisional: learned from a side's own dots (this project only), side: that
# side's folder name (or None); name: the bed profile's name (or None); kind: "dots" or "plane" (no dots); outline:
# bed.Outline.key(), ("circle", cx, cy, r) or ("rect", cx, cy, width, depth, angle deg) in the same plane (None: a
# circle of radius about the centre).
FREEHAND = "Freehand (no bed)"       # the bed list's last entry: no bed is looked for, hinted from or cut (2026-10-25)

BedInfo = namedtuple("BedInfo", "id dots radius dot_mm learned provisional side name kind outline", defaults=(None, "dots", None))


class NullBed:
    """No bed module: Auto remove base only. Learn bed stays hidden; no bed_pose.json or bed.json is ever written."""
    available = False
    learning = False
    learn_status = None
    size = None                         # dots in the learned bed, or None
    provisional = False                 # the bed was learned from a side's own dots (this project only)

    def open(self, project, param_dir, cal, log): pass
    def side_opened(self, side, saved): side.bed = None
    def side_loaded(self, side): pass
    def pass_started(self, side): pass
    def before_step(self, side, d, irl, tracker, last): pass
    def after_step(self, side, d, irl, r, T_mm): return [], None     # (log lines, why the pass must pause or None)
    def every10(self, side): return []
    def platter(self, side): return None
    def found(self, side): return False
    def save(self, side, path, auto_plane, side_no): return False       # True: it wrote table_plane.json itself
    def after_save(self, side, auto_plane, side_no, sides): return []
    def drain(self, sides, picked, side_no): return [], False            # (log lines, the picked side's platter changed)
    def settle(self, sides, picked, side_no, final=False): return [], False     # drain, after waiting (briefly) for queued jobs
    def learn_toggle(self, gain, busy, name=None): return None           # a log line, or None
    def learn_frame(self, d, irl, t, gain): pass
    def take_message(self): return None                                  # Learn bed's result for the hint line, once
    def camera_geo(self, side, recording): return None
    def dots(self, side, recording): return None
    def base_text(self, side, recording): return "base: Auto (no bed learned)"
    def card_base(self, side): return "auto"                              # "bed" or "auto"
    def summary(self): return None                                       # the bed as a BedInfo, or None
    def profiles(self): return ()                                        # the bed profiles' names
    profile_name = None                                                  # the picked profile's name, or None
    def pick_profile(self, name, sides): return None                     # a log line, or None
    def close(self): return []                                           # log lines


def quiet_threads():
    """With the bed adapter: one OpenCV / BLAS thread for the process (bed.quiet_threads; the table finder's small
    matrices made the thread pools spin at 9x the CPU). On the main thread before any other thread starts: threadpoolctl
    walks the loaded libraries, and from another thread that deadlocked against the main thread loading one (Tk scanner)."""
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_adapter.py")): return
    try:
        from app2.scan import bed_adapter
        bed_adapter.quiet_threads()
    except Exception as e: print("bed adapter: threads not limited (%s: %s)" % (type(e).__name__, e), flush=True)


def load_bed_hook():
    """The Learn bed adapter when it is there and imports, else NullBed (an adapter that fails to load says why)."""
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_adapter.py")): return NullBed()
    try:
        from app2.scan.bed_adapter import ScannerBed
        return ScannerBed()
    except Exception as e:
        print("bed adapter did not load (%s: %s); Auto remove base only" % (type(e).__name__, e), flush=True)
        return NullBed()
