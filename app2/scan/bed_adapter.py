# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learn bed in the 2.0 Scan step: scanner/bed.py driven the way the Tk scanner (scanner/pointyoink_scanner.py) drives
it, behind the bed hook (bedhook.NullBed is the protocol). This is the only app2 module that imports bed.py.

What it does, each part as the Tk window does it:
  - the bed: the project's own bed.json first, else the one in the config dir (POINTYOINK_CONFIG's folder, else
    ~/.config/pointyoink). Learn bed installs a new one into both (the old ones stay as bed_prev.json); with no bed at
    all a side's own dots give a provisional one, written into that project only.
  - bed profiles: several named beds in the config dir (bed.py's bed-<name>.json; the single learned bed of before
    becomes "Round dot mat" the first time). The Bed card picks one (pick_profile); the project remembers it
    (bed_profile.json, with its bed.json) and opens with it again. A "plane" bed (no dots: the bare turntable) is never
    looked for in the IR picture and never gets a provisional bed from a side: its base comes from Auto remove base.
  - the table finder: its own thread (find / pool / learn / seed jobs in, results out). The scan worker applies the
    results (drain); the finder only ever sees copies of a frame and never touches the tracker (its CUDA model stays on
    the scan worker, where the hints are handed to it).
  - per side (Side.bed = SideState): the SideBed (bed_pose.json), the dots it has seen, the pass the finder stopped
    for, the pass the part moved in, whether a provisional bed was tried from it.
  - the platter from frame 2-3 (the tint, its rim, "base: bed (N dots)"), hints on a Continue and on lost frames, the
    part moved on the table -> the pass pauses (bed.HINT_ON_MOVE "pause"), the scanner moved -> the table is looked for
    again, table_plane.json "source": "bed" and bed_pose.json written with the Tk scanner's bytes.
With no bed learned it writes what NullBed writes (table_plane.json as SidePlane saves it, no bed_pose.json) and sends
the finder nothing but the provisional-bed job."""
import json, os, queue, shutil, threading, time
from types import SimpleNamespace
import numpy as np
import bed as bedlib
from bed import BedModel, SideBed, BedLearner
from app2.scan.bedhook import BedInfo

BED_CUTS = True             # the table dots' plane goes into table_plane.json when the bed was found (the Tk scanner's gate)
SETTLE_S = 1.0              # a save waits at most this long for the finder's queued frame jobs (then late results re-save)
JOIN_S = 10.0               # closing: how long the finder may take to finish the job in hand
SLOW_S = 1.0                # a finder call slower than this skips the next frames' jobs; the 3rd one stops the finder for the pass


def quiet_threads():
    """One thread for OpenCV and for numpy's BLAS in this process (bed.quiet_threads: the finder's small matrices made
    the pools spin at 9x the CPU). Call it on the main thread before any other thread starts: threadpoolctl walks the
    loaded libraries, and doing that from another thread deadlocked against the main thread loading one."""
    return bedlib.quiet_threads()


def _shape(bed, x_axis):
    """The platter's size for the (n, off, centre, radius) tuples: the radius, as before, for a circle about the spin
    centre; else the outline placed with the bed frame's x axis (bed.PlacedOutline: float() is its reach)."""
    o = bed.outline
    return o.radius if o.centred_circle else o.placed(x_axis)


def learned_text(b):
    """ "29 dots kept (3 shapes that were not round dots left out), platter 194 mm across" for Learn bed's message."""
    q = getattr(b, "quality", None) or {}; o = getattr(b, "outline", None)
    t = "%d dots kept" % len(b)
    if q.get("not_round"): t += " (%d shapes that were not round dots left out)" % q["not_round"]
    if o is not None and o.shape == "rect": t += ", bed %.0f x %.0f mm" % (o.width, o.depth)
    else: t += ", platter radius %.0f mm" % b.radius_mm
    if getattr(b, "board", None) == "ringed": t += ", %d dots in rings" % int(q.get("ringed", 0))
    return t


class SideState:
    """What the bed keeps per side (Side.bed). sb: the side's SideBed; dots_seen: the most table dots one frame showed;
    off_pass: the pass the finder was too slow in (no more jobs in it); moved_pass: the pass the part moved on the table
    in; seed_tried: a provisional bed was asked for from this side; src: what the last save wrote ("bed", "auto", None)."""
    __slots__ = ("sb", "dots_seen", "off_pass", "moved_pass", "seed_tried", "src", "pose_guide", "find_miss", "miss_said", "rim_bad", "fresh")

    def __init__(self, sb):
        self.sb = sb; self.dots_seen = 0; self.off_pass = None; self.moved_pass = None; self.seed_tried = False; self.src = None
        self.pose_guide=None; self.find_miss = 0; self.miss_said = None    # finder results in a row with dots in view and no match; the pass it was said in
        self.rim_bad = 0; self.fresh = None                                # hits in a row off this pass's camera bed; the newest matched hit (ps, idx, Hit)


class ScannerBed:
    """The bed hook with scanner/bed.py behind it. Every method runs on the session's scan worker except the finder's
    own loop; learning is read from the frame source's thread too (a plain attribute read)."""
    available = True

    def __init__(self):
        # an app started before this file existed has the older session in memory (it neither pauses on a part move nor
        # limits the threads): it keeps running without the bed until it is started again
        from app2.scan import session
        if not hasattr(session.ScanSession, "_pause"): raise ImportError("the running app is older than the bed adapter; start it again")
        self.bed = None; self.src = None; self.prior = None; self.miss = 0; self.miss_said = False
        # the picked profile: its name; the bed when it is a plane (no dots), or a dots bed not learned yet (pending)
        self.plane = None; self.pending = None; self.profile = None; self.learn_name = None; self._profiles = ()
        self.project = None; self.param = None; self.cal = None; self.log = print
        self.learner = None; self.learn_status = None; self._msg = None; self.shown_dots = None
        from app2.scan.color_tracking import ColorObserver
        from app2.scan.color_bed import solve_pose
        from app2.scan.preview import Composer
        from app2.scan.sources import COLOR_SYNC_S
        self.color_observer=ColorObserver(Composer.COVER_S,solve_pose,interval=COLOR_SYNC_S)
        self.color_packet=None;self.color_cal=None;self.color_sync_s=COLOR_SYNC_S
        # the finder: frame jobs wait in a queue of 2 (when it falls behind the oldest goes: the newest frame is what
        # matters); a provisional bed (seconds) waits in its own queue so frame jobs never push it out
        self.jobs = queue.Queue(maxsize=2); self.res = queue.Queue(); self.seeds = queue.Queue()
        self.running = False; self.thread = None; self.closing = False; self.closed = False
        self._auto_pending = {}
        self._pending = 0; self._cv = threading.Condition()          # frame jobs queued or running (settle waits for them)
        self._sides = []; self._side_no = None                       # the session's list, for results applied at close
        self.ran = dict(find=0, pool=0, learn=0, seed=0)             # jobs the finder ran (tests read it)

    # ------------------------------------------------------------------ what the session reads
    @property
    def learning(self): return self.learner is not None

    @property
    def size(self): return None if self.bed is None else len(self.bed)

    @property
    def provisional(self): return bool(self.bed is not None and self.bed.provisional)

    def summary(self):
        """The bed for the window (bedhook.BedInfo), made once per bed; None with no bed."""
        b = self.bed if self.bed is not None else getattr(self, "plane", None)
        if b is None: return None
        c = getattr(self, "_summary", None); name = getattr(self, "profile", None) or getattr(b, "name", None)
        if c is None or c[0] is not b or c[1].name != name:
            src = str(b.source or "")
            info = BedInfo(b.id, tuple((round(float(x), 1), round(float(y), 1)) for x, y in b.dots[:, :2]), float(b.radius_mm),
                           None if b.dot_diam_mm is None else float(b.dot_diam_mm), str(b.learned or ""), bool(b.provisional),
                           src.rsplit("/", 1)[-1] if src.startswith("side:") else None, name, b.kind, b.outline.key())
            c = self._summary = (b, info)
        return c[1]

    def learning_map(self):
        """While Learn bed runs: the dots placed so far (seen twice or more), as a BedInfo for the bed drawn big over the
        model panel. None otherwise."""
        lr = self.learner
        if lr is None: return None
        try: M, _ = lr._clusters(2)
        except Exception: return None
        dots = tuple((round(float(x), 1), round(float(y), 1)) for x, y in np.asarray(M).reshape(-1, 2))
        model = self.bed or getattr(self, "pending", None) or getattr(self, "plane", None)
        radius = float(getattr(model, "radius_mm", 0) or 0) or (max(float(np.hypot(x, y)) for x, y in dots) + 10.0 if dots else 100.0)
        return BedInfo(None, dots, radius, None, "", True, None, getattr(self, "learn_name", None), "dots", None)

    def profiles(self):
        """The bed profiles' names for the Bed card's list (read when the session opens and after a pick or a learn),
        and Freehand last."""
        from app2.scan.bedhook import FREEHAND
        return tuple(getattr(self, "_profiles", ())) + (FREEHAND,)

    @property
    def profile_name(self): return getattr(self, "profile", None)

    def _read_profiles(self):
        try: self._profiles = tuple(n for n, _ in bedlib.list_profiles())
        except Exception as e: print("bed profiles:", e, flush=True); self._profiles = ()

    def take_message(self):
        """A message for the hint line, once (Learn bed's result)."""
        m, self._msg = self._msg, None; return m

    def _new_sidebed(self): return SideBed(self.bed, self.prior)

    # ------------------------------------------------------------------ session start, sides
    def open(self, project, param_dir, cal, log):
        """The project's own copy of the bed wins (builds and replays repeat exactly); else the config dir's, shared by
        every project on this turntable. Then the finder starts."""
        self.project, self.param, self.cal, self.log = project, param_dir, cal, log
        if hasattr(self,'color_observer'):self.color_observer.clear()
        self.color_packet=None
        try: bedlib.migrate_profiles(lambda m: print(m, flush=True))
        except Exception as e: print("bed profiles:", e, flush=True)
        self._read_profiles()
        self._choose(project)
        if self.plane is not None:
            print("bed: %s (flat, no dots: the base comes from depth)" % self.profile, flush=True)
        if self.bed is not None:
            print("bed: %d dots (%s, from the %s%s)" % (len(self.bed), self.bed.id, self.src,
                                                     ", profile %s" % self.profile if self.profile else ""), flush=True)
            if self.bed.calib_md5 and self.bed.calib_md5 != bedlib.calib_md5(param_dir):
                print("bed: learned with another calibration file; dots and depth share it, so it still fits", flush=True)
        self.running = True
        self.thread = threading.Thread(target=self._work, name="bed-finder", daemon=True); self.thread.start()

    def _choose(self, project):
        """Which bed this session uses. A project that names its bed profile (bed_profile.json) gets that one: its own
        bed.json when that is the same bed, else the profile. Else, as before: the project's own bed.json, else the
        profile picked last, else the config dir's bed.json."""
        pj = os.path.join(project, "bed.json"); meta = bedlib.read_project_profile(project)
        if meta and meta.get("kind") == "none":                   # Freehand: this project scans without a bed
            from app2.scan.bedhook import FREEHAND
            self.freehand = True; self.profile = FREEHAND; self.src = "project"; return
        own = BedModel.load(pj) if os.path.exists(pj) else None
        if own is not None and len(own) < bedlib.HIT_MIN: own = None
        def take(b, src, pending=False):
            if b is None: return False
            if b.kind == "plane": self.plane, self.src = b, src; self.profile = b.name; return True
            if len(b) < bedlib.HIT_MIN:
                if pending and b.name: self.pending, self.profile = b, b.name; return True     # picked, not learned yet
                return False
            self.bed, self.src, self.prior = b, src, b.camera
            self.profile = b.name or next((n for n, p in bedlib.list_profiles() if (BedModel.load(p) or b).id == b.id), None)
            return True
        if meta and meta.get("name"):
            if meta.get("kind") == "plane":
                if take(bedlib.load_profile(meta["name"]) or bedlib.plane_profile(meta["name"]), "project"): return
            elif own is not None and own.id == meta.get("id"):
                take(own, "project"); self.profile = meta["name"]; return
            elif take(bedlib.load_profile(meta["name"]), "profile", pending=True): return
        if take(own, "project"): return
        pk = bedlib.picked_profile()
        if pk and take(bedlib.load_profile(pk), "config", pending=True): return
        cp = BedModel.config_path()
        take(BedModel.load(cp) if os.path.exists(cp) else None, "config")

    def pick_profile(self, name, sides):
        """The Bed card's list: use the bed profile `name` from now on, for this project (its bed.json and
        bed_profile.json) and as the one picked next time. Sides not scanned yet look for it. Returns a log line."""
        if self.learner is not None or self.closing: return None
        from app2.scan.bedhook import FREEHAND
        self.freehand = name == FREEHAND
        if self.freehand:
            self.bed, self.plane, self.prior, self.pending = None, None, None, None
            pj = os.path.join(self.project, "bed.json")
            if os.path.exists(pj): os.replace(pj, os.path.join(self.project, "bed_prev.json"))
            self.src, self.profile, self.miss, self.miss_said = "profile", name, 0, False
            bedlib.pick_profile(name); bedlib.write_project_profile(self.project, SimpleNamespace(name=name, id=None, kind="none"))
            for o in sides:
                if not o.frames and o.bed is not None: o.bed.sb = self._new_sidebed()
            return "Bed: Freehand. No bed is looked for or cut: move the part or the scanner as you like, and take the table off in Edit."
        b = bedlib.load_profile(name)
        if b is None: return "No bed named %s" % name
        pj = os.path.join(self.project, "bed.json"); unlearned = b.kind != "plane" and len(b) < bedlib.HIT_MIN
        self.pending = b if unlearned else None
        if b.kind == "plane" or unlearned:
            self.bed, self.plane, self.prior = None, (b if b.kind == "plane" else None), None
            if os.path.exists(pj): os.replace(pj, os.path.join(self.project, "bed_prev.json"))
        else:
            self.bed, self.plane, self.prior = b, None, b.camera
            for path in (BedModel.config_path(), pj):
                old = BedModel.load(path) if os.path.exists(path) else None
                if old is not None and old.id != b.id: shutil.copyfile(path, os.path.join(os.path.dirname(path), "bed_prev.json"))
                b.save(path)
        self.src, self.profile, self.miss, self.miss_said = "profile", name, 0, False
        bedlib.pick_profile(name); bedlib.write_project_profile(self.project, b); self._read_profiles()
        for o in sides:
            if not o.frames and o.bed is not None: o.bed.sb = self._new_sidebed()
        if b.kind == "plane": return "Bed: %s (flat, no dots). The base comes from depth (Auto remove base)." % name
        if unlearned: return "Bed: %s is not learned yet. Take the part off the turntable and press Learn bed." % name
        return "Bed: %s (%d dots)" % (name, len(b))

    def _load_sidebed(self, side):
        bp = os.path.join(side.folder, "bed_pose.json")
        return SideBed.load(bp, self.bed, self.prior) if os.path.exists(bp) else self._new_sidebed()

    def side_opened(self, side, saved):
        """A side of the project (saved: as it was saved) or a new one."""
        side.bed = SideState(self._load_sidebed(side) if saved else self._new_sidebed())

    def side_loaded(self, side):
        """A saved side's model was rebuilt for a Continue: its bed as saved (no looking for it again over the frames)."""
        side.bed.sb = self._load_sidebed(side)
        from app2.scan.bed_pose import BedPoseGuide
        from app2.scan.bed_resume import restore
        side.bed.pose_guide=BedPoseGuide()
        if side.bed.sb.B_scan() is None:restore(side,self.cal,self.bed,side.bed.pose_guide)
        else:side.bed.pose_guide.reference=side.bed.sb.B_scan()

    def pass_started(self, side):
        """Start / Continue (called before side.ps counts the new pass): a bed learned since is used by a side that had
        not found the old one; the table dots get a new chance every pass."""
        st = side.bed
        if st.sb.bed is not self.bed and not st.sb.found: st.sb = self._new_sidebed();st.pose_guide=None
        st.sb.new_pass(side.ps + 1); self.shown_dots = None; st.find_miss = 0
        pending = getattr(self, "pending", None)
        if pending is not None:                                   # 2026-10-25: the picked bed has no dots yet, say so where it is seen
            self._msg = "Bed %s is not learned yet, so this side scans without it. Take the part off and press Learn bed first." % pending.name

    # ------------------------------------------------------------------ every frame (scan worker)
    def before_step(self, side, d, irl, tracker, last):
        """On the first frame of a Continue, or after a lost frame: place the camera from the table dots
        (B_scan @ inv(B_cam)) and let ICP start there. Much faster than the feature search; ICP must still fit, within
        the guard (a fit far from the dots' pose is refused). tracker: the session's TrackerHost."""
        st = side.bed
        if st is None or st.off_pass == side.ps: return
        if hasattr(tracker,'marker_required'):tracker.marker_required=False
        far=getattr(side,"scan_profile",{}).get("mode")=="far"
        if not far and hasattr(tracker,'marker_hint') and getattr(st.sb,'bed',None) is not None:
            from app2.scan.bed_pose import BedPoseGuide
            if st.pose_guide is None:st.pose_guide=BedPoseGuide()
            resume=(side.ps>0 and side.idx==0) or (last is not None and not last['ok'])
            if st.pose_guide.step(self,side,d,irl,tracker,resume=resume):return
            if resume and st.pose_guide.reference is not None:
                tracker.marker_required=True
                return
            if st.pose_guide.last_hit is not None:
                from app2.scan.bed_pose import flat_scene
                if flat_scene(d,self.cal,st.pose_guide.last_hit,st.sb.bed):
                    tracker.marker_required=True
                    return
        if not (far or (side.ps > 0 and side.idx == 0) or (last is not None and not last["ok"])): return
        Tn = st.sb.hint_pose(d, irl, self.cal, side.ps)
        if Tn is not None:
            quality=getattr(st.sb,'last_hint_quality',None)
            if far and quality is not None and hasattr(tracker,'marker_hint'):
                tracker.marker_hint(Tn,quality)
            else:tracker.hint(Tn, guard_mm=bedlib.HINT_GUARD_MM, guard_deg=bedlib.HINT_GUARD_DEG)

    def after_step(self, side, d, irl, r, T_mm):
        """After a frame: whether the dots still place the part, whether the scanner moved (camera bed), and what the
        finder looks at next (find while the pass is not placed, then every 10th tracked frame to check it; pool
        tracked frames 10-90 while the side has no bed). Returns (log lines, why the pass must pause or None)."""
        st = side.bed; lines = []; pause = None
        if st is None: return lines, pause
        if st.pose_guide is not None:st.pose_guide.accepted(r)
        sb = st.sb
        msg = sb.hint_outcome(r, side.ps)
        if msg:
            st.moved_pass = side.ps
            # the part moved on the table: safer than guessing where it went, the session pauses once this frame is stored
            if getattr(sb, "move_pause", None) == side.ps: pause = msg
            else: lines.append(msg)
        if r["ok"]: lines += sb.check_camera(side.ps, side.idx, T_mm)
        # after a move the dots' hits no longer match the side's placement (they would veto it): no finder jobs for the
        # rest of the pass
        if sb.bed is None or sb.state == "off" or sb.state.startswith("vetoed") or st.off_pass == side.ps or st.moved_pass == side.ps:
            return lines, pause
        T = T_mm if r["ok"] else None; n_ok = side.pass_ok - (1 if r["ok"] else 0)       # tracked frames before this one
        base = dict(side=side, ps=side.ps, idx=side.idx, d=d, ir=irl, bed=sb.bed)
        if sb.camera_bed(side.ps) is None:
            if side.idx - sb.look_from(side.ps) <= bedlib.POOL_TO: self._job(dict(base, kind="find", T=T, prior=sb.prior_for(side.ps)))
        elif r["ok"] and n_ok % 10 == 0 and sb.found: self._job(dict(base, kind="find", T=T, prior=sb.prior_for(side.ps)))
        if sb.state == "looking" and r["ok"] and bedlib.POOL_FROM <= n_ok <= bedlib.POOL_TO and n_ok % bedlib.POOL_EVERY == 0:
            pl = side.table.cached_plane()
            self._job(dict(base, kind="pool", T=T_mm, snap=sb.pool_snapshot(), extra=[(pl[0], pl[1])] if pl is not None else []))
        return lines, pause

    def _mismatch(self, st, sb, n_dots, hit, ps, no):
        """Dots in view, none of them matching the picked bed, three finder looks in a row: the wrong bed is picked or the
        table changed (the round mat's map was matched against the calibration board and the rim
        wandered). Said once per pass, on the hint line and in the log; None otherwise."""
        matched = hit is not None and hit.matched is not None and int(np.count_nonzero(hit.matched)) >= bedlib.HIT_MIN
        if n_dots < bedlib.HIT_MIN:                                 # few dots: the bed is out of the picture (said once per pass when it had been found)
            if not sb.found: return None
            st.find_miss += 1
            if st.find_miss < 3 or st.miss_said == ps: return None
            st.miss_said = ps
            return "side %d: the bed is out of view (%d dots). Keep some table dots in the picture so the turn stays locked." % (no, n_dots)
        if sb.found or matched: st.find_miss = 0; return None
        st.find_miss += 1
        if st.find_miss < 3 or st.miss_said == ps: return None
        st.miss_said = ps; name = getattr(self, "profile", None) or (sb.bed.name if sb.bed is not None else "the bed")
        return "side %d: %d dots in view but none match the bed %s. Pick the bed that is on the table (Bed card), or Learn bed again." % (no, n_dots, name)

    def _rim_check(self, st, sb, hit, ps, idx):
        """A matched hit against this pass's camera bed (the rim and tint). Off it three looks in a row: the scanner moved
        while tracking said it stood still (the rim sat far from the table), so the camera bed is dropped
        and found again. The newest matched hit is kept so the rim can follow the dots meanwhile. Returns log lines."""
        if hit is None or hit.matched is None or int(np.count_nonzero(hit.matched)) < bedlib.HIT_MIN: return []
        st.fresh = (ps, idx, hit); cam = sb.camera_bed(ps)
        if cam is None: st.rim_bad = 0; return []
        if bedlib.agree(hit, cam): st.rim_bad = 0; return []
        st.rim_bad += 1
        if st.rim_bad < 3: return []
        st.rim_bad = 0
        return sb.drop_camera(ps, idx, "the dots put the table %.0f mm from where this pass had it" % float(np.linalg.norm(hit.centre - np.asarray(cam["centre"]))))

    def every10(self, side):
        """Every 10th tracked frame: V2 (bed normal vs the spin axis) and V3 (bed vs Auto remove base)."""
        self.request_auto(side)
        return [] if side.bed is None else side.bed.sb.check(side.table, cached=True)

    def request_auto(self, side):
        """Queue the observed top detector at the existing cadence; never fit on the scan worker."""
        table = getattr(side, 'table', None)
        if not self.running or table is None or len(table.pts) < table.plane.__defaults__[0]: return
        if table._cache[0] == len(table.pts): return
        token = (id(table), side.ps)
        with self._cv:
            if token in self._auto_pending: return
            # Keep the existing newest-frame queue policy without displacing a queued dot find.
            if self.jobs.full(): return
            snapshot = table.snapshot()
            self._auto_pending[token] = len(snapshot.pts)
        self._job(dict(kind='auto', side=side, table=table, ps=side.ps, snapshot=snapshot, token=token))

    # ------------------------------------------------------------------ the platter
    def found(self, side): return side.bed is not None and side.bed.sb.found

    def platter(self, side):
        """The table dots' platter (n, off, centre, radius, "bed") in the side's scan frame once the bed was found, else
        None (the session then takes Auto remove base's)."""
        sb = None if side.bed is None else side.bed.sb
        if sb is None or not sb.found or sb.fixed is None: return None
        f = sb.fixed; return f["n"], f["off"], f["centre"], _shape(sb.bed, f["x_axis"]), "bed"

    def camera_geo(self, side, recording):
        """This pass's camera bed (the table dots, camera frame) for the violet tint and its rim: no pose needed, so it
        is right on lost frames too."""
        if not recording or side is None or side.bed is None or not side.bed.sb.found: return None
        sb = side.bed.sb; cam = sb.camera_bed(side.ps); fresh = side.bed.fresh
        if fresh is not None and fresh[0] == side.ps and side.idx - fresh[1] <= 15 and (cam is None or not bedlib.agree(fresh[2], cam)):
            h = fresh[2]; return (h.n, h.off, h.centre, _shape(sb.bed, h.x_axis))      # the dots as seen just now beat a camera bed they disagree with
        return None if cam is None else (cam["n"], cam["off"], cam["centre"], _shape(sb.bed, cam["x_axis"]))

    def dots(self, side, recording):
        """The table dots to circle on the IR picture: learning, the learner's last frame (green = placed on the map);
        recording, the newest finder hit of this pass (green = matched to the bed), for 4 frames."""
        lr = self.learner
        if lr is not None:
            last = lr.last
            return None if last is None else (last[0][:, :3], last[1])
        s = self.shown_dots
        if recording and s is not None and side is not None and s[0] is side and s[1] == side.ps and side.idx - s[2] <= 4:
            return s[3], s[4]
        return None

    def base_text(self, side, recording):
        if getattr(self, "freehand", False): return "base: none (freehand)"
        sb = None if side.bed is None else side.bed.sb
        if getattr(self, "plane", None) is not None and (sb is None or sb.bed is None): return "base: Auto (flat bed, no dots)"
        if sb is None or sb.bed is None: return "base: Auto (no bed learned)"
        if sb.found: return "base: bed (%d dots)" % sb.dots_matched
        if sb.state.startswith("vetoed"): return "base: Auto (bed disagreed)"
        return "base: looking for the bed" if (recording and side.idx - sb.look_from(side.ps) <= bedlib.POOL_TO) else "base: Auto"

    def card_base(self, side): return "bed" if self.found(side) else "auto"

    # ------------------------------------------------------------------ saving
    def _write_platter(self, side, tp):
        """table_plane.json for the build: the table dots' plane when the bed was found ("source": "bed"), else Auto
        remove base's ("source": "auto", only while a bed is in play: with no bed the file stays exactly as SidePlane
        writes it), else none; and bed_pose.json beside it (what the bed saw). Atomic writes (the build may be reading
        them); the bytes are the Tk scanner's. Returns the source written."""
        sb = side.bed.sb
        if BED_CUTS and sb.found and sb.fixed is not None:
            j = sb.table_plane_json(1.0, side.table.spin_axis())
            tmp = tp + ".tmp.%d" % os.getpid()
            with open(tmp, "w") as f: json.dump(j, f)
            os.replace(tmp, tp); src = "bed"
        elif side.table.save(tp):
            src = "auto"
            if sb.bed is not None:
                try:
                    with open(tp) as f: j = json.load(f)
                    j["source"] = "auto"; tmp = tp + ".tmp.%d" % os.getpid()
                    with open(tmp, "w") as f: json.dump(j, f)
                    os.replace(tmp, tp)
                except Exception as e: print("table:", e, flush=True)
        else:
            src = None
            if os.path.exists(tp): os.remove(tp)
        if sb.bed is not None: sb.save(os.path.join(side.folder, "bed_pose.json"))
        side.bed.src = src
        return src

    def save(self, side, path, auto_plane, side_no):
        """Project.save_platter's table_plane.json, written here in every case (bed, auto or none). True: done."""
        if side.bed is None: return False
        self._write_platter(side, path)
        return True

    def after_save(self, side, auto, side_no, sides):
        """What a save says: where the base cut comes from, the project's copy of the bed it was scanned with, "bed not
        recognised", and with no bed at all a provisional one asked for from this side (on the finder's thread)."""
        st = side.bed; out = []
        if st is None: return out
        sb = st.sb; src = st.src
        if src == "bed" and auto is not None and sb.auto_check:
            ac = sb.auto_check
            out.append("side %d: base from the table dots (%d dots); Auto remove base: %s, %+.1f mm, %.1f deg"
                       % (side_no, sb.dots_matched, ac.get("verdict"), ac.get("mm", 0), ac.get("deg", 0)))
        elif src == "auto" and sb.bed is not None and not sb.found: out.append("side %d: table not seen, Auto remove base" % side_no)
        if sb.bed is not None:
            sb.bed.copy_to(self.project)
            if getattr(self, "profile", None) and bedlib.read_project_profile(self.project) is None and sb.bed.name == self.profile:
                bedlib.write_project_profile(self.project, sb.bed)
            if not sb.found and st.dots_seen >= bedlib.HIT_MIN:          # dots in view, none matched: the table changed?
                self.miss += 1
                if self.miss >= 3 and not self.miss_said: self.miss_said = True; out.append("bed not recognised: Learn bed again?")
            elif sb.found: self.miss = 0
        elif self.bed is None and getattr(self, "plane", None) is None and getattr(self, "pending", None) is None \
                and not getattr(self, "freehand", False) and auto is not None and not st.seed_tried:
            # no bed at all: a provisional one from this side's own dots, so the next side has it. On the finder's thread
            # (2-4 s), once per side, into this project only (the config's bed is written by Learn bed alone)
            st.seed_tried = True
            self.seeds.put(dict(kind="seed", side=side, folder=side.folder, plane=(np.array(auto[0], float), float(auto[1])),
                                source="side:%s/%s" % (os.path.basename(self.project), os.path.basename(side.folder))))
        return out

    def _resave(self, side):
        """A finder result that came in after the side was saved: its table_plane.json and bed_pose.json again."""
        try:
            src = self._write_platter(side, os.path.join(side.folder, "table_plane.json"))
            from app2.scan.turn_coverage import save as save_turn
            save_turn(side)
            print("side %s: platter saved again (%s): a table-finder result came in after the save" % (os.path.basename(side.folder), src), flush=True)
        except Exception as e: print("table:", e, flush=True)

    # ------------------------------------------------------------------ the finder's results (scan worker)
    def drain(self, sides, picked, side_no):
        """Apply the finder's results. A result that changes what a side saves and comes in after the side was saved
        writes that side's files again at once. Returns (log lines, the picked side's platter changed)."""
        self._sides, self._side_no = sides, side_no
        lines = []; changed = False
        while True:
            try: r = self.res.get_nowait()
            except queue.Empty: break
            kind = r["kind"]
            if kind == 'auto':
                sd = r['side']
                if r.get('error'):
                    print('table:', r['error'], flush=True)
                elif sd in sides and sd.table is r['table'] and sd.ps == r['ps']:
                    if sd.table.install_snapshot(r['snapshot']) and sd is picked:
                        changed = True
                continue
            if kind == "learn": lines += self._learn_result(r, sides); continue
            if kind == "seed": lines += self._seed_result(r, sides, side_no); continue
            if r.get("error"): print("bed:", r["error"], flush=True)
            sd = r["side"]
            if sd not in sides or sd.bed is None or r["bed"] is not sd.bed.sb.bed: continue
            st = sd.bed; sb = st.sb; sig = sb.signature(); state0 = sb.state; no = side_no(sd)
            if r.get("slow") == 3:
                st.off_pass = r["ps"]; lines.append("side %d: the table finder is too slow on this PC; it stops for this pass" % no)
            if kind == "find":
                st.dots_seen = max(st.dots_seen, r.get("n_dots", 0)); h = r["res"]
                if h is not None: self.shown_dots = (sd, r["ps"], r["idx"], h.dots_px, h.matched)
                had = sb.camera_bed(r["ps"]) is not None
                lines += ["side %d: %s" % (no, e) for e in sb.add_hit(r["ps"], r["idx"], h, r["T"])]
                m = self._mismatch(st, sb, r.get("n_dots", 0), h, r["ps"], no)
                if m: lines.append(m); self._msg = m
                lines += ["side %d: %s" % (no, e) for e in self._rim_check(st, sb, h, r["ps"], r["idx"])]
                if not had and sb.camera_bed(r["ps"]) is not None:           # accepted: tint it now
                    self.prior = sb.camera_bed(r["ps"])
                    if sd is picked: changed = True
            elif kind == "pool":
                if r.get("feats") is not None: sb.add_pool(r["ps"], r["idx"], r["feats"], r["T"])
                for e in sb.apply_pool(r["res"], r["ps"], r["idx"]):
                    lines.append("side %d: %s" % (no, e))
                    if sd is picked: changed = True
            if sb.state != state0 and sd is picked: changed = True
            if sd.frames and sb.signature() != sig and not sd.dirty:
                self._resave(sd)                  # already saved: bring its table_plane.json / bed_pose.json up to date now
        return lines, changed

    def settle(self, sides, picked, side_no, final=False):
        """Before a save: wait (at most SETTLE_S) for the frame jobs the finder already has, then apply every result, so
        what is saved is what the window showed. final (closing): stop the finder once the job in hand is done, run the
        frame jobs it still had here, apply every result."""
        if final:
            self._stop_finder(run_left=True)
        else:
            t_end = time.monotonic() + SETTLE_S
            with self._cv:
                while self._pending > 0 and self.running:
                    left = t_end - time.monotonic()
                    if left <= 0: break
                    self._cv.wait(left)
        return self.drain(sides, picked, side_no)

    def _learn_result(self, r, sides):
        if r.get("learner") is not self.learner or self.closing: return []       # stopped (or started again) meanwhile
        if r.get("error"): print("learn bed:", r["error"], flush=True)
        s = r.get("res")
        if s is None: return []
        want = getattr(self, "learn_want", None)
        if want: s = dict(s, want=want)
        self.learn_status = s
        if s["state"] == "done":
            self.learner = None; self.learn_status = None; b = r.get("bed"); n = 0 if b is None else len(b)
            if b is None or n < bedlib.LEARN_MIN_DOTS:
                msg = "Only %d dots seen. Try IR gain 3 to 6 and Learn bed again." % n
            elif want and n < 0.9 * want:                        # the user counted them: well short is not saved
                msg = "Found %d of the %d dots you counted. Not saved: turn slower, try IR gain 3 to 6, and Learn bed again." % (n, want)
            else:
                try:
                    self._install(b, sides)
                    msg = "Bed learned: %s. Put the part on and start a side." % learned_text(b)
                    if want and n < want: msg = "Bed learned with %d of the %d dots you counted (close enough). Put the part on and start a side." % (n, want)
                    elif want and n > want: msg = "Bed learned: %d dots, more than the %d you counted. Check for stray bright spots on the table. Put the part on and start a side." % (n, want)
                    warn = bedlib.outline_check(b) if hasattr(b, "outline") else None
                    if warn: msg += " Check the bed: %s." % warn
                    if b.self_ambiguity is not None and b.self_ambiguity > 0.6: msg += " (Its dots repeat: matching may be unsure.)"
                except Exception as e: msg = "Could not save the bed: %s" % e
            if self.bed is not b or b is None:
                self.learn_status = dict(state="failed", msg=msg)
            self._msg = msg; return [msg]
        if s["state"] == "failed":
            self.learner = None; self._msg = s["msg"]; return ["Learn bed: " + s["msg"]]
        return []

    def _install(self, b, sides):
        """A freshly learned bed goes to the config dir (every project on this turntable) and this project; the old ones
        stay as bed_prev.json. Sides not scanned yet look for the new one."""
        b.name = b.name or self.learn_name or self.profile or "Bed " + str(b.learned)[:10]
        pre = bedlib.load_profile(b.name)                     # learning a preset (or the same bed again): its measured size
        if pre is not None and pre.nominal and not b.nominal: b.nominal = pre.nominal
        for path in (BedModel.config_path(), os.path.join(self.project, "bed.json")):
            if os.path.exists(path): shutil.copyfile(path, os.path.join(os.path.dirname(path), "bed_prev.json"))
            b.save(path)
        bedlib.save_profile(b); bedlib.pick_profile(b.name); bedlib.write_project_profile(self.project, b)
        self.bed, self.src, self.prior, self.miss, self.miss_said = b, "learn", b.camera, 0, False
        self.plane, self.pending, self.profile, self.learn_name = None, None, b.name, None; self._read_profiles()
        for o in sides:
            if not o.frames and o.bed is not None: o.bed.sb = self._new_sidebed()

    def _seed_result(self, r, sides, side_no):
        """A provisional bed learned from a side's own dots: into this project only, marked provisional."""
        b = r.get("new_bed")
        if b is None:
            print("bed: no provisional bed from side %s (%s)" % (os.path.basename(r["folder"]), r.get("error") or r.get("why")), flush=True)
            return []
        if self.bed is not None or r["side"] not in sides: return []          # Learn bed finished meanwhile, or the side is gone
        try: b.save(os.path.join(self.project, "bed.json"))
        except Exception as e: print("bed:", e, flush=True); return []
        self.bed, self.src, self.prior = b, "side", b.camera
        for o in sides:
            if not o.frames and o.bed is not None: o.bed.sb = self._new_sidebed()
        return ["bed learned from side %d's own dots (%d dots), for this project only. Learn bed (empty table, one turn) "
                "makes one for every project." % (side_no(r["side"]), len(b))]

    # ------------------------------------------------------------------ Learn bed
    def learn_toggle(self, gain, busy, name=None, want=None):
        """Learn bed: empty table, one turn, no tracking. Pressed again while learning: stop, nothing saved. Refused while
        recording, busy or building (busy). name: the new bed profile's name (None: the picked profile is learned
        again). Returns a log line or None."""
        if self.learner is not None:
            self.learner = None; self.learn_status = None; return "Learn bed stopped; nothing saved"
        if busy or self.closing: return None
        from app2.scan.color_bed import ColorBedLearner
        model=self.bed or self.pending
        self.learner = ColorBedLearner(self.cal, gain=gain,radius_mm=getattr(model,'radius_mm',None)); self.learn_status = None; self._msg = None
        self.learn_name = name or (self.profile if self.plane is None else None); self.learn_want = int(want) if want else None   # dots the user counted (2026-10-25)
        return "Learn bed: keep the empty bed in view while it turns. Color can finish the check sooner."

    def learn_frame(self, d, irl, t, gain):
        lr = self.learner
        if lr is not None: self._job(dict(kind="learn", learner=lr, d=d, ir=irl, t=t, gain=gain,color=self.color_packet,color_cal=self.color_cal))

    def set_color_frame(self,packet,calibration):
        self.color_packet=packet;self.color_cal=calibration

    # ------------------------------------------------------------------ the finder's thread
    def _job(self, job):
        """Queue a job; when the finder is behind, the oldest waiting one goes."""
        if not self.running: return
        frame = job["kind"] in ("find", "pool", "auto")
        if frame:
            with self._cv: self._pending += 1
        try: self.jobs.put_nowait(job); return
        except queue.Full: pass
        try: self._job_done(self.jobs.get_nowait())
        except queue.Empty: pass
        try: self.jobs.put_nowait(job)
        except queue.Full: self._job_done(job)

    def _job_done(self, job):
        if job["kind"] in ("find", "pool", "auto"):
            with self._cv:
                self._pending -= 1
                if job['kind'] == 'auto':
                    self._auto_pending.pop(job['token'], None)
                self._cv.notify_all()

    def _run_job(self, job):
        """One finder job -> its result (for drain). On the finder's thread; at close, on the scan worker."""
        t0 = time.monotonic(); out = dict(job, res=None); k = job["kind"]
        try:
            if k == 'auto':
                snapshot = job['snapshot']
                out['res'] = snapshot.plane()
                if out['res'] is not None: snapshot.disc()
            elif k == "learn":
                lr = job["learner"]
                if job.get('color') is not None and hasattr(lr,'add_color'):
                    out['res']=lr.add_color(job['d'],job['ir'],job['color']['jpeg'],job.get('color_cal'),t=job['t'],gain=job['gain'])
                else:out["res"] = lr.add(job["d"], job["ir"], t=job["t"], gain=job["gain"])
                if out["res"]["state"] == "done": out["bed"] = lr.result(self.param)
            elif k == "find":
                rings = getattr(job["bed"], "ringed", None) is not None          # a ringed board: dots in broken rings
                dl, fr = bedlib.detect_marks(job["ir"], rings); out["n_dots"] = len(dl)
                out["res"] = bedlib.find_bed(job["d"], job["ir"], self.cal, job["bed"], prior=job["prior"],
                                             dots=(dl, fr) if rings else dl)
            elif k == "pool":
                f = bedlib.pool_features(job["d"], job["ir"], self.cal); out["feats"] = f
                out["res"] = bedlib.solve_pool(job["snap"] + [(f, job["T"])], job["bed"], self.cal, job["extra"])
            elif k == "seed":
                out["new_bed"], out["why"] = bedlib.provisional_bed(job["folder"], job["plane"], job["source"])
                if out["new_bed"] is not None: out["new_bed"].calib_md5 = bedlib.calib_md5(self.param)
        except Exception as e:
            out["error"] = "%s: %s" % (type(e).__name__, e)
        self.ran[k] = self.ran.get(k, 0) + 1
        out["secs"] = time.monotonic() - t0
        out.pop("d", None); out.pop("ir", None)
        return out

    def _work(self):
        skip = 0; slow = {}
        while self.running:
            try: job = self.seeds.get_nowait()
            except queue.Empty:
                try: job = self.jobs.get(timeout=0.5)
                except queue.Empty: continue
            frame = job["kind"] in ("find", "pool")
            try:
                if skip > 0 and frame: skip -= 1; continue          # a slow call: let the next few frames go
                out = self._run_job(job)
                if out["secs"] > SLOW_S and frame:
                    skip = 5; key = (id(job.get("side")), job.get("ps")); slow[key] = slow.get(key, 0) + 1; out["slow"] = slow[key]
                self.res.put(out)
            finally: self._job_done(job)

    def _stop_finder(self, run_left):
        """Stop the finder once the job in hand is done; run_left: its waiting frame jobs run here (learning stops)."""
        if not self.running: return
        self.running = False; self.closing = True
        if self.thread is not None: self.thread.join(JOIN_S)
        while True:
            try: job = self.jobs.get_nowait()
            except queue.Empty: break
            try:
                if run_left and job["kind"] in ("find", "pool"): self.res.put(self._run_job(job))
            finally: self._job_done(job)

    def close(self):
        """Closing, after the last saves: the finder stops, and with no bed yet a provisional one from a side saved just
        now (Finish while recording queues it) is made here; the first one that works is kept. Returns log lines."""
        if self.closed: return []
        self.closed = True; self.closing = True
        observer=getattr(self,'color_observer',None)
        if observer is not None:observer.close()
        self._stop_finder(run_left=False)
        lines = []
        if self._side_no is not None:
            while self.bed is None:
                try: job = self.seeds.get_nowait()
                except queue.Empty: break
                lines += self._seed_result(self._run_job(job), self._sides, self._side_no)
        return lines
