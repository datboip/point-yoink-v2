# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the Scan step's buttons, hint, status line and banner say, worked out from a ScanState snapshot without any
window (the Tk scanner's refresh_buttons / refresh_learn and the bar in tick(), in shorter words). The Qt page only
applies it, so every text and rule can be tested headless."""
import math


class ScanState:
    """A copy of the session's state for the window (made on the worker thread, never changed after).
    picked: the picked side's number (1-based) or None. cam: None, "noframes", "gone" or "replay_end".
    cal: (fx, fy, cx, cy) of the depth camera, for the 3D view's field of view. bed: the bed hook's BedInfo or None.
    rewind_room: frames Back may still take from the picked side; rewound: frames taken back since the pause (Forward
    can bring them back); forward_n: how many the next Forward brings back."""
    __slots__ = ("outside_below_mm","bed_lift_mm","live_build","live_detail","live_pass","live_undo","light_during_scans","surface_name","surface_hint","overlay","range_test","calibration_missing","framing","scene_setup","range_undo","led_requested", "led_status", "stream_warning", "range_guide", "lost_display", "mode_notice", "mode_evidence", "calibration_error", "calibration_source", "far_calibration_path", "light_remaining", "capture_mode", "scan_mode", "range_limits", "recording", "busy", "building", "learning", "learn_status", "learn_map", "bed_available", "bed_size", "bed_provisional",
                 "bed", "bed_profiles", "bed_profile", "picked", "picked_has_frames", "n_sides", "any_frames", "model_exists", "build_msg", "cam", "slow",
                 "near", "far", "gain", "auto_range", "auto_gain", "led", "has_led", "project", "source_kind",
                 "source_label", "cal", "model_view", "build_engine", "below_bed", "remove_base", "save_color", "picked_touch",
                 "rewind_room", "rewound", "forward_n")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))

    def key(self):
        """For "did anything change": every field, learn_status as text."""
        return tuple(str(getattr(self, k)) if k == "learn_status" else getattr(self, k) for k in self.__slots__)


class StatusView:
    """The status line and the banner. kind: tracking, lost, paused, idle, building, noframes, gone, replay_end, slow (it
    picks the colour of the pill and of the rings round the IR picture and the model). stats: [(template with {} for the value,
    value text, colour name or None)]. auto_msg: what Auto gain is doing (the controls show it; the status line only
    when the controls have no room). banner: (level "warn" / "lost" / "info", text, more) or None."""
    __slots__ = ("kind", "pill", "stats", "hint", "banner", "auto_msg")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))


# the colour each status kind shows in (theme token names)
KIND_COLOUR = dict(tracking="kept", lost="lost", paused="warn", idle="ink2", noframes="warn", gone="lost", replay_end="accent",
                   slow="warn", building="accent")
RING = dict(tracking="kept", lost="lost", paused="warn", noframes="warn", gone="lost", slow="warn")   # idle: no ring

HINT_START = "Set the Range so the part is green, then press Start."
HINT_RECORDING = "Pause to move the scanner. To flip the part: Pause, then + New side."
HINT_BUILDING = "Building the model. Scanning is paused."
HINT_LEARNING = "Empty bed: let it turn once. Stops by itself."
HINT_TOUCH = "Touch up: press Start, aim at red and purple spots."
HINT_TOUCH_RECORDING = "Aim at red and purple spots. Hold still to average more."
BACK_STEPS = (30, 10, 5, 1)               # the Back buttons, left to right (the page shows them in this order)
HINT_REWOUND = "Took back %d frame%s. Continue picks up from the last frame kept; Forward brings them back."
REWIND_TIP = ("Take back the newest frames of this side (moved too fast, bumped the part, tracking drifted). The model "
              "updates at once. Continue picks up from the last frame kept; the frames taken back are then set aside for good.")


def buttons(s):
    """Texts, looks and enabled flags of every button, and the hint line: {name: (text, kind, enabled)} plus "hint"
    and "learn_visible". Unlike the Tk window (which left the other buttons as they were while busy), a busy session
    greys out everything that would be refused anyway."""
    b = {}
    locked = bool(s.busy or s.building or s.learning)
    b["learn_visible"] = bool(s.bed_available)
    if s.learning: b["learn"] = ("Stop learning", "warn", True)
    else:
        # a provisional bed (from a side's own dots, this project only) still asks for a real one, as the Tk scanner did
        text = "Learn bed" if s.bed_size is None else ("Learn bed (now %d dots from a side)" % s.bed_size if s.bed_provisional
                                                       else "Bed: %d dots (learn again)" % s.bed_size)
        b["learn"] = (text, "plain", bool(s.bed_available) and not (s.recording or s.busy or s.building))
    b["build"] = ("Build combined model", "build", not s.recording and not locked and bool(s.any_frames))
    b["stop_build"] = ("Stop build", "danger", bool(s.building))
    b["view"] = ("View in 3D", "plain", bool(s.model_exists) and not s.building)
    b["new"] = ("+ New side", "plain", not s.recording and not locked)
    b["touch"] = ("Touch up", "plain", bool(s.model_exists) and not s.recording and not locked)
    b["delete"] = ("Delete", "danger", not s.recording and not locked)       # any side card, not only the picked one
    b["new_project"] = ("New project", "quiet", not s.busy and not s.building)
    b["open_project"] = ("Open project", "quiet", not s.busy and not s.building)
    b["finish"] = ("Finish", "plain", True)
    b["led"] = ("Fill LED" if s.has_led else "Fill LED not set up", "lamp", bool(s.has_led) and not s.busy)
    b["auto_range"] = ("Auto", "chip", not s.busy)
    b["auto_gain"] = ("Auto", "chip", not s.busy)
    b["below_bed"] = ("Cut below bed", "chip", not s.busy)
    b["remove_base"] = ("Remove base", "chip", not s.busy)
    hint = s.build_msg or ""
    if s.busy: b["go"] = (s.busy, "primary", False)
    elif s.learning:
        st = s.learn_status or {}
        b["go"] = ("Cancel learning", "pause", True)             # the same button stops it: nothing is saved
        sp = st.get("speed"); need = st.get("need") or 2.5
        pace = "" if sp is None else (" Turning %.1f deg/s: fine." % sp if sp >= need else " Turning %.1f deg/s: too slow, it needs %.1f. Turn the table up." % (sp, need))
        hint = "Learning the bed: %d of 360 deg, %d dots.%s %s" % (min(360, st.get("deg", 0)), st.get("dots", 0), pace, HINT_LEARNING)
    elif s.building:
        b["go"] = ("Building model...", "primary", False)
        hint = HINT_BUILDING
    elif s.picked is None:
        b["go"] = ("Start side %d" % (s.n_sides + 1), "primary", True)
        hint = hint or HINT_START
    elif s.recording:
        b["go"] = ("Pause side %d" % s.picked, "pause", True)
        hint = HINT_TOUCH_RECORDING if s.picked_touch else HINT_RECORDING
    else:
        verb = "Continue" if s.picked_has_frames else "Start"
        b["go"] = ("%s side %d" % (verb, s.picked), "primary", True)
        if s.picked_touch and not s.picked_has_frames: hint = hint or HINT_TOUCH
        hint = hint or (("Side %d picked: Continue to fill its gaps, or flip the part and + New side." % s.picked)
                        if s.picked_has_frames else "Side %d is ready. Press the big button to start scanning it." % s.picked)
    if (s.learn_status or {}).get("state") == "failed" and not (s.recording or locked):
        hint = "Learn bed failed. See Log, then learn again."
    rew = int(s.rewound or 0)
    b["rewind_visible"] = bool(s.picked is not None and s.picked_has_frames and not s.recording and not s.building and not s.learning)
    for k in BACK_STEPS: b["back%d" % k] = ("Back %d" % k, "plain", b["rewind_visible"] and not locked and bool(s.rewind_room))
    b["forward"] = ("Forward %d" % s.forward_n if s.forward_n else "Forward", "plain", b["rewind_visible"] and not locked and bool(s.forward_n))
    b["rewind_note"] = ("%d frame%s taken back" % (rew, "" if rew == 1 else "s")) if rew else "Undo"
    if rew and b["rewind_visible"] and not locked: hint = HINT_REWOUND % (rew, "" if rew == 1 else "s")
    if s.capture_mode in ("single", "burst") and not locked:
        burst = s.capture_mode == "burst"
        b["go"] = (("Cancel burst" if burst else "Cancel single shot") if s.recording else ("Capture a burst" if burst else "Capture a shot"), "pause" if s.recording else "primary", True)
        hint = ("Hold still. Takes a burst of tracked frames, then pauses." if burst else "Hold still. Takes several tracked frames (more on each press), then pauses.") if s.recording else hint
    if getattr(s,'calibration_error',None) and not s.recording and not s.learning:
        mode=(getattr(s,'mode_evidence',None) or {}).get('mode')
        if mode in ('near','far') and mode!=getattr(s,'scan_mode','near') and not locked:
            from app2.scan.object_modes import label
            b['go']=('Use '+label(mode),'pause',True);b['go_action']='switch_camera_match'
            hint='The frames match '+label(mode)+'. Click to switch. Capture stays paused; a new side is made if needed.'
        else:b['go']=("Checking camera" if getattr(s,"calibration_missing",None) is False else "Choose camera calibration","primary",False);hint=s.calibration_error
        b['learn']=(b['learn'][0],b['learn'][1],False)
    b["hint"] = hint
    if (getattr(s,'range_test',None) or {}).get('active'):
        b['go']=('Range test running','primary',False);b['hint']='Range test open'
        for key in ('build','new','touch','learn'):
            if key in b:b[key]=(b[key][0],b[key][1],False)
    return b


def _when(learned, today=None):
    """"16:52" for a bed learned today, "Sep 26 16:52" this year, else "Sep 26 2025"."""
    import datetime
    try: t = datetime.datetime.strptime(learned[:19], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError): return ""
    today = today or datetime.date.today()
    if t.date() == today: return t.strftime("%H:%M")
    return "%s %d %s" % (t.strftime("%b"), t.day, t.strftime("%H:%M") if t.year == today.year else t.year)


def bed_card(s, side_names=(), today=None):
    """The Bed card on top of the sides: None without the bed adapter, else dict(kind: "none" / "learning" / "learned"
    / "provisional", sub: the line under "Bed", deg: the degrees turned while learning (else None), clickable: whether
    a click may show the bed big (a learned bed, nothing running)). side_names: the sides' folder names in card order
    (a provisional bed names the side it came from)."""
    if not s.bed_available: return None
    info = s.bed; deg = None
    if s.learning:
        st = s.learn_status or {}; deg = min(360, int(st.get("deg", 0)))
        kind, sub = "learning", ("Learning: %d of 360\u00b0, %d of %d dots" % (deg, st.get("dots", 0), st["want"])) if st.get("want") else ("Learning: %d of 360\u00b0, %d dots" % (deg, st.get("dots", 0)))
    elif (s.learn_status or {}).get("state") == "failed":
        kind = "failed"
        sub = "Learn failed; using the last learned bed" if info is not None else "Learn failed; try Learn bed again"
    elif info is None:
        kind, sub = "none", "Not learned yet \u00b7 press Learn bed"
        from app2.scan.bedhook import FREEHAND
        if s.bed_profile == FREEHAND: sub = "Freehand \u00b7 nothing cut, move freely"
        elif s.bed_profile: sub = "%s \u00b7 not learned yet" % s.bed_profile
    elif info.kind == "plane":
        kind, sub = "plane", "No dots seen \u00b7 bed found from depth"
    elif info.provisional:
        no = side_names.index(info.side) + 1 if info.side in side_names else None
        kind = "provisional"
        sub = "%d dots \u00b7 from %s (this project only)" % (len(info.dots), "side %d" % no if no else "a side")
    else:
        when = _when(info.learned, today)
        kind, sub = "learned", "%d dots" % len(info.dots) + (" \u00b7 learned %s" % when if when else "")
    clickable = info is not None and not (s.learning or s.recording or s.busy or s.building)
    return dict(kind=kind, sub=sub, deg=deg, clickable=clickable)


def _num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def status_view(s, last, side_stats, fps, base_text, auto_msg, quiet=0.0, gone_text=""):
    """The status line from the state (Tk tick()'s bar in sentence case). last: the newest tracking result or None;
    side_stats: (frames, kept, lost) of the picked side or None; quiet: seconds without a frame; gone_text: the
    source's own words when it is gone (scanner unplugged, replay played out)."""
    no = s.picked; banner = None
    if s.cam == "noframes":
        kind, pill = "noframes", "No frames"
        banner = ("warn", ("No frames from the scanner for %d s. Side %d is paused and will reconnect by itself." % (quiet, no))
                  if no is not None else "No frames from the scanner for %d s. It reconnects by itself." % quiet,
                  "Check the USB cable and that the scanner is still in PC mode.")
    elif s.cam == "gone":
        kind, pill = "gone", "Scanner disconnected"; banner = ("lost", gone_text, "")
    elif s.cam == "replay_end":
        kind, pill = "replay_end", "Replay finished"; banner = ("info", gone_text, "")
    elif s.slow and not s.recording:
        kind, pill = "slow", "Tracking too slow"
        banner = ("warn", "Tracking got too slow, so scanning paused. Press Finish and report it.", "")
    elif s.building and not s.recording:
        kind, pill = "building", "Building the model"
    elif not s.recording:
        kind, pill = ("idle", "Not scanning") if no is None else ("paused", "Side %d paused" % no)
    elif last is None or last["ok"]:
        kind = "tracking"
        pill = "Side %d tracking" % no + (" (found again by the table dots)" if last and last.get("hinted")
                                          else " (found again)" if last and last.get("reloc") else "")
    elif last.get("few"): kind, pill = "lost", "Side %d: too little of the part in view" % no
    elif last.get("unsure") and s.picked_touch: kind, pill = "lost", "Side %d: finding the part on the model" % no
    elif last.get("unsure"): kind, pill = "lost", "Side %d lost: finding where it was after the pause" % no
    else: kind, pill = "lost", "Side %d lost: turn back to where it was green" % no
    if s.recording and getattr(s,'lost_display',False) and last and last['ok']:
        kind,pill='lost','Side %d: tracking is settling' % no
    if getattr(s,'stream_warning',None) and banner is None:banner=('warn',s.stream_warning,'If this keeps happening, pause, then restart the scanner.')
    stats = []
    if side_stats is not None:
        stats.append(("{} frames", str(side_stats[0]), None))
        stats.append(("{} lost", str(side_stats[2]), "lost" if side_stats[2] else None))
    stats.append(("{} fps", "%.1f" % fps, "warn" if (s.recording and fps < 1.0) else None))
    if last and last["ok"] and s.recording and _num(last.get("fit")) and _num(last.get("rmse")):
        stats.append(("fit {}", "%.2f" % last["fit"], None)); stats.append(("error {} mm", "%.2f" % last["rmse"], None))
    if no is not None and base_text:
        word, _, val = base_text.partition(": ")
        stats.append((word + ": {}", val, "platter" if val.startswith("bed") else None))
    return StatusView(kind=kind, pill=pill, stats=stats, hint=buttons(s)["hint"], banner=banner,
                      auto_msg=auto_msg if (auto_msg and s.recording) else "")


def stats_text(view):
    """The stats as plain text ("412 frames  0 lost  3.1 fps ..."), for logs and tests."""
    return "  ".join(t.replace("{}", v) for t, v, _ in view.stats)
