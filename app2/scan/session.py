# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scan session without a window: the project and its sides, the frame source, the tracker, the platter, Auto
range / Auto gain, the stream watchdog, saving, and build runs. Everything the Tk scanner did in its window's tick().

Threads (the window's thread never waits on any of them):
  scan-worker   owns the project, every Side, the tracker (one CUDA model at a time, only ever touched here), the
                platter, Auto range / Auto gain, the image-mode orbit and the bed hook's state. Takes commands and
                frames, in that order: a Pause pressed now stops at the frame being handled; frames behind it are only
                shown. Slow steps (reloading a side, saving, making a tracker) run here too, with `busy` set so the
                window shows them and greys out what would clash.
  source pumps  LiveSource: depth + colour readers (v4l2-ctl pipes); ReplaySource: one player. They queue frames, and
                while the worker is busy or behind hand the newest one to the composer too (_preview): the IR and
                depth pictures never wait for tracking, only the model view does.
  composer      turns the newest frame into the live pictures (drops older ones), preview.py.
  device        gain / fill LED calls to the scanner, one at a time (sources.DeviceControl).
  build         a build process and its STAGE lines (builder/runner.py).
emit(kind, payload) is called from those threads; the Qt bridge turns each kind into a signal. Payloads are never
changed after they are emitted. Kinds: state, status, sides, live, pose, cloud, model_image, turn, range, gain, log,
build, build_line, closed, error (see the table in the 2.0 spec, section 4).

model_view "points": the window draws the tracker's own cloud in a GL view (cloud and pose events, the camera lives in
the window). "image": no GL; the worker renders the model picture as the Tk scanner did (model_image events)."""
import glob, os, queue, subprocess, sys, threading, time, traceback
from collections import deque
import numpy as np, cv2
from app2 import REPO
from app2.scan import fmt
from app2.scan.assist import AutoRange, AutoGain, auto_platter, well_covered, dial, Orbit
from app2.scan.bedhook import NullBed
from app2.scan.bedcut import BedCut, LOW_MM
BED_LIFT_MAX_MM = 50.0        # Bed height goes this far up or down
from app2.scan.preview import Composer, decode_colour
from app2.scan.project import Project
from app2.scan.tracking import TrackerHost, cuda_tracker, free_gpu, LIVE_MAX_DEPTH_MM
from app2.scan.ui_model import ScanState, status_view
from app2.builder.runner import BuildRunner, LegacyBuild
from track import F, REGISTRATION_MIN_POINTS
from fuse import read_calib
from table_plane import to_camera

VIEWER_TITLE = "built model (red = scan here)"
REC_QUEUE = 300             # frames kept waiting while recording (or held for a Start): the Tk scanner's RAWQ_REC (~580 MB)
# what a command is called in "... did not work" (a command that fails is logged; the session carries on)
CMD_NAMES = dict(_go="Start / Pause", _new_side="+ New side", _pick_no="Picking a side", _delete="Delete side",
                 _set="A setting", _set_auto_gain="Auto gain", _set_led="Fill LED", _orbit="Turning the view", _home="Home view",
                 _build="Build model", _cancel_build="Stop build", _learn="Learn bed", _pick_bed="Pick bed", _view_model="View in 3D",
                 _build_done="The build's result", _set_remove_base="Remove base",
                 _side_mesh_ready="Showing a saved side", _touch_up="Touch up", _rewind="Back", _forward="Forward")
SIDE_CLOUD_KEEP = 300_000   # points of a saved side's cloud kept for showing it again (this session's sides)


def _turn(rad, R):
    """A platter tuple's size into another frame: a radius stays, a placed outline (bed.PlacedOutline, a bed that is not
    a circle about the spin centre) has its x axis turned by R."""
    return rad.rotated(R) if hasattr(rad, "rotated") else rad


def launch_viewer(argv):
    """The built model in its own 3D window (viewer.py, a separate process: a slow load never holds up the scan).
    Returns the process, so the session opens at most one such window."""
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def raise_viewer(proc):
    """The viewer's window to the front (xdotool by its pid, when there is one); fire and forget."""
    import shutil
    pid = getattr(proc, "pid", None)
    if not pid or not shutil.which("xdotool"): return
    try: subprocess.Popen(["xdotool", "search", "--onlyvisible", "--pid", str(pid), "windowactivate"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError: pass


BURST_FRAMES = 25; SHOT_FIRST = 10; SHOT_STEP = 5      # a burst, and Single shot's first / added frames (about 15-25 frames turn Coverage blue)

class ScanConfig:
    """projects_dir: where new ProjectLive folders go. param_dir: the calibration folder copied into every new side.
    project / last: continue that project / the newest one. near, far (mm), gain, auto_range, auto_gain: the start
    settings (the Tk scanner's). model_view: "points" (GL view in the window) or "image" (worker renders).
    replay_settings: frames that carry a saved stamp (replay) are cut and stamped with the range, gain and LED they were
    recorded with (repeats a scan exactly). tracker_factory(cal, w, h), release(): make a tracker, free the GPU.
    view_launcher(argv): opens the built model and returns its process (tests pass a recorder); while that
    process runs, view_raiser(proc) brings its window to the front instead of opening another. below_bed: Cut below bed at the start (None: on,
    except with replay_settings, whose recordings do not say whether it was on: off, so they repeat exactly).
    remove_base: the model view leaves the platter out (the build always cuts it)."""
    def __init__(self, projects_dir, param_dir, project=None, last=False, near=None, far=None, gain=None, auto_range=None,
                 auto_gain=None, model_view="points", build_backend=None, replay_settings=False,
                 tracker_factory=cuda_tracker, release=free_gpu, view_launcher=launch_viewer, w=fmt.W, h=fmt.H,
                 below_bed=None, remove_base=True, view_raiser=raise_viewer, touch_factory=None, save_color=True, outside_below_mm=None):
        self.projects_dir, self.param_dir, self.project, self.last = projects_dir, param_dir, project, last
        from app2.scan.settings import load
        saved=load()
        def valid(key,value,default,lo,hi):
            try: n=int(saved.get(key,default) if value is None else value)
            except (TypeError,ValueError): n=default
            return max(lo,min(hi,n))
        self.near=valid("near",near,100,0,2000);self.far=valid("far",far,450,1,2000)
        try: stored_gain=int(saved.get("gain"))
        except (TypeError,ValueError): stored_gain=None
        self.initial_auto_gain=gain is None and not (stored_gain is not None and 1<=stored_gain<=15)
        self.gain=valid("gain",gain,1,1,15)
        self.auto_range=bool(saved.get("auto_range",True)) if auto_range is None else bool(auto_range)
        self.saved_range=near is None and far is None and "near" in saved and "far" in saved
        self.auto_gain=bool(saved.get("auto_gain",self.initial_auto_gain)) if auto_gain is None else bool(auto_gain)
        self.save_color=bool(save_color)
        from app2.scan.tracking import LIVE_MAX_DEPTH_MM
        self.outside_below_mm=valid('outside_below_mm',outside_below_mm,0,0,LIVE_MAX_DEPTH_MM)
        self.model_view = model_view; self.build_backend = build_backend or LegacyBuild()
        self.replay_settings = replay_settings; self.tracker_factory, self.release = tracker_factory, release
        self.view_launcher = view_launcher; self.view_raiser = view_raiser; self.w, self.h = w, h
        self.below_bed = (not replay_settings) if below_bed is None else bool(below_bed); self.remove_base = bool(remove_base)
        # touch_factory(model, cands, platters) -> tracker factory for Touch up (None: touchup.model_tracker_factory)
        self.touch_factory = touch_factory
        self.scan_mode=saved.get('scan_mode','near')
        if self.scan_mode not in ('near','far'):self.scan_mode='near'
        self.far_param=saved.get('far_param');self.near_param=saved.get('near_param',param_dir)
        self.motion_provider=None;self.motion_calibration=None;self.motion_log=None
        value=saved.get('near_range')
        self.near_range=tuple(value) if isinstance(value,list) and len(value)==2 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in value) else (self.near,self.far)


from app2.scan.session_framing import FramingSession
from app2.scan.object_modes import label as object_label


from app2.scan.session_guides import ScanGuides


from app2.scan.session_distance import DistanceSession,distance_frame


from app2.scan.session_live_build import LiveBuildSession

class ScanSession(FramingSession,ScanGuides,DistanceSession,LiveBuildSession):
    def __init__(self, cfg, source, emit, bed=None):
        self.cfg, self.source, self._emit_raw = cfg, source, emit
        self.bed = bed or NullBed(); self.gl = cfg.model_view == "points"
        self._cmds = queue.Queue(); self._frames = deque(); self._flock = threading.Lock(); self._wake = threading.Event()
        self._arrivals = deque(maxlen=64); self._stop = False; self._closed_sent = False; self.closed = threading.Event()
        # scan state: written by the worker thread only
        self.project = None; self.host = None; self.cal = None; self.composer = None; self.picked = None
        self.mode_detector=None;self.mode_evidence={};self.mode_notice=''
        from app2.scan.light_timer import LightTimer
        self.light_timer = LightTimer(); self._light_seen = None; self._led_auto = False
        try: self.light_timer.idle_wait = 60.0 * __import__("app2.general_prefs", fromlist=["get"]).get("light_minutes")
        except Exception: pass   # True only while the light is on because the app turned it on for color
        self.scan_mode=getattr(cfg,"scan_mode","near");self._near_range=getattr(cfg,"near_range",(cfg.near,cfg.far));self._scan_depth=None
        self.motion_provider=getattr(cfg,"motion_provider",None);self.motion_log=None
        self.capture_mode = "continuous"; self.shot_deadline = None; self.shot_need = 1; self.shot_got = 0; self.shot_presses = {}; self._gain_t = 0.0; self._smart = dict(seen=0, skipped=0, lv=None, at=0)
        self.recording = False; self.busy = None; self.building = False; self.build_msg = None; self.runner = None
        self.cam = None; self.quiet = 0.0; self.cam_tried = 0.0; self.slow = False; self.last = None
        self.table = None; self.table_pose = None; self.orbit = None; self.orbit_dirty = False; self.home_dirty = False
        self.near, self.far, self.gain = cfg.near, cfg.far, cfg.gain
        self._initial_auto_gain=cfg.initial_auto_gain and cfg.auto_gain and not cfg.replay_settings
        from app2.scan import settings as _settings
        self.save_color=bool(_settings.load().get('save_color',cfg.save_color));cfg.save_color=self.save_color   # the user's choice sticks across projects and restarts
        self._color_writer=None;self._color_packet=None
        self.auto_range, self.auto_gain, self.led = cfg.auto_range, cfg.auto_gain, False
        self.autorange = AutoRange(); self.autogain = AutoGain(); self._shown = None; self._seq = 0; self._behind_said = False
        if cfg.saved_range: self.autorange.rng=(self.near,self.far);self.autorange.wide=self.autorange.tight=0
        self._holding = False; self._dropped = 0; self._dropped_said = 0.0; self._then = None; self.home_seq = 0
        from app2.scan.stream_health import StreamHealth
        self.stream_health=StreamHealth(self._arrivals.maxlen);self.stream_warning="";self.range_guide=None;self.lost_display_until=0.
        self._unsure_said = None
        # showing a saved side again (picked, not loaded on the tracker): its last cloud from this session (GL mode),
        # else its own mesh from the last build, else its picture. _viewer: the one separate 3D window (viewer.py)
        self._side_clouds = {}; self._viewer = None
        # Cut below bed: the pixels under / beside the platter are zeroed before tracking and saving (bedcut.py). The
        # platter in the camera frame is kept for the pass (it does not move while the table turns), so lost frames are
        # cut too. cut_poses: the pass's last tracked poses (Auto's platter is seen from their average: one pose alone
        # tipped it by up to 1 deg). cut_px: pixels cut from the last frame; cut_said: the (side, pass) the log named
        self.below_bed, self.remove_base = cfg.below_bed, cfg.remove_base
        self.outside_below_mm=cfg.outside_below_mm
        self.bed_lift_mm=0.0                     # Bed height: the live cut plane raised (+) or lowered (-), this session only
        self.bedcut = None; self._cut_geo_kept = None; self._cut_poses = deque(maxlen=30); self.cut_px = 0; self._cut_said = None
        # the live pictures (_preview, _show): the newest frame handed to the composer, and what the worker last cut
        # with / tinted, for the source thread's previews (plain attribute reads)
        self._shown_t = None; self._cut_geo_last = None; self._overlay_last = None; self._preview_cut = None; self._marks_last = None
        self._init_live_build()
        self.thread = threading.Thread(target=self._run, name="scan-worker")

    def start(self): self.thread.start(); return self

    # ------------------------------------------------------------------ commands: any thread, never wait
    def _post(self, fn, *a, **k): self._cmds.put((fn, a, k)); self._wake.set()
    def toggle_scan(self): self._post(self._go)                       # the big button (and Space)
    def new_side(self): self._post(self._new_side)
    def pick_side(self, no): self._post(self._pick_no, no)            # 1-based, as the cards number them
    def delete_side(self, no): self._post(self._delete, no)           # any side, by its number
    def set_range(self, near, far): self._post(self._set, near=int(near), far=int(far))
    def set_auto_range(self, on): self._post(self._set, auto_range=bool(on))
    def set_gain(self, g): self._post(self._set, who="the slider or the [ ] keys", gain=int(min(15, max(1, g))))
    def set_auto_gain(self, on): self._post(self._set_auto_gain, bool(on))
    def extend_led(self): self._post(self._extend_led)

    def _extend_led(self):
        self.light_timer.extend(time.monotonic()); self._emit_state()

    def _tick_light(self):
        self.light_timer.activity(time.monotonic(),getattr(self,"recording",False) or getattr(getattr(self,"bed",None),"learning",False))
        device=getattr(getattr(self,"source",None),"device",None)
        state=getattr(device,"led_state",None)
        if state is not None and getattr(getattr(self,"source",None),"has_led",False):self.led=state=="on"
        if self.light_timer.tick(time.monotonic()) and getattr(self,"led_requested",self.led) and getattr(self,'_led_auto',True):   # a manual light is never turned off for the user
            try: self._set_led(False); self._log("Turned the white light off: no answer to the reminder.")
            except Exception as e: self._log("Could not turn off fill light: %s" % e)
        remaining = self.light_timer.remaining(time.monotonic())
        if remaining != self._light_seen:
            self._light_seen = remaining; self._emit_state()

    def set_save_color(self,on):self._post(self._set_save_color,bool(on))
    def set_smooth_color(self,on):
        """Off (the default): the color picture moves with the depth frames, so the depth dots sit where they belong
        (the 10-14 build's feel). On: color runs ahead at its own 26 fps."""
        self.smooth_color=bool(on)
    def set_ir_view(self,name):
        if name in ('Left','Right','Both'):
            self.ir_view=name
            if self.composer is not None:self.composer.ir_mode=name
    def _set_save_color(self,on):
        self.save_color=on;self.cfg.save_color=on
        from app2.scan import settings
        settings.save(save_color=bool(on));self._emit_state(force=True)

    def set_led(self, on): self._post(self._set_led_manual, bool(on))
    def _set_led_manual(self, on):
        self._manual_led=bool(on);self._scan_led_owned=False;self._led_auto=False;self._set_led(on)            # the user's own on/off: never turned off for them
    def set_below_bed(self, on): self._post(self._set, below_bed=bool(on))     # Cut below bed
    def set_outside_below(self, mm): self._post(self._set_outside_below, mm)
    def _set_outside_below(self, mm):
        from app2.scan.tracking import LIVE_MAX_DEPTH_MM
        try:value=float(mm)
        except (ValueError,TypeError):return
        if not np.isfinite(value) or not 0<=value<=LIVE_MAX_DEPTH_MM:return
        self.outside_below_mm=value;self.cfg.outside_below_mm=value
        for cut in (self.bedcut,self._preview_cut):
            if cut is not None:cut.outside_below_mm=value
        if self.composer is not None:self.composer.outside_below_mm=value
        from app2.scan.settings import save
        save(outside_below_mm=value);self._emit_state()
    def set_bed_lift(self, mm): self._post(self._set_bed_lift, mm)
    def _set_bed_lift(self, mm):
        """Bed height: the live Cut below bed plane (and its purple tint) moved up or down by mm, while recording too.
        The user's hand when the plane sits above the part (2026-10-22: the part cut, nothing fused). Not saved."""
        try: value=float(mm)
        except (ValueError,TypeError): return
        if not np.isfinite(value): return
        value=max(-BED_LIFT_MAX_MM,min(BED_LIFT_MAX_MM,value))
        if value!=self.bed_lift_mm: self._log("bed height %+g mm" % value)
        self.bed_lift_mm=value; self._emit_state()
    def _lifted(self, geo):
        """A camera platter (n, off, centre, size) with the plane raised by bed_lift_mm (n points at the camera)."""
        lift=getattr(self,'bed_lift_mm',0.0)
        return geo if geo is None or not lift else (geo[0], geo[1]-lift, geo[2], geo[3])
    def set_remove_base(self, on): self._post(self._set_remove_base, bool(on))
    def orbit_drag(self, dx, dy): self._post(self._orbit, dx, dy, 0.0)   # image mode; the GL view turns by itself
    def zoom(self, step): self._post(self._orbit, 0, 0, step)
    def home(self): self._post(self._home)
    def build(self, backend=None): self._post(self._build, backend)
    def set_motion_provider(self,provider):self._post(self._set_motion_provider,provider)

    def _set_motion_provider(self,provider):
        previous=getattr(getattr(self,'motion_provider',None),'__self__',None)
        if self.motion_log is not None and hasattr(previous,'set_recorder'):previous.set_recorder(None)
        self.motion_provider=provider
        if provider is not None and self.motion_log is None and self.project is not None:
            from app2.scan.motion_seed import MotionLog
            context=dict(mode=getattr(self,'scan_mode','near'));self._stamp_scan_profile(context)
            log_path=getattr(self.cfg,'motion_log',None) or os.path.join(self.project.path,'build','motion',str(time.time_ns())+'.jsonl')
            self.motion_log=MotionLog(log_path,getattr(self.cfg,'motion_calibration',None),provenance=getattr(self.cfg,'motion_provenance','recorded'),context=context)
        if self.motion_log is not None:
            client=getattr(provider,'__self__',None)
            if hasattr(client,'set_recorder'):client.set_recorder(self.motion_log.append)

    def set_scan_mode(self,mode):self._post(self._set_scan_mode,mode)

    def _set_scan_mode(self,mode):
        if getattr(self,"_range_sweep",None) is not None:return
        from app2.scan.scan_mode import profile,write
        if mode not in ('near','far'):raise ValueError('Choose Small part or Large part.')
        if self.recording or self.busy or self.building or self.bed.learning:return
        if self.picked is not None and self.picked.frames:
            self._log('Start a new side before changing the object setting.');return
        if self.scan_mode=='near' and mode=='far':self._near_range=(self.near,self.far)
        self.scan_mode=mode
        if not self._select_calibration(mode):
            self._log(self.calibration_error);self._emit_state(force=True);return
        if getattr(self,'motion_log',None) is not None:self.motion_log.set_context(mode=mode)
        if mode=='near':self.near,self.far=self._near_range
        result=profile(mode,self.cal,self._scan_depth,self.near,self.far)
        self.near,self.far=result['near'],result['far']
        self._stamp_scan_profile(result)
        from app2.scan.settings import save
        save(scan_mode=mode,near=self.near,far=self.far,near_range=list(self._near_range))
        if self.picked is not None:
            self.picked.scan_profile=result;write(self.picked.folder,result)
            if self.host is not None and self.host.has(self.picked):self.host.new(self.picked)
        self._sync_scan_geometry();self._emit('range',(self.near,self.far));self._emit_sides();self._emit_state()

    def set_far_calibration(self,folder):self._post(self._set_far_calibration,folder)

    def _set_far_calibration(self,folder):
        if self.recording or self.busy or self.building or self.bed.learning:return
        from app2.scan.calibration import store_far
        from app2.scan.settings import path,save
        selected=store_far(folder,path().parent,(self.cfg.w,self.cfg.h))
        self.cfg.far_param=selected['folder'];save(far_param=selected['folder']);self._reset_mode_detection()
        if self.scan_mode=='far' and not (self.picked is not None and self.picked.frames):self._set_scan_mode('far')
        self._emit_state(force=True)

    def _select_calibration(self,mode):
        from app2.scan.calibration import load,CalibrationUnavailable,copy_parameters
        cfg=self.cfg
        folder=getattr(cfg,'far_param',None) if mode=='far' else getattr(cfg,'near_param',cfg.param_dir)
        try:selected=load(folder or '',mode,(cfg.w,cfg.h))
        except CalibrationUnavailable as error:
            self.calibration_error=str(error);return False
        self.calibration_error='';self.cal=selected['intrinsics'];self.calibration_source=selected['source'];cfg.param_dir=selected['folder']
        project=getattr(self,'project',None)
        if project is not None:project.param_dir=cfg.param_dir
        side=getattr(self,'picked',None)
        if side is not None and not side.frames:copy_parameters(selected,os.path.join(side.folder,'param'))
        host=getattr(self,'host',None)
        if host is not None:host.release();host.cal=self.cal
        from app2.scan import preview
        preview.set_rgb_calib(cfg.param_dir)
        self._setup_guide_calibration(selected)
        composer=getattr(self,'composer',None)
        if composer is not None:composer.cal=self.cal;composer.area=None;composer.scan_geometry=None;composer.zref=None;composer.reference_plane=None
        if composer is not None and hasattr(composer,'color_observer'):composer.color_observer.clear()
        observer=getattr(getattr(self,'bed',None),'color_observer',None)
        if observer is not None:observer.clear()
        if composer is not None:
            from app2.scan.stereo_view import Stereo
            composer.stereo=Stereo.load(self.cfg.param_dir,self.cal)
            if hasattr(composer,"footprint"):composer.footprint.reset()
        if getattr(self,"bedcut",None) is not None:self.bedcut=BedCut(self.cal,cfg.w,cfg.h,outside_below_mm=self.outside_below_mm);self._preview_cut=BedCut(self.cal,cfg.w,cfg.h,outside_below_mm=self.outside_below_mm)
        bed=getattr(self,'bed',None)
        if project is not None and bed is not None:
            bed.close();bed.open(project.path,cfg.param_dir,self.cal,self._log)
            if side is not None:bed.side_opened(side,False)
        self._reset_framing()
        return True

    def set_near_calibration(self,folder):self._post(self._set_near_calibration,folder)

    def _set_near_calibration(self,folder):
        if self.recording or self.busy or self.building or self.bed.learning:return
        from app2.scan.calibration import store
        from app2.scan.settings import path,save
        selected=store(folder,path().parent,'near',(self.cfg.w,self.cfg.h))
        self.cfg.near_param=selected['folder'];save(near_param=selected['folder'])
        if self.scan_mode=='near' and not (self.picked is not None and self.picked.frames):self._set_scan_mode('near')
        self._reset_mode_detection();self._emit_state(force=True)

    def _capture_calibration_error(self):
        mode=getattr(self,'mode_evidence',{}).get('mode')
        if mode and mode!=getattr(self,'scan_mode','near'):
            return 'The camera looks like %s. Change Object to match before you start.' % object_label(mode)
        # No dot evidence (a bed that was never learned, a part covering the dots) is not evidence of the
        # other camera: the Object setting stands, and Learn bed stays open so a new bed can be learned.
        return getattr(self,'calibration_error','')

    def _reset_mode_detection(self,preserve=False):
        if getattr(self,'mode_detector',None) is not None:self.mode_detector.close()
        previous=getattr(self,'mode_evidence',{}) if preserve else {}
        self.mode_detector=None;self.mode_evidence=dict(previous);self.mode_notice=''
        self._mode_checked_at=None;self._mode_signature=None;self._mode_background=False;self._mode_has_result=bool(previous.get('mode'))
        if getattr(getattr(self,'source',None),'kind',None)!='live':return
        from app2.scan.calibration import load,CalibrationUnavailable
        from app2.scan.mode_detection import Detector
        bundles={}
        for mode in ('near','far'):
            folder=getattr(self.cfg,mode+'_param',None)
            if not folder:continue
            try:bundles[mode]=load(folder,mode,(self.cfg.w,self.cfg.h))
            except CalibrationUnavailable:pass
        if len(bundles)<2:
            self.mode_notice='Set up both cameras (Settings, Camera setup) so the app can check which one is in use.';return
        self.mode_detector=Detector(bundles,getattr(self.bed,'bed',None))
        self.mode_notice='Checking which camera is in use...'

    def _check_camera_mode(self,fr):
        detector=getattr(self,'mode_detector',None)
        if detector is None:return
        if self.recording and (getattr(self,'mode_evidence',None) or {}).get('mode'):return   # settled before Start: not every frame
        from app2.scan.mode_detection import signature,changed
        sig=signature(fr.depth,fr.irl);now=fr.t
        previous=getattr(self,'mode_evidence',{});confirmed=previous.get('mode')
        scene_changed=changed(getattr(self,'_mode_signature',None),sig)
        evidence=detector.poll()
        if evidence is not None and evidence.get('frames') and changed(evidence['frames'][-1].get('_signature'),sig):
            detector.begin();evidence=None;self._mode_background=bool(confirmed and not scene_changed)
            if scene_changed:self.mode_evidence={};self._reset_framing(preserve_undo=True)
        if evidence is not None:
            mode=evidence.get('mode');state=evidence.get('state');was_recording=self.recording
            votes=evidence.get('votes',{});other='far' if confirmed=='near' else 'near'
            from scanner.bed import ACCEPT_HITS
            contrary=bool(confirmed and votes.get(other,0)>=ACCEPT_HITS and votes.get(other,0)>votes.get(confirmed,0))
            held=bool(confirmed and not mode and not scene_changed and not contrary)
            if not held:self.mode_evidence=evidence
            if mode:
                row=evidence['frames'][-1].get(mode,{})
                count=row.get('inliers',0);rms=row.get('rms_mm');z=sig['distance']
                detail=('%d dots, %.2f mm fit' % (count,rms)) if rms is not None else 'color circles agree'
                self.mode_notice='Camera match: '+object_label(mode)+'.'
                self._mode_signature=sig
                if mode!=self.scan_mode and self.recording:
                    self._pause(self.picked,'Camera changed. Use '+object_label(mode)+' on a new side before continuing.')
                if self.bed.learning and mode!=self.scan_mode:self._learn()
                if mode!=self.scan_mode and not getattr(self,'_mode_has_result',False) and not was_recording and not (self.picked is not None and self.picked.frames):
                    self._set_scan_mode(mode)
                self._mode_has_result=True
                self._refine_overlay(row,evidence)
            elif held:
                # Losing sight of dots is not evidence of another camera. Keep
                # both the confirmed notice and Start gate while logging this check.
                pass
            elif state in ('uncertain','unavailable'):
                self.mode_notice='Could not check the camera (no bed dots seen), so your Object setting is used. Learn bed to check it.'
                if contrary and self.recording:self._pause(self.picked,'The camera looks different now. Make sure the bed dots are in view before you go on.')
            else:self.mode_notice='Checking which camera is in use...'
            if state!='checking' or mode:
                self._mode_checked_at=now;self._mode_background=False
                if not confirmed and not mode:self._mode_signature=sig
                prefix='Camera check held. ' if held else self.mode_notice+' '
                self._log(prefix+'Evidence: '+__import__('json').dumps(evidence,sort_keys=True))
            if self.picked is not None and not held:
                from app2.scan.scan_mode import write
                self.picked.scan_profile['camera_detection']=self.mode_evidence;write(self.picked.folder,self.picked.scan_profile)
            if not held:self._emit_state(force=True)
        checking=detector.evidence.get('state')=='checking'
        elapsed=now-getattr(self,'_mode_checked_at',now) if getattr(self,'_mode_checked_at',None) is not None else 0
        large=changed(getattr(self,'_mode_signature',None),sig)
        if not checking and detector.future is None and ((not self.recording and elapsed>=Composer.COVER_S) or large):
            detector.begin();self._mode_background=not large
            if large:self.mode_evidence={};self._reset_framing(preserve_undo=True)
            if large or not self.mode_evidence.get('mode'):self.mode_notice='Checking the camera again after a scene change.' if large else 'Checking the camera again.'
            self._emit_state(force=True)
        if detector.evidence.get('state')=='checking':
            jpeg=self.source.colour() if getattr(self.source,'colour',None) else None
            detector.offer(fr.depth,fr.irl,jpeg,sample_signature=sig)

    def _stamp_scan_profile(self,profile):
        import hashlib,copy

        try:digest=hashlib.sha256(open(os.path.join(self.cfg.param_dir,'Pl.bin'),'rb').read()).hexdigest()
        except OSError:digest=None
        profile.update(calibration_sha256=digest,intrinsics=list(self.cal),depth_size=[self.cfg.w,self.cfg.h],sensor_mode_verified=False,
                       calibration_source=getattr(self,'calibration_source',None),calibration_error=getattr(self,'calibration_error',''),
                       camera_detection=getattr(self,'mode_evidence',{}))

    def _sync_scan_geometry(self):
        composer=getattr(self,'composer',None)
        if composer is not None and hasattr(composer,'set_scan_geometry'):
            composer.set_scan_geometry(self.near,self.far,getattr(self,'scan_mode','near'))
            if getattr(self,'range_profile',None) is not None:composer.dot_bands=self.range_profile

    def _motion_snapshot(self,t):
        provider=getattr(self,'motion_provider',None)
        if provider is None:return None
        calibration=getattr(self.cfg,"motion_calibration",None)
        if not isinstance(calibration,dict):calibration={}
        offset=calibration.get("received_to_depth_offset_s",0.)
        if not isinstance(offset,(int,float)) or not np.isfinite(offset):offset=0.
        try:return provider(t-offset)
        except TypeError:
            try:return provider()
            except Exception:return None
        except Exception:return None

    def set_capture_mode(self, mode): self._post(self._set_capture_mode, mode)

    def _set_capture_mode(self, mode):
        if mode not in ("continuous", "single", "burst", "smart", "live") or self.recording or self.busy or self.building: return
        self.capture_mode = mode
        if mode=='live' and self.picked is not None:
            self._prepare_live_side(self.picked)
            self.host.release()
            if self.picked.frames:self._load(self.picked)
            else:self.host.new(self.picked)
        self._emit_state()

    def _shot_count(self, sd):
        """Frames this press takes. Burst: always the full burst. Single shot: 10 on a side's first press, 5 more on each later one."""
        if self.capture_mode == "burst": return BURST_FRAMES
        n = self.shot_presses.get(id(sd), 0); self.shot_presses[id(sd)] = n + 1
        return min(BURST_FRAMES, SHOT_FIRST + SHOT_STEP * n)

    def _check_shot(self, accepted=False):
        if not self.recording or self.capture_mode not in ("single", "burst"): return
        if accepted:
            self.shot_got += 1; self.shot_deadline = time.monotonic() + 8.0         # the wait is for the next tracked frame, not the whole shot
            if self.shot_got < self.shot_need: return
        elif self.shot_deadline is None or time.monotonic() < self.shot_deadline: return
        self.shot_deadline = None; got = self.shot_got
        self._pause(self.picked, ("Saved %d tracked frames." % got) if got and got >= self.shot_need else
                    ("Only %d of %d tracked frames came in. Aim at the part and try again." % (got, self.shot_need)) if got else
                    "No tracked frame. Aim at the part and try again.")

    def cancel_build(self): self._post(self._cancel_build)
    def pause_build(self, on): self._post(self._pause_build, bool(on))       # freeze / go on
    def learn_bed(self, name=None, want=None): self._post(self._learn, name, want)     # name: a new bed profile (None: learn the picked one again); want: dots the user counted
    def pick_bed(self, name): self._post(self._pick_bed, name)        # the Bed card's list: another bed profile
    def view_model(self): self._post(self._view_model)
    def leave_touch_up(self): self._post(self._leave_touch_up)

    def touch_up(self): self._post(self._touch_up)                   # a new side tracked on the built model
    def close(self, then=None): self._post(self._close, then)       # then: None, "new" or ("open", path)
    def rewind(self, n): self._post(self._rewind, int(n))            # Back 5 / 10 / 30 on the paused side
    def forward(self): self._post(self._forward)                     # undo the newest Back

    # ------------------------------------------------------------------ frames in (source threads)
    def _live_jpeg(self, fr):
        """The color frame the live pictures draw with this depth frame: the one exposed nearest to it, so IR, depth and
        color show the same instant (as the 10-14 build did); the newest when Smooth color is on."""
        if not getattr(self,'smooth_color',False) and hasattr(self.source,'color_sample'):
            sample=self.source.color_sample(fr.t)
            if sample is not None: return sample['jpeg']
        return self.source.colour() if getattr(self.source,'colour',None) else None

    def _on_frame(self, fr):
        from app2.scan import timing; timing.mark('frame_in')
        if self.composer is not None: self.composer.lockstep = not getattr(self,'smooth_color',False)
        if self.save_color and hasattr(self.source,"color_sample"):
            fr.meta=dict(fr.meta or {});fr.meta["saved_color"]=self.source.color_sample(fr.t)
        with self._flock:
            self._arrivals.append(fr.t)
            # only shown: the newest frame is all that matters. Kept while recording, and while a command waits or the
            # worker is busy or starting a side (a Start being set up must not lose the frames that come in meanwhile; the
            # Tk window recorded those too). Past REC_QUEUE the oldest go, and the worker says so.
            hold = self.busy or self._holding or not self._cmds.empty()
            if not (self.recording or self.bed.learning or hold):
                while self._frames: self._frames.popleft(); self.source.ack()
            elif len(self._frames) >= REC_QUEUE:
                self._frames.popleft(); self.source.ack(); self._dropped += 1
            self._frames.append(fr)
            behind = len(self._frames) > 1                        # the worker has not taken the one before yet
        # the live pictures show this frame now, before tracking (a frame costs ~250 ms plus the compose, which left the
        # IR and depth tiles a scanner frame behind the color); the worker's post refreshes the marks when it is done
        if self.composer is not None: self._preview(fr)
        self._wake.set()

    def _preview(self, fr):
        """Source thread: the newest frame straight to the live pictures while the worker is busy or behind, so IR and
        depth never wait for tracking (only the model view lags), as the colour view never did. The range cut, and Cut
        below bed with the platter the worker last cut with (its own BedCut: the per-platter set-up is not shared
        between threads). The live pictures never go back to an older frame (shown_t)."""
        d = fr.depth.copy()
        lo,hi=sorted((self.near,self.far));d[(d<lo*10)|(d>hi*10)]=0
        busy = bool(self.busy); geo = self._cut_geo_last; cut = None
        if not busy and geo is not None and self.below_bed and self.recording:
            try: cut = self._preview_cut.apply(d, geo)
            except Exception as e: print("preview cut:", e, flush=True)
        last = self.last
        job = dict(d=d, irl=fr.irl, irr=getattr(fr,"irr",None), lost_now=bool(not busy and self.recording and last is not None and not last["ok"]),
                   geo=None if busy else self._overlay_last, jpeg=self._live_jpeg(fr), seq=fr.seq, cut=cut,
                   framing=None if busy else getattr(self,'framing_view',None))
        if not busy: job.update(getattr(self, '_marks_last', None) or {})   # same marks as the worker's post: no flicker between the two
        self._show(fr.t, job, arrival=True)

    def _show(self, t, job, arrival=False):
        """Hand a frame to the composer unless a newer one was already handed (worker and source thread both post).
        The worker's post for a frame the arrival preview already showed is dropped: composing every frame twice
        held the next frame's picture behind the previous compose (225 ms from arrival to the IR tile, 670 worst)."""
        with self._flock:
            if self._shown_t is not None and (t < self._shown_t or (not arrival and t == self._shown_t)): return
            self._shown_t = t; self.composer.post(job)

    # ------------------------------------------------------------------ the worker
    def _run(self):
        try:
            self._open()
            self.source.start(self._on_frame)
            while not self._stop:
                self._wake.wait(0.25); self._wake.clear()
                self._drain_cmds()
                if self._stop: break
                self._drain_bed(); self._check_shot(); self._tick_light()
                while not self._stop:
                    with self._flock:
                        fr = self._frames.popleft() if self._frames else None; left = len(self._frames)
                    if fr is None: break
                    self._tick_light(); self._check_shot()
                    self._drain_cmds()                            # every command sent before this frame came in applies to it
                    if self._stop: break
                    self._drain_bed()                             # the table finder's results for the frames before this one
                    placing = self.picked is not None and self.picked.ps > 0 and self.picked.pass_ok == 0   # a Continue's check
                    if left > 30 and self.recording and not self._behind_said and not placing:   # far behind: say so
                        self._behind_said = True; self._log("falling behind: %d frames queued" % left)
                    elif left < 5: self._behind_said = False
                    if self._dropped and time.monotonic() - self._dropped_said > 5.0:
                        self._dropped_said = time.monotonic()
                        self._log("the scan is not keeping up: %d frames dropped so far (%d waiting)" % (self._dropped, left))
                    try: self._frame(fr)
                    except Exception as e:                        # one bad frame (disk full, a tracker error): pause, keep going
                        traceback.print_exc()
                        if self.recording: self.recording = False; self._log("frame failed (%s): scanning paused" % e)
                    self.source.ack()
                if self._stop: break
                self._render_pending(); self._watch_stream(); self._emit_state()
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True); self._emit("error", tb)
            try: self._close(self._then, crashed=True)
            except Exception: traceback.print_exc()
        finally:
            if not self._closed_sent:                             # _close itself failed: the window must still hear it
                if self._color_writer is not None:self._color_writer.close();self._color_writer=None
                self._stop = True
                self._emit("closed", dict(then=self._then, project=self.project.path if self.project else None, lines=[], crashed=True))
            self.closed.set()

    def _drain_bed(self):
        """Apply the bed hook's results (the Tk scanner did it every round of its window): log lines, Learn bed's result
        for the hint line, and the picked side's platter when the table dots placed it."""
        lines, changed = self.bed.drain(self.project.sides, self.picked, self._no)
        for line in lines: self._log(line)
        msg = self.bed.take_message()
        if msg: self.build_msg = msg
        if changed and self.picked is not None: self.table = self._platter(self.picked); self._emit_sides();self._emit_turn(self.picked)

    def _drain_cmds(self):
        """Run the waiting commands. One that fails (a CUDA out-of-memory making a tracker, a full disk while saving) is
        logged and the session carries on, as a Tk button callback did; only a failing close ends it."""
        while not self._stop:
            try: fn, a, k = self._cmds.get_nowait()
            except queue.Empty: return
            if fn == self._close: fn(*a, **k); continue
            try: fn(*a, **k)
            except Exception as e:
                traceback.print_exc(); self.busy = None; self._holding = False
                self._log("%s did not work: %s" % (CMD_NAMES.get(getattr(fn, "__name__", ""), "A command"), e))
                try: self._emit_sides(); self._emit_state(force=True)
                except Exception: traceback.print_exc()

    def _open(self):
        cfg = self.cfg
        path = cfg.project or (Project.newest(cfg.projects_dir) if cfg.last else None) or Project.new_path(cfg.projects_dir)
        self.cal = read_calib(os.path.join(cfg.param_dir, "Pl.bin"))
        self.calibration_error='';self.calibration_source=dict(kind='startup calibration',declared_mode='unknown')
        self._select_calibration(getattr(self,'scan_mode','near'))
        from app2.scan import preview; preview.set_rgb_calib(cfg.param_dir)      # the color view's depth guides
        self.host = TrackerHost(self.cal, cfg.w, cfg.h, cfg.tracker_factory, cfg.release)    # made on this thread
        self.bedcut = BedCut(self.cal, cfg.w, cfg.h, outside_below_mm=self.outside_below_mm); self._preview_cut = BedCut(self.cal, cfg.w, cfg.h, outside_below_mm=self.outside_below_mm)
        self.project = Project(path, cfg.param_dir, cfg.w, cfg.h)
        self.bed.open(path, cfg.param_dir, self.cal, self._log)
        for sd in self.project.sides: self.bed.side_opened(sd, True)
        self.composer = Composer(self.cal, self._emit,
                                 colour_fn=self.source.colour if self.source is not None else None)
        self._sync_scan_geometry();self._reset_mode_detection()
        self.composer.outside_below_mm=self.outside_below_mm;self.composer.ir_mode=getattr(self,'ir_view','Left')
        self.composer.bed_color_observer=getattr(self.bed,'color_observer',None)
        calibration=getattr(cfg,'motion_calibration',None)
        if calibration is None:
            import json
            try:calibration=json.loads(open(os.path.join(path,'motion-calibration.json')).read())
            except (OSError,ValueError):calibration=None
        cfg.motion_calibration=calibration
        for side in self.project.sides:side.motion_calibration=calibration
        if getattr(cfg,'motion_log',None) or self.motion_provider is not None:
            from app2.scan.motion_seed import MotionLog
            log_path=getattr(cfg,'motion_log',None) or os.path.join(path,'build','motion',str(time.time_ns())+'.jsonl')
            context=dict(mode=getattr(self,'scan_mode','near'));self._stamp_scan_profile(context)
            self.motion_log=MotionLog(log_path,calibration,provenance=getattr(cfg,'motion_provenance','recorded'),context=context);self._set_motion_provider(self.motion_provider)
        self._log("project folder: %s" % path)
        if self.bed.available and self.bed.size is None:
            self._log("Tip: Learn bed once (empty bed, one turn) and the bed is found right away on every side.")
        self._emit("range", (self.near, self.far)); self._emit("gain", self.gain)
        self._emit_sides(); self._emit_state()

    def _emit(self, kind, payload):
        if self._closed_sent: return                              # a late picture or build line from before the close
        if kind == "closed": self._closed_sent = True
        try: self._emit_raw(kind, payload)
        except Exception: traceback.print_exc()

    def _log(self, msg): print(msg, flush=True); self._emit("log", time.strftime("%H:%M:%S ") + msg)

    def _no(self, sd): return self.project.side_no(sd)

    def _next_seq(self): self._seq += 1; return self._seq

    # ------------------------------------------------------------------ one frame
    @distance_frame
    def _frame(self, fr):
        from app2.scan import timing; timing.mark('frame_start')
        try: return self._frame_body(fr)
        finally: timing.mark('frame_end')

    def _frame_body(self, fr):
        t = fr.t
        if not self.recording and not getattr(self,'_warmed',False):
            self._warmed=True;self._warm_thread=threading.Thread(target=self._warm_up,name='warm-up',daemon=True);self._warm_thread.start()
        self._color_packet=(fr.meta or {}).get("saved_color")
        if hasattr(self.bed,'set_color_frame'):
            from app2.scan.preview import RGB_CAL
            packet=self.source.color_sample(fr.t) if hasattr(self.source,'color_sample') else None
            self.bed.set_color_frame(packet,RGB_CAL)
        if self.cam: self.cam = None; self._reset_mode_detection(); self._log("frames from the scanner again")
        d = fr.depth.copy()
        self._check_camera_mode(fr)
        self._range_test_frame(fr)
        self._scan_depth=fr.depth.copy()
        self._update_bed_observation(fr)
        from app2.scan import timing; timing.mark('scene_start'); scene=self._measure_scene(fr); timing.mark('scene_end')
        previous_warning=self.stream_warning
        self.stream_warning=self.stream_health.add(fr.t,np.count_nonzero(fr.depth))
        if self.stream_warning and not previous_warning:
            self._log('Stream check: '+__import__('json').dumps(self.stream_health.evidence,sort_keys=True))
        note=getattr(self.stream_health,'gap_note',None)
        if note and not note.get('logged'):
            note['logged']=True;self._log('Stream check (late frames, depth fine, no warning): '+__import__('json').dumps({k:v for k,v in note.items() if k!='logged'},sort_keys=True))
        st = fr.meta.get("stamp") if (self.cfg.replay_settings and fr.meta) else None
        if st:                                                    # replay: the settings the frame was recorded with
            self.near, self.far = int(st["near"]), int(st["far"]); self.gain = int(st.get("gain", self.gain)); self.led = bool(st.get("led"))
        if getattr(self,"_initial_auto_gain",False) and self.auto_gain and st is None:
            g=self.autogain.initial(fr.irl,self.gain)
            if g is not None:
                self.gain=g;self._initial_auto_gain=False
                from app2.scan.settings import save
                save(gain=g);self._emit("gain",g)
                self._log("Auto gain chose %d from the IR image." % g)
        g_now = self.gain; self.source.device.gain(g_now)
        if self.auto_range and st is None:
            from app2.scan.surface import safe_range
            span=safe_range(scene,self.scan_mode,(self.near,self.far),(getattr(self,'surface_start',None) or {}).get('range_margin_mm')) if scene['source']=='part' else None
            r = self.autorange.connected(span) if span is not None else self.autorange.update(d)
            if r is not None:
                self.near,self.far=r;self._emit("range",r)
                self._sync_scan_geometry()                         # the preview grays out what this Range leaves out: tell it (it kept an old one)
                from app2.scan.settings import save
                save(near=self.near,far=self.far)
        side=self.picked
        if getattr(self,'scan_mode','near')=='far' and side is not None and not side.frames and 'distance_mm' not in getattr(side,'scan_profile',{}):
            from app2.scan.scan_mode import profile,write
            side.scan_profile.update(profile('far',self.cal,fr.depth,self.near,self.far));self._stamp_scan_profile(side.scan_profile);write(side.folder,side.scan_profile)
            self.near,self.far=side.scan_profile['near'],side.scan_profile['far']
            if self.host.has(side):self.host.new(side)
            self._sync_scan_geometry()
        self._scene_part_mask=scene['mask'] if scene['source']=='part' else None
        from app2.scan.surface import hint as surface_hint
        self.surface_hint=surface_hint(fr.depth,fr.irl,scene)
        if self.auto_gain and st is None and scene.get('source')=='part' and scene.get('mask') is not None:
            part=int(scene['mask'].sum())
            if part>=REGISTRATION_MIN_POINTS:
                g=self.autogain.brightness(self.surface_hint['dark_pixels']/part,self.surface_hint['shiny_pixels']/part,self.gain,time.monotonic())
                if g is not None:
                    self.gain=g;self._gain_t=time.monotonic();self.source.device.gain(g);self._emit("gain",g)
                    from app2.scan.settings import save
                    save(gain=g);self._log("auto gain: %s" % self.autogain.msg[len('Auto gain: '):])
        guides=self._update_scene(fr,scene)
        lo, hi = sorted((self.near, self.far))
        d[(d < lo * 10) | (d > hi * 10)] = 0                      # outside the range: gone from tracking, the view and the saved frame
        irl = fr.irl; sd = self.picked
        cut = self._cut_below_bed(sd, d)                          # below the bed: gone the same way
        if self.bed.learning and not self.recording: self.bed.learn_frame(d, irl, t, g_now)
        if self.recording and sd is not None and not self.busy: self._record(sd, d, irl, t, lo, hi, g_now, irr=getattr(fr,"irr",None))
        lost_now = self.recording and self.last is not None and not self.last["ok"]
        geo = self._overlay_geo(sd); self._overlay_last = geo
        self._marks_last = dict(dots=self.bed.dots(sd, self.recording), **guides)   # the arrival preview draws these too
        self._show(t, dict(d=d, irl=irl, irr=getattr(fr,"irr",None), lost_now=lost_now, geo=geo, jpeg=self._live_jpeg(fr),
                           seq=fr.seq, cut=cut, **self._marks_last))
        self._emit_status()

    def _cut_below_bed(self, sd, d):
        """Cut below bed on this frame (in place) while the picked side records and its platter is known. Returns the
        cut pixels' flat indices (the IR picture dims them) or None."""
        self.cut_px = 0; self._cut_geo_last = None
        if not (self.below_bed and self.recording and sd is not None and not self.busy): return None
        # the table dots' camera platter (this pass's, right on lost frames too), else Auto remove base's seen from the
        # pass's last tracked poses. The platter does not move in the camera's view while the table turns, so Auto's is
        # worked out once per new plane (every 10th tracked frame) and kept for the pass, lost frames included
        geo = self.bed.camera_geo(sd, self.recording); src = "the table dots"
        if geo is None:
            # the scan-frame platter (the dots' fixed plane while the finder looks for them again after the scanner
            # moved, else Auto remove base's) seen from the newest tracked pose. One pose, every frame: averaging the
            # pass's last 30 poses assumed the scanner stays put while the table turns, and a hand-moved scanner got a
            # plane tilted through the part, which cut the part itself (2026-10-22, the whole side purple, nothing fused)
            k = self._cut_geo_kept; tb = self.table; r = self.last
            src = "Auto remove base" if tb is None or len(tb) < 5 or tb[4] != "bed" else "the table dots carried by tracking"
            same = k is not None and k[0] is sd and k[1] == sd.ps
            if tb is not None and r is not None and r.get("ok") and r.get("T") is not None:
                Tm = r["T"] @ F; Tm = Tm.copy(); Tm[:3, 3] *= 1000.0
                geo = self._camera_platter(tb, [Tm]); self._cut_geo_kept = (sd, sd.ps, geo, tb)
            elif same: geo = k[2]                                 # a lost frame: the last one worked out
            else: return None
        geo = self._lifted(geo)
        idx = self.bedcut.apply(d, geo); self.cut_px = len(idx); self._cut_geo_last = geo
        said = self._cut_said
        if said is None or said[:2] != (sd, sd.ps):
            self._log("side %d: cutting below the bed from frame %d on (platter from %s)" % (self._no(sd), sd.idx, src))
        elif said[2] != src:                                      # the console only: the dots' platter may come and go
            print("side %d frame %d: Cut below bed now uses the platter from %s" % (self._no(sd), sd.idx, src), flush=True)
        self._cut_said = (sd, sd.ps, src)
        return idx

    def _camera_platter(self, tb, poses=None):
        """A platter (n, off, centre, radius in the side's scan frame) in the camera frame: from poses (camera -> scan,
        mm), else averaged over the pass's last tracked poses."""
        Ns, Os, Cs = [], [], []
        poses = list(poses) if poses is not None else list(self._cut_poses)
        for Tm in poses:
            nc, oc = to_camera(tb[0], tb[1], Tm); Ns.append(nc); Os.append(oc)
            if tb[2] is not None: Cs.append(Tm[:3, :3].T @ (np.asarray(tb[2], float) - Tm[:3, 3]))
        n = np.mean(Ns, 0); n /= np.linalg.norm(n)
        return n, float(np.mean(Os)), (np.mean(Cs, 0) if Cs else None), (_turn(tb[3], poses[-1][:3, :3].T) if Cs else None)

    def _record(self, sd, d, irl, t, lo, hi, g_now, irr=None):
        host = self.host; no = self._no(sd)
        t_step = time.monotonic()
        self._live_before(sd,t)
        try: self.bed.before_step(sd, d, irl, host, self.last)   # a Continue or a lost frame: start from the table dots
        except Exception as e: print("bed hint:", e, flush=True)
        sparse=getattr(sd,'scan_profile',{}).get('mode')=='far'
        if sparse:
            try:board=self.bed.camera_geo(sd,self.recording)
            except Exception:board=None
            r=host.step(d,after_gap=(sd.ps,sd.idx) if sd.pass_ok==0 else None,timestamp=t,motion=self._motion_snapshot(t),board=board)
        else:
            from app2.scan import timing; timing.mark('step_start')
            r=host.step(d,after_gap=(sd.ps,sd.idx) if sd.pass_ok==0 else None); timing.mark('step_end')
        self.last = r   # a Continue: placed first
        if r["ok"]: sd.pass_ok += 1
        elif r.get("refused"): print("frame %d: a fit far from where the table dots put the camera was refused" % sd.idx, flush=True)
        placed = r.get("placed")
        if placed is not None: self._placed(sd, no, placed, r["ok"])
        took = time.monotonic() - t_step - (placed or {}).get("secs", 0.0)       # the global search is slow on purpose
        if took > 3.0 and sd.idx >= 3:                            # something is badly wrong: stop instead of piling up (not the first
                                                                  # frames of a pass, where the GPU may still be loading its code)
            self.recording = False; self.slow = True
            self._log("frame took %.1f s: scanning paused" % took)
        name = fmt.frame_name(sd.ps, sd.idx)
        M = r["T"] @ F; M[:3, 3] *= 1000.0
        keep = self._live_keep(sd,d,r,t) if self.capture_mode=="live" else not (self.capture_mode == "smart" and self._smart_skip(sd, r))     # Smart frames: a frame that adds nothing is not saved
        color_stamp=dict(part_distance_mm=(getattr(self,"framing_view",None) or {}).get("depth_mm"),right_ir_saved=irr is not None)
        if keep:
            fmt.write_frame(sd.cache, name, d, irl, M, irr=irr)
            if self.save_color:
                packet=self._color_packet
                if packet is not None:
                    if self._color_writer is None:
                        from .color_writer import ColorWriter
                        self._color_writer=ColorWriter()
                    accepted=self._color_writer.submit(os.path.join(sd.cache,name+".img"),packet["jpeg"])
                    color_stamp.update(color_saved=accepted,color_t=packet["t"],color_offset_ms=round((packet["t"]-t)*1000,3),color_seq=packet["seq"],color_frame_id=packet.get("frame_id"),color_sync="saved frame id" if packet.get("frame_id") else "host arrival")
                else:color_stamp.update(color_saved=False,color_sync="no color within 100 ms")
        if r["ok"]:
            if keep: sd.entries.append((sd.ps, sd.idx, M))
            tm = self._touch_model(sd)
            if tm is None:
                if keep: host.tracked(len(sd.entries))          # every 30th: the model is kept for Back
            else:
                try:
                    for k in tm.update(d, r["T"], self.cal):
                        done, total = tm.counts(); sp = tm.spots[k]
                        self._log("side %d: %s spot %d done (%d of %d)" % (no, sp["kind"], k + 1, done, total))
                except Exception as e: print("touch coverage:", e, flush=True)
            try:
                sd.table.add(d, self.cal, M @ F)
                if sd.idx % 10 == 0:
                    for e in self.bed.every10(sd): self._log("side %d: %s" % (no, e))
                    self.table = self._platter(sd)                # cached between new frames
                    if self.auto_gain: self._auto_gain(sd)
                    else: self.autogain.reset()
                self.table_pose = M @ F; self._cut_poses.append(self.table_pose)
            except Exception as e: self.table = None; print("table:", e, flush=True)
        else:
            if self.capture_mode!='live':sd.lost+=1
            self.table_pose=None
        moved = None
        try:
            lines, moved = self.bed.after_step(sd, d, irl, r, M @ F) or ((), None)
            for e in lines: self._log("side %d: %s" % (no, e))
        except Exception as e: print("bed:", e, flush=True)
        if keep:
            from app2.scan.range_guide import distance_zones,measured_profile
            color_stamp['capture_bands']=distance_zones(self._live_fields[sd].profile if self.capture_mode=='live' else getattr(self,'range_profile',None) or measured_profile(getattr(self,'scan_mode','near')))
        if keep: sd.stamps.append(dict(frame=name, t=t, ok=r["ok"], fit=r["fit"], rmse=r["rmse"], reloc=r["reloc"],
                                       near=lo, far=hi,registration_points=r.get("registration_points"),refusal_reason=r.get("refusal_reason"), led=self.led,led_status=getattr(self.source.device,"led_state","unknown"),led_requested=getattr(self,"led_requested",self.led), gain=g_now, hinted=bool(r.get("hinted")),scan_mode=getattr(sd,"scan_profile",{}).get("mode","near"),seed_sources=r.get("seed_sources",[]),**color_stamp))
        if keep and r["ok"]:self._emit_turn(sd, every_frame=True)
        if sd.rgb_thumb is None and self.source.colour() is not None:
            try: sd.rgb_thumb = cv2.resize(decode_colour(self.source.colour()), (320, 240)); sd.version += 1
            except Exception: pass
        sd.idx += 1
        if self.gl:
            T = r["T"].copy(); T[:3, 3] *= 1000.0
            self._emit("pose", dict(side_no=no, T_mm=T, ok=bool(r["ok"]), home_seq=self.home_seq))
        if r["ok"] and sd.idx % 3 == 1:                           # the model so far, every 3rd frame (Tk's cadence)
            if self.gl: self._emit_cloud(sd)
            else: self._render(sd, self.orbit.pose() if self.orbit else r["T"])
        if sd.frames % 10 == 1: self._emit_sides()
        # the part moved on the table (the bed's refused fits agree on a move): this frame is kept, the pass pauses and
        # is saved, so no frame lands in the wrong place (the Tk scanner's pause_why)
        if moved and self.recording: self._pause(sd, moved)
        self._check_shot(bool(r["ok"]))
        if self.capture_mode=="live" and not r["ok"] and self.recording:self._pause(sd,"Tracking lost. Undo recent can take back the last few seconds.")

    def _pause(self, sd, why=None):
        """Stop recording and save the side, as the Pause button does; why goes to the log."""
        self.recording = False;self._sync_scan_light()
        if why: self._log("side %d paused: %s" % (self._no(sd), why))
        self._busy("Saving side %d..." % self._no(sd), lambda: self._save(sd, quick=True))

    def _placed(self, sd, no, placed, ok):
        """What the Continue check found on a pass's first frame(s) (tracking.GapPlacer), in the log once per pass."""
        why = placed.get("why")
        print("side %d pass %d frame %d: after the pause %s (ICP %s, search %s, gap %s, %.2f s)" % (
            no, sd.ps, sd.idx, why, placed.get("icp"), placed.get("search"), placed.get("gap"), placed.get("secs", 0.0)), flush=True)
        touch = self._touch_model(sd) is not None
        if touch:                                                 # Touch up: finding the part on the built model
            if ok: self._log("side %d: found the part on the built model (%s)" % (no, "two searches agree" if why == "search" else why))
            elif self._unsure_said != (sd, sd.ps):
                self._unsure_said = (sd, sd.ps)
                self._log("side %d: looking for the part on the built model; frames count as lost until it is found" % no)
        elif why == "search" and ok: self._log("side %d: the part moved while paused; found again by the global search" % no)
        elif why in ("unsure", "few") and self._unsure_said != (sd, sd.ps):
            self._unsure_said = (sd, sd.ps)
            self._log("side %d: not sure where the part is after the pause, so these frames count as lost. "
                      "Turn it back to where it was when you paused." % no)
        elif ok and self._unsure_said == (sd, sd.ps): self._log("side %d: found again after the pause" % no)
        if ok:                                                    # the check takes about 1 s a frame: frames wait meanwhile
            with self._flock: left = len(self._frames)
            if left > 30: self._behind_said = True; self._log("%d frames came in while side %d was being placed; catching up" % (left, no))

    SMART_DONE, SMART_MIN_VIEW, SMART_KEEP_EVERY, SMART_SETTLE = 0.03, 2000, 10, 0.5

    def _smart_skip(self, sd, r):
        """Smart frames: True when this tracked frame is not worth saving. It looks at the model so far from the frame's pose:
        when under 3% of what it sees is still thin (red or yellow) it adds nothing, so it is dropped, except one in ten (noise
        averaging) and every frame while there is little model yet. Frames right after an Auto gain change are dropped too:
        the exposure is still settling. Lost frames are always kept (their stamps tell the build where tracking failed)."""
        if not r["ok"]: return False
        st = self._smart; st["seen"] += 1
        if time.monotonic() - self._gain_t < self.SMART_SETTLE: st["skipped"] += 1; return True
        share = self._unfinished_share(r["T"])
        if share is None or share >= self.SMART_DONE or st["seen"] % self.SMART_KEEP_EVERY == 0: return False
        st["skipped"] += 1
        if st["skipped"] % 100 == 0: self._log("smart frames: kept %d of %d tracked frames" % (st["seen"] - st["skipped"], st["seen"]))
        return True

    def _unfinished_share(self, T):
        """The share of the model in this frame's view that is still thin (None: too little model to tell). T: camera -> scan, metres.
        The thin/solid levels are refreshed every 30 tracked frames (reading them costs two surface extractions)."""
        st = self._smart; trk = self.host.raw()
        if trk is None or getattr(trk, "model", None) is None: return None
        if st["lv"] is None or st["seen"] - st["at"] >= 30:
            P = trk.model.point.positions.cpu().numpy(); st["P"] = P; st["N"] = trk.model.point.normals.cpu().numpy()
            st["lv"] = trk.coverage_levels(P); st["at"] = st["seen"]
        P, N, lv = st["P"], st["N"], st["lv"]
        if not len(P): return None
        Ti = np.linalg.inv(T); Pc = P @ Ti[:3, :3].T + Ti[:3, 3]; Nc = N @ Ti[:3, :3].T
        fx, fy, cx, cy = self.cal; zc = np.maximum(Pc[:, 2], 1e-6)
        u = np.rint(Pc[:, 0] * fx / zc + cx); v = np.rint(Pc[:, 1] * fy / zc + cy)
        facing = np.abs((Nc * Pc).sum(1)) > 0.5 * np.linalg.norm(Pc, axis=1)
        vis = (Pc[:, 2] > 0.02) & (u >= 0) & (u < 800) & (v >= 0) & (v < 600) & facing
        n = int(vis.sum())
        if n < self.SMART_MIN_VIEW: return None
        return float((lv[vis] < 2).sum()) / n

    def _auto_gain(self, sd):
        # the turn stepper only keeps the dial's turn marks now; the gain itself follows the part's IR brightness each
        # frame (AutoGain.brightness), so nobody waits a turn per try and the steps are one gain at a time
        msg = self.autogain.msg
        self.autogain.tick(sd, sd.table.turned_deg(), self.gain, lambda: 0)
        self.autogain.msg = msg

    def _emit_turn(self, sd, every_frame=False):
        """The turn ring and dial. The per-frame call is held to once a second: the coverage snapshot walks every kept
        frame of the side, so on a long side it would cost more than the scanner's gap between frames."""
        if every_frame:
            now=time.monotonic()
            if now-getattr(self,"_turn_t",0.0)<1.0:return
            self._turn_t=now
        from app2.scan.turn_coverage import snapshot
        coverage=dict(snapshot(sd))
        value=getattr(self,'_bed_live',None) or {}
        coverage.update(live_angle_deg=value.get('angle_deg'),bed_seen=bool(value.get('bed_seen')),start_angle_deg=getattr(sd,'scan_profile',{}).get('start_angle_deg'))
        field=(self._live_fields.get(sd) if getattr(self,'_live_fields',None) else None)
        if field is not None:coverage['quality_done_share']=field.state()['done_share']
        table=getattr(sd,"table",None);auto=getattr(self,"autogain",None);enabled=getattr(self,"auto_gain",False)
        dl = dial(table.turned_deg() if hasattr(table,"turned_deg") else None,auto if enabled else None,sd)
        self._emit("turn", dict(deg=None if dl is None else dl[0], n=None if dl is None else dl[1], gain=getattr(self,"gain",None),
                                msg=auto.msg if enabled and auto is not None else "",coverage=coverage))

    def _platter(self, sd):
        """The platter the build will cut, (n, off, centre, radius, source) in the side's scan frame, or None: the table
        dots' when the bed hook has it, else Auto remove base's. The tint, Auto gain and the save all use this."""
        background = callable(getattr(self.bed, 'request_auto', None))
        if background and not getattr(self, 'recording', False):
            self.bed.request_auto(sd)
        return self.bed.platter(sd) or auto_platter(sd.table, cached=background)

    def _base_plane(self, sd):
        """The side's platter plane (n, off) in its scan frame for the model view, or None: the one cached while it
        records, else the bed's or Auto remove base's."""
        pl = self.bed.platter(sd)                             # the bed found from its dots wins, recording or not: the quick
        if pl is not None: return pl[0], pl[1]                # table fit can take a big flat face of the part for the table
        if sd is self.picked and self.recording and self.table is not None: return self.table[0], self.table[1]
        table = getattr(sd, "table", None)
        if table is None: return None
        if callable(getattr(self.bed, 'request_auto', None)):
            if not self.recording: self.bed.request_auto(sd)
            p = table.cached_plane()
        else:
            p = getattr(table, 'legacy_plane', table.plane)()
        return None if p is None else (p[0], p[1])

    def _overlay_geo(self, sd):
        """The platter in the camera frame for the violet tint, raised by Bed height like the cut itself."""
        return self._lifted(self._overlay_geo_raw(sd))

    def _overlay_geo_raw(self, sd):
        """The bed's camera platter (right on lost frames too), else the side's one plane seen from where the camera is now."""
        if not self.recording or sd is None: return None
        cam = self.bed.camera_geo(sd, self.recording)
        if cam is not None:
            rim=(getattr(self,'_bed_live',None) or {}).get('rim')
            if rim and np.isscalar(cam[3]):return np.asarray(rim['normal']),rim['offset_mm'],np.asarray(rim['center']),rim['radius_mm']
            return cam
        tb = self.table
        if tb is None or self.table_pose is None: return None
        Tm = self.table_pose; nc, oc = to_camera(tb[0], tb[1], Tm)
        return nc, oc, None if tb[2] is None else Tm[:3, :3].T @ (tb[2] - Tm[:3, 3]), _turn(tb[3], Tm[:3, :3].T)

    # ------------------------------------------------------------------ the model view
    def _view_bed_up(self, sd):
        """Matched bed's part-facing normal in this side's displayed scan frame."""
        if sd is None: return None
        plane = self._base_plane(sd)                              # the dot-found bed, else the table fit or Auto remove base: a side
        if plane is None:                                         # still on base: Auto keeps its bottom down too
            n = self._live_bed_up(sd)                             # the first frames of a side: the bed seen while framing
            if n is None: return None
        else: n = np.asarray(plane[0], dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all(): return None
        length = float(np.linalg.norm(n))
        return None if length < 1e-8 else tuple(float(v) for v in n / length)

    def _view_bed_front(self, sd):
        """The direction the scanner looked at the start of the side, in the side's scan frame: that frame is the first
        camera's, so it is the camera's +z. The view's Front face is set from it (and Bottom from the bed) until the
        user picks faces from the cube's menu."""
        return None if sd is None else (0.0, 0.0, 1.0)

    def _live_bed_up(self, sd):
        """Up for a side that has no plane yet, from the bed the framing watcher last saw (camera frame) carried into
        the side's scan frame by the current pose. The camera looks down on the bed, so up is the sign that points
        back at the camera. Before this the first frames of a side showed the part upside down, in the camera's frame,
        until the table fit arrived."""
        geo = (getattr(self, '_bed_live', None) or {}).get('geo')
        if not geo or sd is not self.picked: return None
        n = np.asarray(geo[0], dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8: return None
        if n[2] > 0: n = -n                                       # toward the camera (camera z points away from it)
        r = getattr(self, 'last', None)
        if r is not None and r.get('T') is not None: n = np.asarray(r['T'], float)[:3, :3] @ n
        return n

    def _emit_cloud(self, sd, empty=False):
        """GL mode: the model so far as a point cloud (mm) for the window's 3D view; empty arrays clear it."""
        from app2.scan import timing; timing.mark('cloud_start')
        try: self._emit_cloud_body(sd, empty)
        finally: timing.mark('cloud_end')

    def _emit_cloud_body(self, sd, empty=False):
        if not self.gl: return
        tm = None if empty else self._touch_model(sd)
        if tm is not None and self.host.has(sd): self._emit_touch(sd, tm); return
        if empty or not self.host.has(sd) or not self.host.fused:
            z3 = np.zeros((0, 3), np.float32)
            self._emit("cloud", dict(side_no=None if sd is None else self._no(sd), P=z3, N=z3, L=np.zeros(0, np.uint8),
                                     covered=None, pivot=np.zeros(3, np.float32), seq=self._next_seq(), home_seq=self.home_seq, bed_up=None))
            return
        try: P, N, L, cov = self.host.cloud()
        except Exception as e: print("cloud:", e, flush=True); return
        if self.remove_base and len(P):                           # the platter and anything under it: the build cuts them
            pl = self._base_plane(sd)
            if pl is not None:
                keep = P @ np.asarray(pl[0], np.float32) + np.float32(pl[1]) > LOW_MM
                if not keep.all():
                    P, N, L = P[keep], N[keep], L[keep]; cov = float((L == 2).mean()) if len(L) else 0.0
        pivot = P[::max(1, len(P) // 50000)].mean(0).astype(np.float32) if len(P) else np.zeros(3, np.float32)
        B=np.full(len(P),255,np.uint8);shown=self._side_clouds.get(sd)                  # what the view showed before this refresh
        self._side_clouds[sd] = dict(P=P, N=N, L=L, B=B, covered=cov, pivot=pivot)      # the side as last shown
        payload=dict(side_no=self._no(sd), P=P, N=N, L=L, B=B, covered=cov, pivot=pivot, seq=self._next_seq(),home_seq=self.home_seq,bed_up=self._view_bed_up(sd),bed_front=self._view_bed_front(sd))
        field=(self._live_fields.get(sd) if getattr(self,'_live_fields',None) else None)
        if field is not None and self.capture_mode=='live':
            payload['B'],payload['L']=field.levels(P);payload['covered']=field.state()['done_share'];payload['live_build']=field.state();self._side_clouds[sd]=dict(payload)
            self._add_surface(sd,payload)                     # Live build showed dots only: Surface never had a mesh to draw (2026-10-22)
            self._emit('cloud',payload)
        else:
            payload['B']=self._carry_distance(sd,P,B,shown);self._side_clouds[sd]=dict(self._side_clouds[sd],B=payload['B'])
            self._add_surface(sd,payload)
            self._publish_distance(sd,payload);self._emit('cloud',payload)

    def _add_surface(self,sd,payload):
        """The lit surface for the live view (a real shape instead of loose dots), with each vertex taking the coverage
        and distance of the dots in its cell. Same base cut as the dots."""
        try:surface=self.host.surface() if hasattr(self.host,'surface') else None
        except Exception as error:print('surface:',error,flush=True);surface=None
        if surface is None or not len(payload['P']):
            if getattr(getattr(self.host,'trk',None),'vbg',None) is not None and len(payload['P']):payload['surface_soon']=True
            return
        V,N,Fc=surface
        if self.remove_base:
            pl=self._base_plane(sd)
            if pl is not None:
                keep=V@np.asarray(pl[0],np.float32)+np.float32(pl[1])>LOW_MM
                if not keep.all():
                    tri=keep[Fc].all(1);Fc=Fc[tri];used=np.zeros(len(V),bool);used[Fc.ravel()]=True
                    remap=np.cumsum(used)-1;V,N,Fc=V[used],N[used],remap[Fc].astype(np.int32)
        if not len(Fc):payload['surface_soon']=True;return
        payload.update(V=V,VN=N,F=Fc,VL=cell_lookup(payload['P'],payload['L'],V,0),VB=cell_lookup(payload['P'],payload['B'],V,255))

    def _carry_distance(self,sd,P,B,shown=None):
        """Distance colors for a new model cloud from the last measured one, by 2 mm grid cell, so the view shows the
        growing model at once. Without this the view held the old colored cloud until the background measure caught
        up, which while recording was almost never: the model looked frozen."""
        last=(getattr(self,'_side_colored',None) or {}).get(sd)
        if last is not None and len(P) and len(last['P']):B=cell_lookup(last['P'],np.asarray(last['B']),P,255)
        # best wins against what the view already showed:
        if shown is not None and len(P) and len(shown.get('P',())):      # the live band and the measure never argue a point down
            prev=cell_lookup(shown['P'],np.asarray(shown['B']),P,255);known=(prev!=255)&(B!=255)
            B=B.copy();B[known]=np.maximum(B[known],prev[known]);B[(B==255)&(prev!=255)]=prev[(B==255)&(prev!=255)]
        r=getattr(self,'last',None)
        if self.recording and r is not None and r.get('ok') and len(P):   # points newer than the last measure: their band now
            from app2.scan.range_guide import measured_profile
            M=r['T']@F;M=M.copy();M[:3,3]*=1000.0
            profile=getattr(self,'range_profile',None) or measured_profile(getattr(sd,'scan_profile',{}).get('mode','near'))
            B=depth_band_fill(P,B,M@F,profile,getattr(self,'_scan_depth',None),self.cal)
        return B

    def _touch_model(self, sd):
        """The picked touch-up side's model (touchup.TouchModel) while it is loaded, else None."""
        return getattr(sd, "touch_model", None) if sd is not None else None

    def _emit_touch(self, sd, tm):
        """GL mode, a touch-up side: the built model colored by what still needs scanning (the same point arrays every
        time, so the view only recolors), where the camera looks now, and the spots done so far."""
        done,total=tm.counts()
        left=tm.remaining_counts() if hasattr(tm,"remaining_counts") else {}
        L=tm.levels();B=getattr(tm,'capture_distance',None)
        if B is None:B=np.full(len(tm.P),255,np.uint8)
        payload=dict(side_no=self._no(sd),P=tm.P,N=tm.N,L=L,B=B,covered=None,pivot=tm.pivot,seq=self._next_seq(),home_seq=self.home_seq,bed_up=self._view_bed_up(sd),touch=dict(done=done,total=total,**left),scan_guide=getattr(tm,'scan_guide',None))
        mesh=getattr(tm,'mesh',None);vi=getattr(tm,'v_index',None)
        if mesh is not None and vi is not None:                   # Surface style: the built model, colored like the dots
            payload.update(V=mesh[0],VN=mesh[1],F=mesh[2],VL=np.asarray(L)[vi],VB=np.asarray(B)[vi])
        self._publish_distance(sd,payload);self._emit('cloud',payload)


    def _bind_touch_guide(self, sd, tm):
        def changed():
            if not self._stop: self._post(self._touch_guide_ready, sd, tm)
        tm.on_guide_changed = changed

    def _touch_guide_ready(self, sd, tm):
        if not self._stop and self.picked is sd and self._touch_model(sd) is tm:
            self._emit_touch(sd, tm)

    def _leave_touch_up(self):
        if self.recording or self.busy or self.building:return
        if self._drop_empty_touch_side(self.picked):
            self._emit_cloud(None,empty=True);self._emit_image(None,"live");self._emit_sides();self._emit_state()

    def _drop_empty_touch_side(self,sd):
        if sd is None or sd not in getattr(self.project,"made_sides",[]):return False
        if getattr(sd,"touch_info",None) is None or sd.frames or sd.entries or glob.glob(os.path.join(sd.cache,"*.dph")):return False
        if self.host.has(sd):self.host.release()
        self.project.delete_side(sd)
        if self.picked is sd:self.picked=None
        return True

    def _touch_up(self):
        """Touch up: a new side tracked in the built model's frame from its first frame (touchup.py), so the build
        places it as it is. The model shows coloured by its quality map: red holes, purple rough spots, turning green as
        the new frames cover them."""
        if self.recording or self.busy or self.building or self.bed.learning: return
        empty=next((s for s in self.project.sides if getattr(s,"touch_info",None) is not None and not s.frames and not s.entries and not glob.glob(os.path.join(s.cache,"*.dph"))),None)
        if empty is not None:
            if self.picked is not empty:
                if self.picked is not None:self._save(self.picked)
                self.host.release();self.picked=empty
            if not self.host.has(empty):
                def reload_empty():
                    make=self._load_touch(empty)
                    if make is None:raise ValueError("Touch up needs the current built model. Build again first.")
                    self.host.new(empty,factory=make)
                self._busy("Preparing touch up...",reload_empty)
            if self._touch_model(empty) is not None:self._emit_touch(empty,empty.touch_model)
            self._emit_sides();self._emit_state();return
        mp = os.path.join(self.project.path,"model.ply")
        if not os.path.exists(mp): self._log("Touch up needs a built model: Build combined model first."); return
        from app2.scan import touchup
        import model_frames
        frames = model_frames.read(self.project.path)
        if not frames: self._log("Touch up: can't tell where the sides sit in the built model; build it again first."); return
        self._forget_rewound()
        if self.picked is not None: self._save(self.picked)
        self.host.release()
        self._holding = True
        try:
            def prep():
                tm = touchup.TouchModel.load(mp, self._log)
                try:
                    from app2.builder.scan_guide import guide
                    from app2.scan.guide_mode import selected_mode
                    tm.scan_guide = guide(self.project.path, mp, mode=selected_mode(self.project.path), distance=(self.near,self.far), frames=frames, live=True)
                except Exception as e: self._log("Scan guide: %s" % e)
                cands = touchup.start_poses(self.project.path, frames); platters = touchup.platter_priors(self.project.path, frames)
                sd = self.project.make_side(); model_frames.write_touch(sd.folder, frames, mp)
                sd.touch_model = tm; self._bind_touch_guide(sd, tm); sd.touch_info = model_frames.touch_info(sd.folder)
                self.bed.side_opened(sd, False)
                self.picked = sd; self.last = None; self._home();self._emit_turn(sd)
                make = (self.cfg.touch_factory or touchup.model_tracker_factory)(tm, cands, platters)
                self.host.new(sd, factory=make)
                done, total = tm.counts()
                self._log("side %d: touch up on the built model, %d spot%s to go over (red = hole, purple = rough). Put the "
                          "part on the table any way up, press Start and aim at the spots until they turn green; holding "
                          "still on a spot averages more frames into it." % (self._no(sd), total, "" if total == 1 else "s"))
            self._busy("Preparing touch up...", prep)
        finally: self._holding = False
        sd = self.picked
        if sd is not None and self._touch_model(sd) is not None:
            self._emit_image(None, "live"); self._emit_touch(sd, sd.touch_model)
            T = self._touch_home(sd)
            if T is not None: self._emit("pose", dict(side_no=self._no(sd), T_mm=T, ok=True, home_seq=self.home_seq))
        self._emit_sides(); self._emit_state()

    def _touch_home(self, sd):
        """Home for a touch-up side before its first frame: where the tracker starts (mm)."""
        try: T = self.host.T.copy(); T[:3, 3] *= 1000.0; return T
        except Exception: return None

    def _load_touch(self, sd):
        """Continue on a saved touch-up side: its tracker holds the built model again, when model.ply is the one the side
        was scanned on (else it carries on as a plain side: its frames stay in that model's frame either way)."""
        from app2.scan import touchup
        import model_frames
        info = getattr(sd, "touch_info", None) or model_frames.touch_info(sd.folder)
        mp = os.path.join(self.project.path, "model.ply"); same = False
        try:
            st = os.stat(mp); m = (info or {}).get("model") or {}
            same = abs(float(m.get("mtime", -1)) - st.st_mtime) < 1e-3 and int(m.get("size", -1)) == st.st_size
        except OSError: pass
        if not same: return None
        frames = {k: v for k, v in info["ref"].items()}
        tm = touchup.TouchModel.load(mp, self._log)
        try:
            from app2.builder.scan_guide import guide
            from app2.scan.guide_mode import selected_mode
            tm.scan_guide = guide(self.project.path, mp, mode=selected_mode(self.project.path), distance=(self.near,self.far), frames=frames, live=True)
        except Exception as e: self._log("Scan guide: %s" % e)
        make = (self.cfg.touch_factory or touchup.model_tracker_factory)(tm, touchup.start_poses(self.project.path, frames),
                                                                        touchup.platter_priors(self.project.path, frames))
        sd.touch_model = tm
        self._bind_touch_guide(sd, tm)
        return make

    def _keep_cloud(self, sd):
        """A side saved: its cloud as last shown is kept (thinned to SIDE_CLOUD_KEEP points) for picking it again."""
        c = self._side_clouds.get(sd)
        if c is None or len(c["P"]) <= SIDE_CLOUD_KEEP: return
        k = int(np.ceil(len(c["P"]) / SIDE_CLOUD_KEEP))
        self._side_clouds[sd] = dict(c, P=c["P"][::k].copy(), N=c["N"][::k].copy(), L=c["L"][::k].copy(), B=c["B"][::k].copy() if c.get("B") is not None else None)

    def _side_pose(self, sd):
        """The scanner's last pose on a side (camera -> world, mm): Home when the side is shown again."""
        if not sd.entries: return None
        return np.asarray(sd.entries[-1][2] @ F, np.float64)

    def _show_saved(self, sd):
        """A saved side picked (its model is not on the tracker): its model in the GL view. Its cloud from this
        session when there is one (right away), else its own mesh from the last build (read on a thread). Returns
        False when neither is there: the picture then stays alone."""
        if not self.gl or sd is None or not (sd.entries or sd.frames): return False     # a side from an earlier session has no entries
        T = self._side_pose(sd); c = self._side_clouds.get(sd)
        if c is not None and len(c["P"]):
            self._emit("cloud", dict(c, side_no=self._no(sd), seq=self._next_seq(), home_seq=self.home_seq, saved=True, bed_up=self._view_bed_up(sd)))
            self._emit("pose", dict(side_no=self._no(sd), T_mm=T, ok=True, home_seq=self.home_seq))
            return True
        path = self.project.side_mesh(sd)
        if not path:
            self._flush_disk();live=os.path.join(sd.folder,'live-model.ply');path=live if os.path.exists(live) else None
        if not path: return False
        no = self._no(sd)

        def read():
            try:
                import open3d as o3d
                m = o3d.io.read_triangle_mesh(path)
                if not len(m.triangles): return
                if not m.has_vertex_normals(): m.compute_vertex_normals()
                V = np.asarray(m.vertices, np.float32); N = np.asarray(m.vertex_normals, np.float32)
                Fc = np.asarray(m.triangles, np.int32)
            except Exception as e: print("side mesh:", e, flush=True); return
            from app2.scan.capture_distance import project_levels
            try:B=project_levels(self.project.path,V,sd.folder)
            except (OSError,ValueError):B=None
            pivot = V[::max(1, len(V) // 50000)].mean(0).astype(np.float32)
            self._post(self._side_mesh_ready, sd, dict(side_no=no, V=V, N=N, F=Fc, B=B, T_mm=T, pivot=pivot, path=path))
        threading.Thread(target=read, name="side-mesh", daemon=True).start()
        return True

    def _side_mesh_ready(self, sd, m):
        """The mesh read: shown only if that side is still picked and not loaded on the tracker meanwhile."""
        if self.picked is not sd or self.recording or (self.host is not None and self.host.has(sd) and self.host.fused): return
        self._emit("side_mesh", dict(m, seq=self._next_seq(), home_seq=self.home_seq, bed_up=self._view_bed_up(sd)))

    def _emit_image(self, img, kind, covered=None, your_view=False, needs=False):
        """The model panel's picture: kind "live" (image mode's render) or "thumb" (a saved side). img None clears it.
        (The built model is the Build step's: it is never shown on the Scan step.)"""
        self._emit("model_image", dict(img=img, kind=kind, covered=covered, your_view=your_view, needs=needs, seq=self._next_seq()))

    def _render(self, sd, T):
        """Image mode: the model picture from pose T (tracker metres) at scale 0.8, as the Tk scanner drew it."""
        if not self.host.has(sd): return
        try: img, cov = self.host.render(T, scale=0.8, coverage=True)
        except Exception as e: print("render:", e, flush=True); return
        self._emit_image(img, "live", cov, self.orbit is not None)

    def _render_pending(self):
        if self.gl: return
        sd = self.picked
        if sd is None or not self.host.has(sd): self.orbit_dirty = self.home_dirty = False; return
        if self.orbit_dirty and self.orbit is not None:
            self.orbit_dirty = False; self._render(sd, self.orbit.pose())
        if self.home_dirty:
            self.home_dirty = False
            if self.host.fused: self._render(sd, self.host.T)

    def _show_home(self, sd):
        """GL mode, after a side's model was (re)built: its cloud and the scanner's last pose as Home."""
        if not self.gl or sd is None or not self.host.has(sd) or not self.host.fused: return
        T = self.host.T.copy(); T[:3, 3] *= 1000.0
        self._emit_cloud(sd); self._emit("pose", dict(side_no=self._no(sd), T_mm=T, ok=True, home_seq=self.home_seq))

    # ------------------------------------------------------------------ sides
    def _warm_up(self):
        """Once, on its own thread while the live pictures run: start the GPU and the distance worker, so pressing Start
        goes straight to building instead of waiting seconds for both (seen live: the color view sat there after Start).
        On the frame thread it froze the pictures for its 15 s."""
        start=time.monotonic()
        try:
            if self.gl and self.capture_mode!='live' and not hasattr(self,'_distance_pool'):self._prepare_distance()
        except Exception as error:print('warm distance:',error,flush=True)
        try:
            from app2.scan.tracking import cuda_device
            import open3d.core as o3c
            o3c.Tensor.zeros((1,),o3c.float32,cuda_device())   # the CUDA context: the slow part of a first side
            if self.cal is not None:                           # and one throwaway track of a flat wall: the GPU loads its code now,
                from app2.scan.tracking import cuda_tracker   # not on the first frame after Start (6 s, which paused the scan)
                trk=cuda_tracker(self.cal,self.cfg.w,self.cfg.h);v,u=np.mgrid[0:self.cfg.h,0:self.cfg.w]
                bowl=(2500+0.004*((u-self.cfg.w/2)**2+(v-self.cfg.h/2)**2)).astype(np.uint16)   # a curved fake surface: a flat one fuses nothing
                for _ in range(3):trk.step(bowl)
                from app2.scan.tracking import cuda_cache_release
                del trk;cuda_cache_release()
        except Exception as error:print('warm gpu:',error,flush=True)
        self._log('Ready to scan (GPU and helpers started in %.1f s).' % (time.monotonic()-start))

    def _busy(self, msg, fn):
        self.busy = msg; self._emit_state(force=True)
        try: fn()
        finally: self.busy = None; self._emit_sides(); self._emit_state(force=True)

    def _defer_disk(self, fn):
        """Disk work that nothing waits for (the voxel cache, the live model): one background writer, in order.
        _flush_disk waits for it before anything reads those files or the session closes."""
        if getattr(self, "_disk_thread", None) is None or not self._disk_thread.is_alive():
            self._disk_q = queue.Queue(); self._disk_thread = threading.Thread(target=self._disk_loop, name="side-writer", daemon=True); self._disk_thread.start()
        self._disk_q.put(fn)

    def _disk_loop(self):
        while True:
            fn = self._disk_q.get()
            if fn is None: return
            try: fn()
            except Exception as e: traceback.print_exc(); self._log("Could not finish saving a side's live model: %s" % e)
            finally: self._disk_q.task_done()

    def _flush_disk(self):
        q = getattr(self, "_disk_q", None)
        if q is not None: q.join()

    def _save(self, sd, quick=False):
        """Save a side (and its model picture while its tracker is here). A side with no tracked frame is removed.
        quick (Pause): the big files are written in the background so Continue is there at once."""
        t0 = time.monotonic()
        if getattr(self,"_color_writer",None) is not None:self._color_writer.flush()
        self._settle_bed()
        thumb = None
        if self.host.has(sd) and sd.entries and sd.dirty:
            T = sd.entries[-1][2] @ F; T = T.copy(); T[:3, 3] /= 1000.0
            try: thumb = self.host.render(T, scale=0.5, weight_threshold=2.0)
            except Exception as e: print("render:", e, flush=True)
        field=(self._live_fields.get(sd) if getattr(self,'_live_fields',None) else None)
        if field is not None:
            sd.scan_profile['live_build']['latest']=field.state();field.save(sd.folder)
        self._save_live_surface(sd, later=self._defer_disk if quick else None)
        if self.project.save_side(sd, thumb, self.bed, self._log) and quick:
            print("side %s paused: saved in %.2f s, live model writing in the background" % (sd.name, time.monotonic() - t0), flush=True)
        self._keep_cloud(sd)
        if sd in self.project.sides:self._emit_turn(sd)
        if sd not in self.project.sides:                          # never tracked a frame: gone, and no longer picked
            self._side_clouds.pop(sd, None)
            if self.host.has(sd): self.host.release()
            if self.picked is sd: self.picked = None; self._emit_cloud(None, empty=True); self._emit_image(None, "live")

    def _settle_bed(self, final=False):
        """Before a save: the bed hook's late results applied first, so what is saved is what the window showed.
        final (closing): the table finder stops and what it still had is applied."""
        lines, changed = self.bed.settle(self.project.sides, self.picked, self._no, final=final)
        for line in lines: self._log(line)
        if changed and self.picked is not None: self.table = self._platter(self.picked)

    def _read_depth(self, sd, ps, idx): return fmt.read_depth(sd.cache, fmt.frame_name(ps, idx), self.cfg.w, self.cfg.h)

    def _load(self, sd):
        no = self._no(sd)
        def prog(k, n): self.busy = "Loading side %d... %d of %d frames" % (no, k, n); self._emit_state()
        make = None
        if getattr(sd, "touch_info", None) is not None:
            try: make = self._load_touch(sd)
            except Exception as e: self._log("side %d: the built model did not load (%s); carrying on without it" % (no, e))
        try:
            if not self._restore_live_surface(sd):self.host.reseed(sd, self._read_depth, prog, factory=make)
            if make is not None:
                tm = sd.touch_model
                for ps, idx, M in sd.entries:                     # what the side's frames already covered
                    raw = self._read_depth(sd, ps, idx)
                    if raw is not None: T = M @ F; T = T.copy(); T[:3, 3] /= 1000.0; tm.update(raw, T, self.cal)
                tm.view[:] = False
        except Exception:
            self.host.release(); raise                            # never carry on from a half-loaded model
        self.bed.side_loaded(sd)
        self._home(); self._show_home(sd)

    def _go(self):
        if getattr(self,"_range_sweep",None) is not None:return
        error=self._capture_calibration_error()
        if not self.recording and error:
            self._log(error);self._emit_state(force=True);return
        if self.busy or self.building or self.bed.learning: return
        self.build_msg = None
        if self.recording:
            sd = self.picked
            self.recording = False; self._busy("Saving side %d..." % self._no(sd), lambda: self._save(sd, quick=True)); return
        self._forget_rewound()                                    # Continue: the frames taken back are gone for good
        warm=getattr(self,'_warm_thread',None)
        if warm is not None and warm.is_alive():self._busy('Starting the GPU...',lambda:warm.join())   # never two distance workers
        self._holding = True                                      # frames that come in while the side is set up are kept
        try:
            if self.picked is None: self._new_side()
            sd = self.picked
            if sd is None: return
            error=self._capture_calibration_error()
            if error:self._log(error);self._emit_state(force=True);return
            self._prepare_live_side(sd)
            if not self.host.has(sd):
                self._busy(("Loading side %d..." if sd.entries else "Preparing side %d...") % self._no(sd),
                           lambda: self._load(sd) if sd.entries else self.host.new(sd))
            if self.gl and self.capture_mode!='live' and not hasattr(self,'_distance_pool'):
                self._busy('Preparing side %d...' % self._no(sd),self._prepare_distance)
            self.bed.pass_started(sd)
            sd.ps += 1; sd.idx = 0; self.recording = True; self.last = None; self.slow = False; sd.dirty = True; sd.pass_ok = 0; self._smart = dict(seen=0, skipped=0, lv=None, at=0)
            self._start_scan_light()
            if self.save_color and self.source.has_led and not self.led and not getattr(self,'light_during_scans',False) and not getattr(self,'_manual_led',False):
                self._led_auto=True;self._set_led(True);self._log("Asked the white light to turn on for color. Check that it is on.")
            self.shot_got = 0; self.shot_need = self._shot_count(sd) if self.capture_mode in ("single", "burst") else 1
            self.shot_deadline = time.monotonic() + 8.0 if self.capture_mode in ("single", "burst") else None
        finally: self._holding = False
        with self._flock: held = len(self._frames)
        self._behind_said = held > 30
        if held > 30: self._log("%d frames came in while side %d was being set up; catching up" % (held, self._no(sd)))
        self.table = self._platter(sd) if self.bed.found(sd) else None; self.table_pose = None; self._cut_poses.clear()
        self._emit_sides(); self._emit_state()

    def _new_side(self):
        if self.recording or self.busy or self.building or self.bed.learning: return
        self._forget_rewound()
        self._reset_framing()
        if self.picked is not None and not self._drop_empty_touch_side(self.picked):self._save(self.picked)
        if not self.save_color and getattr(self,'_led_auto',False):   # only a light the app turned on for color
            if self.source.has_led:self.source.device.led(False)
            self.led=False;self.led_requested=False;self._led_auto=False;self.light_timer.stop();self._light_seen=None
        self.autogain.reset();self.source.device.gain(self.gain)
        self._emit("gain",self.gain);self._emit_state()
        self.host.release()                                       # only one model on the GPU at a time
        mode=getattr(self,'scan_mode','near');declared=(getattr(self,'calibration_source',None) or {}).get('declared_mode')
        if (mode=='far' or declared not in (None,mode)) and not self._select_calibration(mode):
            self._log(self.calibration_error);self._emit_state(force=True);return
        self._reset_mode_detection(preserve=True)
        sd = self.project.make_side()
        from app2.scan.scan_mode import profile,write
        sd.scan_profile=profile(getattr(self,'scan_mode','near'),self.cal,getattr(self,'_scan_depth',None),self.near,self.far)
        sd.scan_profile['surface_start']=getattr(self,'surface_start',None)
        sd.motion_calibration=getattr(self.cfg,'motion_calibration',None);self._stamp_scan_profile(sd.scan_profile);write(sd.folder,sd.scan_profile)
        self._prepare_live_side(sd)
        self._live_recent=None
        self.bed.side_opened(sd, False)
        self.picked = sd; self.last = None; self._home();self._emit_turn(sd)
        self._emit_cloud(None, empty=True); self._emit_image(None, "live")
        self._busy("Preparing side %d..." % self._no(sd), lambda: self.host.new(sd))      # the first one starts CUDA: seconds

    def _pick_no(self, no):
        sides = self.project.sides
        self._pick(sides[no - 1] if (no is not None and 1 <= no <= len(sides)) else None)

    def _pick(self, sd):
        if self.recording or self.busy or self.building or self.bed.learning: return
        if sd is self.picked: return
        self._reset_framing()
        self._forget_rewound()
        if self.picked is not None:
            if not self._drop_empty_touch_side(self.picked):self._save(self.picked);self.host.release()
        self.picked = sd if sd in self.project.sides else None
        self.last = None; self._home()
        self._emit_cloud(None, empty=True)
        if self.picked is not None:                          # its picture at once, then its model when there is one
            self.cal=read_calib(os.path.join(self.picked.folder,'param','Pl.bin'))
            self.host.cal=self.cal
            saved_profile=getattr(self.picked,'scan_profile',{}) or {}
            saved_mode=saved_profile.get('mode')
            if saved_mode in ('near','far'):self.scan_mode=saved_mode
            self.calibration_source=saved_profile.get('calibration_source') or dict(kind='saved side',declared_mode='unknown')
            self.calibration_error=('Large part camera files are missing. Open Camera setup and press Rebuild Large part copy.'
                if self.scan_mode=='far' and self.calibration_source.get('declared_mode')!='far' else '')
            if self.scan_mode=='far' and self.calibration_source.get('declared_mode')=='far':
                import hashlib
                actual=hashlib.sha256(open(os.path.join(self.picked.folder,'param','Pl.bin'),'rb').read()).hexdigest()
                if actual!=self.calibration_source.get('calibration_sha256'):
                    self.calibration_error='Large part camera files changed. Open Camera setup and press Rebuild Large part copy.'
            from app2.scan import preview
            preview.set_rgb_calib(os.path.join(self.picked.folder,'param'))
            self.composer.cal=self.cal;self.composer.area=None;self.composer.scan_geometry=None
            from app2.scan.stereo_view import Stereo
            self.composer.stereo=Stereo.load(os.path.join(self.picked.folder,"param"),self.cal)
            if hasattr(self.composer,"footprint"):self.composer.footprint.reset()
            self._sync_scan_geometry()
            self._emit_image(self.picked.model_thumb, "thumb"); self._show_saved(self.picked);self._emit_turn(self.picked)
        else:
            self._emit("turn",dict(deg=None,coverage=None,msg=""));self._emit_image(None,"live")
            # nothing picked: the live camera in the model panel
        self._emit_sides(); self._emit_state()

    def _delete(self, no):
        sd = next((x for x in self.project.sides if self._no(x) == no), None)
        if sd is None or self.recording or self.busy or self.building or self.bed.learning: return
        self._log("deleted side %d (%s, %d frames)" % (no, sd.name, sd.frames))
        if sd is not self.picked:                                 # another card's side: the live model is not touched
            self.project.delete_side(sd); self._side_clouds.pop(sd, None); self._emit_sides(); self._emit_state(); return
        if self.host.has(sd): self.host.release()
        self.project.delete_side(sd); self.picked = None; self._home(); self._side_clouds.pop(sd, None)
        self._reset_framing();self._emit("turn",dict(deg=None,coverage=None,msg=""))
        self._emit_cloud(None, empty=True); self._emit_image(None, "live"); self._emit_sides(); self._emit_state()

    # ------------------------------------------------------------------ Back / Forward while paused
    def _rewind(self, n):
        if getattr(self,"_color_writer",None) is not None: self._color_writer.flush()
        """Back n frames on the paused picked side (Back 5 / 10 / 30): the newest recorded frames leave its stamps and
        pose table (their files move to discarded/), the side is saved, and the model is made again without them
        (TrackerHost.rewind: the newest checkpoint plus the frames after it). Continue then starts from the last kept
        frame's pose, placed on this model like any Continue (lost until it is sure)."""
        sd = self.picked
        if sd is None or self.recording or self.busy or self.building or self.bed.learning: return
        t0 = time.monotonic(); no = self._no(sd)
        got = self.project.rewind(sd, n)
        if got is None: return
        st, en = got; how = []

        def go():
            try:
                if en: how.append(self._refit(sd))
            finally: self._save(sd)                           # the pose table and track json match the frames left
        self._busy("Going back %d frames on side %d..." % (len(st), no), go)
        self._after_refit(sd, en)
        print("side %d: back %d frames (%d tracked), model %s in %.2f s" % (no, len(st), len(en), how[0] if how else "-",
                                                                            time.monotonic() - t0), flush=True)
        self._log("side %d: took back %d frame%s (%d tracked). Forward brings them back until you Continue." % (
            no, len(st), "" if len(st) == 1 else "s", len(en)))

    def _forward(self):
        if getattr(self,"_color_writer",None) is not None: self._color_writer.flush()
        """Undo the newest Back: its frames come back into the side and onto the model."""
        sd = self.picked
        if sd is None or self.recording or self.busy or self.building or self.bed.learning: return
        no = self._no(sd)
        got = self.project.forward(sd)
        if got is None: return
        st, en = got

        def go():
            try:
                if en and (getattr(sd, "touch_info", None) is not None or not self.host.has(sd)): self._load(sd)
                elif en: self.host.extend(sd, en, self._read_depth)
            finally: self._save(sd)
        self._busy("Bringing back %d frames on side %d..." % (len(st), no), go)
        self._after_refit(sd, en)
        self._log("side %d: brought back %d frame%s" % (no, len(st), "" if len(st) == 1 else "s"))

    def _refit(self, sd):
        """The picked side's model for its entries after a Back: from a checkpoint when its tracker is here, else loaded
        again from its frames (a touch-up side always: its coverage map is made from the frames too)."""
        if getattr(sd, "touch_info", None) is not None or not self.host.has(sd): self._load(sd); return "full"
        return self.host.rewind(sd, self._read_depth)

    def _after_refit(self, sd, en):
        """After Back / Forward: the model view, the platter and the cards show the side as it is now."""
        if self.capture_mode=='live':
            self._live_fields.pop(sd,None);self._live_recent=None;self._prepare_live_side(sd)
        self.last = None; self.table = self._platter(sd) if sd.entries else None
        if en and self.host.has(sd): self._home(); self._show_home(sd)
        self._emit_sides(); self._emit_state(force=True)

    def _forget_rewound(self):
        """Frames taken back become final (Continue, another side, a build, closing): Forward can no longer bring them
        back; they stay in the side's discarded/ folder, in no table, and no build reads them."""
        for sd in getattr(self.project, "sides", None) or ():
            if not getattr(sd, "rewound", None): continue
            n = self.project.forget_rewound(sd)
            if n: self._log("side %d: %d frame%s taken back for good (kept in its discarded folder)" % (self._no(sd), n, "" if n == 1 else "s"))

    # ------------------------------------------------------------------ settings, view
    def _set(self, who=None, **kw):
        if any(k in kw for k in ('near','far')):self.frame_range_undo=None
        before_gain = self.gain
        for k,v in kw.items(): setattr(self,k,v)
        if "gain" in kw and int(kw["gain"]) != before_gain:            # who moves the gain: Auto gain logs its own steps
            self._log("gain set to %d by %s" % (int(kw["gain"]), who or "a setting"))
        if any(k in kw for k in ("gain","near","far","auto_range")):
            from app2.scan.settings import save
            save(gain=self.gain,near=self.near,far=self.far)
            self._sync_scan_geometry()
            if "auto_range" in kw: save(auto_range=self.auto_range)
            if "gain" in kw: self._initial_auto_gain=False
        self._emit_state()

    def _set_auto_gain(self, on):
        self.auto_gain = on
        from app2.scan.settings import save
        save(auto_gain=bool(on))
        if not on: self._initial_auto_gain=False;self.autogain.reset()
        self._emit_state()

    def _set_remove_base(self, on):
        """Remove base on / off: the model view shows the part alone or with the platter (the build cuts it anyway)."""
        self.remove_base = on; self._emit_state()
        sd = self.picked
        if self.gl and sd is not None and self.host.has(sd) and self.host.fused: self._emit_cloud(sd)

    def _set_led(self, on):
        if not self.source.has_led: return
        self.source.device.led(on); self.led = False
        self.led_requested=bool(on)
        if on and getattr(self,'_led_auto',False): self.light_timer.start(time.monotonic())   # the reminder only for a light the app turned on
        else: self.light_timer.stop()
        self._emit_state()

    def _orbit(self, dx, dy, step):
        if self.gl: return
        sd = self.picked
        if sd is None or not self.host.has(sd) or self.host.fused == 0: return
        if self.orbit is None: self.orbit = Orbit(self.host.T, self.host.model_centre())
        self.orbit.drag(dx, dy); self.orbit.zoom += step; self.orbit_dirty = True

    def _home(self):
        # back to the scanner's own view. While paused nothing else redraws the model, so ask for one redraw here. GL
        # mode: the view goes Home when a cloud or pose with a newer home_seq comes in (New side, pick, delete, reload)
        self.orbit = None; self.orbit_dirty = False; self.home_dirty = True; self.home_seq += 1

    def _view_model(self):
        """The built model in viewer.py. One window at most: while it is open, it comes to the front instead."""
        m = self.project.model_path()
        if not m: return
        p = self._viewer
        if p is not None and getattr(p, "poll", lambda: 0)() is None:
            try: self.cfg.view_raiser(p)
            except Exception as e: print("raise viewer:", e, flush=True)
            return
        self._viewer = self.cfg.view_launcher([sys.executable, os.path.join(REPO, "viewer.py"), m, VIEWER_TITLE])

    def _pick_bed(self, name):
        """Use another bed profile (not while recording, saving, building or learning)."""
        if self.recording or self.busy or self.building or self.bed.learning or not hasattr(self.bed, "pick_profile"): return
        msg = self.bed.pick_profile(name, self.project.sides)
        if msg: self._log(msg)
        from app2.scan.bedhook import FREEHAND
        if name == FREEHAND and self.remove_base: self._set_remove_base(False); self._log("Remove base turned off for Freehand.")
        self._reset_mode_detection()
        if self.picked is not None: self.table = self._platter(self.picked)
        self._emit_sides(); self._emit_state()

    def _learn(self, name=None, want=None):
        if not self.bed.learning and self._capture_calibration_error():
            self._log(self._capture_calibration_error());self._emit_state(force=True);return
        busy = bool(self.recording or self.busy or self.building)
        msg = self.bed.learn_toggle(self.gain, busy, name, want=want) if name else self.bed.learn_toggle(self.gain, busy)
        if msg: self._log(msg)
        if self.bed.learning: self.build_msg = None               # the hint line is Learn bed's until its result
        self._emit_state()

    # ------------------------------------------------------------------ build
    def _build(self, backend):
        if self.recording or self.busy or self.building or self.bed.learning: return
        if not [sd for sd in self.project.sides if sd.frames]: return
        self._forget_rewound()
        for sd in list(self.project.sides): self._save(sd)        # every side: a late bed result can touch any of them
        self.host.release()                                       # free the GPU for the build
        self.building = True; self.build_msg = None; self.runner_backend = backend or self.cfg.build_backend
        self.runner = BuildRunner(self.project.path, backend or self.cfg.build_backend, self._on_build_event,
                                  lambda prog: self._post(self._build_done, prog)).start()
        self._log("build started (%s)" % (backend or self.cfg.build_backend).name)
        self._emit_sides(); self._emit_state()

    def _on_build_event(self, kind, payload):                     # build thread: straight on to the window
        if self._stop: return
        if kind == "log": self._log(payload)
        else: self._emit(kind, payload)

    def _build_done(self, prog):
        self.runner = None; self.building = False
        if prog.done:                                            # the built model shows on the Build step, not here
            done_text = getattr(getattr(self, "runner_backend", None), "done_text", None)
            self.build_msg = done_text(prog) if done_text else "Model built: %s" % prog.model
        else: self.build_msg = prog.failed
        self._log(self.build_msg); self._emit_state()

    def _cancel_build(self):
        if self.runner is not None: self.runner.cancel()

    def _pause_build(self, on):
        if self.runner is not None: (self.runner.pause if on else self.runner.resume)()

    # ------------------------------------------------------------------ stream watchdog
    def _watch_stream(self):
        """No frames for 4 s: pause the side (and save it) and say so. Then retry the stream every 3 s while the device
        is there. A replay that played to the end is "gone": nothing to reconnect to."""
        if self.busy: return
        quiet = time.monotonic() - self.source.last_t
        if quiet < 4.0 and not self.source.dead: return
        there = self.source.present(); first = self.cam is None
        self.cam = "noframes" if there else ("replay_end" if self.source.kind == "replay" else "gone"); self.quiet = quiet
        if first:
            if self.recording and self.picked is not None:
                self.recording = False; self._save(self.picked); self._emit_sides()
            if self.cam == "replay_end": self._log("replay finished: no more frames")
            elif self.cam == "gone": self._log("Scanner disconnected: device %s is missing. Waiting for USB and PC mode." % getattr(self.source, "depth_dev", "unknown"))
            else: self._log("no frames from the scanner for %d s: side paused (USB cable? still in PC mode?)" % quiet)
        if there and time.monotonic() - self.cam_tried > 3.0 and (self.source.dead or quiet > 6.0):
            self.cam_tried = time.monotonic(); self._log("reconnecting to the scanner")
            try: self.source.restart()
            except Exception as e: self._log("reconnect failed: %s" % e)
        self._emit_state(); self._emit_status()

    # ------------------------------------------------------------------ to the window
    def _state(self):
        sd = self.picked; sides = self.project.sides
        return ScanState(**self._guide_state(),**self._live_state(),calibration_missing=bool(getattr(self,'calibration_error','')),framing=getattr(self,'framing_view',None),scene_setup=getattr(self,'scene_setup',None),range_undo=getattr(self,'frame_range_undo',None) is not None,led_requested=getattr(self,"led_requested",self.led),led_status=getattr(self.source.device,"led_state","unknown"),stream_warning=getattr(self,"stream_warning",""),range_guide=getattr(self,"range_guide",None),lost_display=time.monotonic()<getattr(self,"lost_display_until",0),mode_notice=getattr(self,'mode_notice',''),mode_evidence=getattr(self,'mode_evidence',{}),calibration_error=self._capture_calibration_error(),calibration_source=getattr(self,'calibration_source',None),far_calibration_path=getattr(self.cfg,'far_param',None),scan_mode=getattr(self,"scan_mode","near"),range_limits=(0,int(LIVE_MAX_DEPTH_MM)),light_remaining=None if (self.recording or self.bed.learning) else self.light_timer.remaining(time.monotonic()), capture_mode=self.capture_mode, recording=self.recording, busy=self.busy, building=self.building, learning=self.bed.learning,
                         learn_status=self.bed.learn_status, learn_map=self.bed.learning_map() if hasattr(self.bed, "learning_map") else None, bed_available=self.bed.available, bed_size=self.bed.size,
                         bed_provisional=bool(getattr(self.bed, "provisional", False)),
                         bed=self.bed.summary() if hasattr(self.bed, "summary") else None,
                         bed_profiles=tuple(self.bed.profiles()) if hasattr(self.bed, "profiles") else (),
                         bed_profile=getattr(self.bed, "profile_name", None),
                         picked=self._no(sd) if sd in sides else None, picked_has_frames=bool(sd is not None and sd.frames),
                         n_sides=len(sides), any_frames=any(s.frames for s in sides),
                         model_exists=os.path.exists(os.path.join(self.project.path, "model.ply")), build_msg=self.build_msg,
                         cam=self.cam, slow=self.slow, near=self.near, far=self.far, gain=self.gain,
                         auto_range=self.auto_range, auto_gain=self.auto_gain, led=self.led, has_led=self.source.has_led,
                         project=self.project.path, source_kind=self.source.kind, source_label=self.source.label,
                         cal=tuple(self.cal), model_view=self.cfg.model_view,
                         build_engine=getattr(self.cfg.build_backend, "label", ""), below_bed=self.below_bed, outside_below_mm=self.outside_below_mm, bed_lift_mm=self.bed_lift_mm,
                         remove_base=self.remove_base,save_color=self.save_color,
                         picked_touch=bool(sd is not None and getattr(sd, "touch_info", None) is not None),
                         rewind_room=sd.rewind_room if sd in sides else 0, rewound=sd.rewound_frames if sd in sides else 0,
                         forward_n=len(sd.rewound[-1][0]) if (sd in sides and sd.rewound) else 0)

    def _emit_state(self, force=False):
        self._sync_scan_light()
        s = self._state(); key = s.key()
        if force or key != self._shown:
            self._shown = key; self._emit("state", s); self._emit_status(s)

    def _emit_status(self, s=None):
        s = s or self._state(); sd = self.picked
        with self._flock: a = list(self._arrivals)
        now = time.monotonic(); a = [x for x in a if now - x < 5.0]
        fps = (len(a) - 1) / max(1e-3, a[-1] - a[0]) if (len(a) > 1 and now - a[-1] < 2.0) else 0.0   # quiet: 0, not the last rate
        stats = (sd.frames, len(sd.entries), sd.lost) if sd is not None else None
        if self.recording and self.last and not self.last["ok"]:self.lost_display_until=now+Composer.COVER_S;s.lost_display=True
        self._emit("status", status_view(s, self.last, stats, fps, self.bed.base_text(sd, self.recording) if sd else "",
                                         self.autogain.msg if self.auto_gain else "", self.quiet, self.source.gone_text()))

    def _emit_sides(self):
        cards = []
        for i, sd in enumerate(self.project.sides):
            cards.append(dict(scan_mode=getattr(sd,"scan_profile",{}).get("mode"),no=i + 1, name=sd.name, frames=sd.frames, kept=len(sd.entries), lost=sd.lost, picked=sd is self.picked,
                              scanning=sd is self.picked and self.recording, base=self.bed.card_base(sd) if sd.frames else None,
                              rgb_thumb=sd.rgb_thumb, model_thumb=sd.model_thumb, version=sd.version,
                              touch=getattr(sd, "touch_info", None) is not None))
        self._emit("sides", cards)

    # ------------------------------------------------------------------ closing
    def _close(self, then, crashed=False):
        """Finish: leave the scanner as found (LED off, default gain), stop the streams, save every side, free the GPU,
        drop an empty project. A running build is stopped and waited for (the window asks first): it must not go on
        writing into the project after the app is gone."""
        if getattr(self,'_range_sweep',None) is not None:self._finish_range_test()
        if getattr(self,'_bed_observer',None) is not None:self._bed_observer.close();self._bed_observer=None
        self._distance_closing=True
        if hasattr(self,'_distance_pool'):
            from app2.scan.capture_distance import build_side
            for sd in getattr(self.project,'sides',[]):
                if any(x.get('capture_bands') for x in sd.stamps):self._distance_pool.submit(build_side,sd.folder,[(a,b,M.copy()) for a,b,M in sd.entries],__import__('fuse').read_calib(os.path.join(sd.folder,'param','Pl.bin')),(self.cfg.h,self.cfg.w),getattr(sd,'scan_profile',{}).get('mode','near'),list(sd.stamps))
            self._distance_pool.shutdown(wait=True)
        if getattr(self,'mode_detector',None) is not None:self.mode_detector.close();self.mode_detector=None
        client=getattr(getattr(self,'motion_provider',None),'__self__',None)
        motion_log=getattr(self,'motion_log',None)
        if motion_log is not None:
            if hasattr(client,'set_recorder'):client.set_recorder(None)
            motion_log.close();self.motion_log=None
        self._then = then; self.recording = False
        if self.runner is not None:
            try: self.runner.stop(5.0)
            except Exception: traceback.print_exc()
        try: self.source.stop()
        except Exception as e: print("source:", e, flush=True)
        lines = []
        if self.project is not None:
            try: self._settle_bed(final=True)
            except Exception: traceback.print_exc()
            try: self._forget_rewound()
            except Exception: traceback.print_exc()
            for sd in list(self.project.sides):
                try: self._save(sd)
                except Exception: traceback.print_exc()
            try:                                                  # the bed's last word: a provisional bed from a side
                for ln in self.bed.close() or (): self._log(ln)   # saved just now, then its finder ends
            except Exception: traceback.print_exc()
            try: self._flush_disk()                               # a Pause's background write finishes before the folder is final
            except Exception: traceback.print_exc()
            lines = self.project.finish()
        for ln in lines: print(ln, flush=True)
        if self.host is not None: self.host.release()          # before Python tears down (a live model made Open3D abort at exit)
        if self.composer is not None: self.composer.stop(); self.composer.thread.join(2.0)
        try: self.source.join(2.0)                             # no thread of this session still in numpy / OpenCV at exit
        except Exception: pass
        try: self.bed.close()
        except Exception: traceback.print_exc()
        if self._color_writer is not None:self._color_writer.close();self._color_writer=None
        self._stop = True
        self._emit("closed", dict(then=then, project=self.project.path if self.project else None, lines=lines, crashed=crashed))


def depth_band_fill(P,B,T,profile,depth=None,cal=None,tol_mm=6.0):
    """Distance bands from this frame, by how far each point sits from the camera at pose T (camera to scan, mm).
    Points still unmeasured (255) take their band; with the depth frame and calibration, points the camera sees now
    (projected onto a depth pixel within tol_mm) move up to this frame's band, best wins, as the background measure
    does. So new surface shows its band at once and a part held in Best turns green instead of keeping the band it
    was first seen in."""
    from app2.scan.range_guide import distance_levels
    T=np.asarray(T,np.float64);Q=(np.asarray(P,np.float64)-T[:3,3])@T[:3,:3];z=Q[:,2]
    band=distance_levels(z,profile).astype(B.dtype);out=B.copy();todo=B==255
    out[todo]=band[todo]
    if depth is None or cal is None:return out
    seen=z>1.0
    u=np.full(len(z),-1,np.int64);v=np.full(len(z),-1,np.int64)
    u[seen]=np.rint(Q[seen,0]/z[seen]*cal[0]+cal[2]).astype(np.int64);v[seen]=np.rint(Q[seen,1]/z[seen]*cal[1]+cal[3]).astype(np.int64)
    inside=seen&(u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0])
    dz=np.zeros(len(z));dz[inside]=depth[v[inside],u[inside]]*0.1
    seen=inside&(dz>0)&(np.abs(dz-z)<=tol_mm)&~todo
    out[seen]=np.maximum(out[seen],band[seen]);return out


def cell_lookup(src,values,dst,missing,mm=2.0):
    """values of src points (mm) for dst points that share a cell mm wide with one of them, else missing. A sorted
    key search: tens of milliseconds where a tree search on every refresh slowed the frames down."""
    values=np.asarray(values);out=np.full(len(dst),missing,values.dtype);todo=np.arange(len(dst))
    for shift in (0.0,mm/2):                              # a second grid, half a cell over, catches points just across a wall
        def keys(Q):
            Q=np.floor((np.asarray(Q,np.float64)+shift)/mm).astype(np.int64);return (Q[:,0]*73856093)^(Q[:,1]*19349663)^(Q[:,2]*83492791)
        ks=keys(src);order=np.argsort(ks,kind='stable');ks=ks[order];kd=keys(np.asarray(dst)[todo])
        i=np.clip(np.searchsorted(ks,kd),0,len(ks)-1);hit=ks[i]==kd
        out[todo[hit]]=values[order[i[hit]]];todo=todo[~hit]
        if not len(todo):break
    return out
