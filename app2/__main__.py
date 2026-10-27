# SPDX-License-Identifier: AGPL-3.0-or-later
"""PointYoink 2.0.
usage: python -m app2 [--last | --project <ProjectLive folder>] [--param <scan>/param] [--root <projects dir>]
                      [--near 100] [--far 450] [--build legacy|stages|run] [--no-gl]
                      [--replay <ProjectLive folder> --root <projects dir> [--fps 3] [--replay-sides 1,2] [--replay-settings]]
--replay plays a saved project's frames instead of the scanner (nothing is sent to any device, and the replayed project
is never written to: new projects go to --root, which must be given and must lie outside it). --no-gl draws the model
as a picture rendered by the scan worker instead of in a GL view (also chosen by itself when there is no OpenGL 3.3)."""
import argparse, os, sys


def real(p): return os.path.realpath(os.path.expanduser(p))


def inside(p, folder):
    """Is p the folder or anywhere under it (symlinks resolved)?"""
    p, folder = real(p), real(folder)
    return p == folder or p.startswith(folder.rstrip(os.sep) + os.sep)


def parse_sides(text):
    """--replay-sides "1,2" or folder names -> [1, 2] / ["09272026002103"]; numbers start at 1."""
    out = []
    for x in text.split(","):
        x = x.strip()
        if not x: continue
        if x.isdigit():
            if int(x) < 1: raise ValueError("side numbers start at 1, not %s" % x)
            out.append(int(x))
        else: out.append(x)
    return out


def parse(argv=None):
    from app2.version import APP_TITLE
    ap = argparse.ArgumentParser(prog="app2", description=APP_TITLE)
    ap.add_argument("--version", action="version", version=APP_TITLE)
    ap.add_argument("--param", default=None, help="calibration folder copied into every new side (explicit Small part export; otherwise use its stored calibration)")
    ap.add_argument("--far-param", default=None, help="explicit Large part calibration folder from this scanner")
    ap.add_argument("--root", default=None, help="where new projects go (default: the Scans folder in ~/PointYoink, set in Preferences)")
    ap.add_argument("--near", type=float, default=None, help="mm, closer than this is cut")
    ap.add_argument("--far", type=float, default=None, help="mm, further than this is cut")
    ap.add_argument("--project", default=None, help="continue this ProjectLive folder")
    ap.add_argument("--last", action="store_true", help="continue the newest ProjectLive folder")
    ap.add_argument("--replay", default=None, help="play this project's saved frames instead of the scanner")
    ap.add_argument("--fps", type=float, default=3.0, help="replay speed, frames per second")
    ap.add_argument("--replay-sides", default=None, help="which sides to replay: 1-based numbers or folder names, comma separated")
    ap.add_argument("--replay-settings", action="store_true", help="use the range, gain and LED each frame was recorded with")
    ap.add_argument("--build", choices=("legacy", "stages", "run"), default="run",
                    help="the Scan step's Build button: build_project.py, the staged engine in legacy events, or the staged engine's own run")
    ap.add_argument("--no-gl", action="store_true", help="no GL view: the model is a picture")
    a = ap.parse_args(argv)
    if a.replay:
        a.replay = real(a.replay)
        if a.root is None: ap.error("--replay needs --root (where the new projects go)")
        if inside(a.root, a.replay): ap.error("--root may not be inside the replayed project")
        if a.project and inside(a.project, a.replay): ap.error("--project may not be the replayed project")
        if a.replay_sides:
            from app2.scan.sources import ReplaySource
            try: ReplaySource.pick_sides(a.replay, parse_sides(a.replay_sides))
            except ValueError as e: ap.error("--replay-sides: %s" % e)
    from app2.library_dir import default_dir
    a.root = os.path.abspath(os.path.expanduser(a.root)) if a.root else default_dir()     # --root, else the default set on Import
    return a


def prepare_env():
    """Before anything imports Qt or OpenGL: no input method (it stalled the Tk window), a thread dump on SIGUSR1, the
    Qt platform and PyOpenGL's matching platform. Returns the Qt platform name."""
    os.environ["XMODIFIERS"] = "@im=none"
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1)                # kill -USR1 <pid> prints where every thread is
    plat = os.environ.get("QT_QPA_PLATFORM")
    if not plat:
        plat = "wayland" if (os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY")) else "xcb"
        os.environ["QT_QPA_PLATFORM"] = plat
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl" if plat.startswith("wayland") else "glx")
    return plat


def probe_gl():
    """Can this display give an OpenGL 3.3 core context? (ok, what it said)."""
    from PySide6.QtGui import QOpenGLContext, QOffscreenSurface, QSurfaceFormat
    ctx = QOpenGLContext(); ctx.setFormat(QSurfaceFormat.defaultFormat())
    if not ctx.create(): return False, "no OpenGL context"
    v = ctx.format().version()
    if tuple(v) < (3, 3): return False, "OpenGL %d.%d is older than 3.3" % tuple(v)
    surf = QOffscreenSurface(); surf.setFormat(ctx.format()); surf.create()
    if not ctx.makeCurrent(surf): return False, "could not use the OpenGL context"
    try: renderer = ctx.functions().glGetString(0x1F01)            # GL_RENDERER
    except Exception: renderer = "?"
    ctx.doneCurrent()
    return True, "OpenGL %d.%d, %s" % (v[0], v[1], renderer)


TK_PARAM = os.environ.get("POINTYOINK_PARAM", "")      # an extra calibration folder to fall back on (developer setups)
NO_CALIBRATION = ("The app has no camera files yet. On the MIRACO itself (not PC mode), scan anything with Small part and save it, "
                  "then plug in USB File Transfer and use File, Import scanner projects. That import brings the camera's "
                  "files, and the app keeps a copy. Do the same with Large part once to scan large parts.")


def has_cal(folder): return bool(folder) and os.path.exists(os.path.join(folder, "Pl.bin"))


def side_params(project):
    """The param folders of a project's sides that hold a calibration, newest side first."""
    data = os.path.join(project or "", "data")
    try: names = sorted(os.listdir(data), reverse=True)
    except OSError: return []
    return [p for p in (os.path.join(data, x, "param") for x in names) if has_cal(p)]


def find_param(given, project, root, fallback=None):
    """The calibration folder: --param, else the project's own (every side keeps a copy), else this PC's Tk scanner
    one (fallback, default POINTYOINK_PARAM), else the newest side's under the projects folder. None when there is none."""
    if given: return given
    from app2.scan.settings import load
    registered=load().get('near_param')
    if has_cal(registered):return registered
    fallback = TK_PARAM if fallback is None else fallback
    for p in side_params(project):
        return p
    if has_cal(os.path.expanduser(fallback or "")): return os.path.expanduser(fallback)
    try: others = sorted((os.path.join(root, x) for x in os.listdir(root)), key=lambda p: os.path.getmtime(p), reverse=True)
    except OSError: others = []
    for other in others:
        for p in side_params(other): return p
    return None


def offline_safe_source(**kw):
    """A LiveSource that opens without a scanner: with none attached (or POINTYOINK_NO_DEVICE=1) starting the stream
    fails, and the session carries on with no frames (the watchdog then says "not connected"), so a saved project can
    still be built, edited and exported. Reconnects when the scanner shows up; nothing is sent to it before that."""
    from app2.scan.sources import LiveSource, no_device

    class OfflineSafeSource(LiveSource):
        offline = False

        def present(self): return not no_device() and super().present()

        def gone_text(self):
            from app2.scan.usb_mode import help_text
            return help_text()                     # says File Transfer mode when the scanner is plugged in that way

        def start(self, sink):
            try: super().start(sink)
            except Exception as e:
                self.offline = True; self.running = True; self.dead = True; self.last_t = 0.0
                print("scanner not streaming: %s" % e, flush=True)

        def restart(self): super().restart(); self.offline = False

        def stop(self):
            if self.offline: self.device.close(); self.running = False; return        # nothing was sent: nothing to undo
            super().stop()
    return OfflineSafeSource(**kw)


def session_factory(a, model_view):
    """make(bridge, **over) -> a new ScanSession (not started). over: project / last for New project and Open project."""
    from app2.scan.session import ScanConfig, ScanSession
    from app2.scan.sources import ReplaySource
    from app2.scan.bedhook import load_bed_hook
    from app2.scan.project import Project
    from app2.builder.runner import LegacyBuild, StagesBuild, StagesRun

    def make(bridge, **over):
        project = over.get("project", a.project); last = over.get("last", a.last)
        if not (project or last or a.replay):
            from app2 import general_prefs
            last = bool(general_prefs.get("open_last"))                   # Preferences, General: start on the newest project
        if last and not project: project = Project.newest(a.root); last = False
        param = find_param(a.param, project, a.root)
        if a.replay:
            src = ReplaySource(a.replay, sides=parse_sides(a.replay_sides) if a.replay_sides else None, fps=a.fps)
            if not a.param and src.folders: param = os.path.join(src.folders[0], "param")
            if project and (inside(project, a.replay) or inside(a.replay, project)):
                raise ValueError("that is the project being replayed; it is never written to")
        else: src = offline_safe_source()
        if not has_cal(param): raise ValueError(NO_CALIBRATION if not a.param else "No calibration (Pl.bin) in %s." % a.param)
        os.makedirs(a.root, exist_ok=True)
        cfg = ScanConfig(a.root, param, project=project, last=last, near=a.near, far=a.far, model_view=model_view,
                         replay_settings=a.replay_settings, build_backend={"stages": StagesBuild, "run": StagesRun}.get(a.build, LegacyBuild)())
        if not a.replay:
            from app2.scan.settings import load as settings_load,path as settings_path,save as settings_save
            from app2.scan.calibration import store
            if a.param:
                selected=store(a.param,settings_path().parent,'near');cfg.near_param=selected['folder'];settings_save(near_param=cfg.near_param)
            else:cfg.near_param=settings_load().get('near_param')
        if getattr(a,'far_param',None):                  # registered like --param: Settings, Camera setup owns it from then on
            from app2.scan.calibration import store_far
            selected=store_far(a.far_param,settings_path().parent,(cfg.w,cfg.h));cfg.far_param=selected['folder'];settings_save(far_param=cfg.far_param)
        return ScanSession(cfg, src, bridge, bed=load_bed_hook())
    return make


def main(argv=None):
    a = parse(argv)
    plat = prepare_env()
    import app2  # noqa: F401 (puts the repo and scanner/ on sys.path)
    from app2.memory_watch import trace_start; trace_start()      # POINTYOINK_MEMTRACE=<file>: developer memory tracing
    from app2.version import VERSION as _V
    from app2.diagnostics import log as _log
    _log.setup(_V, "(platform %s)" % plat)               # the log file, the error hooks and the crash trace, from the start
    # Learn bed: one OpenCV / BLAS thread for the process, here on the main thread before any other thread starts (the
    # Tk scanner's order: from another thread it deadlocked against the main thread loading a library)
    from app2.scan.bedhook import quiet_threads
    quiet_threads()
    gl_wanted = not a.no_gl and plat != "offscreen"
    if gl_wanted:
        from app2.qt.view3d import default_format
        default_format()                               # before the QApplication exists
    from PySide6.QtWidgets import QApplication
    from app2.qt import theme
    from app2.version import VERSION
    app = QApplication(sys.argv[:1]); app.setApplicationName("PointYoink"); app.setApplicationVersion(VERSION)
    app.setDesktopFileName("point-yoink")                       # the menu entry (and its icon) the window belongs to
    theme.install(app)
    splash = None
    if plat != "offscreen":                            # the splash: up before the slow imports below
        from app2.qt.splash import Splash
        splash = Splash(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png"), VERSION)
        splash.show(); splash.step("Getting things ready...", 0.05)
    step = splash.step if splash else (lambda *_: None)
    step("Loading the 3D tools...", 0.2)
    from app2.qt.shell import MainWindow
    step("Checking the graphics card...", 0.6)
    if gl_wanted: gl, why = probe_gl()
    else: gl, why = False, ("--no-gl" if a.no_gl else "platform %s has no OpenGL" % plat)
    print("model view: %s (%s)" % ("GL points" if gl else "picture from the scan worker", why), flush=True)
    step("Opening your workspace...", 0.8)
    w = MainWindow(session_factory(a, "points" if gl else "image"), gl=gl, projects_dir=a.root,
                   defer_new_project=not (a.project or a.last or a.replay or __import__("app2.general_prefs",fromlist=["get"]).get("open_last")))
    step("Ready", 1.0)
    if splash: splash.hold()                           # seen, not a blink, then the window
    from app2.diagnostics.driver import maybe_start
    maybe_start(w)                                      # the developer driver: only when POINTYOINK_DRIVER names a socket
    if splash and splash.screen() is not None:          # open on the splash's screen, not wherever the mouse went meanwhile
        area = splash.screen().availableGeometry(); w.move(area.center() - w.rect().center())
    w.show()
    if splash: splash.finish(w)
    # Ctrl+C in the terminal or a plain kill: Finish (every side saved, the scanner left as found), no questions asked
    import signal
    from PySide6.QtCore import QTimer
    def finish(*_): w.force_close = True; QTimer.singleShot(0, w.close)
    signal.signal(signal.SIGINT, finish); signal.signal(signal.SIGTERM, finish)
    wake = QTimer(); wake.timeout.connect(lambda: None); wake.start(250)      # lets Python see a signal while Qt waits
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
