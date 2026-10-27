# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 2.0 log file. Before this, errors only reached the in-app Log popover and the terminal and were gone when the app closed.

Writes `app2.log` (1 MB, 3 files) in the cache folder: the version and system at the start, every line the app prints, uncaught
errors (main thread, other threads, Qt's own warnings) and native crashes (`crash.log`). A small marker file says whether the
last session ended cleanly, so the next start can offer to report a crash. The raw log keeps real paths: the Report a problem
window scrubs a copy, never this file in place.
"""
import atexit, faulthandler, io, json, logging, os, platform, sys, threading, time, traceback
from logging.handlers import RotatingFileHandler

NAME = "pointyoink.app2"
_state = {}


def log_dir():
    d = os.environ.get("POINTYOINK_LOG_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "pointyoink")
    os.makedirs(d, exist_ok=True)
    return d


class _Tee(io.TextIOBase):
    """Passes writes through to the real stream and logs each finished line."""
    def __init__(self, real, logger, level):
        self.real, self.logger, self.level, self.buf = real, logger, level, ""

    def write(self, text):
        try: self.real.write(text)
        except Exception: pass
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip(): self.logger.log(self.level, line.rstrip())
        return len(text)

    def flush(self):
        try: self.real.flush()
        except Exception: pass

    def isatty(self): return False
    def fileno(self): return self.real.fileno()


def crashed_last_time():
    """True when the previous session never wrote its clean-exit mark (read before setup() starts a new one)."""
    try:
        with open(os.path.join(log_dir(), "session.json")) as f: return not json.load(f).get("clean", True)
    except (OSError, ValueError): return False


def _write_marker(clean):
    try:
        with open(os.path.join(log_dir(), "session.json"), "w") as f: json.dump({"pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "clean": clean}, f)
    except OSError: pass


LISTENERS = []     # called with (kind, exc) on every uncaught error, from whatever thread it happened on


def setup(version, extra=""):
    """Start logging. Returns the logger. Safe to call once per process; teardown() undoes it (tests)."""
    if _state: return _state["logger"]
    d = log_dir(); logger = logging.getLogger(NAME); logger.setLevel(logging.INFO); logger.propagate = False
    handler = RotatingFileHandler(os.path.join(d, "app2.log"), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(handler)
    _state.update(logger=logger, handler=handler, was_crash=crashed_last_time(),
                  excepthook=sys.excepthook, thread_hook=threading.excepthook, stdout=sys.stdout, stderr=sys.stderr)
    logger.info("=== PointYoink %s starting%s", version, (" " + extra) if extra else "")
    logger.info("system: %s, Python %s, pid %d", platform.platform(), platform.python_version(), os.getpid())
    if _state["was_crash"]: logger.warning("the previous session did not end cleanly")

    def hook(kind, exc_type, exc, tb):
        logger.critical("uncaught error (%s): %s\n%s", kind, exc, "".join(traceback.format_exception(exc_type, exc, tb)).rstrip())
        for fn in list(LISTENERS):
            try: fn(kind, exc)
            except Exception: pass
    def main_hook(t, v, tb):
        hook("main thread", t, v, tb); _state["excepthook"](t, v, tb)
    def thread_hook(args):
        hook("thread %s" % getattr(args.thread, "name", "?"), args.exc_type, args.exc_value, args.exc_traceback); _state["thread_hook"](args)
    sys.excepthook = main_hook; threading.excepthook = thread_hook
    try:
        crash = open(os.path.join(d, "crash.log"), "a"); crash.write("\n--- %s pid %d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid())); crash.flush()
        faulthandler.enable(file=crash, all_threads=True); _state["crash_file"] = crash
    except OSError: pass
    sys.stdout = _Tee(sys.stdout, logger, logging.INFO); sys.stderr = _Tee(sys.stderr, logger, logging.WARNING)
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO, QtMsgType.QtWarningMsg: logging.WARNING,
                  QtMsgType.QtCriticalMsg: logging.ERROR, QtMsgType.QtFatalMsg: logging.CRITICAL}
        qInstallMessageHandler(lambda kind, ctx, msg: logger.log(levels.get(kind, logging.INFO), "Qt: %s", msg))
    except Exception: pass
    _write_marker(False); atexit.register(_clean_exit)
    return logger


def _clean_exit():
    if _state and _state.get("logger"): _state["logger"].info("=== clean exit"); _write_marker(True)


def was_crash():
    """True when this process started right after a session that never exited cleanly."""
    return bool(_state.get("was_crash"))


def teardown():
    """Put the streams and hooks back and close the files (tests, or an orderly shutdown)."""
    if not _state: return
    sys.excepthook = _state["excepthook"]; threading.excepthook = _state["thread_hook"]; sys.stdout = _state["stdout"]; sys.stderr = _state["stderr"]
    try: faulthandler.disable()
    except Exception: pass
    _state["logger"].removeHandler(_state["handler"]); _state["handler"].close()
    if _state.get("crash_file"): _state["crash_file"].close()
    try:
        from PySide6.QtCore import qInstallMessageHandler
        qInstallMessageHandler(None)
    except Exception: pass
    _state.clear()


def tail(lines=300):
    """The last lines of the current app log (raw), for the report."""
    try:
        with open(os.path.join(log_dir(), "app2.log"), encoding="utf-8", errors="replace") as f: return f.read().splitlines()[-lines:]
    except OSError: return []
