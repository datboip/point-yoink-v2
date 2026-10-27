# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a project's model in a separate process and follow it through its STAGE lines ("STAGE <name> <json>"): a
progress share that never moves backwards, a short step text, full lines for the log, the finished model and its
preview. The build itself is a backend: build_project.py today (LegacyBuild); the staged engine (scanner/stages) prints
the same stream with --legacy-events (StagesBuild), so the Scan step's Build button works with either. Later a backend
that reads the engine's own events (plan, node, progress, published, done, error) runs through this same runner, still
as a subprocess (memory caps, and a crash in the build never takes the window with it)."""
import json, os, signal, subprocess, sys, threading, time
from app2 import SCANNER

# where each build step sits on the bar (rough shares of the time: every side, every pair, then the rest)
SPAN = {"side": (0.0, 0.35), "table": (0.0, 0.35), "align": (0.35, 0.70), "aligned": (0.35, 0.70), "base": (0.70, 0.72),
        "refine": (0.72, 0.80), "fuse": (0.80, 0.95), "done": (1.0, 1.0)}


class BuildProgress:
    """frac: 0..1 for the bar; text: the Tk scanner's step line ("Building: ..."); done: the "done" event's fields (or
    the model found after a clean exit); failed: why it failed; preview: the picture to show when it is done (and
    preview_needs: that picture is model_needs.png, red = scan here); finished: the process has ended."""
    def __init__(self):
        self.frac = 0.0; self.text = "Building: starting..."; self.sides_n = None; self.done = None; self.failed = None
        self.t0 = time.time(); self.t_end = None; self.returncode = None; self.finished = False
        self.preview = None; self.preview_needs = False; self.model = None

    def snapshot(self):
        s = BuildProgress(); s.__dict__.update(self.__dict__); return s

    @property
    def step(self):
        """The step text without "Building: ", for the progress line ("Fusing all sides, frame 812 of 3380")."""
        t = self.text[len("Building: "):] if self.text.startswith("Building: ") else self.text
        return t[:1].upper() + t[1:]

    def elapsed(self): return (self.t_end or time.time()) - self.t0

    def elapsed_text(self):
        return "%d%%  %d:%02d" % (100 * self.frac, *divmod(int(self.elapsed()), 60))


def parse_line(ln, prog):
    """One line of the build's output into prog; returns the log lines it makes. The Tk scanner's parser, unchanged
    (build_model's worker loop): unknown stage names and bad JSON are ignored as there."""
    out = []
    if ln.startswith(("median of pair medians", "kept ")): out.append(ln.strip())       # refine and keep-part summaries
    if not ln.startswith("STAGE "): return out
    parts = ln.split(" ", 2)
    try: info = json.loads(parts[2])
    except Exception: info = {}
    name = parts[1]
    if name == "frames":                                            # fuse's frame count: the bar creeps, the text counts
        k, n = info.get("done", 0), max(1, info.get("total", 1))
        if info.get("step") == "all":
            prog.frac = max(prog.frac, 0.80 + 0.15 * k / n); prog.text = "Building: fusing %d / %d" % (k, n)
        else:
            sn, ns = info.get("side", 1), max(1, info.get("sides", 1))
            prog.frac = max(prog.frac, 0.35 * (sn - 1 + k / n) / ns)
            prog.text = "Building: side %d/%d, %d / %d" % (sn, ns, k, n)
        return out
    if name in SPAN:                                                # move the bar: never backwards
        lo, hi = SPAN[name]; k = info.get("done", info.get("side", 1) - 1 if "side" in info else 0); n = info.get("total", info.get("count", 1))
        if name == "side": prog.sides_n = n
        if name == "table": n = prog.sides_n or 1; k = info.get("side", 1)
        prog.frac = max(prog.frac, lo + (hi - lo) * min(1.0, k / max(1, n)))
    if name == "table": out.append("side %s: %s" % (info.get("side", "?"), info.get("msg", "")))
    elif name == "aligned": out.append("sides %s line up: fit %s, %s mm" % (info.get("sides"), info.get("fit"), info.get("rmse_mm")))
    elif name in ("warn", "error", "base", "done"): out.append("%s: %s" % (name, info.get("msg") or info.get("model", "")))
    elif "msg" in info and name not in ("align",): out.append(info["msg"])
    if name == "table": info["msg"] = "platter cut on side %s" % info.get("side", "?")   # the cutter's report is a paragraph
    if "msg" in info:
        m = info["msg"] if len(info["msg"]) <= 60 else info["msg"][:57].rsplit(" ", 1)[0] + "..."   # two short lines at most
        prog.text = "Building: " + m
    if name == "done": prog.done = info
    if name == "error": prog.failed = "Build failed: " + info.get("msg", "")
    return out


class LegacyBuild:
    """build_project.py: the one-shot build of today."""
    name = "build_project.py"
    label = "Engine: build_project.py"
    def argv(self, project): return [sys.executable, "-u", os.path.join(SCANNER, "build_project.py"), project]
    def exit_text(self, rc): return None


class StagesBuild:
    """The staged engine (scanner/stages), printing build_project.py's stream (its --legacy-events mode)."""
    name = "stages"
    label = "Engine: staged build (legacy events)"
    def argv(self, project): return [sys.executable, "-u", os.path.join(SCANNER, "stages", "cli.py"), project, "run", "--legacy-events"]
    def exit_text(self, rc): return "The project is being built somewhere else." if rc == 3 else None


def StagesRun(*a, **kw):
    """The staged engine with its own events and a control per run (--to, --force, --with, --no-publish): the Build
    step's backend (app2/builder/stagehost.py)."""
    from app2.builder.stagehost import StagesRun as _S
    return _S(*a, **kw)


class LogReplayBuild:
    """Tests: prints a recorded build log (timestamps like "00:43:31 " dropped) with a pause between lines."""
    name = "log replay"
    label = "Engine: recorded build log"
    def __init__(self, log, delay=0.0): self.log, self.delay = log, delay

    def argv(self, project):
        code = ("import sys,time,re\nfor ln in open(sys.argv[1]):\n ln=re.sub(r'^\\d\\d:\\d\\d:\\d\\d ','',ln)\n"
                " sys.stdout.write(ln); sys.stdout.flush(); time.sleep(float(sys.argv[2]))\n")
        return [sys.executable, "-u", "-c", code, self.log, str(self.delay)]

    def exit_text(self, rc): return None


class BuildRunner:
    """Runs one build on its own thread. emit(kind, payload): "build" (a BuildProgress copy, at most 10 a second and on
    the last line), "build_line" (every line of output except fuse's frame counter) and "log" (the lines worth keeping);
    on_finished(progress) once the process has ended. cancel() stops the build and everything it started (it runs in
    its own process group): SIGTERM, then SIGKILL after the backend's cancel_grace seconds (3 by default; the staged
    engine's CLI needs longer to stop its own children, which run in sessions of their own)."""
    def __init__(self, project, backend, emit, on_finished):
        self.project, self.backend, self.emit, self.on_finished = project, backend, emit, on_finished
        self.prog = BuildProgress(); self.p = None; self.cancelled = False; self.parse = getattr(backend, "parse", parse_line)
        self.thread = threading.Thread(target=self._run, name="build", daemon=True)

    def start(self): self.thread.start(); return self

    # Pause / Resume: freeze the build process and everything it started (the jobs run in groups of their own, so the whole
    # tree is found through /proc), then let it go on. A stopped process cannot hear a Stop, so cancel() resumes first.
    @staticmethod
    def _tree(pid):
        kids = {}
        try:
            for name in os.listdir("/proc"):
                if not name.isdigit(): continue
                try:
                    with open("/proc/%s/stat" % name) as f: rest = f.read().rsplit(")", 1)[1].split()
                    kids.setdefault(int(rest[1]), []).append(int(name))
                except (OSError, IndexError, ValueError): continue
        except OSError: pass
        out, todo = [], [pid]
        while todo:
            q = todo.pop()
            for k in kids.get(q, []): out.append(k); todo.append(k)
        return [pid] + out

    def _signal_tree(self, sig):
        p = self.p
        if p is None or p.poll() is not None: return False
        for q in self._tree(p.pid):
            try: os.kill(q, sig)
            except (ProcessLookupError, PermissionError): pass
        return True

    def pause(self):
        """Freeze the build. Twice, a moment apart, to catch a job started in the instant between."""
        if getattr(self, "paused", False): return False
        if not self._signal_tree(signal.SIGSTOP): return False
        self.paused = True; time.sleep(0.15); self._signal_tree(signal.SIGSTOP); return True

    def resume(self):
        if not getattr(self, "paused", False): return False
        self.paused = False; self._signal_tree(signal.SIGCONT); return True

    def cancel(self):
        self.resume()                                   # a stopped build could not hear the Stop
        self.cancelled = True
        p = self.p
        if p is None or p.poll() is not None: return
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: return
        grace = float(getattr(self.backend, "cancel_grace", 3))
        def kill_later():
            try: p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        threading.Thread(target=kill_later, name="build-kill", daemon=True).start()

    def stop(self, timeout=5.0):
        """Closing: cancel and wait (bounded) for the build to end. SIGKILL for its group when SIGTERM was not enough in
        time: cancel()'s own SIGKILL comes from a daemon thread that dies with the app, and a build left running would
        go on writing into the project."""
        self.cancel(); self.thread.join(max(timeout, float(getattr(self.backend, "cancel_grace", 3)) + 1.0))
        p = self.p
        if p is not None and p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            try: p.wait(2.0)
            except subprocess.TimeoutExpired: pass
        self.thread.join(1.0)

    def _run(self):
        prog = self.prog; last = 0.0
        try:
            self.p = subprocess.Popen(self.backend.argv(self.project), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1, start_new_session=True, errors="replace",
                                      env=dict(os.environ, **getattr(self.backend, 'env', {})))
            if self.cancelled: self.cancel()
            for ln in self.p.stdout:
                if not ln.startswith("STAGE frames "): self.emit("build_line", ln.rstrip("\n"))
                for out in self.parse(ln, prog): self.emit("log", out)
                if time.monotonic() - last > 0.1: last = time.monotonic(); self.emit("build", prog.snapshot())
            self.p.wait(); prog.returncode = self.p.returncode
        except Exception as e:
            prog.failed = "Build failed: %s" % e
        self._finish(prog)
        prog.finished = True; prog.t_end = time.time()
        self.emit("build", prog.snapshot())
        self.on_finished(prog)

    def _finish(self, prog):
        """Success: a parsed done, or a clean exit with a model.ply newer than the build's start (a build whose last line
        got lost still built its model). Then the picture to show."""
        model = os.path.join(self.project, "model.ply")
        if self.cancelled and not prog.done: prog.failed = "Build stopped"; return
        if not prog.done and prog.returncode == 0 and os.path.exists(model) and os.path.getmtime(model) >= prog.t0:
            prog.done = {"model": model}
        if not prog.done:
            if not prog.failed:
                prog.failed = self.backend.exit_text(prog.returncode) or "Build failed (exit %s)" % prog.returncode
            return
        prog.failed = None; prog.frac = 1.0
        # a --no-publish run (a stage built for the Build step) made no model.ply: never report an older one as built
        prog.model = prog.done.get("model") or (model if getattr(self.backend, "publish", True) else None)
        need = os.path.join(self.project, "model_needs.png"); prev = prog.done.get("preview")
        if prev and os.path.exists(prev):
            if os.path.exists(need) and os.path.getmtime(need) >= os.path.getmtime(prev): prev = need
        else:                                          # no preview named: the newest picture this build made
            made = [p for p in (need, os.path.join(self.project, "model.png")) if os.path.exists(p) and os.path.getmtime(p) >= prog.t0]
            prev = max(made, key=os.path.getmtime) if made else None
        prog.preview = prev; prog.preview_needs = prev is not None and os.path.abspath(prev) == os.path.abspath(need)
