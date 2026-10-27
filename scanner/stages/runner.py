# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runs one heavy step as a child process under a hard memory cap, one at a time.

The saved Processing preferences apply to every stage, including Poisson, cleanup, and previews.
One processing job runs per config at a time, including its worker pool, under systemd memory and CPU caps.
Available RAM is checked after queue waits; the desktop keeps 15 GiB or 25% of RAM, whichever is larger.
Custom wrappers are refused because they can launch scopes outside the enforced budget.
An isolated POINTYOINK_CONFIG also isolates both processing and GPU locks.

Every child starts in its own session; cancel() kills only those process groups (never pkill). A cancel is for what is
running when it is made: every Engine entry point (run, job, preview, view, detect, delete, adopt, ...) runs inside
Runner.scope(), and cancel() marks only the scopes active at that moment, so the next call starts clean. A cancel that
lands while a step waits for the GPU lock or just after its child started still stops it. Calls made outside any
scope (direct Runner users) share one flag that reset() clears.

A failed child is classified (StepError.kind):
  "launch"  the wrapper never started it or could not exec it (MARK below never printed; systemd-run "Failed to connect
            to bus" / "Failed to start transient scope unit"; heavy.sh / flock / nice errors; the shell's exec failing)
  "killed"  a signal (negative code, or 128+N through a shell: a memory-cap kill shows as 137) or a kill / out-of-memory
            message
  "crash"   a Python traceback, a segfault or an abort
  "exit"    a clean non-zero exit of the script itself that says it is deliberate: a line "STAGE no_result {...}"
            (the only recognised form; no repo script prints it yet)
  "unknown" any other failure
Only "exit" is a deliberate outcome a soft-fail step may keep in the store for good; everything else is transient and
retried (engine._transient). Unrecognised failures are never taken as deliberate.
"""
import fcntl, os, re, shlex, signal, subprocess, sys, threading, time
from contextlib import contextmanager
from . import resources

KILL_MARKS = re.compile(r"^Killed\b|MemoryError|std::bad_alloc|[Oo]ut of memory|oom-kill|\bOOM\b", re.M)
CRASH_MARKS = re.compile(r"Traceback \(most recent call last\)|Segmentation fault|core dumped|Aborted|Fatal Python error")
# the wrapper failing before (or while) starting the child: systemd-run, the user bus, heavy.sh / flock / nice
LAUNCH_MARKS = re.compile(r"Failed to connect to bus|Failed to create bus connection|Failed to start transient (scope|service) unit|"
                          r"Failed to (start|execute|find executable)|^systemd-run: |^flock: |^nice: |heavy\.sh: |"
                          r"Transport endpoint is not connected|Job for \S+ failed", re.M)
# dash: "exec: /x: not found"; bash as /bin/sh (SteamOS, Arch, Fedora): "/x: line 1: /x: No such file or directory"
EXEC_FAIL = re.compile(r"(exec: |line \d+: ).*(not found|Permission denied|Exec format error|No such file)", re.M)
DELIBERATE_MARK = re.compile(r"^STAGE no_result ", re.M)


class Cancelled(Exception): pass


class StepError(Exception):
    def __init__(self, msg, tail="", rc=None, kind=None):
        super().__init__(msg); self.tail = tail; self.rc = rc
        self.kind = kind or classify(rc, tail)

    @property
    def deliberate(self):
        """The script itself said no, cleanly and on purpose (STAGE no_result): safe to keep as the step's result."""
        return self.kind == "exit"

    @property
    def sig(self):
        """What makes two failures 'the same' (engine retry count): the kind and the last meaningful output line."""
        lines = [ln.strip() for ln in (self.tail or "").splitlines() if ln.strip() and not ln.startswith(("$ ", "# exit"))]
        return "%s|%s" % (self.kind, lines[-1][:200] if lines else "rc %s" % self.rc)


def classify(rc, text, started=None, pre=None):
    """launch | killed | crash | exit | unknown for a failed child (see the module doc). started: whether the child
    really started (the MARK line was seen; None = not known); pre: the output printed before it (the wrapper's)."""
    if rc is None: return "crash"
    text = text or ""
    if started is False: return "launch"
    if LAUNCH_MARKS.search(text if started is None else (pre or "")): return "launch"
    if rc in (126, 127) and EXEC_FAIL.search(text): return "launch"
    if rc < 0 or rc >= 128: return "killed"
    if KILL_MARKS.search(text): return "killed"
    if CRASH_MARKS.search(text): return "crash"
    if DELIBERATE_MARK.search(text): return "exit"
    return "unknown"


MARK = "@@stages-start@@"
# a shell prefix that prints MARK once the child really starts (after heavy.sh got its slot / GPU lock), so the time
# spent queued is told apart from the step's own time
PREFIX = ["/bin/sh", "-c", 'echo "%s"; exec "$0" "$@"' % MARK]
NO_SCOPES = "no systemd user session, so RAM and CPU limits are not enforced"
_scopes = None


def scopes_available():
    """Whether `systemd-run --user --scope` works here (checked once). Without it (no systemd, no user bus, some
    containers) steps still run, uncapped. POINTYOINK_NO_SCOPES=1 forces the fallback."""
    global _scopes
    if os.environ.get("POINTYOINK_NO_SCOPES") == "1": return False
    if _scopes is None:
        try:
            _scopes = subprocess.run(["systemd-run", "--user", "--scope", "-q", "true"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError): _scopes = False
    return _scopes


class Runner:
    def __init__(self, wrap=None, nice=10, memory_gb=None):
        self.wrap = wrap if wrap is not None else (os.environ.get("POINTYOINK_STAGES_WRAP") or None)
        # re-entrant: cancel() may run in a signal handler on the thread that holds it (the CLI's SIGINT)
        self.nice = nice; self.pgids = set(); self.lock = threading.RLock(); self._flag = False
        self._tl = threading.local(); self._active = set(); self._cancelled = set(); self._next_scope = 0
        self.last_wait = 0.0
        self.memory_gb = memory_gb
        self.processing_lock_path = str(resources.config_dir() / "processing.lock")
        self.gpu_lock_path = os.path.join(os.path.expanduser("~/.cache/pointyoink"), "gpu.lock")
        if os.environ.get("POINTYOINK_CONFIG"):
            self.gpu_lock_path = str(resources.config_dir() / "gpu.lock")
        self.poll_s = 0.2

    @property
    def last_wait(self):return getattr(self._tl,"last_wait",0.0)

    @last_wait.setter
    def last_wait(self,value):self._tl.last_wait=value

    @contextmanager
    def batch(self,gpu_limit=2):
        """Hold the config lock once; workers receive slices of the same budget."""
        from .parallel import Jobs
        with self._processing_slot():
            budget=resources.budget(self.memory_gb);jobs=Jobs(self,budget,gpu_limit)
            gl=None
            try:
                os.makedirs(os.path.dirname(self.gpu_lock_path),exist_ok=True)
                gl=open(self.gpu_lock_path,"a+")
                while True:
                    if self.cancelled:raise Cancelled()
                    try:fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(self.poll_s)
                yield jobs
            finally:
                if gl:fcntl.flock(gl,fcntl.LOCK_UN);gl.close()

    @contextmanager
    def worker(self,scope,jobs,kind):
        """Propagate the entry scope so cancellation also covers executor threads."""
        self._tl.scope=scope;self._tl.depth=1;self._tl.batch=jobs;self._tl.kind=kind
        try:yield
        finally:self._tl.scope=None;self._tl.depth=0;self._tl.batch=None;self._tl.budget=None

    def reset(self):
        """An earlier cancel no longer applies to calls made outside any scope."""
        self._flag = False

    @contextmanager
    def scope(self):
        """One engine entry point (nested calls share the outermost one): a cancel() made while it is active stops it;
        one made before it started, or after it ended, does not."""
        depth = getattr(self._tl, "depth", 0)
        if depth == 0:
            with self.lock:
                self._next_scope += 1; sid = self._next_scope; self._active.add(sid)
            self._tl.scope = sid
        self._tl.depth = depth + 1
        try:
            yield
        finally:
            self._tl.depth -= 1
            if self._tl.depth == 0:
                sid = self._tl.scope; self._tl.scope = None
                with self.lock: self._active.discard(sid); self._cancelled.discard(sid)

    @property
    def cancelled(self):
        sid = getattr(self._tl, "scope", None)
        if sid is None: return self._flag
        return sid in self._cancelled

    def _budget(self):
        return getattr(self._tl, "budget", None) or resources.budget(self.memory_gb)

    def argv(self, cmd, mem_gb, gpu, light=False):
        # Legacy wrappers can launch scopes outside our cgroup. Do not silently bypass a user limit.
        if self.wrap:
            raise StepError("Custom processing wrappers cannot enforce these limits. Clear POINTYOINK_STAGES_WRAP to use the built-in runner.", kind="launch")
        b = self._budget()
        if not scopes_available():                  # no systemd user session: run, but nothing caps RAM or CPU
            return ["nice", "-n", str(self.nice)] + list(cmd)
        return ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=%d" % b.memory_bytes,
                "-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", "-p", "CPUQuota=%d%%" % b.cpu_quota,
                "nice", "-n", str(self.nice)] + list(cmd)

    def env(self, extra=None):
        e = dict(os.environ)
        e.pop("POINTYOINK_STAGES_WRAP", None)
        if extra: e.update({k: str(v) for k, v in extra.items()})
        threads = str(self._budget().cpu_threads)
        e.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONUNBUFFERED="1")
        # systemd bounds resident memory. The old virtual-address limit would still cap
        # CPU fusion at 12 GiB and conflicts with CUDA mappings even for small jobs.
        e["POINTYOINK_MEM_CAP_GB"] = "100000"
        return e

    @contextmanager
    def _processing_slot(self):
        # Serialize child jobs across this config's UI, build, and preview runners.
        # Each child's cgroup includes its worker pool, so budgets do not multiply.
        os.makedirs(os.path.dirname(self.processing_lock_path), exist_ok=True)
        with open(self.processing_lock_path, "a+") as slot:
            while True:
                if self.cancelled: raise Cancelled()
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(self.poll_s)
            try:
                if self.cancelled: raise Cancelled()
                yield
            finally:
                fcntl.flock(slot, fcntl.LOCK_UN)

    def run(self, cmd, mem_gb=4, gpu=False, log=None, on_line=None, env=None, cwd=None, light=False):
        queued_at = time.time()
        self.last_wait = 0.0
        jobs=getattr(self._tl,"batch",None)
        if jobs is not None:
            with jobs.lease(mem_gb,gpu,getattr(self._tl,"kind",None)) as budget:
                self._tl.budget=budget
                try:return self._run(cmd,mem_gb,gpu,log,on_line,env,cwd,light,queued_at)
                finally:self._tl.budget=None
        with self._processing_slot():
            try:
                return self._run(cmd, mem_gb, gpu, log, on_line, env, cwd, light, queued_at)
            finally:
                self._tl.budget = None

    def _run(self, cmd, mem_gb=4, gpu=False, log=None, on_line=None, env=None, cwd=None, light=False, queued_at=None):
        """Run cmd (list; cmd[0] is the interpreter). on_line(line) is called for every output line.
        Returns (text, peak_mb, secs). Raises StepError on a non-zero exit, Cancelled after cancel()."""
        if self.cancelled: raise Cancelled()
        gl = None; t0 = queued_at if queued_at is not None else time.time()
        if gpu and getattr(self._tl,"batch",None) is None:
            os.makedirs(os.path.dirname(self.gpu_lock_path), exist_ok=True)
            gl = open(self.gpu_lock_path, "a+")
            while True:                           # wait for the GPU without going deaf to a cancel
                try:
                    fcntl.flock(gl, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except OSError:
                    if self.cancelled: gl.close(); raise Cancelled()
                    time.sleep(self.poll_s)
            if self.cancelled:
                fcntl.flock(gl, fcntl.LOCK_UN); gl.close(); raise Cancelled()
        out = []; pre = []; started = [None]
        lf = None
        try:
            try:
                if getattr(self._tl,"budget",None) is None:self._tl.budget = resources.budget(self.memory_gb)
            except ValueError as ex:
                raise StepError(str(ex), kind="launch") from ex
            full = self.argv(PREFIX + list(cmd), mem_gb, gpu, light)
            lf = open(log, "a") if log else None
            if lf:
                b = self._budget()
                if scopes_available():
                    lf.write("# processing limits: %.3f GiB RAM, %d threads, CPU quota %d%%\n" % (b.memory_gb, b.cpu_threads, b.cpu_quota))
                else: lf.write("# processing limits: %d threads; %s\n" % (b.cpu_threads, NO_SCOPES))
            if lf: lf.write("$ %s\n" % " ".join(shlex.quote(c) for c in full)); lf.flush()
            p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env=self.env(env), cwd=cwd, start_new_session=True)
            with self.lock: self.pgids.add(p.pid)
            if self.cancelled: self._kill([p.pid])        # cancel() ran between Popen and the line above: it missed this child
            try:
                for ln in p.stdout:
                    if started[0] is None and ln.strip() == MARK:
                        started[0] = time.time(); continue
                    out.append(ln)
                    if started[0] is None: pre.append(ln)
                    if lf: lf.write(ln)
                    if on_line:
                        try: on_line(ln.rstrip("\n"))
                        except Exception: pass
                _, status, ru = os.wait4(p.pid, 0)
                p.returncode = os.waitstatus_to_exitcode(status)
            finally:
                with self.lock: self.pgids.discard(p.pid)
            peak = int(ru.ru_maxrss / 1024)
            text = "".join(out); t1 = time.time()
            self.last_wait = (started[0] - t0) if started[0] else 0.0
            if lf: lf.write("# exit %d, %.1f s (queued %.1f s), peak ~%d MB\n" % (p.returncode, t1 - (started[0] or t0), self.last_wait, peak))
            if self.cancelled: raise Cancelled()
            if p.returncode != 0:
                kind = classify(p.returncode, text, started[0] is not None, "".join(pre))
                raise StepError("%s failed (exit %d%s)" % (os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0]), p.returncode,
                                                           "" if kind == "exit" else ", " + kind),
                                text[-600:], p.returncode, kind)
            return text, peak, t1 - (started[0] or t0)
        finally:
            if lf: lf.close()
            if gl:
                fcntl.flock(gl, fcntl.LOCK_UN); gl.close()

    def _kill(self, pg):
        for g in pg:
            try: os.killpg(g, signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass
        if pg:
            time.sleep(1.5)
            for g in pg:
                try: os.killpg(g, signal.SIGKILL)
                except (ProcessLookupError, PermissionError): pass

    def cancel(self):
        """Stop what is running now: the active scopes (and calls outside any scope) see Cancelled; their children are
        killed. Scopes that start later are not affected."""
        with self.lock:
            self._flag = True; self._cancelled |= self._active; pg = list(self.pgids)
        self._kill(pg)
