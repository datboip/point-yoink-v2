# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory watchdog: how much RAM this app and everything it started hold, against what the PC has.

The 2026-10-22 runaway (the app at 36 GB plus a 7 GB build worker on a 62 GB PC, swapping, the window still
"measuring coverage") gave no warning at all. The window polls `check` every few seconds and says so when the app
takes more than its share or the PC is nearly out.
Linux only (reads /proc); elsewhere `usage` returns None and nothing is said."""
import os

GIB = 1024 ** 3
OWN_SHARE = 0.45            # warn when the app and its workers hold this share of the PC's RAM...
LOW_AVAILABLE = 0.08        # ...or when what is left for everything else drops to this share
REPEAT_S = 60.0             # the warning is said again at most this often while it still holds


def _meminfo():
    try:
        with open("/proc/meminfo") as f:
            m = {k.strip(): int(v.split()[0]) * 1024 for k, v in (ln.split(":", 1) for ln in f if ":" in ln)}
        return m.get("MemTotal"), m.get("MemAvailable")
    except (OSError, ValueError): return None, None


def _procs():
    """{pid: (ppid, rss_bytes)} for every process we can read."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit(): continue
        try:
            with open("/proc/%s/stat" % name) as f: st = f.read()
            # pid (comm) state ppid ... rss is field 24 (1-based); comm may hold spaces so split after the last ')'
            tail = st[st.rindex(")") + 2:].split()
            out[int(name)] = (int(tail[1]), int(tail[21]) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError): continue
    return out


def usage(root_pid=None):
    """(ours, total, available) in bytes: ours is this process plus every descendant (build workers, quality
    subprocesses). None when /proc is not there."""
    total, available = _meminfo()
    if total is None: return None
    procs = _procs()
    if not procs: return None
    root = os.getpid() if root_pid is None else root_pid
    kids = {}
    for pid, (ppid, _) in procs.items(): kids.setdefault(ppid, []).append(pid)
    ours = 0; stack = [root]; seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in procs: continue
        seen.add(pid); ours += procs[pid][1]; stack.extend(kids.get(pid, ()))
    return ours, total, available


def warning(ours, total, available):
    """The sentence to show, or None when memory is fine."""
    if total <= 0: return None
    big = ours >= OWN_SHARE * total; low = available <= LOW_AVAILABLE * total
    if not (big or low): return None
    why = ("PointYoink holds %.0f of the PC's %.0f GB of RAM" % (ours / GIB, total / GIB)) if big else \
          ("the PC has %.1f GB of RAM left and PointYoink holds %.0f GB" % (available / GIB, ours / GIB))
    return "Memory warning: %s. Stop the build or coverage measure, or close the project, before the PC runs out." % why


TRACE_ENV = "POINTYOINK_MEMTRACE"       # developer: =<file> writes tracemalloc's biggest allocation sites there every minute


def trace_start():
    """Developer: POINTYOINK_MEMTRACE=<file> starts tracemalloc (call before the heavy imports)."""
    if os.environ.get(TRACE_ENV):
        import tracemalloc; tracemalloc.start(1)        # deeper stacks slow tracking enough to trip its too-slow guard


def trace_dump(path, ours=None):
    import tracemalloc, time
    if not tracemalloc.is_tracing(): return
    snap = tracemalloc.take_snapshot().filter_traces((tracemalloc.Filter(False, tracemalloc.__file__),))
    lines = ["== %s traced %.2f GB, RSS tree %s" % (time.strftime("%H:%M:%S"), tracemalloc.get_traced_memory()[0] / GIB,
                                                   "%.2f GB" % (ours / GIB) if ours else "?")]
    for st in snap.statistics("lineno")[:25]:
        lines.append("%8.1f MB %6d blocks  %s" % (st.size / 2 ** 20, st.count, st.traceback.format(most_recent_first=False)[-1].strip()))
    lines.append("-- by traceback (top 6)")
    for st in snap.statistics("traceback")[:6]:
        lines.append("%8.1f MB %6d blocks" % (st.size / 2 ** 20, st.count))
        lines.extend("      " + ln.strip() for ln in st.traceback.format(most_recent_first=True)[:8])
    with open(path, "a") as f: f.write("\n".join(lines) + "\n")


class Watch:
    """Polled by the window: `check(now)` returns a warning to say (at most once a minute) or None. `last` keeps the
    latest reading for the log."""
    def __init__(self): self.last = None; self.said_t = None; self.text = None; self.trace_t = None

    def check(self, now):
        u = usage()
        self.last = u
        path = os.environ.get(TRACE_ENV)
        if path and (self.trace_t is None or now - self.trace_t >= REPEAT_S):
            self.trace_t = now
            try: trace_dump(path, u[0] if u else None)
            except Exception as e: print("memtrace failed:", e, flush=True)
        if u is None: return None
        text = warning(*u)
        self.text = text
        if text is None: self.said_t = None; return None
        if self.said_t is not None and now - self.said_t < REPEAT_S: return None
        self.said_t = now
        return text
