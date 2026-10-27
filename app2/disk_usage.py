# SPDX-License-Identifier: AGPL-3.0-or-later
"""How much disk each project takes. Walks the folder without following links, off the GUI thread (a 12-side project
is tens of GB in thousands of files); results are cached until the folder's newest change."""
import os, threading

_cache = {}                                   # path -> (stamp, bytes)
_lock = threading.Lock()


def folder_bytes(path):
    """Total size of the files under path (links not followed, unreadable entries skipped)."""
    total, stack = 0, [path]
    while stack:
        try: it = os.scandir(stack.pop())
        except OSError: continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False): stack.append(e.path)
                    elif e.is_file(follow_symlinks=False): total += e.stat(follow_symlinks=False).st_blocks * 512
                except OSError: pass
    return total


def _stamp(path):
    """Cheap change marker: the project folder's and its data/ and stages/ folders' mtimes (new sides and builds land there)."""
    out = []
    for p in (path, os.path.join(path, "data"), os.path.join(path, "stages")):
        try: out.append(os.stat(p).st_mtime_ns)
        except OSError: out.append(0)
    return tuple(out)


def project_bytes(path):
    """Cached size of one project, or None while it has never been measured."""
    with _lock: hit = _cache.get(path)
    return hit[1] if hit and hit[0] == _stamp(path) else None


def measure(paths, done):
    """Measure paths on a worker thread; done(path, bytes) is called from that thread for each (wrap it in a Qt signal)."""
    def run():
        for p in paths:
            s = _stamp(p)
            with _lock: hit = _cache.get(p)
            if hit and hit[0] == s: n = hit[1]
            else:
                n = folder_bytes(p)
                with _lock: _cache[p] = (s, n)
            done(p, n)
    t = threading.Thread(target=run, name="disk-usage", daemon=True); t.start(); return t


def nice(n):
    """Bytes as the user reads them: 850 KB, 12.4 MB, 33 GB."""
    if n is None: return "..."
    for unit, k in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= k: v = n / k; return ("%.0f %s" if v >= 100 or unit == "KB" else "%.1f %s") % (v, unit)
    return "%d B" % n
