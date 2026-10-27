# SPDX-License-Identifier: AGPL-3.0-or-later
"""Artifact store: stages/nodes/<node or list>.ops/<key16>/ with meta.json written last.

An artifact is only ever written into stages/tmp/<name>.<key16>.<pid>/ and then renamed into place, so a crash never
leaves anything half-written in nodes/. A lookup is a hit when meta.json exists, its key matches and every declared
output still has its recorded size; a hit touches meta.json so its mtime records the last use (for GC)."""
import fcntl, json, os, shutil, time, uuid

from .keys import key16, output_digest
from .keys import jload, jdump


def _atomic_json(path, obj):
    tmp = path + ".tmp.%s.%d" % (uuid.uuid4().hex, os.getpid())
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _timings_path():
    """Step timings across every project on this PC (so a new project starts from real numbers): next to the app's config."""
    cfg = os.environ.get("POINTYOINK_CONFIG")
    return os.path.join(os.path.dirname(cfg) if cfg else os.path.expanduser("~/.config/pointyoink"), "timings.json")


MIN_SECS = 0.5                                              # shorter "runs" are cache hits: never a rate to plan with


def global_secs():
    """{step kind: [seconds, ...]} from earlier projects (the last 30 of each), or {}."""
    try:
        with open(_timings_path()) as f: d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError): return {}


def global_size_secs():
    """{step kind: [[size, seconds], ...]} from earlier projects: a rate a new project's own size can scale."""
    d = global_secs().get("~sized")
    return d if isinstance(d, dict) else {}


def note_global_secs(kind, secs, size=None):
    if not secs or secs < MIN_SECS: return                  # a cache hit or a no-op is not a timing
    try:
        d = global_secs(); h = d.setdefault(kind, []); h.append(round(float(secs), 2)); del h[:-30]
        if size is not None and size > 0:
            sized = d.setdefault("~sized", {}).setdefault(kind, []); sized.append([round(float(size), 2), round(float(secs), 2)]); del sized[:-30]
        p = _timings_path(); os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".part"
        with open(tmp, "w") as f: json.dump(d, f)
        os.replace(tmp, p)
    except (OSError, ValueError, TypeError): pass          # a missing history only costs a worse estimate


class Store:
    def __init__(self, project, memo):
        self.p = project; self.memo = memo
        self.nodes = os.path.join(project.stages, "nodes"); self.tmp = os.path.join(project.stages, "tmp")
        self.state_path = os.path.join(project.stages, "state.json")

    # ------------------------------------------------------------------------------------------------ artifacts
    def dir(self, store, key): return os.path.join(self.nodes, store, key16(key))

    def lookup(self, store, key):
        d = self.dir(store, key); mp = os.path.join(d, "meta.json")
        try: meta = jload(mp)
        except Exception: return None
        if meta.get("key") != key: return None
        for name, o in (meta.get("outputs") or {}).items():
            f = os.path.join(d, o["file"])
            try:
                if os.path.getsize(f) != o["bytes"]: return None
            except OSError: return None
        try: os.utime(mp, None)
        except OSError: pass
        meta["_dir"] = d
        return meta

    def peek(self, store, key):
        """lookup without touching (status/plan)."""
        d = self.dir(store, key); mp = os.path.join(d, "meta.json")
        try: meta = jload(mp)
        except Exception: return None
        if meta.get("key") != key: return None
        for name, o in (meta.get("outputs") or {}).items():
            try:
                if os.path.getsize(os.path.join(d, o["file"])) != o["bytes"]: return None
            except OSError: return None
        meta["_dir"] = d
        return meta

    def tmp_dir(self, store, key):
        d = os.path.join(self.tmp, "%s.%s.%s.%d" % (store.replace("/", "_"), key16(key), uuid.uuid4().hex, os.getpid()))
        os.makedirs(d)
        return d

    def clean_tmp(self):
        """Remove tmp folders of processes that are gone (an interrupted run leaves only its tmp folder)."""
        if not os.path.isdir(self.tmp): return
        for name in os.listdir(self.tmp):
            try: pid = int(name.rsplit(".", 1)[1])
            except (IndexError, ValueError): pid = None
            if pid is not None and pid != os.getpid():
                try: os.kill(pid, 0); continue
                except ProcessLookupError: pass
                except PermissionError: continue
            elif pid == os.getpid(): continue
            shutil.rmtree(os.path.join(self.tmp, name), ignore_errors=True)

    def publish(self, tmp, store, key, meta, outputs, geo_fn=None):
        """outputs: {name: file name in tmp}. Digests the files, writes meta.json last, renames tmp into place.
        An existing artifact under the same key (a forced re-run) is replaced. Returns the final meta.
        Mesh artifacts also get "geo_digest" (order-independent, keys.DigestMemo.geo): downstream keys use it, so the
        same surface in another vertex order is an early cutoff; "digest" (bytes) stays the storage identity."""
        files = {}; digests = {}; geos = {}
        for name, fn in (outputs or {}).items():
            fp = os.path.join(tmp, fn)
            dg = self.memo.digest(fp); digests[name] = dg
            files[name] = {"file": fn, "digest": dg, "bytes": os.path.getsize(fp)}
            if meta.get("kind") == "mesh" and fn.endswith(".ply"):
                g = self.memo.geo(fp, geo_fn)
                if g: files[name]["geo"] = g; geos[name] = g
        meta = dict(meta); meta["key"] = key; meta["outputs"] = files
        if outputs: meta["digest"] = output_digest(digests)
        if geos: meta["geo_digest"] = output_digest({n: geos.get(n, d) for n, d in digests.items()})
        meta.setdefault("created", time.time()); meta.setdefault("adopted", False); meta.setdefault("pinned", False)
        meta.setdefault("status", "ok"); meta.setdefault("warn", []); meta["engine"] = 1
        _atomic_json(os.path.join(tmp, "meta.json"), meta)
        final = self.dir(store, key)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        if os.path.exists(final):
            old = final + ".old.%d" % os.getpid()
            os.rename(final, old); os.rename(tmp, final); shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, final)
        # the memo is keyed on the real path: move the digests (and a known geo digest) to the files' final place
        with self.memo.lock:
            for name, o in files.items():
                fp = os.path.join(final, o["file"]); st = os.stat(fp)
                old = self.memo.memo.pop(os.path.realpath(os.path.join(tmp, o["file"])), None)
                self.memo.memo[os.path.realpath(fp)] = [st.st_size, st.st_mtime_ns, st.st_ino, o["digest"]] + \
                    (old[4:5] if old and len(old) > 4 else [])
                self.memo.dirty = True
        meta["_dir"] = final
        return meta

    def update_meta(self, d, **kw):
        mp = os.path.join(d, "meta.json")
        try: meta = jload(mp)
        except Exception: return None
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(meta.get(k), dict): meta[k].update(v)
            else: meta[k] = v
        st = os.stat(mp); _atomic_json(mp, meta)
        try: os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns))       # a stats update is not a use
        except OSError: pass
        return meta

    def artifacts(self):
        """Every stored artifact: [(store name, key16, dir, meta)]."""
        out = []
        if not os.path.isdir(self.nodes): return out
        for store in sorted(os.listdir(self.nodes)):
            sd = os.path.join(self.nodes, store)
            if not os.path.isdir(sd): continue
            for k in sorted(os.listdir(sd)):
                d = os.path.join(sd, k)
                try: meta = jload(os.path.join(d, "meta.json"))
                except Exception: meta = None
                out.append((store, k, d, meta))
        return out

    # ------------------------------------------------------------------------------------------------ state.json
    def state(self):
        try: return jload(self.state_path)
        except Exception: return {"nodes": {}, "secs": {}}

    def save_state(self, st): _atomic_json(self.state_path, st)

    def note_state(self, node, key=None, status=None, secs=None, kind=None, size=None):
        with open(self.state_path + ".lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try: return self._note_state_locked(node, key, status, secs, kind, size)
            finally: fcntl.flock(handle, fcntl.LOCK_UN)

    def _note_state_locked(self, node, key=None, status=None, secs=None, kind=None, size=None):
        st = self.state(); e = st.setdefault("nodes", {}).setdefault(node, {})
        if key: e["last_key"] = key
        if status: e["status"] = status
        if status in ("done", "cached", "warn", "adopted") and key: e["last_ok_key"] = key
        if secs is not None:
            e["secs"] = round(secs, 2)
            if size is not None and size > 0:
                e["size"] = round(float(size), 2)
            if kind and secs >= MIN_SECS:
                h = st.setdefault("secs", {}).setdefault(kind, []); h.append(round(secs, 2)); del h[:-15]
                note_global_secs(kind, secs, size)
                if size is not None and size > 0:
                    sized = st.setdefault("size_secs", {}).setdefault(kind, [])
                    sized.append([round(float(size), 2), round(secs, 2)]); del sized[:-15]
        self.save_state(st)

    def dir_bytes(self, d):
        n = 0
        for root, _, files in os.walk(d):
            for f in files:
                try: n += os.path.getsize(os.path.join(root, f))
                except OSError: pass
        return n
