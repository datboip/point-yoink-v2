# SPDX-License-Identifier: AGPL-3.0-or-later
"""The recipe (everything the user chose; the only user state) and its history of snapshots.

history.json is the one source of truth: the recipe is entries[cursor].recipe, read from that single atomically
replaced file (stages/recipe.json is a copy for people and older readers; nothing reads it back). Every user change
appends one snapshot (the entries after the cursor are dropped first); undo/redo only move the cursor. At most 200
entries. A change is a read-modify-write under stages/history.lock (edit()), so two editors (CLI + UI) never lose each
other's changes: the second one's change is applied on top of the first one's recipe."""
import copy, fcntl, json, os, time
from contextlib import contextmanager

from . import params as PR
from .keys import jload, jdump

SCHEMA = 1
MAX_ENTRIES = 200


def new_recipe(preset="today"):
    r = {"schema": SCHEMA, "preset": preset, "params": {}, "sides": {"leave_out": [], "platter": {}, "params": {}},
         "ops": {}, "next_op": 1}
    if preset != "today": r = PR.apply_preset(r, preset)
    return normalise(r)


def normalise(r):
    """Same content -> same bytes: fixed top-level keys, sorted leave-out, empty dicts pruned."""
    r = copy.deepcopy(r)
    r.setdefault("schema", SCHEMA); r.setdefault("preset", "today"); r.setdefault("params", {}); r.setdefault("ops", {})
    s = r.setdefault("sides", {}); s["leave_out"] = sorted(set(s.get("leave_out", []))); s.setdefault("platter", {}); s.setdefault("params", {})
    r["next_op"] = int(r.get("next_op", 1))
    PR.promote_fused_cuts(r)
    PR.prune(r)
    return r


def dumps(r): return json.dumps(normalise(r), indent=1, sort_keys=True)


def _atomic_write(path, text):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f: f.write(text)
    os.replace(tmp, path)


class ReadOnly(ValueError): pass


class History:
    def __init__(self, stages_dir, read_only=False):
        self.dir = stages_dir; self.read_only = read_only
        self.rp = os.path.join(stages_dir, "recipe.json"); self.hp = os.path.join(stages_dir, "history.json")
        self.lp = os.path.join(stages_dir, "history.lock")
        self._memo = (None, None)

    @contextmanager
    def locked(self):
        if self.read_only: raise ReadOnly("opened read-only: no changes")
        f = open(self.lp, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX); yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN); f.close()

    def exists(self): return os.path.exists(self.hp)

    def init(self, recipe, note):
        with self.locked():
            if os.path.exists(self.hp): return
            h = {"schema": SCHEMA, "cursor": 0, "entries": [{"t": time.time(), "note": note, "recipe": normalise(recipe)}]}
            self._write(h)

    def _raw(self):
        """history.json, parsed once per version of the file (every write replaces it, so its inode changes). Shared:
        callers must not change it (_read() gives a private copy)."""
        try:
            st = os.stat(self.hp); sig = (st.st_ino, st.st_size, st.st_mtime_ns)
            if self._memo[0] == sig: return self._memo[1]
            h = jload(self.hp); self._memo = (sig, h)
            return h
        except Exception:
            r = new_recipe()
            return {"schema": SCHEMA, "cursor": 0, "entries": [{"t": time.time(), "note": "New build, preset today", "recipe": r}]}

    def _read(self): return copy.deepcopy(self._raw())

    def _write(self, h):
        _atomic_write(self.hp, json.dumps(h, indent=1, sort_keys=True))
        _atomic_write(self.rp, dumps(h["entries"][h["cursor"]]["recipe"]))       # a copy for people; never read back

    def recipe(self):
        h = self._raw(); return normalise(h["entries"][h["cursor"]]["recipe"])

    @staticmethod
    def _append(h, recipe, note):
        cur = normalise(h["entries"][h["cursor"]]["recipe"])
        if dumps(cur) == dumps(recipe): return False
        h["entries"] = h["entries"][:h["cursor"] + 1]
        h["entries"].append({"t": time.time(), "note": note, "recipe": recipe})
        h["cursor"] = len(h["entries"]) - 1
        if len(h["entries"]) > MAX_ENTRIES:
            drop = len(h["entries"]) - MAX_ENTRIES
            h["entries"] = h["entries"][drop:]; h["cursor"] -= drop
        return True

    def record(self, recipe, note):
        """One user change given as a whole recipe. Returns False (nothing recorded) when the recipe did not change.
        Prefer edit(): a recipe read before another editor's change would overwrite that change."""
        recipe = normalise(recipe)
        with self.locked():
            h = self._read()
            if not self._append(h, recipe, note): return False
            self._write(h)
        return True

    def edit(self, fn):
        """One user change as a read-modify-write under the lock: fn(recipe) changes the current recipe in place (it may
        raise to refuse) and returns (note, result). -> (result, changed)."""
        with self.locked():
            h = self._read(); r = normalise(h["entries"][h["cursor"]]["recipe"])
            note, result = fn(r)
            r = normalise(r)
            changed = self._append(h, r, note)
            if changed: self._write(h)
        return result, changed

    def move(self, delta):
        with self.locked():
            h = self._read(); c = h["cursor"] + delta
            if c < 0 or c >= len(h["entries"]): return None
            note = h["entries"][h["cursor"]]["note"] if delta < 0 else h["entries"][c]["note"]
            h["cursor"] = c; self._write(h)
            return note

    def undo(self): return self.move(-1)

    def redo(self): return self.move(+1)

    def entries(self):
        h = self._read(); return h["entries"], h["cursor"]
