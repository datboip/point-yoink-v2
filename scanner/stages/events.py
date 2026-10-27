# SPDX-License-Identifier: AGPL-3.0-or-later
"""Events of a run: new STAGE events (plan, node, progress, published, done, error) and, in legacy mode, exactly the
stream build_project.py prints (its event names, 1-based positions over all sides, and its plain lines), so
pointyoink_scanner.py can read either. New events never carry a "msg" key (the scanner logs any event's msg); only
error does, and done/error are shared by both streams (printed once).
Every event is also appended to stages/runs/<run id>.jsonl; stages/runs/events.jsonl is a copy of the current run."""
import json, os, sys, threading, time

NEW = ("plan", "node", "progress", "published", "done", "error")


class Events:
    def __init__(self, project=None, run_id=None, legacy=False, out=None, on_event=None, quiet_new=False):
        self.legacy_mode = legacy; self.out = out if out is not None else sys.stdout; self.on_event = on_event
        self.quiet_new = quiet_new; self.lock = threading.RLock(); self.run_id = run_id
        self.f = None; self.f2 = None
        if project is not None and run_id is not None:
            rd = os.path.join(project.stages, "runs"); os.makedirs(rd, exist_ok=True)
            self.f = open(os.path.join(rd, "%s.jsonl" % run_id), "a")
            self.f2 = open(os.path.join(rd, "events.jsonl"), "w")

    def _record(self, kind, name, kw):
        rec = {"t": round(time.time(), 3), "kind": kind, "event": name}; rec.update(kw)
        line = json.dumps(rec)
        for f in (self.f, self.f2):
            if f:
                f.write(line + "\n"); f.flush()
        if self.on_event:
            try: self.on_event(kind, name, kw)
            except Exception: pass

    def _print(self, text):
        try:
            self.out.write(text + "\n"); self.out.flush()
        except (BrokenPipeError, ValueError): pass

    def emit(self, name, **kw):
        """A new event (or done/error, shared by both streams)."""
        if name != "error" and "msg" in kw: kw["text"] = kw.pop("msg")
        with self.lock:
            self._record("stage", name, kw)
            if not self.quiet_new or name in ("done", "error"):
                self._print("STAGE %s %s" % (name, json.dumps(kw)))

    def legacy(self, name, **kw):
        """A build_project.py event: printed only in legacy mode, always recorded."""
        with self.lock:
            self._record("legacy", name, kw)
            if self.legacy_mode: self._print("STAGE %s %s" % (name, json.dumps(kw)))

    def line(self, text):
        """A plain line build_project.py prints (pairs: ..., placed side ..., kept ..., needs_scan's summary)."""
        with self.lock:
            self._record("line", "line", {"text": text})
            if self.legacy_mode: self._print(text)

    def say(self, text):
        """A human line for the CLI (not in legacy mode: the scanner's parser must see only the legacy stream)."""
        with self.lock:
            if not self.legacy_mode and not self.quiet_new: self._print(text)

    def close(self):
        for f in (self.f, self.f2):
            try:
                if f: f.close()
            except Exception: pass
