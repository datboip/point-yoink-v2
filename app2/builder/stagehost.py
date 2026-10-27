# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step's binding to the staged engine (scanner/stages).

  host = StageHost(project_root, emit)         # one per open project; opens stages/ and ONE in-process Engine
  host.status()                                # coalesced: at most one in flight, one more queued
  t = host.call("set", {"fused.method": "points"})   # any Engine method by name, or a helper below (h_*)
  host.cancel()                                # stops the call running now and drops the queued ones
  host.close()

All Engine calls run on ONE thread ("stages-edit"), in order: the Engine is not thread-safe (_begin, _run, _ev, the
runner's pgid set). The GUI thread only posts (call() takes well under 5 ms) and receives through emit:
  stages_status   Snapshot (latest wins)
  stages_result   {"ticket", "name", "ok", "value", "error", "secs"} (queued, in ticket order)
  stages_busy     {"name", "since"} or None (latest wins)
Runs do NOT go through here: they go through the scan session's BuildRunner with the StagesRun backend below (a
`cli.py run` subprocess with its own Engine and Runner), so a cancel of a run never kills a preview, a crash in a run
never takes the window, and the Scan step's Build button and this page share one lock-free path (session.build).

Helpers (h_*) are what the page needs that the engine does not offer as one call (see the spec's API requests): plan
counts, a detect with its mask read and its files removed, a positions-mask delete with its count checked, the
One-tap summary of a preset, several undos in one go, the Line-up numbers."""
import collections, copy, json, os, shutil, sys, threading, time, traceback, uuid
import numpy as np
from app2 import SCANNER

MUTATING = {"set", "reset", "preset", "add_op", "drop_op", "toggle_op", "set_op", "clear_ops", "leave_out", "include",
            "flip", "match", "turn", "nudge", "platter", "undo", "redo", "delete", "adopt_legacy", "pin", "h_mask_delete", "h_history_walk",
            "h_platter_plane", "h_apply", "h_apply_custom", "h_finish", "h_toggle_many"}
MASK_RADIUS = 0.001        # mm: the mask points ARE the selected element positions, so only they match (0.02 took duplicates)


class Snapshot:
    """What the page shows, made on the edit thread, never changed after. groups: [{name, label, nodes: [NodeInfo as
    dict]}]; files: {step id: {output name: path}} of every built step; chains: {list token: [(list key, index, op)]};
    lists / finals: the graph's; sides: [{id, n, active}]; fused_kind: points | mesh; methods: {"fused": [(name, label)]}."""
    __slots__ = ("groups", "est_s", "entries", "cursor", "running", "confirm", "recipe", "files", "chains", "parked", "lists",
                 "finals", "sides", "fused_kind", "fused_method", "meshed_method", "methods", "all_stale", "t", "secs", "error")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))

    def node(self, nid):
        for g in self.groups or ():
            for n in g["nodes"]:
                if n["id"] == nid: return n
        return None

    def group(self, name):
        return next((g for g in self.groups or () if g["name"] == name), None)

    def out(self, step, kind=None):
        f = (self.files or {}).get(step) or {}
        if kind: return f.get(kind)
        return next((f[k] for k in ("mesh", "points", "lineup", "placement", "transforms", "pairs") if k in f), None)

    def final(self, token): return (self.finals or {}).get(token, token)

    def end_of(self, token):
        """The step a new op on this list token edits (the list's end) and the file shown for it."""
        return self.final(token)


def _newest_meta(e, st):
    """The meta of the most recent stored result of this step under any earlier key (with "_dir"), or None."""
    try:
        sd = os.path.join(e.store.nodes, st.store)
        keys = [k for k in os.listdir(sd) if os.path.isfile(os.path.join(sd, k, "meta.json"))]
        if not keys: return None
        k = max(keys, key=lambda x: os.path.getmtime(os.path.join(sd, x, "meta.json")))
        with open(os.path.join(sd, k, "meta.json")) as f: meta = json.load(f)
        meta["_dir"] = os.path.join(sd, k); return meta
    except (OSError, ValueError, KeyError, TypeError, AttributeError): return None


def _newest_stored(e, st, meta=None):
    """{output name: path} of the most recent stored result of this step under any earlier key, or None."""
    try:
        meta = meta or _newest_meta(e, st)
        if not meta: return None
        files = {n: os.path.join(meta["_dir"], o["file"]) for n, o in (meta.get("outputs") or {}).items()}
        return files if files and all(os.path.exists(v) for v in files.values()) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError): return None


def elsewhere_words(stages_dir):
    """What the other build is doing, for the page that is locked out of its tools: who started it, when, and the newest progress line of
    its run (the run's events file), e.g. "Built from the terminal since 03:52: Fused, frame 810 of 1800 (56%)". None when no lock."""
    import glob, time
    j = running_elsewhere(stages_dir)
    if not j: return None
    cmd = " ".join(str(x) for x in (j.get("cmd") or [])) if isinstance(j.get("cmd"), list) else str(j.get("cmd") or "")
    who = "YOLO Yoink or Build in another window" if "app2" in cmd else "the terminal" if "cli" in cmd else "another program"
    try: since = time.strftime("%H:%M", time.localtime(float(j.get("started") or 0)))
    except (TypeError, ValueError): since = ""
    line = ""
    try:
        runs = sorted(glob.glob(os.path.join(stages_dir, "runs", "*.jsonl")), key=os.path.getmtime)
        if runs:
            with open(runs[-1], "rb") as f:
                f.seek(max(0, os.path.getsize(runs[-1]) - 20000)); tail = f.read().decode("utf-8", "replace").splitlines()
            for ln in reversed(tail):
                try: ev = json.loads(ln)
                except ValueError: continue
                if ev.get("event") == "progress":
                    line = "%s, %s (%d%%)" % (step_words(ev), ev.get("text") or "", round(100 * float(ev.get("overall") or 0))); break
                if ev.get("event") == "node" and ev.get("state") == "start": line = step_words(ev) + " started"; break
    except OSError: pass
    return "Built from %s%s%s" % (who, (" since " + since) if since else "", (": " + line) if line else "")


def running_elsewhere(stages_dir):
    """The run lock's holder ({pid, cmd, started}) while a run is going on (ours through BuildRunner, or a CLI), else
    None. Engine.status() does not say (API request R12); lock.json exists only while the lock is held."""
    try:
        with open(os.path.join(stages_dir, "lock.json")) as f: j = json.load(f)
        os.kill(int(j["pid"]), 0); return j
    except Exception: return None


def _plain(v):
    """Results cross threads as plain data (RunResult / NodeInfo dataclasses -> dicts)."""
    if hasattr(v, "__dataclass_fields__"): return dict(v.__dict__)
    return v


class StageHost:
    def __init__(self, root, emit, mem_gb=None, engine_factory=None):
        self.root = root; self._emit_to = emit; self.mem_gb = mem_gb; self._factory = engine_factory
        self.q = collections.deque(); self.cv = threading.Condition(); self.e = None; self.p = None
        self._status_queued = False; self._current = None; self._stop = False; self._ticket = 0
        self.opened = threading.Event(); self.open_error = None; self.adopted = None
        self.calls = []                                  # names run, in order (tests)
        self.thread = threading.Thread(target=self._loop, name="stages-edit", daemon=True); self.thread.start()

    # ---------------- GUI side (never blocks) ----------------
    def call(self, name, *a, **kw):
        with self.cv:
            self._ticket += 1; t = self._ticket
            self.q.append((t, name, a, kw)); self.cv.notify()
        return t

    def status(self):
        """Asked after a page show, a run's node event, a file change of history.json (a CLI edit), the slow poll. Queued
        once at most: ten asks while one runs make one more status, not ten."""
        with self.cv:
            if self._status_queued: return
            self._status_queued = True; self.q.append((0, "_status", (), {})); self.cv.notify()

    @property
    def busy(self): return self._current is not None or any(x[1] != "_status" for x in list(self.q))

    def cancel(self):
        """Stops the call running now (a detect, delete, preview) and drops the queued ones (asked before the cancel).
        Engine.cancel() sleeps 1.5 s between SIGTERM and SIGKILL, so it runs on its own short thread."""
        with self.cv:
            dropped = [x for x in self.q if x[1] != "_status"]; self.q = collections.deque(x for x in self.q if x[1] == "_status")
        for t, name, _a, _k in dropped:
            self.emit("stages_result", {"ticket": t, "name": name, "ok": False, "value": None, "error": "cancelled", "secs": 0})
        if self.e is not None and self._current is not None:
            threading.Thread(target=self.e.cancel, name="stages-cancel", daemon=True).start()

    def emit(self, kind, payload):
        """Nothing leaves a closed host (a call still running when it was dropped finishes silently)."""
        if not self._stop: self._emit_to(kind, payload)

    def close(self, timeout=5.0):
        """Stops the loop; timeout 0 = do not wait (the thread is a daemon; the page drops what it would still say)."""
        self.cancel()
        with self.cv: self._stop = True; self.cv.notify()
        if timeout: self.thread.join(timeout)

    # ---------------- edit thread ----------------
    def _open(self):
        from stages.project import Project
        from stages.engine import Engine
        if self._factory is not None:
            self.e = self._factory(self.root); self.p = getattr(self.e, "p", None); return
        self.p = Project.open(self.root, create=True)
        self.e = Engine(self.p, mem_gb=self.mem_gb)
        if self.p.created and os.path.isdir(self.p.build):     # what cli.py does on a first open (Engine() does not: R19)
            try: self.adopted = _plain(self.e.adopt_legacy())
            except Exception as ex: self.adopted = {"error": str(ex)}

    def _loop(self):
        try: self._open()
        except Exception as ex:
            self.open_error = "%s: %s" % (type(ex).__name__, ex)
            self.emit("stages_result", {"ticket": 0, "name": "open", "ok": False, "value": None, "error": self.open_error, "secs": 0})
            self.opened.set(); return
        self.opened.set(); self._do_status()
        while True:
            with self.cv:
                while not self.q and not self._stop: self.cv.wait(0.5)
                if self._stop: return
                t, name, a, kw = self.q.popleft()
                if name == "_status": self._status_queued = False
            try:
                if name == "_status": self._do_status(); continue
                changed = self._run_one(t, name, a, kw)
                if changed and not self._stop:
                    with self.cv:                          # the next status comes from here; a queued one is not needed
                        self.q = collections.deque(x for x in self.q if x[1] != "_status"); self._status_queued = False
                    self._do_status()
            except Exception:                              # an emit or a result that would not convert: the loop lives on
                traceback.print_exc(); self._current = None

    def _run_one(self, t, name, a, kw):
        self._current = name; t0 = time.time(); self.calls.append(name)
        self.emit("stages_busy", {"name": name, "since": t0})
        ok, val, err = True, None, None
        try:
            fn = getattr(self.e, name) if hasattr(self.e, name) else getattr(self, name)     # a fake engine may answer h_*
            val = fn(*a, **kw)
        except Exception as ex:
            ok = False; kind = type(ex).__name__
            err = "cancelled" if kind == "Cancelled" else "%s" % (ex.args[0] if ex.args else ex)
            if kind not in ("Cancelled", "ValueError", "KeyError", "ReadOnly", "Locked", "DeniedError"):
                err = "%s\n%s" % (err, traceback.format_exc(limit=4))
        finally:
            self._current = None; self.emit("stages_busy", None)
        try: v = _plain(val)
        except Exception as ex: ok, v, err = False, None, "%s: %s" % (type(ex).__name__, ex)
        self.emit("stages_result", {"ticket": t, "name": name, "ok": ok, "value": v, "error": err, "secs": round(time.time() - t0, 3)})
        return name in MUTATING               # a failed change may have half-applied (h_apply, a history walk): status after it too

    def _do_status(self):
        t0 = time.time(); stop = hist = None
        if os.environ.get("POINTYOINK_TRACE") == "2":           # developer: where a slow snapshot spends its time (printed when over 3 s)
            import collections; me = threading.get_ident(); hist = collections.Counter(); stop = threading.Event()
            def sample():                                       # this thread's stack four frames deep, four times a second
                while not stop.wait(0.25):
                    f = sys._current_frames().get(me); st = []
                    while f is not None and len(st) < 5: st.append("%s:%d" % (os.path.basename(f.f_code.co_filename), f.f_lineno)); f = f.f_back
                    hist[" < ".join(st)] += 1
            threading.Thread(target=sample, name="status-sampler", daemon=True).start()
        try: snap = self.e.snapshot() if hasattr(self.e, "snapshot") else make_snapshot(self.e, self.p)   # a fake engine
        except Exception as ex:
            snap = Snapshot(error="%s: %s" % (type(ex).__name__, ex), t=time.time())
            traceback.print_exc()
        snap.secs = round(time.time() - t0, 3); snap.t = time.time()
        if stop is not None:
            stop.set()
            if snap.secs > 3: print("trace slow status %.1fs\n" % snap.secs + "\n".join("%4d  %s" % (n, k) for k, n in hist.most_common(8)), flush=True)
        self.emit("stages_status", snap)

    # ================================================================ helpers (edit thread)
    def h_fill_preview(self, path, params):
        return self.e.job("tools.common:tool_job", {"in": path, "params": params, "preview_fill": True}, mem_gb=3)

    def h_build_memory(self, to=None, force=(), from_=None, with_params=None):
        from app2.builder.build_memory import planned_memory
        return planned_memory(self.e,to,force,from_,with_params)

    def h_plan(self, to=None, force=(), preset=None, recipe=None):
        """Counts only (R20): {to_run, est_s, stale: [ids], waiting: [ids], cached}. preset: what that preset would run
        (nothing is committed)."""
        from stages import params as PR
        if preset: recipe = PR.apply_preset(self.e.recipe(), preset)
        g, info, to_run, est = self.e.plan(to=to, force=force, recipe=recipe)
        return {"to_run": len(to_run), "est_s": round(est), "ids": [st.id for st in to_run],
                "stale": [i for i, v in info.items() if v["status"] == "stale"],
                "waiting": [i for i, v in info.items() if v["status"] == "waiting"],
                "cached": sum(1 for v in info.values() if v["status"] in ("fresh", "adopted", "warn", "check") and not v.get("retry"))}

    def _toggle_recipe(self, recipe, list_token, changes):
        """Validate the whole batch before changing any operation in this recipe."""
        if not isinstance(changes, dict) or any(not isinstance(key, str) or type(on) is not bool for key, on in changes.items()):
            raise ValueError('Choose on or off for each applied edit.')
        chain = self.e.chain_ops(list_token, recipe)
        by_id = {operation.get('id'): (key, index, operation) for key, index, operation in chain}
        missing = set(changes) - set(by_id)
        if missing:
            raise ValueError('Applied edits changed. Refresh and try again: ' + ', '.join(sorted(missing)))
        selected = [(by_id[key], on) for key, on in changes.items()
                    if by_id[key][2].get('on', True) != on]
        # Default tool lists acquire persistent ids when first edited. Resolve all old ids first.
        for (key, index, _operation), on in selected:
            self.e._ops(recipe, key)[index]['on'] = on
        return len(selected)

    def h_toggle_many(self, list_token, changes):
        """Set checkbox states atomically, with one Undo entry for the entire batch."""
        def edit(recipe):
            count = self._toggle_recipe(recipe, list_token, changes)
            return 'Applied edits: %d states changed' % count, {'changed': count, 'states': dict(changes)}
        return self.e._edit(edit)[0]

    def h_toggle_plan(self, list_token, changes, to=None):
        """Plan the exact proposed checkbox combination without writing its recipe."""
        recipe = copy.deepcopy(self.e.recipe())
        changed = self._toggle_recipe(recipe, list_token, changes)
        result = self.h_plan(to=to, recipe=recipe)
        result['changed'] = changed
        return result

    def h_finish(self, params):
        """Adjust the last enabled finish; retain all earlier edits."""
        chain = self.e.chain_ops("meshed")
        if chain and chain[-1][2].get("tool") == "finish" and chain[-1][2].get("on", True):
            return self.e.set_op("meshed", chain[-1][2]["id"], params)
        return self.e.add_op("meshed", {"tool": "finish", "params": params})

    def h_custom_onetap(self, fragment, preset):
        from app2.builder.preset_store import validate_fragment
        return self.h_onetap(preset, validate_fragment(fragment))

    def h_apply_custom(self, fragment, preset):
        from app2.builder.preset_store import validate_fragment
        from stages import params as PR
        fragment = validate_fragment(fragment)
        def edit(r):
            new = PR.apply_preset(r, preset, fragment)
            r.clear(); r.update(new)
            return "Custom preset: " + preset, preset
        return self.e._edit(edit)[0]

    def h_ultra_plan(self, presets):
        """Ultra Yoink's warning: per recipe, the steps it would run, the time, and about how much disk its results add
        (what this project's last Fused / Meshed / Coverage results take; 25 MB a side when nothing is built yet)."""
        import glob
        nodes = os.path.join(self.e.p.stages, "nodes"); per = 0
        for group in ("fused", "meshed", "report"):
            dirs = [d for d in glob.glob(os.path.join(nodes, group + "*", "*")) if os.path.isdir(d)]
            by = {}
            for d in dirs: by.setdefault(os.path.basename(os.path.dirname(d)), []).append(d)
            for ds in by.values():
                newest = max(ds, key=os.path.getmtime)
                per += sum(os.path.getsize(os.path.join(r, f)) for r, _x, fs in os.walk(newest) for f in fs)
        if per == 0: per = 25 * 2**20 * max(1, len(self.e.p.sides()))
        out = []
        for name in presets:
            o = self.h_onetap(name); o["est_bytes"] = per if o["to_run"] else 0; out.append(o)
        return out

    def h_onetap(self, preset, fragment=None):
        """Summarize the proposed recipe without saving it."""
        from stages import params as PR
        r0 = self.e.recipe(); r = PR.apply_preset(r0, preset, fragment); g = self.e.graph(r)
        f = g.by_id["fused"]; m = g.by_id["meshed"]
        fname = {"tsdf": "Standard", "points": "Points"}.get(f.method.name, f.method.label or f.method.name)
        fusion = "%s %s mm" % (fname, PR.fmt(f.params["point_distance"]))
        ft = [g.by_id[i] for i in g.chain("fused")]
        ftools = ", ".join(tool_words(o.method.name, o.params, f.produces) for o in ft if not o.alias and o.method.name != "delete") or "none"
        if m.method.name == "poisson":
            try: ext = self.e._part_extent(g)                 # R6: no public part extent
            except Exception: ext = None
            q = m.params["quality"]
            grid = ("(about %.2f mm detail)" % PR.poisson_plan(ext, q)["grid_mm"]) if ext else "(detail known once Fused is built)"
            mesh = "Watertight %.1f %s, close small holes %s" % (q, grid, "on" if m.params["hole_fill"] else "off")
        else:
            mesh = {"as_fused": "As scanned", "bpa": "Joined points"}.get(m.method.name, m.method.label or m.method.name)
        mt = [g.by_id[i] for i in g.chain("meshed")]
        mtools = ", ".join(tool_words(o.method.name, o.params, "mesh") for o in mt if not o.alias and o.method.name != "delete") or "none"
        deletes = sum(1 for lk, ops in (r.get("ops") or {}).items() for o in ops if o.get("tool") == "delete" and o.get("on", True))
        pl = self.h_plan(recipe=r)
        return {"preset": preset, "fusion": fusion, "ftools": ftools, "mesh": mesh, "mtools": mtools, "deletes": deletes,
                "to_run": pl["to_run"], "est_s": pl["est_s"], "note": PR.PRESETS[preset]["note"] if fragment is None else "Your build settings."}

    def h_export_stl(self, src, dst, orient=None):
        """Export STL: src (a mesh PLY) -> dst as a binary STL (app2/builder/export_stl.py), here and not on the GUI thread.
        orient: the 3x3 rotation chosen on the Orient block (the model is turned and set on z = 0 first)."""
        from app2.builder.export_stl import write_stl
        tmp = None
        try:
            if orient is not None:
                from app2.builder.export_model import orient_ply
                src = tmp = orient_ply(src, orient)
            return write_stl(src, dst)
        finally:
            if tmp and os.path.exists(tmp): os.remove(tmp)

    def h_export(self, src, dst, fmt="stl", zip_root=None, orient=None):
        """Export: src (a mesh PLY) -> dst as STL / OBJ / PLY / 3MF, or a ZIP with the project's side folders
        (zip_root = the project) (app2/builder/export_model.py). orient as in h_export_stl."""
        from app2.builder.export_model import export
        tmp = None
        try:
            if orient is not None:
                from app2.builder.export_model import orient_ply
                src = tmp = orient_ply(src, orient)
            return export(src, dst, fmt, zip_root)
        finally:
            if tmp and os.path.exists(tmp): os.remove(tmp)

    def h_side_agreement(self):
        """How well each placed side agrees with the others, from the line-up as built: {"sides": {n: {id, alone, agree}}, "bad": [n]}."""
        import json as _j
        from side_agreement import agreement, flag_bad
        self.e._begin(); p = self.e.output("lineup.adjust", kind="lineup")
        sp = self.e.output("lineup.adjust", kind="sides") or (os.path.join(os.path.dirname(p), "sides.json") if p else None)
        if not sp or not os.path.exists(sp): return {"sides": {}, "bad": []}
        with open(sp) as f: J = _j.load(f)
        sides = J["sides"]; res = agreement(sides, self.p.root)
        ids = {s["n"]: s["id"] for s in sides}; base = next((s["n"] for s in sides if s["id"] == J.get("base")), None)
        return {"sides": {str(n): dict(r, id=ids.get(n)) for n, r in res.items()}, "bad": [ids[n] for n in flag_bad(res, base)
                if n in ids], "bad_n": flag_bad(res, base)}

    def h_scan_guide(self, src, mode='turntable'):
        from app2.builder.scan_guide import guide
        if os.path.realpath(src)==os.path.realpath(os.path.join(self.p.root,"model.ply")):
            return guide(self.p.root,src,mode)
        self.e._begin()
        src_lineup=self.e.output("lineup.adjust",kind="lineup")
        sides=self.e.output("lineup.adjust",kind="sides") or (os.path.join(os.path.dirname(src_lineup),"sides.json") if src_lineup else "")
        return guide(self.p.root,src,mode,lineup_file=sides)

    def h_capture_distance(self,src,side=None):
        from app2.builder.capture_distance import colors
        from app2.builder.scan_guide import selected_lineup
        self.e._begin();sp=self.e.output('lineup.adjust',kind='sides')
        rows,_=selected_lineup(self.p.root,sp) if sp else ([],None)
        placements={row['id']:np.asarray(row['T']) for row in rows} if rows else None
        C,counts=colors(self.p.root,src,placements,side)
        import hashlib
        key=hashlib.sha256(C.tobytes()).hexdigest();out=os.path.join(self.p.root,'build','capture-distance-'+key+'.npy');os.makedirs(os.path.dirname(out),exist_ok=True);np.save(out,C)
        return dict(rgb=out,counts=counts)

    def h_quality(self, src):
        """How well each spot of the shown result was scanned (frames that saw it): a color per vertex or point, saved next to the
        project's build files and returned as {"rgb": path, "share": [red, yellow, blue]}. Measured by scanner/frame_quality.py in its own
        process (about 30 s); the same result file reuses the saved colors."""
        import hashlib, json as _j, subprocess, sys
        self.e._begin(); p = self.e.output("lineup.adjust", kind="lineup")
        sp = self.e.output("lineup.adjust", kind="sides") or (os.path.join(os.path.dirname(p), "sides.json") if p else None)
        if not sp or not os.path.exists(sp): raise RuntimeError("Line up the sides first: the quality view reads their frames.")
        from app2.builder.quality_cache import identity
        key = identity(src, sp)
        out = os.path.join(self.p.root, "build", "quality-%s.npy" % key)
        if not os.path.exists(out):
            self.e._begin(); p = self.e.output("lineup.adjust", kind="lineup")
            sp = self.e.output("lineup.adjust", kind="sides") or (os.path.join(os.path.dirname(p), "sides.json") if p else None)
            if not sp or not os.path.exists(sp): raise RuntimeError("Line up the sides first: the quality view reads their frames.")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(here, "scanner"), here]))
            tmp = out + ".part.npy"
            r = subprocess.run(["nice", "-n", "10", sys.executable, os.path.join(here, "scanner", "frame_quality.py"), self.p.root, sp, src,
                                "--out", out + ".q.npz", "--rgb", tmp], capture_output=True, text=True, env=env)
            if r.returncode or not os.path.exists(tmp): raise RuntimeError((r.stderr or r.stdout or "quality map failed").strip().splitlines()[-1][:200])
            os.replace(tmp, out)
        c = np.load(out, mmap_mode="r"); n = max(1, len(c))
        return {"rgb": out, "share": [round(float((c[::7] == p).all(1).mean()), 3) for p in ([255, 70, 50], [255, 200, 40], [90, 170, 255])], "n": int(n)}

    def h_model_dims(self, src):
        """The model's size in mm ({"w", "d", "h", "text"}), from its vertices only."""
        from app2.builder.export_model import model_dims
        return model_dims(src)

    def h_detect(self, list_token, op):
        """Engine.detect, then the mask read into a bool array (len n) and the detect files removed (R16). "in" = the
        file the mask indexes (R4: the list end _parent_for_new_op uses), "png" the picture as an RGB array."""
        res = self.e.detect(list_token, op)
        lk = self.e.list_key(list_token)
        try:
            _g, pid, parent, _outs = self.e._parent_for_new_op(lk)
            fk = "points" if "points" in parent.files else "mesh"; inp = parent.files[fk]
        except Exception: pid, inp = None, None
        return dict(res, **_read_detect(res, remove=True), step=pid, input=inp, pieces=pieces_gone(res.get("stats")))

    def h_side_detect(self, side, rate_pct):
        """Isolation Detect on a side (R2): the isolation tool in detect mode on the side's platter-cut mesh (what
        side.keep's min_share keeps pieces of). count / mask = EVERY piece under rate % of the biggest (on that mesh),
        whatever the side's rate is now; back = what the side's rate removes now and rate keeps. view_*: the same rate
        on the finished side (the list end the view shows, side.ply): only what a Delete would still take (the rest
        the side's rate already removed; removing whole pieces leaves the biggest, so the share is the same)."""
        from stages.keys import jload  # noqa: F401  (keeps the engine's import order)
        sid = self.p.resolve_side(side).id if self.p is not None else side
        inp = self.e.output("side.platter@" + sid, kind="mesh")
        if not inp: raise ValueError("side.platter@%s is not built yet: run first" % sid)

        def detect(rate, path=inp):
            r = self.e.job("tools.common:tool_job", {"tool": "isolation", "kind": "mesh", "in": path, "params": {"rate": float(rate)},
                                                     "op": {"tool": "isolation"}, "parent_digest": "", "stages": self.e.p.stages,
                                                     "detect": True, "T_now": None}, mem_gb=4)
            try: return dict(count=r.get("count"), n=r.get("n"), pieces=pieces_gone(r.get("stats")), **_read_detect(r, remove=False))
            finally: shutil.rmtree(r.get("_dir", ""), ignore_errors=True)
        out = detect(rate_pct); out["rate"] = float(rate_pct)
        try:
            _g, _pid, parent, _o = self.e._parent_for_new_op("side.keep@" + sid); vin = parent.files.get("mesh")
        except Exception: vin = None
        if vin and os.path.exists(vin):
            v = detect(rate_pct, vin)
            out.update(view_input=vin, view_mask=v.get("mask"), view_count=int(v.get("count") or 0), view_n=v.get("n"),
                       view_pieces=int(v.get("pieces") or 0))
        # was: the mask minus what the side's rate removes now, so any rate at or over it (3%) found nothing (R2 bug)
        try: now = 100.0 * float(self.e.graph(self.e.recipe()).by_id["side.keep@" + sid].params.get("min_share") or 0)
        except Exception: now = 0.0
        out["now_pct"] = now; out["back"] = 0
        if now > float(rate_pct) + 1e-9 and out.get("mask") is not None:       # lower than now: some pieces come back
            m0 = detect(now).get("mask")
            if m0 is not None and len(m0) == len(out["mask"]): out["back"] = int((m0 & ~out["mask"]).sum())
        return dict(out, step="side.platter@" + sid, input=inp)

    def h_mask_delete(self, node, positions, want):
        """A Rectangle / Lasso delete: the selected elements' own positions as a point PLY, delete as a mask with radius
        0.001 mm (only they match), then the count checked against the selection (R14: no delete by indices)."""
        from stages import plyio
        d = os.path.join(self.e.p.stages, "tmp"); os.makedirs(d, exist_ok=True)
        ply = os.path.join(d, "mask_%s.ply" % uuid.uuid4().hex[:8])
        P = np.asarray(positions, np.float32)
        plyio.write_points_ply(ply, P, np.zeros_like(P))
        try: det = self.e.delete(node, {"shape": "mask"}, mask_pts=ply, radius=MASK_RADIUS)
        finally:
            try: os.remove(ply)
            except OSError: pass
        _read_detect(det, remove=True, want_mask=False)
        return {"id": det.get("id"), "got": det.get("count"), "want": int(want), "n": det.get("n")}

    def h_history_walk(self, steps):
        """History click: undo (steps < 0) or redo (steps > 0) as many times as needed, one status after."""
        notes = []
        for _ in range(abs(int(steps))):
            r = self.e.undo() if steps < 0 else self.e.redo()
            if r is None: break
            notes.append(r)
        return notes

    def h_lineup(self):
        """Line-up numbers: per side T (model frame), placement score, base, agreement before / after refining."""
        from stages.keys import jload
        out = {"T": {}, "order": [], "base": None, "score": {}, "agree": None, "meshes": {}}
        _st, o = self.e._resolve("lineup.adjust")                   # one resolve (a graph build) for both files, not two
        lp = o.files.get("lineup") if o is not None else None; sp = o.files.get("sides") if o is not None else None
        if not lp:                                                  # mid-rebuild or stale: the last line-up still places the sides for a look
            st = self.e.graph().by_id.get("lineup.adjust"); old = _newest_stored(self.e, st) if st is not None else None
            if old: lp = old.get("lineup"); sp = sp or old.get("sides"); out["stale"] = True
        if lp:
            j = jload(lp); out.update(T=j.get("T", {}), order=j.get("order", []), base=j.get("base"))
        if sp:
            for s in jload(sp).get("sides", []):
                out["meshes"][s["id"]] = s.get("mesh") and os.path.join(self.root, s["mesh"])
        return out

    def h_view_file(self, node, color="shaded", key=None):
        """The decimated view.ply of a node (made when missing), without starting the viewer (R13)."""
        cmd = self.e.view(node, color=color, key=key, dry_run=True)
        return cmd[2] if len(cmd) > 2 else None

    def h_platter_plane(self, side, plane):
        """Manual platter plane {n, d, margin, keep_above} for a side: written to stages/tmp as the engine wants a file."""
        d = os.path.join(self.e.p.stages, "tmp"); os.makedirs(d, exist_ok=True)
        pp = os.path.join(d, "plane_%s.json" % uuid.uuid4().hex[:8])
        with open(pp, "w") as f: json.dump(plane, f)
        try: return self.e.platter(side, pp)
        finally:
            try: os.remove(pp)
            except OSError: pass

    def h_apply(self, calls):
        """Several engine calls in a row (Line-up's "Apply 3 changes"): one entry each (R3), one status after. All or
        nothing: when one is refused, the ones already made are undone (their entries stay only as redo)."""
        out = []
        try:
            for name, a, kw in calls: out.append(_plain(getattr(self.e, name)(*a, **(kw or {}))))
        except Exception:
            for _ in out:
                try: self.e.undo()
                except Exception: break
            raise
        return out


def pieces_gone(st):
    """Pieces (mesh) or clusters (points) an isolation detect would drop, from its stats; None for other tools."""
    st = st or {}
    for a, b in (("pieces", "kept_pieces"), ("clusters", "kept_clusters")):
        if a in st and b in st: return max(0, int(st[a]) - int(st[b]))
    return None


def _read_detect(res, remove=True, want_mask=True):
    """{mask: bool array or None, img: RGB array or None} from a detect result; the files go when remove."""
    out = {"mask": None, "img": None}
    m, png, n = res.get("mask"), res.get("png"), res.get("n")
    try:
        if want_mask and m and os.path.exists(m):
            a = np.unpackbits(np.load(m)); out["mask"] = a[:n].astype(bool) if n is not None else a.astype(bool)
        if want_mask and png and os.path.exists(png):
            from PIL import Image
            out["img"] = np.asarray(Image.open(png).convert("RGB"))
    finally:
        if remove:
            for f in (m, png):
                if f:
                    try: os.remove(f)
                    except OSError: pass
    return out


def tool_words(name, params, kind):
    """A tool with its numbers: "Isolation 1%", "Overlap merge 0.21 mm", "Smooth 10 x2"."""
    from stages import params as PR
    p = dict(params or {})
    try: p = PR.resolve_tool(name, kind, p)            # the defaults of what was not given
    except Exception: pass
    if name == "isolation": return "Isolation %s%%" % PR.fmt(float(p.get("rate", 0)))
    if name == "overlap": return "Overlap merge %.2f mm" % float(p.get("distance", 0))
    if name == "smooth":
        s = PR.fmt(float(p.get("strength", 0)))
        return ("Smooth %s%% x%d" if kind == "mesh" else "Smooth %s x%d") % (s, int(p.get("times", 1)))
    if name == "simplify": return "Simplify %s%%" % PR.fmt(float(p.get("keep_pct", 0)))
    if name == "mirror": return "Mirror (gap %g mm)" % float(p.get("gap_mm", 1.5))
    if name == "clone": return "Clone (%s %g mm)" % (p.get("shape", "cylinder"), float(p.get("width" if p.get("shape") == "box" else "diameter", 0)))
    if name == "local_fix": return "Smooth area (%s %g mm)" % (p.get("shape", "cylinder"), float(p.get("width" if p.get("shape") == "box" else "diameter", 0)))
    if name == "cut_plane": return "Cut base (below %.1f mm)" % float(p.get("d", 0))
    if name == "match_cleaned_sides": return "Match the cleaned sides %s mm" % PR.fmt(float(p.get("distance_mm", 0)))
    if name == "fill_holes":
        h = float(p.get("hole_mm", 0)); return "Fill holes %s" % ("Auto" if not h else "%s mm" % PR.fmt(h))
    if name == "slivers": return "Remove slivers %s" % PR.fmt(float(p.get("factor", 0)))
    if name == "finish":
        from stages.tools.finish import faces_text
        return "Finish for print · %s triangles · smooth %d" % (faces_text(p.get("faces",0)),int(p.get("smooth",0)))
    if name == "delete": return "Delete"
    return name


def make_snapshot(e, p):
    """Status, history, recipe, output paths and op chains in one go (edit thread)."""
    from stages import params as PR
    from stages.methods import get_method
    groups, g, est = e.status()
    entries, cursor = e.history()
    r = e.recipe()
    try: confirm = e.confirm_line(r)
    except Exception: confirm = None
    files = {}
    try:
        outs, _info, _ = e._walk(g, "plan", None)     # output paths of every built step (status gives names only: R24)
        for sid, o in outs.items():
            if o is not None: files[sid] = dict(o.files or {})
    except Exception: pass
    last = {}                                # a stale step still has its last result on disk: show that (marked out of date) instead of nothing
    for st in g.steps:
        if st.id not in files and not st.alias:
            meta = _newest_meta(e, st)
            if meta: last[st.id] = meta
            old = _newest_stored(e, st, meta)
            if old: files[st.id] = old
    chains = {}
    for tok in ["fused", "meshed", "lineup.fix", "lineup.adjust"] + ["side@" + s.id for s in g.sides]:
        try: chains[tok] = [(lk, j, copy.deepcopy(o)) for lk, j, o in e.chain_ops(tok, r)]
        except Exception: chains[tok] = []
    try: parked = [(lk, j, copy.deepcopy(o)) for lk, j, o in e.parked(r)]
    except Exception: parked = []
    methods = {"fused": []}
    for m in PR.METHODS["fused"]:
        try: methods["fused"].append((m, get_method("fused", m).label or m))
        except Exception: pass
    gd = [{"name": gr.name, "label": gr.label, "nodes": [dict(n.__dict__) for n in gr.nodes]} for gr in groups]
    for gr in gd:                            # what the last build found (its stats and warnings) on a step waiting to run again
        for n in gr["nodes"]:
            m = last.get(n["id"])
            n["last_stats"] = (m.get("stats") if m and n.get("stats") is None else None) or None
            n["last_warn"] = (m.get("warn") if m and not n.get("warn") else None) or None
    st = [n["status"] for gr in gd if gr["name"] not in ("raw", "textured") for n in gr["nodes"] if n["status"] not in ("off", "n/a", "parked")]
    fm = g.by_id["fused"]; mm = g.by_id["meshed"]
    return Snapshot(groups=gd, est_s=est, entries=[{"t": x.get("t"), "note": x.get("note")} for x in entries], cursor=cursor,
                    running=running_elsewhere(p.stages) if p is not None else None, confirm=confirm, recipe=copy.deepcopy(r),
                    files=files, chains=chains, parked=parked, lists={k: list(v) for k, v in g.lists.items()}, finals=dict(g.finals),
                    sides=[{"id": s.id, "n": s.n, "active": s.active} for s in g.sides], fused_kind=fm.produces,
                    fused_method=fm.method.name, meshed_method=mm.method.name, methods=methods,
                    all_stale=bool(st) and st.count("stale") >= max(3, len(st) // 3) and not any(s in ("fresh", "adopted", "warn", "check") for s in st))


# ================================================================ runs: a BuildRunner backend
STAGE_WORD = {"side": "Side", "lineup": "Line-up", "fused": "Fused", "meshed": "Meshed", "report": "Coverage", "post": "Previews"}


class StagesRun:
    """cli.py run with the engine's own events (plan, node, progress, published, done, error). Goes through
    session.build(backend), so the session saves every side and frees the GPU first, the Scan page's BuildBox shows the
    same run, and ui_model's rules (no build while recording, no second build) hold."""
    name = "stages"
    cancel_grace = 10.0            # SIGTERM -> the CLI cancels its children (1.5 s + exit); SIGKILL only after this

    def __init__(self, to=None, from_=None, force=(), with_params=None, publish=True, what=None, defer_report=True):
        self.to, self.from_, self.force, self.with_params, self.publish = to, from_, tuple(force or ()), dict(with_params or {}), publish
        self.what = what               # "onetap" / a stage word, for the page's progress card
        self.defer_report = defer_report
        self.label = "Engine: staged build" + (" to %s" % to if to else "")

    def argv(self, project):
        a = [sys.executable, "-u", os.path.join(SCANNER, "stages", "cli.py"), project, "run"]
        if self.to: a += ["--to", self.to]
        if self.from_: a += ["--from", self.from_]
        for f in self.force: a += ["--force", f]
        for k, v in self.with_params.items(): a += ["--with", "%s=%s" % (k, v)]
        if not self.publish: a.append("--no-publish")
        elif self.defer_report: a.append("--defer-report")
        return a

    def exit_text(self, rc):
        return {3: "The project is being built somewhere else.", 2: "That stage or setting is not known to the engine."}.get(rc)

    def done_text(self, prog):
        if self.to and self.to not in ("report", "all"):
            return "%s built%s" % (stage_word(self.to), took(prog))
        return ("Model built: %s%s" % (prog.model or "model.ply", took(prog))) if self.publish else "Model built" + took(prog)

    def parse(self, ln, prog):
        """One output line into the BuildProgress; returns log lines. Also keeps prog.nodes {id: event} and prog.plan for
        the Build page (it reads them from the "build" snapshots)."""
        if not ln.startswith("STAGE "):
            s = ln.strip()
            return [s] if s.startswith(("error", "Traceback", "refused")) else []
        parts = ln.split(" ", 2)
        try: info = json.loads(parts[2])
        except Exception: return []
        return apply_event(parts[1], info, prog)


def stage_word(token):
    head = str(token).split("@")[0].split("#")[0].split(".")[0]
    return STAGE_WORD.get(head, token)


def apply_event(name, info, prog):
    """One engine event into prog. prog.nodes is replaced, never changed in place: the "build" snapshots the window
    reads share it with this (build) thread."""
    out = []
    nodes = dict(getattr(prog, "nodes", None) or {})
    if name == "plan":
        prog.plan = info; prog.frac = max(prog.frac, 0.0)
        prog.text = "Building: %d step%s to run, %d cached, about %s" % (info.get("to_run", 0), "" if info.get("to_run") == 1 else "s",
                                                                      info.get("cached", 0), mins(info.get("est_s", 0)))
        for n in info.get("nodes", []): nodes[n["id"]] = {"id": n["id"], "state": n["status"], "retry": n.get("retry", False)}
    elif name == "node":
        cur = dict(nodes.get(info["id"], {})); cur.update(info)
        if info["state"] == "retry": cur.update(state="warn", retry=True)
        nodes[info["id"]] = cur
        if info["state"] == "start": prog.text = "Building: %s" % step_words(info); prog.current = info["id"]
        if info["state"] in ("failed", "warn", "check", "retry"):
            out.append("%s: %s %s" % (step_words(info), info["state"], (info.get("error") or "; ".join(info.get("warn") or []))[:300]))
    elif name == "progress":
        prog.frac = max(prog.frac, float(info.get("overall", 0) or 0))
        cur = dict(nodes.get(info["id"], {"id": info["id"]})); cur.update(state="start", frac=info.get("frac"), text=info.get("text"))
        nodes[info["id"]] = cur; prog.current = info["id"]
        prog.text = "Building: %s, %s" % (step_words(info), info.get("text") or "%d of %d" % (info.get("done", 0), info.get("total", 1)))
    elif name == "published":
        prog.model = info.get("model"); prog.preview = info.get("needs") or info.get("preview"); prog.preview_needs = bool(info.get("needs"))
        out.append("published %s" % info.get("model"))
    elif name == "done":
        prog.done = info; prog.frac = 1.0
        if info.get("model"): prog.model = info["model"]
    elif name == "error":
        prog.failed = "Build failed: %s" % info.get("msg", ""); out.append("error: %s" % info.get("msg", ""))
    active=[key for key,value in nodes.items() if value.get("state")=="start"]
    if name in ("node","progress") and len(active)>1:
        from app2.builder.progress_words import headline
        prog.text="Building: "+headline(nodes)
    if name=="node" and info.get("state")!="start" and getattr(prog,"current",None)==info.get("id"):
        prog.current=active[0] if active else None
    prog.nodes = nodes
    return out


def step_words(info):
    """side.fuse@<id> -> "Side 2 fuse"; fused.points#op-0003 -> "Fused points tool op-0003"."""
    i = info.get("id", "")
    head = i.split("@")[0].split("#")[0]
    g = STAGE_WORD.get(head.split(".")[0], head)
    if info.get("n"): g += " %d" % info["n"]
    rest = head.split(".", 1)[1] if "." in head else ""
    if "#" in i: rest = (rest + " " if rest else "") + "tool " + i.split("#", 1)[1]
    return (g + " " + rest).strip()


def took(prog):
    """" in 3:12 (est. 4 min)": how long the run really took, next to what it was expected to take."""
    secs = int(prog.elapsed()) if prog is not None else 0
    if secs < 2: return ""                                  # everything cached: nothing worth saying
    est = (getattr(prog, "plan", None) or {}).get("est_s") or 0
    return " in %d:%02d%s" % (secs // 60, secs % 60, (" (est. %s)" % mins(est)) if est >= 5 else "")


def mins(s):
    s = int(s or 0)
    if s < 60: return "%d s" % max(s, 1)
    return "%d min" % max(1, round(s / 60.0))


# ================================================================ element positions (what a delete tests)
def element_positions(path):
    """What the engine's delete tests (tools/edits.elements): points, or face centroids of a mesh, as float32."""
    from stages import plyio
    if path.endswith(".npz"): return plyio.load_points(path)["P"].astype(np.float32)
    V, F, _N = plyio.read_mesh(path)
    return V[F].mean(1).astype(np.float32) if len(F) else np.zeros((0, 3), np.float32)
