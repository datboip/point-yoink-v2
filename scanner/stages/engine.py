# SPDX-License-Identifier: AGPL-3.0-or-later
"""The staged build engine: plan / run / status / edits / publish / versions / adopt / gc.

Each step's result is kept in the store under a key made from exactly what it was built from (resolved params, method
version, upstream OUTPUT digests, leaf fingerprints). A run walks the steps in order: a stored key is a cache hit, a
static alias passes its input through, anything else runs. Keys use output digests, so a re-run that writes identical
bytes invalidates nothing downstream (early cutoff) and switching a setting back is a cache hit.

A key also carries a digest of the code the step runs (its scripts, its stages modules, the package versions that
matter; methods/__init__.py), computed once per run, so editing a script re-runs exactly the steps that use it. Mesh
outputs are keyed downstream by an order-independent geometry digest (plyio.geo_digest): the same surface written in
another vertex order is still an early cutoff; the byte digest stays the storage and exact-replay identity (Out.bdigest).

A soft-fail step whose failure was not deliberate (killed, crashed, a launch failure, anything unrecognised:
runner.StepError.kind) is stored under its key flagged for retry (meta.transient, meta.retry): the plan treats it as
built (everything below it is fresh, the step shows "warn (will retry)"), and the next run re-executes only that step
(same output: early cutoff below it). After RETRY_LIMIT identical failures with the same inputs it is kept as a normal
soft fail with a warning saying so (a launch failure is always retried). Recipe changes are read-modify-writes under the
history lock (recipe.History.edit). Every entry point runs in its own cancel scope (runner.Runner.scope)."""
import copy, fcntl, functools, hashlib, json, os, shutil, statistics, sys, time, uuid, threading
from dataclasses import dataclass, field
from contextlib import contextmanager

from . import graph as G
from . import params as PR
from .events import Events
from .keys import ENGINE_VERSION, H, DigestMemo, key16, output_digest, file_digest_raw
from .methods import JOB, lib_version
from .project import DeniedError
from .recipe import History, new_recipe, normalise, ReadOnly
from .runner import Runner, StepError, Cancelled
from . import resources
from .store import Store
from . import store as _store
from .keys import jload, jdump

DEFAULT_SECS = {"side.fuse": 25, "side.platter": 4, "side.keep": 2, "lineup.pairs": 3, "lineup.place": 20, "lineup.fix": 5,
                "lineup.refine": 60, "lineup.adjust": 1, "fused.tsdf": 180, "fused.points": 240, "meshed.poisson": 120,
                "meshed.bpa": 60, "op": 10, "report": 15}
GLOBAL_PREVIEW = ("lineup.adjust", "fused", "meshed")


def entry(fn):
    """An engine entry point: runs in its own cancel scope, so a cancel stops what is running when it is made and never
    a call made after it (runner.Runner.scope; nested entry points share the outermost scope)."""
    @functools.wraps(fn)
    def wrapped(self, *a, **kw):
        with self.runner.scope(): return fn(self, *a, **kw)
    return wrapped


def flip_axis_text(op):
    a = op.get("axis", "auto")
    return "auto: %s" % op["chose"] if a == "auto" and op.get("chose") else a


class Locked(Exception):
    def __init__(self, pid): super().__init__("a build is running (pid %s)" % pid); self.pid = pid


class NotLight(Exception): pass


class Out:
    """A step's result as the walk sees it. digest: what downstream keys use (the geometry digest of a mesh, else the
    bytes); bdigest: the bytes (exact delete replay, publish)."""
    __slots__ = ("step", "digest", "bdigest", "files", "kind", "meta", "alias_of")

    def __init__(self, step, digest, files, kind, meta=None, alias_of=None, bdigest=None):
        self.step = step; self.digest = digest; self.files = files; self.kind = kind; self.meta = meta; self.alias_of = alias_of
        self.bdigest = bdigest or digest


@dataclass
class RunResult:
    ok: bool
    secs: float
    failed: str = None
    model: str = None
    ran: list = field(default_factory=list)
    cached: list = field(default_factory=list)
    error: str = None


@dataclass
class NodeInfo:
    id: str
    status: str
    method: str
    key: str = None
    params: dict = None
    stats: dict = None
    outputs: dict = None
    preview: str = None
    ops: list = None
    secs: float = None
    adopted: bool = False
    group: str = None
    side: str = None
    n: int = None
    summary: str = ""
    warn: list = None
    retry: bool = False             # a soft fail stored for retry: shown "warn (will retry)", re-run by the next run


@dataclass
class Group:
    name: str
    label: str
    nodes: list


class Ctx:
    """What a method or tool sees while it runs."""

    def __init__(self, engine, st, key, tmp, ins, ev, rs, light_only=False, forced=False):
        self.engine = engine; self.project = engine.p; self.step = st; self.key = key; self.out_dir = tmp
        self.forced = forced            # the user asked for this step to run again (run --force)
        self.inputs = ins; self.params = st.params; self.ev = ev; self.rs = rs; self.light_only = light_only
        self.log = os.path.join(tmp, "log.txt"); self.peak_mb = 0; self.warns = []; self._jobs = 0; self.wait_s = 0.0
        self.ops = st.op

    @property
    def fuse_cap(self): return self.engine.mem_gb

    @property
    def n_sides(self): return len(self.rs["sides"])

    @property
    def side_ids_all(self): return [s.id for s in self.rs["sides"]]

    def side_n(self, sid): return self.rs["pos"][sid]

    @property
    def sides_json(self):
        lu = self.inputs.get("lineup"); return lu.files.get("sides") if lu else None

    def call(self, args, mem_gb=4, gpu=False, python=None, env=None, parse=None, on_line=None, light=False):
        if self.light_only: raise NotLight()
        cmd = [python or sys.executable] + [str(a) for a in args]

        def ol(line):
            if parse:
                pr = parse(line)
                if pr: self.progress(pr.get("done", 0), pr.get("total", 1), pr.get("text", ""))
            if on_line: on_line(line)
        try:
            text, peak, _secs = self.engine.runner.run(cmd, mem_gb, gpu, log=self.log, on_line=ol, env=env, light=light)
        finally:
            self.wait_s += self.engine.runner.last_wait
        self.peak_mb = max(self.peak_mb, peak)
        return text

    def job(self, fn, payload, mem_gb=3, gpu=False, on_line=None, python=None, light=False):
        """Run stages/job.py <job.json> (fn = "module.path:function" under stages/) under the runner; returns its result."""
        self._jobs += 1
        jp = os.path.join(self.out_dir, "job_%d.json" % self._jobs); rp = os.path.join(self.out_dir, "result_%d.json" % self._jobs)
        payload = dict(payload, fn=fn, out_dir=self.out_dir, result=rp)
        jdump(payload, jp)
        self.call([JOB, jp], mem_gb=mem_gb, gpu=gpu, on_line=on_line, python=python, light=light)
        try: return jload(rp)
        except Exception: return {}

    def progress(self, done, total, text=""):
        self.engine._progress(self.step, done, total, text)

    def legacy(self, name, **kw):
        if self.ev: self.ev.legacy(name, **kw)

    def line(self, text):
        if self.ev: self.ev.line(text)

    def warn(self, text): self.warns.append(text)


class Engine:
    def __init__(self, project, mem_gb=None, wrap=None, on_event=None, out=None, read_only=None):
        """read_only (default: the project's): nothing is written (no history, no memo, no runs); status / show /
        history / versions work, every change raises recipe.ReadOnly."""
        self.p = project; self.mem_gb = mem_gb; self.on_event = on_event; self.out = out
        self.read_only = getattr(project, "read_only", False) if read_only is None else read_only
        self.hist = History(project.stages, read_only=self.read_only)
        if not self.read_only and not self.hist.exists(): self.hist.init(new_recipe("today"), "New build, preset today")
        self.memo = DigestMemo(os.path.join(project.stages, "cache", "digests.json"), read_only=self.read_only)
        self.store = Store(project, self.memo)
        self.runner = Runner(wrap, memory_gb=mem_gb)
        self._ev = None; self._run = None; self._lockf = None; self._run_lock=threading.RLock()
        self._begin()

    def _writable(self):
        if self.read_only: raise ReadOnly("%s is open read-only" % self.p.root)

    def _begin(self):
        """Start of a plan / run / status: forget the leaf fingerprints and code digests (frames added since, a script
        edited since), so a long-lived engine sees them. Everything below computes them at most once until the next call."""
        self.p.refresh(); self._codes = {}; self._fcode = {}; self._libs = {}
        self._side_frames = {s.id: s.frames for s in self.p.sides()}

    # ------------------------------------------------------------------------------------------------ recipe
    def recipe(self): return self.hist.recipe()

    def graph(self, recipe=None, with_params=None):
        return G.build(self.p, recipe if recipe is not None else self.recipe(), self.canon_with(with_params))

    def canon_with(self, wp):
        """`run --with` overrides, checked and made canonical like set(): every key must be a param path (a typo is an
        error, never a silently ignored override), a side given by position becomes its id (side[2].keep.min_share),
        and so does a lineup.place.base position."""
        if not wp: return wp
        out = {}
        for k, v in wp.items():
            PR.param_of(k)                                  # unknown stage / method / param: ValueError
            k2, v2 = self._canon(k, v)
            if k2 in out and out[k2] != v2: raise ValueError("--with gives %s twice (%r and %r)" % (k2, out[k2], v2))
            out[k2] = v2
        return out

    # ------------------------------------------------------------------------------------------------ stage tokens
    def canon_token(self, token, g=None):
        """A stage token whose side is given by id or position (side@2, side.keep@2, side.keep@2#op-0003) -> the same
        token with the side id. An unknown side is an error (never an empty selection)."""
        if not token or "@" not in str(token): return token
        head, rest = str(token).split("@", 1); sid, sep, tail = rest.partition("#")
        s = self.p.resolve_side(sid, g.sides if g is not None else None)
        return "%s@%s%s%s" % (head, s.id, sep, tail)

    def _no_stage(self, token, g):
        return ("no stage %r (give report, meshed, fused, lineup, lineup.pairs|place|fix|refine|adjust, sides, side@<side>, "
                "side.fuse|platter|keep@<side> or a step id from `status --json`; <side> = an id or a position 1..%d)"
                % (token, len(g.sides)))

    def targets(self, g, token):
        """Step ids a --to token names (checked: an unknown stage or side is an error)."""
        t = self.canon_token(token, g)
        try: ids = g.target_ids(t)
        except KeyError: raise ValueError(self._no_stage(token, g))
        if not ids: raise ValueError(self._no_stage(token, g))
        return ids

    def node_ids(self, g, token):
        """Step ids a --force / --from / show / versions token names (checked)."""
        t = self.canon_token(token, g)
        if t in g.by_id: return [t]
        try: ids = g.node_steps(t)
        except KeyError: raise ValueError(self._no_stage(token, g))
        if not ids: raise ValueError(self._no_stage(token, g))
        return ids

    def _edit(self, fn):
        """One history entry: fn(recipe) changes the CURRENT recipe in place and returns (note, result); the graph must
        still build. Read, change and write happen under the history lock, so a change made meanwhile by another editor
        (the UI, a second CLI) is never lost. -> (result, changed)."""
        def wrapped(r):
            note, result = fn(r)
            G.build(self.p, r)
            return note, result
        return self.hist.edit(wrapped)

    def _commit(self, recipe, note):
        """Record a whole recipe (validated). Kept for callers that built one; the editors below use _edit."""
        G.build(self.p, recipe)
        return self.hist.record(recipe, note)

    def _canon(self, path, v=None):
        """A side given by position in a param path (side[2].keep.min_share) or as lineup.place.base becomes its id; an
        unknown side is an error, never a silently ignored override."""
        d = PR.parse_path(path)
        if d["side"] is not None:
            path = PR.side_path(path, self.p.resolve_side(d["side"]).id)
        if d["node"] == "lineup.place" and d["name"] == "base" and v is not None:
            vs = str(v).strip()
            if vs != "auto": v = self.p.resolve_side(vs).id
        return path, v

    def set(self, changes, note=None):
        ch = {}
        for path, v in changes.items():
            k, v = self._canon(path, v); ch[k] = v

        def fn(r):
            notes = []
            for path, v in ch.items():
                old, new = PR.set_path(r, path, v)
                p, _d = PR.param_of(path)
                notes.append("%s %s -> %s" % (p.label or path, PR.fmt(old), PR.fmt(new)))
            return note or self._label(ch) + ": " + "; ".join(notes), "; ".join(notes)
        res, changed = self._edit(fn)
        return res if changed else None

    def _label(self, changes):
        k = next(iter(changes)); node = PR.parse_path(k)["node"]
        return {"side": "Side", "lineup": "Line-up", "fused": "Fused", "meshed": "Meshed", "report": "Coverage"}.get(node.split(".")[0], node)

    def reset(self, paths):
        ps = [self._canon(p)[0] for p in paths]

        def fn(r):
            notes = []
            for path in ps:
                old, new = PR.reset_path(r, path); notes.append("%s %s -> %s (default)" % (path, PR.fmt(old), PR.fmt(new)))
            return "Reset: " + "; ".join(notes), notes
        return self._edit(fn)[0]

    def preset(self, name):
        if name not in PR.PRESETS: raise ValueError("no preset %r (have: %s)" % (name, ", ".join(PR.PRESETS)))

        def fn(r):
            new = PR.apply_preset(r, name); r.clear(); r.update(new)
            return "Preset: " + name, PR.PRESETS[name]["note"]
        return self._edit(fn)[0]

    def confirm_line(self, recipe=None):
        """Revo's confirm line for One-tap, from the recipe. The Poisson grid is the one this part really gets (from the
        size of the fused result, when one is built)."""
        r = recipe or self.recipe(); g = self.graph(r)
        f = g.by_id["fused"]; m = g.by_id["meshed"]
        s = "Fusion: %s %s mm" % ("Points" if f.method.name == "points" else "TSDF", PR.fmt(f.params["point_distance"]))
        fo = [g.by_id[i] for i in g.chain("fused")]
        s += " / %s: %s" % ("Point tools" if f.produces == "points" else "Mesh tools",
                            ", ".join("%s %s" % (o.method.label, o.method.summary(o.params)) for o in fo if not o.alias) or "none")
        if m.method.name == "poisson":
            ext = self._part_extent(g)
            holes = "on" if m.params["hole_fill"] else "off"
            if ext: s += " / Mesh: Poisson %s, holes %s" % (PR.poisson_label(PR.poisson_plan(ext, m.params["quality"])), holes)
            else: s += " / Mesh: Poisson Q%d (grid set by the part size, known once Fused is built), holes %s" % (m.params["quality"], holes)
        else:
            s += " / Mesh: %s" % m.method.label
        mo = [g.by_id[i] for i in g.chain("meshed")]
        s += " / Mesh tools: %s" % (", ".join("%s %s" % (o.method.label, o.method.summary(o.params)) for o in mo if not o.alias) or "none")
        return s

    def _part_extent(self, g):
        """Largest bbox extent (mm) of the fused result: the current one when built, else the newest stored one."""
        self._begin()
        outs, _info, _ = self._walk(g, "plan", None)
        for i in [g.final("fused"), "fused"]:
            o = outs.get(i)
            bb = ((o.meta or {}).get("stats") or {}).get("bbox_mm") if o is not None else None
            if bb: return float(max(bb))
        best = None
        for st in ("fused",) + tuple(sorted(x for x in os.listdir(self.store.nodes) if x.startswith("fused.")) if os.path.isdir(self.store.nodes) else ()):
            sd = os.path.join(self.store.nodes, st)
            if not os.path.isdir(sd): continue
            for k in os.listdir(sd):
                try: meta = jload(os.path.join(sd, k, "meta.json"))
                except Exception: continue
                bb = (meta.get("stats") or {}).get("bbox_mm")
                if bb and (best is None or meta.get("created", 0) > best[0]): best = (meta.get("created", 0), float(max(bb)))
        return best[1] if best else None

    # ------------------------------------------------------------------------------------------------ op lists
    def list_key(self, token, recipe=None):
        """CLI list names -> recipe op list keys: fused (current method's tools), meshed, side@<side>, lineup.fix,
        lineup.adjust. The Fused stage's manual deletes and base cuts live in the list "fused" (see graph.py)."""
        r = recipe or self.recipe()
        if token == "fused": return "fused." + PR.method_of(r, "fused")
        if token in ("meshed", "lineup.fix", "lineup.adjust"): return token
        if token.startswith("fused."): return token
        if token.startswith("side@") or token.startswith("side.keep@"):
            s = self.p.resolve_side(token.split("@", 1)[1]); return "side.keep@" + s.id
        raise ValueError("no op list %r (fused, meshed, side@<side>, lineup.fix, lineup.adjust)" % token)

    def chain_ops(self, token, r=None):
        """[(list key, index in that list, op)] of the ops a list token shows, in the order they apply: fused = the
        current method's tools, then the Fused stage's manual deletes and base cuts. Indices given to drop / toggle / set count here."""
        r = r if r is not None else self.recipe()
        if token == "fused":
            m = PR.method_of(r, "fused"); lk = "fused." + m
            tools = G.op_list(r, lk, G.get_method("fused", m))
            return [(lk, i, o) for i, o in enumerate(tools)] + [("fused", i, o) for i, o in enumerate(r.get("ops", {}).get("fused", []))]
        lk = self.list_key(token, r)
        meth = G.get_method("fused", lk.split(".", 1)[1]) if lk.startswith("fused.") else None
        return [(lk, i, o) for i, o in enumerate(G.op_list(r, lk, meth))]

    def parked(self, r=None):
        """[(list key, index, op)] of fused tool lists of the methods not in use (kept for when you switch back)."""
        r = r if r is not None else self.recipe(); cur = "fused." + PR.method_of(r, "fused"); out = []
        for lk in sorted(r.get("ops", {})):
            if lk.startswith("fused.") and lk != cur:
                out += [(lk, i, o) for i, o in enumerate(r["ops"][lk])]
        return out

    def _pick(self, token, i, r):
        ch = self.chain_ops(token, r)
        if isinstance(i, str) and not i.lstrip("-").isdigit():
            for e in ch:
                if e[2].get("id") == i: return e
            raise ValueError("no op %r in %s" % (i, token))
        i = int(i)
        if not -len(ch) <= i < len(ch): raise ValueError("no op %d in %s (it has %d)" % (i, token, len(ch)))
        return ch[i]

    def _ops(self, r, lk):
        ops = r.setdefault("ops", {})
        if lk not in ops:
            if lk.startswith("fused."):
                m = G.get_method("fused", lk.split(".", 1)[1])
                ops[lk] = [dict({"on": True}, **copy.deepcopy(o)) for o in m.default_ops]
                for o in ops[lk]: o["id"] = self._new_id(r)
            else: ops[lk] = []
        return ops[lk]

    def _new_id(self, r):
        r["next_op"] = int(r.get("next_op", 1)); i = "op-%04d" % r["next_op"]; r["next_op"] += 1; return i

    def _kind_of_list(self, lk, r):
        if lk.startswith("fused."): return G.get_method("fused", lk.split(".", 1)[1]).produces
        if lk == "fused": return G.get_method("fused", PR.method_of(r, "fused")).produces
        return "mesh"

    def add_op(self, list_key, op, at=None, note=None):
        # A hand-picked base cut belongs to the model, whichever fusion method is used.
        if op.get("tool") == "cut_plane" and (list_key == "fused" or list_key.startswith("fused.")):
            return self._add_op(lambda _r: "fused", op, at, note)
        return self._add_op(lambda r: self.list_key(list_key, r), op, at, note)

    def _add_op(self, lk_of, op, at=None, note=None):
        def fn(r):
            lk = lk_of(r); ops = self._ops(r, lk)
            o = dict(op); o.setdefault("on", True); o["id"] = self._new_id(r)
            if lk == "fused" and o.get("tool") not in ("delete", "cut_plane"):
                raise ValueError("the Fused stage's own list holds manual deletes and base cuts; other tools go to fused.<method>")
            if o.get("tool") == "delete" and o.get("shape") not in ("box", "sphere", "mask"):
                raise ValueError("a delete needs a shape: use `delete <node> --box=.. | --sphere=.. | --mask pts.ply`")
            if o.get("tool") not in ("flip", "match", "turn", "nudge", "delete"):
                kind = self._kind_of_list(lk, r)
                if G.get_tool(o["tool"], kind) is None: raise ValueError("no tool %r for %s results" % (o["tool"], kind))
                given = o.get("params") or {}
                res = PR.resolve_tool(o["tool"], kind, given)          # unknown keys (typos) raise here
                o["params"] = {k: v for k, v in res.items() if k in given}
            if at is None: ops.append(o)
            else: ops.insert(int(at), o)
            return note or "%s: + %s" % (self._list_label(lk), self._op_text(o, lk, r)), o["id"]
        return self._edit(fn)[0]

    def _list_label(self, lk):
        if lk.startswith("fused.") or lk == "fused": return "Fused"
        if lk == "meshed": return "Meshed"
        if lk.startswith("side.keep@"): return "Side %d" % self.p.resolve_side(lk.split("@")[1]).n
        return "Line-up"

    def _op_text(self, op, lk, r):
        if op["tool"] == "flip": return "flip side %d (%s axis)" % (self.p.resolve_side(op["side"]).n, flip_axis_text(op))
        if op["tool"] == "match":
            return "match side %d to side %d (%d points)" % (self.p.resolve_side(op["side"]).n, self.p.resolve_side(op["to"]).n, len(op.get("pairs", [])))
        if op["tool"] == "turn": return "turn side %d by %g deg" % (self.p.resolve_side(op["side"]).n, op.get("deg", 0))
        if op["tool"] == "nudge": return "nudge side %d" % self.p.resolve_side(op["side"]).n
        if op["tool"] == "delete": return "delete %s" % op.get("shape")
        kind = self._kind_of_list(lk, r); t = G.get_tool(op["tool"], kind)
        return "%s %s" % (t.label, t.summary(PR.resolve_tool(op["tool"], kind, op.get("params"))))

    def drop_op(self, list_key, i):
        def fn(r):
            lk, j, _o = self._pick(list_key, i, r); op = self._ops(r, lk).pop(j)
            return "%s: - %s" % (self._list_label(lk), self._op_text(op, lk, r)), op
        return self._edit(fn)[0]

    def toggle_op(self, list_key, i):
        def fn(r):
            lk, j, _o = self._pick(list_key, i, r); op = self._ops(r, lk)[j]; op["on"] = not op.get("on", True)
            return "%s: %s %s" % (self._list_label(lk), "on" if op["on"] else "off", self._op_text(op, lk, r)), op
        return self._edit(fn)[0]

    def set_op(self, list_key, i, changes):
        """Change an op's params in place: ONE history entry (undo goes straight back; no drop + add with a never-built
        state between). Tools: their params (unknown keys are errors); flip: axis; nudge: t, r ("x,y,z"); any op: on."""
        def fn(r):
            lk, j, _o = self._pick(list_key, i, r); op = self._ops(r, lk)[j]; tool = op.get("tool"); notes = []
            for k, v in changes.items():
                if k == "on":
                    old = op.get("on", True); op["on"] = PR.Param("on", "bool", True).coerce(v); notes.append("on %s -> %s" % (old, op["on"]))
                elif tool == "flip":
                    if k != "axis" or str(v) not in ("auto", "x", "y", "z"): raise ValueError("a flip takes axis=auto|x|y|z")
                    notes.append("axis %s -> %s" % (op.get("axis", "auto"), v)); op["axis"] = str(v)
                    op.pop("chose", None); op.pop("rel", None)      # the kept pose was for the old axis: fitted at the run
                elif tool == "match":
                    raise ValueError("a point match is picked again, not edited: drop it and line up from points again")
                elif tool == "turn":
                    if k != "keep": raise ValueError("a turn takes keep=true|false (turn it again for another angle)")
                    kv = PR.Param("keep", "bool", False).coerce(v); notes.append("keep %s -> %s" % (op.get("keep", False), kv)); op["keep"] = kv
                elif tool == "nudge":
                    if k not in ("t", "r"): raise ValueError("a nudge takes t=x,y,z and r=rx,ry,rz")
                    vv = [float(x) for x in (v.split(",") if isinstance(v, str) else v)]
                    if len(vv) != 3: raise ValueError("%s: expected 3 numbers" % k)
                    notes.append("%s %s -> %s" % (k, op.get(k, [0, 0, 0]), vv)); op[k] = vv
                elif tool == "delete":
                    raise ValueError("a delete is redrawn, not edited: drop it and delete again")
            if tool not in ("flip", "match", "turn", "nudge", "delete"):
                tc = {k: v for k, v in changes.items() if k != "on"}
                if tc:
                    kind = self._kind_of_list(lk, r)
                    cur = PR.resolve_tool(tool, kind, op.get("params"))
                    new = PR.resolve_tool(tool, kind, dict(op.get("params") or {}, **tc))     # unknown keys raise
                    prm = dict(op.get("params") or {})
                    for k in tc:
                        prm[k] = new[k]
                        q = next((q for q in PR.tool_params(tool, kind) if q.name == k), None)
                        notes.append("%s %s -> %s" % ((q.label if q and q.label else k), PR.fmt(cur[k]), PR.fmt(new[k])))
                    op["params"] = prm
            return "%s: %s %s" % (self._list_label(lk), tool, "; ".join(notes)), op
        op, changed = self._edit(fn)
        return op if changed else None

    def clear_ops(self, list_key):
        def fn(r):          # fused: the whole chain the user sees (the method's tools and the manual edits)
            lks = ["fused." + PR.method_of(r, "fused"), "fused"] if list_key == "fused" else [self.list_key(list_key, r)]
            for lk in lks: r.setdefault("ops", {})[lk] = []
            return "%s: tools cleared" % self._list_label(lks[0]), None
        self._edit(fn)

    def ops_of(self, list_key):
        """-> (label, [(list key, index, op)] in chain order, parked [(list key, index, op)] for fused)."""
        r = self.recipe()
        return ("fused" if list_key == "fused" else self.list_key(list_key, r)), self.chain_ops(list_key, r), \
            (self.parked(r) if list_key == "fused" else [])

    # ------------------------------------------------------------------------------------------------ side edits
    def leave_out(self, side):
        s = self.p.resolve_side(side)

        def fn(r):
            lo = set(r["sides"].get("leave_out", [])); lo.add(s.id)
            if len(lo) >= len(self.p.sides()): raise ValueError("at least one side must stay in")
            r["sides"]["leave_out"] = sorted(lo)
            return "Side %d left out" % s.n, None
        self._edit(fn)

    def include(self, side):
        s = self.p.resolve_side(side)

        def fn(r):
            r["sides"]["leave_out"] = sorted(set(r["sides"].get("leave_out", [])) - {s.id})
            return "Side %d back in" % s.n, None
        self._edit(fn)

    @entry
    def flip(self, side, axis="auto", preview=False):
        """Flip a side 180 deg about one of its own axes, FITTED on the line-up as it is now (auto: the axis that lands
        best after the fit) and kept as that pose against the placed sides (tools/edits.py fix_job), so a refit, base
        change or leave-out does not turn it again. preview: the fitted pose only, nothing added (the Flip preview uses
        this same fit). Before any line-up is built the flip is fitted when the line-up runs."""
        s = self.p.resolve_side(side)
        if str(axis) not in ("auto", "x", "y", "z"): raise ValueError("a flip takes axis=auto|x|y|z")
        fit = self._fit_flip(s.id, axis)
        if preview:
            if fit is None: raise ValueError("the line-up is not built yet: run first")
            return fit
        op = {"tool": "flip", "side": s.id, "axis": axis}
        if fit: op.update(chose=fit["axis"], rel=fit["rel"])
        return self.add_op("lineup.fix", op, note="Line-up: flip side %d (%s axis)" % (s.n, flip_axis_text(op)))

    def _lineup_now(self):
        """(graph, outs, {side: T} the line-up gives before nudges) or None when it is not built for the current recipe."""
        self._begin(); g = self.graph(); outs, _info, _ = self._walk(g, "plan", None)
        o = outs.get("lineup.refine")
        if o is None: return None
        tj = jload(o.files.get("transforms") or o.files["placement"])
        return g, outs, {k: tj["T"][k] for k in tj["order"]}

    def _fit_flip(self, sid, axis):
        now = self._lineup_now()
        if now is None or sid not in now[2]: return None
        g, outs, T = now; ins = g.by_id["lineup.fix"].inputs
        meshes = {k[5:]: outs[v].files["mesh"] for k, v in ins.items() if k.startswith("mesh@") and k[5:] in T}
        digests = {k[5:]: outs[v].digest for k, v in ins.items() if k.startswith("mesh@") and k[5:] in T}
        return self.job("tools.edits:resolve_flip_job", {"T": T, "side": sid, "axis": axis, "meshes": meshes, "digests": digests,
                                                         "cache": os.path.join(self.p.stages, "cache", "samples")}, mem_gb=3, light=True)

    def match(self, side, to, pairs):
        """Line up a side from picked point pairs [[to_xyz, side_xyz]], each in its own side's scan frame (so the match
        survives changes upstream): a rigid fit to the picks, then a local surface fit (tools/edits.py solve_match)."""
        s = self.p.resolve_side(side); t = self.p.resolve_side(to)
        if s.id == t.id: raise ValueError("match a side to another side")
        from .methods import REPO
        if REPO not in sys.path: sys.path.insert(0, REPO)
        import registration_metrics as RM
        pr = RM.validate_landmark_pairs(pairs)
        op = {"tool": "match", "side": s.id, "to": t.id, "pairs": [[[round(float(x), 4) for x in p] for p in q] for q in pr]}
        return self.add_op("lineup.fix", op, note="Line-up: match side %d to side %d (%d points)" % (s.n, t.n, len(pr)))

    @entry
    def turn(self, side, R, keep=False, move=None):
        """Turn a side by the world rotation R (3x3) about its own centre, as the Nudge view shows it (the line-up after
        nudges). Kept in the side's own frame (M) and, on a built line-up, as that pose against the placed sides (rel, like
        a flip), so a refit or a base change does not turn it again; the line-up run then fits it LOCALLY from that pose
        (tools/edits.py fix_job) unless keep ("Keep exactly as placed"). move = a nudge (mm) added in the same history
        entry, so Nudge's Apply with a turn and a move is one undo. -> the turn op's id."""
        op, note = self._turn_op(side, R, keep)
        if not move or not any(float(x) for x in move): return self.add_op("lineup.fix", op, note=note)
        nop, nnote = self._nudge_op(side, move, (0, 0, 0))

        def fn(r):
            ids = []
            for lk, o in (("lineup.fix", op), ("lineup.adjust", nop)):
                o = dict(o, on=True); o["id"] = self._new_id(r); self._ops(r, lk).append(o); ids.append(o["id"])
            return note + "; " + nnote[len("Line-up: "):], ids[0]
        return self._edit(fn)[0]

    def _turn_op(self, side, R, keep=False):
        """The turn op and its history note (see turn)."""
        import numpy as np
        s = self.p.resolve_side(side); R = np.asarray(R, float)
        if R.shape != (3, 3) or abs(np.linalg.det(R) - 1) > 1e-3 or np.abs(R.T @ R - np.eye(3)).max() > 1e-3:
            raise ValueError("a turn takes a 3x3 rotation")
        deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        if deg < 0.01: raise ValueError("nothing to turn")
        res = self._resolve_turn(s.id, R)
        if res is None: raise ValueError("the line-up is not built yet: run first")
        op = {"tool": "turn", "side": s.id, "M": [[round(float(x), 9) for x in row] for row in res["M"]], "c": res["c"],
              "deg": round(deg, 2), "keep": bool(keep)}
        if res.get("rel"): op["rel"] = res["rel"]
        return op, "Line-up: turn side %d by %.1f deg%s" % (s.n, deg, " (kept as placed)" if keep else "")

    def _resolve_turn(self, sid, R):
        now = self._lineup_now()
        if now is None or sid not in now[2]: return None
        g, outs, T = now; ins = g.by_id["lineup.fix"].inputs
        adj = outs.get("lineup.adjust"); Td = jload(adj.files["lineup"])["T"] if adj is not None and adj.files.get("lineup") else T
        meshes = {k[5:]: outs[v].files["mesh"] for k, v in ins.items() if k.startswith("mesh@") and k[5:] in T}
        digests = {k[5:]: outs[v].digest for k, v in ins.items() if k.startswith("mesh@") and k[5:] in T}
        return self.job("tools.edits:resolve_turn_job", {"T": T, "T_disp": Td.get(sid, T[sid]), "side": sid, "R": R.tolist(), "meshes": meshes,
                                                         "digests": digests, "cache": os.path.join(self.p.stages, "cache", "samples")}, mem_gb=3, light=True)

    def nudge(self, side, t=(0, 0, 0), r=(0, 0, 0)):
        op, note = self._nudge_op(side, t, r)
        return self.add_op("lineup.adjust", op, note=note)

    def _nudge_op(self, side, t, r):
        s = self.p.resolve_side(side)
        return ({"tool": "nudge", "side": s.id, "t": [float(x) for x in t], "r": [float(x) for x in r]},
                "Line-up: nudge side %d by %s mm / %s deg" % (s.n, list(t), list(r)))

    def platter(self, side, how):
        """how: auto | search | none | a plane.json path (manual)."""
        s = self.p.resolve_side(side)
        plane = None
        if how not in ("auto", "search", "none", "plate"):
            j = jload(how); plane = {"n": [float(x) for x in j["n"]], "d": float(j["d"]), "margin": float(j.get("margin", 1.0)),
                                     "keep_above": bool(j.get("keep_above", True))}

        def fn(r):
            pl = r["sides"].setdefault("platter", {})
            if how == "auto": pl.pop(s.id, None)
            elif how in ("search", "none", "plate"): pl[s.id] = {"method": how}
            else: pl[s.id] = {"method": "manual", "plane": plane}
            return "Side %d: platter %s" % (s.n, how if how in ("auto", "search", "none", "plate") else "manual plane"), None
        self._edit(fn)

    # ------------------------------------------------------------------------------------------------ history
    def undo(self): return self.hist.undo()

    def redo(self): return self.hist.redo()

    def history(self): return self.hist.entries()

    # ------------------------------------------------------------------------------------------------ keys + walk
    def _file_code(self, path):
        if path not in self._fcode:
            try:
                with open(path, "rb") as f: self._fcode[path] = hashlib.blake2b(f.read(), digest_size=16).hexdigest()
            except OSError: self._fcode[path] = "-"
        return self._fcode[path]

    def _lib(self, name):
        if name not in self._libs: self._libs[name] = lib_version(name)
        return self._libs[name]

    def code_digest(self, st):
        """Digest of the code a step runs: its scripts' and stages modules' contents and the package versions that
        matter (Method.code), once per run (see _begin)."""
        if st.method is None or not hasattr(st.method, "code"): return None
        files, libs = st.method.code(st.params)
        if getattr(st,"produces",None) in ("mesh", "points"):
            from .methods import HERE, script
            files = list(files) + [("stages/color.py",os.path.join(HERE,"color.py")), ("color_frames.py",script("color_frames.py"))]
            libs = list(libs) + ["open3d", "numpy", "scipy", "Pillow", "opencv-python"]
        ck = (type(st.method).__name__, tuple(files), tuple(libs))
        if ck not in self._codes:
            lv = {}
            for l in libs:
                if "=" in l: k, v = l.split("=", 1); lv[k] = v
                else: lv[l] = self._lib(l)
            self._codes[ck] = H({"files": {n: self._file_code(pth) for n, pth in files}, "libs": lv})
        return self._codes[ck]

    def step_key(self, st, outs):
        ins = {n: outs[i].digest for n, i in st.inputs.items()}
        code = self.code_digest(st)
        if st.kind == "op":
            payload = {k: v for k, v in st.op.items() if k not in ("id", "recorded", "on", "params")}
            payload["params"] = st.params
            return H({"engine": ENGINE_VERSION, "parent": ins["parent"], "tool": st.method.name, "version": st.method.version,
                      "op": payload, "lineup": ins.get("lineup"), "code": code})
        return H({"engine": ENGINE_VERSION, "node": st.id, "method": st.method.name, "version": st.method.version,
                  "params": st.params, "inputs": ins, "leaf": st.leaf, "code": code})

    def _out_meta(self, st, meta, outs):
        if meta.get("alias_of"):
            src = outs[st.inputs[meta["alias_of"]]]
            return Out(st.id, src.digest, src.files, src.kind, meta, alias_of=meta["alias_of"], bdigest=src.bdigest)
        files = {n: os.path.join(meta["_dir"], o["file"]) for n, o in (meta.get("outputs") or {}).items()}
        from .color import has_colors
        identity=meta["digest"] if any(has_colors(path) for path in files.values() if path.endswith((".ply",".npz"))) else meta.get("geo_digest") or meta["digest"]
        return Out(st.id, identity, files, meta.get("kind", st.produces), meta, bdigest=meta["digest"])

    def _status_of_meta(self, meta):
        if meta.get("transient"): return "warn"
        if meta.get("check"): return "check"
        if meta.get("adopted"): return "adopted"
        if meta.get("status") == "warn": return "warn"
        return "fresh"

    def _walk(self, g, mode="plan", needed=None, force=frozenset(), ev=None, rs=None, adopter=None):
        """mode plan: statuses only (no touching). run: execute stale steps. adopt: adopt legacy files or run light steps.
        Returns (outs, info, failed step id or None)."""
        if mode=="run":return self._walk_run(g,needed,force,ev,rs)
        outs, info = {}, {}; failed = None
        for st in g.steps:
            if needed is not None and st.id not in needed: continue
            if failed is not None or any(i not in outs for i in st.inputs.values()):
                info[st.id] = {"status": "waiting"}; continue
            if st.alias:
                outs[st.id] = outs[st.inputs[st.alias]]
                info[st.id] = {"status": "n/a" if st.na else "off"}
                if mode == "run": self._node_event(ev, st, "off", rs)
                continue
            key = self.step_key(st, outs)
            meta = None; prev = None
            if st.id not in force:
                meta = (self.store.peek if mode == "plan" else self.store.lookup)(st.store, key)
            if meta is not None and meta.get("transient") and mode == "run":
                prev = meta; meta = None            # a soft fail stored for retry: this run tries it again (only it)
            if meta is not None:
                outs[st.id] = self._out_meta(st, meta, outs)
                info[st.id] = {"status": self._status_of_meta(meta), "key": key, "meta": meta}
                if meta.get("transient"): info[st.id]["retry"] = True
                if mode == "run":
                    self._legacy_before(ev, st, rs, cached=True)
                    self._node_event(ev, st, "check" if meta.get("check") else "cached", rs, key=key, meta=meta)
                    rs["cached"].append(st.id)
                continue
            if mode == "plan":
                info[st.id] = {"status": "stale", "key": key}; continue
            if mode == "adopt":
                o = adopter.adopt(self, st, key, outs, g) if adopter else None
                if o is None and st.method is not None and getattr(st.method, "light", False):
                    try:
                        if adopter: adopter.before_light(self, st, outs, g)
                        o = self._execute(st, key, outs, None, rs, light_only=True)
                    except NotLight: o = None
                    except Exception as e: o = None; info[st.id] = {"status": "stale", "key": key, "why": str(e)[:200]}
                if o is None:
                    info.setdefault(st.id, {"status": "stale", "key": key}); continue
                outs[st.id] = o; info[st.id] = {"status": self._status_of_meta(o.meta or {}), "key": key, "meta": o.meta}
                continue
            # run
            self._legacy_before(ev, st, rs, cached=False)
            try:
                o = self._execute(st, key, outs, ev, rs, prev=prev, forced=st.id in force)
            except Cancelled:
                info[st.id] = {"status": "failed", "key": key}; failed = st.id
                self._node_event(ev, st, "failed", rs, key=key, error="cancelled"); rs["error"] = "cancelled"; break
            except Exception as e:
                tail = getattr(e, "tail", "") or ""
                msg = "%s: %s" % (st.id, str(e)); info[st.id] = {"status": "failed", "key": key, "error": msg}; failed = st.id
                self._node_event(ev, st, "failed", rs, key=key, error=(str(e) + (" | " + tail[-600:] if tail else "")))
                if ev: ev.emit("error", msg=msg[:300] + ((" | " + tail[-300:]) if tail else ""), node=st.id)
                rs["error"] = msg
                self.store.note_state(st.id, key, "failed")
                continue
            outs[st.id] = o; meta = o.meta or {}
            info[st.id] = {"status": self._status_of_meta(meta), "key": key, "meta": meta}
            if meta.get("transient"): info[st.id]["retry"] = True
            state = "check" if meta.get("check") else ("warn" if meta.get("status") == "warn" else "done")
            self._node_event(ev, st, state, rs, key=key, meta=meta)
            if meta.get("check") and ev:
                for w in meta.get("warn", []): ev.legacy("warn", msg=w)
            rs["ran"].append(st.id)
        return outs, info, failed

    def _walk_run(self,g,needed,force,ev,rs):
        try:return self._walk_ready(g,needed,force,ev,rs)
        except Cancelled:
            steps=[st for st in g.steps if needed is None or st.id in needed]
            info={st.id:{"status":"waiting"} for st in steps}
            if not steps:return {},info,None
            first=steps[0];info[first.id]={"status":"failed"};rs["error"]="cancelled"
            self._node_event(ev,first,"failed",rs,error="cancelled")
            return {},info,first.id

    def _walk_ready(self,g,needed,force,ev,rs):
        """Run ready nodes; a failure blocks descendants but not independent sides."""
        from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
        from .parallel import Admission,gpu_slots
        steps=[st for st in g.steps if needed is None or st.id in needed]
        pending=list(steps);outs={};info={};errors={};inflight={};keys=set();cancelled=False
        requested=max(1,int(os.environ.get("POINTYOINK_BUILD_GPU_SLOTS","2")))
        gpu_limit,free=gpu_slots(requested)
        workers=max(1,int(os.environ.get("POINTYOINK_BUILD_WORKERS","100000")))
        # The old scanner parser derives progress from serial side positions.
        if ev and getattr(ev,"legacy_mode",False):workers=1;gpu_limit=1
        scope=getattr(self.runner._tl,"scope",None)
        with self.runner.batch(gpu_limit) as jobs:
            admission=Admission(jobs.budget,gpu_limit,workers)
            if ev:ev.say("Parallel build: up to %d GPU fusion jobs; %.2f GiB RAM and %d CPU threads shared%s" %
                         (gpu_limit,jobs.budget.memory_gb,jobs.budget.cpu_threads,
                          "; requested %d GPU jobs, limited by VRAM or the two-job ceiling"%requested if requested>gpu_limit else ""))
            def execute(st,key,ins,previous):
                with self.runner.worker(scope,jobs,st.kind):
                    return self._execute(st,key,ins,ev,rs,prev=previous,forced=st.id in force)
            last_limits = resources.limit_options()
            with ThreadPoolExecutor(max_workers=min(workers,resources.cpu_count())) as pool:
                while pending or inflight:
                    limits = resources.limit_options()
                    if limits != last_limits:
                        gpu_limit, free = gpu_slots(requested)
                        jobs.gpu_limit = gpu_limit
                        last_limits = limits
                    try: admission.refresh(resources.budget(self.runner.memory_gb), gpu_limit)
                    except ValueError: pass  # The child launch reports unavailable desktop headroom.
                    if self.runner.cancelled:cancelled=True
                    changed=False
                    if not cancelled:
                        for st in list(pending):
                            inputs=list(st.inputs.values())
                            if any(info.get(i,{}).get("status") in ("failed","blocked") for i in inputs):
                                info[st.id]={"status":"blocked"};pending.remove(st);changed=True;continue
                            if any(i not in outs for i in inputs):continue
                            if st.alias:
                                outs[st.id]=outs[st.inputs[st.alias]];info[st.id]={"status":"n/a" if st.na else "off"}
                                self._node_event(ev,st,"off",rs);pending.remove(st);changed=True;continue
                            key=self.step_key(st,outs);identity=(st.store,key)
                            if identity in keys:continue
                            meta=None if st.id in force else self.store.lookup(st.store,key)
                            previous=meta if meta and meta.get("transient") else None
                            if previous:meta=None
                            if meta is not None:
                                outs[st.id]=self._out_meta(st,meta,outs);info[st.id]={"status":self._status_of_meta(meta),"key":key,"meta":meta}
                                self._legacy_before(ev,st,rs,cached=True)
                                self._node_event(ev,st,"check" if meta.get("check") else "cached",rs,key=key,meta=meta)
                                with self._run_lock:rs["cached"].append(st.id)
                                pending.remove(st);changed=True;continue
                            if not admission.accepts(st):continue
                            ins={i:outs[i] for i in inputs};self._legacy_before(ev,st,rs,cached=False)
                            admission.take(st);keys.add(identity);pending.remove(st)
                            future=pool.submit(execute,st,key,ins,previous);inflight[future]=(st,key,identity);changed=True
                    if inflight:
                        done,_=wait(inflight,timeout=.2,return_when=FIRST_COMPLETED)
                        for future in sorted(done,key=lambda f:steps.index(inflight[f][0])):
                            st,key,identity=inflight.pop(future);admission.release(st);keys.discard(identity)
                            try:o=future.result()
                            except Cancelled:
                                cancelled=True;errors[st.id]="cancelled";info[st.id]={"status":"failed","key":key}
                                self._node_event(ev,st,"failed",rs,key=key,error="cancelled")
                                if not self.runner.cancelled:self.runner.cancel()
                            except Exception as ex:
                                tail=getattr(ex,"tail","") or "";msg="%s: %s"%(st.id,ex);errors[st.id]=msg
                                info[st.id]={"status":"failed","key":key,"error":msg}
                                self._node_event(ev,st,"failed",rs,key=key,error=str(ex)+(" | "+tail[-600:] if tail else ""))
                                if ev:ev.emit("error",msg=msg[:300]+(" | "+tail[-300:] if tail else ""),node=st.id)
                                with self._run_lock:self.store.note_state(st.id,key,"failed")
                            else:
                                outs[st.id]=o;meta=o.meta or {};info[st.id]={"status":self._status_of_meta(meta),"key":key,"meta":meta}
                                if meta.get("transient"):info[st.id]["retry"]=True
                                state="check" if meta.get("check") else "warn" if meta.get("status")=="warn" else "done"
                                self._node_event(ev,st,state,rs,key=key,meta=meta)
                                if meta.get("check") and ev:
                                    for warning in meta.get("warn",[]):ev.legacy("warn",msg=warning)
                                with self._run_lock:rs["ran"].append(st.id)
                    elif not changed:break
            if cancelled and not errors and pending:
                st=pending[0];errors[st.id]="cancelled";info[st.id]={"status":"failed"}
                self._node_event(ev,st,"failed",rs,error="cancelled")
            for st in pending:info.setdefault(st.id,{"status":"waiting"})
            for value in info.values():
                if value.get("status")=="blocked":value["status"]="waiting"
            order={st.id:i for i,st in enumerate(steps)}
            with self._run_lock:
                rs["ran"].sort(key=order.get);rs["cached"].sort(key=order.get)
                failed=min(errors,key=order.get) if errors else None
                rs["error"]=errors.get(failed)
                rs["admission"]={"gpu_requested":requested,"gpu_limit":gpu_limit,"free_vram_bytes":free,
                    "memory_bytes":jobs.budget.memory_bytes,"cpu_threads":jobs.budget.cpu_threads,
                    "peak_child_caps_bytes":jobs.peak_mem,"peak_child_threads":jobs.peak_cpu,"peak_children":jobs.peak_jobs}
        return outs,info,failed

    RETRY_LIMIT = 3            # identical failures with the same inputs before a soft fail is kept for good

    def _execute(self, st, key, outs, ev, rs, light_only=False, prev=None, forced=False):
        if st.id == "report":
            path = os.path.join(self.p.stages, "coverage.lock")
            self.p.guard_write(path)
            with open(path, "a+") as handle:
                while True:
                    if self.runner.cancelled: raise Cancelled()
                    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                    except BlockingIOError: time.sleep(0.05)
                try: return self._execute_locked(st, key, outs, ev, rs, light_only, prev, forced)
                finally: fcntl.flock(handle, fcntl.LOCK_UN)
        return self._execute_locked(st, key, outs, ev, rs, light_only, prev, forced)

    def _execute_locked(self, st, key, outs, ev, rs, light_only=False, prev=None, forced=False):
        """prev: the stored retry result this run replaces (a soft fail flagged transient), for the retry count."""
        tmp = self.store.tmp_dir(st.store, key)
        ins = {n: outs[i] for n, i in st.inputs.items()}
        ctx = Ctx(self, st, key, tmp, ins, ev, rs or self._rs(None), light_only, forced=forced)
        if rs is not None:
            with self._run_lock:rs["current"]=st.id;rs["cur_frac"]=0.0;rs.setdefault("active",{})[st.id]=0.0
        if ev and not light_only: self._node_event(ev, st, "start", rs, key=key)
        t0 = time.time()
        try:
            res = st.method.execute(ctx) or {}
            filekey = "mesh" if st.produces=="mesh" else "points" if st.produces=="points" else None
            if filekey and res.get("outputs",{}).get(filekey):
                from .color import has_colors
                target=os.path.join(ctx.out_dir,res["outputs"][filekey])
                if st.kind=="fused" and filekey=="points" and ctx.params.get("color")=="on":
                    ctx.job("color:frame_job",{"target":target,"sides_json":ins["lineup"].files["sides"],"project":self.p.root,"frames_per_side":ctx.params.get("frames_per_side",300),"white":bool(ctx.params.get("white_balance",True))},mem_gb=4)
                source=next((path for item in ins.values() for name,path in item.files.items() if name in ("mesh","points") and has_colors(path)),None)
                if source and (st.kind=="meshed" or (st.kind=="op" and st.method.name!="isolation") or not has_colors(target)):
                    ctx.job("color:carry_job",{"source":source,"target":target},mem_gb=4)
        except BaseException:
            try:
                if os.path.exists(ctx.log) and rs is not None and rs.get("run_id"):
                    shutil.copy(ctx.log, os.path.join(self.p.stages, "runs", "%s.%s.log" % (rs["run_id"], st.id.replace("/", "_"))))
            except Exception: pass
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        secs = time.time() - t0 - ctx.wait_s                 # the step's own time; time queued for a slot / the GPU apart
        warn = list(res.get("warn", [])) + ctx.warns
        meta = {"node": st.id, "version": st.method.version, "params": st.params, "kind": st.produces,
                "inputs": {n: {"node": i, "digest": outs[i].digest} for n, i in st.inputs.items()},
                "stats": {k: v for k, v in (res.get("stats") or {}).items() if v is not None}, "secs": round(secs, 2),
                "wait_s": round(ctx.wait_s, 1),
                "peak_mb": ctx.peak_mb, "status": "warn" if warn else "ok", "warn": warn}
        if st.kind == "op":
            meta["tool"] = st.method.name; meta["parent"] = outs[st.inputs["parent"]].digest
            meta["op"] = {k: v for k, v in st.op.items() if k not in ("recorded",)}
            if res.get("check"): meta["check"] = True
        else:
            meta["method"] = st.method.name
        if res.get("adopted"): meta["adopted"] = True      # built only from adopted inputs (pairs from the build's cache)
        alias_of = res.get("alias_of")
        outputs = {} if alias_of else res.get("outputs", {})
        state = "warn" if warn else "done"
        if res.get("transient"):
            meta, outputs = self._transient(st, meta, res, tmp, prev); state = "retry" if meta.get("transient") else "warn"
        if alias_of:
            meta["alias_of"] = alias_of; meta["digest"] = ins[alias_of].bdigest; meta["geo_digest"] = ins[alias_of].digest
        try:
            final = self.store.publish(tmp, st.store, key, meta, outputs, geo_fn=self._geo_job)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)         # nothing half-stored (a cancelled geometry digest included)
            raise
        with self._run_lock:self.store.note_state(st.id,key,state,secs,self._wkind(st),self._size(st))
        return self._out_meta(st, final, outs)

    def _transient(self, st, meta, res, tmp, prev=None):
        """A soft fail that was not deliberate (res["fail"]: kind, sig, rc): stored under its key flagged for retry
        (meta.transient, meta.retry.count = identical failures in a row with these inputs), or, at RETRY_LIMIT identical
        ones, kept as a normal soft fail with a warning saying so (a launch failure is always retried). A report keeps
        the outputs it did make. -> (meta, outputs)."""
        fail = res.get("fail") or {}
        kind = fail.get("kind") or "unknown"; sig = fail.get("sig") or kind
        pr = (prev or {}).get("retry") or {}
        count = int(pr.get("count", 0)) + 1 if pr.get("sig") == sig else 1
        retry = {"kind": kind, "sig": sig, "rc": fail.get("rc"), "count": count}
        warn = list(meta.get("warn") or [])
        why = sig.split("|", 1)[-1][:120]
        if kind != "launch" and count >= self.RETRY_LIMIT:
            warn.append("%s failed the same way %d runs in a row with the same inputs (%s): kept as it is, not retried any "
                        "more; `run --force %s` tries again" % (st.id, count, why, st.id))
            meta = dict(meta, warn=warn, status="warn", retry=dict(retry, gave_up=True))
        else:
            warn.append("will retry next run (%s: %s; failure %d%s)" % (kind, why, count, "" if kind == "launch" else " of %d" % self.RETRY_LIMIT))
            meta = dict(meta, warn=warn, status="warn", transient=True, retry=retry)
        outputs = {n: fn for n, fn in (res.get("outputs") or {}).items() if os.path.exists(os.path.join(tmp, fn))}
        return meta, ({} if res.get("alias_of") else outputs)

    def _geo_job(self, path):
        """Geometry digest of a mesh too big for the engine process: a small job under a memory cap. A cancel propagates
        (the artifact is then not stored); another failure is retried once with a bigger cap, then raised: an artifact
        is never stored without the geometry digest its downstream keys need."""
        err = None
        for gb in (4, 8):
            try:
                g = self.job("plyio:geo_digest_job", {"in": path}, mem_gb=gb, light=False).get("geo")
                if g: return g
                err = "no digest in the job's result"
            except Cancelled: raise
            except Exception as e: err = str(e)[:200]
        raise StepError("geometry digest of %s failed: %s" % (os.path.basename(path), err))

    # ------------------------------------------------------------------------------------------------ events
    def _wkind(self, st):
        if st.kind == "op":
            tool = st.op.get("tool", "")
            action = "." + st.params.get("action", "smooth") if tool == "local_fix" else ""
            return "op." + tool + action
        if st.kind in ("fused", "meshed"): return "%s.%s" % (st.kind, st.method.name)
        return st.kind

    def _size(self, st):
        """Cheap size known during planning (frames of a side, pairs of a line-up, capped frames of a fuse)."""
        if st.side:
            return max(1, self._side_frames.get(st.side, 1))
        n = len([k for k in st.inputs if k.startswith("mesh@")]) or len(self._side_frames)
        if st.kind == "lineup.pairs":
            return max(1, n * (n - 1) // 2)
        if st.kind == "fused":
            cap = st.params.get("frames_per_side", 0)
            active = st.leaf.get("frames", self._side_frames)
            return max(1, sum(min(self._side_frames.get(sid, 0), cap) if cap else self._side_frames.get(sid, 0)
                              for sid in active))
        return max(1, n)

    def _weight(self, st, history):
        """Seconds a step will take: its own last run scaled by size, else this kind's rate, else every project's, else the typical number."""
        k = self._wkind(st)
        if st.kind == "op":
            previous = history.get("secs", {}).get(k)
            if previous: return max(0.5, previous[-1])
            # Older histories pooled every tool. Recover the last matching tool
            # from its artifacts rather than estimating a smooth from a delete.
            matches = []
            root = os.path.join(self.store.nodes, st.store)
            if os.path.isdir(root):
                for key in os.listdir(root):
                    m = jload(os.path.join(root, key, "meta.json"))
                    op = m.get("op", {})
                    if op.get("tool") != st.op.get("tool") or m.get("status") not in ("ok", "warn", "done"): continue
                    if op.get("tool") == "local_fix" and op.get("params", {}).get("action", "smooth") != st.params.get("action", "smooth"): continue
                    if m.get("secs", 0) >= 0.5: matches.append((m.get("created", 0), m["secs"]))
            if matches: return max(0.5, max(matches)[1])
            return 10.0
        size = self._size(st)
        scales_with_size = st.kind in ("side.fuse", "fused", "lineup.pairs")
        previous = history.get("nodes", {}).get(st.id, {})
        if previous.get("secs", 0) >= 0.5:
            ratio = size / previous["size"] if scales_with_size and previous.get("size") else 1.0
            return max(0.5, previous["secs"] * max(0.1, min(20.0, ratio)))
        ok = lambda rows: [r for r in rows or [] if r[1] >= _store.MIN_SECS]      # older files hold cache hits as 0 s runs
        sized = ok(history.get("size_secs", {}).get(k)) or ok(_store.global_size_secs().get(k))    # this project's rate, else earlier projects'
        if sized:
            if scales_with_size:
                rates = [secs / measure for measure, secs in sized if measure > 0]
                if rates:
                    return max(0.5, statistics.median(rates) * size)
            return max(0.5, statistics.median(secs for _, secs in sized))
        h = [x for x in history.get("secs", {}).get(k) or [] if x >= _store.MIN_SECS] or [x for x in _store.global_secs().get(k) or [] if x >= _store.MIN_SECS]
        base = statistics.median(h) if h else DEFAULT_SECS.get(k, 10)
        if not scales_with_size or h:                      # past runs carry no size: their seconds as they were, not scaled from the typical size
            return max(0.5, base)
        reference = 300 if st.side else (3 if st.kind == "lineup.pairs" else 900 if st.kind == "fused" else 3)
        return max(0.5, base * max(0.1, min(20.0, size / reference)))

    def _rs(self, ev, plan_info=None, g=None):
        sides = self.p.sides()
        return {"sides": sides, "pos": {s.id: s.n for s in sides}, "ran": [], "cached": [], "error": None, "done_w": 0.0,
                "total_w": 1.0, "weights": {}, "current": None, "cur_frac": 0.0, "active": {}, "completed": set(), "last_overall": 0.0, "plan": plan_info or {}, "g": g, "run_id": None}

    def _progress(self, st, done, total, text):
        with self._run_lock:
            rs=self._run
            if rs is None or self._ev is None:return
            frac=min(1.0,done/max(1,total));rs["cur_frac"]=frac
            active=rs.setdefault("active",{});active[st.id]=max(active.get(st.id,0.0),frac)
            partial=sum(rs["weights"].get(i,0)*f for i,f in active.items())
            overall=min(1.0,(rs["done_w"]+partial)/max(1e-9,rs["total_w"]))
            overall=max(rs.get("last_overall",0.0),overall);rs["last_overall"]=overall
            self._ev.emit("progress",id=st.id,done=done,total=total,frac=round(frac,3),overall=round(overall,3),text=text)

    def _node_event(self,ev,st,state,rs,key=None,meta=None,error=None):
        with self._run_lock:
            if rs is not None and state in ("done","warn","check","failed","cached","off"):
                completed=rs.setdefault("completed",set())
                if st.id not in completed:rs["done_w"]+=rs["weights"].get(st.id,0);completed.add(st.id)
                rs.setdefault("active",{}).pop(st.id,None)
            if ev is None:return
            kw={"id":st.id,"group":st.group,"side":st.side,"n":st.n,"state":state}
            if key:kw["key"]=key16(key)
            if meta:
                kw["secs"]=meta.get("secs");kw["stats"]=meta.get("stats") or {}
                if meta.get("wait_s"):kw["wait_s"]=meta["wait_s"]
                if meta.get("warn"):kw["warn"]=meta["warn"]
                if meta.get("transient"):kw["retry"]=True
                pv=os.path.join(meta.get("_dir",""),"preview.png");kw["preview"]=pv if os.path.exists(pv) else None
            if error:kw["error"]=error[-600:]
            ev.emit("node",**kw)

    def _legacy_before(self, ev, st, rs, cached):
        """build_project.py's per-side line at the start of each side."""
        if ev is None or st.kind != "side.fuse": return
        plan = rs.get("plan", {}); g = rs.get("g")
        sid = st.side; n = st.n; N = len(rs["sides"])
        side_steps = [s.id for s in g.steps if s.group == "side" and s.side == sid] if g else []
        fresh = all(plan.get(i, {}).get("status") in ("fresh", "adopted", "warn", "off", "n/a", "check") and not plan.get(i, {}).get("retry")
                    for i in side_steps)
        ev.legacy("side", done=n - 1, total=N, msg=("side %d unchanged, reusing it" % n) if fresh else ("building side %d" % n))

    # ------------------------------------------------------------------------------------------------ plan / run
    def _needed(self, g, to=None):
        return g.ancestors(self.targets(g, to))

    def _force(self, g, force=(), from_=None):
        f = set()
        for t in force or ():
            ids = self.node_ids(g, t)
            f.update(ids)
            for i in ids:          # forcing a node with ops re-runs its ops too
                st = g.by_id[i]
                if st.kind != "op": f.update(g.chain(st.id))
        if from_:
            f.update(g.descendants(self.node_ids(g, from_)))
        return f

    def plan(self, to=None, force=(), from_=None, with_params=None, recipe=None, _fresh=True):
        if _fresh: self._begin()
        g = self.graph(recipe, with_params); needed = self._needed(g, to); fs = self._force(g, force, from_)
        _outs, info, _ = self._walk(g, "plan", needed, fs)
        for i in fs:
            if i in info and info[i]["status"] not in ("waiting", "off", "n/a"): info[i]["status"] = "stale"
        # anything below a forced/stale step is waiting in reality
        stale = [i for i, v in info.items() if v["status"] == "stale"]
        below = g.descendants(stale) - set(stale)
        for i in below:
            if i in info and info[i]["status"] not in ("off", "n/a"): info[i]["status"] = "waiting"
        hist = self.store.state()
        to_run = [st for st in g.steps if st.id in info and (info[st.id]["status"] in ("stale", "waiting") or info[st.id].get("retry"))]
        est = self._wall_estimate(to_run, {st.id: self._weight(st, hist) for st in to_run})
        return g, info, to_run, est

    def _wall_estimate(self, to_run, secs):
        """Wall seconds for these steps when ready sides run side by side: a dry run of the same admission the build uses
        (two GPU fusions, RAM and CPU shared). Planning does not ask the GPU, so a busy GPU can make the build a bit slower."""
        import heapq, itertools
        from .parallel import Admission
        try: budget = resources.budget(self.runner.memory_gb)
        except Exception: return sum(secs.values())
        gpu = max(1, min(2, int(os.environ.get("POINTYOINK_BUILD_GPU_SLOTS", "2"))))
        adm = Admission(budget, gpu, max(1, int(os.environ.get("POINTYOINK_BUILD_WORKERS", "100000"))))
        ids = {st.id for st in to_run}; done = set(); pending = list(to_run); running = []; t = 0.0; order = itertools.count()
        while pending or running:
            for st in list(pending):
                if any(i in ids and i not in done for i in st.inputs.values()): continue
                if not adm.accepts(st) and running: continue          # alone it always runs, as in the build
                adm.take(st); pending.remove(st); heapq.heappush(running, (t + secs[st.id], next(order), st))
            if not running: break
            t, _, st = heapq.heappop(running); adm.release(st); done.add(st.id)
        return t

    @entry
    def run(self, to=None, from_=None, force=(), with_params=None, publish=True, legacy=False, dry_run=False, quiet=False, defer_report=False):
        t0 = time.time()
        self._writable()
        self.runner.reset()                 # a cancel stops the run it was made in, not every run after it
        self._begin()
        if from_:
            g0 = self.graph(None, with_params); ids = self.node_ids(g0, from_)
            _, info0, _ = self._walk(g0, "plan", g0.ancestors(ids) - set(ids))
            bad = [i for i, v in info0.items() if v["status"] in ("stale", "waiting")]
            if bad: raise ValueError("--from %s: %s is not built yet (run --to it first)" % (from_, bad[0]))
        self._lock()
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        ev = Events(self.p, run_id, legacy=legacy, out=self.out, on_event=self.on_event, quiet_new=quiet)
        self._ev = ev; self._run_lock=ev.lock
        try:
            self.store.clean_tmp()
            recipe = self.recipe()
            follow_on = bool(defer_report and publish and to in (None, "all", "report"))
            if follow_on: to = "meshed"
            g, info, to_run, est = self.plan(to, force, from_, with_params, recipe, _fresh=False)
            hist = self.store.state()
            rs = self._rs(ev, info, g); rs["run_id"] = run_id; self._run = rs
            rs["weights"] = {st.id: self._weight(st, hist) for st in to_run}
            rs["total_w"] = max(1e-9, sum(rs["weights"].values()))
            ev.emit("plan", run=run_id, to_run=len(to_run), cached=sum(1 for v in info.values() if v["status"] in ("fresh", "adopted", "warn") and not v.get("retry")),
                    est_s=round(est), nodes=[dict({"id": i, "status": v["status"]}, **({"retry": True} if v.get("retry") else {}))
                                             for i, v in info.items()])
            if legacy: ev.legacy("sides", count=len(g.sides))
            if dry_run: return RunResult(True, time.time() - t0)
            needed = self._needed(g, to); fs = self._force(g, force, from_)
            outs, info2, failed = self._walk(g, "run", needed, fs, ev, rs)
            self.memo.save()
            res = RunResult(failed is None, time.time() - t0, failed, ran=rs["ran"], cached=rs["cached"], error=rs["error"])
            if failed is None and follow_on:
                model, generation = self._publish_mesh(g, outs, ev, recipe, with_params, "report" in fs)
                res.model = model
                maintenance_start = time.monotonic()
                try: self.gc()
                except Exception: pass
                ev.emit("done", model=model, preview=None, sides=len(g.sides), base=None,
                        secs=round(time.time() - t0), coverage_pending=True, generation=generation,
                        maintenance_s=round(time.monotonic() - maintenance_start, 4))
            elif failed is None and "report" in outs and "report" in needed:
                try:
                    model = self._publish(g, outs, ev, publish)
                except DeniedError as e:            # a publish target resolves into a protected folder: nothing written
                    ev.emit("error", msg=str(e)[:300], node="publish")
                    res.ok = False; res.failed = "publish"; res.error = str(e)
                else:
                    res.model = model
                    try: self._post(g, outs, ev)
                    except Exception as e: ev.emit("node", id="post", group="post", side=None, n=None, state="warn", error=str(e)[-300:])
                    base = None
                    try: base = rs["pos"][jload(outs["lineup.adjust"].files["lineup"])["base"]]
                    except Exception: pass
                    ev.emit("done", model=os.path.join(self.p.root, "model.ply"), preview=os.path.join(self.p.root, "model.png"),
                            sides=len(g.sides), base=base, secs=round(time.time() - t0))
                    try: self.gc()
                    except Exception: pass
            elif failed is None:
                ev.emit("done", model=None, preview=None, sides=len(g.sides), base=None, secs=round(time.time() - t0))
            self.memo.save()
            res.secs = time.time() - t0
            return res
        finally:
            self._run = None; self._ev = None; ev.close(); self._unlock()

    def cancel(self): self.runner.cancel()

    # ------------------------------------------------------------------------------------------------ lock
    def _lock(self):
        self._writable()
        lp = os.path.join(self.p.stages, "lock"); f = open(lp, "a+")
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try: pid = jload(os.path.join(self.p.stages, "lock.json")).get("pid")
            except Exception: pid = "?"
            f.close(); raise Locked(pid)
        self._lockf = f
        jdump({"pid": os.getpid(), "cmd": " ".join(sys.argv), "started": time.time()}, os.path.join(self.p.stages, "lock.json"))

    def _unlock(self):
        if self._lockf:
            try: os.remove(os.path.join(self.p.stages, "lock.json"))
            except OSError: pass
            fcntl.flock(self._lockf, fcntl.LOCK_UN); self._lockf.close(); self._lockf = None

    # ------------------------------------------------------------------------------------------------ publish
    PUBLISHED = (("mesh", "model.ply"), ("preview", "model.png"), ("needs", "model_needs.ply"), ("needs_png", "model_needs.png"),
                 ("needs_json", "model_needs.json"))

    @contextmanager
    def _publication_lock(self):
        path = os.path.join(self.p.stages, "publication.lock")
        self.p.guard_write(path)
        with open(path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(handle, fcntl.LOCK_UN)

    def _publish_mesh(self, g, outs, ev, recipe, with_params=None, force_report=False):
        mo = outs[g.final("meshed")]
        record = {"digest": mo.bdigest, "node_key": (mo.meta or {}).get("key"),
                  "node": mo.step, "generation": uuid.uuid4().hex, "coverage_pending": True,
                  "report_key": self.step_key(g.by_id["report"], outs),
                  "recipe": recipe, "with_params": with_params or {}, "force_report": force_report, "t": time.time()}
        model = os.path.join(self.p.root, "model.ply")
        pp = os.path.join(self.p.stages, "published.json")
        for _, filename in self.PUBLISHED: self.p.guard_write(os.path.join(self.p.root, filename))
        self.p.guard_write(pp)
        with self._publication_lock():
            shutil.copyfile(mo.files["mesh"], model + ".tmp"); os.replace(model + ".tmp", model)
            for _, filename in self.PUBLISHED[1:]:
                path = os.path.join(self.p.root, filename)
                if os.path.lexists(path): os.remove(path)
            self._model_frames(outs)
            jdump(record, pp + ".tmp"); os.replace(pp + ".tmp", pp)
        ev.emit("published", model=model, preview=None, needs=None, digest=mo.bdigest,
                generation=record["generation"], coverage_pending=True)
        return model, record["generation"]

    @entry
    def measure_published(self, generation):
        """Measure immutable cached inputs without holding the foreground build lock.

        A newer publication discards this result, including a byte-identical mesh
        with a different line-up. Cached results remain useful for later builds.
        """
        self._writable(); self._begin()
        pp = os.path.join(self.p.stages, "published.json")
        pub = jload(pp)
        if pub.get("generation") != generation: return {"discarded": True}
        g = self.graph(pub["recipe"], pub.get("with_params"))
        outs, _, _ = self._walk(g, "plan", self._needed(g, "report"))
        st = g.by_id["report"]
        if any(i not in outs for i in st.inputs.values()): return {"discarded": True}
        key = self.step_key(st, outs)
        if key != pub.get("report_key"): return {"discarded": True}
        previous = outs.get("report")
        if previous is None or pub.get("force_report") or (previous.meta or {}).get("transient"):
            outs["report"] = self._execute(st, key, outs, None, None, prev=previous.meta if previous else None,
                                           forced=bool(pub.get("force_report")))
        rep = outs["report"]
        with self._publication_lock():
            current = jload(pp)
            if any(current.get(k) != pub.get(k) for k in ("generation", "digest", "report_key")):
                return {"discarded": True}
            for name, filename in self.PUBLISHED[1:]:
                destination = os.path.join(self.p.root, filename); self.p.guard_write(destination)
                source = rep.files.get(name)
                if source and os.path.exists(source):
                    shutil.copyfile(source, destination + ".tmp"); os.replace(destination + ".tmp", destination)
                elif os.path.lexists(destination): os.remove(destination)
            incomplete = bool((rep.meta or {}).get("transient") or (rep.meta or {}).get("retry"))
            current.update(report=rep.bdigest, coverage_pending=False, coverage_error=incomplete)
            jdump(current, pp + ".tmp"); os.replace(pp + ".tmp", pp)
        if not self.runner.cancelled:
            self._post(g, outs, None)
        return {"generation": generation, "discarded": False, "error": "Coverage could not finish." if incomplete else None}

    def _publish(self, g, outs, ev, publish=True):
        with self._publication_lock():
            return self._publish_locked(g, outs, ev, publish)

    def _publish_locked(self, g, outs, ev, publish=True):
        """Copy the model and its report files to the project root (same names as build_project). A file the new report
        does not have (no scan-here map this time) is removed, so an older map never sits next to a newer model. Every
        target is checked against POINTYOINK_STAGES_DENY by its real path first (DeniedError: nothing is written)."""
        mo = outs[g.final("meshed")]; rep = outs["report"]
        model = os.path.join(self.p.root, "model.ply")
        pp = os.path.join(self.p.stages, "published.json")
        try: pub = jload(pp)
        except Exception: pub = {}
        if not publish: return None
        srcs = {"mesh": mo.files["mesh"], **{k: rep.files.get(k) for k in ("preview", "needs", "needs_png", "needs_json")}}
        targets = [(srcs.get(k), os.path.join(self.p.root, fn)) for k, fn in self.PUBLISHED]
        for _src, dst in targets + [(None, pp)]: self.p.guard_write(dst)
        stale = [dst for src, dst in targets if not (src and os.path.exists(src)) and os.path.lexists(dst)]
        if pub.get("digest") == mo.bdigest and pub.get("report") == rep.bdigest and os.path.exists(model) and not stale:
            if not os.path.exists(os.path.join(self.p.root, "build", "frames.json")): self._model_frames(outs)       # a build published before this existed
            return model

        def put(src, dst):
            if not src or not os.path.exists(src):
                if os.path.lexists(dst): os.remove(dst)
                return None
            tmp = dst + ".tmp"; shutil.copyfile(src, tmp); os.replace(tmp, dst); return dst
        for src, dst in targets: put(src, dst)
        rec = {"digest": mo.bdigest, "report": rep.bdigest, "node_key": (mo.meta or {}).get("key"), "node": mo.step, "t": time.time()}
        jdump(rec, pp + ".tmp"); os.replace(pp + ".tmp", pp)
        self._model_frames(outs)
        ev.emit("published", model=model, preview=os.path.join(self.p.root, "model.png"),
                needs=os.path.join(self.p.root, "model_needs.png") if rep.files.get("needs_png") else None, digest=mo.bdigest)
        return model

    def _model_frames(self, outs):
        """Where each side sits in the model just published (build/frames.json: folder name -> side-to-model 4x4, the file Touch up reads to track
        a new side against the built model)."""
        import model_frames
        try: sj = jload(outs["lineup.adjust"].files["sides"])
        except Exception: return
        work = os.path.join(self.p.root, "build")
        try:
            self.p.guard_write(os.path.join(work, "frames.json"))                      # a denied path: nothing is written, nothing fails
            os.makedirs(work, exist_ok=True)
            model_frames.write(work, {os.path.basename(str(s["folder"]).rstrip("/")): s["T"] for s in sj["sides"] if s.get("T") is not None})
            model_frames.commit(work)
        except Exception: pass

    def _post(self, g, outs, ev):
        """Stats and a 1-view preview for the global artifacts of this run that lack them (one small job, cap 3)."""
        ids = ["lineup.adjust", "fused"] + g.chain("fused") + ["meshed"] + g.chain("meshed")
        if g.final("meshed") not in ids: ids.append(g.final("meshed"))
        targets = []
        for i in ids:
            o = outs.get(i)
            if o is None or o.meta is None or o.alias_of or o.step != i: continue
            d = o.meta["_dir"]
            if os.path.exists(os.path.join(d, "preview.png")) and o.meta.get("post"): continue
            st = g.by_id[i]
            targets.append({"dir": d, "kind": o.kind, "files": o.files, "node": i, "label": "%s %s" % (i, st.method.name if st.method else ""),
                            "sides_json": outs["lineup.adjust"].files["sides"] if i == "lineup.adjust" else None, "views": 6 if i == "lineup.adjust" else 1})
        if not targets: return
        r = self.job("preview:post_job", {"targets": targets, "project": self.p.root}, mem_gb=3)
        for t in targets:
            pj = os.path.join(t["dir"], "post.json")
            try: stats = jload(pj)
            except Exception: continue
            self.store.update_meta(t["dir"], stats=stats, post=True)

    @staticmethod
    def _small(files, limit=1_500_000):
        """True when every mesh / point file here has fewer than limit faces / points (header reads only): the jobs
        that read them then stay well under 1 GB (measured), so they count as light."""
        from .plyio import counts, points_count
        for f in (files or {}).values():
            try:
                if f.endswith(".npz"): n = points_count(f)
                elif f.endswith(".ply"): c = counts(f); n = max(c.get("face", 0), c.get("vertex", 0) if not c.get("face") else 0)
                else: continue
            except Exception: return False
            if n >= limit: return False
        return True

    @entry
    def job(self, fn, payload, mem_gb=3, gpu=False, on_line=None, light=None):
        """A job outside any step (post, preview, detect, view): runs in a scratch folder under stages/tmp. Light (may
        skip a crowded wrap) when the result it reads is small."""
        self._writable()
        if light is None:
            fl = dict(payload.get("files") or {})
            for t in payload.get("targets", []): fl.update({"%s:%s" % (t["node"], k): v for k, v in t["files"].items()})
            if payload.get("in"): fl["in"] = payload["in"]
            light = self._small(fl) and len(payload.get("targets", [])) <= 8
        d = os.path.join(self.p.stages, "tmp", "job.%s.%d" % (uuid.uuid4().hex[:8], os.getpid())); os.makedirs(d)
        try:
            jp = os.path.join(d, "job.json"); rp = os.path.join(d, "result.json")
            jdump(dict(payload, fn=fn, out_dir=d, result=rp), jp)
            self.runner.run([sys.executable, JOB, jp], mem_gb, gpu, log=os.path.join(d, "log.txt"), on_line=on_line, light=light)
            try: r = jload(rp)
            except Exception: r = {}
            r["_dir"] = d
            return r
        finally:
            if fn != "tools.common:tool_job": shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------ status
    def status(self, with_params=None):
        self._begin()
        r = self.recipe(); g = self.graph(r, with_params)
        _o, info, _ = self._walk(g, "plan", None)          # every step, left-out sides too
        needed = self._needed(g); hist = self.store.state()
        todo = [st for st in g.steps if st.id in needed and (info.get(st.id, {}).get("status") in ("stale", "waiting")
                                                              or info.get(st.id, {}).get("retry"))]
        est = self._wall_estimate(todo, {st.id: self._weight(st, hist) for st in todo})
        groups = []
        raw = [NodeInfo("raw@" + s.id, "fresh", "-", group="raw", side=s.id, n=s.n, stats=self.p.raw_stats(s.id),
                        summary="" if s.active else "left out") for s in g.sides]
        groups.append(Group("raw", "Raw Data", raw))
        for gname, label in G.GROUPS[1:]:
            nodes = []
            for st in g.steps:
                if st.group != gname: continue
                i = info.get(st.id, {"status": "waiting"}); meta = i.get("meta") or {}
                if not getattr(st, "active", True) and i["status"] in ("stale", "waiting"):
                    i = dict(i, status="off")           # a left-out side's steps are not part of any build: never "will rebuild"
                pv = os.path.join(meta.get("_dir", ""), "preview.png") if meta else None
                nodes.append(NodeInfo(st.id, i["status"], st.method.name if st.method else "-", i.get("key"), st.params,
                                      meta.get("stats"), {k: v["file"] for k, v in (meta.get("outputs") or {}).items()},
                                      pv if pv and os.path.exists(pv) else None, None, meta.get("secs"), bool(meta.get("adopted")),
                                      gname, st.side, st.n, st.method.summary(st.params) if st.method else "", meta.get("warn"),
                                      bool(i.get("retry"))))
            if gname == "fused":        # tool lists of the fused methods not in use: kept, shown, not run
                for lk, _j, o in self.parked(r):
                    nodes.append(NodeInfo("%s#%s" % (lk, o.get("id")), "parked", o.get("tool", "-"), group="fused",
                                          params=o.get("params"), summary="parked: used when fused.method=%s" % lk.split(".", 1)[1]))
            if gname == "textured": nodes = [NodeInfo("textured", "n/a", "-", summary="colour comes later", group="textured")]
            groups.append(Group(gname, label, nodes))
        return groups, g, est

    # ------------------------------------------------------------------------------------------------ outputs
    def _resolve(self, node, key=None):
        """-> (step, Out) of a node's current artifact, or of the stored version key (key16). An alias resolves to
        what it passes through."""
        self._begin()
        g = self.graph(); outs, _info, _ = self._walk(g, "plan", None)
        node = self.canon_token(node, g)
        tgt = g.final(node) if node in g.finals else node
        if tgt not in g.by_id: tgt = self.node_ids(g, node)[-1]
        st = g.by_id[tgt]
        if key:
            d = os.path.join(self.store.nodes, st.store, key[:16])
            meta = jload(os.path.join(d, "meta.json")); meta["_dir"] = d
            if meta.get("alias_of"): raise ValueError("version %s of %s is an alias of its input" % (key, node))
            return st, self._out_meta(st, meta, outs)
        o = outs.get(st.id)
        if o is not None and o.step != st.id and o.step in g.by_id: st = g.by_id[o.step]
        return st, o

    def output(self, node, kind=None, key=None):
        st, o = self._resolve(node, key)
        if o is None: return None
        name = kind or next((k for k in ("mesh", "points", "lineup", "placement", "transforms", "pairs", "preview") if k in o.files), None)
        return o.files.get(name)

    def versions(self, node):
        self._begin()
        g = self.graph(); node = self.canon_token(node, g)
        st = g.by_id[g.final(node) if node in ("fused", "meshed") else self.node_ids(g, node)[-1]]
        while st.alias == "parent" and "parent" in st.inputs:
            st = g.by_id[st.inputs["parent"]]
        out = []
        sd = os.path.join(self.store.nodes, st.store)
        if not os.path.isdir(sd): return out
        for k in sorted(os.listdir(sd)):
            try: meta = jload(os.path.join(sd, k, "meta.json"))
            except Exception: continue
            if st.kind != "op" or meta.get("op", {}).get("id") == st.op.get("id"):
                out.append({"key16": k, "node": meta.get("node"), "method": meta.get("method") or meta.get("tool"), "params": meta.get("params"),
                            "stats": meta.get("stats"), "created": meta.get("created"), "pinned": meta.get("pinned"),
                            "adopted": meta.get("adopted"), "dir": os.path.join(sd, k),
                            "last_used": os.path.getmtime(os.path.join(sd, k, "meta.json"))})
        return sorted(out, key=lambda x: -(x["created"] or 0))

    def pin(self, node, k16, on=True):
        for v in self.versions(node):
            if v["key16"] == k16:
                self.store.update_meta(v["dir"], pinned=on); return True
        return False

    def _extra(self, color):
        extra = {}
        if color in ("sides", "scanhere"):
            g = self.graph(); _o, full, _ = self._walk(g, "plan", None)
            lu = full.get("lineup.adjust", {}).get("meta")
            if color == "sides" and lu: extra["sides_json"] = os.path.join(lu["_dir"], lu["outputs"]["sides"]["file"])
            rp = full.get("report", {}).get("meta")
            if color == "scanhere" and rp and "needs" in rp.get("outputs", {}):
                extra["needs"] = os.path.join(rp["_dir"], rp["outputs"]["needs"]["file"])
        return extra

    @entry
    def preview(self, node, color="shaded", views=1, key=None):
        st, o = self._resolve(node, key)
        if o is None or o.meta is None: raise ValueError("%s is not built yet" % node)
        d = o.meta["_dir"]; name = "preview.png" if (color == "shaded" and views == 1) else "preview_%s_%d.png" % (color, views)
        png = os.path.join(d, name)
        if os.path.exists(png): return png
        extra = self._extra(color)
        if color == "removed" and st.kind == "op":
            g = self.graph(); outs, _i, _ = self._walk(g, "plan", None)
            par = outs.get(st.inputs["parent"])
            if par is not None: extra.update(parent=par.files, tool=st.method.name, params=st.params)
        self.job("preview:preview_job", {"files": o.files, "kind": o.kind, "dir": d, "png": png, "color": color, "views": views,
                                         "label": "%s %s" % (st.id, st.method.name if st.method else ""), "project": self.p.root,
                                         **extra}, mem_gb=3)
        return png

    @entry
    def view(self, node, color="shaded", key=None, dry_run=False):
        st, o = self._resolve(node, key)
        if o is None or o.meta is None: raise ValueError("%s is not built yet" % node)
        d = o.meta["_dir"]
        vp = os.path.join(d, "view.ply" if color == "shaded" else "view_%s.ply" % color)
        if not os.path.exists(vp):
            self.job("preview:view_job", {"files": o.files, "kind": o.kind, "dir": d, "out": vp, "color": color, **self._extra(color)}, mem_gb=3)
        from .methods import script
        cmd = [sys.executable, script("viewer.py"), vp, "%s %s (%s)" % (st.id, st.method.name if st.method else "", color)]
        if dry_run: return cmd
        import subprocess
        env = dict(os.environ); env["XMODIFIERS"] = "@im=none"
        subprocess.Popen(cmd, start_new_session=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cmd

    # ------------------------------------------------------------------------------------------------ detect + delete
    def _parent_for_new_op(self, lk, at=None):
        """The step whose output a new op in list lk would get (the list's end), and its Out. The Fused stage's delete
        list ("fused") starts after the current method's tools."""
        self._begin()
        g = self.graph(); outs, info, _ = self._walk(g, "plan", None)
        if lk.startswith("fused.") and lk != "fused." + g.by_id["fused"].method.name:
            raise ValueError("%s is parked (fused.method is %s)" % (lk, g.by_id["fused"].method.name))
        ids = g.lists.get(lk, [])
        if at is not None: ids = ids[:int(at)]
        if ids: pid = ids[-1]
        elif lk == "fused":
            tl = g.lists.get("fused." + g.by_id["fused"].method.name, []); pid = tl[-1] if tl else "fused"
        elif lk.startswith("fused."): pid = "fused"
        else: pid = lk
        if pid not in outs: raise ValueError("%s is not built yet: run first" % pid)
        return g, pid, outs[pid], outs

    @entry
    def detect(self, list_key, op):
        """Run a tool in detect mode on the current end of the list: nothing changes. -> {count, png, mask}."""
        r = self.recipe(); lk = self.list_key(list_key, r)
        return self._detect(lk, op, *self._parent_for_new_op(lk))

    def _detect(self, lk, op, g, pid, parent, outs):
        kind = parent.kind
        tool = G.get_tool(op["tool"], kind)
        if tool is None or not getattr(tool, "detect", False) or kind not in getattr(tool, "kinds", ()):
            raise ValueError("%s has no detect on %s results (detect: isolation, overlap, delete)" % (op["tool"], kind))
        params = PR.resolve_tool(op["tool"], kind, op.get("params") or {}) if op["tool"] != "delete" else {}
        T_now = None
        if "lineup.adjust" in outs: T_now = jload(outs["lineup.adjust"].files["lineup"])["T"]
        fkey = "points" if "points" in parent.files else "mesh"
        full = dict(op); full.setdefault("frame", "side" if lk.startswith("side.keep@") else "model")
        res = self.job("tools.common:tool_job", {"tool": op["tool"], "kind": kind, "in": parent.files[fkey], "params": params,
                                                 "op": full, "parent_digest": parent.bdigest, "stages": self.p.stages, "detect": True,
                                                 "T_now": T_now}, mem_gb=4)
        dd = os.path.join(self.p.stages, "detect"); os.makedirs(dd, exist_ok=True)
        stem = os.path.join(dd, "%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), lk.replace("@", "-"), op["tool"]))
        png = mask = None
        if res.get("png") and os.path.exists(res["png"]): png = stem + ".png"; shutil.copyfile(res["png"], png)
        if res.get("mask") and os.path.exists(res["mask"]): mask = stem + ".npy"; shutil.copyfile(res["mask"], mask)
        shutil.rmtree(res.get("_dir", ""), ignore_errors=True)
        return {"count": res.get("count"), "png": png, "mask": mask, "n": res.get("n"), "area_mm2": res.get("area_mm2"),
                "stats": res.get("stats") or {}}

    @entry
    def delete(self, node, shape, mask_pts=None, radius=0.5, detect=False):
        """node: side@<side>, fused or meshed. shape: {"shape": "box", "box": {...}} / {"shape": "sphere", ...} / mask.
        A delete on fused goes to the Fused stage's own list (kept whatever the fused method)."""
        r = self.recipe()
        # a delete on fused (or on fused.<method>) goes to the Fused stage's own list: kept whatever the method (B1)
        lk = "fused" if node == "fused" or str(node).startswith("fused.") else self.list_key(node, r)
        g, pid, parent, outs = self._parent_for_new_op(lk)
        frame = "side" if lk.startswith("side.keep@") else "model"
        op = {"tool": "delete", "on": True, "frame": frame}; op.update(shape)
        if frame == "model":
            lu = jload(outs["lineup.adjust"].files["lineup"])
            op["anchor"] = {"order": lu["order"], "T": {s: lu["T"][s] for s in lu["order"]}}
        if op["shape"] == "mask":
            fkey = "points" if "points" in parent.files else "mesh"
            res = self.job("tools.edits:make_mask_job", {"kind": parent.kind, "in": parent.files[fkey], "mask": mask_pts, "radius": radius,
                                                         "input_digest": parent.bdigest, "regions": os.path.join(self.p.stages, "regions")}, mem_gb=3)
            op["region"] = res["region"]; op["radius"] = radius
        # recorded BEFORE the preview: detect then replays exactly what the run will delete (the exact indices of a
        # mask while its input is unchanged), not a geometric approximation of it
        op["recorded"] = {"input_digest": parent.bdigest}
        det = self._detect(lk, op, g, pid, parent, outs)
        if detect: return det
        op["recorded"].update(count=det["count"], area_mm2=det.get("area_mm2"))
        oid = self._add_op(lambda _r: lk, op, note="%s: delete %s (%d %s)" % (self._list_label(lk), op["shape"], det["count"] or 0,
                                                                            "points" if parent.kind == "points" else "faces"))
        return dict(det, id=oid)

    # ------------------------------------------------------------------------------------------------ adopt
    @entry
    def adopt_legacy(self, accept_versions=None, refit_old_pairs=False):
        """Adopt build/ (section 18). Pair fits are taken whatever pair_all version made them when both side files'
        stamps match the adopted sides (a VERSION bump does not by itself make an old fit wrong; `run --force
        lineup.pairs` refits). refit_old_pairs: only take fits of the current version (plus accept_versions)."""
        from .legacy import Adopter
        self._writable(); self._begin()
        ad = Adopter(self.p, accept_versions, refit_old=refit_old_pairs)
        if not ad.ok: return {"adopted": [], "stale": [], "why": ad.why}
        self._lock()
        try:
            self.store.clean_tmp()
            if refit_old_pairs: ad.notes.append(self._drop_old_pair_fits())
            g = self.graph(); rs = self._rs(None, {}, g)
            outs, info, _ = self._walk(g, "adopt", None, rs=rs, adopter=ad)
            ad.before_light(self, g.by_id["lineup.pairs"], outs, g)      # the fits of the adopted sides, when pairs could not run
            self.memo.save()
        finally:
            self._unlock()
        adopted = [i for i, v in info.items() if v["status"] == "adopted"]
        stale = [i for i, v in info.items() if v["status"] in ("stale", "waiting")]
        return {"adopted": adopted, "stale": stale, "notes": ad.notes}

    def _drop_old_pair_fits(self):
        """adopt --refit-old-pairs on a project that took an older pair_all's fits before: remove them from the pair cache
        and drop the stored lineup.pairs results made with them, so the next run refits exactly those pairs."""
        from .methods.pairs import PairCache
        c = PairCache(os.path.join(self.p.stages, "cache", "pairs.json")); n = c.drop_old()
        if n: c.save()
        gone = 0
        for store, _k, d, meta in self.store.artifacts():
            if store == "lineup.pairs" and meta and (meta.get("stats") or {}).get("old_fits"):
                shutil.rmtree(d, ignore_errors=True); gone += 1
        return "refit old pairs: %d older fit%s removed from the pair cache, %d line-up result%s dropped (the next run refits them)" % (
            n, "" if n == 1 else "s", gone, "" if gone == 1 else "s")

    # ------------------------------------------------------------------------------------------------ gc
    def gc(self, max_gb=None, keep=3):
        self._writable(); self._begin()
        budget = float(max_gb if max_gb is not None else os.environ.get("POINTYOINK_STAGES_CACHE_GB", 8)) * 1024 ** 3
        protected = set()
        entries, cur = self.hist.entries()
        recipes = [self.recipe()] + [e["recipe"] for k, e in enumerate(entries) if abs(k - cur) <= 10]
        for r in recipes:
            try:
                g = self.graph(r); _o, info, _ = self._walk(g, "plan", None)
            except Exception: continue
            for v in info.values():
                if v.get("meta"): protected.add(os.path.realpath(v["meta"]["_dir"]))
        arts = self.store.artifacts(); by_store = {}
        for store, k, d, meta in arts:
            if meta is None: shutil.rmtree(d, ignore_errors=True); continue
            if meta.get("pinned"): protected.add(os.path.realpath(d))
            by_store.setdefault(store, []).append((os.path.getmtime(os.path.join(d, "meta.json")), d))
        for store, lst in by_store.items():
            for _t, d in sorted(lst, reverse=True)[:keep]: protected.add(os.path.realpath(d))
        sizes = {d: self.store.dir_bytes(d) for _s, _k, d, m in arts if m is not None}
        from .local_points_cache import collect as collect_side_cache
        cache = collect_side_cache(self.p.stages, float("inf"), 0)
        total = sum(sizes.values()) + cache["bytes"]; removed = []
        # transient files first: viewer exports, on-request point PLYs, previews of old keys
        for _s, _k, d, m in arts:
            if m is None: continue
            for f in os.listdir(d):
                if f.startswith("view") and f.endswith(".ply") or f == "points.ply" or \
                        (os.path.realpath(d) not in protected and f.endswith(".png") and f.startswith(("preview_", "detect"))):
                    p = os.path.join(d, f); b = os.path.getsize(p); os.remove(p); total -= b; sizes[d] -= b
        protected_bytes = sum(size for d, size in sizes.items() if os.path.realpath(d) in protected) + cache["protected_bytes"]
        budget = max(8 * 1024 ** 3, budget, protected_bytes + 2 * 1024 ** 3)
        if total > budget:
            cands = sorted((os.path.getmtime(os.path.join(d, "meta.json")), d) for _s, _k, d, m in arts
                           if m is not None and os.path.realpath(d) not in protected)
            for _t, d in cands:
                if total <= budget: break
                shutil.rmtree(d, ignore_errors=True); total -= sizes.get(d, 0); removed.append(d)
        node_bytes = total - cache["bytes"]
        cache = collect_side_cache(self.p.stages, budget, node_bytes)
        total = node_bytes + cache["bytes"]; removed.extend(cache["removed"])
        self.memo.forget_missing(); self.memo.save()
        result = {"removed": removed, "total_gb": round(total / 1024 ** 3, 3), "budget_gb": round(budget / 1024 ** 3, 3),
                  "protected": len(protected), "protected_gb": round(protected_bytes / 1024 ** 3, 3)}
        if cache["bytes"] or cache["removed"]:
            result.update(side_cache_gb=round(cache["bytes"] / 1024 ** 3, 3),
                          protected_side_caches=cache["protected"], over_budget_gb=round(max(0, total - budget) / 1024 ** 3, 3))
        return result
