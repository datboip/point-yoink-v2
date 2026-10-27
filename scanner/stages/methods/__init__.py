# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plug-in contract: a Method computes one node, a Tool applies one op on top of a node's result. Both register here.

A Method either returns subprocess commands from argv(ctx) (run in order under the runner's memory cap) or does light
work in run(ctx). execute(ctx) returns a Result dict:
    {"outputs": {name: file name in ctx.out_dir}, "stats": {...}, "warn": [...], "alias_of": input name or None}
Methods never read build/ and never write outside ctx.out_dir. Bake winners slot in as new Method classes
(fused.<name> producing "mesh" or "points", meshed.<name>, lineup.refine.<name>).

Code in the keys: every Method and Tool names the repo scripts it runs (`scripts`), the stages modules whose code makes
its output (`modules`) and the packages whose version matters (`libs`). The engine hashes their contents once per run
into each step's key, so editing fuse.py, keep_part.py, pair_all.py, ... or a method module re-runs exactly the steps
that use it (and early cutoff stops the wave where the bytes come out the same). Output-shaping tables and formulas in
params.py (the Poisson quality plan, the overlap schedule, the mesh smooth lambda) are named in `tables` and their source
is hashed the same way. The `version` integers stay as a manual override. POINTYOINK_STAGES_REPO=<dir>: take the repo scripts from <dir> (same layout: fuse.py, process.py,
align.py, viewer.py at the top, the rest under scanner/) wherever a copy exists there; for trying a script edit on a
copy without touching the repo."""
import glob, hashlib, inspect, os, subprocess, sys

from .. import params as PR

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # scanner/stages
SCANNER = os.path.dirname(HERE)
REPO = os.path.dirname(SCANNER)
JOB = os.path.join(HERE, "job.py")
ROOT_SCRIPTS = ("fuse.py", "process.py", "viewer.py", "align.py", "registration_metrics.py")


def repo_override():
    d = os.environ.get("POINTYOINK_STAGES_REPO")
    return os.path.abspath(os.path.expanduser(d)) if d else None


def script(name):
    """A repo script: fuse.py, process.py, align.py and viewer.py live in the repo root, the rest in scanner/ (a copy under
    POINTYOINK_STAGES_REPO wins when it exists)."""
    rel = name if name in ROOT_SCRIPTS else os.path.join("scanner", name)
    ov = repo_override()
    if ov and os.path.exists(os.path.join(ov, rel)): return os.path.join(ov, rel)
    return os.path.join(REPO, rel)


def scanner_dir():
    """The folder pair_all.py and its helpers are imported from (the override copy when there is one)."""
    ov = repo_override()
    if ov and os.path.exists(os.path.join(ov, "scanner", "pair_all.py")): return os.path.join(ov, "scanner")
    return SCANNER


def stage_file(rel): return os.path.join(HERE, rel)


# params.py names whose values / source shape a step's output (Method.tables / Tool.tables)
TABLES = {"poisson": ("poisson_plan", "QUALITY_GRID", "QUALITY_STEPS", "POISSON_SCALE", "DEPTH_MIN", "DEPTH_MAX"),
          "overlap": ("overlap_schedule",),
          "mesh_smooth": ("mesh_smooth_lambda",)}


def table_digest(name):
    """"params.<name>=<digest>" of a TABLES entry: the source of its functions and the repr of its constants, read
    now (so an edit, or a test that patches a table, changes the key)."""
    h = hashlib.blake2b(digest_size=8)
    for attr in TABLES[name]:
        obj = getattr(PR, attr)
        try: src = inspect.getsource(obj) if callable(obj) else repr(obj)
        except (OSError, TypeError): src = repr(obj)
        h.update(attr.encode() + b"\0" + src.encode() + b"\0")
    return "params.%s=%s" % (name, h.hexdigest())


def argv_digest(node, method, params=None, skip=(), schema=None):
    """"argv=<digest>" of how a node's params become script flags: legacy_flags' source and, per param of the node, its
    emitted flag for the resolved value (Method.argv_params)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(inspect.getsource(legacy_flags).encode())
    schema=PR.node_params(node,method) if schema is None else schema
    if params is None:params={p.name:p.default for p in schema}
    h.update(repr(legacy_flags(params, schema, skip=skip)).encode())
    return "argv=" + h.hexdigest()


def lib_version(name):
    """Installed version of a package (read from its metadata, nothing imported); pymeshlab from the PyMeshLab python."""
    if name == "pymeshlab":
        from .mesh import pml_python
        py = pml_python()
        if not py: return "-"
        root = os.path.dirname(os.path.dirname(py))            # <venv>/bin/python: never realpath (a venv links to /usr)
        di = sorted(glob.glob(os.path.join(root, "lib", "python*", "site-packages", "pymeshlab-*.dist-info")))
        if di: return os.path.basename(di[-1])[len("pymeshlab-"):-len(".dist-info")]
        try:
            return subprocess.run([py, "-c", "import importlib.metadata as m; print(m.version('pymeshlab'))"], capture_output=True,
                                  text=True, timeout=30).stdout.strip() or "?"
        except Exception: return "?"
    try:
        from importlib.metadata import version
        return version(name)
    except Exception: return "?"


def legacy_flags(params, schema, skip=()):
    """Script flags for the params that differ from the script's own default (so the default recipe gives exactly
    build_project.py's argv)."""
    out = []
    for p in schema:
        if not p.legacy_flag or p.name in skip or p.name not in params: continue
        sd = p.default if p.script_default is None else p.script_default
        if params[p.name] != sd: out += [p.legacy_flag, str(params[p.name])]
    return out


REGISTRY = {"methods": {}, "tools": {}}


def register(cls):
    if issubclass(cls, Tool):
        for k in cls.kinds: REGISTRY["tools"][(cls.name, k)] = cls
    else:
        REGISTRY["methods"][(cls.node, cls.name)] = cls
    return cls


_loaded = False


def _load():
    global _loaded
    if _loaded: return
    _loaded = True
    from . import side, lineup, fuse_tsdf, fuse_points, fuse_plus, mesh, report    # noqa: F401  (registration)
    from ..tools import point_tools, mesh_tools, edits, finish, fin, slice_gap, drill, match_sides, plane_cut, clone, local_fix, mirror  # noqa: F401


def get_method(node, name):
    _load()
    cls = REGISTRY["methods"].get((node, name))
    if cls is None: raise ValueError("%s has no method %r (have: %s)" % (node, name, ", ".join(sorted(n for (nd, n) in REGISTRY["methods"] if nd == node))))
    return cls()


def get_tool(name, kind):
    _load()
    cls = REGISTRY["tools"].get((name, kind))
    if cls is None:
        if any(n == name for (n, _k) in REGISTRY["tools"]): return _WrongKind(name)
        return None if name != "_missing" else _WrongKind(name)
    return cls()


def methods_of(node):
    _load()
    return [cls() for (nd, _n), cls in REGISTRY["methods"].items() if nd == node]


def all_tools():
    _load()
    return sorted(REGISTRY["tools"].items())


class Method:
    node = ""
    name = ""
    label = ""
    version = 1
    consumes = {}
    produces = "mesh"
    mem_gb = 4.0
    gpu = False
    python = "venv"            # venv = sys.executable, pml = the PyMeshLab python
    soft_fail = False
    default_ops = []
    light = False              # cheap in-process work: runs during adopt too
    outputs = {}
    scripts = ()               # repo scripts it runs (names as script() takes them), and the repo modules they import
    modules = ()               # stages modules whose code makes its output (paths under scanner/stages)
    libs = ()                  # packages whose version changes its output
    tables = ()                # params.py TABLES entries that shape its output
    argv_params = False        # its params reach a script through legacy_flags
    argv_schema = "common"     # the same schema actually used by the command builder
    argv_skip = ()             # parameters emitted manually, rather than through legacy_flags

    def code(self, params):
        """([(name, path)], [lib]) of everything whose code makes this step's output (hashed into its key)."""
        schema = PR.METHOD_PARAMS.get((self.node, self.name), ()) if self.argv_schema == "method" else PR.COMMON.get(self.node, ())
        argv = [argv_digest(self.node, self.name, params, skip=self.argv_skip, schema=schema)] if self.argv_params else []
        executor = []
        if type(self).execute is Method.execute:
            from ..keys import H
            executor.append("method.outputs=" + H(self.outputs))
            for name in ("execute", "argv", "run", "parse", "stats"):
                function = getattr(Method, name)
                if getattr(type(self), name) is function:
                    source = inspect.getsource(function)
                    executor.append("method." + name + "=" + hashlib.blake2b(source.encode(), digest_size=16).hexdigest())
        return ([(n, script(n)) for n in self.scripts] + [("stages/" + m, stage_file(m)) for m in self.modules],
                list(self.libs) + [table_digest(t) for t in self.tables] + argv + executor)

    def static_alias(self, step, graph):
        """An input name when this node changes nothing for these params (known at plan time), else None."""
        return None

    def argv(self, ctx):
        return None

    def run(self, ctx):
        raise NotImplementedError

    def parse(self, line):
        """A child's stdout line -> {"done", "total", "text"} progress, or None."""
        return None

    def stats(self, ctx, stdout):
        return {}

    def adopt(self, ctx):
        """Legacy build/ files for this node (see engine.adopt_legacy), or None."""
        return None

    def execute(self, ctx):
        cmds = self.argv(ctx)
        if cmds is None: return self.run(ctx)
        text = ""
        for c in cmds: text += ctx.call(c, mem_gb=self.mem_gb, gpu=self.gpu, parse=self.parse)
        return {"outputs": dict(self.outputs), "stats": self.stats(ctx, text)}

    def summary(self, params):
        return ""


class Tool:
    name = ""
    label = ""
    kinds = ()
    version = 1
    mem_gb = 4.0
    detect = False
    scripts = ()
    modules = ()
    libs = ()
    tables = ()

    def code(self, params):
        return ([(n, script(n)) for n in self.scripts] + [("stages/" + m, stage_file(m)) for m in self.modules],
                list(self.libs) + [table_digest(t) for t in self.tables])

    def execute(self, ctx):
        raise NotImplementedError

    def summary(self, params):
        return ", ".join("%s %s" % (k, v) for k, v in params.items())


class _WrongKind(Tool):
    """A tool on a result of the wrong kind (e.g. Overlap Detection on a TSDF mesh): shown n/a, passes through."""

    def __init__(self, name):
        self.name = name; self.label = name; self.kinds = ()

    def summary(self, params):
        return "n/a here"


def py():
    return sys.executable
