# SPDX-License-Identifier: AGPL-3.0-or-later
"""The steps for (project, recipe): every node expanded per side, methods resolved, op lists expanded into one step per
op (each op prefix is its own cached artifact), static aliases marked.

Order (a topological order; run executes in it):
  per side (position order): side.fuse@id, side.platter@id, side.keep@id, side.keep@id#op...
  lineup.pairs, lineup.place, lineup.fix, lineup.refine, lineup.adjust
  fused, fused.<method>#op..., fused#op... (manual edits), meshed, meshed#op..., meshed.retained_cuts (when needed), report
Per-side steps exist for every side, left-out ones too (so they can be looked at); only active sides feed Line-up.
The Fused stage has two op lists: the tools of the current method (fused.<method>: switching method switches the list)
and its manual deletes and base cuts (fused: kept whatever the method, replayed geometrically onto its output), in that order."""
import copy, os
from dataclasses import dataclass, field

from . import params as PR
from .methods import get_method, get_tool

GROUPS = [("raw", "Raw Data"), ("side", "Side"), ("lineup", "Line-up"), ("fused", "Fused"), ("meshed", "Meshed"),
          ("report", "Coverage"), ("textured", "Textured")]
GROUP_LABEL = dict(GROUPS)


@dataclass
class Step:
    id: str
    kind: str                      # node kind (side.fuse, ..., report) or "op"
    group: str
    side: str = None
    n: int = None
    method: object = None          # Method (node) or Tool (op) instance
    params: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)      # input name -> step id
    leaf: dict = field(default_factory=dict)
    store: str = ""
    alias: str = None              # static alias: this input passes straight through
    na: bool = False               # alias because the tool does not fit (shown n/a)
    op: dict = None
    list_key: str = None
    index: int = None              # op position in its list (0-based)
    produces: str = "mesh"
    frame: str = None              # "side" | "model" for deletes
    active: bool = True

    @property
    def name(self):
        return self.method.name if self.method is not None else "-"


class Graph:
    def __init__(self):
        self.steps = []; self.by_id = {}; self.finals = {}; self.lists = {}; self.sides = []; self.active = []

    def add(self, st):
        self.steps.append(st); self.by_id[st.id] = st; return st

    def final(self, node): return self.finals.get(node, node)

    def ancestors(self, ids):
        need = set(); stack = list(ids)
        while stack:
            s = stack.pop()
            if s in need: continue
            need.add(s); stack.extend(self.by_id[s].inputs.values())
        return need

    def descendants(self, ids):
        out = set(ids); changed = True
        while changed:
            changed = False
            for st in self.steps:
                if st.id not in out and any(i in out for i in st.inputs.values()): out.add(st.id); changed = True
        return out

    def target_ids(self, token):
        """Step ids a CLI target names: a group (sides, side@<id>, lineup, fused, meshed, report), a node id (with its
        ops) or a step id."""
        if token in (None, "", "report", "all"): return [self.final("report")]
        if token == "sides": return [st.id for st in self.steps if st.group == "side"]
        if token.startswith("side@"):
            sid = token[5:]; return [st.id for st in self.steps if st.group == "side" and st.side == sid]
        if token == "lineup": return ["lineup.adjust"]
        if token in ("fused", "meshed"): return [self.final(token)]
        if token in self.finals: return [self.finals[token]]
        if token in self.by_id: return [token]
        # a node id of a list (fused.tsdf) or a group label
        for k, v in self.lists.items():
            if token == k and v: return [v[-1]]
        raise KeyError("no stage %r" % token)

    def chain(self, node):
        """The op steps after a node, in order (fused: the method's tools, then the manual deletes and base cuts)."""
        if node == "fused":
            return self.lists.get("fused." + self.by_id["fused"].method.name, []) + self.lists.get("fused", [])
        return list(self.lists.get(node, []))

    def node_steps(self, token):
        """The steps a `show`/`--force` node token names (a node id without its ops, a step id, or a group)."""
        if token in self.by_id: return [token]
        if token == "sides": return [st.id for st in self.steps if st.group == "side" and st.kind != "op"]
        if token.startswith("side@"): return [st.id for st in self.steps if st.side == token[5:] and st.kind != "op"]
        if token == "lineup": return [st.id for st in self.steps if st.group == "lineup"]
        raise KeyError("no stage %r" % token)


def default_ops(method):
    return [dict({"id": "default-%d" % (k + 1), "on": True}, **copy.deepcopy(o))
            for k, o in enumerate(getattr(method, "default_ops", []) or [])]


def op_list(recipe, list_key, method=None):
    ops = recipe.get("ops", {})
    if list_key in ops: return ops[list_key]
    if list_key.startswith("fused.") and method is not None: return default_ops(method)
    return []


def build(project, recipe, with_params=None):
    wp = with_params or {}
    g = Graph(); s_cfg = recipe.get("sides", {})
    leave = set(s_cfg.get("leave_out", []))
    sides = project.sides(leave_out=leave)
    g.sides = sides; g.active = [s for s in sides if s.active]
    if not g.active and sides: raise ValueError("every side is left out")

    def add_ops(list_key, parent, group, kind, side=None, n=None, frame="model", method=None):
        ids = []; prev = parent
        for k, op in enumerate(op_list(recipe, list_key, method)):
            tool = get_tool(op["tool"], kind)
            st = Step(id="%s#%s" % (list_key, op["id"]), kind="op", group=group, side=side, n=n, method=tool,
                      inputs={"parent": prev}, store=list_key + ".ops", op=op, list_key=list_key, index=k, produces=kind,
                      frame=op.get("frame", frame) if op["tool"] == "delete" else None)
            if tool is None or kind not in getattr(tool, "kinds", ()):
                st.alias = "parent"; st.na = True
                if tool is None: st.method = get_tool("_missing", kind)
            else:
                st.params = PR.resolve_tool(op["tool"], kind, op.get("params"))
                if not op.get("on", True): st.alias = "parent"
                if (op["tool"] == "delete" and st.frame == "model") or op["tool"] == "match_cleaned_sides":
                    st.inputs["lineup"] = "lineup.adjust"
            g.add(st); ids.append(st.id); prev = st.id
        g.lists[list_key] = ids
        return prev

    # ------------------------------------------------------------------------------------------------ per side
    from scanner.color_frames import available
    def side_has_color(side):
        images=bool(project._scan(side.id).get("img",[])) if hasattr(project,"_scan") else None
        return available(os.path.join(project.folder(side.id),"cache"),has_images=images)
    color_mode=PR.resolve(recipe,"fused",PR.method_of(recipe,"fused",wp),None,wp)["color"]
    color_actual="off" if color_mode=="off" or (color_mode=="auto" and not all(side_has_color(s) for s in g.active)) else "on"
    for s in sides:
        m = get_method("side.fuse", PR.method_of(recipe, "side.fuse", wp))
        fuse_params=PR.resolve(recipe,"side.fuse",m.name,s.id,wp)
        from app2.scan.scan_mode import read as scan_profile
        capture=scan_profile(project.folder(s.id))
        spec=next(p for p in PR.COMMON['side.fuse'] if p.name=='voxel')
        explicit=recipe.get('sides',{}).get('params',{}).get(s.id,{}).get('side.fuse',{})
        global_explicit=recipe.get('params',{}).get('side.fuse',{})
        if capture.get('mode')=='far' and 'voxel_mm' in capture and 'voxel' not in explicit and 'voxel' not in global_explicit and fuse_params['voxel']==spec.default and 'side.fuse.voxel' not in wp and ('side[%s].fuse.voxel'%s.id) not in wp:
            fuse_params['voxel']=spec.coerce(capture['voxel_mm'])
        depth_spec=next(p for p in PR.COMMON['side.fuse'] if p.name=='max_depth')
        if capture.get('mode')=='far' and 'far' in capture and 'max_depth' not in explicit and 'max_depth' not in global_explicit and fuse_params['max_depth']==depth_spec.default and 'side.fuse.max_depth' not in wp and ('side[%s].fuse.max_depth'%s.id) not in wp:
            fuse_params['max_depth']=depth_spec.coerce(max(depth_spec.default,capture['far']))
        g.add(Step(id="side.fuse@" + s.id, kind="side.fuse", group="side", side=s.id, n=s.n, method=m,
                   params=fuse_params, leaf={"frames": project.frames_fp(s.id)},
                   store="side.fuse@" + s.id, active=s.active))
        ov = s_cfg.get("platter", {}).get(s.id) or {}
        pm = ov.get("method") or PR.method_of(recipe, "side.platter", wp)
        m = get_method("side.platter", pm)
        prm = PR.resolve(recipe, "side.platter", pm, s.id, wp)
        if pm == "manual": prm["plane"] = ov.get("plane")
        leaf = {"live_plane": project.live_plane_fp(s.id)} if pm in ("auto", "plate") else {}
        if pm in ("auto", "plate") and hasattr(project, "board_frames_fp"):
            leaf["board_ir"] = project.board_frames_fp(s.id)
            leaf["board_depth"] = project.frames_fp(s.id)
        st = g.add(Step(id="side.platter@" + s.id, kind="side.platter", group="side", side=s.id, n=s.n, method=m, params=prm,
                        inputs={"raw": "side.fuse@" + s.id}, leaf=leaf, store="side.platter@" + s.id, active=s.active))
        st.alias = m.static_alias(st, g)
        m = get_method("side.keep", PR.method_of(recipe, "side.keep", wp))
        g.add(Step(id="side.keep@" + s.id, kind="side.keep", group="side", side=s.id, n=s.n, method=m,
                   params=PR.resolve(recipe, "side.keep", m.name, s.id, wp), inputs={"cut": "side.platter@" + s.id},
                   store="side.keep@" + s.id, active=s.active))
        g.finals["side.keep@" + s.id] = add_ops("side.keep@" + s.id, "side.keep@" + s.id, "side", "mesh", s.id, s.n, frame="side")
        g.finals["side.fuse@" + s.id] = "side.fuse@" + s.id; g.finals["side.platter@" + s.id] = "side.platter@" + s.id

    # ------------------------------------------------------------------------------------------------ line-up
    def side_inputs(planes=True):
        d = {}
        for s in g.active:
            d["mesh@" + s.id] = g.final("side.keep@" + s.id)
            if planes: d["plane@" + s.id] = "side.platter@" + s.id
        return d

    def node(kind, inputs, extra_params=None):
        m = get_method(kind, PR.method_of(recipe, kind, wp))
        prm = PR.resolve(recipe, kind, m.name, None, wp)
        if extra_params: prm.update(extra_params)
        st = g.add(Step(id=kind, kind=kind, group=kind.split(".")[0], method=m, params=prm, inputs=inputs, store=kind,
                        produces=m.produces))
        st.alias = m.static_alias(st, g); g.finals[kind] = kind
        return st

    node("lineup.pairs", side_inputs())
    pl = node("lineup.place", dict(side_inputs(), pairs="lineup.pairs"))
    if pl.params.get("base") not in ("auto", None) and pl.params["base"] not in [s.id for s in sides]:
        # set() stores ids; a position given through --with is resolved here, anything else is an error (never ignored)
        pl.params["base"] = project.resolve_side(pl.params["base"], sides).id
    flips = [dict((k, v) for k, v in o.items() if k not in ("id", "recorded")) for o in recipe.get("ops", {}).get("lineup.fix", [])
             if o.get("on", True) and o.get("tool") in ("flip", "match", "turn") and o.get("side") in [s.id for s in g.active]]
    node("lineup.fix", dict(side_inputs(planes=False), placement="lineup.place"), {"flips": flips})
    node("lineup.refine", dict(side_inputs(planes=False), placement="lineup.fix"), {"active": len(g.active)})
    nudges = [dict((k, v) for k, v in o.items() if k not in ("id", "recorded")) for o in recipe.get("ops", {}).get("lineup.adjust", [])
              if o.get("on", True) and o.get("tool") == "nudge" and o.get("side") in [s.id for s in g.active]]
    node("lineup.adjust", dict(side_inputs(), transforms="lineup.refine"), {"nudges": nudges})
    g.lists["lineup.fix"] = []; g.lists["lineup.adjust"] = []

    for side_step in g.steps:
        if side_step.kind=="side.fuse":side_step.params["color"]=color_actual if side_step.active else "off"
    # ------------------------------------------------------------------------------------------------ fused
    fm = get_method("fused", PR.method_of(recipe, "fused", wp))
    side_deletes = {s.id: [{k: v for k, v in op.items() if k != "recorded"}
                           for op in recipe.get("ops", {}).get("side.keep@" + s.id, [])
                           if op.get("tool") == "delete" and op.get("on", True)] for s in g.active}
    side_deletes = {sid: ops for sid, ops in side_deletes.items() if ops}
    fst = g.add(Step(id="fused", kind="fused", group="fused", method=fm, params=PR.resolve(recipe, "fused", fm.name, None, wp),
                     inputs={"lineup": "lineup.adjust"}, leaf={"frames": {s.id: project.frames_fp(s.id) for s in g.active},
                                                                  "side_deletes": side_deletes},
                     store="fused", produces=fm.produces))
    fst.params["color"]=color_actual
    # an absent fused.<method> list means the method's default_ops (they live in code until the first edit)
    tools_end = add_ops("fused." + fm.name, "fused", "fused", fm.produces, method=fm)
    # the Fused stage's manual deletes and base cuts: one list whatever the method (a method switch keeps them), after its tools
    g.finals["fused"] = add_ops("fused", tools_end, "fused", fm.produces)

    # ------------------------------------------------------------------------------------------------ meshed
    mm = PR.method_of(recipe, "meshed", wp)
    if mm == "auto": mm = "as_fused" if fm.produces == "mesh" else "poisson"
    if mm == "as_fused" and fm.produces != "mesh":
        raise ValueError("meshed.method as_fused needs a mesh from Fused, but fused.method %s makes %s: use meshed.method "
                         "auto, poisson or bpa" % (fm.name, fm.produces))
    m = get_method("meshed", mm)
    mst = g.add(Step(id="meshed", kind="meshed", group="meshed", method=m, params=PR.resolve(recipe, "meshed", mm, None, wp),
                     inputs={"fused": g.final("fused")}, store="meshed", produces="mesh"))
    mst.params["fused_kind"] = fm.produces
    mst.alias = m.static_alias(mst, g)
    g.finals["meshed"] = add_ops("meshed", "meshed", "meshed", "mesh")
    # Meshing, hole filling, and smoothing can grow faces back into explicit cut regions.
    # Retain those user exclusions on the final mesh; side platter planes are not global cuts.
    cuts = [{"source": i, "params": dict(g.by_id[i].params)}
            for i in g.chain("fused") + g.chain("meshed")
            if g.by_id[i].method.name == "cut_plane" and not g.by_id[i].alias]
    if cuts:
        from .methods.retained_cuts import RetainedCuts
        parent = g.final("meshed")
        g.add(Step(id="meshed.retained_cuts", kind="meshed.retained_cuts", group="meshed",
                   method=RetainedCuts(), params={"cuts": cuts}, inputs={"parent": parent},
                   store="meshed.retained_cuts", produces="mesh"))
        g.finals["meshed"] = "meshed.retained_cuts"

    # ------------------------------------------------------------------------------------------------ report
    m = get_method("report", PR.method_of(recipe, "report", wp))
    g.add(Step(id="report", kind="report", group="report", method=m, params=PR.resolve(recipe, "report", m.name, None, wp),
               inputs={"mesh": g.final("meshed")}, store="report", produces="report"))
    g.finals["report"] = "report"
    return g
