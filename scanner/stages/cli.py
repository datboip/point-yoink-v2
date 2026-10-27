#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PointYoink 2.0 staged build, command line.

    ~/pointyoink/venv/bin/python ~/pointyoink/scanner/stages/cli.py <project> <command> ...

  status [--json] [-v]                 Revo-style stage list: state, method, key params, stats, est. rebuild time
  list [--methods] [--tools] [--params]
  show <node> [--json]
  run [--to N] [--from N] [--force N ...] [--with k=v ...] [--preset P] [--no-publish] [--legacy-events] [--mem-gb 6] [--dry-run]
  set k=v [k=v ...] [--note ..] [--run]        value "revo" = the Revo value
  reset k [k ...] [--run]
  preset <today|revo|revo-device|plus|print|quick> [--run]
  op <list> add <tool> [k=v ...] [--at i] [--detect] [--run]
  op <list> list | drop <i> | toggle <i> | set <i> k=v [k=v ...] | clear
                                        <list> = fused | meshed | side@<side> | lineup.fix | lineup.adjust; <i> = the
                                        index `op <list> list` shows, or an op id (fused lists its method's tools, then
                                        its manual deletes, which stay whatever the method; other methods' lists are
                                        shown as parked)
  delete <node> (--box=x0,y0,z0,x1,y1,z1 | --sphere=x,y,z,r | --mask pts.ply [--radius 0.5]) [--detect] [--run]
                                        (write --box=/--sphere= with "=": coordinates can start with "-")
  leave-out <side> | include <side>
  flip <side> [--axis auto|x|y|z] | nudge <side> --t x,y,z [--r rx,ry,rz] | platter <side> auto|search|none|<plane.json>
  match <side> --to <side> --pairs pairs.json   (pairs: [[to_xyz, side_xyz], ...], each in its side's own scan frame, 3+)
  turn <side> --r rx,ry,rz [--keep]    (deg about the world axes, about the side's own centre; then a local fit unless --keep)
  undo [--run] | redo [--run] | history
  versions <node> | pin <node> <key16> | unpin <node> <key16>
  preview <node> [--color shaded|sides|removed|scanhere|support] [--views 1|6] [--key K]
  view <node> [--color ..] [--key K] [--dry-run]
  export <out.ply|.stl|.obj> [--node meshed]
  adopt [--refit-old-pairs] | gc [--max-gb 8] [--keep 3]
<side> = an id or a 1-based position, wherever a stage or param names one (--to side@2, --force side.platter@2,
show side.keep@2, --with side[2].keep.min_share=0.2); an unknown stage, side or param is an error (exit 2).
Exit codes: 0 ok, 1 failed, 2 bad args, 3 locked.
status, show (without --views), history, list and versions only read: they never create or change anything in the
project (no stages/ folder, no adopt of build/, no history)."""
import argparse, json, os, signal, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE); REPO = os.path.dirname(SCANNER)
if sys.path and os.path.abspath(sys.path[0]) == HERE: sys.path.pop(0)
for _p in (REPO, SCANNER):
    if _p not in sys.path: sys.path.insert(0, _p)

from stages import params as PR                           # noqa: E402
from stages.engine import Engine, Locked                  # noqa: E402
from stages.project import Project, DeniedError           # noqa: E402
from stages.runner import Cancelled                       # noqa: E402

ICON = {"fresh": "ok", "adopted": "ok*", "stale": "..", "waiting": "--", "off": "off", "n/a": "n/a", "failed": "!!", "warn": "ok!",
        "check": "chk", "running": ">>", "parked": "pk"}
READ_ONLY = ("status", "history", "list", "versions", "show")


def kv(items):
    out = {}
    for it in items or []:
        if "=" not in it: raise ValueError("expected k=v, got %r" % it)
        k, v = it.split("=", 1); out[k.strip()] = v.strip()
    return out


def floats(s, n):
    v = [float(x) for x in s.split(",")]
    if len(v) != n: raise ValueError("expected %d numbers, got %r" % (n, s))
    return v


def fmt_stats(st):
    if not st: return ""
    bits = []
    for k in ("faces", "points", "pieces", "open_edge_mm", "fitted", "cached", "base", "tilt_deg", "source", "removed", "depth", "grid_mm",
              "same_as", "backend", "agreement", "kept_pct"):
        if k in st and st[k] is not None:
            v = st[k]
            if k == "agreement" and isinstance(v, dict): v = "%.3f -> %.3f mm" % (v.get("before", 0), v.get("after", 0))
            elif isinstance(v, float): v = PR.fmt(round(v, 3))
            elif isinstance(v, int) and v >= 10000: v = "%.0fk" % (v / 1000)
            bits.append("%s %s" % (k.replace("_", " "), v))
    return ", ".join(bits)


def icon(n):
    """A node's state as status prints it; a soft fail stored for retry reads "warn (will retry)"."""
    return "warn (will retry)" if getattr(n, "retry", False) else ICON.get(n.status, n.status)


def print_status(e, verbose=False, as_json=False):
    groups, g, est = e.status()
    if as_json:
        print(json.dumps({"groups": [{"name": gr.name, "label": gr.label, "nodes": [n.__dict__ for n in gr.nodes]} for gr in groups],
                          "est_s": round(est)}, default=str, indent=1))
        return
    r = e.recipe()
    print("project %s   preset %s   %d sides (%d active)" % (e.p.root, r.get("preset"), len(g.sides), len(g.active)))
    for gr in groups:
        if gr.name == "raw":
            fr = sum(n.stats["frames"] for n in gr.nodes); tr = sum(n.stats["tracked"] for n in gr.nodes)
            print("\n%-11s %d sides, %d frames (%d tracked, %d lost)" % (gr.label, len(gr.nodes), fr, tr, fr - tr))
            if verbose:
                for n in gr.nodes:
                    print("   %2d %s  %4d frames, %4d lost%s%s" % (n.n, n.side, n.stats["frames"], n.stats["lost"],
                                                                   ", live plane" if n.stats["live_plane"] else "", "  (left out)" if n.summary else ""))
            continue
        if gr.name == "side":
            sides = {}
            for n in gr.nodes: sides.setdefault(n.side, []).append(n)
            cnt = {}
            for n in gr.nodes: cnt[n.status] = cnt.get(n.status, 0) + 1
            print("\n%-11s %s" % (gr.label, ", ".join("%d %s" % (v, k) for k, v in sorted(cnt.items()))))
            if verbose or any(n.status not in ("fresh", "adopted") for n in gr.nodes):
                for sid, ns in sides.items():
                    lo = sid in r["sides"].get("leave_out", [])
                    parts = []
                    for n in ns:
                        nm = n.id.split("@")[0].split(".")[-1] if "#" not in n.id else "op " + n.id.split("#")[1]
                        parts.append("%s %s%s" % (nm, icon(n), (" (" + n.summary + ")") if n.summary and nm == "platter" else ""))
                    st = next((x.stats for x in reversed(ns) if x.stats), None) or {}
                    print("   %2d %s  %s%s%s" % (ns[0].n, sid, " | ".join(parts), ("   " + fmt_stats(st)) if verbose else "", "   LEFT OUT" if lo else ""))
            continue
        print("\n%-11s" % gr.label, end="")
        first = True
        for n in gr.nodes:
            if gr.name == "textured":
                print("n/a (colour comes later)"); break
            name = n.id if "#" not in n.id else ("  + " if n.status != "parked" else "  ~ ") + n.id.split("#")[1]
            line = "%-22s %-4s %-12s %s" % (name, icon(n), n.method, n.summary)
            s = fmt_stats(n.stats)
            if s: line += "   [" + s + "]"
            if n.adopted: line += "   (from earlier build)"
            if n.warn: line += "   warn: " + "; ".join(n.warn)[:80]
            print(("" if first else " " * 12) + line); first = False
    print("\nest. rebuild: %s" % ("nothing to do" if est < 0.5 else "~%d s" % est))


def print_show(e, node, as_json=False):
    g = e.graph(); outs, info, _ = e._walk(g, "plan", None)
    node = e.canon_token(node, g)                     # side@2 / side.keep@2: a side by position or id
    ids = e.node_ids(g, node)                         # an unknown stage or side is an error, never an empty listing
    if node in ("fused", "meshed"):
        ids = [node] + g.chain(node)
        if g.final(node) not in ids: ids.append(g.final(node))
    res = []
    for i in ids:
        st = g.by_id[i]; inf = info.get(i, {}); meta = inf.get("meta") or {}
        o = outs.get(i)
        d = {"id": i, "status": "warn (will retry)" if inf.get("retry") else inf.get("status"),
             "method": st.method.name if st.method else None, "key": inf.get("key"),
             "params": st.params, "inputs": {k: {"node": v, "digest": outs[v].digest if v in outs else None} for k, v in st.inputs.items()},
             "alias": st.alias, "stats": meta.get("stats"), "outputs": o.files if o else None, "secs": meta.get("secs"),
             "preview": os.path.join(meta["_dir"], "preview.png") if meta and os.path.exists(os.path.join(meta["_dir"], "preview.png")) else None,
             "warn": meta.get("warn"), "adopted": meta.get("adopted"), "dir": meta.get("_dir")}
        if st.op: d["op"] = st.op
        res.append(d)
    if as_json: print(json.dumps(res, indent=1, default=str)); return
    for d in res:
        print("%s   %s   method %s   key %s" % (d["id"], d["status"], d["method"], (d["key"] or "-")[:16]))
        if d.get("op"): print("  op      %s" % json.dumps({k: v for k, v in d["op"].items() if k not in ("anchor",)}))
        if d["alias"]: print("  alias   passes %s through" % d["alias"])
        print("  params  %s" % json.dumps(d["params"]))
        for k, v in d["inputs"].items(): print("  input   %-22s %s  %s" % (k, v["node"], (v["digest"] or "-")[:20]))
        if d["stats"]: print("  stats   %s" % json.dumps(d["stats"]))
        if d["outputs"]:
            for k, v in d["outputs"].items(): print("  output  %-10s %s" % (k, v))
        if d["preview"]: print("  preview %s" % d["preview"])
        if d["secs"] is not None: print("  secs    %s%s" % (d["secs"], "   (from earlier build)" if d["adopted"] else ""))
        if d["warn"]: print("  warn    %s" % "; ".join(d["warn"]))


def do_run(e, a, legacy=False):
    wp = kv(getattr(a, "with_", None))
    res = e.run(to=getattr(a, "to", None), from_=getattr(a, "from_", None), force=getattr(a, "force", None) or (), with_params=wp or None,
                publish=not getattr(a, "no_publish", False), legacy=legacy or getattr(a, "legacy_events", False), dry_run=getattr(a, "dry_run", False),
                defer_report=getattr(a, "defer_report", False))
    if not res.ok:
        print("failed at %s: %s" % (res.failed, (res.error or "")[:400]), file=sys.stderr); return 1
    if not legacy and not getattr(a, "legacy_events", False):
        print("ran %d step%s, %d cached, %.1f s%s" % (len(res.ran), "" if len(res.ran) == 1 else "s", len(res.cached), res.secs,
                                                    ("; model %s" % res.model) if res.model else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stages", description=__doc__.split("\n")[0])
    ap.add_argument("project"); ap.add_argument("--mem-gb", type=float, default=None, help="lower the saved Processing memory limit for this run")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("status"); s.add_argument("--json", action="store_true"); s.add_argument("-v", action="store_true")
    s = sp.add_parser("list"); s.add_argument("--methods", action="store_true"); s.add_argument("--tools", action="store_true"); s.add_argument("--params", action="store_true")
    s = sp.add_parser("show"); s.add_argument("node"); s.add_argument("--json", action="store_true"); s.add_argument("--views", type=int, default=0)

    def run_args(s):
        s.add_argument("--to"); s.add_argument("--from", dest="from_"); s.add_argument("--force", nargs="*")
        s.add_argument("--with", dest="with_", action="append"); s.add_argument("--preset"); s.add_argument("--no-publish", action="store_true")
        s.add_argument("--legacy-events", action="store_true"); s.add_argument("--dry-run", action="store_true")
        s.add_argument("--defer-report", action="store_true")
    s = sp.add_parser("run"); run_args(s)
    s = sp.add_parser("set"); s.add_argument("pairs", nargs="+"); s.add_argument("--note"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("reset"); s.add_argument("paths", nargs="+"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("preset"); s.add_argument("name"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("op"); s.add_argument("list"); s.add_argument("action", choices=["add", "list", "drop", "toggle", "set", "clear"])
    s.add_argument("rest", nargs="*"); s.add_argument("--at", type=int); s.add_argument("--detect", action="store_true"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("delete"); s.add_argument("node"); s.add_argument("--box"); s.add_argument("--sphere"); s.add_argument("--mask")
    s.add_argument("--radius", type=float, default=0.5); s.add_argument("--detect", action="store_true"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("leave-out"); s.add_argument("side"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("include"); s.add_argument("side"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("flip"); s.add_argument("side"); s.add_argument("--axis", default="auto", choices=["auto", "x", "y", "z"]); s.add_argument("--run", action="store_true")
    s = sp.add_parser("match"); s.add_argument("side"); s.add_argument("--to", required=True); s.add_argument("--pairs", required=True); s.add_argument("--run", action="store_true")
    s = sp.add_parser("turn"); s.add_argument("side"); s.add_argument("--r", required=True); s.add_argument("--keep", action="store_true"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("nudge"); s.add_argument("side"); s.add_argument("--t", default="0,0,0"); s.add_argument("--r", default="0,0,0"); s.add_argument("--run", action="store_true")
    s = sp.add_parser("platter"); s.add_argument("side"); s.add_argument("how"); s.add_argument("--run", action="store_true")
    for nm in ("undo", "redo"):
        s = sp.add_parser(nm); s.add_argument("--run", action="store_true")
    sp.add_parser("history")
    s = sp.add_parser("versions"); s.add_argument("node")
    for nm in ("pin", "unpin"):
        s = sp.add_parser(nm); s.add_argument("node"); s.add_argument("key16")
    s = sp.add_parser("preview"); s.add_argument("node"); s.add_argument("--color", default="shaded"); s.add_argument("--views", type=int, default=1); s.add_argument("--key")
    s = sp.add_parser("view"); s.add_argument("node"); s.add_argument("--color", default="shaded"); s.add_argument("--key"); s.add_argument("--dry-run", action="store_true")
    s = sp.add_parser("export"); s.add_argument("out"); s.add_argument("--node", default="meshed")
    s = sp.add_parser("adopt"); s.add_argument("--accept-pairs-version", type=int, action="append", default=[],
                                                help="with --refit-old-pairs: still adopt fits of this pair_all version")
    s.add_argument("--refit-old-pairs", action="store_true",
                   help="only adopt pair fits of the current pair_all version (default: any version whose side files match)")
    s = sp.add_parser("gc"); s.add_argument("--max-gb", type=float); s.add_argument("--keep", type=int, default=3)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return 2 if ex.code else 0
    ro = a.cmd in READ_ONLY and not (a.cmd == "show" and a.views)
    try:
        p = Project.open(a.project, create=not ro, read_only=ro)
    except (DeniedError, FileNotFoundError) as ex:
        print("error: %s" % ex, file=sys.stderr); return 2
    e = Engine(p, mem_gb=a.mem_gb, read_only=ro)

    def on_signal(sig, frm):
        e.cancel()
    signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
    if not ro and getattr(p, "created", False) and os.path.isdir(p.build) and a.cmd not in ("adopt",):
        try:
            r = e.adopt_legacy(); print("adopted %d stages from build/ (%d stale)" % (len(r.get("adopted", [])), len(r.get("stale", []))))
        except Locked: pass
        except Cancelled:
            print("cancelled", file=sys.stderr); return 1
    try:
        return dispatch(e, a)
    except Locked as ex:
        print("STAGE error %s" % json.dumps({"msg": str(ex)}), flush=True); return 3
    except DeniedError as ex:
        print("error: %s" % ex, file=sys.stderr); return 2
    except (ValueError, KeyError) as ex:
        print("error: %s" % (ex.args[0] if ex.args else ex), file=sys.stderr); return 2
    except Cancelled:
        print("cancelled", file=sys.stderr); return 1


def dispatch(e, a):
    c = a.cmd
    if c == "status": print_status(e, a.v, a.json); return 0
    if c == "list":
        from stages.methods import methods_of, all_tools
        from stages import graph as G
        every = not (a.methods or a.tools or a.params)
        if a.methods or every:
            print("stages and methods (default first):")
            for node, ms in PR.METHODS.items(): print("  %-14s %s" % (node, ", ".join(ms)))
        if a.tools or every:
            print("tools:")
            for kind, names in PR.TOOLS_BY_KIND.items():
                for nm in names:
                    ps = PR.tool_params(nm, kind)
                    print("  %-6s %-10s %s" % (kind, nm, "; ".join(q.describe().strip() for q in ps) if ps else ""))
        if a.params or every:
            print("params (path, default, unit, range, Revo):")
            for path, q in PR.all_paths(): print("  %-34s %s" % (path, q.describe()[19:] if len(q.describe()) > 19 else q.describe()))
        return 0
    if c == "show":
        print_show(e, a.node, a.json)
        if a.views: print(e.preview(a.node, views=a.views))
        return 0
    if c == "run":
        if a.preset:
            print(e.preset(a.preset)); print(e.confirm_line())
        return do_run(e, a)
    if c == "set":
        note = e.set(kv(a.pairs), a.note); print(note or "no change")
        return do_run(e, a) if a.run else 0
    if c == "reset":
        for n in e.reset(a.paths): print(n)
        return do_run(e, a) if a.run else 0
    if c == "preset":
        print(e.preset(a.name)); print(e.confirm_line())
        return do_run(e, a) if a.run else 0
    if c == "op":
        if a.action == "list":
            def line(k, o, note=""):
                print("  %s  %-8s %-10s %s%s%s" % (k, o["id"], o["tool"], json.dumps(o.get("params") or {k2: v for k2, v in o.items() if k2 not in ("id", "tool", "on", "anchor", "recorded", "rel")}),
                                                "" if o.get("on", True) else "   (off)", note))
            label, chain, parked = e.ops_of(a.list)
            last = None
            for k, (lk, _j, o) in enumerate(chain):
                if lk != last:
                    print("%s:" % (lk if lk != "fused" else "fused (manual deletes, kept whatever the method)")); last = lk
                line("%2d" % k, o)
            if not chain: print("%s:\n  (none)" % label)
            last = None
            for lk, _j, o in parked:
                if lk != last: print("parked %s (used when fused.method=%s):" % (lk, lk.split(".", 1)[1])); last = lk
                line(" -", o)
            return 0
        if a.action == "add":
            if not a.rest: raise ValueError("op add <tool> [k=v ...]")
            tool = a.rest[0]; prm = kv(a.rest[1:])
            if a.detect:
                r = e.detect(a.list, {"tool": tool, "params": prm}); print("detect: %s of %s would go; %s" % (r["count"], r["n"], r["png"])); return 0
            print("added", e.add_op(a.list, {"tool": tool, "params": prm}, at=a.at))
        elif a.action == "drop": print("dropped", e.drop_op(a.list, a.rest[0])["id"])
        elif a.action == "toggle":
            o = e.toggle_op(a.list, a.rest[0]); print("%s is %s" % (o["id"], "on" if o["on"] else "off"))
        elif a.action == "set":
            if len(a.rest) < 2: raise ValueError("op <list> set <i> k=v [k=v ...]")
            o = e.set_op(a.list, a.rest[0], kv(a.rest[1:]))
            print(("%s: %s" % (o["id"], json.dumps(o.get("params") or {k: v for k, v in o.items() if k in ("axis", "t", "r", "on")}))) if o else "no change")
        elif a.action == "clear": e.clear_ops(a.list); print("cleared")
        return do_run(e, a) if a.run else 0
    if c == "delete":
        if a.box:
            v = floats(a.box, 6); shape = {"shape": "box", "box": {"min": v[:3], "max": v[3:], "T": None}}
        elif a.sphere:
            v = floats(a.sphere, 4); shape = {"shape": "sphere", "center": v[:3], "radius": v[3]}
        elif a.mask:
            shape = {"shape": "mask"}
        else: raise ValueError("give --box, --sphere or --mask")
        r = e.delete(a.node, shape, mask_pts=os.path.abspath(a.mask) if a.mask else None, radius=a.radius, detect=a.detect)
        print("%s: %s of %s selected; %s" % ("detect" if a.detect else "delete %s" % r.get("id"), r["count"], r["n"], r["png"]))
        return do_run(e, a) if (a.run and not a.detect) else 0
    if c in ("leave-out", "include"):
        (e.leave_out if c == "leave-out" else e.include)(a.side); print("ok")
        return do_run(e, a) if a.run else 0
    if c == "flip":
        print("added", e.flip(a.side, a.axis)); return do_run(e, a) if a.run else 0
    if c == "match":
        print("added", e.match(a.side, a.to, json.load(open(a.pairs)))); return do_run(e, a) if a.run else 0
    if c == "turn":
        from .tools.edits import r_xyz
        print("added", e.turn(a.side, r_xyz(floats(a.r, 3)), a.keep)); return do_run(e, a) if a.run else 0
    if c == "nudge":
        print("added", e.nudge(a.side, floats(a.t, 3), floats(a.r, 3))); return do_run(e, a) if a.run else 0
    if c == "platter":
        e.platter(a.side, a.how); print("ok"); return do_run(e, a) if a.run else 0
    if c in ("undo", "redo"):
        n = e.undo() if c == "undo" else e.redo()
        print(("%s: %s" % (c, n)) if n else "nothing to %s" % c)
        return do_run(e, a) if a.run else 0
    if c == "history":
        ents, cur = e.history()
        for k, en in enumerate(ents):
            import time as _t
            print("%s %3d  %s  %s" % (">" if k == cur else " ", k, _t.strftime("%m-%d %H:%M:%S", _t.localtime(en["t"])), en["note"]))
        return 0
    if c == "versions":
        for v in e.versions(a.node):
            import time as _t
            print("%s  %s  %-10s %s%s%s   %s" % (v["key16"], _t.strftime("%m-%d %H:%M", _t.localtime(v["created"] or 0)), v["method"],
                                                fmt_stats(v["stats"]), "  PINNED" if v["pinned"] else "", "  (from earlier build)" if v["adopted"] else "",
                                                json.dumps(v["params"])[:100]))
        return 0
    if c in ("pin", "unpin"):
        print("ok" if e.pin(a.node, a.key16, c == "pin") else "no such version"); return 0
    if c == "preview":
        print(e.preview(a.node, a.color, a.views, a.key)); return 0
    if c == "view":
        cmd = e.view(a.node, a.color, a.key, a.dry_run); print(" ".join(cmd)); return 0
    if c == "export":
        src = e.output(a.node)
        if not src: raise ValueError("%s is not built yet" % a.node)
        e.p.guard_write(a.out)            # the target (symlinks followed) must not land under POINTYOINK_STAGES_DENY
        if a.out.lower().endswith(".ply"):
            import shutil
            out = os.path.realpath(a.out); tmp = out + ".tmp.%d" % os.getpid()
            shutil.copyfile(src, tmp); os.replace(tmp, out)
        else:
            r = e.job("methods.mesh:export_job", {"in": src, "out": os.path.realpath(a.out)}, mem_gb=4)
        print(a.out); return 0
    if c == "adopt":
        r = e.adopt_legacy(a.accept_pairs_version, refit_old_pairs=a.refit_old_pairs)
        if r.get("why"): print("nothing adopted: %s" % r["why"]); return 0
        print("adopted %d: %s" % (len(r["adopted"]), ", ".join(r["adopted"])))
        if r["stale"]: print("stale %d: %s" % (len(r["stale"]), ", ".join(r["stale"])))
        for n in r.get("notes", []): print("  note: " + n)
        return 0
    if c == "gc":
        print(json.dumps(e.gc(a.max_gb, a.keep), indent=1)); return 0
    raise ValueError("unknown command %s" % c)


if __name__ == "__main__":
    sys.exit(main())
