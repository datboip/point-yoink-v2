#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drop-in for scanner/build_project.py: same argv, same stdout stream (build_project's STAGE events and plain lines,
plus the engine's new events, which the scanner's parser ignores), same published files, same exit codes.

    compat.py <project> [--voxel 0.4] [--mem-gb 6] [--legacy-build full]

1. open the project (a new stages/ starts with preset today and adopts build/)
2. --voxel V becomes the transient overrides side.fuse.voxel=V and fused.point_distance=V (build_project used one voxel
   for both); without it the recipe is used as it stands, so the user's stage choices carry into the Build button
3. run to the end with legacy events, publish
4. write build/placement.json (positions), build/T_refined_<k>.json and build/side<n>_plane.json (what bake/frames.py
   and other tools read); --legacy-build full also writes build/side<n>.ply and build/all_raw.ply. Never writes
   sides_cache.json or pairs_cache.json.
Hookup (by the owners of those files, later): pointyoink_scanner.py runs this instead of build_project.py, or
build_project.py's main becomes os.execv(PY, [PY, "<scanner>/stages/compat.py", project] + args)."""
import argparse, json, os, signal, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.dirname(HERE); REPO = os.path.dirname(SCANNER)
if sys.path and os.path.abspath(sys.path[0]) == HERE: sys.path.pop(0)
for _p in (REPO, SCANNER):
    if _p not in sys.path: sys.path.insert(0, _p)

from stages.engine import Engine, Locked            # noqa: E402
from stages.project import Project, DeniedError     # noqa: E402
from stages.legacy import write_legacy_build        # noqa: E402


def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)


def main(argv=None, out=None):
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--voxel", type=float, default=None); ap.add_argument("--mem-gb", type=float, default=None, help="lower the saved Processing memory limit for this run")
    ap.add_argument("--legacy-build", choices=["min", "full"], default="min")
    a = ap.parse_args(argv)
    try:
        p = Project.open(a.project, create=True)
    except (DeniedError, FileNotFoundError) as e:
        emit("error", msg=str(e)); return 2
    if not p.side_folders():
        emit("error", msg="no saved sides in " + a.project); return 2
    e = Engine(p, mem_gb=a.mem_gb, out=out)
    signal.signal(signal.SIGTERM, lambda s, f: e.cancel()); signal.signal(signal.SIGINT, lambda s, f: e.cancel())
    try:
        if getattr(p, "created", False) and os.path.isdir(p.build): e.adopt_legacy()
        wp = None
        if a.voxel is not None: wp = {"side.fuse.voxel": a.voxel, "fused.point_distance": a.voxel}
        res = e.run(with_params=wp, legacy=True)
    except Locked as ex:
        emit("error", msg=str(ex)); return 3
    if not res.ok: return 1
    try:
        g = e.graph(None, wp); outs, _i, _ = e._walk(g, "plan", None)
        lu = outs["lineup.adjust"]; fx = outs["lineup.fix"]
        full = a.legacy_build == "full"
        meshes = {s.id: outs[g.final("side.keep@" + s.id)].files["mesh"] for s in g.active} if full else None
        fused_raw = outs["fused"].files.get("mesh") if full and "fused" in outs else None
        write_legacy_build(p, lu.files["lineup"], fx.files["placement"], lu.files["sides"], full, meshes, fused_raw)
    except Exception as ex:
        emit("warn", msg="legacy build/ files not written: %s" % str(ex)[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
