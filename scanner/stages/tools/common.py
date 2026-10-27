# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared plumbing of in-engine tools: the engine side (build the job, run it under the tool's cap) and the job side
(load the parent result, apply, write the new artifact plus a packed 'removed' mask for previews and detect)."""
import json, os

import numpy as np

from .. import plyio
from ..keys import jload, jdump


def out_name(kind):
    return "points.npz" if kind == "points" else "mesh.ply"


def run_tool_job(ctx, tool, detect=False):
    parent = ctx.inputs["parent"]
    job = {"tool": tool.name, "kind": ctx.step.produces, "in": parent.files[parent_file_key(parent)], "params": ctx.params,
           "op": ctx.step.op, "parent_digest": parent.bdigest, "stages": ctx.project.stages, "detect": detect,
           "T_now": lineup_T(ctx)}
    if tool.name == "match_cleaned_sides":
        job["sides_json"] = ctx.inputs["lineup"].files["sides"]
        job["project"] = ctx.project.root
    # light when the input is small: measured 0.3 GB (mesh smooth, 750k faces) and 0.5 GB (point isolation, 2.4M points)
    small = ctx.engine._small({"in": job["in"]}, 1_500_000 if ctx.step.produces == "mesh" else 3_000_000)
    light = small and tool.name in ("smooth", "delete", "isolation", "cut_plane", "clone", "local_fix", "mirror") and not (tool.name == "smooth" and ctx.step.produces == "points")
    r = ctx.job("tools.common:tool_job", job, mem_gb=tool.mem_gb, light=light)
    if detect: return r
    res = {"outputs": {ctx.step.produces: out_name(ctx.step.produces)}, "stats": r.get("stats", {})}
    if r.get("stats", {}).get("check"):
        res["warn"] = ["delete replay removed %d (%.0f mm2), %.0f%% off what was recorded: check this delete" % (
            r["stats"]["removed"], r["stats"].get("removed_mm2", 0), 100 * r["stats"]["drift"])]
        res["check"] = True
    return res


def parent_file_key(out):
    for k in ("points", "mesh"):
        if k in out.files: return k
    return next(iter(out.files))


def lineup_T(ctx):
    lu = ctx.inputs.get("lineup")
    if lu is None or "lineup" not in lu.files: return None
    return jload(lu.files["lineup"])["T"]


def load(kind, path):
    if kind == "points": return plyio.load_points(path)
    return plyio.read_mesh(path)


def save(kind, data, out_dir):
    p = os.path.join(out_dir, out_name(kind))
    if kind == "points":
        plyio.save_points(p, data["P"], data["N"], data["W"], data["NF"], data["S"], data["side_ids"], float(data["cell"]), colors=data.get("C") if data.get("C") is not None and len(data["C"])==len(data["P"]) else None)
    else:
        V, F, N = data
        if len(F): N = plyio.vertex_normals(V, F)
        plyio.write_mesh(p, V, F, N)
    return p


def count(kind, data):
    return len(data["P"]) if kind == "points" else len(data[1])


def tool_job(job):
    """Job side: apply job['tool'] to job['in']. detect: write detect.png + removed mask only, change nothing."""
    from . import point_tools, mesh_tools, edits, match_sides, plane_cut, clone, local_fix, mirror
    if job.get("preview_fill"):return local_fix.preview_job(job)
    kind = job["kind"]; data = load(kind, job["in"]); n0 = count(kind, data)
    name = job["tool"]
    if name == "match_cleaned_sides":
        new, sel, st = match_sides.job_apply(job, kind, data)
    elif name == "cut_plane":
        new, sel, st = plane_cut.job_apply(job, kind, data)
    elif name == "clone":
        new, sel, st = clone.job_apply(job, kind, data)
    elif name == "local_fix":
        new, sel, st = local_fix.job_apply(job, kind, data)
    elif name == "mirror":
        new, sel, st = mirror.job_apply(job, kind, data)
    elif name == "delete":
        new, sel, st = edits.delete_job(job, kind, data)
    elif kind == "points":
        new, sel, st = point_tools.apply(name, data, job["params"], detect=job.get("detect"))
    else:
        new, sel, st = mesh_tools.apply(name, data, job["params"], detect=job.get("detect"))
    out = job["out_dir"]
    if sel is not None:
        np.save(os.path.join(out, "removed.npy"), np.packbits(sel.astype(bool)))
        jdump({"n": int(len(sel)), "removed": int(sel.sum())}, os.path.join(out, "removed.json"))
    if job.get("detect"):
        from ..preview import render_removed
        png = os.path.join(out, "detect.png")
        render_removed(kind, data, sel if sel is not None else np.zeros(n0, bool), png,
                       title="%s detect: %d of %d would go" % (name, int(sel.sum()) if sel is not None else 0, n0))
        return {"count": int(sel.sum()) if sel is not None else 0, "n": n0, "png": png,
                "mask": os.path.join(out, "removed.npy"), "stats": st, "area_mm2": st.get("removed_mm2")}
    save(kind, new, out)
    st = dict(st); st.setdefault("removed", n0 - count(kind, new)); st["in"] = n0; st["out"] = count(kind, new)
    return {"stats": st}
