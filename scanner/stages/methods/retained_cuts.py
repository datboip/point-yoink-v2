# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep explicit model-space base cuts on the final mesh after reconstruction and edits."""
import os

from . import Method


class RetainedCuts(Method):
    node = "meshed.retained_cuts"
    name = "retained_cuts"
    label = "Keep base cuts"
    version = 1
    produces = "mesh"
    modules = ("methods/retained_cuts.py", "tools/plane_cut.py", "tools/edits.py", "plyio.py")
    libs = ("numpy", "scipy")

    def execute(self, ctx):
        stats = ctx.job("methods.retained_cuts:job", {"in": ctx.inputs["parent"].files["mesh"],
                                                   "cuts": ctx.params["cuts"]}, mem_gb=3)
        return {"outputs": {"mesh": "mesh.ply"}, "stats": stats}

    def summary(self, params):
        count = len(params["cuts"])
        return "Keep %d base cut%s" % (count, "" if count == 1 else "s")


def job(payload):
    from .. import plyio
    from ..tools.plane_cut import apply

    data = plyio.read_mesh(payload["in"])
    faces_in = len(data[1])
    details = []
    for cut in payload["cuts"]:
        data, _selected, stats = apply("mesh", data, cut["params"])
        details.append({"source": cut["source"], "removed": stats["removed"]})
    vertices, faces, normals = data
    plyio.write_mesh(os.path.join(payload["out_dir"], "mesh.ply"), vertices, faces, normals)
    return {"faces_in": faces_in, "faces": len(faces), "removed": faces_in - len(faces),
            "retained_cuts": details}
