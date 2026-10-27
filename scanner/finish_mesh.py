#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finish a built model for printing: a smooth, closed, light mesh.
The recipe picked over gentler smoothing, a finer Poisson and cutting flakes out first:
  screened Poisson (depth 9) over the model -> drop Poisson surface more than --trim mm from the real scan (it balloons
  across openings; a tighter trim re-opens every hole Poisson had filled) -> drop loose pieces -> close holes up to
  --hole-edges edges -> Taubin smoothing (no shrinking) -> simplify to --faces triangles (250k is about a 12 MB STL).
usage: finish_mesh.py model.ply out.ply [--stl out.stl] [--faces 250000] [--depth 9] [--trim 3.0] [--smooth 5]
Prints STAGE lines (name {json}) like the other build steps."""
import argparse, json, os, sys, time


def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--stl", default=None)
    ap.add_argument("--faces", type=int, default=250000); ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--trim", type=float, default=3.0, help="mm: Poisson surface further than this from the scan goes")
    ap.add_argument("--hole-edges", type=int, default=400); ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--cores", type=int, default=6, help="CPU cores to use: PyMeshLab ignores OMP_NUM_THREADS and took 17-20 "
                    "cores, starving the live scanner (seen: 11 s tracking lag)")
    a = ap.parse_args()
    try:
        cores = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cores[-max(1, a.cores):]); os.nice(10)
    except (AttributeError, OSError): pass
    import pymeshlab as ml
    t0 = time.time(); ms = ml.MeshSet(); ms.load_new_mesh(a.src); n0 = ms.current_mesh().face_number()
    emit("poisson", depth=a.depth)
    ms.generate_surface_reconstruction_screened_poisson(depth=a.depth, pointweight=4, preclean=True)
    ms.set_current_mesh(1)
    emit("trim", mm=a.trim)
    ms.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=1, refmesh=0, maxdist=ml.PureValue(a.trim * 2.5), signeddist=False)
    ms.compute_selection_by_condition_per_vertex(condselect="(q > %g)" % a.trim); ms.meshing_remove_selected_vertices()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=2000)
    emit("holes", max_edges=a.hole_edges); ms.meshing_close_holes(maxholesize=a.hole_edges)
    if a.smooth: emit("smooth", passes=a.smooth); ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=a.smooth)
    emit("simplify", faces=a.faces)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=a.faces, preservenormal=True, qualitythr=0.5)
    if ms.current_mesh().face_number() > a.faces * 1.1:        # PyMeshLab sometimes stops short: finish with Open3D
        tmp = a.out + ".tmp.ply"; ms.save_current_mesh(tmp)
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(tmp).simplify_quadric_decimation(a.faces); m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(tmp, m); ms = ml.MeshSet(); ms.load_new_mesh(tmp); os.remove(tmp)
    m = ms.current_mesh(); ms.save_current_mesh(a.out)
    if a.stl: ms.save_current_mesh(a.stl, binary=True)
    emit("done", out=a.out, stl=a.stl, faces_in=n0, faces=m.face_number(), secs=round(time.time() - t0, 1),
         mb_stl=round(os.path.getsize(a.stl) / 1e6, 1) if a.stl else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
