#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Line up every pair of sides in one process: each side is read, sampled and given its FPFH features once, and every
pair is fitted with the same steps as align.py --auto (FGR, RANSAC when FGR is weak, then coarse and fine ICP).
align.py run once per pair spent 2.1 of its 2.5 s reading and sampling the two meshes: with 14 sides that was about 3
of the 4.5 minutes, spent reading the same 14 files over and over.
Results are cached in <work>/pairs_cache.json, keyed on both side files (size and time): a rebuild after adding sides
only fits the new pairs. Writes <work>/T_<j>_to_<i>.json for place_sides.py (older T_*_to_* files are removed first,
so a deleted side's pairs cannot leak in).
With --beds <build>/beds.json (build_project.py writes it: each side's platter plane, and where the table dots put the
side when a bed is learned) every candidate fit must keep both parts above both tables (support.py): a flipped fit on a
near-symmetric part pushes one side into the other's table and is thrown out. When both sides saw the table dots, the
dots' placement is one more start ("bed"); if it fits as well as the winner the part did not move between the sides.
usage: pair_all.py --work <build dir> --n <sides> [--beds <build>/beds.json]
Prints STAGE lines: align {done,total,msg}, aligned {sides,fit,rmse_mm,bed} and warn {msg}."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# align.py caps the address space of whatever imports it (8 GB); CUDA Open3D alone maps more than that, so reads fail.
# The build runs this under a systemd memory cap instead.
os.environ.setdefault("POINTYOINK_MEM_CAP_GB", "100000")
from align import cloud
from seeded_sample import sample
import support

R = o3d.pipelines.registration
VERSION = 3                                     # bump when the fitting changes, so old cached fits are not reused
SAME_POSE_MM = 2.0                              # the table dots' placement this close to the winner: the part stayed put


def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)


def stamp(path):
    s = os.stat(path); return "%d:%d:%d" % (VERSION, s.st_size, s.st_mtime_ns)


def prepare(path, voxel=1.0, samples=250000):
    """A side as align.py sees it: working cloud (voxel mm), coarse cloud for the first ICP, feature cloud + FPFH."""
    p, n = sample(path, samples); B = cloud(p, n, voxel); fv = voxel * 3
    D = B.voxel_down_sample(fv)
    f = R.compute_fpfh_feature(D, o3d.geometry.KDTreeSearchParamHybrid(radius=fv * 5, max_nn=100))
    return dict(B=B, C=B.voxel_down_sample(max(voxel * 2.5, 2.0)), D=D, f=f, fv=fv, voxel=voxel)


def refine(base, mov, T0, max_dist):
    """align.py's two ICP passes from a rough fit: coarse clouds, then the working clouds."""
    res = R.registration_icp(mov["C"], base["C"], max_dist, T0, R.TransformationEstimationPointToPlane(), R.ICPConvergenceCriteria(max_iteration=40))
    T = np.asarray(res.transformation); f, e = float(res.fitness), float(res.inlier_rmse)
    res2 = R.registration_icp(mov["B"], base["B"], max(base["voxel"] * 1.5, 1.0), T, R.TransformationEstimationPointToPlane(),
                              R.ICPConvergenceCriteria(max_iteration=15))
    if res2.fitness > 0.2: T = np.asarray(res2.transformation); f, e = float(res2.fitness), float(res2.inlier_rmse)
    return T, f, e


def ransac(base, mov):
    fv = base["fv"]
    r = R.registration_ransac_based_on_feature_matching(
        mov["D"], base["D"], mov["f"], base["f"], True, fv * 1.5, R.TransformationEstimationPointToPoint(False), 3,
        [R.CorrespondenceCheckerBasedOnEdgeLength(0.9), R.CorrespondenceCheckerBasedOnDistance(fv * 1.5)],
        R.RANSACConvergenceCriteria(400000, 0.999))
    return np.asarray(r.transformation)


def candidates(base, mov, max_dist=6.0, extra=()):
    """Moving side -> base side. Rough fits from both FGR and RANSAC (align.py only tried RANSAC when FGR looked weak)
    and any extra starts, each finished with ICP. Returns [(start, 4x4, fitness, rmse)]."""
    fv = base["fv"]; starts = []
    try:
        r = R.registration_fgr_based_on_feature_matching(mov["D"], base["D"], mov["f"], base["f"],
                                                         R.FastGlobalRegistrationOption(maximum_correspondence_distance=fv * 1.5))
        T = np.asarray(r.transformation)
        if np.isfinite(T).all() and np.linalg.norm(T[:3, 3]) < 5000: starts.append(("fgr", T))
    except Exception: pass
    starts.append(("ransac", ransac(base, mov)))
    starts += list(extra)
    return [(name,) + refine(base, mov, T0, max_dist) for name, T0 in starts]


def fit(base, mov, max_dist=6.0):
    """The one that ends up overlapping most wins (no platter check). Returns (4x4, fitness, rmse)."""
    return max((c[1:] for c in candidates(base, mov, max_dist)), key=lambda x: x[1])


def load_beds(path, n):
    """build/beds.json -> per side: plane dict (fuse.py's n, d) or None, B_scan 4x4 or None, bed id, and a digest of
    the entry (a side whose planes change must be fitted again)."""
    out = {}
    try: j = json.load(open(path)).get("sides", {})
    except Exception: j = {}
    for i in range(n):
        e = j.get(str(i + 1)) or {}
        out[i] = dict(plane=e.get("plane"), B=None if e.get("B_scan") is None else np.array(e["B_scan"], float),
                      bed_id=e.get("bed_id"), digest=hashlib.md5(json.dumps(e, sort_keys=True).encode()).hexdigest()[:12])
    return out


def pick(cands, Pi, Pj, beds_i, beds_j, name, base=None, mov=None):
    """The fit to keep: highest fitness among the candidates that keep both parts above both tables. Returns
    (4x4, fitness, rmse, support dict, bed dict, warn messages)."""
    warns = []; under = {}
    pl_i, pl_j = beds_i["plane"], beds_j["plane"]
    check = pl_i is not None and pl_j is not None
    for c in cands: under[id(c)] = support.pair_under(Pi, pl_i, Pj, pl_j, c[1]) if check else 0.0
    ok = [c for c in cands if under[id(c)] <= support.REJECT]
    if check and not ok and base is not None:                  # one more RANSAC start from another seed
        o3d.utility.random.seed(7919)
        c = ("ransac2",) + refine(base, mov, ransac(base, mov), 6.0); cands = cands + [c]
        under[id(c)] = support.pair_under(Pi, pl_i, Pj, pl_j, c[1])
        if under[id(c)] <= support.REJECT: ok = [c]
    best_all = max(cands, key=lambda c: c[2])
    flag = False
    if ok: win = max(ok, key=lambda c: c[2])
    else:
        win = best_all; flag = True
        warns.append("sides %s: no fit keeps both parts above both tables; kept the best (flagged)" % name)
    if best_all is not win and under[id(best_all)] > support.REJECT:
        warns.append("sides %s: a flipped fit was thrown out (%.1f%% of a side would sit inside the table)" % (name, 100 * under[id(best_all)]))
    sup = dict(under=round(under[id(win)], 5), rejected=sum(1 for c in cands if under[id(c)] > support.REJECT), start=win[0], flag=flag)
    bed = None
    bc = [c for c in cands if c[0] == "bed"]
    if bc:
        Q = Pj[np.random.default_rng(0).choice(len(Pj), min(5000, len(Pj)), replace=False)]
        gap = float(np.sqrt((((Q @ bc[0][1][:3, :3].T + bc[0][1][:3, 3]) - (Q @ win[1][:3, :3].T + win[1][:3, 3])) ** 2).sum(1).mean()))
        bed = dict(same_pose=bool(bc[0] is win or gap <= SAME_POSE_MM), gap_mm=round(gap, 2))
    return win[1], win[2], win[3], sup, bed, warns


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--beds", default=None, help="build/beds.json: platter planes (support check) and table-dot placements")
    a = ap.parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    t0 = time.time(); w = a.work
    paths = [os.path.join(w, "side%d.ply" % (i + 1)) for i in range(a.n)]
    keys = [stamp(p) for p in paths]
    beds = load_beds(a.beds, a.n) if a.beds else None
    cp = os.path.join(w, "pairs_cache.json")
    try: cache = json.load(open(cp))
    except Exception: cache = {}
    for f in glob.glob(os.path.join(w, "T_*_to_*.json")): os.remove(f)
    todo = [(i, j) for i in range(a.n) for j in range(i + 1, a.n)]
    sides = {}; fresh = 0
    for k, (i, j) in enumerate(todo):
        name = "%d_to_%d" % (j + 1, i + 1); c = cache.get(name); pn = "%d-%d" % (i + 1, j + 1)
        bkey = None if beds is None else beds[i]["digest"] + beds[j]["digest"]
        sup = bed = None
        if c and c.get("base") == keys[i] and c.get("moving") == keys[j] and c.get("beds") == bkey:
            M, f, e = np.array(c["matrix"]), c["fitness"], c["rmse"]; sup, bed = c.get("support"), c.get("bed")
            for m in c.get("warns", []): emit("warn", msg=m)
        else:
            emit("align", done=k, total=len(todo), msg="lining up side %d with side %d" % (j + 1, i + 1))
            for s in (i, j):
                if s not in sides: sides[s] = prepare(paths[s])
            if beds is None:
                M, f, e = fit(sides[i], sides[j]); warns = []
            else:
                extra = []
                if beds[i]["B"] is not None and beds[j]["B"] is not None and beds[i]["bed_id"] == beds[j]["bed_id"]:
                    extra.append(("bed", beds[i]["B"] @ np.linalg.inv(beds[j]["B"])))     # the part stayed put on the table?
                cands = candidates(sides[i], sides[j], extra=extra)
                Pi, Pj = np.asarray(sides[i]["B"].points), np.asarray(sides[j]["B"].points)
                M, f, e, sup, bed, warns = pick(cands, Pi, Pj, beds[i], beds[j], pn, sides[i], sides[j])
                for m in warns: emit("warn", msg=m)
            fresh += 1
            cache[name] = dict(base=keys[i], moving=keys[j], matrix=M.tolist(), fitness=f, rmse=e, beds=bkey, support=sup, bed=bed, warns=warns)
        json.dump({"matrix": M.tolist(), "fitness": f, "rmse": e, "support": sup, "bed": bed}, open(os.path.join(w, "T_%s.json" % name), "w"))
        emit("aligned", sides=pn, fit=round(f, 2), rmse_mm=round(e, 2),
             bed="-" if bed is None else ("same pose" if bed["same_pose"] else "moved"))
        if bed and bed["same_pose"]: emit("bed", msg="sides %s: same part pose, placed by the table dots" % pn)
    tmp = cp + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, cp)
    print("pairs: %d fitted, %d from the cache, %.0f s" % (fresh, len(todo) - fresh, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
