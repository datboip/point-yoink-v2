#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Place scanned sides into one frame by consensus, from all pairwise fits (build/T_<j>_to_<i>.json, side j -> side i).
Start with the base side. Repeatedly, for every unplaced side and every placed side k, the candidate placement is
T_k @ T_(j->k); it is scored by the share of the side's points that land within 1.5 mm of the sides placed so far. The
best (side, candidate) is placed next. A pair fit that came out flipped on a near-symmetric part lands mostly on empty
space and never wins. With --beds (build/beds.json: every side's platter plane) a candidate that puts a side into the
table of a side already placed, or a placed side into its own table, is skipped (support.py); when every candidate of a
round is skipped the best is placed anyway and flagged. A table that at least 2/3 of its side's good pair fits (fitness
>= 0.5, 3 or more) go through is not trusted (the part stood on something else, or the plane is off): with it every
right candidate was skipped and weak wrong ones were placed first (a 12-side scan, 2026-09-28). And a landing under 60%
is never placed while one 25 points better exists that goes through a table: that one is placed (and flagged).
Writes build/placement.json: {"order": [...], "T": {side: 4x4 side->base, mm}, "score": {...}, "flagged": [...]}.
usage: place_sides.py --work <build dir> --n <sides> --base <index> [--beds <build>/beds.json]"""
import argparse, glob, json, os, sys, numpy as np, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support
from placement_hints import hint_ok as orientation_hint_ok

ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--n", type=int, required=True)
ap.add_argument("--base", type=int, required=True); ap.add_argument("--tol", type=float, default=1.5)
ap.add_argument("--beds", default=None); ap.add_argument("--hints", default="")     # "idx:same|flipped|sideways,..." how each side was scanned vs the base
a = ap.parse_args()
GOOD_FIT, BAD_TABLE = 0.5, 2 / 3      # pair fitness that counts as good; share of those crossing a table that voids it
WEAK, GAP = 0.6, 0.25                 # a landing under WEAK is not placed while one GAP better goes through a table
pair = {}; fit = {}
for f in glob.glob(os.path.join(a.work, "T_*_to_*.json")):
    name = os.path.basename(f)[2:-5]
    if "_to_" not in name: continue
    j, i = [int(x) - 1 for x in name.split("_to_")]
    jf = json.load(open(f)); M = np.array(jf["matrix"], float); pair[(i, j)] = M; pair[(j, i)] = np.linalg.inv(M)
    fit[(i, j)] = fit[(j, i)] = float(jf.get("fitness", 0.0))
clouds = []
o3d.utility.random.seed(0)           # the same sides place the same way every run (sampling was unseeded: 3 of 8 runs differed)
for s in range(a.n):
    pc = o3d.io.read_triangle_mesh(os.path.join(a.work, "side%d.ply" % (s + 1))).sample_points_uniformly(60000).voxel_down_sample(1.5)
    clouds.append(np.asarray(pc.points))
planes = {}
if a.beds:
    try: bj = json.load(open(a.beds)).get("sides", {})
    except Exception: bj = {}
    for s in range(a.n):
        pl = (bj.get(str(s + 1)) or {}).get("plane")
        if pl is not None: planes[s] = pl
hints = {}
for part in a.hints.split(","):
    if ":" in part:
        k, how = part.split(":"); hints[int(k)] = how
planes_all = dict(planes)             # kept whole: the tables the line-up rules out may be dropped below, a side's up never is


def hint_ok(j, C):
    """Does placing side j by C give it the orientation the user said (against the base side's up)? Needs both sides' tables."""
    how = hints.get(j)
    side_up = planes_all[j]["n"] if j in planes_all else None
    base_up = planes_all[a.base]["n"] if a.base in planes_all else None
    return orientation_hint_ok(how, C, side_up, base_up)


samp = [c[np.random.default_rng(0).choice(len(c), min(3000, len(c)), replace=False)] for c in clouds]
for s in sorted(planes):          # a table most of the side's good pair fits go through is not what the part rested on
    good = [j for j in range(a.n) if j != s and fit.get((s, j), 0.0) >= GOOD_FIT]
    cross = sum(support.under(samp[j] @ pair[(s, j)][:3, :3].T + pair[(s, j)][:3, 3], planes[s]) > support.REJECT for j in good)
    if len(good) >= 3 and cross >= BAD_TABLE * len(good):
        del planes[s]
        msg = "Side %d: its base plane was ignored for placing. %d of its %d good fits run through it, so it is probably not the real base." % (s + 1, cross, len(good))
        print(msg); print("STAGE warn %s" % json.dumps({"msg": msg}), flush=True)


def through_table(j, C, T):
    """The worst share of side j (placed by C) inside a placed side's table, or of a placed side inside side j's."""
    worst = 0.0
    if j not in planes: return worst
    for m in T:
        if m not in planes: continue
        Qj = samp[j] @ C[:3, :3].T + C[:3, 3]; Qm = samp[m] @ T[m][:3, :3].T + T[m][:3, 3]
        worst = max(worst, support.under(Qj, support.plane_in(planes[m], T[m])), support.under(Qm, support.plane_in(planes[j], C)))
    return worst


T = {a.base: np.eye(4)}; order = [a.base]; score = {a.base: 1.0}; flagged = []
def placed_tree():
    P = np.concatenate([clouds[k] @ T[k][:3, :3].T + T[k][:3, 3] for k in T])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); return o3d.geometry.KDTreeFlann(pc)
while len(T) < a.n:
    tree = placed_tree(); best = None; best_any = None; skipped = 0; hint_skipped = 0
    for j in range(a.n):
        if j in T: continue
        for k in T:
            if (k, j) not in pair: continue
            C = T[k] @ pair[(k, j)]
            Q = clouds[j] @ C[:3, :3].T + C[:3, 3]
            idx = np.random.default_rng(0).choice(len(Q), min(3000, len(Q)), replace=False)
            hits = sum(1 for q in Q[idx] if tree.search_radius_vector_3d(q, a.tol)[0] > 0)
            s = hits / len(idx)
            if best_any is None or s > best_any[0]: best_any = (s, j, k, C)
            if hints and not hint_ok(j, C): hint_skipped += 1; continue
            if planes and through_table(j, C, T) > support.REJECT: skipped += 1; continue
            if best is None or s > best[0]: best = (s, j, k, C)
    if best is None or (best[0] < WEAK and best_any[0] - best[0] > GAP):     # never a weak landing over a far better one
        best = best_any; flagged.append(best[1])
        msg = "Side %d was placed from side %d but crosses another side's base plane. Check it in Line-up." % (best[1] + 1, best[2] + 1)
        print(msg); print("STAGE warn %s" % json.dumps({"msg": msg}), flush=True)
    if hints and best[1] in hints and not hint_ok(best[1], best[3]):
        msg = "side %d could not be placed the way you said (%s): no fit matches it" % (best[1] + 1, {"same": "same way up", "flipped": "flipped over", "sideways": "on its side"}[hints[best[1]]])
        print(msg); print("STAGE warn %s" % json.dumps({"msg": msg}), flush=True)
    s, j, k, C = best
    T[j] = C; order.append(j); score[j] = s
    print("placed side %d via side %d: %.0f%% of it lands on the model so far%s" % (
        j + 1, k + 1, 100 * s, "" if not skipped else " (%d candidate%s through a table skipped)" % (skipped, "s" if skipped > 1 else "")))
json.dump({"order": order, "T": {str(k): v.tolist() for k, v in T.items()}, "score": {str(k): v for k, v in score.items()},
           "flagged": flagged}, open(os.path.join(a.work, "placement.json"), "w"))
