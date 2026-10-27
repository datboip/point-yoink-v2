#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build one model from every side of a live-scan project: fuse each side, drop floating bits, line every side up with
the biggest one (align.py --auto), refine all placements together (refine_sides.py), fuse all sides into one model,
keep the part, render a preview. Each heavy step runs as its own process under a memory cap, one after another, so a
big scan can never take the desktop down with it.
usage: build_project.py <ProjectLive folder> [--voxel 0.4] [--mem-gb 6] [--fuse classic|plus [--plus-voxel 0.3] [--no-drop-enclosed]]
Output: <project>/model.ply and <project>/model.png; work files in <project>/build/"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_frames
PY = sys.executable
PYK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDE_VERSION = 1            # bump when fusing, cutting or keep_part change, so cached side meshes are rebuilt
FUSE_FRAMES = 300           # per side in the final fuse


def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)


def run(args, mem_gb, quiet=True, frames=None, relay=False):
    """One heavy step in its own process with a hard memory limit (systemd scope); returns its output text.
    frames: a dict of extra fields; fuse.py's frame count is passed on as STAGE frames with them, so the window can
    show the build is still moving during a long fuse. relay: pass the step's own STAGE lines straight on."""
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=%dG" % mem_gb, PY] + args
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for ln in p.stdout:
        out.append(ln)
        if relay and ln.startswith("STAGE "): print(ln, end="", flush=True)
        if frames is not None and ln.startswith("STAGE integrate "):
            try: emit("frames", **frames, **json.loads(ln.split(" ", 2)[2]))
            except ValueError: pass
    p.wait(); text = "".join(out)
    if p.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (os.path.basename(args[0]), p.returncode, text[-600:]))
    return text


def bed_file(project):
    """The bed (Learn bed, bed.py) this project was scanned with: its own copy, else the one in the config dir."""
    p = os.path.join(project, "bed.json")
    if os.path.exists(p): return p
    c = os.environ.get("POINTYOINK_CONFIG")
    p = os.path.join(os.path.dirname(os.path.abspath(c)) if c else os.path.expanduser("~/.config/pointyoink"), "bed.json")
    return p if os.path.exists(p) else None


def write_beds(project, sides, work):
    """build/beds.json for pair_all.py and place_sides.py: each side's platter plane (the one the build cut it with) for
    the support check, and where the table dots put the side (B_scan, when a bed is known) for the same-pose start.
    Sides scanned before the bed existed get their bed placement from their saved frames (bed.py side-pose, CPU, a few
    seconds, written into build/ only). Those files are named after the side's folder and say which folder they are
    for: side numbers shift when a side is deleted, and a file keyed by number was reused for the wrong side."""
    bedf = bed_file(project); out = {"version": 1, "sides": {}}
    for i, s in enumerate(sides):
        pose = os.path.join(s, "bed_pose.json")
        if not os.path.exists(pose) and bedf:
            folder = os.path.basename(s)
            alt = os.path.join(work, "bed_pose_%s.json" % folder)
            src = [bedf, os.path.join(s, "cache", "pointyoink_register_pose.pose")]
            try: same = json.load(open(alt)).get("folder") == folder
            except Exception: same = False
            if not same or os.path.getmtime(alt) < max(os.path.getmtime(f) for f in src if os.path.exists(f)):
                try: run([os.path.join(HERE, "bed.py"), "side-pose", s, "--bed", bedf, "--out", alt], 2)
                except RuntimeError as e: emit("warn", msg="side %d: table dots not placed (%s)" % (i + 1, str(e)[:100]))
            pose = alt
        try: pj = json.load(open(pose))
        except Exception: pj = {}
        pp = os.path.join(work, "side%d_plane.json" % (i + 1))
        try: pl = json.load(open(pp)); pl = {"n": pl["n"], "d": pl["d"]}
        except Exception: pl = None
        out["sides"][str(i + 1)] = {"folder": os.path.basename(s), "plane": pl,
                                    "B_scan": pj.get("B_scan") if pj.get("state") == "found" else None, "bed_id": pj.get("bed_id")}
    p = os.path.join(work, "beds.json"); tmp = p + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, p)
    found = sum(1 for e in out["sides"].values() if e["B_scan"] is not None)
    if bedf: emit("bed", msg="table dots placed %d of %d sides" % (found, len(sides)))
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--voxel", type=float, default=0.4); ap.add_argument("--mem-gb", type=int, default=6)
    ap.add_argument("--fuse", default="classic", choices=("classic", "plus"),
                    help="step 4: classic = today's TSDF fuse; plus = fuse.py --mode plus (the bake-off's tsdf_plus: smoother "
                         "skin, fewer doubled walls), then drop_enclosed.py; falls back to classic when it fails")
    ap.add_argument("--plus-voxel", type=float, default=0.3,
                    help="plus: voxel in mm (not --voxel, which is for the per-side fuses). 0.3 = the tuned bake winner; 0.4 "
                         "had no tab tear and kept the inner blade on the headrest, with a little less of the gain")
    ap.add_argument("--no-drop-enclosed", action="store_true", help="plus: keep the surface sealed inside the part (drop_enclosed.py off)")
    a = ap.parse_args()
    data = os.path.join(a.project, "data"); work = os.path.join(a.project, "build"); os.makedirs(work, exist_ok=True)
    sides = sorted(d for d in glob.glob(os.path.join(data, "*")) if os.path.exists(os.path.join(d, "cache", "pointyoink_track.json")))
    if not sides: emit("error", msg="no saved sides in " + a.project); return 2
    # Touch-up sides (touchup.json: tracked in an earlier model's frame) are not lined up with the others: they are
    # placed from that frame after the ordinary sides are (model_frames.touch_placement). Without any, nothing changes.
    touch = [s for s in sides if model_frames.touch_info(s) is not None]
    if touch and len(touch) < len(sides): sides = [s for s in sides if s not in touch]
    else: touch = []
    emit("sides", count=len(sides)); t0 = time.time()
    # a side's meshes are reused when nothing it was built from changed: same folder, same frames, same platter plane,
    # same settings. Adding two sides to a 12-side project then builds 2 sides, not 14.
    cache_p = os.path.join(work, "sides_cache.json")
    try: cache = json.load(open(cache_p))
    except Exception: cache = {}
    def digest(f): return hashlib.md5(open(f, "rb").read()).hexdigest() if os.path.exists(f) else "-"
    def side_key(s):
        dph = glob.glob(os.path.join(s, "cache", "*.dph")); pl = os.path.join(s, "table_plane.json")
        pose = os.path.join(s, "cache", "pointyoink_register_pose.pose")      # re-tracking changes the poses, not the frames
        return "%d|%s|%d|%d|%s|%s|%s" % (SIDE_VERSION, os.path.basename(s), len(dph), max((os.stat(f).st_mtime_ns for f in dph), default=0),
                                         digest(pose), digest(pl), a.voxel)
    for f in glob.glob(os.path.join(work, "side*_plane.json")):         # planes of sides that are gone must not leak in
        if int(os.path.basename(f)[4:-11]) > len(sides): os.remove(f)

    # 1. each side on its own, floating bits dropped (touch-up sides too, after the others, named by their folder)
    parts = []; tparts = []; every = sides + touch
    for i, s in enumerate(every):
        tag = "side%d" % (i + 1) if i < len(sides) else "touch_" + os.path.basename(s)
        name = "side %d" % (i + 1) if i < len(sides) else "touch-up side %s" % os.path.basename(s)
        fused = os.path.join(work, "%s_raw.ply" % tag); part = os.path.join(work, "%s.ply" % tag)
        key = side_key(s); plane_out = os.path.join(work, "%s_plane.json" % tag)
        out_parts = parts if i < len(sides) else tparts
        if cache.get(tag) == key and os.path.exists(part):
            emit("side", done=i, total=len(every), msg="%s unchanged, reusing it" % name); out_parts.append(part); continue
        emit("side", done=i, total=len(every), msg="building %s" % name)
        cache.pop(tag, None)
        if os.path.exists(plane_out): os.remove(plane_out)                 # an older build's plane must not leak in
        run([os.path.join(PYK, "fuse.py"), "--frames", os.path.join(s, "cache"), "--calib", os.path.join(s, "param", "Pl.bin"),
             "--out", fused, "--voxel", str(a.voxel), "--gpu"], a.mem_gb,
            frames=dict(step="side", side=i + 1, sides=len(every)))
        # cut the turntable platter first: with the platter in range every side has the same big flat disc, and the
        # alignment then matches disc to disc instead of part to part (seen on a live project)
        cut = os.path.join(work, "%s_cut.ply" % tag)
        try:
            live = os.path.join(s, "table_plane.json")          # found while scanning (Auto remove base): no hunting
            msg = run([os.path.join(HERE, "strip_table.py"), s, cut, "--mesh", fused,
                       "--plane-out", plane_out]
                      + (["--plane", live] if os.path.exists(live) else []), 8).strip()
            emit("table", side=i + 1, msg=msg)
        except RuntimeError as e:
            emit("warn", msg="%s: table cut failed (%s); using it uncut" % (name, str(e)[:120])); cut = fused
        run([os.path.join(HERE, "keep_part.py"), cut, part], 4)
        out_parts.append(part)
        cache[tag] = key; json.dump(cache, open(cache_p, "w"))

    # 2. line up every pair; the base is the side that fits best with all the others (biggest-file was measured worse:
    #    it chose a side that overlaps the rest poorly)
    #    All pairs in one process (pair_all.py): each side is read once, and pairs of unchanged sides come from its cache.
    #    With build/beds.json every fit must keep both parts above both platters (a flipped fit goes through a table)
    beds = write_beds(a.project, sides, work)
    n = len(parts); pair = {}                                   # (i, j) -> (4x4 j->i, fitness)
    print(run([os.path.join(HERE, "pair_all.py"), "--work", work, "--n", str(n), "--beds", beds], 6, relay=True).splitlines()[-1], flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            jj = json.load(open(os.path.join(work, "T_%d_to_%d.json" % (j + 1, i + 1)))); M = np.array(jj["matrix"], float); f = float(jj.get("fitness", 0))
            pair[(i, j)] = (M, f); pair[(j, i)] = (np.linalg.inv(M), f)
    base = max(range(n), key=lambda i: sum(pair[(i, j)][1] for j in range(n) if j != i)) if n > 1 else 0
    emit("base", side=base + 1, msg="side %d fits best with the others" % (base + 1))
    # 2b. place sides one at a time by consensus. A near-symmetric part lets a pair fit come out flipped 180 deg while
    #     scoring well on its own (seen on the headrest: sides agreed with each other only through some routes). So
    #     every candidate placement of a new side (via each side already placed) is scored by how much of it lands on
    #     the sides placed so far; a flipped one lands on empty space and loses.
    print(run([os.path.join(HERE, "place_sides.py"), "--work", work, "--n", str(n), "--base", str(base), "--beds", beds], 4).strip(), flush=True)
    placed = json.load(open(os.path.join(work, "placement.json")))
    order = placed["order"]
    inits = {}
    for i in order:
        p = os.path.join(work, "T_side%d_init.json" % (i + 1))
        json.dump({"matrix": placed["T"][str(i)]}, open(p, "w")); inits[i] = p
    ident = inits[base]

    # 3. all placements together
    final = {base: ident}
    if len(parts) > 2:
        emit("refine", msg="adjusting all sides together")
        args = [os.path.join(HERE, "refine_sides.py"), "--out-prefix", os.path.join(work, "T_refined_")]
        for i in order: args += ["--mesh", parts[i], "--init", inits[i]]
        print(run(args, 5), flush=True)
        for k, i in enumerate(order): final[i] = os.path.join(work, "T_refined_%d.json" % (k + 1))
    else:
        final.update(inits)
    # where every side sits in the new model (build/frames.json once the model is written: Touch up tracks in it), and
    # the touch-up sides placed from the frame they were tracked in: no pair fitting
    placed_f = {os.path.basename(sides[i]): np.array(json.load(open(final[i]))["matrix"], float) for i in order}
    tsets = []
    for k, s in enumerate(touch):
        folder = os.path.basename(s); M, via = model_frames.touch_placement(model_frames.touch_info(s), placed_f,
                                                                          prefer=os.path.basename(sides[base]))
        if M is None:
            emit("warn", msg="touch-up side %s: none of the sides it was scanned against are left; it is left out" % folder); continue
        p = os.path.join(work, "T_touch_%s.json" % folder); json.dump({"matrix": M.tolist(), "via": via}, open(p, "w"))
        placed_f[folder] = M; tsets.append((s, p, os.path.join(work, "touch_%s_plane.json" % folder)))
        emit("touch", msg="touch-up side %s placed from the model it was scanned on (no pair fitting)" % folder)
    model_frames.write(work, placed_f)

    # 4. one model from every frame of every side, then keep the part
    emit("fuse", msg="fusing all sides into one model")
    raw = os.path.join(work, "all_raw.ply"); model = os.path.join(a.project, "model.ply")
    # at most FUSE_FRAMES frames per side: a 790-frame side outvoted the others and erased a thin lip it saw edge-on
    args = [os.path.join(PYK, "fuse.py"), "--out", raw, "--voxel", str(a.voxel), "--gpu", "--set-max-frames", str(FUSE_FRAMES)]
    sets = []
    for i in order:
        s = sides[i]; plane = os.path.join(work, "side%d_plane.json" % (i + 1))
        sets += ["--set", "%s,%s,%s%s" % (os.path.join(s, "cache"), os.path.join(s, "param", "Pl.bin"), final[i],
                                          ("," + plane) if os.path.exists(plane) else "")]
    for s, tp, plane in tsets:
        sets += ["--set", "%s,%s,%s%s" % (os.path.join(s, "cache"), os.path.join(s, "param", "Pl.bin"), tp,
                                          ("," + plane) if os.path.exists(plane) else "")]
    args += sets
    fused_with = "classic"
    if a.fuse == "plus":
        # the tsdf_plus method: 16^3 blocks (free-space reach), trunc 1.2 mm, weight > 2,
        # field blur of 0.24 mm only where 3+ sides saw. Same frames, clean-up, platter cut and placements as classic.
        plus = [os.path.join(PYK, "fuse.py"), "--mode", "plus", "--out", raw, "--gpu", "--set-max-frames", str(FUSE_FRAMES),
                "--voxel", str(a.plus_voxel), "--plus-trunc", "1.2", "--plus-block-res", "16", "--plus-min-weight", "2",
                "--plus-field-smooth", "%g" % round(0.24 / a.plus_voxel, 4), "--plus-smooth-min-sides", "3"] + sets
        try:
            run(plus, a.mem_gb, frames=dict(step="all")); fused_with = "plus"
        except RuntimeError as e:       # CUDA out of memory (the live scanner shares the GPU) or anything else
            emit("warn", msg="plus fuse failed, using the classic fuse (%s)" % str(e)[-160:].replace("\n", " "))
    if fused_with == "classic":
        run(args, a.mem_gb, frames=dict(step="all"))
    print(run([os.path.join(HERE, "keep_part.py"), raw, model], 4).strip(), flush=True)
    model_frames.commit(work)
    if fused_with == "plus":
        emit("fused", msg="fused with plus", mode="plus")
        if not a.no_drop_enclosed:      # loose flakes sealed inside the part: no scanner can see them, so no "scan here" there
            try: print(run([os.path.join(HERE, "drop_enclosed.py"), model, model, "--rays", "32", "--tidy-rounds", "4"], 4).strip().splitlines()[-1], flush=True)
            except RuntimeError as e: emit("warn", msg="enclosed-surface clean-up failed (%s); model kept as fused" % str(e)[-120:])
    # carve.py (drop what two or more sides looked straight through) is NOT run here yet: on the 9-side headrest build
    # it took 1,016 points off the real surface and nearly tripled the open edge (128 -> 335 mm)
    run([os.path.join(HERE, "render_views.py"), model, os.path.join(a.project, "model.png")], 4)
    # where the model still needs scanning: open edges marked red, for the next side to aim at
    try:
        print(run([os.path.join(HERE, "needs_scan.py"), model, os.path.join(a.project, "model_needs.ply"), "--png",
                   os.path.join(a.project, "model_needs.png"), "--json", os.path.join(a.project, "model_needs.json")], 6).strip(), flush=True)
    except RuntimeError as e: emit("warn", msg="scan-here map failed: " + str(e)[:120])
    emit("done", model=model, preview=os.path.join(a.project, "model.png"), sides=len(sides) + len(tsets), base=base + 1, secs=round(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
