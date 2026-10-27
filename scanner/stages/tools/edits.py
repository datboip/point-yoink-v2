# SPDX-License-Identifier: AGPL-3.0-or-later
"""Manual edits: delete (box / sphere / points mask) with replay, flip and nudge matrices.

delete replay, in order (spec 10.3):
 1. input unchanged (digest == recorded.input_digest) and the op has exact indices (mask): delete exactly those
 2. model-frame ops are re-anchored when the model frame moved (base side changed): the first id of anchor.order that is
    still placed gives T_fix = T_now[id] @ inv(anchor.T[id]), applied to the box frame, sphere centre or region samples
 3. geometry: box / sphere by the shape; mask: an element goes when its nearest dead sample is closer than its nearest
    live sample and nearer than 1.0 mm (the other skin of a thin wall is live, so it stays)
 4. the removed amount more than 30 % off the recorded one flags the op 'check'. The amount is the removed AREA (mesh
    faces, or the surface the points cover: points_area), so re-fusing at another resolution (more, smaller faces) or
    switching tsdf <-> points is not a false alarm; ops recorded without an area fall back to the element count
Points are selected when inside the shape; mesh faces when their centroid is inside (unreferenced vertices then go)."""
import hashlib, io, json, os

import numpy as np

from .. import params as PR
from ..methods import Tool, register
from .. import plyio
from ..keys import jload, jdump

DEAD_VOXEL = 0.3
LIVE_REACH = 3.0
MASK_MAX = 200000
REPLAY_NEAR = 1.0
CHECK_DRIFT = 0.30


# ------------------------------------------------------------------------------------------------------ geometry
def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); x, y, z = a; t = np.radians(deg)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    if abs(deg % 360 - 180) < 1e-12:      # exact half turn: R = 2 a a^T - I (no sin/cos rounding)
        R = 2 * np.outer(a, a) - np.eye(3)
    return R


def tr(t):
    M = np.eye(4); M[:3, 3] = t; return M


def hom(R):
    M = np.eye(4); M[:3, :3] = R; return M


def r_xyz(r):
    rx, ry, rz = [float(v) for v in r]
    return rot([0, 0, 1], rz) @ rot([0, 1, 0], ry) @ rot([1, 0, 0], rx)


def nudge_matrix(c, t, r):
    """D = Tr(c + t) R_xyz(r) Tr(-c): rotate about the side's placed centroid c (model frame), then move by t."""
    c = np.asarray(c, float)
    return tr(c + np.asarray(t, float)) @ hom(r_xyz(r)) @ tr(-c)


def sample_surface(V, F, n, seed=0):
    """n area-weighted surface samples (numpy, seeded)."""
    rng = np.random.default_rng(seed)
    if len(F) == 0:
        return V[rng.choice(len(V), min(n, len(V)), replace=False)] if len(V) else V
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    f = rng.choice(len(F), n, p=a / a.sum())
    u = rng.random(n); v = rng.random(n); s = u + v > 1; u[s] = 1 - u[s]; v[s] = 1 - v[s]
    A, B, C = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
    return A + u[:, None] * (B - A) + v[:, None] * (C - A)


def pca_frame(P):
    """Centroid and axes (columns, x = largest variance)."""
    c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T))
    return c, v[:, np.argsort(w)[::-1]]


def flip_matrix(c, axis_vec):
    """Fl = Tr(c) R(axis, 180 deg) Tr(-c), in the side frame."""
    return tr(c) @ hom(rot(axis_vec, 180.0)) @ tr(-np.asarray(c, float))


def landing_share(Q, tree, tol=1.5, probes=3000, seed=0):
    """place_sides' score: the share of probe points that land within tol mm of the placed model."""
    idx = np.random.default_rng(seed).choice(len(Q), min(probes, len(Q)), replace=False)
    d, _ = tree.query(Q[idx], distance_upper_bound=tol)
    return float(np.isfinite(d).mean())


def choose_flip(P_side, T_side, others, axis="auto", tol=1.5):
    """P_side: side samples (side frame). others: placed samples of the other sides (model frame).
    Returns (flip matrix in the side frame, chosen axis name, scores)."""
    from scipy.spatial import cKDTree
    c, ax = pca_frame(P_side)
    names = ("x", "y", "z")
    cands = [axis] if axis in names else list(names)
    tree = cKDTree(others) if len(others) else None
    scores = {}
    for nm in cands:
        Fl = flip_matrix(c, ax[:, names.index(nm)]); M = T_side @ Fl
        Q = P_side @ M[:3, :3].T + M[:3, 3]
        scores[nm] = landing_share(Q, tree, tol) if tree is not None else 0.0
    best = max(cands, key=lambda k: scores[k])
    return flip_matrix(c, ax[:, names.index(best)]), best, scores


# ------------------------------------------------------------------------------------------------------ flip / match fits
# A flip or a match is ABSOLUTE: the pose it lands on is kept against a placed side ("to"), T[s] = T[to] @ rel, so a
# refit, a base change or a leave-out moves it with that side instead of turning a new placement. Every fit is LOCAL
# point-to-plane ICP seeded from the flipped / picked pose (never a global search), facing surfaces only (thin walls).
FACING = 0.7                    # cos of the largest angle between paired surfaces
FIT_STEPS = (6.0, 3.0, 1.5)     # mm pairing gates, coarse to fine
FLIP_STEPS = (12.0, 6.0, 3.0, 1.5)
KEEP_STEPS = (3.0, 1.5)         # a stored pose only follows small changes of its anchor
LAND_TOL = 1.5                  # mm: place_sides' landing tolerance
MATCH_MOVE = (5.0, 10.0)        # mm / deg ICP may move a picked pose before the picks win
MATCH_PICK = 3.0                # mm median pick residual ICP may push the fit to
SAMPLES = 60000
SAMPLE_VERSION = 1


def sample_with_normals(V, F, n, seed=0):
    """Area-weighted samples and their face normals (numpy, seeded)."""
    rng = np.random.default_rng(seed)
    if len(F) == 0:
        i = rng.choice(len(V), min(n, len(V)), replace=False) if len(V) else np.zeros(0, int)
        return V[i], np.zeros((len(i), 3))
    e = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(e, axis=1)
    f = rng.choice(len(F), n, p=a / a.sum())
    u = rng.random(n); v = rng.random(n); s = u + v > 1; u[s] = 1 - u[s]; v[s] = 1 - v[s]
    A, B, C = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
    return A + u[:, None] * (B - A) + v[:, None] * (C - A), e[f] / np.maximum(a[f], 1e-12)[:, None]


def side_samples(path, digest=None, cache=None, n=SAMPLES):
    """(P, N) of a side mesh in its own frame; cached by the mesh digest (stages/cache/samples) when one is given."""
    fp = os.path.join(cache, "%s_%d_v%d.npz" % (str(digest).replace(":", "_"), n, SAMPLE_VERSION)) if cache and digest else None
    if fp and os.path.exists(fp):
        try:
            z = np.load(fp); return z["P"].astype(np.float64), z["N"].astype(np.float64)
        except Exception: pass
    V, F, _ = plyio.read_mesh(path); P, N = sample_with_normals(V, F, n, 0)
    if fp:
        try:
            os.makedirs(cache, exist_ok=True); tmp = fp + ".tmp.%d.npz" % os.getpid()
            np.savez(tmp, P=P.astype(np.float32), N=N.astype(np.float32)); os.replace(tmp, fp)
        except OSError: pass
    return P, N


def moved(Ta, Tb, c):
    """mm the point c (side frame) moves and deg the pose turns from Ta to Tb."""
    D = Tb @ np.linalg.inv(Ta); ca = Ta[:3, :3] @ c + Ta[:3, 3]
    ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(D[:3, :3] @ ca + D[:3, 3] - ca)), float(ang)


class Target:
    """Placed samples a fit lands on (model frame), their normals and a tree."""
    def __init__(self, Q, NQ):
        from scipy.spatial import cKDTree
        self.Q, self.NQ = Q, NQ; self.tree = cKDTree(Q) if len(Q) else None


def placed(P, N, T): return P @ T[:3, :3].T + T[:3, 3], N @ T[:3, :3].T


def pairs_at(P, N, T, tgt, gate):
    X, Xn = placed(P, N, T)
    d, j = tgt.tree.query(X, distance_upper_bound=gate, workers=-1)
    ok = np.isfinite(d); ok[ok] = (Xn[ok] * tgt.NQ[j[ok]]).sum(1) > FACING
    return X, ok, j, d


def local_icp(P, N, tgt, T0, steps=FIT_STEPS, iters=15):
    """Point-to-plane ICP of samples P (normals N, own frame) onto tgt from pose T0, one round per gate. -> T."""
    T = np.array(T0, float)
    if tgt.tree is None or not len(P): return T
    for gate in steps:
        for _ in range(iters):
            X, ok, j, _d = pairs_at(P, N, T, tgt, gate)
            if ok.sum() < 30: break
            x = X[ok]; q = tgt.Q[j[ok]]; n = tgt.NQ[j[ok]]; c = x.mean(0)
            r = ((x - q) * n).sum(1); L = max(float(np.sqrt(((x - c) ** 2).sum(1).mean())), 1e-6)
            A = np.hstack([np.cross(x - c, n) / L, n])            # rotation in mm at the fit's own size, like the shift
            # damped: a direction the surfaces do not pin (a plane slides along itself) stays put instead of drifting on noise
            sol = np.linalg.solve(A.T @ A + 1e-3 * len(x) * np.eye(6), -A.T @ r); w, t = sol[:3] / L, sol[3:]
            ang = np.linalg.norm(w)
            R = rot(w / ang, np.degrees(ang)) if ang > 1e-12 else np.eye(3)
            T = tr(c + t) @ hom(R) @ tr(-c) @ T
            if ang < 1e-5 and np.linalg.norm(t) < 1e-4: break
    return T


def fit_score(P, N, T, tgt, tol=LAND_TOL):
    """(share of the samples within tol mm of a facing surface, rms of those distances)."""
    if tgt.tree is None or not len(P): return 0.0, float("nan")
    _X, ok, _j, d = pairs_at(P, N, T, tgt, tol)
    return float(ok.mean()), (float(np.sqrt((d[ok] ** 2).mean())) if ok.any() else float("nan"))


def _rank(sc): return (sc[0], -(sc[1] if sc[1] == sc[1] else 99.0))


def probe(P, N, k=15000, seed=1):
    i = np.random.default_rng(seed).choice(len(P), min(k, len(P)), replace=False); return P[i], N[i]


def into_box(P, T, lo, hi):
    """The shift that moves the placed samples' box inside the box lo..hi (each axis, as far as it fits)."""
    X = P @ T[:3, :3].T + T[:3, 3]; a, b = X.min(0), X.max(0); t = np.zeros(3)
    for k in range(3):
        if b[k] - a[k] >= hi[k] - lo[k]: t[k] = (lo[k] + hi[k] - a[k] - b[k]) / 2
        elif a[k] < lo[k]: t[k] = lo[k] - a[k]
        elif b[k] > hi[k]: t[k] = hi[k] - b[k]
    return t


def fit_flip(P, N, T_side, tgt, axis="auto"):
    """Flip a side 180 deg about one of its own principal axes and FIT it: every candidate (the three axes for auto) from
    its flipped pose and from that pose moved inside the other sides' box (a flipped shell belongs at the far end of the
    part), local ICP each, keep the best landing share (then the lowest rms). -> dict(T, axis, vec, scores, share, rms)."""
    c, ax = pca_frame(P); names = ("x", "y", "z")
    cands = [axis] if axis in names else list(names)
    Pp, Np = probe(P, N)
    box = (tgt.Q.min(0), tgt.Q.max(0)) if tgt.tree is not None else None
    best = None; scores = {}
    for nm in cands:
        vec = ax[:, names.index(nm)]; T0 = T_side @ flip_matrix(c, vec)
        top = None
        for S in [T0] + ([tr(into_box(Pp, T0, *box)) @ T0] if box is not None else []):
            T1 = local_icp(Pp, Np, tgt, S, FLIP_STEPS, 12); sc = fit_score(Pp, Np, T1, tgt)
            if top is None or _rank(sc) > _rank(top[1]): top = (T1, sc)
        scores[nm] = round(top[1][0], 4)
        if best is None or _rank(top[1]) > _rank(best[1]): best = (top[0], top[1], nm, vec)
    T = local_icp(P, N, tgt, best[0], KEEP_STEPS)
    share, rms = fit_score(Pp, Np, T, tgt)
    return {"T": T, "axis": best[2], "vec": [float(x) for x in best[3]], "scores": scores, "share": share, "rms": rms}


def solve_match(pairs, P, N, T_to, tgt):
    """pairs [[to_xyz, side_xyz]] in each side's own frame. Kabsch, then local ICP seeded from it; the picks win when ICP
    moves further than MATCH_MOVE or pushes the median pick residual above MATCH_PICK. -> dict(T, residuals, icp, warn)."""
    import registration_metrics as RM
    pr = RM.validate_landmark_pairs(pairs)
    B = np.array([p[0] for p in pr], float); M = np.array([p[1] for p in pr], float); T_to = np.asarray(T_to, float)
    T0 = T_to @ RM.kabsch_transform(M, B)
    Bw = B @ T_to[:3, :3].T + T_to[:3, 3]
    res = lambda T: np.linalg.norm(M @ T[:3, :3].T + T[:3, 3] - Bw, axis=1)
    T1 = local_icp(P, N, tgt, T0, FIT_STEPS); mv = moved(T0, T1, P.mean(0) if len(P) else M.mean(0))
    r0, r1 = res(T0), res(T1); warn = None
    if mv[0] > MATCH_MOVE[0] or mv[1] > MATCH_MOVE[1]: warn = "the surface fit moved %.1f mm / %.1f deg away from the points" % mv
    elif np.median(r1) > MATCH_PICK and np.median(r1) > np.median(r0) + 1.0: warn = "the surface fit drifts %.1f mm from the points" % np.median(r1)
    return {"T": T0 if warn else T1, "residuals": [round(float(x), 2) for x in (r0 if warn else r1)],
            "kabsch": [round(float(x), 2) for x in r0], "icp": warn is None, "warn": warn}


# A turn (Nudge's rotate rings) is kept in the side's OWN frame, M = Tr(c0) R Tr(-c0) with c0 the side's vertex mean, so it
# turns the side about its own centre whatever the placement; made on a built line-up it also keeps the user's pose against
# the placed sides (rel, like a flip), so a refit or a base change does not turn it again. Then a LOCAL fit seeded from that
# pose (never a global search); a fit that moves it more than MATCH_MOVE keeps the user's pose, with a warning. keep: no fit.
def turn_side_matrix(R_world, T_disp, c0):
    """The world rotation R_world about the placed centre T_disp(c0) as a side-frame matrix: T_disp @ M == D @ T_disp."""
    Ra = np.asarray(T_disp, float)[:3, :3]; c0 = np.asarray(c0, float)
    return tr(c0) @ hom(Ra.T @ np.asarray(R_world, float) @ Ra) @ tr(-c0)


def fit_turn(P, N, T0, tgt):
    """Local ICP from the user's pose T0. -> dict(T, moved (mm, deg), warn); the user's pose when the fit runs off."""
    T1 = local_icp(P, N, tgt, T0, FIT_STEPS); mv = moved(T0, T1, P.mean(0) if len(P) else np.zeros(3))
    warn = "the surface fit moved %.1f mm / %.1f deg away from where you put it" % mv if mv[0] > MATCH_MOVE[0] or mv[1] > MATCH_MOVE[1] else None
    return {"T": T0 if warn else T1, "moved": [round(mv[0], 2), round(mv[1], 2)], "warn": warn}


def resolve_turn_job(job):
    """engine.turn: the side-frame matrix of a world rotation R about the side's centre as the view shows it (T_disp, after
    nudges), and the user's pose (on the line-up before nudges, T) kept against the placed sides. -> {M, c, rel}."""
    s = job["side"]; c0 = plyio.vertex_mean(job["meshes"][s]); M = turn_side_matrix(job["R"], job["T_disp"], c0)
    out = {"M": M.tolist(), "c": [float(x) for x in c0], "rel": []}
    T = {k: np.array(m, float) for k, m in (job.get("T") or {}).items()}
    if s in T and len(T) > 1:
        samp = _sampler(job); P, N = samp(s)
        out["rel"] = anchors(P, N, T[s] @ M, {k: v for k, v in T.items() if k != s}, samp)
    return out


def _target(Ts, samp, skip=None, only=None):
    ks = [k for k in (only or Ts) if k != skip and k in Ts]
    if not ks: return Target(np.zeros((0, 3)), np.zeros((0, 3)))
    Xs = [placed(*samp(k), Ts[k]) for k in ks]
    return Target(np.vstack([x[0] for x in Xs]), np.vstack([x[1] for x in Xs]))


def anchors(P, N, T, Ts, samp, k=3):
    """The placed sides a pose is kept against, most overlap first: [[to, inv(T[to]) @ T]]."""
    Pp, Np = probe(P, N, 4000)
    sc = sorted(((fit_score(Pp, Np, T, _target(Ts, samp, only=[o]))[0], o) for o in Ts), key=lambda x: -x[0])
    return [[o, (np.linalg.inv(Ts[o]) @ T).tolist()] for _s, o in sc[:k]]


def _sampler(job):
    got = {}
    def samp(s):
        if s not in got: got[s] = side_samples(job["meshes"][s], (job.get("digests") or {}).get(s), job.get("cache"))
        return got[s]
    return samp


def resolve_flip_job(job):
    """engine.flip and its preview (one code path): fit the flip on the placement the user sees and keep it against the
    placed sides. -> {T (model frame), axis, vec, scores, share, rel [[to, 4x4]]}."""
    T = {s: np.array(m, float) for s, m in job["T"].items()}; s = job["side"]; samp = _sampler(job)
    P, N = samp(s); r = fit_flip(P, N, T[s], _target(T, samp, s), job.get("axis", "auto"))
    rel = anchors(P, N, r["T"], {k: v for k, v in T.items() if k != s}, samp)
    return {"T": r["T"].tolist(), "axis": r["axis"], "vec": r["vec"], "scores": r["scores"], "share": r["share"], "rel": rel}


def fix_job(job):
    """lineup.fix: the flip, match and turn ops in order. A flip resolved when it was made (rel) lands on its kept pose against
    the first of its anchor sides still placed, then a small local fit; one without rel (made before any line-up) is
    fitted here. A match solves its picks against its "to" side. Never a global search."""
    pl = jload(job["placement"]); T = {s: np.array(m, float) for s, m in pl["T"].items()}; samp = _sampler(job)
    flipped, matched, turned, fits, warns = {}, {}, {}, {}, []
    for op in job["flips"]:
        s = op["side"]
        if s not in T: continue
        P, N = samp(s)
        if op.get("tool") == "match":
            to = op.get("to")
            if to not in T or to == s: warns.append("side %s: the side its points match is left out, so they are not used" % s); continue
            try: r = solve_match(op["pairs"], P, N, T[to], _target(T, samp, only=[to]))
            except ValueError as e: warns.append("side %s: %s" % (s, e)); continue
            T[s] = r["T"]; matched[s] = {"to": to, "residuals": r["residuals"], "kabsch": r["kabsch"], "icp": r["icp"]}
            if r["warn"]: warns.append("side %s: %s; kept the fit to the points" % (s, r["warn"])); matched[s]["warn"] = r["warn"]
            print("matched side %s to side %s: point residuals %s mm%s" % (s, to, r["residuals"], "" if r["icp"] else " (points only)"), flush=True)
        elif op.get("tool") == "turn":
            anc = next(((o, np.array(A, float)) for o, A in (op.get("rel") or []) if o in T and o != s), None)
            seed = T[anc[0]] @ anc[1] if anc is not None else T[s] @ np.array(op["M"], float)
            if op.get("keep"): T[s] = seed; turned[s] = {"fit": False}
            else:
                r = fit_turn(P, N, seed, _target(T, samp, s)); T[s] = r["T"]; turned[s] = {"fit": r["warn"] is None, "moved": r["moved"]}
                if r["warn"]: warns.append("side %s: %s; kept it as placed" % (s, r["warn"])); turned[s]["warn"] = r["warn"]
            print("turned side %s%s%s" % (s, " on side %s" % anc[0] if anc else "", ", refitted" if turned[s]["fit"] else " (kept as placed)"), flush=True)
        else:
            anc = next(((o, np.array(M, float)) for o, M in (op.get("rel") or []) if o in T and o != s), None)
            tgt = _target(T, samp, s)
            if anc is not None:
                T[s] = local_icp(P, N, tgt, T[anc[0]] @ anc[1], KEEP_STEPS); flipped[s] = op.get("chose") or op.get("axis", "auto")
                print("flipped side %s: kept pose on side %s, refitted" % (s, anc[0]), flush=True)
            else:
                r = fit_flip(P, N, T[s], tgt, op.get("axis", "auto")); T[s] = r["T"]; flipped[s] = r["axis"]
                print("flipped side %s about its %s axis (landing %s)" % (s, r["axis"], ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in r["scores"].items())), flush=True)
        sh, rms = fit_score(*probe(P, N), T[s], _target(T, samp, s))
        fits[s] = {"share": round(sh, 3), "rms": None if rms != rms else round(rms, 3)}
    pl["T"] = {s: T[s].tolist() for s in pl["order"]}; pl["flipped"] = flipped; pl["matched"] = matched; pl["turned"] = turned
    jdump(pl, os.path.join(job["out_dir"], "placement.json"))
    return {"flipped": flipped, "matched": matched, "turned": turned, "fits": fits, "warn": warns}


# ------------------------------------------------------------------------------------------------------ delete
def elements(kind, data):
    """Positions the selection tests: points, or face centroids of a mesh."""
    if kind == "points": return data["P"].astype(np.float64)
    V, F, _ = data
    return V[F].mean(1) if len(F) else np.zeros((0, 3))


def anchor_fix(op, T_now):
    """T_fix that carries the op's recorded model frame into the current one (identity for side-frame ops)."""
    if op.get("frame", "model") != "model" or not op.get("anchor") or not T_now: return np.eye(4)
    an = op["anchor"]
    for s in an.get("order", []):
        if s in T_now and s in an.get("T", {}):
            return np.asarray(T_now[s], float) @ np.linalg.inv(np.asarray(an["T"][s], float))
    return np.eye(4)


def select_shape(X, op, Tfix=np.eye(4)):
    """Boolean mask of the positions X inside the op's box or sphere (after re-anchoring)."""
    if op["shape"] == "box":
        b = op["box"]; Tb = np.asarray(b.get("T") if b.get("T") is not None else np.eye(4), float)
        M = np.linalg.inv(Tfix @ Tb); Y = X @ M[:3, :3].T + M[:3, 3]
        lo, hi = np.minimum(b["min"], b["max"]), np.maximum(b["min"], b["max"])
        return ((Y >= lo) & (Y <= hi)).all(1)
    if op["shape"] == "sphere":
        c = Tfix[:3, :3] @ np.asarray(op["center"], float) + Tfix[:3, 3]
        return np.linalg.norm(X - c, axis=1) <= float(op["radius"])
    raise ValueError("shape %r is selected through its region" % op["shape"])


def voxel_down(P, voxel, cap=MASK_MAX, seed=0):
    if not len(P): return P.astype(np.float32)
    k = np.floor(P / voxel).astype(np.int64)
    _, first = np.unique(k, axis=0, return_index=True)
    Q = P[np.sort(first)]
    if len(Q) > cap: Q = Q[np.random.default_rng(seed).choice(len(Q), cap, replace=False)]
    return Q.astype(np.float32)


def make_region(X, sel, input_digest):
    """dead = selected positions, live = unselected ones within 3 mm of them (the other skin of a thin wall), idx."""
    from scipy.spatial import cKDTree
    dead = voxel_down(X[sel], DEAD_VOXEL)
    rest = np.nonzero(~sel)[0]
    if len(dead) and len(rest):
        d, _ = cKDTree(dead).query(X[rest], distance_upper_bound=LIVE_REACH)
        live = voxel_down(X[rest[np.isfinite(d)]], DEAD_VOXEL)
    else: live = np.zeros((0, 3), np.float32)
    return {"dead": dead, "live": live, "idx": np.nonzero(sel)[0].astype(np.int32), "input_digest": np.array(input_digest)}


def save_region(regions_dir, reg):
    buf = io.BytesIO(); np.savez(buf, **reg); b = buf.getvalue()
    sha = hashlib.blake2b(b, digest_size=8).hexdigest()
    os.makedirs(regions_dir, exist_ok=True); p = os.path.join(regions_dir, sha + ".npz")
    if not os.path.exists(p):
        tmp = p + ".tmp.%d" % os.getpid()
        with open(tmp, "wb") as f: f.write(b)
        os.replace(tmp, p)
    return "regions/%s.npz" % sha


def select_mask_region(X, reg, Tfix=np.eye(4)):
    from scipy.spatial import cKDTree
    dead = reg["dead"].astype(np.float64) @ Tfix[:3, :3].T + Tfix[:3, 3]
    live = reg["live"].astype(np.float64) @ Tfix[:3, :3].T + Tfix[:3, 3]
    if not len(dead) or not len(X): return np.zeros(len(X), bool)
    dd, _ = cKDTree(dead).query(X, distance_upper_bound=REPLAY_NEAR + 1e-9)
    near = np.isfinite(dd)
    if len(live):
        dl, _ = cKDTree(live).query(X[near]); ok = dd[near] < dl
    else: ok = np.ones(near.sum(), bool)
    out = np.zeros(len(X), bool); out[np.nonzero(near)[0][ok]] = True
    return out


def replay_mask(X, op, parent_digest, T_now, stages_dir):
    """-> (selected mask, how) following the replay rules."""
    Tfix = anchor_fix(op, T_now)
    if op["shape"] == "mask":
        reg = dict(np.load(os.path.join(stages_dir, op["region"])))
        rec = (op.get("recorded") or {}).get("input_digest")
        if rec and rec == parent_digest and "idx" in reg and (len(reg["idx"]) == 0 or reg["idx"].max() < len(X)):
            m = np.zeros(len(X), bool); m[reg["idx"]] = True; return m, "exact"
        return select_mask_region(X, reg, Tfix), "geometric"
    return select_shape(X, op, Tfix), "shape"


def mask_side_depth(depth, pose, cal, ops, stages_dir, depth_scale=0.1):
    """Remove a side's saved delete regions from its depth frame before any multi-side fusion.

    pose is camera to the side's scan frame in millimeters. The regions were recorded
    in that same side frame, so another side can still contribute in an overlap.
    """
    if not ops or not np.any(depth):
        return depth
    v, u = np.nonzero(depth)
    fx, fy, cx, cy = cal
    z = depth[v, u].astype(np.float64) * depth_scale
    camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    camera[:, 1:] *= -1  # OpenCV camera to scanner camera frame
    side = camera @ pose[:3, :3].T + pose[:3, 3]
    removed = np.zeros(len(side), bool)
    for op in ops:
        if not op.get("on", True):
            continue
        selected, _ = replay_mask(side, op, None, None, stages_dir)
        removed |= selected
    if removed.any():
        depth = depth.copy()
        depth[v[removed], u[removed]] = 0
    return depth


def apply_mask(kind, data, sel):
    """Remove the selected elements. Returns the new data."""
    if kind == "points":
        keep = ~sel
        return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(sel) and k not in ("side_ids",) else v)
                for k, v in data.items()}
    V, F, N = data
    F2 = F[~sel]
    used = np.unique(F2); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[F2], (N[used] if N is not None else None)


@register
class Delete(Tool):
    name = "delete"; label = "Delete (box / sphere / lasso mask)"; kinds = ("mesh", "points"); version = 2; mem_gb = 3; detect = True
    modules = ("tools/edits.py", "tools/common.py", "plyio.py"); libs = ("numpy", "scipy")

    def execute(self, ctx):
        from .common import run_tool_job
        return run_tool_job(ctx, self)

    def summary(self, p):
        return ""


def points_area(data, sel, k=16, cand=40, chunk=100000):
    """mm^2 of surface the selected points cover, comparable to a mesh's face area (so a delete recorded on a tsdf mesh
    and replayed on points after a method switch is not a false 'check'). Each selected point stands for the surface
    area around it: pi r^2 / m, r = the tangent-plane distance to its m-th nearest same-facing neighbour (m = k, or fewer
    when it has fewer), the usual nearest-neighbour density estimate. Distances are measured in the point's tangent
    plane (noise across the surface does not count) and only between points facing the same way (the other skin of a
    thin wall is its own surface). So several points per cell (normal bins, an overlap dedupe at a finer cell) do not
    inflate the area, unlike count x cell^2. A point with fewer than 3 such neighbours counts cell^2."""
    idx = np.nonzero(sel)[0]
    if not len(idx): return 0.0
    from scipy.spatial import cKDTree
    P = np.asarray(data["P"], np.float64); N = np.asarray(data["N"], np.float64); c = float(data.get("cell", 0.4))
    tree = cKDTree(P); kk = min(cand, len(P)); total = 0.0
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; X = P[ii]; Nx = N[ii]
        _d, nb = tree.query(X, k=kk, workers=8)
        nb = nb.reshape(len(ii), kk)
        D = P[nb] - X[:, None, :]
        along = np.einsum("nkj,nj->nk", D, Nx)
        t2 = np.maximum((D * D).sum(-1) - along * along, 0.0)
        same = np.einsum("nkj,nj->nk", N[nb], Nx) > 0.3
        t2 = np.where(same, t2, np.inf); t2.sort(1)            # the point itself first (distance 0)
        m = np.minimum(k, same.sum(1) - 1)
        r2 = t2[np.arange(len(ii)), np.clip(m, 0, kk - 1)]
        a = np.where(m >= 3, np.pi * r2 / np.maximum(m, 1), c * c)
        total += float(a.sum())
    return total


def removed_area(kind, data, sel):
    """mm^2 the selection covers: face areas for a mesh, points_area for points."""
    if kind == "points": return points_area(data, sel)
    V, F, _ = data
    if not sel.any(): return 0.0
    f = F[sel]
    return float(0.5 * np.linalg.norm(np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]]), axis=1).sum())


def delete_job(job, kind, data):
    """Called by tools.common.tool_job for delete: returns (new data, removed mask, extra stats)."""
    op = job["op"]; X = elements(kind, data)
    sel, how = replay_mask(X, op, job.get("parent_digest"), job.get("T_now"), job["stages"])
    area = removed_area(kind, data, sel)
    st = {"replay": how, "removed": int(sel.sum()), "removed_mm2": round(area, 2)}
    rec = op.get("recorded") or {}
    if how != "exact" and (rec.get("area_mm2") is not None or rec.get("count") is not None):
        if rec.get("area_mm2") is not None: drift = abs(area - rec["area_mm2"]) / max(1e-9, rec["area_mm2"])
        else: drift = abs(int(sel.sum()) - rec["count"]) / max(1, rec["count"])
        st["drift"] = round(drift, 3)
        if drift > CHECK_DRIFT: st["check"] = True
    return apply_mask(kind, data, sel), sel, st


def make_mask_job(job):
    """delete --mask pts.ply: select the elements within radius of any mask point and save the region file."""
    from scipy.spatial import cKDTree
    from .common import load
    kind = job["kind"]; data = load(kind, job["in"]); X = elements(kind, data)
    M = plyio.xyz(plyio.read(job["mask"], faces=False)["vertex"])
    d, _ = cKDTree(M).query(X, distance_upper_bound=float(job["radius"]) + 1e-9)
    sel = np.isfinite(d)
    reg = make_region(X, sel, job["input_digest"])
    return {"region": save_region(job["regions"], reg), "count": int(sel.sum())}
