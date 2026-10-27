#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learn bed: know the turntable once, then find it in every frame from the first second.

The platter is dark with light marker dots. The dots are the brightest round blobs in the IR picture at any gain, they
sit on the platter by construction, and their layout is not symmetric. So once the dot layout (the bed map) is known,
one frame is enough to place the whole platter: pick a plane for the dots, cast each dot's IR centre onto it, and match
the pattern to the map in 2D. That gives the platter plane from frame 2-3 (Auto remove base needs 8+ frames and a turned
part, and sometimes takes a flat face of the part: the part has no dots) and the table's angle, which places the camera
from the dots alone (a lost frame is found again without the slow feature search).

Frames and units: mm everywhere (only the Tracker works in metres). Camera = OpenCV depth camera; depth d is uint16 in
0.1 mm; cal = (fx, fy, cx, cy) from fuse.read_calib. A plane is n . x + off = 0 with n pointing toward the camera and the
part (off > 0 in the camera frame); moving it lift mm away from the camera is off += lift. table_plane.json stores
d = -off (fuse.py keeps n . x >= d + margin). A pose-table entry M (scanner camera -> scan) gives T_scan<-cam = M @ F.
The bed frame has its origin where the spin axis meets the platter top, z = the platter normal toward the part, x = the
direction of the most-seen dot when it was learned; the dots are 2D points on z = 0. B_cam (bed -> camera) comes from
one frame; B_scan = T_scan<-cam @ B_cam is constant for a side while the part stays put on the table.

Measured on two projects (20 sides): 29-32 dots at radius 19-96 mm, 18-25 mm apart, 4-22 dots per frame. Some depth
sits inside the dots at every gain (they read ~0.8 mm high); the platter around them gives depth from gain ~4 up. So
the plane comes from both (dot depth and the ring around the dots), and every dot is placed by its IR centre's ray
meeting the plane, never by its own depth. The height comes from the platter surface when it gives depth, else from
the dots minus the learned lift.

Beds and outlines: several named beds (profiles, bed-<name>.json next to bed.json) of kind "dots" (a dot map; a
calibration board's dots sit in broken rings, board "ringed", with a flag per dot that matching uses) or "plane" (no
dots: the bare turntable, found from depth only; its IR letters are never looked at). A bed's outline is a circle or a
rectangle in the bed frame (Outline), measured when learning; every in-bed check (nothing beneath, the height refit,
the cut below the bed, the rim drawn on the IR picture) uses it. A circle about the spin centre keeps the disc maths
of before.

usage:
  bed.py learn-sides <side dir>... --out bed.json         learn a bed from saved, tracked sides (offline seed)
  bed.py side-pose <side dir> --bed bed.json --out f.json  where the bed is in a saved side (the build runs this)
"""
import os, sys, json, time, hashlib, argparse
from collections import namedtuple
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from table_plane import fit_plane, to_camera, F, SidePlane      # noqa: E402

W, H = 800, 600

# ---- dots in the IR picture
DOT_THR_FLOOR = 120            # primary threshold: max(120, p99 // 2); fewer than HIT_MIN dots -> max(25, 0.7 * max)
DOT_THR_ALT = (25, 0.7)
DOT_AREA = (12, 2500)          # px
DOT_ASPECT = 3.5               # bbox long / short side: a dot seen from low is a flat ellipse (3.5 found both low-camera sides, 0 wrong)
DOT_FILL = 0.7                 # area / ellipse area of the bbox
DOT_BORDER = 3                 # px: a dot cut by the picture edge has a wrong centre
DOT_RING = 0.45                # ring median <= this x inside mean: a glint on a bright part is not a dot
DARK_IR = 90                   # platter pixels are darker than this
RING_SHARE = 0.15              # a dot is ringed when this share of the band 1.45-3.2 r around it is bright (an arc)
# ---- planes
RANSAC_DOT = (1.5, 300, 4)     # mm, iterations, min inliers (dot depth)
RANSAC_RING = (1.0, 200, 60)   # mm, iterations, min points (depth ring around the dots)
ON_PLANE = 2.5                 # mm: a dot with depth further than this from a candidate plane is not on it
# ---- 2D match
PAIR_RANGE = (15.0, 140.0)     # mm, frame dot pairs used as hypotheses
PAIR_MAX = 60                  # hypotheses pairs sampled per frame
PAIR_TOL, INLIER_TOL = 1.5, 2.0
PAIR_MIN_BEFORE_EXIT = 20      # pairs looked at before the early exit may stop the search (keeps the runner-up honest)
RUNNER_UP_DEG, RUNNER_UP_MM = 5.0, 5.0
HIT_MIN, HIT_RATIO, HIT_RMS = 7, 2.0, 1.4
# ---- acceptance, height, checks
AGREE_DEG, AGREE_MM, AGREE_CENTRE = 1.5, 2.0, 4.0
ACCEPT_HITS, ACCEPT_WINDOW = 3, 30
SURFACE_MIN, SURFACE_BAND0, SURFACE_BAND, SURFACE_DEG = 300, 3.0, 1.0, 1.0   # refit band +-1 mm: +-2 took in a face 1.5 mm up
SURFACE_VS_DOTS = 1.5          # mm: a surface refit this far OVER the dots' own plane is a dark face of the part near the platter
LOW_MM = 6.0                   # mm: the platter's layer is looked for down to this far under the candidate plane
LAYER_MIN = 150                # depth pixels (every 2nd row and column) a 2 mm layer needs to be the platter's; the refit itself
                               # needs SURFACE_MIN (300) within +-SURFACE_BAND of it (fewer made plain frames worse: p90 1.03 -> 1.27 mm)
LAYER_UP = 4.0                 # mm: from the lowest dense layer the platter's peak is climbed to, at most this far up (noise spreads it down)
PRIOR_HEIGHT_MM = 1.5          # a prior plane is moved to the dots' height only when they disagree by more than this
V1_BELOW_MM, V1_MAX = 4.0, 0.03
OUTLINE_RECT_SHARE = 0.8      # dots filling a rectangle under this share of their circle's area: a rectangular bed
OUTLINE_PAST = 15.0            # mm: the dark surface counts toward the outline only this far past the dots (a dark table top
                               # at the same height gave 120-270 mm)
V2_DEG = 3.0
V3_DEG, V3_MM = 3.0, 3.0
V3_UNDER_MM = 8.0            # Auto this far or more under the dots found something UNDER the table (floor, turntable base:
                             # seen live zoomed out, 58 mm down), not the platter: the dots stay
V4_DEG, V4_MM, V4_RUN = 3.0, 5.0, 3
POOL_FROM, POOL_EVERY, POOL_TO = 10, 10, 90
POOL_RANSAC = (2.0, 500)
POOL_CLUSTER, POOL_MIN_OBS = 4.0, 2
POOL_MAX_DOTS, POOL_MAX_RING, POOL_MAX_FRAMES = 25, 400, 90
LEARN_TURN_DEG, LEARN_MIN_DOTS, LEARN_TIMEOUT = 370.0, 12, 150.0        # 150 s: a slow table (one turn in two minutes) still learns
HINT_GUARD_MM, HINT_GUARD_DEG = 15.0, 5.0     # a hinted frame must land this close to the dots' pose (worst dots pose seen: 8.5 mm, 2.4 deg)
HINT_GIVE_UP = 2               # refused hinted fits that agree on where the part now sits on the table: it moved there, the dots no longer place it
HINT_AGREE_MM, HINT_AGREE_DEG = 10.0, 5.0     # ... agree: the same move of the part within this (at the platter centre)
ON_TABLE_MM, ON_TABLE_DEG = 5.0, 5.0          # a part moved on the table stays on it: the platter's plane in the part's frame moves by at most this
HINT_APART_DEG = 8.0           # ... and the agreeing fits come from views this far apart (the table turned between them): from nearly
                               # the same view, ICP stopped part way from the hint lands in nearly the same wrong place (seen: 8 and 11 deg
                               # for a 30 deg move, two frames in a row)
HINT_ON_MOVE = "pause"         # the part moved on the table: "pause" the pass and ask for a new side (default: in replays,
                               # "follow" (hint from where the agreeing fits put the part) fused frames 2-6 deg / 12-40 mm
                               # off in half the moves it took, which the guard alone would have left lost); "stop" hints
HINT_FOLLOW = HINT_ON_MOVE == "follow"
CAM_RUN = 3                    # tracked frames in a row that put the platter elsewhere than this pass's camera bed: the scanner moved
CAM_CENTRE_MM = 25.0           # ... its plane by V4_DEG / V4_MM, or its centre by this (20 still sides: plane 2.5 deg / 2.6 mm, centre 16 mm at most;
                               # a slide along the platter moves only the centre, and only the rim ring drawn from it)
LEARN_KEEP = 0.15              # a learned dot must be seen at least this share as often as a typical dot (specks are seen a few times)
LEARN_SEEN_MIN = 0.25          # a learned map is refused when its dots were seen in fewer than this share (median) of the frames
                               # that showed their place (7 good maps: 0.49-0.72; a smeared one, dots lost on the way: 0.07) ...
LEARN_HALVES_MIN = 0.7         # ... and the maps of the even and the odd frames agree on fewer than this share of the dots (good 0.9-1.0, smeared 0.49)
LEARN_REG_MAX = 60             # registration uses at most this many of the most-seen dots (a platter has ~30; a map smeared into
                               # hundreds made every frame's match take seconds and GBs)
LEARN_STILL_DEG = 20.0         # turned less than this in LEARN_TIMEOUT: the table did not turn
RINGED_BOARD = 0.3             # this share of the learned dots in bright rings: a ringed board (every other dot, as on a ringed calibration board)

Reg = namedtuple("Reg", "R t inliers rms runner_up mask nn")


def _jl(x, nd=6):
    """numpy -> JSON-able lists, rounded."""
    if x is None: return None
    a = np.asarray(x, float)
    return np.round(a, nd).tolist()


def _unit(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-12)


def ang_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def backproject(u, v, z, cal):
    """Pixels (u, v) at depth z (mm) -> camera-frame points (..., 3)."""
    return np.stack([(u - cal[2]) * z / cal[0], (v - cal[3]) * z / cal[1], z], -1)


def basis(n):
    """(e1, e2) spanning the plane with normal n, e1 x e2 = n (right-handed, so a mirrored match is impossible)."""
    n = _unit(n); e1 = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.3: e1 = np.cross(n, [0, 1.0, 0])
    e1 = _unit(e1); return e1, np.cross(n, e1)


def circle_centre(C, n):
    """Centre (3D, in the plane through the points' mean along n) and radius of the circle the points C lie near."""
    e1, e2 = basis(n); m = C.mean(0); Q = C - m; x, y = Q @ e1, Q @ e2
    a, b, c = np.linalg.lstsq(np.stack([x, y, np.ones(len(x))], 1), x * x + y * y, rcond=None)[0]
    return m + (a / 2) * e1 + (b / 2) * e2, float(np.sqrt(max(c + (a / 2) ** 2 + (b / 2) ** 2, 0.0)))


# ------------------------------------------------------------------------------------------------ the bed's outline
class Outline:
    """The bed's shape in the bed frame (mm, on z = 0, origin where the spin axis meets the platter): a "circle"
    (centre, radius) or a "rect" (centre, width along its own x, depth along its own y, angle = degrees from the bed
    frame's x to the rectangle's x). A round platter is a circle about the spin centre; a square mat or a calibration
    board lying on the turntable is a rect, anywhere on it."""
    __slots__ = ("shape", "centre", "radius", "width", "depth", "angle")

    def __init__(self, shape="circle", centre=(0.0, 0.0), radius=97.0, width=0.0, depth=0.0, angle=0.0):
        self.shape = "rect" if shape == "rect" else "circle"; self.centre = np.asarray(centre, float).reshape(2)
        self.radius = float(radius or 0.0); self.width = float(width or 0.0); self.depth = float(depth or 0.0)
        self.angle = float(angle or 0.0)

    @classmethod
    def from_json(cls, j, radius_mm=97.0):
        if not j: return cls("circle", (0.0, 0.0), radius_mm)
        return cls(j.get("shape", "circle"), j.get("centre") or (0.0, 0.0), j.get("radius", radius_mm), j.get("width"),
                   j.get("depth"), j.get("angle"))

    def to_json(self):
        c = [round(float(v), 2) for v in self.centre]
        if self.shape == "circle": return {"shape": "circle", "centre": c, "radius": round(self.radius, 2)}
        return {"shape": "rect", "centre": c, "width": round(self.width, 2), "depth": round(self.depth, 2),
                "angle": round(self.angle, 2)}

    def key(self):
        """A plain tuple (the window's BedInfo carries it): ("circle", cx, cy, r) or ("rect", cx, cy, w, d, angle)."""
        c = tuple(round(float(v), 1) for v in self.centre)
        if self.shape == "circle": return ("circle",) + c + (round(self.radius, 1),)
        return ("rect",) + c + (round(self.width, 1), round(self.depth, 1), round(self.angle, 2))

    @property
    def centred_circle(self):
        """A circle about the spin centre: the platter every learned round bed has (its maths stay the old ones)."""
        return self.shape == "circle" and not self.centre.any()

    def _local(self, xy):
        q = np.asarray(xy, float).reshape(-1, 2) - self.centre
        a = np.radians(self.angle); c, s = np.cos(a), np.sin(a)
        return np.stack([q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c], 1)

    def sdf2(self, xy):
        """Signed distance (mm) of bed-frame points (N, 2) to the outline: > 0 outside, < 0 inside."""
        if self.shape == "circle": return np.linalg.norm(np.asarray(xy, float).reshape(-1, 2) - self.centre, axis=1) - self.radius
        L = np.abs(self._local(xy)) - [self.width / 2, self.depth / 2]
        return np.linalg.norm(np.maximum(L, 0), axis=1) + np.minimum(L.max(1), 0)

    def poly(self, k=120):
        """Points around the outline (k, 2), bed frame, counter-clockwise."""
        if self.shape == "circle":
            a = np.linspace(0, 2 * np.pi, k, endpoint=False); return self.centre + self.radius * np.stack([np.cos(a), np.sin(a)], 1)
        hw, hd = self.width / 2, self.depth / 2; m = max(1, k // 4)
        t = np.linspace(-1, 1, m, endpoint=False)
        L = np.concatenate([np.stack([hw * t, -hd * np.ones(m)], 1), np.stack([hw * np.ones(m), hd * t], 1),
                            np.stack([-hw * t, hd * np.ones(m)], 1), np.stack([-hw * np.ones(m), -hd * t], 1)])
        a = np.radians(self.angle); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return L @ R.T + self.centre

    @property
    def reach(self):
        """The farthest the outline gets from the spin centre (a radius for code that wants one)."""
        if self.shape == "circle": return float(np.linalg.norm(self.centre) + self.radius)
        return float(np.linalg.norm(self.poly(8), axis=1).max())

    def placed(self, x_axis): return PlacedOutline(self, x_axis)


class PlacedOutline:
    """An outline placed in some frame: the bed frame's x axis there. The platter tuples (n, off, centre, radius) carry
    it in the radius's place for a bed that is not a circle about the spin centre; n and the centre come from the tuple.
    float() is its reach, so code that only wants a radius still gets one."""
    __slots__ = ("outline", "x_axis")

    def __init__(self, outline, x_axis): self.outline = outline; self.x_axis = np.asarray(x_axis, float)

    def _axes(self, n):
        n = _unit(n); x = _unit(self.x_axis - (self.x_axis @ n) * n); return x, np.cross(n, x)

    def sdf(self, P, n, c):
        """Signed distance (mm, > 0 outside) of 3D points P (N, 3) to the outline, measured in the plane (along n)."""
        x, y = self._axes(n); q = np.asarray(P, float).reshape(-1, 3) - np.asarray(c, float)
        return self.outline.sdf2(np.stack([q @ x, q @ y], 1))

    def ring(self, n, c, k=120):
        """(k, 3) points around the outline in this frame."""
        x, y = self._axes(n); Q = self.outline.poly(k)
        return np.asarray(c, float) + Q[:, :1] * x + Q[:, 1:] * y

    def rotated(self, R): return PlacedOutline(self.outline, np.asarray(R, float) @ self.x_axis)

    def __float__(self): return self.outline.reach

    def __eq__(self, o):
        return isinstance(o, PlacedOutline) and o.outline.key() == self.outline.key() and np.array_equal(o.x_axis, self.x_axis)

    __hash__ = None


def _min_area_rect(Q):
    """The smallest rectangle round 2D points: (centre, width, depth, angle deg), width along the angle, width >= depth."""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(Q, np.float32).reshape(-1, 1, 2))
    if h > w: w, h, a = h, w, a + 90.0
    a = (a + 90.0) % 180.0 - 90.0
    return np.array([cx, cy]), float(w), float(h), float(a)


def fit_outline(M, dot_diam=None, surf=None, reach=None, shape=None):
    """The bed's outline from its learned dots M (N, 2, bed frame) and, when seen, the dark flat surface around them
    (surf (K, 2), same frame, only out to OUTLINE_PAST mm past the dots). A circle about the spin centre (radius: reach
    when given, the learner's own number) unless the dots fill a rectangle much better than that circle
    (min-area rectangle under OUTLINE_RECT_SHARE of the circle's area: a square mat, a calibration board), then a
    rectangle grown by half a dot and 3 mm, and out to the surface's 98th percentile on each side. shape forces one."""
    M = np.asarray(M, float).reshape(-1, 2); grow = (dot_diam / 2 if dot_diam else 4.0) + 3.0
    rad = float(reach) if reach is not None else float(np.linalg.norm(M, axis=1).max() + grow) if len(M) else 97.0
    circle = Outline("circle", (0.0, 0.0), rad)
    if len(M) < 4 or shape == "circle": return circle
    c, w, h, a = _min_area_rect(M)
    w, h = w + 2 * grow, h + 2 * grow
    if shape != "rect" and w * h >= OUTLINE_RECT_SHARE * np.pi * rad * rad: return circle
    o = Outline("rect", c, width=w, depth=h, angle=a)
    if surf is not None and len(surf) >= 300:
        L = o._local(surf); lim = np.array([w / 2, h / 2]) + OUTLINE_PAST
        L = L[(np.abs(L) < lim).all(1)]
        if len(L) >= 300:
            ext = [max(w / 2, float(np.percentile(np.abs(L[:, 0]), 98))), max(h / 2, float(np.percentile(np.abs(L[:, 1]), 98)))]
            o.width, o.depth = 2 * ext[0], 2 * ext[1]
    return o


def quiet_threads():
    """One thread for OpenCV and for numpy's BLAS in this process. The bed's matrices are small (a few thousand points,
    3 x 300 planes): a thread pool gains nothing there and its spinning threads burned 9-10x the wall time in CPU
    (find_bed, 24 cores: 36 ms wall either way, 340 ms CPU with the pools, 37 ms with one thread; the pools also spun
    for 450 ms of CPU in the idle second after). OpenBLAS's pool is process-wide, so this also holds for the window's own small
    numpy work; the tracker runs on the GPU, and OpenMP (Open3D's feature search) is left alone. Call it on the main
    thread before other threads start (threadpoolctl walks the loaded libraries: from a worker thread that deadlocked
    against the main thread loading one). Returns the threadpoolctl limiter or None without threadpoolctl."""
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except Exception:
        return None


def calib_md5(param_dir):
    p = os.path.join(param_dir, "Pl.bin")
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------ dots and depth
def _dots_at(ir, thr, rings=False, stats=None):
    """Round bright blobs at threshold thr: [(cx, cy, r, area, w, h, ringed)]. rings (a board whose dots sit in broken
    rings, bed kind "dots" with board "ringed"): a blob whose ring test fails is still a dot when the gap right around
    it is dark and a bright ring lies past it (ringed = True); arcs themselves are thin and fail the fill test, and a
    short arc that passes it next to a ringed dot is dropped (_drop_arcs). stats: counts of the bright blobs of dot size
    ("blobs") and those left out as not round ("not_round")."""
    n, lab, st, ce = cv2.connectedComponentsWithStats((ir >= thr).astype(np.uint8), connectivity=8)
    if n <= 1: return []
    x, y, w, h, a = (st[1:, k] for k in range(5))
    size = (a >= DOT_AREA[0]) & (a <= DOT_AREA[1])
    ok = (size & (np.maximum(w, h) <= DOT_ASPECT * np.minimum(w, h))
          & (a >= DOT_FILL * np.pi / 4 * w * h) & (x >= DOT_BORDER) & (y >= DOT_BORDER)
          & (x + w <= ir.shape[1] - DOT_BORDER) & (y + h <= ir.shape[0] - DOT_BORDER))
    if stats is not None:
        inner = size & (x >= DOT_BORDER) & (y >= DOT_BORDER) & (x + w <= ir.shape[1] - DOT_BORDER) & (y + h <= ir.shape[0] - DOT_BORDER)
        stats["blobs"] = stats.get("blobs", 0) + int(inner.sum()); stats["not_round"] = stats.get("not_round", 0) + int((inner & ~ok).sum())
    out = []
    for i in np.nonzero(ok)[0] + 1:
        bx, by, bw, bh, ba = st[i]; cx, cy = ce[i]; r = 0.5 * max(bw, bh)
        pad = int(3.2 * r + 4) if rings else int(2.4 * r + 3)
        x0, x1 = max(0, int(cx) - pad), min(ir.shape[1], int(cx) + pad + 2)
        y0, y1 = max(0, int(cy) - pad), min(ir.shape[0], int(cy) + pad + 2)
        patch = ir[y0:y1, x0:x1].astype(np.float32); comp = lab[y0:y1, x0:x1] == i
        yy, xx = np.mgrid[y0:y1, x0:x1]; rr = np.hypot(xx - cx, yy - cy)
        ring = (rr > 1.6 * r + 1) & (rr < 2.4 * r + 2)
        if ring.sum() < 8: continue
        inside = float(patch[comp].mean()); outside = float(np.median(patch[ring]))
        ringed = False
        if outside > DOT_RING * inside:
            if not rings: continue
            gap = (rr > 1.15 * r + 1) & (rr < 1.45 * r + 1) & ~comp
            past = (rr > 1.45 * r + 1) & (rr < 3.2 * r + 3)
            if gap.sum() < 8 or float(np.median(patch[gap])) > DOT_RING * inside: continue
            outside = float(np.median(patch[gap])); ringed = True
        elif rings:
            past = (rr > 1.45 * r + 1) & (rr < 3.2 * r + 3)
            ringed = bool(past.sum() >= 8 and float((patch[past] >= thr).mean()) >= RING_SHARE)
        m = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        g = np.maximum(patch - outside, 0) * m; tot = float(g.sum())
        if tot <= 0: continue
        out.append(((g * xx).sum() / tot, (g * yy).sum() / tot, r, float(ba), float(bw), float(bh), ringed))
    if rings: out = _drop_arcs(out, stats)
    return out


def _drop_arcs(out, stats=None):
    """A short arc of a broken ring can pass as a small dot: a blob within 3.2 r of a bigger ringed dot and under 0.7 of
    its area is part of that dot's ring, not a dot (real dots lie 18 mm or more apart)."""
    drop = set()
    for i, a in enumerate(out):
        if not a[6]: continue
        for j, b in enumerate(out):
            if j != i and b[3] < 0.7 * a[3] and np.hypot(a[0] - b[0], a[1] - b[1]) < 3.2 * a[2] + 3: drop.add(j)
    if drop and stats is not None: stats["not_round"] = stats.get("not_round", 0) + len(drop)
    return [o for k, o in enumerate(out) if k not in drop]


def detect_marks(ir, rings=False, stats=None):
    """The platter's marker dots in an IR picture and whether each sits in a bright ring: ((N, 6) [cx, cy, r, area, w,
    h] in pixels, (N,) bool). cx/cy the intensity-weighted centre (the dark ring's level taken off first). Fixed floor
    first (found more dots in the gain sweep); when that finds fewer than HIT_MIN, a threshold at 0.7 of the brightest
    pixel (low gains peak far below 255). rings: see _dots_at (off: exactly the round-mat detection)."""
    ir = np.ascontiguousarray(ir, np.uint8)
    hist = np.bincount(ir.ravel(), minlength=256); p99 = int(np.searchsorted(np.cumsum(hist), 0.99 * ir.size))
    s1 = {} if stats is not None else None
    best = _dots_at(ir, max(DOT_THR_FLOOR, p99 // 2), rings, s1)
    if len(best) < HIT_MIN:
        s2 = {} if stats is not None else None
        alt = _dots_at(ir, max(DOT_THR_ALT[0], int(DOT_THR_ALT[1] * int(ir.max()))), rings, s2)
        if len(alt) > len(best): best, s1 = alt, s2
    if stats is not None:
        for k, v in (s1 or {}).items(): stats[k] = stats.get(k, 0) + v
    a = np.array(best, np.float64).reshape(-1, 7)
    return np.ascontiguousarray(a[:, :6]), a[:, 6] > 0.5


def detect_dots(ir, rings=False, stats=None):
    """The platter's marker dots: (N, 6) [cx, cy, r, area, w, h] in pixels (detect_marks without the ring flags)."""
    return detect_marks(ir, rings, stats)[0]


def dot_depth_points(d, cal, dots):
    """3D point (camera, mm) from the depth ON each dot: median of the back-projected pixels within 1.3 r + 1 px.
    NaN where fewer than 5 pixels have depth. Retro-reflective dots read high (about 0.8 mm)."""
    out = np.full((len(dots), 3), np.nan)
    for k, (cx, cy, r, a, w, h) in enumerate(dots):
        rad = 1.3 * r + 1
        x0, x1 = max(0, int(cx - rad)), min(d.shape[1], int(cx + rad) + 2)
        y0, y1 = max(0, int(cy - rad)), min(d.shape[0], int(cy + rad) + 2)
        dd = d[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1]
        m = (np.hypot(xx - cx, yy - cy) < rad) & (dd > 0)
        if m.sum() < 5: continue
        out[k] = np.median(backproject(xx[m].astype(np.float64), yy[m].astype(np.float64), dd[m] * 0.1, cal), 0)
    return out


def ring_depth_points(d, ir, cal, dots, max_pts=4000, rng=None):
    """Depth of the dark platter right around each dot (annulus 1.6 r + 2 .. 3 r + 4 px, IR darker than DARK_IR):
    platter surface by construction. (M, 3) camera frame, mm."""
    P = []
    for (cx, cy, r, a, w, h) in dots:
        r0, r1 = 1.6 * r + 2, 3 * r + 4
        x0, x1 = max(0, int(cx - r1)), min(d.shape[1], int(cx + r1) + 2)
        y0, y1 = max(0, int(cy - r1)), min(d.shape[0], int(cy + r1) + 2)
        dd = d[y0:y1, x0:x1]; ii = ir[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1]; rr = np.hypot(xx - cx, yy - cy)
        m = (rr > r0) & (rr < r1) & (dd > 0) & (ii < DARK_IR)
        if m.any(): P.append(backproject(xx[m].astype(np.float64), yy[m].astype(np.float64), dd[m] * 0.1, cal))
    if not P: return np.zeros((0, 3))
    P = np.concatenate(P)
    if len(P) > max_pts: P = P[(rng or np.random.default_rng(0)).choice(len(P), max_pts, replace=False)]
    return P


# ------------------------------------------------------------------------------------------------ planes
def ransac_plane(P, thr, iters, min_in=4, rng=None, score_max=5000):
    """Plane through the most of P (N x 3) within thr; ties go to the plane farther from the camera (the platter lies
    under the part). Returns (n, off, inlier mask) with n toward the camera (off > 0), refitted on its inliers."""
    P = np.asarray(P, float); N = len(P)
    if N < max(3, min_in): return None
    rng = rng or np.random.default_rng(1)
    S = P if N <= score_max else P[rng.choice(N, score_max, replace=False)]
    idx = rng.integers(0, N, (iters, 3)); a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nn = np.cross(b - a, c - a); L = np.linalg.norm(nn, axis=1); g = L > 1e-6
    if not g.any(): return None
    nn = nn[g] / L[g, None]; off = -(nn * a[g]).sum(1)
    flip = off < 0; nn[flip] *= -1; off[flip] *= -1
    cnt = (np.abs(S @ nn.T + off) < thr).sum(0)
    j = np.lexsort((off, cnt))[-1]
    if cnt[j] * N / len(S) < min_in: return None
    inl = np.abs(P @ nn[j] + off[j]) < thr
    if inl.sum() >= 3:
        n2, c2 = fit_plane(P[inl]); o2 = float(-n2 @ c2)
        if o2 < 0: n2, o2 = -n2, -o2
        if ang_deg(n2, nn[j]) < 10: n, o = n2, o2
        else: n, o = nn[j], float(off[j])
    else: n, o = nn[j], float(off[j])
    inl = np.abs(P @ n + o) < thr
    if inl.sum() < min_in: return None
    return n, float(o), inl


def plane_from_depth(d, cal, iters=300, tol=1.0, rng=None):
    """Support plane of one frame: the plane with the most depth points within tol that has next to nothing beyond it
    (seen from the camera): things stand on it, nothing lies under it. (n, off, count) or None."""
    rng = rng or np.random.default_rng(0)
    v, u = np.nonzero(d[::2, ::2]); v = v * 2; u = u * 2
    if len(u) < 2000: return None
    P = backproject(u.astype(np.float64), v.astype(np.float64), d[v, u] * 0.1, cal)
    S = P[rng.choice(len(P), min(20000, len(P)), replace=False)]; best = None
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]; nn = np.cross(b - a, c - a); l = np.linalg.norm(nn)
        if l < 1e-6: continue
        nn /= l
        if nn @ a > 0: nn = -nn
        off = -nn @ a; h = S @ nn + off
        inl = int((np.abs(h) < tol).sum())
        if (h < -3 * tol).sum() > 0.03 * len(S): continue
        if best is None or inl > best[2]: best = (nn, off, inl)
    if best is None: return None
    nn, off, _ = best; sel = P[np.abs(P @ nn + off) < tol]
    if len(sel) < 100: return None
    n2, c2 = fit_plane(sel)
    if n2 @ c2 > 0: n2 = -n2
    return n2, float(-n2 @ c2), len(sel)


def plane_candidates(dots, P3, ring, prior=None, rng=None):
    """Planes the dots may lie on, in the order tried: [(how, n, off)]: the prior (last known camera bed), up to three
    planes through the dots' own depth, one through the depth ring around them. Near-duplicates are dropped."""
    rng = rng or np.random.default_rng(1); out = []
    def add(how, n, off):
        for _, n2, o2 in out:
            if ang_deg(n, n2) < 0.5 and abs(o2 - off) < 0.5: return
        out.append((how, _unit(n), float(off)))
    if prior is not None:
        pn, po = (prior["n"], prior["off"]) if isinstance(prior, dict) else prior
        add("prior", np.asarray(pn, float), float(po))
    have = ~np.isnan(P3[:, 0]) if len(P3) else np.zeros(0, bool)
    if have.sum() >= RANSAC_DOT[2]:
        P = P3[have]; left = np.ones(len(P), bool)
        for k in range(3):
            if left.sum() < RANSAC_DOT[2]: break
            r = ransac_plane(P[left], RANSAC_DOT[0], RANSAC_DOT[1], RANSAC_DOT[2], rng)
            if r is None: break
            add("dots%d" % (k + 1), r[0], r[1]); left &= ~(np.abs(P @ r[0] + r[1]) < RANSAC_DOT[0])
    if len(ring) >= RANSAC_RING[2]:
        r = ransac_plane(ring, RANSAC_RING[0], RANSAC_RING[1], RANSAC_RING[2], rng)
        if r is not None and r[2].sum() >= RANSAC_RING[2]: add("ring", r[0], r[1])
    return out


def lift(dots, cal, n, off):
    """Each dot's IR centre ray meeting the plane (camera frame, mm). (N, 3), NaN where the ray misses (behind the
    camera or closer than 50 mm)."""
    out = np.full((len(dots), 3), np.nan)
    if not len(dots): return out
    ray = np.stack([(dots[:, 0] - cal[2]) / cal[0], (dots[:, 1] - cal[3]) / cal[1], np.ones(len(dots))], 1)
    den = ray @ np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"): t = -off / den
    ok = (np.abs(den) > 1e-6) & (t > 0) & (ray[:, 2] * t > 50)
    out[ok] = ray[ok] * t[ok, None]
    return out


# ------------------------------------------------------------------------------------------------ 2D match
_MAPTAB = {}


def _map_pairs(M2):
    key = (M2.shape, hash(M2.tobytes()))
    if key not in _MAPTAB:
        if len(_MAPTAB) > 16: _MAPTAB.clear()
        a, b = np.nonzero(~np.eye(len(M2), dtype=bool))
        dM = np.linalg.norm(M2[a] - M2[b], axis=1); o = np.argsort(dM)
        _MAPTAB[key] = (np.stack([a[o], b[o]], 1), dM[o])
    return _MAPTAB[key]


def _kabsch2(A, B):
    ca, cb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb)); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: return None
    return R, cb - R @ ca


def _rot2(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])


def register_2d(F2, M2, tol_pair=PAIR_TOL, tol_in=INLIER_TOL, max_pairs=PAIR_MAX, exhaustive=False, rng=None, init=None,
                flags=None):
    """Rigid 2D placement (rotation + translation, never a mirror) of frame dots F2 onto map dots M2: map = R f + t.
    Hypotheses from frame dot pairs 15-140 mm apart matched to map pairs of the same length; the one that puts the most
    frame dots within tol_in of a map dot wins and is refined (Kabsch on its inliers, 3 times). runner_up = the best
    hypothesis that places the frame dots somewhere else (more than 5 deg or 5 mm away at their centroid): a pattern
    that matches two ways is not a match. flags: (frame, map) bool per dot (a ringed board): a frame dot counts only on a
    map dot with the same flag, so a pose turned 180 deg on a checkered grid of ringed dots scores nothing. Returns Reg
    or None."""
    F2 = np.asarray(F2, float); M2 = np.asarray(M2, float); nf = len(F2)
    if nf < 3 or len(M2) < 3: return None
    ff = mf = None
    if flags is not None and flags[0] is not None and flags[1] is not None and len(flags[0]) == nf and len(flags[1]) == len(M2):
        ff, mf = np.asarray(flags[0], bool), np.asarray(flags[1], bool)
    def near(X):            # distance to the nearest map dot (a flagged frame dot: of the same flag), and which one
        D = np.linalg.norm(X[:, None] - M2[None], axis=2)
        if ff is not None: D = np.where(ff[:, None] == mf[None], D, 1e9)
        return D.min(1), D.argmin(1)
    rng = rng or np.random.default_rng(0)
    hyp_c, hyp_a, hyp_t = [], [], []
    best = None
    if init is not None:
        R0, t0 = init; X = F2 @ R0.T + t0
        cnt = int((near(X)[0] < tol_in).sum())
        best = (cnt, float(np.arctan2(R0[1, 0], R0[0, 0])), np.asarray(t0, float))
    iu = np.array(np.triu_indices(nf, 1)).T
    dF = np.linalg.norm(F2[iu[:, 0]] - F2[iu[:, 1]], axis=1)
    keep = (dF > PAIR_RANGE[0]) & (dF < PAIR_RANGE[1]); iu, dF = iu[keep], dF[keep]
    if not exhaustive and len(iu) > max_pairs:
        s = rng.choice(len(iu), max_pairs, replace=False); iu, dF = iu[s], dF[s]
    am, dMs = _map_pairs(M2)
    exit_at = max(HIT_MIN, int(0.8 * nf))
    mx, my = M2[:, 0], M2[:, 1]
    for k in range(len(iu)):
        i, j = iu[k]
        lo, hi = np.searchsorted(dMs, [dF[k] - tol_pair, dF[k] + tol_pair])
        if hi <= lo: continue
        ab = am[lo:hi]
        vf = F2[j] - F2[i]; vm = M2[ab[:, 1]] - M2[ab[:, 0]]
        th = np.arctan2(vm[:, 1], vm[:, 0]) - np.arctan2(vf[1], vf[0]); c, s_ = np.cos(th), np.sin(th)
        tx = M2[ab[:, 0], 0] - (c * F2[i, 0] - s_ * F2[i, 1]); ty = M2[ab[:, 0], 1] - (s_ * F2[i, 0] + c * F2[i, 1])
        X = c[:, None] * F2[None, :, 0] - s_[:, None] * F2[None, :, 1] + tx[:, None]
        Y = s_[:, None] * F2[None, :, 0] + c[:, None] * F2[None, :, 1] + ty[:, None]
        dd = (X[:, :, None] - mx) ** 2 + (Y[:, :, None] - my) ** 2
        if ff is not None: dd = np.where(ff[None, :, None] == mf[None, None, :], dd, 1e18)
        cnt = (dd.min(2) < tol_in ** 2).sum(1)
        g = cnt >= 3
        if g.any(): hyp_c.append(cnt[g]); hyp_a.append(th[g]); hyp_t.append(np.stack([tx[g], ty[g]], 1))
        h = int(cnt.argmax())
        if best is None or cnt[h] > best[0]: best = (int(cnt[h]), float(th[h]), np.array([tx[h], ty[h]]))
        if not exhaustive and k + 1 >= PAIR_MIN_BEFORE_EXIT and best[0] >= exit_at: break
    if best is None or best[0] < 3: return None
    R, t = _rot2(best[1]), best[2]
    for _ in range(3):
        dmin, nn = near(F2 @ R.T + t); inl = dmin < tol_in
        if inl.sum() < 3: break
        r = _kabsch2(F2[inl], M2[nn[inl]])
        if r is None: break
        R, t = r
    dmin, nn = near(F2 @ R.T + t); inl = dmin < tol_in
    rms = float(np.sqrt((dmin[inl] ** 2).mean())) if inl.any() else 99.0
    ru = 0
    if hyp_c:
        C = np.concatenate(hyp_c); A = np.concatenate(hyp_a); T = np.concatenate(hyp_t)
        qc = F2.mean(0); ang = np.arctan2(R[1, 0], R[0, 0])
        pc = np.stack([np.cos(A) * qc[0] - np.sin(A) * qc[1], np.sin(A) * qc[0] + np.cos(A) * qc[1]], 1) + T
        far = (np.linalg.norm(pc - (R @ qc + t), axis=1) > RUNNER_UP_MM) | \
              (np.degrees(np.abs((A - ang + np.pi) % (2 * np.pi) - np.pi)) > RUNNER_UP_DEG)
        if far.any(): ru = int(C[far].max())
    return Reg(R, t, int(inl.sum()), rms, ru, inl, nn)


def is_hit(reg, min_in=HIT_MIN):
    return reg is not None and reg.inliers >= min_in and reg.inliers >= HIT_RATIO * reg.runner_up and reg.rms <= HIT_RMS


def bed_frame(n, off, R, t):
    """The bed frame (4x4, bed -> the frame n/off live in) from a plane and the 2D placement map = R f + t, where f are
    coordinates in basis(n). Returns (B, centre, x_axis)."""
    n = _unit(n); e1, e2 = basis(n)
    f0 = -R.T @ t                                       # the map's origin (spin centre) in plane coordinates
    c = f0[0] * e1 + f0[1] * e2 - off * n
    xf = R.T @ np.array([1.0, 0.0]); x3 = _unit(xf[0] * e1 + xf[1] * e2)
    B = np.eye(4); B[:3, 0] = x3; B[:3, 1] = np.cross(n, x3); B[:3, 2] = n; B[:3, 3] = c
    return B, c, x3


# ------------------------------------------------------------------------------------------------ one frame
class Hit:
    """The bed found in one frame (camera frame, mm)."""
    __slots__ = ("n", "off", "centre", "x_axis", "B_cam", "inliers", "runner_up", "rms", "how", "height_from",
                 "surface_shift", "dots_px", "matched", "beneath", "ms", "n_dots")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))

    def set_plane(self, n, off):
        """Move to plane (n, off) keeping the angle: the centre drops onto it along the new normal."""
        n = _unit(n); c = self.centre - (n @ self.centre + off) * n
        x = _unit(self.x_axis - (self.x_axis @ n) * n)
        self.n, self.off, self.centre, self.x_axis = n, float(off), c, x
        B = np.eye(4); B[:3, 0] = x; B[:3, 1] = np.cross(n, x); B[:3, 2] = n; B[:3, 3] = c; self.B_cam = B

    def camera(self):
        return dict(n=self.n.copy(), off=float(self.off), centre=self.centre.copy(), x_axis=self.x_axis.copy())


def _dot_mask(dots, shape, grow=1.3, pad=2):
    m = np.zeros(shape, np.uint8)
    for (cx, cy, r, a, w, h) in dots: cv2.circle(m, (int(round(cx)), int(round(cy))), int(grow * r + pad), 1, -1)
    return m.astype(bool)


def _frame_pts(d, stride=2):
    v, u = np.nonzero(d[::stride, ::stride]); v = v * stride; u = u * stride
    return u, v


def _lowest_layer(h):
    """Height (mm over the candidate plane) of the platter's peak in the dark in-disc heights h between -LOW_MM and
    +SURFACE_BAND0, or None. The platter is the lowest surface there: nothing lies under it, while a dark face of the part
    near it is a layer above, often a far denser one (a sprayed part at gain 1: 10x the platter's pixels). So: the lowest
    dense 2 mm layer (LAYER_MIN pixels, 3x the 2 mm under it), then up 1 mm windows while they get denser (at most
    LAYER_UP): depth noise spreads the platter downward and the lowest dense layer sat under its peak (1.2 mm at 1 mm
    noise, 2.4 at 1.5), while the climb stops at the platter's own peak, before the valley under a face."""
    c = np.histogram(h, bins=np.arange(-LOW_MM, SURFACE_BAND0 + 0.01, 0.5))[0]
    s = np.convolve(c, np.ones(4, int), "valid")                  # 2 mm layers, one every 0.5 mm
    for i in range(len(s)):
        under = s[i - 4] if i >= 4 else c[:i].sum()                # the 2 mm right under it
        if s[i] >= LAYER_MIN and s[i] >= 3 * under:
            c2 = c[:-1] + c[1:]                                    # 1 mm windows, one every 0.5 mm
            k = k0 = i + int(np.argmax(c2[i:i + 3]))               # the densest 1 mm inside the lowest dense 2 mm
            while k + 1 < len(c2) and k + 1 - k0 <= LAYER_UP / 0.5 and c2[k + 1] >= c2[k]: k += 1
            return -LOW_MM + 0.5 * k + 0.5
    return None


def refine_height(d, cal, hit, bed, ir=None, dots=None, P3=None):
    """Put the plane on the platter top. The dot plane reads high (the dots do), the ring plane is the platter itself.
    Best: the platter's dark depth pixels inside the disc (not on a dot): the platter's layer from 6 mm under the
    candidate plane to 3 mm over it (_lowest_layer), at least SURFACE_MIN pixels within +-SURFACE_BAND of it -> refit
    (twice, same band), kept if the normal stays within 1 deg and the refit is not more than SURFACE_VS_DOTS over the
    matched dots' own plane (their depth minus the learned lift). A dark face of the part 1.5-3.5 mm over the platter
    pulled the old +-3 mm refit up to it (+3.8 mm, and V1 cannot see that); the dots can read high themselves (a face next
    to a dot lifts its depth too), so only a refit OVER them is refused. Otherwise a plane from dot depth is moved down by the
    learned lift; a prior plane is moved to the dots' height when they disagree by more than PRIOR_HEIGHT_MM, a ring plane
    when it lies more than SURFACE_VS_DOTS over them. A prior with fewer than 4 matched dots giving depth is only the
    prior: height_from = "prior-only" (SideBed does not count such a hit toward accepting a plane)."""
    lift_mm = float(bed.lift_mm) if bed is not None else 0.0
    Pd = None                                           # matched dots with depth: the platter + lift_mm, by construction
    if P3 is not None and len(P3):
        m = ~np.isnan(P3[:, 0])
        if hit.matched is not None and len(hit.matched) == len(P3): m &= hit.matched
        if m.sum() >= 4: Pd = P3[m]
    def dots_off(n): return None if Pd is None else float(np.median(lift_mm - Pd @ n))
    u, v = _frame_pts(d)
    if len(u):
        P = backproject(u.astype(np.float64), v.astype(np.float64), d[v, u] * 0.1, cal)
        keep = np.ones(len(u), bool)
        if ir is not None: keep &= ir[v, u] < DARK_IR
        if dots is not None and len(dots): keep &= ~_dot_mask(dots, d.shape)[v, u]
        P = P[keep]
        n, off, c = hit.n, hit.off, hit.centre
        h = P @ n + off; inner = _in_outline(P, hit, bed, -5.0)            # 5 mm inside the rim
        z = _lowest_layer(h[inner])
        sel = (np.abs(h - (0.0 if z is None else z)) < SURFACE_BAND) & inner
        if z is not None and sel.sum() >= SURFACE_MIN:
            n2, o2 = n, off
            for _ in range(2):
                nn, cc = fit_plane(P[sel])
                if nn @ n < 0: nn = -nn
                n2, o2 = nn, float(-nn @ cc)
                sel = (np.abs(P @ n2 + o2) < SURFACE_BAND) & inner
                if sel.sum() < SURFACE_MIN // 2: break
            if ang_deg(n2, n) <= SURFACE_DEG and sel.sum() >= SURFACE_MIN // 2:
                od = dots_off(n2)
                if od is None or o2 >= od - SURFACE_VS_DOTS:        # not over the dots' own plane (under it is fine)
                    shift = -float(n2 @ c + o2)
                    hit.set_plane(n2, o2); hit.height_from = "surface"; hit.surface_shift = round(shift, 3)
                    return hit
    if hit.how.startswith("dots"):
        hit.set_plane(hit.n, hit.off + lift_mm); hit.height_from = "lift"; hit.surface_shift = -lift_mm
        return hit
    od = dots_off(hit.n)
    if hit.how == "prior" and od is None: hit.height_from = "prior-only"
    elif od is not None and (abs(od - hit.off) > PRIOR_HEIGHT_MM if hit.how == "prior" else hit.off < od - SURFACE_VS_DOTS):
        hit.surface_shift = round(hit.off - od, 3); hit.set_plane(hit.n, od); hit.height_from = "lift"
    else: hit.height_from = hit.how                   # "prior" (the dots agree with it) or "ring"
    return hit


def _in_outline(P, hit, bed, pad):
    """Which camera-frame points P lie over the bed's outline grown by pad mm (negative: shrunk), seen along the hit's
    normal. bed: a BedModel, an Outline or a radius (None: 95 mm). A circle about the spin centre uses the old maths."""
    o = bed if isinstance(bed, Outline) else Outline("circle", (0, 0), 95.0 if bed is None else bed) \
        if not hasattr(bed, "outline") else bed.outline
    n, c = hit.n, hit.centre; q = P - c
    if o.centred_circle:
        return np.linalg.norm(q - np.outer(q @ n, n), axis=1) < o.radius + pad
    return PlacedOutline(o, hit.x_axis).sdf(P, n, c) < pad


def v1_nothing_beneath(d, cal, hit, radius):
    """Share of the depth pixels inside the disc (radius + 5 mm, seen along the normal) that lie more than 4 mm under
    the plane. The platter is opaque: a real platter has nothing under it. A flat face of the part does. radius: a
    number (a circle about the centre), an Outline or the BedModel (its outline)."""
    u, v = _frame_pts(d, 3)
    if not len(u): return 0.0
    P = backproject(u.astype(np.float64), v.astype(np.float64), d[v, u] * 0.1, cal)
    h = P @ hit.n + hit.off
    inside = _in_outline(P, hit, radius, 5.0)
    if inside.sum() < 50: return 0.0
    return float((h[inside] < -V1_BELOW_MM).mean())


def find_bed(d, ir, cal, bed, prior=None, budget_s=0.5, rng=None, dots=None):
    """The bed in one frame, or None. Tries the plane candidates in order (prior, dot planes, ring plane) until
    budget_s has passed; the most inliers wins, then the lowest rms. Only a HIT counts: >= 7 dots matched, at least
    twice the runner-up, rms <= 1.4 mm. Returns a Hit (camera frame) with its height set (refine_height). dots:
    detect_dots's dots, or detect_marks' (dots, ringed) for a ringed board. A bed with no dot map (kind "plane") is
    never looked for here."""
    t0 = time.monotonic(); rng = rng or np.random.default_rng(0)
    if not has_dots(bed): return None             # a plane bed (no dot map): its letters and marks are never dots
    rings = bed.ringed is not None
    if dots is None: dl, fring = detect_marks(ir, rings)
    elif isinstance(dots, tuple): dl, fring = dots                  # detect_marks' own (dots, ringed)
    else: dl, fring = dots, None
    rings = rings and fring is not None
    if len(dl) < HIT_MIN: return None
    P3 = dot_depth_points(d, cal, dl)
    ring = ring_depth_points(d, ir, cal, dl, rng=rng)
    best = None; M2 = bed.map2d
    for how, n, off in plane_candidates(dl, P3, ring, prior, rng):
        if time.monotonic() - t0 > budget_s: break
        h = P3 @ n + off
        use = np.isnan(h) | (np.abs(h) < ON_PLANE)
        L = lift(dl[use], cal, n, off); ok = ~np.isnan(L[:, 0])
        if ok.sum() < HIT_MIN: continue
        e1, e2 = basis(n); F2 = np.stack([L[ok] @ e1, L[ok] @ e2], 1)
        reg = register_2d(F2, M2, rng=rng, flags=None if not rings else (fring[use][ok], bed.ringed))
        if not is_hit(reg): continue
        if best is None or (reg.inliers, -reg.rms) > (best[0].inliers, -best[0].rms):
            idx = np.nonzero(use)[0][ok]; best = (reg, how, n, off, idx)
    if best is None: return None
    reg, how, n, off, idx = best
    B, c, x3 = bed_frame(n, off, reg.R, reg.t)
    matched = np.zeros(len(dl), bool); matched[idx[reg.mask]] = True
    hit = Hit(n=_unit(n), off=float(off), centre=c, x_axis=x3, B_cam=B, inliers=reg.inliers, runner_up=reg.runner_up,
              rms=reg.rms, how=how, height_from=None, surface_shift=None, dots_px=dl[:, :3].copy(), matched=matched,
              beneath=None, n_dots=len(dl))
    refine_height(d, cal, hit, bed, ir, dl, P3)
    hit.beneath = v1_nothing_beneath(d, cal, hit, bed)
    hit.ms = round(1000 * (time.monotonic() - t0), 1)
    return hit


def agree(a, b):
    """Two camera beds are the same platter: normals within 1.5 deg, each centre within 2 mm of the other's plane,
    centres within 4 mm. (Within a pass the camera bed is steady: the table spins about its own axis.)"""
    an, ao, ac = (a.n, a.off, a.centre) if isinstance(a, Hit) else (np.asarray(a["n"]), a["off"], np.asarray(a["centre"]))
    bn, bo, bc = (b.n, b.off, b.centre) if isinstance(b, Hit) else (np.asarray(b["n"]), b["off"], np.asarray(b["centre"]))
    return (ang_deg(an, bn) <= AGREE_DEG and abs(bn @ ac + bo) <= AGREE_MM and abs(an @ bc + ao) <= AGREE_MM
            and np.linalg.norm(ac - bc) <= AGREE_CENTRE)


def mean_hits(hits):
    """One camera bed from agreeing hits: mean normal, mean centre; x_axis of the newest hit (the angle turns)."""
    n = _unit(np.sum([h.n for h in hits], 0)); c = np.mean([h.centre for h in hits], 0)
    return dict(n=n, off=float(-n @ c), centre=c, x_axis=hits[-1].x_axis.copy())


def to_scan(hit, T):
    """A camera bed into the scan frame with T (OpenCV camera -> scan, mm): (n, off, centre, x_axis, B_scan)."""
    if isinstance(hit, Hit): n, c, x, B = hit.n, hit.centre, hit.x_axis, hit.B_cam
    else:
        n, c, x = np.asarray(hit["n"]), np.asarray(hit["centre"]), np.asarray(hit["x_axis"])
        B = np.eye(4); B[:3, 0] = x; B[:3, 1] = np.cross(n, x); B[:3, 2] = n; B[:3, 3] = c
    R, t = T[:3, :3], T[:3, 3]; ns = R @ n; cs = R @ c + t
    return dict(n=ns, off=float(-ns @ cs), centre=cs, x_axis=R @ x, B_scan=T @ B)


def plane_to_camera(n, off, centre, T):
    """A scan-frame plane and centre as seen from a camera at pose T (camera -> scan): (n_c, off_c, centre_c)."""
    nc, oc = to_camera(np.asarray(n, float), float(off), T)
    return nc, oc, T[:3, :3].T @ (np.asarray(centre, float) - T[:3, 3])


# ------------------------------------------------------------------------------------------------ the bed map
def config_dir():
    """Where the bed of this turntable lives: next to PointYoink's config."""
    c = os.environ.get("POINTYOINK_CONFIG")
    return os.path.dirname(os.path.abspath(c)) if c else os.path.expanduser("~/.config/pointyoink")


class BedModel:
    """A bed: dot map (bed frame), outline, dot size, the dots' lift, and where the camera saw it. As a bed profile it
    also has a name and a kind: "dots" (a dot map; board "ringed" when its dots sit in broken rings, like a calibration
    board, with a flag per dot) or "plane" (a flat surface with no dots, found from depth alone). quality: what the
    learning measured (dots kept, shapes left out, the map check). radius_mm is the outline's reach (a circle's
    radius) for code that wants a disc."""
    def __init__(self, **kw):
        self.version = 1; self.id = kw.get("id") or time.strftime("bed-%Y%m%d-%H%M%S")
        self.learned = kw["learned"] if kw.get("learned") is not None else time.strftime("%Y-%m-%dT%H:%M:%S")   # "": a preset not learned yet
        self.source = kw.get("from", "learn"); self.frames = int(kw.get("frames", 0)); self.turned_deg = kw.get("turned_deg")
        self.ir_gain = kw.get("ir_gain") or []
        self.dots = np.asarray(kw.get("dots", np.zeros((0, 4))), float).reshape(-1, 4)
        self.outline = kw["outline"] if isinstance(kw.get("outline"), Outline) else \
            Outline.from_json(kw.get("outline"), float(kw.get("radius_mm", 97.0)))
        self.dot_diam_mm = kw.get("dot_diam_mm")
        self.rms_mm = kw.get("rms_mm"); self.lift_mm = float(kw.get("lift_mm", 0.0) or 0.0)
        self.self_ambiguity = kw.get("self_ambiguity"); self.calib_md5 = kw.get("calib_md5")
        self.provisional = bool(kw.get("provisional", False))    # learned from one side's own dots, kept in that project only
        self.name = kw.get("name"); self.kind = "plane" if kw.get("kind") == "plane" else "dots"
        self.rim=kw.get("rim");self.rim_markers=kw.get("rim_markers")
        self.board = kw.get("board"); self.quality = kw.get("quality") or {}
        self.nominal = kw.get("nominal")        # the outline as measured by hand (a preset's), JSON; Learn bed checks against it
        rg = kw.get("ringed")
        self.ringed = None if rg is None else np.asarray(rg, bool).reshape(-1)
        if self.ringed is not None and len(self.ringed) != len(self.dots): self.ringed = None
        cam = kw.get("camera")
        self.camera = None if not cam else dict(n=np.asarray(cam["n"], float), off=float(cam["off"]),
                                                 centre=np.asarray(cam["centre"], float), x_axis=np.asarray(cam["x_axis"], float))
        self.map2d = np.ascontiguousarray(self.dots[:, :2])

    @property
    def radius_mm(self): return self.outline.reach

    @radius_mm.setter
    def radius_mm(self, r): self.outline = Outline("circle", (0.0, 0.0), float(r))

    @classmethod
    def from_json(cls, j): return cls(**j)

    @classmethod
    def load(cls, path):
        try: return cls.from_json(json.load(open(path)))
        except Exception: return None

    def to_json(self):
        cam = None if self.camera is None else {k: (_jl(v) if k != "off" else round(float(v), 4)) for k, v in self.camera.items()}
        j = {"version": 1, "id": self.id, "learned": self.learned, "from": self.source, "frames": self.frames,
             "turned_deg": None if self.turned_deg is None else round(float(self.turned_deg), 1), "ir_gain": self.ir_gain,
             "dots": _jl(self.dots, 3), "radius_mm": round(self.radius_mm, 2),
             "dot_diam_mm": None if self.dot_diam_mm is None else round(float(self.dot_diam_mm), 2),
             "rms_mm": None if self.rms_mm is None else round(float(self.rms_mm), 3), "lift_mm": round(self.lift_mm, 3),
             "self_ambiguity": None if self.self_ambiguity is None else round(float(self.self_ambiguity), 3),
             "calib_md5": self.calib_md5, "camera": cam, "provisional": self.provisional,
             "outline": self.outline.to_json(), "kind": self.kind}
        if self.rim:j["rim"]=self.rim
        if self.rim_markers:j["rim_markers"]=self.rim_markers
        if self.name: j["name"] = self.name
        if self.board: j["board"] = self.board
        if self.ringed is not None: j["ringed"] = [int(x) for x in self.ringed]
        if self.quality: j["quality"] = self.quality
        if self.nominal: j["nominal"] = self.nominal
        return j

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.%d" % os.getpid(); json.dump(self.to_json(), open(tmp, "w"), indent=1); os.replace(tmp, path)

    def copy_to(self, project):
        """The bed a project was scanned with goes into the project, so builds and replays repeat exactly."""
        p = os.path.join(project, "bed.json")
        if not os.path.exists(p): self.save(p)
        return p

    @staticmethod
    def config_path(): return os.path.join(config_dir(), "bed.json")

    def __len__(self): return len(self.dots)


def has_dots(bed):
    """Is there a dot map to look for? Not for a plane bed (the bare turntable: letters and marks, no dots)."""
    return bed is not None and getattr(bed, "kind", "dots") != "plane" and len(bed.map2d) >= 3


# ------------------------------------------------------------------------------------------------ bed profiles
# Several named beds live next to bed.json in the config dir: bed-<name>.json each (a BedModel with a name and a kind),
# bed_pick.json {"name": ...} the one picked last. bed.json stays the last picked dots bed (older code reads it); a
# project keeps the bed it was scanned with (its bed.json) and which profile that was (its bed_profile.json).
PROFILE_PREFIX = "bed-"
PICK_FILE = "bed_pick.json"
PROJECT_PROFILE = "bed_profile.json"
MIGRATED_NAME = "Round dot mat"


def profile_slug(name):
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name).strip())
    while "--" in s: s = s.replace("--", "-")
    return s.strip("-") or "bed"


def profile_path(name): return os.path.join(config_dir(), PROFILE_PREFIX + profile_slug(name) + ".json")


def _write_json(path, j):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid(); json.dump(j, open(tmp, "w"), indent=1); os.replace(tmp, path)


def list_profiles():
    """The bed profiles in the config dir, [(name, path)] sorted by name (migrate_profiles first)."""
    out = []
    d = config_dir()
    try: names = sorted(os.listdir(d))
    except OSError: return out
    for f in names:
        if not (f.startswith(PROFILE_PREFIX) and f.endswith(".json")) or ".tmp." in f: continue
        p = os.path.join(d, f); b = BedModel.load(p)
        if b is not None and b.name: out.append((b.name, p))
    return sorted(out, key=lambda x: x[0].lower())


def load_profile(name):
    """The profile with this name, or None."""
    for n, p in list_profiles():
        if n == name: return BedModel.load(p)
    return None


def save_profile(bed, name=None):
    """Store a bed as a named profile (bed-<name>.json; a profile of the same name is replaced). Returns the path."""
    if name: bed.name = name
    if not bed.name: bed.name = "Bed " + str(bed.learned)[:10]
    p = profile_path(bed.name); bed.save(p); return p


def picked_profile():
    """The name of the profile picked last (bed_pick.json), or None."""
    try: return json.load(open(os.path.join(config_dir(), PICK_FILE))).get("name")
    except Exception: return None


def pick_profile(name):
    _write_json(os.path.join(config_dir(), PICK_FILE), {"name": name})


# The user's three beds, measured by hand: usable (their outline) before they are learned. (name, kind, board, outline)
PRESETS = ((MIGRATED_NAME, "dots", None, Outline("circle", (0.0, 0.0), 99.5)),             # 199 mm across, ~32 dots
           ("Calibration board", "dots", "ringed", Outline("rect", (0.0, 0.0), width=410.0, depth=259.0)),   # ringed calibration board
           ("Bare turntable", "plane", None, Outline("circle", (0.0, 0.0), 100.0)))       # ridges, letters, no dots
NOMINAL_MM = 5.0               # a learned outline this far off the bed's measured size: Learn bed warns


def migrate_profiles(log=None):
    """The first time (no profile yet): the presets are written, and the single learned bed (bed.json) becomes the
    profile "Round dot mat" (same id, dots, date and outline; its measured size kept as "nominal") and the picked one.
    bed.json itself is left as it is. Returns the paths written."""
    if list_profiles(): return []
    out = []; b = BedModel.load(BedModel.config_path())
    if b is not None and (b.provisional or len(b) < HIT_MIN): b = None
    for name, kind, board, o in PRESETS:
        if b is not None and name == MIGRATED_NAME:
            b.name = name; b.kind = "dots"; b.nominal = o.to_json()
            if not b.quality: b.quality = {"dots": len(b), "rms_mm": b.rms_mm, "self_ambiguity": b.self_ambiguity, "frames": b.frames}
            if b.outline.centred_circle and abs(b.outline.radius - o.radius) > NOMINAL_MM:
                # the learner's edge (the dark surface's 98th percentile) ran long: 109 mm on the 2026-09-27 bed, where the
                # frames show depth under the plane from 95-100 mm on and the mat measures 199 mm across. Only the rim
                # (the tint, the cut, the in-disc checks) uses it: the dots' placement did not change on replayed sides
                b.quality["learned_radius_mm"] = round(b.outline.radius, 2); b.outline = Outline("circle", (0.0, 0.0), o.radius)
            out.append(save_profile(b))
            if picked_profile() is None: pick_profile(name)
            if log: log("bed: the learned bed is now the bed profile \"%s\" (%s)" % (name, os.path.basename(out[-1])))
            continue
        pre = plane_profile(name, o) if kind == "plane" else BedModel(name=name, kind=kind, board=board, outline=o, **{"from": "preset"})
        pre.nominal = o.to_json(); pre.learned = ""; out.append(save_profile(pre))
    return out


def outline_check(bed):
    """Learn bed's check of the learned outline against the bed's measured one (nominal): a warning, or None."""
    if not bed.nominal: return None
    nom = Outline.from_json(bed.nominal); o = bed.outline
    if nom.shape != o.shape:
        return "it came out %s, but this bed is %s" % ("a rectangle" if o.shape == "rect" else "round",
                                                     "a rectangle" if nom.shape == "rect" else "round")
    if o.shape == "circle":
        if abs(o.radius - nom.radius) > NOMINAL_MM:
            return "it measured %.0f mm across, this bed is %.0f mm" % (2 * o.radius, 2 * nom.radius)
        return None
    a, b = sorted((o.width, o.depth)), sorted((nom.width, nom.depth))
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > NOMINAL_MM:
        return "it measured %.0f x %.0f mm, this bed is %.0f x %.0f mm" % (a[1], a[0], b[1], b[0])
    return None


def plane_profile(name, outline=None):
    """A bed with no dots (kind "plane"): only its flat surface is used, found from depth (the bare turntable). The
    outline defaults to a 100 mm circle about the spin centre."""
    return BedModel(name=name, kind="plane", outline=outline or Outline("circle", (0.0, 0.0), 100.0), **{"from": "plane"})


def read_project_profile(project):
    """{"name", "id", "kind"} of the bed profile a project was scanned with, or None."""
    try: return json.load(open(os.path.join(project, PROJECT_PROFILE)))
    except Exception: return None


def write_project_profile(project, bed):
    _write_json(os.path.join(project, PROJECT_PROFILE), {"name": bed.name, "id": bed.id, "kind": bed.kind})


# ------------------------------------------------------------------------------------------------ plane beds
def find_plane_bed(d, cal, rng=None):
    """A plane bed (no dots) in one frame: the support plane (plane_from_depth: most points, next to nothing under it),
    then put on its LOWEST dense layer (_lowest_layer, the same climb refine_height uses) and refit there, so raised
    ridges or letters on the platter neither lift nor tilt it. (n, off, points) camera frame, or None."""
    r = plane_from_depth(d, cal, rng=rng)
    if r is None: return None
    n, off, _ = r
    u, v = _frame_pts(d)
    P = backproject(u.astype(np.float64), v.astype(np.float64), d[v, u] * 0.1, cal)
    z = _lowest_layer(P @ n + off)
    if z is not None: off = off - z
    for _ in range(2):
        sel = np.abs(P @ n + off) < SURFACE_BAND
        if sel.sum() < SURFACE_MIN: break
        nn, cc = fit_plane(P[sel])
        if nn @ n < 0: nn = -nn
        if ang_deg(nn, n) > SURFACE_DEG * 3: break
        n, off = nn, float(-nn @ cc)
    return _unit(n), float(off), int((np.abs(P @ n + off) < SURFACE_BAND).sum())


def match_relief(d, cal, bed, plane):
    """TODO (the bare turntable, kind "plane"): its raised radial ridges are uneven, so the relief over the plane gives
    the table's angle too, from depth alone (no IR blobs: the printed letters must never count). A matcher plugs in
    here: heights over `plane` (find_plane_bed) sampled on rings about the centre, matched to a learned relief profile
    -> a Hit like find_bed's. Until then the plane alone is used (Auto remove base's path)."""
    return None


def compute_self_ambiguity(M2, ringed=None):
    """Best match of the map onto itself other than doing nothing, as a share of the dots (> 0.6: repeating pattern).
    ringed: the dots' ring flags (a ringed board), which matching also uses."""
    if len(M2) < 4: return 1.0
    r = register_2d(M2, M2, exhaustive=True, init=(np.eye(2), np.zeros(2)), flags=None if ringed is None else (ringed, ringed))
    return 0.0 if r is None else r.runner_up / len(M2)


def _cluster2d(Q, rad, min_n, w=None):
    """Greedy clusters (densest first), then points reassigned to the nearest centre within rad: (centres, counts, rms)."""
    Q = np.asarray(Q, float)
    if not len(Q): return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
    w = np.ones(len(Q)) if w is None else np.asarray(w, float)
    D = np.linalg.norm(Q[:, None] - Q[None], axis=2) if len(Q) <= 2000 else None      # <= 32 MB
    dens = (D < rad).astype(float) @ w if D is not None else w
    used = np.zeros(len(Q), bool); cents = []
    for i in np.argsort(-dens):
        if used[i]: continue
        dist = D[i] if D is not None else np.linalg.norm(Q - Q[i], axis=1)
        m = (~used) & (dist < rad)
        used |= m
        if w[m].sum() >= min_n: cents.append((w[m, None] * Q[m]).sum(0) / w[m].sum())
    if not cents: return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
    C = np.array(cents)
    for _ in range(2):
        Dc = np.linalg.norm(Q[:, None] - C[None], axis=2); a = Dc.argmin(1); ok = Dc.min(1) < rad
        newC, cnt, rms = [], [], []
        for k in range(len(C)):
            m = ok & (a == k)
            if w[m].sum() < min_n: continue
            ck = (w[m, None] * Q[m]).sum(0) / w[m].sum(); newC.append(ck); cnt.append(w[m].sum())
            rms.append(np.sqrt((w[m] * ((Q[m] - ck) ** 2).sum(1)).sum() / w[m].sum()))
        if not newC: return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
        C = np.array(newC)
    return C, np.array(cnt), np.array(rms)


def _merge_close(C, cnt, rms, rad=8.0, keep_heavier=False):
    """Dots closer than rad are one dot (real dots are >= 18 mm apart): count-weighted, or (keep_heavier) at the better
    seen position, when the two are the same dot placed differently by different sides (averaging lands between them)."""
    C, cnt, rms = list(C), list(cnt), list(rms)
    while True:
        if len(C) < 2: break
        A = np.array(C); D = np.linalg.norm(A[:, None] - A[None], axis=2) + np.eye(len(A)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        if D[i, j] >= rad: break
        if keep_heavier and D[i, j] > 2 * INLIER_TOL:
            if cnt[j] > cnt[i]: C[i], rms[i] = C[j], rms[j]
            cnt[i] = cnt[i] + cnt[j]
        else:
            wsum = cnt[i] + cnt[j]; C[i] = (cnt[i] * A[i] + cnt[j] * A[j]) / wsum
            rms[i] = float(np.hypot(max(rms[i], rms[j]), D[i, j] / 2)); cnt[i] = wsum
        del C[j], cnt[j], rms[j]
    return np.array(C).reshape(-1, 2), np.array(cnt), np.array(rms)


def _orient_map(M, cnt):
    """Rotate the map so the most-seen dot lies on +x (the bed frame's x)."""
    k = int(np.argmax(cnt)); a = np.arctan2(M[k, 1], M[k, 0]); return M @ _rot2(-a).T, a


def _side_io(side):
    from fuse import read_calib, read_global_poses
    cal = read_calib(os.path.join(side, "param", "Pl.bin"))
    gp = read_global_poses(os.path.join(side, "cache", "pointyoink_register_pose.pose"))
    return cal, gp


def read_frame(side, key):
    p = os.path.join(side, "cache", "frame_%03d_%04d" % tuple(key))
    d = np.frombuffer(open(p + ".dph", "rb").read(), np.uint16).reshape(H, W)
    ir = np.frombuffer(open(p + ".irl", "rb").read(), np.uint8).reshape(H, W) if os.path.exists(p + ".irl") else None
    return d, ir


def learn_from_side(side, every=3, plane=None, max_frames=None):
    """A dot map from a saved, tracked side (offline seed; also the provisional bed from a side scanned without one).
    Each frame's dots are cast onto the side's platter plane (seen from that frame's camera, refined on the depth ring
    around the dots), taken into the scan frame with the tracked pose, and clustered (4 mm, seen >= 3 times). Centre =
    the centre of the circle the camera moved on, dropped onto the plane. plane: (n, off) in the scan frame, default the
    side's table_plane.json. Returns a dict, or None."""
    cal, gp = _side_io(side)
    if plane is None:
        tp = os.path.join(side, "table_plane.json")
        if not os.path.exists(tp): return None
        j = json.load(open(tp)); plane = (np.asarray(j["n"], float), -float(j["d"]))
    n0, off0 = _unit(plane[0]), float(plane[1])
    keys = sorted(gp)[::max(1, every)]
    if max_frames: keys = keys[:max_frames]
    pts, lifts, ring_s, cams, diam = [], [], [], [], []
    rng = np.random.default_rng(0)
    for k in keys:
        try: d, ir = read_frame(side, k)
        except OSError: continue
        if ir is None: continue
        T = gp[k] @ F; cams.append(T[:3, 3].copy())
        nc, oc = to_camera(n0, off0, T)
        dl = detect_dots(ir)
        if len(dl) < 3: continue
        P3 = dot_depth_points(d, cal, dl); ring = ring_depth_points(d, ir, cal, dl, 2000, rng)
        refined = False
        if len(ring) >= 60:
            sel = np.abs(ring @ nc + oc) < 3.0
            if sel.sum() >= 60:
                nn, cc = fit_plane(ring[sel])
                if nn @ nc < 0: nn = -nn
                if ang_deg(nn, nc) < 2.0: nc, oc = nn, float(-nn @ cc); refined = True
                Rs = ring[sel][rng.choice(sel.sum(), min(150, sel.sum()), replace=False)]
                ring_s.append(Rs @ T[:3, :3].T + T[:3, 3])
        h = P3 @ nc + oc; on = np.isnan(h) | (np.abs(h) < ON_PLANE)
        have = ~np.isnan(h) & on
        if refined and have.any(): lifts += list(h[have])          # the dots' height over the platter right around them
        L = lift(dl[on], cal, nc, oc); ok = ~np.isnan(L[:, 0])
        if not ok.any(): continue
        pts.append(L[ok] @ T[:3, :3].T + T[:3, 3])
        z = L[ok][:, 2]; diam += list(2 * dl[on][ok][:, 2] * z / cal[0])
    if not pts or len(cams) < 10: return None
    P = np.concatenate(pts)
    n, off = n0, off0
    if ring_s:
        Rr = np.concatenate(ring_s)
        r = ransac_plane(Rr, 1.0, 300, 100)
        if r is not None and ang_deg(r[0], n0) < 3 and abs(r[1] - off0) < 5: n, off = r[0], r[1]
    C = np.array(cams); c3, cam_r = circle_centre(C, n); c3 = c3 - (c3 @ n + off) * n
    e1, e2 = basis(n)
    P = P[np.abs(P @ n + off) < 3.0]
    Q = np.stack([(P - c3) @ e1, (P - c3) @ e2], 1)
    M, cnt, rms = _cluster2d(Q, POOL_CLUSTER, 3)
    if len(M) < 3: return None
    M, cnt, rms = _merge_close(M, cnt, rms)
    ok = cnt >= max(3, 0.1 * np.median(cnt))          # a speck seen a few times among dots seen 50+ times is not a dot
    M, cnt, rms = M[ok], cnt[ok], rms[ok]
    # the platter edge: only the dots measure it here (the depth ring lies next to them)
    rad = float(np.linalg.norm(M, axis=1).max() + (np.median(diam) / 2 if diam else 4) + 3)
    lift_mm = float(np.median(lifts)) if len(lifts) >= 10 else 0.0
    # camera placement at the side's last frame, the first prior for the next side
    Tl = gp[keys[-1]] @ F; ncam, ocam, ccam = plane_to_camera(n, off, c3, Tl)
    return dict(dots=np.column_stack([M, cnt, rms]), n=n, off=off, centre=c3, e1=e1, e2=e2, lift_mm=lift_mm,
                frames=len(cams), radius_mm=rad, dot_diam_mm=float(np.median(diam)) if diam else None, side=side,
                cam_radius=cam_r, camera=dict(n=ncam, off=ocam, centre=ccam, x_axis=Tl[:3, :3].T @ e1))


def provisional_bed(side, plane, source):
    """A bed from one saved side's own dots, for a project scanned without a learned bed (the window runs this on its
    table-finder thread; it takes 2-4 s). plane: the side's platter (n, off), scan frame. (BedModel, None) or (None, why)."""
    m = learn_from_side(side, every=3, plane=plane)
    if m is None: return None, "too few tracked frames showed dots"
    b, _ = merge_maps([m], source)
    if b is None or len(b) < LEARN_MIN_DOTS: return None, "only %d dots" % (0 if b is None else len(b))
    if b.self_ambiguity is None or b.self_ambiguity >= 0.6: return None, "its dots repeat (%.2f)" % (b.self_ambiguity or 1.0)
    b.provisional = True
    return b, None


def register_maps(A, B):
    """Map B onto map A (both bed frames, 2D): Reg or None. Exhaustive, so the runner-up is real."""
    return register_2d(np.asarray(B, float)[:, :2], np.asarray(A, float)[:, :2], exhaustive=True)


def merge_maps(maps, source=None):
    """One BedModel from several side maps (dicts from learn_from_side). Each map is registered onto the one with the
    most dots (2D, with a runner-up); dots within 6 mm merge (count-weighted), dots closer than 8 mm are one dot, and a
    dot is kept when seen on >= 2 sides or >= 6 times. Origin = the mean of the maps' spin centres; x = most-seen dot."""
    maps = [m for m in maps if m is not None and len(m["dots"]) >= 7]
    if not maps: return None, []
    ref = max(range(len(maps)), key=lambda i: len(maps[i]["dots"])); A = maps[ref]["dots"]
    pool, origins, used, notes = [], [], [], []
    for i, m in enumerate(maps):
        if i == ref: R, t = np.eye(2), np.zeros(2)
        else:
            r = register_maps(A, m["dots"])
            if r is None or r.inliers < HIT_MIN or r.inliers < HIT_RATIO * r.runner_up or r.rms > 1.6:
                notes.append("%s: not matched (%s)" % (os.path.basename(m.get("side", "?")), None if r is None else
                             "%d in, runner-up %d, rms %.2f" % (r.inliers, r.runner_up, r.rms)))
                continue
            R, t = r.R, r.t; notes.append("%s: %d in, runner-up %d, rms %.2f" % (os.path.basename(m.get("side", "?")), r.inliers, r.runner_up, r.rms))
        D = m["dots"][:, :2] @ R.T + t
        for k in range(len(D)): pool.append((D[k, 0], D[k, 1], m["dots"][k, 2], i))
        origins.append(t); used.append(m)
    Pp = np.array(pool)
    order = np.argsort(-Pp[:, 2]); taken = np.zeros(len(Pp), bool); dots = []
    for k in order:
        if taken[k]: continue
        m = (~taken) & (np.linalg.norm(Pp[:, :2] - Pp[k, :2], axis=1) < 6.0); taken |= m
        c = (Pp[m, :2] * Pp[m, 2:3]).sum(0) / Pp[m, 2].sum()
        for _ in range(3):              # trimmed: a side that saw the dot from low (a flat ellipse) must not drag it
            mm = m & (np.linalg.norm(Pp[:, :2] - c, axis=1) < 2.5)
            if not mm.any(): break
            c = (Pp[mm, :2] * Pp[mm, 2:3]).sum(0) / Pp[mm, 2].sum()
        mm = m & (np.linalg.norm(Pp[:, :2] - c, axis=1) < 2.5)
        if not mm.any(): mm = m
        wsum = Pp[mm, 2].sum(); rms = float(np.sqrt((Pp[mm, 2] * ((Pp[mm, :2] - c) ** 2).sum(1)).sum() / wsum))
        dots.append((c[0], c[1], wsum, rms, len(set(Pp[mm, 3].astype(int)))))
    Dd = np.array(dots)
    C, cnt, rms = _merge_close(Dd[:, :2], Dd[:, 2], Dd[:, 3], keep_heavier=True)
    nsides = np.array([Dd[np.linalg.norm(Dd[:, :2] - c, axis=1) < 8.0, 4].max() for c in C])
    # real dots are seen hundreds of times over many sides; specks off the pattern (glints on the part near the platter)
    # a few dozen times on one side. The plan's "2 sides or 6 sightings" kept 4-11 such specks: both are needed here.
    keep = (cnt >= max(6.0, 0.15 * np.median(cnt))) & ((nsides >= 2) if len(used) > 1 else True)
    C, cnt, rms = C[keep], cnt[keep], rms[keep]
    o = np.mean(origins, 0); C = C - o
    C, a = _orient_map(C, cnt)
    lifts = [m["lift_mm"] for m in used if m.get("lift_mm") is not None]
    diams = [m["dot_diam_mm"] for m in used if m.get("dot_diam_mm")]
    rad = float(np.linalg.norm(C, axis=1).max() + (np.median(diams) / 2 if diams else 4) + 3)
    cam = used[-1].get("camera")      # the last side's camera placement: only its plane is used (as the first prior)
    bed = BedModel(dots=np.column_stack([C, cnt, rms]), radius_mm=rad, lift_mm=float(np.median(lifts)) if lifts else 0.0,
                   dot_diam_mm=float(np.median(diams)) if diams else None, rms_mm=float(np.sqrt(np.mean(rms ** 2))) if len(rms) else None,
                   frames=int(sum(m["frames"] for m in used)), turned_deg=None,
                   **{"from": source or "offline:%d sides" % len(used)})
    bed.self_ambiguity = compute_self_ambiguity(bed.map2d)
    if cam is not None: bed.camera = dict(n=cam["n"], off=cam["off"], centre=cam["centre"], x_axis=cam["x_axis"])
    return bed, notes


# ------------------------------------------------------------------------------------------------ learning, empty table
class BedLearner:
    """Learn bed with the camera standing still and the table turning (no tracker). Frames 1-5 fix the plane; then
    every frame's dots are matched to the map grown so far and added to it. Done after one full turn (370 deg) with at
    least 12 dots seen 3 times or more; gives up after 75 s. add() -> {deg, dots, frames, state, msg}."""
    def __init__(self, cal, t0=None, gain=None):
        self.cal = cal; self.t0 = t0; self.gain = gain; self.frames = 0; self.state = "plane"; self.msg = "looking at the table"
        self.early = []; self.plane = None; self.lift_mm = 0.0; self.height_from = None
        self.cs = []; self.cn = []; self.cq = []            # cluster sums, counts, sum of squares (map coordinates)
        self.Rs = []; self.ts = []; self.angs = []; self.fidx = []; self.fails = 0; self.times = []
        self.surf = []; self.diam = []; self.last = None; self.c_est = None; self.gains = set()
        self.surf_map = []                  # the same surface samples placed on the map (the outline of a bed that is not round)
        self.cr = []                        # per cluster: sightings in a bright ring (a ringed board's dots)
        self.shapes = {}                    # bright blobs of dot size seen, and those left out as not round (detect_marks)
        self.view = []; self.obs = []       # every frame's dots on the plane (the camera stands still: its view), the placed frames'
        self.rim_views=[]
        self.map_check = None               # the map check's numbers, when it ran

    def _clusters(self, min_n=1):
        if not self.cs: return np.zeros((0, 2)), np.zeros(0)
        n = np.array(self.cn, float); C = np.array(self.cs) / n[:, None]; m = n >= min_n
        return C[m], n[m]

    def _add_dots(self, X, flags=None):
        for i, x in enumerate(X):
            f = 0 if flags is None else int(bool(flags[i]))
            if self.cs:
                C = np.array(self.cs) / np.array(self.cn, float)[:, None]; dd = np.linalg.norm(C - x, axis=1); k = int(dd.argmin())
                if dd[k] < 3.0:
                    self.cs[k] = self.cs[k] + x; self.cn[k] += 1; self.cq[k] += float(dd[k] ** 2); self.cr[k] += f; continue
            self.cs.append(x.copy()); self.cn.append(1); self.cq.append(0.0); self.cr.append(f)

    def _set_plane(self):
        """Plane from the first frames: dot depth (RANSAC), the depth ring around the dots and the frame's support plane;
        the platter surface (ring or support) wins when the dots agree with it, and the dots' lift over it is measured."""
        P3 = np.concatenate([e[1] for e in self.early]) if self.early else np.zeros((0, 3))
        P3 = P3[~np.isnan(P3[:, 0])] if len(P3) else P3
        ring = np.concatenate([e[2] for e in self.early]) if self.early else np.zeros((0, 3))
        dotp = ransac_plane(P3, 1.5, 500, 6) if len(P3) >= 6 else None
        cands = []
        if len(ring) >= RANSAC_RING[2]:
            r = ransac_plane(ring, 1.0, 300, RANSAC_RING[2])
            if r is not None: cands.append(("ring", r[0], r[1]))
        sup = [e[3] for e in self.early if e[3] is not None]
        if sup: s = max(sup, key=lambda x: x[2]); cands.append(("surface", s[0], s[1]))
        best = None
        for how, n, off in cands:
            if dotp is not None and ang_deg(n, dotp[0]) > 2.0: continue
            on = int((np.abs(P3 @ n + off) < ON_PLANE).sum()) if len(P3) else 0
            if best is None or on > best[0]: best = (on, how, n, off)
        if best is not None and (dotp is None or best[0] >= 0.5 * dotp[2].sum()):
            _, how, n, off = best
            h = P3 @ n + off if len(P3) else np.zeros(0); h = h[np.abs(h) < ON_PLANE]
            self.lift_mm = float(np.median(h)) if len(h) >= 4 else 0.0; self.height_from = how
        elif dotp is not None:
            n, off = dotp[0], dotp[1]; self.lift_mm = 0.0; self.height_from = "dots"
        else: return False
        self.plane = (_unit(n), float(off)); self.e = basis(n); return True

    def _register(self, dl, P3, d, ir, fr=None):
        n, off = self.plane; e1, e2 = self.e
        h = P3 @ n + off; expect = 0.0 if self.height_from == "dots" else self.lift_mm     # dots read lift over the surface
        use = np.isnan(h) | (np.abs(h - expect) < ON_PLANE)
        L = lift(dl[use], self.cal, n, off); ok = ~np.isnan(L[:, 0]); idx = np.nonzero(use)[0][ok]
        F2 = np.stack([L[ok] @ e1, L[ok] @ e2], 1)
        matched = np.zeros(len(dl), bool)
        if len(F2): self.view.append(F2)
        if len(F2) < 5: self.fails += 1; self.last = (dl, matched); return None
        if not self.Rs: R, t = np.eye(2), np.zeros(2); inl = np.ones(len(F2), bool)
        else:
            nreg = len(self.Rs); M2, n2 = self._clusters(2 if nreg >= 5 else 1)
            if len(M2) < 5: M2, n2 = self._clusters(1)
            mf = self._ringed(M2)
            ff = None if fr is None else fr[idx]
            if len(M2) > LEARN_REG_MAX:
                keep = np.argsort(-n2)[:LEARN_REG_MAX]; M2 = M2[keep]
                if mf is not None: mf = mf[keep]
            R = t = None
            # predicted placement first: with the spin centre known, the last angle step again (3 mm); before that, the
            # last frame's placement (the dots move at most ~6 mm a frame at 3.4 deg, and lie >= 18 mm apart: 8 mm)
            if self.c_est is not None and nreg >= 2:
                rate = (self.angs[-1] - self.angs[-2]) / max(1, self.fidx[-1] - self.fidx[-2])
                Rp = _rot2(self.angs[-1] + rate * (self._fno - self.fidx[-1])); tp = (np.eye(2) - Rp) @ self.c_est; rad, need = 3.0, 4
            else: Rp, tp, rad, need = self.Rs[-1], self.ts[-1], 8.0, 5
            M1, _ = self._clusters(1)                         # near a predicted placement every sighting counts
            m1f = self._ringed(M1)
            def distances(X):
                D = np.linalg.norm(X[:, None] - M1[None], axis=2)
                if ff is not None and m1f is not None: D = np.where(ff[:, None] == m1f[None], D, 1e9)
                return D
            X = F2 @ Rp.T + tp; D = distances(X); nn = D.argmin(1); m = D.min(1) < rad
            if m.sum() >= need:
                r = _kabsch2(F2[m], M1[nn[m]])
                for _ in range(2):
                    if r is None: break
                    R, t = r; D = distances(F2 @ R.T + t); nn = D.argmin(1); inl = D.min(1) < INLIER_TOL
                    r = _kabsch2(F2[inl], M1[nn[inl]]) if inl.sum() >= 3 else None
                if r is not None:
                    R, t = r; D = distances(F2 @ R.T + t); inl = D.min(1) < INLIER_TOL
                    if inl.sum() < need or np.sqrt((D.min(1)[inl] ** 2).mean()) > HIT_RMS: R = None
                else: R = None
            if R is None:
                reg = register_2d(F2, M2, flags=None if ff is None or mf is None else (ff, mf))
                if reg is None or reg.inliers < 5 or reg.rms > HIT_RMS or reg.inliers < HIT_RATIO * reg.runner_up:
                    self.fails += 1; self.last = (dl, matched); return None
                R, t, inl = reg.R, reg.t, reg.mask
        matched[idx[inl]] = True
        a = float(np.arctan2(R[1, 0], R[0, 0]))
        if self.angs: a = self.angs[-1] + ((a - self.angs[-1] + np.pi) % (2 * np.pi) - np.pi)
        self.Rs.append(R); self.ts.append(t); self.angs.append(a); self.fidx.append(self._fno)
        self._add_dots(F2 @ R.T + t, None if fr is None else fr[idx]); self.obs.append((self._fno, F2, R, t))
        if len(self.Rs) >= 3 and np.ptp(self.angs) > np.radians(8):
            A = np.concatenate([np.eye(2) - _rot2(x) for x in self.angs]); b = np.concatenate(self.ts)
            self.c_est = np.linalg.lstsq(A, b, rcond=None)[0]
        # platter surface for its radius (in-plane coordinates of this frame = map coordinates about the fixed centre)
        u, v = _frame_pts(d, 4)
        if len(u):
            Pp = backproject(u.astype(np.float64), v.astype(np.float64), d[v, u] * 0.1, self.cal)
            s = (np.abs(Pp @ n + off) < 2.0) & (ir[v, u] < DARK_IR)
            if s.any():
                S = Pp[s][:300]; S2 = np.stack([S @ e1, S @ e2], 1); self.surf.append(S2); self.surf_map.append(S2 @ R.T + t)
        z = L[ok][:, 2]; self.diam += list(2 * dl[idx][:, 2] * z / self.cal[0])
        self.last = (dl, matched)
        return R

    def turned(self):
        return float(np.degrees(np.ptp(self.angs))) if len(self.angs) > 1 else 0.0

    def ready(self):
        """IR-only learning needs a full turn to reveal dots outside its view."""
        return self.turned()>=LEARN_TURN_DEG

    def add(self, d, ir, t=None, gain=None):
        t = time.monotonic() if t is None else t
        self.last_t = t
        if self.t0 is None: self.t0 = t
        if gain is not None: self.gains.add(int(gain))
        self.frames += 1; t1 = time.monotonic()
        if self.state in ("done", "failed"): return self.status()
        dl, fr = detect_marks(ir, True, self.shapes); P3 = dot_depth_points(d, self.cal, dl)
        if self.state == "plane":
            ring = ring_depth_points(d, ir, self.cal, dl, 2000)
            sup = plane_from_depth(d, self.cal) if len(self.early) < 5 else None
            self.early.append((dl, P3, ring, sup, d, ir, self.frames, fr))
            self.last = (dl, np.zeros(len(dl), bool))
            if len(self.early) >= 5 and self._set_plane():
                self.state = "learning"
                for (dl0, P30, _, _, d0, ir0, f0, fr0) in self.early: self._fno = f0; self._register(dl0, P30, d0, ir0, fr0)
                self.early = []
            elif len(self.early) >= 12:
                self.state = "failed"; self.msg = "Could not see the table: tilt the scanner so the platter fills the lower half of the view."
        else:
            self._fno = self.frames; self._register(dl, P3, d, ir, fr)
        if self.plane is not None and self.angs:
            bright=(ir==np.iinfo(ir.dtype).max).astype(np.uint8);count,labels,stats,_=cv2.connectedComponentsWithStats(bright,8);points=[]
            for k in range(1,count):
                vv,uu=np.nonzero((labels==k)&(d>0))
                if len(uu):points.append(np.median(backproject(uu.astype(float),vv.astype(float),d[vv,uu]*.1,self.cal),axis=0))
            self.rim_views.append((float(self.angs[-1]),points))
        self.times.append(time.monotonic() - t1)
        if self.state == "learning":
            M, cnt = self._clusters(3)
            if self.ready() and len(M) >= LEARN_MIN_DOTS:
                why = self._check_map()
                if why is None: self.state = "done"; self.msg = "done"
                else: self.state = "failed"; self.msg = why
            elif t - self.t0 > LEARN_TIMEOUT:
                self.state = "failed"; deg = self.turned()
                lost = "%d of %d frames did not match" % (self.fails, self.frames)
                if deg >= LEARN_TURN_DEG:
                    self.msg = "Only %d dots seen. Try IR gain 3 to 6 and Learn bed again." % len(M)
                elif self.fails > 0.25 * self.frames:       # the dots were lost, not the table stopped (seen at gain 1 and 10-12)
                    self.msg = ("Lost track of the table dots (%s). Take everything off the table, set IR gain 3 to 6 and "
                                "Learn bed again." % lost)
                elif deg < LEARN_STILL_DEG:
                    self.msg = "The table did not turn in %d s (%.0f deg). Is it switched on?" % (LEARN_TIMEOUT, deg)
                elif self.fails > 0.05 * self.frames:       # it turned, and the dots were lost on the way (the turn was not followed)
                    self.msg = ("The table turned, but its dots were followed for only %.0f of 360 deg in %d s (%s). Take "
                                "everything off the table, set IR gain 3 to 6 and Learn bed again." % (deg, LEARN_TIMEOUT, lost))
                else:
                    self.msg = ("The table turned only %.0f of 360 deg in %d s: too slow for Learn bed. Turn it faster and "
                                "Learn bed again." % (deg, LEARN_TIMEOUT))
            else: self.msg = "%d of 360 deg, %d dots" % (min(360, self.turned()), len(M))
        return self.status()

    def speed(self):
        """Degrees per second of table turn so far (None before a few seconds), so the user can see whether to turn it up."""
        if self.t0 is None or len(self.angs) < 2: return None
        secs = getattr(self,'last_t',self.t0) - self.t0
        return round(self.turned() / secs, 1) if secs > 3.0 else None

    def status(self):
        M, _ = self._clusters(3)
        return dict(deg=round(min(self.turned(), 999), 1), dots=len(M), frames=self.frames, state=self.state, msg=self.msg, speed=self.speed(),
                    need=round(LEARN_TURN_DEG / LEARN_TIMEOUT, 1))

    def _kept(self):
        """The map's dots (map coordinates), their sighting counts and rms: clusters seen 3+ times, the close ones merged.
        Specks (glints, the part's edge) are seen a few times, the dots dozens of times in a turn: the same share-of-
        typical filter as learn_from_side / merge_maps, typical = the median of the 25 most seen (specks can outnumber
        the dots when the part stays on the table, which would drag a plain median down)."""
        C, cnt = self._clusters(3)
        idx = [i for i, n_ in enumerate(self.cn) if n_ >= 3]
        rms = np.array([np.sqrt(self.cq[i] / self.cn[i]) for i in idx])
        C, cnt, rms = _merge_close(C, cnt, rms)
        if not len(C): return C, cnt, rms
        top = np.sort(cnt)[::-1][:25]
        ok = cnt >= max(3.0, LEARN_KEEP * float(np.median(top)))
        return C[ok], cnt[ok], rms[ok]

    def _check_map(self):
        """Does the learned map hold up against the frames it came from? None if so, else why not (the message).
        The camera stood still, so the hull of every frame's dots on the plane is where dots CAN be seen. A real dot is
        seen in most of the placed frames that put it inside that view (7 good maps: median 0.49-0.72); a map smeared by
        frames placed at the wrong angle is full of dots seen a few times each (210644 with the part taken off: 74 dots
        for 29, median 0.07). And the maps of the even and the odd frames must agree (same place within 2 mm: good maps
        0.9-1.0 of the dots, the smeared one 0.49). Refused only when both say so (LEARN_SEEN_MIN, LEARN_HALVES_MIN)."""
        C, cnt, _ = self._kept()
        if len(C) < LEARN_MIN_DOTS or len(self.obs) < 10 or not self.view: return None
        V = np.concatenate(self.view).astype(np.float32)
        Hv = cv2.convexHull(V).reshape(-1, 2).astype(np.float64)
        e = np.roll(Hv, -1, 0) - Hv
        def inside(P):
            cr = e[None, :, 0] * (P[:, None, 1] - Hv[None, :, 1]) - e[None, :, 1] * (P[:, None, 0] - Hv[None, :, 0])
            return (cr >= 0).all(1) | (cr <= 0).all(1)
        vis = np.zeros(len(C)); seen = np.zeros(len(C))
        for _, F2, R, t in self.obs:
            f = (C - t) @ R; inn = inside(f)                     # map = R f + t  ->  f = R^T (m - t)
            near = np.linalg.norm(f[:, None] - F2[None], axis=2).min(1) < INLIER_TOL
            vis += inn; seen += inn & near
        m = vis >= 5
        seen_share = float(np.median(seen[m] / vis[m])) if m.any() else 1.0
        halves = []
        for par in (0, 1):
            Q = [F2 @ R.T + t for fno, F2, R, t in self.obs if fno % 2 == par]
            Ch, ch, _ = _cluster2d(np.concatenate(Q), 3.0, 2) if Q else (np.zeros((0, 2)), np.zeros(0), None)
            if len(Ch): Ch, ch, _ = _merge_close(Ch, ch, np.zeros(len(Ch)))
            if len(Ch): Ch = Ch[ch >= max(2.0, LEARN_KEEP * float(np.median(np.sort(ch)[::-1][:25])))]
            halves.append(Ch)
        agree_share = float((np.linalg.norm(halves[1][:, None] - halves[0][None], axis=2).min(1) < INLIER_TOL).mean()) \
            if len(halves[0]) and len(halves[1]) else 0.0
        self.map_check = dict(dots=len(C), seen=round(seen_share, 3), halves=round(agree_share, 3))
        if seen_share >= LEARN_SEEN_MIN or agree_share >= LEARN_HALVES_MIN: return None
        return ("The table dots did not line up over the turn (%d dots learned, each seen in only %d%% of the frames that "
                "showed its place; %d of %d frames did not match). Take everything off the table, set IR gain 3 to 6 and "
                "Learn bed again." % (len(C), round(100 * seen_share), self.fails, self.frames))

    def result(self, param_dir=None):
        """The learned BedModel (only when done)."""
        if self.state != "done": return None
        A = np.concatenate([np.eye(2) - _rot2(x) for x in self.angs]); b = np.concatenate(self.ts)
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        C, cnt, rms = self._kept()
        M = C - c; M, phi = _orient_map(M, cnt)
        n, off = self.plane; e1, e2 = self.e
        # the platter edge from the dots: the outermost dot that has a neighbour within 35 mm (the dots lie 18-25 mm
        # apart; a speck that got through lies alone), plus half a dot and 3 mm. A plain percentile fell 3 mm short.
        Dn = np.linalg.norm(M[:, None] - M[None], axis=2) + np.eye(len(M)) * 1e9; r_ = np.linalg.norm(M, axis=1)
        r_ = r_[Dn.min(1) < 35.0] if (Dn.min(1) < 35.0).any() else r_
        rad = float(r_.max() + (np.median(self.diam) / 2 if self.diam else 4) + 3)
        if self.surf:
            # the platter edge from its dark surface: 98th percentile of its radius, only out to 15 mm past the dots
            # (a dark table top or background at the same height gave 120-270 mm)
            S = np.linalg.norm(np.concatenate(self.surf) - c, axis=1); S = S[S < rad + 15.0]
            if len(S) >= 300: rad = max(rad, float(np.percentile(S, 98)))
        # the outline: this circle, unless the dots fill a rectangle (a square mat, a board): then its min-area rectangle
        # out to the dark surface around it. Ringed dots (a board's): a dot seen in a ring in most of its sightings
        Rp = _rot2(-phi)
        surf = (np.concatenate(self.surf_map) - c) @ Rp.T if self.surf_map else None
        outline = fit_outline(M, float(np.median(self.diam)) if self.diam else None, surf, reach=rad)
        ringed = self._ringed(C)
        board = None
        if ringed is not None and len(ringed) and ringed.mean() >= RINGED_BOARD: board = "ringed"
        else: ringed = None
        # camera placement at the last frame: bed coords b = Rot(-phi) (R_L f + t_L - c)  ->  f = R_L^T (Rot(phi) b + c - t_L)
        RL = _rot2(self.angs[-1])
        xf = RL.T @ _rot2(phi) @ np.array([1.0, 0.0]); x3 = _unit(xf[0] * e1 + xf[1] * e2)
        cen = c[0] * e1 + c[1] * e2 - off * n
        bed = BedModel(dots=np.column_stack([M, cnt, rms]), outline=outline, lift_mm=self.lift_mm, board=board, ringed=ringed,
                       dot_diam_mm=float(np.median(self.diam)) if self.diam else None,
                       rms_mm=float(np.sqrt(np.mean(rms ** 2))) if len(rms) else None, frames=self.frames,
                       turned_deg=self.turned(), ir_gain=sorted(self.gains), calib_md5=calib_md5(param_dir) if param_dir else None,
                       camera=dict(n=n, off=off, centre=cen, x_axis=x3), **{"from": "learn"})
        if self.rim_views and bed.dot_diam_mm and bed.outline.centred_circle:
            from app2.scan.rim import RimMarkers,basis
            markers=RimMarkers();bx,by=basis(n);final_angle=np.arctan2(x3@by,x3@bx)
            sigma=max(.1/np.sqrt(12),bed.rms_mm or 0)
            for angle,points in self.rim_views:
                candidates=[]
                for point in points:
                    height=float(point@n+off);delta=point-cen;radius=np.linalg.norm(delta-(delta@n)*n)
                    if height<-3*sigma and abs(radius-bed.radius_mm)<=bed.dot_diam_mm/2+3*sigma:
                        candidates.append(dict(angle_deg=float(np.degrees(np.arctan2(delta@by,delta@bx)))%360,height_mm=height))
                if len(candidates)==1:candidates[0]['reference_deg']=0
                elif len(candidates)==2:
                    a,b=candidates;gap=abs((a['angle_deg']-b['angle_deg']+180)%360-180)
                    if gap<=np.degrees(2*bed.dot_diam_mm/bed.radius_mm):
                        phase=np.angle(np.mean(np.exp(1j*np.radians([a['angle_deg'],b['angle_deg']]))));candidates=[dict(angle_deg=float(np.degrees(phase))%360,height_mm=(a['height_mm']+b['height_mm'])/2,reference_deg=180)]
                markers.observe(candidates,top_angle=float(np.degrees(final_angle+self.angs[-1]-angle))%360,learn=True)
            bed.rim_markers=markers.finish(sigma)
        bed.self_ambiguity = compute_self_ambiguity(bed.map2d, bed.ringed)
        bed.quality = dict(dots=len(M), clusters=int(sum(1 for x in self.cn if x >= 3)), not_round=int(self.shapes.get("not_round", 0)),
                           blobs=int(self.shapes.get("blobs", 0)), rms_mm=None if bed.rms_mm is None else round(float(bed.rms_mm), 3),
                           self_ambiguity=round(float(bed.self_ambiguity), 3), frames=self.frames,
                           turned_deg=round(self.turned(), 1), map_check=self.map_check, ringed=int(0 if ringed is None else ringed.sum()))
        return bed

    def _ringed(self, C):
        """Per kept dot (map coordinates, before centring): seen in a bright ring in most of its sightings. None when
        nothing was ever seen in a ring."""
        if not self.cs or not any(self.cr): return None
        n = np.array(self.cn, float); A = np.array(self.cs) / n[:, None]; r = np.array(self.cr, float)
        out = np.zeros(len(C), bool)
        for i, x in enumerate(C):
            m = np.linalg.norm(A - x, axis=1) < 3.0
            if m.any(): out[i] = r[m].sum() >= 0.5 * n[m].sum()
        return out


# ------------------------------------------------------------------------------------------------ one side
def pool_features(d, ir, cal, rng=None):
    """What the slow path keeps of a frame: its dots (at most 25, biggest first), their depth, <= 400 ring points."""
    dl = detect_dots(ir)
    if len(dl) > POOL_MAX_DOTS: dl = dl[np.argsort(-dl[:, 3])[:POOL_MAX_DOTS]]
    return dict(dots=dl, P3=dot_depth_points(d, cal, dl), ring=ring_depth_points(d, ir, cal, dl, POOL_MAX_RING, rng))


def solve_pool(pool, bed, cal, extra=(), rng=None):
    """The slow path for sides where no single frame shows enough dots (edge-on views, a low camera): pool the tracked
    frames' dots in the scan frame. Plane candidates: RANSAC over the pooled dot depth, over the pooled ring depth, and
    any extra (scan-frame) planes such as Auto remove base's. For each, every frame's dots are cast onto it from that
    frame's camera, taken into the scan frame, clustered (4 mm, >= 2 sightings) and matched to the map. pool: list of
    (features, T_scan_cam). Returns a scan-frame bed dict or None."""
    rng = rng or np.random.default_rng(3)
    if len(pool) < 2 or not has_dots(bed): return None
    def tr(P, T): return P @ T[:3, :3].T + T[:3, 3]
    Pd = [tr(f["P3"][~np.isnan(f["P3"][:, 0])], T) for f, T in pool if len(f["P3"])]
    Pd = np.concatenate(Pd) if Pd else np.zeros((0, 3))
    Pr = [tr(f["ring"], T) for f, T in pool if len(f["ring"])]
    Pr = np.concatenate(Pr) if Pr else np.zeros((0, 3))
    cands = [("auto", _unit(n), float(o)) for n, o in extra]
    left = np.ones(len(Pd), bool)
    for k in range(3):
        if left.sum() < 6: break
        r = ransac_plane(Pd[left], POOL_RANSAC[0], POOL_RANSAC[1], 6, rng)
        if r is None: break
        cands.append(("dots", r[0], r[1])); left &= ~(np.abs(Pd @ r[0] + r[1]) < POOL_RANSAC[0])
    if len(Pr) >= RANSAC_RING[2]:
        r = ransac_plane(Pr, POOL_RANSAC[0], POOL_RANSAC[1], RANSAC_RING[2], rng)
        if r is not None: cands.append(("ring", r[0], r[1]))
    best = None
    for how, n, off in cands:
        e1, e2 = basis(n); Q = []
        for f, T in pool:
            if not len(f["dots"]): continue
            nc, oc = to_camera(n, off, T); h = f["P3"] @ nc + oc
            on = np.isnan(h) | (np.abs(h) < ON_PLANE + 0.5)
            L = lift(f["dots"][on], cal, nc, oc); ok = ~np.isnan(L[:, 0])
            if ok.any(): Ls = tr(L[ok], T); Q.append(np.stack([Ls @ e1, Ls @ e2], 1))
        if not Q: continue
        C, cnt, _ = _cluster2d(np.concatenate(Q), POOL_CLUSTER, POOL_MIN_OBS)
        if len(C) < HIT_MIN: continue
        reg = register_2d(C, bed.map2d, exhaustive=len(C) <= 30, rng=rng)
        if not is_hit(reg): continue
        if best is None or (reg.inliers, -reg.rms) > (best[0].inliers, -best[0].rms): best = (reg, how, n, off)
    if best is None: return None
    reg, how, n, off = best
    height_from = how
    if how == "dots": off = off + bed.lift_mm; height_from = "lift"      # dot depth reads high by the learned lift
    B, c, x3 = bed_frame(n, off, reg.R, reg.t)                          # same (e1, e2): the 2D placement still holds
    return dict(n=_unit(n), off=float(off), centre=c, x_axis=x3, B_scan=B, inliers=reg.inliers, runner_up=reg.runner_up,
                rms=reg.rms, how=how, height_from=height_from, frames=len(pool))


class SideBed:
    """The bed of one side: hits per pass (camera frame), acceptance (3 agreeing hits within 30 frames + V1), the side's
    scan-frame bed (median over tracked hits), the slow pooled path, and the checks V2-V4 that hand removal back to
    Auto remove base when the bed and the side disagree. Used by the live window and by the offline replay alike."""
    def __init__(self, bed=None, prior=None):
        self.bed = bed; self.prior = prior; self.state = "looking" if bed is not None else "off"
        self.passes = {}; self.scan = []; self.fixed = None
        self.found_at = None; self.path = None; self.height_from = None; self.surface_shift = None
        self.dots_matched = 0; self.runner_up = 0; self.rms = None
        self.pool = []; self.v4_bad = 0; self.vetoes = []; self.auto_check = None; self.v3_said = None
        self.v1_said = False; self.hinted = 0; self.cam_drops = 0
        # giving up on the hints (the part moved on the table): per pass, never saved (a Continue tries the dots again)
        self.hint_pass = None; self.hint_moves = []; self.hint_off_pass = None; self.last_hint = None; self.hint_fix = {}; self.move_pause = None

    # --- state
    @property
    def found(self): return self.state == "found"

    def camera_bed(self, ps):
        p = self.passes.get(ps); return None if p is None else p.get("camera")

    def look_from(self, ps):
        """Frame of pass ps from which the table is looked for: 0, or where the scanner was seen to move (check_camera)."""
        p = self.passes.get(ps); return 0 if p is None else p.get("look_from", 0)

    def signature(self):
        """What the saved files (table_plane.json, bed_pose.json) depend on: a change means the side must be saved again."""
        f = self.fixed
        return (self.state, self.path, self.found_at, self.dots_matched, len(self.vetoes), None if f is None else id(f),
                (self.auto_check or {}).get("verdict"),
                tuple(sorted((ps, p.get("accepted")) for ps, p in self.passes.items())))

    def prior_for(self, ps):
        """Camera bed to try first in pass ps: this pass's, else the newest earlier pass's, else the one carried in."""
        cam = self.camera_bed(ps)
        if cam is not None: return cam
        recent=self.passes.get(ps,{}).get("reacquire_prior")
        if recent is not None:return recent
        for q in sorted(self.passes, reverse=True):
            if q < ps and self.passes[q].get("camera") is not None: return self.passes[q]["camera"]
        return self.prior

    def plane(self):
        return None if (not self.found or self.fixed is None) else (self.fixed["n"], self.fixed["off"])

    def disc(self):
        return None if (not self.found or self.fixed is None) else (self.fixed["centre"], self.bed.radius_mm)

    def B_scan(self):
        return None if (not self.found or self.fixed is None) else self.fixed.get("B_scan")

    def _veto(self, why, msg):
        self.state = "vetoed:" + why; self.vetoes.append(msg); return msg

    def _update_fixed(self):
        if not self.scan: return
        N = np.array([s["n"] for s in self.scan]); Cc = np.array([s["centre"] for s in self.scan])
        n = _unit(np.median(N, 0)); c = np.median(Cc, 0)
        Rm = sum(s["B_scan"][:3, :3] for s in self.scan); U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
        x = _unit(Rm[:, 0] - (Rm[:, 0] @ n) * n)
        B = np.eye(4); B[:3, 0] = x; B[:3, 1] = np.cross(n, x); B[:3, 2] = n; B[:3, 3] = c
        self.fixed = dict(n=n, off=float(-n @ c), centre=c, x_axis=x, B_scan=B)

    # --- hits
    def add_hit(self, ps, idx, hit, T=None):
        """One frame's hit (from find_bed) with its tracked pose T (camera -> scan, mm) or None when the frame was lost.
        Returns log lines."""
        ev = []
        if self.bed is None or self.state.startswith("vetoed") or hit is None: return ev
        p = self.passes.setdefault(ps, dict(hits=[], camera=None, accepted=None))
        p["hits"].append((idx, hit, None if T is None else np.asarray(T, float)))
        if p["camera"] is None:
            # a hit whose height is only the prior's (no dot depth to check it) cannot confirm a plane: it would agree
            # with every other such hit by construction. It still counts once a plane is accepted (tracking checks, hints)
            if hit.height_from == "prior-only": return ev
            agreeing = [(i, h, t) for i, h, t in p["hits"]
                        if abs(i - idx) < ACCEPT_WINDOW and h.height_from != "prior-only" and agree(h, hit)]
            if len(agreeing) < ACCEPT_HITS: return ev
            beneath = float(np.median([h.beneath or 0.0 for _, h, _ in agreeing]))
            if beneath >= V1_MAX:
                if not self.v1_said: self.v1_said = True; ev.append("V1: %.0f%% of the depth lies under the dots' plane; not taking it" % (100 * beneath))
                return ev
            p["camera"] = mean_hits([h for _, h, _ in agreeing]); p["accepted"] = idx; p["found_hits"] = [i for i, _, _ in agreeing]
            tracked = [(i, h, t) for i, h, t in agreeing if t is not None]
            if self.state == "looking" or (self.path == "pooled" and tracked):     # single-frame hits beat a pooled find
                first = self.state == "looking"; self.state = "found"
                if first: self.found_at = (ps, idx)
                self.scan = []
                self.path = "prior" if all(h.how == "prior" for _, h, _ in agreeing) else "dots"
                self.height_from = hit.height_from; self.surface_shift = hit.surface_shift
                self.dots_matched = int(max(h.inliers for _, h, _ in agreeing)); self.runner_up = int(max(h.runner_up for _, h, _ in agreeing))
                self.rms = float(np.median([h.rms for _, h, _ in agreeing]))
                for i, h, t in tracked: self.scan.append(dict(to_scan(h, t), idx=i, ps=ps))
                self._update_fixed()
                ev.append(("table found at frame %d (%d dots, %.1f mm)" if first else "table placed by the dots of single frames at frame %d (%d dots, %.1f mm)")
                          % (idx, self.dots_matched, self.rms))
            elif self.fixed is not None and tracked:
                s = to_scan(tracked[-1][1], tracked[-1][2]); dg, mm = self._off(s)
                ev.append("pass %d: table found again at frame %d (%.1f deg, %.1f mm from the side's)" % (ps + 1, idx, dg, mm))
            return ev
        if T is None or self.state != "found": return ev
        s = to_scan(hit, T)
        if self.fixed is None: self.scan.append(dict(s, idx=idx, ps=ps)); self._update_fixed(); return ev
        dg, mm = self._off(s)
        if dg > V4_DEG or mm > V4_MM:
            self.v4_bad += 1
            if self.v4_bad >= V4_RUN:
                ev.append(self._veto("V4", "table dots and tracking disagree (%.1f deg, %.1f mm) %d times in a row: Auto remove base takes over" % (dg, mm, V4_RUN)))
        else:
            self.v4_bad = 0; self.scan.append(dict(s, idx=idx, ps=ps)); self._update_fixed()
        return ev

    def _off(self, s):
        f = self.fixed; dg = ang_deg(s["n"], f["n"])
        mm = max(abs(f["n"] @ s["centre"] + f["off"]), float(np.linalg.norm(s["centre"] - f["centre"])))
        return dg, mm

    # --- camera placed by the dots (lost frames, Continue) and the camera bed while the scanner stays put
    def hint_pose(self, d, ir, cal, ps, budget_s=0.25):
        """Where the camera is (camera -> scan, METRES, for Tracker.hint) from the table dots of this frame, or None:
        B_scan @ inv(B_cam) with a hit that agrees with this pass's camera bed (when one is known); once this pass found
        that the part moved on the table (hint_outcome), that move applied: O @ B_scan @ inv(B_cam)."""
        self.last_hint = None; self.last_hint_quality = None
        B_s = self.B_scan()
        if B_s is None or self.hint_off_pass == ps: return None
        h = find_bed(d, ir, cal, self.bed, prior=self.prior_for(ps), budget_s=budget_s)
        cam = self.camera_bed(ps)
        if h is None or (cam is not None and not agree(h, cam)): return None
        if cam is None and self.passes.get(ps,{}).get("reacquire_prior") is not None and h.height_from=="prior-only":return None
        T = B_s @ np.linalg.inv(h.B_cam)                                    # where the dots put the camera, mm
        O = self.hint_fix.get(ps) if ps == self.hint_pass else None
        Th = T if O is None else O @ T
        self.last_hint = (T.copy(), Th.copy())                              # for hint_outcome
        pixels=getattr(h,'dots_px',None);matched=getattr(h,'matched',None);center=getattr(h,'centre',None)
        span=0
        if pixels is not None and matched is not None and center is not None:
            points=np.asarray(pixels)[np.asarray(matched,bool),:2]
            if len(points):span=min(np.ptp(points[:,0])/cal[0],np.ptp(points[:,1])/cal[1])*abs(float(center[2]))
        self.last_hint_quality = dict(inliers=getattr(h,'inliers',0), runner_up=getattr(h,'runner_up',0), rms_mm=getattr(h,'rms',None), span_mm=float(span))
        Th = Th.copy(); Th[:3, 3] /= 1000.0; self.hinted += 1
        return Th

    def hints_off(self, ps): return self.hint_off_pass == ps

    def moved(self, ps):
        """Did this pass find that the part moved on the table? (O, the move, or None)."""
        return self.hint_fix.get(ps) if ps == self.hint_pass else None

    def new_pass(self, ps):
        """A new pass (Start / Continue): the dots get their chance again, whatever the last pass decided."""
        self.hint_pass = ps; self.hint_moves = []; self.hint_off_pass = None; self.last_hint = None; self.hint_fix = {}

    def _move_of(self, O):
        """(deg, mm) of a move O of the part (scan frame): its rotation, and how far it carries the platter centre."""
        c = self.fixed["centre"]
        return (float(np.degrees(np.arccos(np.clip((np.trace(O[:3, :3]) - 1) / 2, -1.0, 1.0)))),
                float(np.linalg.norm(O[:3, :3] @ c + O[:3, 3] - c)))

    def hint_outcome(self, r, ps):
        """After Tracker.step (r: its result, with the hint hint_pose gave for this frame). Only a fit that was FOUND and
        refused as far from the dots' pose (r["refused"], r["refused_T"]) says anything about the part: a frame too thin
        for ICP, or a fit that failed, says nothing. The refused fit against the dots' pose gives the part's move on the
        table, O = T_fit @ inv(T_dots) (scan frame). HINT_GIVE_UP refused fits whose moves agree (HINT_AGREE_MM /
        HINT_AGREE_DEG), seen from views at least HINT_APART_DEG apart (the table turned between them), that keep the
        platter's plane where it was (ON_TABLE_*: a part moved on the table stays on it; a wrong-way match of a
        near-symmetric part turns the platter over) mean the part really moved. Then, for the rest of this pass, the
        hints come from where the part now sits (O @ the dots' pose; HINT_FOLLOW), with the guard still on: after the
        move the tracker's own search, unguarded, kept landing off (1-2 deg / 10-20 mm, or lost) even when the fits had
        found the move right. (HINT_FOLLOW False: hints stop for the pass and the search runs unguarded, as without a
        bed.) Fits that agree on another move later replace it. A fit starts the count over. Returns a log line when a
        move is taken."""
        hint, self.last_hint = self.last_hint, None
        if ps != self.hint_pass: self.new_pass(ps)
        if r.get("ok"): self.hint_moves = []; return None
        if self.hint_off_pass == ps or not r.get("hinted") or not r.get("refused"): return None
        Tf = r.get("refused_T")
        if hint is None or Tf is None or self.fixed is None: return None
        hint_T = hint[0]                                           # the dots' own pose: moves are measured from it
        Tf = np.asarray(Tf, float).copy(); Tf[:3, 3] *= 1000.0
        O = Tf @ np.linalg.inv(hint_T)
        n, off, c = self.fixed["n"], self.fixed["off"], self.fixed["centre"]
        if ang_deg(O[:3, :3] @ n, n) > ON_TABLE_DEG or abs(n @ (O[:3, :3] @ c + O[:3, 3]) + off) > ON_TABLE_MM:
            return None                                            # not a move along the table: no evidence either way
        same = [P for P, Th in self.hint_moves if self._move_of(np.linalg.inv(P) @ O)[0] <= HINT_AGREE_DEG
                and float(np.linalg.norm((P[:3, :3] - O[:3, :3]) @ c + P[:3, 3] - O[:3, 3])) <= HINT_AGREE_MM
                and self._move_of(np.linalg.inv(Th) @ hint_T)[0] >= HINT_APART_DEG]
        self.hint_moves.append((O, hint_T))
        if len(same) + 1 < HINT_GIVE_UP: return None
        dg, mm = self._move_of(O)
        if HINT_ON_MOVE == "pause":
            self.move_pause = ps; self.hint_moves = []                 # the window pauses the pass; the guard stays on
            return ("the part moved on the table (%d fits put it %.0f deg / %.0f mm from where the table dots do): "
                    "scanning paused so no frame lands in the wrong place. Press + New side to scan it where it now sits"
                    % (len(same) + 1, dg, mm))
        if HINT_FOLLOW:
            self.hint_fix[ps] = O; self.hint_moves = []
            return ("the part moved on the table (%d fits put it %.0f deg / %.0f mm from where the table dots do)? the dots "
                    "place it there for the rest of this pass" % (len(same) + 1, dg, mm))
        self.hint_off_pass = ps
        return ("the part moved on the table (%d fits put it %.0f deg / %.0f mm from where the table dots do)? tracking "
                "without the table dots for this pass" % (len(same) + 1, dg, mm))

    def check_camera(self, ps, idx, T):
        """The camera bed of pass ps holds only while the scanner stays put. On a tracked frame (T: camera -> scan, mm)
        the side's plane seen from T is where the platter must be; CAM_RUN frames in a row that put it more than V4_DEG /
        V4_MM (or its centre more than CAM_CENTRE_MM) from the camera bed (the scanner was moved mid-pass without a
        pause) drop it, and the pass looks for the table again from this frame. Returns log lines."""
        p = self.passes.get(ps)
        if p is None or p.get("camera") is None or self.fixed is None or self.state != "found" or T is None: return []
        f = self.fixed; nc, oc, cc = plane_to_camera(f["n"], f["off"], f["centre"], np.asarray(T, float))
        cam = p["camera"]; dg = ang_deg(nc, cam["n"])
        mm = max(abs(nc @ cam["centre"] + oc), abs(cam["n"] @ cc + cam["off"])); mc = float(np.linalg.norm(cc - cam["centre"]))
        if dg <= V4_DEG and mm <= V4_MM and mc <= CAM_CENTRE_MM: p["cam_bad"] = 0; return []
        p["cam_bad"] = p.get("cam_bad", 0) + 1
        if p["cam_bad"] < CAM_RUN: return []
        p.update(reacquire_prior=cam,camera=None, accepted=None, hits=[], cam_bad=0, look_from=idx); self.cam_drops += 1
        return ["pass %d: the scanner moved at frame %d (%.1f deg, %.1f mm, centre %.1f mm)? looking for the table again" % (ps + 1, idx, dg, mm, mc)]

    def drop_camera(self, ps, idx, why):
        """This pass's camera bed is let go (the dots keep placing the platter elsewhere: the scanner moved while tracking
        said it did not, 2026-10-25); the pass looks for the table again from this frame, the old bed as its prior."""
        p = self.passes.get(ps)
        if p is None or p.get("camera") is None: return []
        p.update(reacquire_prior=p["camera"], camera=None, accepted=None, hits=[], cam_bad=0, look_from=idx); self.cam_drops += 1
        return ["pass %d: %s at frame %d; looking for the table again" % (ps + 1, why, idx)]

    # --- slow path
    def add_pool(self, ps, idx, feats, T):
        if len(self.pool) >= POOL_MAX_FRAMES: return
        self.pool.append((feats, np.asarray(T, float), ps, idx))

    def pool_snapshot(self):
        return [(f, T) for f, T, _, _ in self.pool]

    def apply_pool(self, res, ps, idx):
        """A pooled result (solve_pool) found the bed in the scan frame."""
        if res is None or self.state != "looking": return []
        self.state = "found"; self.found_at = (ps, idx); self.path = "pooled"; self.height_from = res["height_from"]
        self.dots_matched = res["inliers"]; self.runner_up = res["runner_up"]; self.rms = res["rms"]
        self.fixed = dict(n=res["n"], off=res["off"], centre=res["centre"], x_axis=res["x_axis"], B_scan=res["B_scan"])
        return ["table found from %d pooled frames at frame %d (%d dots, %.1f mm)" % (res["frames"], idx, res["inliers"], res["rms"])]

    # --- checks against Auto remove base
    def check(self, sp, cached=False):
        """V2 (bed normal vs the spin axis) and V3 (bed vs Auto remove base's plane). Returns log lines."""
        ev = []
        if self.state != "found" or self.fixed is None or sp is None: return ev
        n, c = self.fixed["n"], self.fixed["centre"]
        ax = sp.spin_axis()
        if ax is not None:
            dg = min(ang_deg(ax, n), ang_deg(-ax, n))
            if dg > V2_DEG:
                ev.append(self._veto("V2", "V2: the table dots' plane is %.1f deg off the spin axis: Auto remove base takes over" % dg)); return ev
        pl = sp.cached_plane() if cached else sp.plane()
        if pl is None: return ev
        na, oa = pl[0], pl[1]; dg = ang_deg(na, n); den = na @ n
        s = -(na @ c + oa) / den if abs(den) > 0.3 else 0.0
        if s < -V3_UNDER_MM: verdict = "auto_under"
        elif s < -V3_MM: verdict = "auto_lower"
        elif s > V3_MM or (dg > V3_DEG and ax is not None): verdict = "auto_higher"
        elif dg > V3_DEG: verdict = "auto_tilted"
        else: verdict = "agree"
        self.auto_check = dict(deg=round(dg, 2), mm=round(float(s), 2), verdict=verdict)
        if verdict != self.v3_said:
            self.v3_said = verdict
            if verdict == "agree": ev.append("Auto remove base agrees with the table dots (%.1f deg, %+.1f mm)" % (dg, s))
            elif verdict == "auto_higher": ev.append("Auto remove base picked a flat face of the part (%+.1f mm, %.1f deg); using the table dots" % (s, dg))
            elif verdict == "auto_under": ev.append("Auto remove base found something %.0f mm under the table (the floor or the turntable's base?); using the table dots" % -s)
            elif verdict == "auto_lower":
                ev.append(self._veto("V3", "V3: Auto remove base is %.1f mm under the table dots' plane: Auto takes over" % -s))
            else: ev.append("Auto remove base is tilted %.1f deg from the table dots; keeping the dots" % dg)
        return ev

    # --- output
    def table_plane_json(self, margin=1.0, spin_axis=None):
        f = self.fixed; n = f["n"]
        tilt = 0.0 if spin_axis is None else min(ang_deg(spin_axis, n), ang_deg(-np.asarray(spin_axis), n))
        return {"n": _jl(n, 9), "d": float(-f["off"]), "margin": margin, "keep_above": True, "points": int(self.dots_matched),
                "tilt_deg": round(tilt, 3), "source": "bed", "dots": int(self.dots_matched), "centre": _jl(f["centre"], 4),
                "radius_mm": round(float(self.bed.radius_mm), 2), "outline": self.bed.outline.to_json(), "x_axis": _jl(f["x_axis"], 9)}

    def to_json(self):
        f = self.fixed
        st = "found" if self.state == "found" else ("none" if self.state == "off" else "auto")
        passes = {}
        for ps, p in self.passes.items():
            cam = p.get("camera")
            passes[str(ps)] = {"camera": None if cam is None else {"n": _jl(cam["n"]), "off": round(float(cam["off"]), 4),
                                                                   "centre": _jl(cam["centre"], 4), "x_axis": _jl(cam["x_axis"])},
                               "hits": [int(i) for i, _, _ in p["hits"]][:200], "accepted": p.get("accepted")}
        return {"version": 1, "bed_id": None if self.bed is None else self.bed.id, "state": st, "detail": self.state,
                "path": self.path, "found_at": None if self.found_at is None else "frame_%03d_%04d" % tuple(self.found_at),
                "n": None if f is None else _jl(f["n"], 9), "off": None if f is None else round(float(f["off"]), 4),
                "centre": None if f is None else _jl(f["centre"], 4),
                "radius_mm": None if self.bed is None else round(float(self.bed.radius_mm), 2),
                "outline": None if self.bed is None else self.bed.outline.to_json(),
                "x_axis": None if f is None else _jl(f["x_axis"]), "B_scan": None if f is None else _jl(f["B_scan"], 6),
                "dots_matched": int(self.dots_matched), "runner_up": int(self.runner_up),
                "rms_mm": None if self.rms is None else round(float(self.rms), 3), "height_from": self.height_from,
                "surface_shift_mm": self.surface_shift, "passes": passes, "auto_check": self.auto_check or {"verdict": "no_auto"},
                "vetoes": list(self.vetoes)}

    def save(self, path):
        tmp = path + ".tmp.%d" % os.getpid(); json.dump(self.to_json(), open(tmp, "w"), indent=1); os.replace(tmp, path)

    @classmethod
    def load(cls, path, bed=None, prior=None):
        """A side's saved bed (load_tracker: no re-detection over the saved frames)."""
        sb = cls(bed, prior)
        if bed is None: return sb                          # no bed any more (deleted): Auto remove base, as before
        try: j = json.load(open(path))
        except Exception: return sb
        if bed is not None and j.get("bed_id") not in (None, bed.id): return sb        # learned again since: look again
        det = j.get("detail") or j.get("state")
        if j.get("n") is not None and j.get("state") == "found":
            n = np.asarray(j["n"], float); c = np.asarray(j["centre"], float)
            B = np.asarray(j["B_scan"], float) if j.get("B_scan") is not None else None
            x = np.asarray(j["x_axis"], float) if j.get("x_axis") is not None else basis(n)[0]
            if B is None: B = np.eye(4); B[:3, 0] = x; B[:3, 1] = np.cross(n, x); B[:3, 2] = n; B[:3, 3] = c
            sb.fixed = dict(n=n, off=float(j["off"]), centre=c, x_axis=x, B_scan=B); sb.state = "found"
            sb.scan = [dict(n=n, off=float(j["off"]), centre=c, x_axis=x, B_scan=B, idx=-1, ps=-1)]
        elif isinstance(det, str) and det.startswith("vetoed"): sb.state = det
        sb.path = j.get("path"); sb.height_from = j.get("height_from"); sb.surface_shift = j.get("surface_shift_mm")
        sb.dots_matched = int(j.get("dots_matched") or 0); sb.runner_up = int(j.get("runner_up") or 0); sb.rms = j.get("rms_mm")
        sb.vetoes = list(j.get("vetoes") or []); sb.auto_check = j.get("auto_check")      # (an old "hint_off" is ignored: per pass now)
        fa = j.get("found_at")
        if fa: sb.found_at = tuple(int(x) for x in fa.split("_")[1:3])
        for ps, p in (j.get("passes") or {}).items():
            cam = p.get("camera")
            sb.passes[int(ps)] = dict(hits=[], accepted=p.get("accepted"), camera=None if cam is None else dict(
                n=np.asarray(cam["n"], float), off=float(cam["off"]), centre=np.asarray(cam["centre"], float), x_axis=np.asarray(cam["x_axis"], float)))
        return sb


# ------------------------------------------------------------------------------------------------ offline replay
def replay_side(side, bed, prior=None, frames=None, find_every=1, v4=True, pool=True, sideplane=True, log=None,
                timing=None, rng=None, find_until=90):
    """Run a saved side's frames through SideBed the way the live window does: find jobs on frames 0-29 of each pass
    (and on to frame find_until while the pass is not accepted), every 10th frame after (V4), pool jobs on tracked
    frames 10-90 while not found, SidePlane checks (V2, V3) every 10th frame. Returns the SideBed."""
    cal, gp = _side_io(side)
    stamps = []
    tj = os.path.join(side, "cache", "pointyoink_track.json")
    if os.path.exists(tj): stamps = [s["frame"] for s in json.load(open(tj)).get("stamps", [])]
    keys = [tuple(int(x) for x in s.split("_")[1:3]) for s in stamps] if stamps else sorted(gp)
    if frames is not None: keys = keys[:frames]
    sb = SideBed(bed, prior); sp = SidePlane() if sideplane else None; rng = rng or np.random.default_rng(0)
    tracked = {}
    for (ps, idx) in keys:
        T = gp.get((ps, idx)); T = None if T is None else T @ F
        n_tr = tracked.get(ps, 0)
        for e in sb.check_camera(ps, idx, T):
            if log: log("%s f%d: %s" % (os.path.basename(side), idx, e))
        i0 = idx - sb.look_from(ps)
        want_find = sb.state != "off" and not sb.state.startswith("vetoed") and (
            (sb.camera_bed(ps) is None and (i0 < ACCEPT_WINDOW or i0 <= find_until) and idx % find_every == 0) or
            (v4 and sb.camera_bed(ps) is not None and T is not None and n_tr % 10 == 0))
        want_pool = pool and sb.state == "looking" and T is not None and POOL_FROM <= n_tr <= POOL_TO and n_tr % POOL_EVERY == 0
        need_sp = sp is not None and T is not None
        if not (want_find or want_pool or need_sp): continue
        try: d, ir = read_frame(side, (ps, idx))
        except OSError: continue
        if ir is None: continue
        if want_find:
            t0 = time.monotonic(); h = find_bed(d, ir, cal, bed, prior=sb.prior_for(ps), rng=rng)
            if timing is not None: timing.setdefault("find", []).append(time.monotonic() - t0)
            for e in sb.add_hit(ps, idx, h, T):
                if log: log("%s f%d: %s" % (os.path.basename(side), idx, e))
        if T is not None:
            if sp is not None: sp.add(d, cal, T)
            if want_pool:
                t0 = time.monotonic()
                sb.add_pool(ps, idx, pool_features(d, ir, cal, rng), T)
                extra = []
                if sp is not None and sp.plane() is not None: extra.append((sp.plane()[0], sp.plane()[1]))
                res = solve_pool(sb.pool_snapshot(), bed, cal, extra, rng)
                if timing is not None: timing.setdefault("pool", []).append(time.monotonic() - t0)
                for e in sb.apply_pool(res, ps, idx):
                    if log: log("%s f%d: %s" % (os.path.basename(side), idx, e))
            tracked[ps] = n_tr + 1
            if sp is not None and n_tr % 10 == 0:
                for e in sb.check(sp):
                    if log: log("%s f%d: %s" % (os.path.basename(side), idx, e))
    sb.sideplane = sp
    return sb


# ------------------------------------------------------------------------------------------------ CLI
def _refuse(path):
    """Never write into a project's scan data (<Project...>/data/...): offline outputs go to build/ or elsewhere."""
    parts = os.path.realpath(os.path.abspath(path)).split(os.sep)
    return any(parts[i] == "data" and parts[i - 1].startswith("Project") for i in range(1, len(parts) - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("learn-sides"); a1.add_argument("sides", nargs="+"); a1.add_argument("--out", required=True)
    a1.add_argument("--every", type=int, default=3)
    a2 = sub.add_parser("side-pose"); a2.add_argument("side"); a2.add_argument("--bed", required=True); a2.add_argument("--out", required=True)
    a2.add_argument("--frames", type=int, default=150)
    a = ap.parse_args()
    if a.cmd == "learn-sides":
        if _refuse(a.out): print("refusing to write into a project's scan data: %s" % a.out); return 2
        maps = []
        for s in a.sides:
            m = learn_from_side(s, a.every); maps.append(m)
            print("%s: %s" % (os.path.basename(s.rstrip("/")), "no map" if m is None else "%d dots, lift %.2f mm, radius %.1f mm" % (len(m["dots"]), m["lift_mm"], m["radius_mm"])), flush=True)
        bed, notes = merge_maps(maps, "offline:%d sides" % len([m for m in maps if m is not None]))
        for n_ in notes: print("  " + n_)
        if bed is None: print("no bed"); return 1
        bed.calib_md5 = calib_md5(os.path.join(a.sides[0], "param")); bed.save(a.out)
        print("bed: %d dots, radius %.1f mm, lift %.2f mm, rms %.2f mm, self-ambiguity %.2f -> %s" % (len(bed), bed.radius_mm, bed.lift_mm, bed.rms_mm or 0, bed.self_ambiguity, a.out))
        return 0
    if a.cmd == "side-pose":
        if _refuse(a.out): print("refusing to write into a project's scan data: %s" % a.out); return 2
        bed = BedModel.load(a.bed)
        if bed is None: print("no bed in %s" % a.bed); return 1
        sb = replay_side(a.side, bed, frames=a.frames, sideplane=True)
        j = dict(sb.to_json(), folder=os.path.basename(os.path.abspath(a.side)))     # the build checks which side this is
        tmp = a.out + ".tmp.%d" % os.getpid(); json.dump(j, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
        print("side %s: bed %s (%s, %d dots)" % (os.path.basename(a.side.rstrip("/")), sb.state, sb.path, sb.dots_matched))
        return 0


if __name__ == "__main__":
    sys.exit(main())
