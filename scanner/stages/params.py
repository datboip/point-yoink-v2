# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every knob of every stage: name, default, range, unit, the Revo label and value when the device has the same knob,
and the script flag it maps to. Also the param paths the CLI and the recipe use:

    side.fuse.voxel            common param of a node
    fused.method               the node's method
    fused.tsdf.trunc           a param of one method of a node
    side[<id>].keep.min_share  a per-side override of a side node's common param

Values are validated and coerced to their type before hashing (an int stays an int, floats are rounded to 9 decimals).
`help` says what OUR algorithm does, never Revo's maths (which we do not know)."""
import copy, math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    name: str
    type: str                     # float | int | bool | choice | str
    default: object
    lo: float = None
    hi: float = None
    step: float = None
    unit: str = ""
    label: str = ""               # Revo wording when Revo has the knob
    revo: object = None           # the device's default value, when known
    help: str = ""
    advanced: bool = False
    choices: tuple = ()
    legacy_flag: str = None       # the script flag; passed only when the value differs from the script's own default
    script_default: object = None  # the script's own default when it differs from ours (None = same as default)

    def coerce(self, v):
        if isinstance(v, str) and self.type != "str":
            s = v.strip()
            if self.type == "bool":
                if s.lower() in ("1", "true", "yes", "on"): return True
                if s.lower() in ("0", "false", "no", "off"): return False
                raise ValueError("%s: expected on/off, got %r" % (self.name, v))
            if self.type == "choice":
                v = s
            elif self.type == "int":
                v = float(s)
            elif self.type == "float":
                v = float(s)
        if self.type == "float":
            v = float(v)
            if not math.isfinite(v): raise ValueError("%s: not a number" % self.name)
            v = round(v, 9) + 0.0
        elif self.type == "int":
            f = float(v)
            if f != int(f): raise ValueError("%s: expected a whole number, got %r" % (self.name, v))
            v = int(f)
        elif self.type == "bool":
            if not isinstance(v, bool):
                if v in (0, 1): v = bool(v)
                else: raise ValueError("%s: expected on/off, got %r" % (self.name, v))
        elif self.type == "choice":
            if v not in self.choices: raise ValueError("%s: expected one of %s, got %r" % (self.name, "/".join(map(str, self.choices)), v))
        elif self.type == "str":
            v = str(v)
        if self.type in ("float", "int"):
            if self.lo is not None and v < self.lo - 1e-12: raise ValueError("%s: %s is below %s" % (self.name, v, self.lo))
            if self.hi is not None and v > self.hi + 1e-12: raise ValueError("%s: %s is above %s" % (self.name, v, self.hi))
        return v

    def describe(self):
        rng = "" if self.lo is None else " [%s..%s]" % (fmt(self.lo), fmt(self.hi))
        if self.choices: rng = " [%s]" % "|".join(map(str, self.choices))
        revo = "" if self.revo is None else "   Revo %s%s" % (('"%s" ' % self.label) if self.label else "", fmt(self.revo))
        return "%-18s %-8s %s%s%s%s" % (self.name, fmt(self.default), self.unit, rng, revo, ("  (advanced)" if self.advanced else ""))


def fmt(v):
    if isinstance(v, float): return ("%.4f" % v).rstrip("0").rstrip(".")
    return str(v)


P = Param
# ---------------------------------------------------------------------------------------------------------------- nodes
# node kind -> methods (default first)
METHODS = {
    "side.fuse": ["tsdf"],
    "side.platter": ["auto", "plate", "search", "manual", "none"],
    "side.keep": ["keep_part"],
    "lineup.pairs": ["fpfh_icp"],
    "lineup.place": ["consensus"],
    "lineup.fix": ["manual"],
    "lineup.refine": ["pose_graph", "none"],
    "lineup.adjust": ["manual"],
    "fused": ["tsdf", "points", "plus"],
    "meshed": ["auto", "as_fused", "poisson", "bpa"],
    "report": ["needs"],
}

TSDF_PARAMS = [
    P("trunc", "float", 3.0, 1, 8, 0.5, "voxels", help="truncation band in voxels: wider merges nearby double surfaces, narrower keeps thin sheets", legacy_flag="--trunc"),
    P("min_weight", "float", 3.0, 1, 20, 1, "views", help="drop voxels seen by fewer frames (GPU path)", legacy_flag="--min-weight"),
    P("filter", "float", 0.0, 0, 10, 0.5, "mm", help="drop depth pixels at jumps bigger than this (0 = off; 3 softened a 2.3 mm recess to 2.1 on 3 test scans, 2026-10-12)", legacy_flag="--filter", script_default=3.0),
    P("grazing", "float", 80.0, 45, 89, 1, "deg", help="drop pixels seen more edge-on than this", legacy_flag="--grazing"),
    P("max_depth", "float", 600.0, 200, 1000, 10, "mm", help="ignore farther pixels", legacy_flag="--max-depth"),
]

COMMON = {
    "side.fuse": [
        P("voxel", "float", 0.4, 0.2, 2.0, 0.05, "mm", label="Point Distance", revo=0.4, help="TSDF voxel of one side on its own (build_project --voxel)"),
        *TSDF_PARAMS,
        P("every", "int", 1, 1, 10, 1, "", help="use every Nth frame", legacy_flag="--every", advanced=True),
        P("gpu", "bool", True, help="fuse on the GPU (fuse.py falls back to the CPU on CUDA OOM by itself)", advanced=True),
    ],
    "side.platter": [
        P("margin", "float", 1.0, 0, 40, 0.1, "mm", help="also drop this much above the platter", legacy_flag="--margin"),
        P("max_tilt", "float", 8.0, 1, 20, 1, "deg", help="how far the platter may lean from the spin axis", legacy_flag="--max-tilt"),
    ],
    "side.keep": [
        P("min_share", "float", 0.03, 0, 1, 0.01, "", label="Isolation rate",
          help="keep every piece at least this share of the biggest (Revo's 15% is not used: a single side breaks into real pieces)",
          legacy_flag="--min-share"),
    ],
    "lineup.pairs": [
        P("voxel", "float", 1.0, 0.3, 5, 0.1, "mm", help="working cloud spacing of the pair fits"),
        P("samples", "int", 250000, 20000, 1000000, 10000, "", help="points sampled per side", advanced=True),
        P("max_dist", "float", 6.0, 1, 20, 0.5, "mm", help="ICP reach", advanced=True),
    ],
    "lineup.place": [
        P("base", "str", "auto", help="the side that stays put: auto (fits best with the others) or a side id"),
        P("tol", "float", 1.5, 0.3, 5, 0.1, "mm", help="a sample lands on the model when this close", legacy_flag="--tol"),
        P("hints", "str", "", help="how each side was scanned compared with the base: side id:same / flipped / sideways, separated by commas", advanced=True),
    ],
    "lineup.fix": [],
    "lineup.refine": [
        P("voxel", "float", 0.5, 0.2, 3, 0.1, "mm", help="spacing of the points used", legacy_flag="--voxel"),
        P("rounds", "int", 2, 1, 6, 1, "", help="pair fits then pose graph, this many times", legacy_flag="--rounds"),
    ],
    "lineup.adjust": [],
    "fused": [
        P("color","choice","auto",choices=("auto","on","off"),label="Scan color",help="Auto uses saved color when every active side has it. Color takes extra time."),
        P("white_balance","bool",True,label="Auto white balance",help="neutralize the camera's tint so a gray part comes out gray (off for parts meant to look warm or cool)"),
        P("point_distance", "float", 0.4, 0.1, 2.0, 0.05, "mm", label="Point Distance", revo=0.4,
          help="tsdf: voxel size; points: cell size of the fused points"),
        P("frames_per_side", "int", 300, 0, 5000, 50, "", help="at most this many frames per side, evenly spread (0 = all); a 790-frame side erased the lip otherwise"),
    ],
    "meshed": [],
    "report": [
        P("min_mm", "float", 8.0, 0, 100, 1, "mm", help="open edges shorter than this are pinholes, not places to scan", legacy_flag="--min-mm"),
        P("red_mm", "float", 2.0, 0.1, 20, 0.5, "mm", help="red within this distance of an open edge", legacy_flag="--red-mm"),
        P("yellow_mm", "float", 6.0, 0.5, 40, 0.5, "mm", help="yellow within this distance", legacy_flag="--yellow-mm"),
    ],
}

METHOD_PARAMS = {
    ("fused", "tsdf"): list(TSDF_PARAMS),
    ("fused", "points"): [
        P("step", "int", 2, 1, 8, 1, "px", help="use every Nth pixel of every frame"),
        P("edge_px", "float", 6.0, 0, 30, 1, "px", help="points closer than this to a depth edge weigh less"),
        P("z0", "float", 260.0, 100, 600, 10, "mm", help="weight = (z0/z)^2: near pixels weigh more", advanced=True),
        P("workers", "int", 6, 1, 12, 1, "", help="sides fused in parallel", advanced=True),
        P("min_frames", "float", 3.0, 1, 50, 1, "", help="keep points seen by at least this many frames"),
        P("min_sides", "int", 1, 1, 10, 1, "", help="keep points seen by at least this many sides", advanced=True),
        P("normal_agree", "float", 0.3, 0, 1, 0.05, "", help="keep cells whose frames' normals agree at least this much", advanced=True),
        P("rescue_mm", "float", 3.0, 0, 10, 0.5, "mm", help="with Min. sides above 1: still keep a one-side point when no point several sides agree on lies within this distance (a scan that fills a hole), 0 = drop them all", advanced=True),
    ],
    ("fused", "plus"): [       # fuse.py --mode plus + drop_enclosed.py (methods/fuse_plus.py); tuned on the 15-side headrest
        P("trunc_mm", "float", 1.2, 0.3, 5, 0.1, "mm", help="truncation band: narrower bands doubled walls where sides disagree", legacy_flag="--plus-trunc"),
        P("block_res", "int", 16, 4, 32, 1, "voxels",
          help="voxels per block edge: blocks set how far free-space votes reach (16 x 0.3 mm kept the thin lip, 8 x 0.3 lost it)",
          legacy_flag="--plus-block-res", script_default=0),
        P("min_weight", "float", 2.0, 0, 20, 1, "views", help="marching cubes needs weight above this at all 8 corners", legacy_flag="--plus-min-weight"),
        P("smooth_mm", "float", 0.24, 0, 1.0, 0.02, "mm",
          help="weighted blur of the distance field (sigma, mm; 0 = off): merges misplaced copies of a wall into one"),
        P("smooth_min_sides", "int", 3, 1, 20, 1, "", help="blur only where at least this many sides saw (1-2-side details such as a thin lip stay sharp)",
          legacy_flag="--plus-smooth-min-sides"),
        P("drop_enclosed", "bool", True, help="drop surface sealed inside the part (no scanner can see it) and ragged rims (drop_enclosed.py)"),
        P("enclosed_rays", "int", 32, 8, 128, 8, "", help="rays per side of each point for the enclosed test", advanced=True),
        P("tidy_rounds", "int", 4, 0, 10, 1, "", help="rounds of dropping rim triangles that hang on by one edge", advanced=True),
        P("fallback", "bool", True, help="if the plus fuse fails (e.g. GPU out of memory), build with the classic fuse and retry next run", advanced=True),
    ],
    ("meshed", "poisson"): [
        P("quality", "int", 6, 1, 7, 1, "", label="Quality", revo=7,
          help="Poisson grid, set by the part size: Q6 = octree depth round(log2(extent / 0.18 mm)) (depth 10 on a 180-190 mm part, "
               "as the bake), Q7 one depth finer (~0.10 mm there; the device's L7 is 0.11), and each level below Q6 about 1.41x "
               "coarser (depth -1 every two levels, octree scale 1.1 or 1.56). The run shows the grid it really used; a level "
               "the depth limits (6..11) cannot tell apart is reported as the same as its neighbour"),
        P("backend", "choice", "auto", choices=("auto", "pml", "o3d"), help="auto = PyMeshLab when a PyMeshLab python is configured, else Open3D"),
        P("pointweight", "float", 4.0, 0, 20, 0.5, "", help="screened Poisson point weight", advanced=True),
        P("samples_per_node", "float", 1.5, 1, 20, 0.5, "", help="Poisson samples per node", advanced=True),
        P("confidence", "bool", True, help="quality = log(1 + weight) (pml only)", advanced=True),
        P("trim", "float", 0.5, 0.1, 5, 0.1, "mm", help="faces with a vertex further than this from the points are cut (no invented surface)"),
        P("hole_fill", "bool", False, label="Hole Filling (Auto)", revo=False, help="on = trim at fill_trim instead, so Poisson closes holes near real data"),
        P("fill_trim", "float", 3.0, 0.5, 10, 0.5, "mm", help="how far from real data a filled hole may reach"),
        P("min_piece", "float", 0.002, 0, 0.1, 0.001, "", help="drop pieces under this share of faces"),
        P("taubin", "int", 3, 0, 20, 1, "", help="light smoothing passes"),
    ],
    ("meshed", "bpa"): [],
    ("meshed", "as_fused"): [],
    ("meshed", "auto"): [],
}

# -------------------------------------------------------------------------------------------------------------- tools
TOOL_PARAMS = {
    ("drill", "mesh"): [P("anchors", "str", "[]"), P("shape", "choice", "cylinder", choices=("cylinder", "box", "sphere")),
        P('diameter', "float", 5.0, 0.2, 100, 0.1, 'mm'),
        P('width', "float", 5.0, 0.2, 100, 0.1, 'mm'),
        P('height', "float", 5.0, 0.2, 100, 0.1, 'mm'),
        P('depth', "float", 10.0, 0.1, 500, 0.1, 'mm'),
        P('tilt', "float", 0.0, -89, 89, 0.1, 'deg'),
        P('rotation', "float", 0.0, 0, 360, 0.1, 'deg'),
        P('offset_u', "float", 0.0, -100, 100, 0.1, 'mm'),
        P('offset_v', "float", 0.0, -100, 100, 0.1, 'mm'),
        P('inset', "float", 0.0, -100, 500, 0.1, 'mm')],
    ("slice_gap", "mesh"): [P("anchors", "str", "[]"), P("width", "float", 1., 0.1, 100, 0.1, "mm"), P("depth", "float", 5., 0.1, 500, 0.1, "mm")],
    ("fin", "mesh"): [P("anchors", "str", "[]"), P("thickness", "float", 2., 0.2, 30, 0.1, "mm"), P("reach", "float", 8., 0.5, 100, 0.5, "mm"), P("shape", "choice", "triangle", choices=("triangle", "rounded"))],
    ("isolation", "points"): [
        P("rate", "float", 1.0, 0, 100, 0.5, "%", label="Isolation rate", revo=15.0,
          help="26-connected clusters at cluster_voxel; drops clusters smaller than max(min_pts, rate % of the biggest)"),
        P("cluster_voxel", "float", 1.0, 0.2, 5, 0.1, "mm", advanced=True),
        P("min_pts", "int", 200, 0, 100000, 50, "", advanced=True),
    ],
    ("overlap", "points"): [
        P("distance", "float", 0.21, 0.05, 2.0, 0.01, "mm", label="Overlap Distance", revo=0.21,
          help="s = distance/0.21: normal-gated weighted plane projection with sig_n = s x (1.0, 0.7, 0.5), then dedupe at s x 0.25 per (cell, normal bin); opposite-facing points never merge"),
        P("radius", "float", 1.5, 0.3, 5, 0.1, "mm", advanced=True),
        P("sig_t", "float", 0.5, 0.1, 3, 0.05, "mm", advanced=True),
        P("gate", "float", 0.7, 0, 1, 0.05, "", advanced=True),
        P("k", "int", 64, 8, 256, 8, "", advanced=True),
        P("schedule", "str", "", help="explicit sig_n list, e.g. 1.0,0.7,0.5 (overrides distance)", advanced=True),
    ],
    ("smooth", "points"): [
        P("strength", "float", 10.0, 1, 25, 1, "", label="Smooth strength", revo=10.0, help="plane-projection passes with sig_n = 0.03 x strength mm"),
        P("times", "int", 2, 1, 50, 1, "", label="Smooth times", revo=10),
        P("sig_t", "float", 0.3, 0.05, 2, 0.05, "mm", advanced=True),
        P("min_support", "int", 3, 0, 32, 1, "", help="then drop points with fewer same-facing neighbours", advanced=True),
    ],
    ("isolation", "mesh"): [
        P("rate", "float", 3.0, 0, 100, 0.5, "%", label="Isolation rate", revo=15.0, help="keep_part.py: keep pieces at least rate % of the biggest"),
    ],
    ("fill_holes", "mesh"): [
        P("hole_mm", "float", 0.0, 0, 500, 1, "mm", label="Hole Filling (Auto)", help="largest hole to fill (0 = auto: 4 % of the bbox diagonal)"),
    ],
    ("simplify", "mesh"): [
        P("keep_pct", "float", 40.0, 1, 99, 1, "%", label="Simplify Ratio", revo=40.0, help="keep this share of faces (fast_simplification)"),
    ],
    ("smooth", "mesh"): [
        P("strength", "float", 30.0, 5, 100, 5, "%", label="Smooth", revo=30.0, help="Taubin lambda = clip(0.5 x strength/30, 0.05, 0.9); runaway vertices pinned back"),
        P("times", "int", 3, 1, 20, 1, "", label="Smooth times", revo=3),
    ],
    ("slivers", "mesh"): [
        P("factor", "float", 4.0, 1.5, 20, 0.5, "x median edge", help="drop faces whose longest edge is this many times the median"),
    ],
    ("finish", "mesh"): [       # tools/finish.py -> scanner/finish_mesh.py: the print recipe the user picked ("E2")
        P("faces", "int", 250000, 100000, 1000000, 10000, "triangles", label="Target triangles",
          help="simplify to this many triangles last (binary STL: 50 bytes a triangle + 84; 250k is about 12.5 MB)"),
        P("depth", "int", 9, 6, 11, 1, "", help="screened Poisson octree depth over the model", advanced=True),
        P("trim_mm", "float", 3.0, 0.5, 10, 0.5, "mm",
          help="drop Poisson surface further than this from the scan (it balloons across openings; tighter re-opens the holes Poisson filled)",
          advanced=True),
        P("hole_edges", "int", 400, 0, 2000, 50, "edges", help="close holes whose rim has up to this many edges (0 = leave them open)"),
        P("smooth", "int", 5, 0, 20, 1, "", help="Taubin smoothing passes (keeps the size; 0 = off)"),
    ],
    ("delete", "mesh"): [],
    ("delete", "points"): [],
    ("match_cleaned_sides", "mesh"): [P("distance_mm", "float", 1.5, 0.2, 10, 0.1, "mm")],
    ("match_cleaned_sides", "points"): [P("distance_mm", "float", 1.5, 0.2, 10, 0.1, "mm")],
    ("cut_plane", "mesh"): [P("nx", "float", 0.0, -1, 1, 0.0001, ""), P("ny", "float", 0.0, -1, 1, 0.0001, ""), P("nz", "float", 1.0, -1, 1, 0.0001, ""),
                            P("d", "float", 0.0, -100000, 100000, 0.01, "mm"), P("margin", "float", 1.5, 0, 40, 0.1, "mm"),
                            P("pad", "float", 4.0, 0, 100, 0.5, "mm"), P("poly", "str", "")],
    ("cut_plane", "points"): [P("nx", "float", 0.0, -1, 1, 0.0001, ""), P("ny", "float", 0.0, -1, 1, 0.0001, ""), P("nz", "float", 1.0, -1, 1, 0.0001, ""),
                            P("d", "float", 0.0, -100000, 100000, 0.01, "mm"), P("margin", "float", 1.5, 0, 40, 0.1, "mm"),
                             P("pad", "float", 4.0, 0, 100, 0.5, "mm"), P("poly", "str", "")],
    ("clone", "points"): [P("src", "str", "[]"), P("dst", "str", "[]"), P("shape", "choice", "cylinder", choices=("cylinder", "box")),
                          P("diameter", "float", 20.0, 1, 300, 0.5, "mm"), P("width", "float", 20.0, 1, 300, 0.5, "mm"), P("height", "float", 20.0, 1, 300, 0.5, "mm"),
                          P("depth", "float", 8.0, 0.5, 100, 0.5, "mm"), P("rotation", "float", 0.0, -180, 180, 1, "deg"), P("mirror", "choice", "no", choices=("no", "yes"))],
    ("clone", "mesh"): [P("src", "str", "[]"), P("dst", "str", "[]"), P("shape", "choice", "cylinder", choices=("cylinder", "box")),
                          P("diameter", "float", 20.0, 1, 300, 0.5, "mm"), P("width", "float", 20.0, 1, 300, 0.5, "mm"), P("height", "float", 20.0, 1, 300, 0.5, "mm"),
                          P("depth", "float", 8.0, 0.5, 100, 0.5, "mm"), P("rotation", "float", 0.0, -180, 180, 1, "deg"), P("mirror", "choice", "no", choices=("no", "yes"))],
    ("local_fix", "points"): [P("spot", "str", "[]"), P("action", "choice", "smooth", choices=("smooth", "fill")), P("shape", "choice", "cylinder", choices=("cylinder", "box")),
                              P("diameter", "float", 20.0, 1, 300, 0.5, "mm"), P("width", "float", 20.0, 1, 300, 0.5, "mm"), P("height", "float", 20.0, 1, 300, 0.5, "mm"),
                              P("depth", "float", 6.0, 0.5, 100, 0.5, "mm"), P("feather", "float", 4.0, 0, 50, 0.5, "mm"),
                              P("strength", "float", 0.8, 0.1, 1.0, 0.05, ""), P("passes", "int", 8, 1, 50, 1, ""),
                              P("reach", "float", 2.5, 0.2, 15, 0.1, "mm")],
    ("local_fix", "mesh"): [P("spot", "str", "[]"), P("action", "choice", "smooth", choices=("smooth", "fill")), P("shape", "choice", "cylinder", choices=("cylinder", "box")),
                              P("diameter", "float", 20.0, 1, 300, 0.5, "mm"), P("width", "float", 20.0, 1, 300, 0.5, "mm"), P("height", "float", 20.0, 1, 300, 0.5, "mm"),
                              P("depth", "float", 6.0, 0.5, 100, 0.5, "mm"), P("feather", "float", 4.0, 0, 50, 0.5, "mm"),
                              P("strength", "float", 0.8, 0.1, 1.0, 0.05, ""), P("passes", "int", 8, 1, 50, 1, ""),
                              P("reach", "float", 2.5, 0.2, 15, 0.1, "mm")],
    ("mirror", "points"): [P("nx", "float", 0.0, -1, 1, 0.001, ""), P("ny", "float", 0.0, -1, 1, 0.001, ""), P("nz", "float", 1.0, -1, 1, 0.001, ""),
                           P("d", "float", 0.0, -100000, 100000, 0.01, "mm"), P("fit", "choice", "yes", choices=("yes", "no")),
                           P("gap_mm", "float", 1.5, 0.2, 20, 0.1, "mm", help="a mirrored point is added only where the original has nothing within this distance")],
    ("mirror", "mesh"): [P("nx", "float", 0.0, -1, 1, 0.001, ""), P("ny", "float", 0.0, -1, 1, 0.001, ""), P("nz", "float", 1.0, -1, 1, 0.001, ""),
                         P("d", "float", 0.0, -100000, 100000, 0.01, "mm"), P("fit", "choice", "yes", choices=("yes", "no")),
                         P("gap_mm", "float", 1.5, 0.2, 20, 0.1, "mm", help="a mirrored face is added only where the original has nothing within this distance")],
    ("flip", "placement"): [],
    ("match", "placement"): [],
    ("turn", "placement"): [],
    ("nudge", "transforms"): [],
}
TOOLS_BY_KIND = {"points": ["isolation", "overlap", "smooth", "match_cleaned_sides", "cut_plane", "clone", "local_fix", "mirror", "delete"],
                 "mesh": ["isolation", "match_cleaned_sides", "fill_holes", "simplify", "smooth", "slivers", "finish", "fin", "slice_gap", "drill", "cut_plane", "clone", "local_fix", "mirror", "delete"]}
DETECT_TOOLS = ("isolation", "overlap", "delete")

QUALITY_GRID = {1: 0.80, 2: 0.60, 3: 0.45, 4: 0.33, 5: 0.25, 6: 0.18, 7: 0.11}     # nominal only (the old table)
# Quality -> (octree depth offset from the Q6 depth, Poisson octree scale). Poisson's cell is extent x scale / 2^depth, so a
# depth step is 2x and scale 1.1 -> 1.1 x sqrt(2) adds the half step: neighbouring levels always differ (unless the
# depth limits clamp them, which the plan reports). 1.1 is PyMeshLab's and Open3D's own default scale.
POISSON_SCALE = 1.1
QUALITY_STEPS = {7: (1, 1.1), 6: (0, 1.1), 5: (0, 1.1 * math.sqrt(2)), 4: (-1, 1.1), 3: (-1, 1.1 * math.sqrt(2)),
                 2: (-2, 1.1), 1: (-2, 1.1 * math.sqrt(2))}
DEPTH_MIN, DEPTH_MAX = 6, 11


def poisson_plan(extent_mm, quality):
    """-> {"quality", "depth", "scale", "grid_mm" (the cell Poisson really uses on this part), "same_as" (the next finer
    level that gives the same depth and scale on this part, or None)}."""
    ext = max(float(extent_mm), 1e-6)

    def one(q):
        d6 = int(min(DEPTH_MAX, max(DEPTH_MIN, round(math.log2(ext / QUALITY_GRID[6])))))
        off, sc = QUALITY_STEPS[int(q)]
        d = int(min(DEPTH_MAX, max(DEPTH_MIN, d6 + off)))
        return d, round(sc, 6)
    q = int(quality); d, sc = one(q)
    same = None
    for q2 in range(q + 1, 8):                 # the nearest finer level that comes out identical on this part
        if one(q2) == (d, sc): same = q2; break
    return {"quality": q, "depth": d, "scale": sc, "grid_mm": round(ext * sc / 2 ** d, 4), "same_as": same}


def poisson_depth(extent_mm, quality):
    """(depth, grid mm really used) of a Quality on a part of this extent."""
    pl = poisson_plan(extent_mm, quality)
    return pl["depth"], pl["grid_mm"]


def poisson_label(plan):
    s = "Q%d: depth %d, grid %s mm" % (plan["quality"], plan["depth"], fmt(plan["grid_mm"]))
    if plan.get("same_as"): s += " (same as Q%d on this part)" % plan["same_as"]
    return s


def overlap_schedule(p):
    """(sig_n list, dedupe cell) of the overlap tool."""
    if p.get("schedule"):
        sch = tuple(float(x) for x in str(p["schedule"]).split(",") if x.strip())
        return sch, round(0.25 * float(p["distance"]) / 0.21, 9)
    s = float(p["distance"]) / 0.21
    return tuple(round(s * f, 9) for f in (1.0, 0.7, 0.5)), round(s * 0.25, 9)


def mesh_smooth_lambda(strength):
    return float(min(0.9, max(0.05, 0.5 * float(strength) / 30.0)))


# ---------------------------------------------------------------------------------------------------------- recipes
def node_params(node, method=None):
    return list(COMMON.get(node, [])) + list(METHOD_PARAMS.get((node, method), []))


def find_param(node, name, method=None):
    for p in COMMON.get(node, []):
        if p.name == name: return p, None
    if method is not None:
        for p in METHOD_PARAMS.get((node, method), []):
            if p.name == name: return p, method
    for m in METHODS.get(node, []):
        for p in METHOD_PARAMS.get((node, m), []):
            if p.name == name: return p, m
    return None, None


def tool_params(tool, kind):
    return list(TOOL_PARAMS.get((tool, kind), []))


def check_tool_params(tool, kind, given):
    """Every given key must be one of the tool's params (a typo like 'strenght' is an error, not silently dropped)."""
    names = [p.name for p in tool_params(tool, kind)]
    bad = [k for k in (given or {}) if k not in names]
    if bad:
        raise ValueError("%s (%s) has no param %s (params: %s)" % (tool, kind, ", ".join(repr(b) for b in bad), ", ".join(names) or "none"))


def resolve_tool(tool, kind, given):
    check_tool_params(tool, kind, given)
    out = {}
    for p in tool_params(tool, kind):
        out[p.name] = p.coerce(given[p.name]) if given and p.name in given else p.default
    return out


def parse_path(path):
    """-> dict(node, method, name, side). name 'method' = the node's method."""
    side = None
    s = path.strip()
    if s.startswith("side[") and "]" in s:
        side = s[5:s.index("]")]; s = "side" + s[s.index("]") + 1:]
    nodes = sorted(METHODS, key=len, reverse=True)
    node = next((n for n in nodes if s == n or s.startswith(n + ".")), None)
    if node is None: raise ValueError("unknown param path %r (try `list --params`)" % path)
    rest = s[len(node) + 1:].split(".") if s != node else []
    if not rest or rest == [""]: raise ValueError("param path %r names a stage, not a param" % path)
    if rest == ["method"]:
        if side is not None: raise ValueError("per-side method overrides are set with `platter`")
        return dict(node=node, method=None, name="method", side=None)
    if len(rest) == 2:
        m, name = rest
        if m not in METHODS[node]: raise ValueError("%s has no method %r (methods: %s)" % (node, m, ", ".join(METHODS[node])))
        p, _ = find_param(node, name, m)
        if p is None or all(q.name != name for q in METHOD_PARAMS.get((node, m), [])):
            raise ValueError("%s.%s has no param %r" % (node, m, name))
        return dict(node=node, method=m, name=name, side=side)
    if len(rest) == 1:
        p, m = find_param(node, rest[0])
        if p is None: raise ValueError("%s has no param %r" % (node, rest[0]))
        if side is not None and m is not None: raise ValueError("per-side overrides are for common side params only")
        return dict(node=node, method=m, name=rest[0], side=side)
    raise ValueError("bad param path %r" % path)


def side_path(path, sid):
    """The same param path with its side token replaced by the side id (side[2].keep.min_share -> side[<id>]...)."""
    s = path.strip()
    if not s.startswith("side[") or "]" not in s: return s
    return "side[%s]%s" % (sid, s[s.index("]") + 1:])


def param_of(path):
    d = parse_path(path)
    if d["name"] == "method":
        return Param("method", "choice", METHODS[d["node"]][0], choices=tuple(METHODS[d["node"]])), d
    p, _ = find_param(d["node"], d["name"], d["method"])
    return p, d


def _slot(recipe, d, create):
    """The dict the value lives in, and the key."""
    if d["side"] is not None:
        sp = recipe.setdefault("sides", {}).setdefault("params", {}) if create else recipe.get("sides", {}).get("params", {})
        e = sp.setdefault(d["side"], {}) if create else sp.get(d["side"], {})
        e = e.setdefault(d["node"], {}) if create else e.get(d["node"], {})
        return e, d["name"]
    pr = recipe.setdefault("params", {}) if create else recipe.get("params", {})
    e = pr.setdefault(d["node"], {}) if create else pr.get(d["node"], {})
    if d["method"] is not None and d["name"] != "method":
        e = e.setdefault(d["method"], {}) if create else e.get(d["method"], {})
    return e, d["name"]


def get_path(recipe, path):
    p, d = param_of(path)
    e, k = _slot(recipe, d, False)
    if k in e: return e[k]
    if d["side"] is not None:                          # a per-side override falls back to the global value
        return get_path(recipe, "%s.%s" % (d["node"], d["name"]))
    return p.default


def set_path(recipe, path, value):
    """Set (sparse: a value equal to the default is removed). value 'revo' = the Revo value. Returns (old, new)."""
    p, d = param_of(path)
    old = get_path(recipe, path)
    if isinstance(value, str) and value.strip().lower() == "revo" and p.type != "str":
        if p.revo is None: raise ValueError("%s has no known Revo value" % path)
        value = p.revo
    v = p.coerce(value)
    e, k = _slot(recipe, d, True)
    e.pop(k, None)
    base = get_path(recipe, "%s.%s" % (d["node"], d["name"])) if d["side"] is not None else p.default
    if v == base: pass
    else: e[k] = v
    prune(recipe)
    return old, v


def reset_path(recipe, path):
    p, d = param_of(path)
    old = get_path(recipe, path)
    e, k = _slot(recipe, d, True); e.pop(k, None); prune(recipe)
    return old, get_path(recipe, path)


def prune(recipe):
    """Drop empty dicts in params / sides.params so equal recipes are byte-equal."""
    def walk(x):
        if isinstance(x, dict):
            for k in list(x):
                walk(x[k])
                if isinstance(x[k], dict) and not x[k]: del x[k]
    walk(recipe.get("params", {}))
    sp = recipe.get("sides", {}).get("params")
    if sp is not None: walk(sp)


def method_of(recipe, node, with_params=None):
    wp = with_params or {}
    if node + ".method" in wp: return param_of(node + ".method")[0].coerce(wp[node + ".method"])
    return recipe.get("params", {}).get(node, {}).get("method", METHODS[node][0])


def resolve(recipe, node, method, side=None, with_params=None):
    """Resolved params of a node for its method: common + the method's own, per-side overrides applied."""
    pr = recipe.get("params", {}).get(node, {})
    sp = recipe.get("sides", {}).get("params", {}).get(side, {}).get(node, {}) if side else {}
    wp = with_params or {}
    out = {}
    for p in COMMON.get(node, []):
        v = pr.get(p.name, p.default)
        if p.name in sp: v = sp[p.name]
        k = node + "." + p.name
        if k in wp: v = wp[k]
        if side and "side[%s].%s" % (side, k[5:]) in wp: v = wp["side[%s].%s" % (side, k[5:])]
        out[p.name] = p.coerce(v)
    mp = pr.get(method, {}) if isinstance(pr.get(method), dict) else {}
    for p in METHOD_PARAMS.get((node, method), []):
        v = mp.get(p.name, p.default)
        k = "%s.%s.%s" % (node, method, p.name)
        if k in wp: v = wp[k]
        out[p.name] = p.coerce(v)
    return out


def all_paths():
    """Every settable path with its Param (for `list --params`)."""
    out = []
    for node, ms in METHODS.items():
        if len(ms) > 1: out.append((node + ".method", Param("method", "choice", ms[0], choices=tuple(ms))))
        for p in COMMON.get(node, []): out.append(("%s.%s" % (node, p.name), p))
        for m in ms:
            for p in METHOD_PARAMS.get((node, m), []): out.append(("%s.%s.%s" % (node, m, p.name), p))
    return out


# ---------------------------------------------------------------------------------------------------------- presets
PRESETS = {
    "today": {"note": "build_project.py today: fused tsdf 0.4 mm, 300 frames/side, meshed as fused, isolation 3 %",
              "set": {"side.platter.method": "plate", "fused.method": "tsdf", "fused.point_distance": 0.4, "fused.frames_per_side": 300, "meshed.method": "auto"},
              "reset_nodes": ["fused", "meshed"]},
    "revo": {"note": "the bake's revo_points: points 0.4 mm, isolation 1 %, overlap 0.21 mm, smooth 10 x2, Poisson Q6",
             "set": {"fused.points.min_sides": 2, "side.platter.method": "plate", "fused.method": "points", "fused.point_distance": 0.4, "fused.points.step": 2, "fused.frames_per_side": 300,
                     "meshed.method": "auto", "meshed.poisson.quality": 6, "meshed.poisson.trim": 0.5, "meshed.poisson.taubin": 3},
             "reset_nodes": ["fused", "meshed"]},
    "revo-device": {"note": "Revo's own numbers wherever a mapping exists (for A/B, not a default): points 0.4, isolation 15 %, overlap 0.21, smooth 10 x10, Poisson Q7",
                    "set": {"fused.points.min_sides": 2, "side.platter.method": "plate", "fused.method": "points", "fused.point_distance": 0.4, "fused.frames_per_side": 300,
                            "meshed.method": "auto", "meshed.poisson.quality": 7},
                    "reset_nodes": ["fused", "meshed"],
                    "ops": {"fused.points": [{"tool": "isolation", "params": {"rate": 15.0}},
                                             {"tool": "overlap", "params": {"distance": 0.21}},
                                             {"tool": "smooth", "params": {"strength": 10.0, "times": 10}}]}},
    "plus": {"note": "the bake-off's tsdf_plus: fused plus 0.4 mm, 300 frames/side, blur on 3+ sides, enclosed surface dropped (0.3 mm ran a 10 GB GPU out of memory on 1,800 frames)",
             "set": {"side.platter.method": "plate", "fused.method": "plus", "fused.point_distance": 0.4, "fused.frames_per_side": 300, "meshed.method": "auto"},
             "reset_nodes": ["fused", "meshed"]},
    "print": {"note": "Deep Clean points, then Finish for print at the triangle target",
              "set": {"side.platter.method": "plate", "fused.method": "points", "fused.points.min_sides": 2,
                      "fused.point_distance": 0.4, "fused.frames_per_side": 300, "meshed.method": "auto", "meshed.poisson.quality": 7},
              "reset_nodes": ["fused", "meshed"],
              "ops": {"fused.points": [{"tool": "isolation", "params": {"rate": 15.0}},
                                       {"tool": "overlap", "params": {"distance": 0.21}},
                                       {"tool": "smooth", "params": {"strength": 10.0, "times": 10}}],
                      "meshed": [{"tool": "finish", "params": {}}]}},
    "quick": {"note": "a look mid-scan: fused tsdf 0.8 mm, 100 frames/side (side nodes stay cached)",
              "set": {"side.platter.method": "plate", "fused.method": "tsdf", "fused.point_distance": 0.8, "fused.frames_per_side": 100, "meshed.method": "auto"},
              "reset_nodes": ["fused", "meshed"]},
}


def promote_fused_cuts(recipe):
    """Move the active method's older manual cuts into the shared Fused edit list.
    Parked methods stay parked until selected; their cuts must not become active unexpectedly.
    Call on a private recipe copy. Keep IDs, enabled states, outlines, and recorded metadata intact.
    """
    ops = recipe.setdefault("ops", {})
    key = "fused." + method_of(recipe, "fused")
    cuts = [o for o in ops.get(key, []) if o.get("tool") == "cut_plane"]
    if cuts:
        ops[key] = [o for o in ops[key] if o.get("tool") != "cut_plane"]
        ops.setdefault("fused", []).extend(cuts)


def apply_preset(recipe, name, fragment=None):
    """One preset = the params AND the tool lists of the nodes it resets: a node's params go back to the code defaults
    before the preset's own values are set, and its tool lists are dropped (a fused method then uses its default_ops)
    unless the preset writes them. Manual deletes and base cuts are retained edits: the shared Fused list and the
    delete ops in the meshed list stay, and a delete still sitting in a fused.<method> tool list (made before the Fused
    stage had its own list) moves to the end of that list instead of being dropped with the tools."""
    pr = PRESETS[name] if fragment is None else fragment
    r = copy.deepcopy(recipe)
    promote_fused_cuts(r)
    ops = r.setdefault("ops", {})
    geometry = [o for o in ops.get("meshed", []) if o.get("tool") in ("fin", "slice_gap", "drill", "cut_plane")] if "meshed" in pr.get("reset_nodes", []) else []
    for n in pr.get("reset_nodes", []):
        r.setdefault("params", {}).pop(n, None)
        if n == "fused":
            moved = [o for lk in sorted(k for k in ops if k.startswith("fused.")) for o in ops[lk] if o.get("tool") == "delete"]
            if moved: ops["fused"] = list(ops.get("fused", [])) + moved
            for lk in [k for k in ops if k.startswith("fused.")]:
                parked_cuts = [o for o in ops[lk] if o.get("tool") == "cut_plane"]
                if parked_cuts: ops[lk] = parked_cuts
                else: del ops[lk]
        elif n in ops:
            keep = [o for o in ops[n] if o.get("tool") == "delete"]
            if keep: ops[n] = keep
            else: del ops[n]
    for k, v in pr["set"].items(): set_path(r, k, v)
    for lk, ops in pr.get("ops", {}).items():
        cuts = [o for o in r.get("ops", {}).get(lk, []) if o.get("tool") == "cut_plane"]
        lst = [o for o in r.get("ops", {}).get(lk, []) if o.get("tool") == "delete"]     # kept manual deletes go first
        for o in ops:
            r["next_op"] = int(r.get("next_op", 1))
            lst.append({"id": "op-%04d" % r["next_op"], "tool": o["tool"], "on": True, "params": dict(o["params"])})
            r["next_op"] += 1
        r.setdefault("ops", {})[lk] = lst + cuts
    if geometry:
        r.setdefault("ops", {}).setdefault("meshed", []).extend(geometry)
    r["preset"] = name
    promote_fused_cuts(r)
    prune(r)
    return r
