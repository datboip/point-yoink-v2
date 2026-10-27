# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scanner's helpers that need no window: Auto range, Auto gain, the platter the build will cut, how much of the
part is well covered, the platter dial, and the hand-turned model picture (image mode). Ported from
scanner/pointyoink_scanner.py with the same numbers; the session calls them on its worker thread (well_covered and the
orbit touch the tracker's model)."""
import numpy as np

W, H = 800, 600
RANGE_STEP_MM = 5     # The existing Range handle step and Auto range rounding.


class AutoRange:
    """Keep the Near/Far window wrapped around the part: the nearest big mass of depth in the middle of the view (the
    part and the platter under it), with room in front and behind. Moves smoothly, like the scanner's own Auto."""
    def __init__(self): self.rng = None; self.wide = 0; self.tight = 0

    def connected(self, span):
        """Full supported part span plus measured noise margin, rounded outward.

        Expansion is immediate so no connected endpoint is lost while hysteresis
        waits. Contraction uses the existing thirty-frame hold.
        """
        import math
        lo,hi=span;lo=max(0,math.floor(lo/RANGE_STEP_MM)*RANGE_STEP_MM);hi=math.ceil(hi/RANGE_STEP_MM)*RANGE_STEP_MM
        if self.rng is None or lo<self.rng[0] or hi>self.rng[1]:
            self.rng=(lo,hi) if self.rng is None else (min(lo,self.rng[0]),max(hi,self.rng[1]));self.tight=0
        else:
            self.tight+=1
            if self.tight<30:return None
            self.rng=(lo,hi);self.tight=0
        return tuple(map(int,self.rng))

    def update(self, d):
        """d: the uncut depth frame. Returns the new (near, far) in mm, rounded to 5, or None to stay put."""
        c = d[H // 5: H * 4 // 5, W // 5: W * 4 // 5]
        z = c[c > 0].astype(np.float32) * 0.1
        if z.size < 2000: return None
        hist, edges = np.histogram(z, bins=np.arange(50, 1000, 10))
        big = np.nonzero(hist > 0.05 * hist.max())[0]
        if not len(big): return None
        z_front = edges[big[0]]                                   # the nearest well-populated 10 mm layer
        body = z[(z >= z_front) & (z <= z_front + 250)]            # the part and its platter lie within 25 cm of that
        lo = max(50.0, float(np.percentile(body, 1)) - 40); hi = float(np.percentile(body, 99)) + 40
        # stay put unless the part has really left the window: widening waits 3 frames (the part is being cut off),
        # narrowing waits 30 and needs 30 mm to gain. Moving every frame made the cut edge, and the saved frames, flicker
        cur = self.rng
        if cur is None: self.rng = (lo, hi); self.wide = self.tight = 0
        else:
            wide = lo < cur[0] - 10 or hi > cur[1] + 10; tight = not wide and (lo > cur[0] + 30 or hi < cur[1] - 30)
            self.wide = self.wide + 1 if wide else 0
            self.tight = self.tight + 1 if tight else 0
            if self.wide >= 3: self.rng = (min(lo, cur[0]), max(hi, cur[1])); self.wide = 0
            elif self.tight >= 30: self.rng = (lo, hi); self.tight = 0
            else: return None
        return int(round(self.rng[0] / RANGE_STEP_MM) * RANGE_STEP_MM), int(round(self.rng[1] / RANGE_STEP_MM) * RANGE_STEP_MM)


def gain_order(g0):
    """The gains Auto gain tries, one per full turn: up first, then down, further out each time (dark plastic wants a
    high gain, shiny spots a low one). 12 is the top: 15 clips the IR and loses depth again."""
    out = []
    for g in (g0, g0 + 3, g0 - 2, g0 + 6, g0 - 3, g0 + 8):
        g = min(12, max(1, g))
        if g not in out: out.append(g)
    return out


def well_covered(trk, pl):
    """How many of the PART's points have been seen often enough (the blue ones in the model view). With the side's
    platter known, only points 2 mm or more above it and inside its circle count: otherwise a gain that lights up more of
    the platter or the background looked like progress. pl: the side's platter (n, off, centre, radius, source) or None."""
    if trk is None or getattr(trk, "model", None) is None: return 0
    Pm = trk.model.point.positions.cpu().numpy(); keep = trk.coverage_levels(Pm) == 2
    if pl is not None:
        P = Pm * 1000.0; n, off = pl[0], pl[1]                  # tracker metres -> the side's scan frame in mm
        keep &= P @ n + off > 2.0
        if pl[2] is not None:
            if hasattr(pl[3], "sdf"): keep &= pl[3].sdf(P, n, pl[2]) < 10.0       # a bed that is not a circle about the centre
            else: h = P - pl[2]; keep &= np.linalg.norm(h - np.outer(h @ n, n), axis=1) < pl[3] + 10.0
    return int(keep.sum())


class AutoGain:
    """Look at how much well-covered surface the last stretch of the turn added and pick the gain from it, so each gain picks up
    what the others could not. Quick (the default): a stretch is `window` degrees (45), the gain moves on as soon as a stretch
    adds next to nothing (the blue at this gain is saturated) and stays while it still adds, the order wraps around, and it is done
    when every gain in a row added nothing. Not quick: a stretch is a full turn and every turn takes the next gain (the first
    version). tick() returns (new gain or None, log line or None); `turn` also carries the dial's full-turn marks."""
    FLAT = 0.03                                               # a stretch that adds under 3% of what is covered adds nothing

    def __init__(self, quick=True, window=45.0): self.turn = None; self.msg = ""; self.quick = quick; self.window = window if quick else 360.0

    @staticmethod
    def initial(ir,gain):
        """Choose from observed IR brightness before the first tracked turn."""
        values=np.asarray(ir).ravel();values=values[values>0]
        if not len(values): return None
        level=float(np.percentile(values,75))
        return int(np.clip(round(gain*128/max(level,1)),1,15))

    def reset(self): self.turn = None; self.msg = ""; self.bright_t = None; self.up_from = None

    CLIP_MAX = 0.02      # more than 2% of the part's IR pixels blown out: a step down
    DARK_MAX = 0.10      # more than 10% of the part dark with no depth (and no clipping to speak of): a step up
    HOLD_S = 2.0         # at most one step every two seconds: a gain change shifts the IR picture under the tracker

    def brightness(self, dark_frac, clip_frac, gain, now):
        """One gain step from what the part's IR shows now, or None. Replaces waiting out 45 degree stretches of the
        turn: that stepper changed gain by whole presets at a time and kept the user waiting a turn per try."""
        lo, hi = getattr(self, 'bounds', (1, 15))
        clip_max = getattr(self, 'clip_max', self.CLIP_MAX); dark_max = getattr(self, 'dark_max', self.DARK_MAX)   # the Surface preset's rule
        last = getattr(self, 'bright_t', None)
        if last is not None and now - last < self.HOLD_S: return None
        up_from = getattr(self, 'up_from', None)                 # the dark share before the last step up
        if clip_frac > clip_max and gain > lo: g = gain - 1; why = "%d%% of the part is blown out" % round(100 * clip_frac); self.up_from = None
        elif dark_frac > dark_max and clip_frac < clip_max / 4 and gain < hi:
            if up_from is not None and dark_frac > up_from - 0.03:   # the last step up changed nothing: the dark is missing depth, not darkness
                self.msg = "Auto gain: gain %d, more brightness did not help" % gain; return None
            g = gain + 1; why = "%d%% of the part is dark with no depth" % round(100 * dark_frac); self.up_from = dark_frac
        else:
            self.msg = "Auto gain: gain %d" % gain; self.up_from = None; return None
        self.bright_t = now; self.msg = "Auto gain: %d, %s" % (g, why); return g

    def tick(self, side_key, turned_deg, gain, count_fn):
        if turned_deg is None: return None, None
        t = self.turn
        if t is None or t["side"] is not side_key:
            self.turn = dict(side=side_key, mark=turned_deg, wmark=turned_deg, count=count_fn(), order=[g for g in gain_order(gain) if getattr(self,'bounds',(1,15))[0]<=g<=getattr(self,'bounds',(1,15))[1]] or [gain], i=0, flat=0, done=False, n=0)
            self.msg = "Auto gain: gain %d" % gain; return None, None
        if abs(turned_deg - t["mark"]) >= 360.0: t["n"] += 1; t["mark"] = turned_deg          # the dial's turn number
        if t["done"] or abs(turned_deg - t["wmark"]) < self.window: return None, None
        now = count_fn(); added = now - t["count"]
        t["wmark"] = turned_deg; t["count"] = now
        flat = added < self.FLAT * max(now, 1)
        t["flat"] = t["flat"] + 1 if flat else 0
        pct = round(100 * added / max(now, 1))
        line = "auto gain: %d degrees at gain %d added %d well-covered points (now %d)" % (round(self.window), gain, added, now)
        if not self.quick:
            if t["flat"] >= 2 or t["i"] + 1 >= len(t["order"]):
                t["done"] = True; self.msg = "Auto gain: done, more turns would not help"; return None, line
            t["i"] += 1; g = t["order"][t["i"]]; self.msg = "Auto gain: turn %d, gain %d (last +%d%%)" % (t["n"] + 1, g, pct)
            return g, line
        if t["flat"] >= len(t["order"]):
            t["done"] = True; self.msg = "Auto gain: done, no brightness helps more"; return None, line
        if not flat: self.msg = "Auto gain: gain %d (+%d%% last %d\u00b0)" % (gain, pct, round(self.window)); return None, line
        t["i"] = (t["i"] + 1) % len(t["order"]); g = t["order"][t["i"]]
        self.msg = "Auto gain: %d is as good as it gets, trying %d" % (gain, g)
        return g, line


def dial(turned_deg, autogain, side_key):
    """The platter dial: (degrees into this turn, turn number from 1). Auto gain's own turns when it runs on this side
    (its turns start where it measured), else plain full turns since the side began. None before the spin axis is known."""
    if turned_deg is None: return None
    t = autogain.turn if autogain is not None else None
    if t is not None and t["side"] is side_key:
        return max(0.0, turned_deg - t["mark"]) % 360.0, t["n"] + 1
    return turned_deg % 360.0, int(turned_deg // 360.0) + 1


def auto_platter(sideplane, cached=False):
    """Auto remove base's platter (n, off, centre, radius, "auto") in the side's scan frame, or None."""
    p = sideplane.cached_plane() if cached else getattr(sideplane, 'legacy_plane', sideplane.plane)()
    if p is None: return None
    dc = sideplane.cached_disc() if cached else sideplane.disc()
    return p[0], p[1], None if dc is None else dc[0], None if dc is None else dc[1], "auto"


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis); x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Orbit:
    """Image mode: the model picture turned and zoomed by hand around the model's middle; Home drops it (back to the
    scanner's view). Drag: 0.01 rad per pixel of the Tk panel; wheel: 10 mm per click. Tracker metres."""
    def __init__(self, T0, c): self.T0 = np.asarray(T0, float).copy(); self.c = np.asarray(c, float); self.yaw = self.pitch = self.zoom = 0.0

    def drag(self, dx, dy): self.yaw -= dx * 0.01; self.pitch += dy * 0.01

    def pose(self):
        T0, c = self.T0, self.c
        R = rot(T0[:3, 1], self.yaw) @ rot(T0[:3, 0], self.pitch)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c - R @ c
        Z = np.eye(4); Z[2, 3] = self.zoom; return M @ T0 @ Z
