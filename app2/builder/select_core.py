# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build step selection: numpy only, no Qt, no GL. The reference for gpu_select and its fallback.

The engine deletes ELEMENTS: points of a cloud, or faces of a mesh by their centroid (stages/tools/edits.py elements()).
So selection works on exactly those positions, and its result is a bool per element in the engine's element order,
which is also the view's flag order (face i = gl_PrimitiveID i).

Elements keeps them sorted along a Morton curve in chunks of CHUNK, each with a bounding box, so a rectangle / lasso /
sphere / box first throws away whole chunks and only tests the elements of chunks it touches. Chunks wholly inside a
"through the model" lasso are taken without testing a single element.

Screen selection, two ways (Revo's "Select Through" toggle):
  through   every element whose centroid projects inside the mask
  visible   ... and is on the surface as drawn (the depth buffer of the frame on screen), and faces the eye when it
            has a normal. "On the surface": its eye depth within tol of its own pixel's depth plus the local slope (how
            much nearer the nearest surface in the 3x3 around is), or within tol of that 3x3 nearest surface plus its own
            slope along the ray (from its normal) times 1.5 px (and a point's sprite reach). A flat wall 1.5 mm behind
            another never passes; a steep front surface does (comparing with the 3x3 nearest alone missed ~80% of a
            sloped wall). Sub-pixel faces of a 10 M face mesh are still found (the centroid test does not need the face
            to own a pixel).
"""
import numpy as np

CHUNK = 2048
BIG = np.float32(3e38)         # "no surface" depth (background)


def morton_order(X, bits=10):
    lo, hi = X.min(0), X.max(0); s = np.float32(2 ** bits - 1) / np.maximum(hi - lo, 1e-9).astype(np.float32)
    q = ((X - lo) * s).astype(np.uint32)

    def spread(v):                                       # 10 bits -> every third bit of 30 (uint32 is enough)
        v = (v | (v << np.uint32(16))) & np.uint32(0x030000FF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x0300F00F)
        v = (v | (v << np.uint32(4))) & np.uint32(0x030C30C3)
        v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
        return v
    code = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint32(1)) | (spread(q[:, 2]) << np.uint32(2))
    return np.argsort(code)


def centroids(V, F):
    """Face centroids (the engine's delete elements) and unit face normals, float32, engine face order."""
    V = np.asarray(V, np.float32); F = np.asarray(F)
    A = V[F[:, 0]]; B = V[F[:, 1]]; C = V[F[:, 2]]
    X = (A + B + C) / np.float32(3.0)
    B -= A; C -= A; n = np.cross(B, C); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), np.float32(1e-12))
    return X, n


class Elements:
    """X (n,3) element positions (points, or face centroids), Nrm (n,3) unit normals or None."""

    def __init__(self, X, Nrm=None):
        X = np.asarray(X, np.float32); self.n = len(X)
        self.perm = morton_order(X).astype(np.int64)                   # sorted slot -> element index
        self.X = X[self.perm]
        self.Nq = None if Nrm is None else np.clip(np.round(np.asarray(Nrm, np.float32)[self.perm] * 127), -127, 127).astype(np.int8)
        nc = (self.n + CHUNK - 1) // CHUNK; self.nc = nc
        pad = nc * CHUNK - self.n
        Xp = np.concatenate([self.X, np.repeat(self.X[-1:], pad, 0)]) if pad else self.X
        B = Xp.reshape(nc, CHUNK, 3); self.lo = B.min(1); self.hi = B.max(1)

    @staticmethod
    def of_mesh(V, F):
        return Elements(*centroids(V, F))

    # ---------------- helpers ----------------
    def _slots(self, chunks):
        """Chunk ids -> the sorted slots they hold."""
        if not len(chunks): return np.zeros(0, np.int64)
        s = (chunks[:, None] * CHUNK + np.arange(CHUNK)[None]).ravel()
        return s[s < self.n]

    def _corners(self, ch):
        lo, hi = self.lo[ch], self.hi[ch]
        return np.stack([np.where(np.array([i & 1, i & 2, i & 4], bool), hi, lo) for i in range(8)], 1)   # (k,8,3)

    def to_elements(self, slots):
        m = np.zeros(self.n, bool); m[self.perm[slots]] = True; return m

    # ---------------- screen: rectangle / lasso ----------------
    def select_screen(self, mask, view, proj, near, far, depth=None, through=False, tol_mm=0.35, tol_px=1.5, facing=True, reach_mm=0.0):
        """mask (h,w) bool (a filled rectangle or lasso, device pixels); view world -> eye, proj eye -> clip (the frame
        on screen, Camera.view / Camera.projection); depth: that frame's depth image (h,w, 1.0 = background), needed
        unless through; reach_mm: how far a drawn element covers its neighbours (a point's sprite size; 0 for faces).
        -> bool per element (engine order)."""
        h, w = mask.shape; M = np.asarray(proj, float) @ np.asarray(view, float)
        orthographic = abs(float(proj[3][3]) - 1.) < 1e-8
        ys, xs = np.nonzero(mask)
        if not len(xs): return np.zeros(self.n, bool)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        # chunks: project their boxes' corners; keep those whose screen box meets the mask's box
        C = self._corners(np.arange(self.nc)).reshape(-1, 3)
        q = C @ M[:3, :3].T + M[:3, 3]; wq = C @ M[3, :3] + M[3, 3]
        behind = (wq <= 1e-9).reshape(self.nc, 8).any(1)
        wq = np.where(wq <= 1e-9, 1e-9, wq)
        sx = ((q[:, 0] / wq * 0.5 + 0.5) * w).reshape(self.nc, 8); sy = ((0.5 - q[:, 1] / wq * 0.5) * h).reshape(self.nc, 8)
        bx0, bx1, by0, by1 = sx.min(1), sx.max(1), sy.min(1), sy.max(1)
        touch = behind | ((bx1 >= x0) & (bx0 <= x1) & (by1 >= y0) & (by0 <= y1))
        cand_chunks = np.nonzero(touch)[0]
        whole = np.zeros(0, np.int64)
        if through and len(cand_chunks) and not orthographic:
            # a chunk whose screen box lies wholly in the mask is taken without testing its elements (summed-area table)
            S = np.zeros((h + 1, w + 1), np.int64); S[1:, 1:] = mask.cumsum(0).cumsum(1)
            ok = ~behind[cand_chunks]
            ix0 = np.clip(np.floor(bx0[cand_chunks]), 0, w).astype(int); ix1 = np.clip(np.ceil(bx1[cand_chunks]) + 1, 0, w).astype(int)
            iy0 = np.clip(np.floor(by0[cand_chunks]), 0, h).astype(int); iy1 = np.clip(np.ceil(by1[cand_chunks]) + 1, 0, h).astype(int)
            inside = (bx0[cand_chunks] >= 0) & (bx1[cand_chunks] < w - 1) & (by0[cand_chunks] >= 0) & (by1[cand_chunks] < h - 1)
            area = (ix1 - ix0) * (iy1 - iy0)
            cnt = S[iy1, ix1] - S[iy0, ix1] - S[iy1, ix0] + S[iy0, ix0]
            full = ok & inside & (cnt == area) & (area > 0)
            whole = cand_chunks[full]; cand_chunks = cand_chunks[~full]
        sl = self._slots(cand_chunks); X = self.X[sl]
        q = X @ M[:3, :3].T.astype(np.float32) + M[:3, 3].astype(np.float32); wq = X @ M[3, :3].astype(np.float32) + np.float32(M[3, 3])
        eye_z = -(X @ np.asarray(view)[2, :3] + np.asarray(view)[2, 3]) if orthographic else wq
        good = (eye_z > near) & (eye_z < far) if orthographic else wq > 1e-6
        x = (q[:, 0] / wq * 0.5 + 0.5) * w; y = (0.5 - q[:, 1] / wq * 0.5) * h
        xi = np.floor(x).astype(np.int32); yi = np.floor(y).astype(np.int32)       # floor, as the GPU path
        good &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        k = np.nonzero(good)[0]; k = k[mask[yi[k], xi[k]]]
        if not through:
            if depth is None: raise ValueError("visible-only selection needs the depth image")
            Z = self.ref_image(depth, near, far, orthographic=orthographic)                               # (h,w,2): slope-aware limit, 3x3 nearest
            ze = eye_z[k]                                                         # eye distance (perspective w)
            px_mm = ze * np.float32(2.0 / (float(proj[1][1]) * h))             # one pixel's size at that distance
            if orthographic: px_mm = np.full(len(k), 2.0 / (float(proj[1][1]) * h))
            zz = Z[yi[k], xi[k]]; lim = zz[:, 0]
            if self.Nq is not None:
                eye = np.linalg.inv(view)[:3, 3].astype(np.float32)
                nk = self.Nq[sl[k]].astype(np.float32); d = eye - X[k]
                if orthographic: d = np.broadcast_to(np.linalg.inv(view)[:3, 2], d.shape)
                cs = np.einsum("ij,ij->i", nk, d) / np.maximum(np.linalg.norm(nk, axis=1) * np.linalg.norm(d, axis=1), np.float32(1e-12))
                tn = np.sqrt(np.maximum(np.float32(1) - cs * cs, 0)) / np.maximum(cs, np.float32(0.2))   # surface slope along the ray
                lim = np.maximum(lim, zz[:, 1] + (np.float32(1.5) * px_mm + np.float32(reach_mm)) * tn)
                if facing: lim = np.where(cs > 0, lim, -np.inf)               # facing the eye (back skins of thin walls go)
            ok = ze <= lim + np.maximum(np.float32(tol_mm), np.float32(tol_px) * px_mm)
            k = k[ok]
        return self.to_elements(np.concatenate([self._slots(whole), sl[k]]))

    @staticmethod
    def eye_dist(depth, n, f, orthographic=False):
        d = depth.astype(np.float64)
        if orthographic: return np.where(d >= 1.0, np.inf, n + d * (f - n)).astype(np.float32)
        z = 2 * n * f / ((f + n) - (2 * d - 1) * (f - n))
        return np.where(d >= 1.0, np.inf, z).astype(np.float32)

    @staticmethod
    def ref_image(depth, n, f, big=BIG, orthographic=False):
        """(h,w,2) float32 per pixel: [0] own depth + how much nearer the nearest surface in the 3x3 around is (a sloped
        surface's own pixels pass, a wall behind a flat one does not); [1] that 3x3 nearest depth. Background = big."""
        zb = Elements.eye_dist(depth, n, f, orthographic)
        zmin = zb.copy(); zmin[1:] = np.minimum(zmin[1:], zb[:-1]); zmin[:-1] = np.minimum(zmin[:-1], zb[1:])
        z2 = zmin.copy(); z2[:, 1:] = np.minimum(z2[:, 1:], zmin[:, :-1]); z2[:, :-1] = np.minimum(z2[:, :-1], zmin[:, 1:])
        fin = np.isfinite(zb); zf = np.where(fin, zb, 0); z2f = np.where(fin, z2, 0)
        ref = np.where(fin, zf + (zf - z2f), big)
        return np.stack([np.minimum(ref, big), np.where(np.isfinite(z2), z2, big)], -1).astype(np.float32)

    # ---------------- 3D: sphere / box (same maths as stages/tools/edits.select_shape) ----------------
    def select_sphere(self, c, r):
        c = np.asarray(c, np.float32); r = np.float32(r)
        near = np.all((self.lo <= c + r) & (self.hi >= c - r), 1)
        sl = self._slots(np.nonzero(near)[0]); X = self.X[sl]
        d = X - c
        return self.to_elements(sl[np.einsum("ij,ij->i", d, d) <= r * r])

    def select_box(self, T, lo, hi):
        """T box -> world (the op's box.T), lo / hi in the box frame."""
        T = np.asarray(T, float); Mi = np.linalg.inv(T)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        cc = self._corners(np.arange(self.nc)) @ Mi[:3, :3].T + Mi[:3, 3]           # chunk corners in the box frame
        near = np.all((cc.min(1) <= hi) & (cc.max(1) >= lo), 1)
        sl = self._slots(np.nonzero(near)[0]); Y = self.X[sl].astype(np.float64) @ Mi[:3, :3].T + Mi[:3, 3]
        return self.to_elements(sl[((Y >= lo) & (Y <= hi)).all(1)])

    def positions(self, sel):
        """World positions of the selected elements, engine order (for the mask PLY a delete op is made from)."""
        inv = np.empty(self.n, np.int64); inv[self.perm] = np.arange(self.n)
        return self.X[inv[np.nonzero(sel)[0]]]


# ---------------- masks from what the hand drew ----------------
def polygon_mask(pts, w, h):
    """Filled polygon (lasso) in device pixels -> (h,w) bool. The app uses QPainterPath into a QImage; PIL here."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 0); ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=1, outline=1)
    return np.asarray(im, bool)


def rect_mask(x0, y0, x1, y1, w, h):
    m = np.zeros((h, w), bool); m[max(0, int(min(y0, y1))):int(max(y0, y1)) + 1, max(0, int(min(x0, x1))):int(max(x0, x1)) + 1] = True
    return m


def clip_box(p_world, n_world, extent):
    """Revo's Clip: everything on the arrow side of a plane (a line drawn on screen, swept along the view) as a box op
    the engine already replays: frame x = the plane normal, from 0 to 4x the model's size; y / z unbounded in practice."""
    n = np.asarray(n_world, float); n /= np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [0, 1.0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = n, a, b, p_world
    E = 4.0 * float(extent)
    return {"min": [0.0, -E, -E], "max": [E, E, E], "T": T.tolist()}
