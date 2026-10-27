# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export scan timelapse: a short clip of a scan being built, for sharing. A project's saved sides are played back
offscreen (no window, no screen recording, no scanner): each side's tracked frames are fused at their saved poses into
the live scanner's own TSDF model (track.Tracker.seed, as reopening a side does; no matching) and every few frames the
model so far is drawn in plain shading, with the camera moving around it during growth. Framing uses the
finished model bounds and a steady camera distance. Optional IR pictures and the PointYoink mark can be included.

Drawing: the 3D view's own GL code (app2/gl_core.py) on a headless EGL context (no window, no X server), else a numpy
splat renderer with the same shading when GL is not there. Encoding: MP4 (H.264) through ffmpeg; GIF through ffmpeg's
palettegen when ffmpeg is there, else Pillow. Nothing is written but the output file (a .part file until it is done).

Each side is fused twice: once whole, to frame the camera on the finished model (seconds on the GPU), then again while
drawing. One TSDF model on the GPU at a time. The app runs this as a child process (python -m app2.scan.timelapse):
its own CUDA and EGL, and the window never waits. Progress lines on stdout: "PROGRESS <0..1> <text>", then
"DONE <json>" or "FAILED <text>"; SIGTERM cancels (the .part file is removed)."""
import json, os, shutil, signal, subprocess, sys, time
import numpy as np

SIZES = {"480p": 480, "720p": 720, "1080p": 1080}
FPS = {"mp4": 24, "gif": 12}
DEFAULT_SECONDS = 8.0
from app2.qt.build_words import TIMELAPSE_SHEET
AUTO_ORBIT_SECONDS = max(length for length, _ in TIMELAPSE_SHEET["lengths"])  # Legacy callers may import it.
SHADED = np.full(3, 200, np.uint8)  # same gray as the built-model viewport
NEW_SURFACE=np.array([245,164,78],np.uint8)
GROWTH_PALETTE=np.linspace(SHADED,NEW_SURFACE,8).astype(np.float32)  # Existing viewport palette slots.
GREEN=np.array([76,201,122],np.float32);YELLOW=np.array([235,200,70],np.float32);RED=np.array([226,84,72],np.float32)
# the coverage look: slot 0 gray, 1-3 gray fading to green, 4 green, 5 yellow (getting there), 6 red (barely seen), 7 red
COVERAGE_PALETTE=np.concatenate([np.linspace(SHADED,GREEN,5),[YELLOW,RED,RED]]).astype(np.float32)
SLOT_YELLOW,SLOT_RED,SLOT_GREEN=5,6,4


class SurfaceAge:
    """First appearance attached to the finished surface, stable across TSDF remeshing."""
    def __init__(self,finished,voxel_mm,seconds):
        from scipy.spatial import cKDTree
        self.tree=cKDTree(finished);self.born=np.full(len(finished),np.inf)
        self.bound=np.sqrt(3.)*voxel_mm
        self.fade_seconds=float(seconds)/len(GROWTH_PALETTE)
    def levels(self,points,now):
        distance,index=self.tree.query(points,distance_upper_bound=self.bound)
        valid=np.isfinite(distance);out=np.zeros(len(points),np.uint8)
        if valid.any():
            indices=index[valid];self.born[indices]=np.minimum(self.born[indices],now)
            age=np.maximum(0,now-self.born[indices])
            out[valid]=np.rint(np.clip(1-age/self.fade_seconds,0,1)*(len(GROWTH_PALETTE)-1)).astype(np.uint8)
        return out

class SurfaceCoverage:
    """The coverage look: each point of the finished surface is red while it is barely seen, yellow while it is getting
    there, green once it is well covered, and then settles to gray over `fade` seconds of clip time."""
    def __init__(self,finished,voxel_mm,fade):
        from scipy.spatial import cKDTree
        self.tree=cKDTree(finished);self.green=np.full(len(finished),np.inf)
        self.bound=np.sqrt(3.)*voxel_mm;self.fade=max(1e-6,float(fade))
    def levels(self,points,level,now):
        out=np.where(level==0,SLOT_RED,np.where(level==1,SLOT_YELLOW,SLOT_GREEN)).astype(np.uint8)
        distance,index=self.tree.query(points,distance_upper_bound=self.bound)
        valid=np.isfinite(distance)&(level==2)
        if valid.any():
            idx=index[valid];self.green[idx]=np.minimum(self.green[idx],now)
            frac=np.clip((now-self.green[idx])/self.fade,0,1)
            out[valid]=np.rint((1-frac)*SLOT_GREEN).astype(np.uint8)
        return out


LAYOUTS = ("model", "camera+model")
COLORS = ("age","coverage")
BACKGROUND = (12, 14, 17)                     # the theme's well (#0c0e11), as SceneGL clears
FOV_Y = 30.0                                  # degrees: a longer lens than the scanner's, less distortion on the part
ELEVATION, AZIMUTH = 26.0, 35.0               # the 3/4 view: above the part, turned from where the scan started
FILL = 0.86                                   # the finished model fills this share of the model picture
HOLD_SECS = 0.4                               # the finished side stays this long before the next one
LOW_MM = 1.0                                  # the build's margin over the platter plane (bedcut.LOW_MM)
POINT_MM = 1.4                                # splat size, mm (the 3D view's: the tracker cloud is 1 mm apart)


class Cancelled(Exception): pass


def ffmpeg():
    return shutil.which("ffmpeg")


def formats():
    """The formats this machine can write: MP4 needs ffmpeg, GIF always works (Pillow)."""
    return ("mp4", "gif") if ffmpeg() else ("gif",)


def frame_size(height):
    """16:9, both even (H.264 with yuv420p needs even sides)."""
    h = int(height) // 2 * 2; return int(round(h * 16 / 9)) // 2 * 2, h


def default_path(project, fmt_):
    """<project>/<project name>_timelapse.<ext>, with _2, _3 ... when that name is taken."""
    stem = os.path.basename(os.path.normpath(project)) + "_timelapse"; p = os.path.join(project, stem + "." + fmt_); k = 2
    while os.path.exists(p): p = os.path.join(project, "%s_%d.%s" % (stem, k, fmt_)); k += 1
    return p


# --------------------------------------------------------------------------------------------------- the sides
class SidePlay:
    """One saved side to play: its tracked frames [(pass, frame, 4x4 mm)] in scan order, calibration, platter plane."""
    def __init__(self, folder):
        from app2.scan import fmt
        from fuse import read_calib
        self.folder = folder; self.cache = os.path.join(folder, "cache")
        saved = fmt.read_side(folder)
        self.entries = saved[0] if saved else []
        self.cal = read_calib(os.path.join(folder, "param", "Pl.bin"))
        from register import frame_size
        depths = [os.path.join(self.cache, fmt.frame_name(ps, idx) + ".dph") for ps, idx, _ in self.entries]
        depths = [path for path in depths if os.path.isfile(path)]
        self.size = frame_size(self.cache, depths) if depths else (fmt.W, fmt.H)
        scale = self.size[0] / 800.0
        self.cal = tuple(value * scale for value in self.cal)
        # (n, d, keep_above) of table_plane.json (fuse.py keeps x . n >= d + margin): the model view's Remove base cut
        self.plane = None; self.margin = LOW_MM; self.plane_source = None
        cached = None; self.final_surface = None; self.final_voxel_mm = None
        plane_path = os.path.join(folder, 'table_plane.json')
        try:
            from scanner.stages.engine import Engine
            from scanner.stages.project import Project
            engine=Engine(Project(os.path.dirname(os.path.dirname(folder))), read_only=True)
            token='side.platter@'+os.path.basename(folder)
            cached = engine.output(token, kind='plane')
            final_mesh=engine.output(token,kind='mesh') if cached else None
            if final_mesh:
                import open3d as o3d
                from scipy.spatial import cKDTree
                points=np.asarray(o3d.io.read_triangle_mesh(final_mesh).vertices)
                if len(points):self.final_surface=cKDTree(points)
                self.final_voxel_mm=float(engine.graph().by_id['side.fuse@'+os.path.basename(folder)].params['voxel'])
            if cached: plane_path = cached
        except (OSError, ValueError): pass
        try:
            with open(plane_path) as f: j = json.load(f)
            n = np.asarray(j["n"], float)
            if n.shape == (3,) and np.isfinite(n).all() and np.linalg.norm(n) > 1e-6:
                self.plane = (n / np.linalg.norm(n), float(j["d"]) / np.linalg.norm(n), bool(j.get("keep_above", True)))
                self.margin = float(j.get("margin", LOW_MM)); self.plane_source = "Cut base" if cached else "saved side plane"
        except (OSError, ValueError, KeyError, TypeError): pass

    def depth(self, k):
        from app2.scan import fmt
        ps, idx, _M = self.entries[k]
        try: raw = np.fromfile(os.path.join(self.cache, fmt.frame_name(ps, idx) + ".dph"), np.uint16)
        except OSError: return None
        w, h = self.size
        return raw.reshape(h, w) if raw.size == w * h else None

    def ir(self, k):
        from app2.scan import fmt
        ps, idx, _M = self.entries[k]; p = os.path.join(self.cache, fmt.frame_name(ps, idx) + ".irl")
        try: a = np.fromfile(p, np.uint8)
        except OSError: return None
        w, h = self.size
        return a.reshape(h, w) if a.size == w * h else None

    def pose(self, k):
        """Frame k's camera -> scan frame (OpenCV), mm."""
        from track import F
        return self.entries[k][2] @ F

    def up(self):
        """The scan frame's up: the platter's normal (towards the part) when there is one, else the first camera's up."""
        if self.plane is not None: return self.plane[0] if self.plane[2] else -self.plane[0]
        return -self.pose(0)[:3, 1]

    def cut(self, P, N, L):
        """Remove base: the platter and everything under it, as the Scan step's model view drops them."""
        if self.plane is None or not len(P): return P, N, L
        n, d, above = self.plane; s = P @ n.astype(np.float32) - np.float32(d)
        keep = s > self.margin if above else s < -self.margin
        return P[keep], N[keep], L[keep]


def pick_sides(project, sides=None):
    from app2.scan.sources import ReplaySource
    out = [SidePlay(f) for f in ReplaySource.pick_sides(project, sides)]
    out = [s for s in out if s.entries]
    if not out: raise ValueError("This project has no saved side with tracked frames")
    return out


def make_tracker(cal, device=None, width=800, height=600):
    """The live scanner's tracker (its voxel size and depth cut) on the GPU, or on the CPU when there is no CUDA."""
    import open3d.core as o3c
    from app2.scan.fmt import W, H
    from track import Tracker
    if device is None: device = "CUDA:0" if o3c.cuda.is_available() else "CPU:0"
    import inspect
    tracker=Tracker(cal, width, height, o3c.Device(device), max_depth_mm=1000.0)
    tracker.timelapse_voxel_mm=inspect.signature(Tracker).parameters['voxel_mm'].default
    return tracker


def fuse(trk, side, k):
    raw = side.depth(k)
    if raw is None: return
    T = side.pose(k).copy(); T[:3, 3] /= 1000.0; trk.seed(raw, T)


def model_cloud(trk, side):
    """The model so far (mm, normals, coverage level), platter cut."""
    from app2.scan.tracking import cloud_of
    z = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.uint8))
    if not trk.fused: return z
    try: P, N, L, _c = cloud_of(trk)
    except RuntimeError: return z                     # Open3D refuses an empty surface
    P,N,L=side.cut(P,N,L)
    if side.final_surface is not None and len(P):
        # Bound the disagreement of two voxelizations by their summed cell diagonals.
        bound=np.sqrt(3.)*(float(trk.timelapse_voxel_mm)+side.final_voxel_mm)
        workers=max(1,int(os.environ.get('OMP_NUM_THREADS','1')))
        distance=side.final_surface.query(P,distance_upper_bound=bound,workers=workers)[0]
        accepted=np.isfinite(distance)
        return P[accepted],N[accepted],L[accepted]
    return P,N,L



# --------------------------------------------------------------------------------------------------- the camera
class ShotCamera:
    """A camera fitted to one side's finished model: looking down ELEVATION degrees, turned AZIMUTH from where the scan
    started, far enough back that the model fills FILL of the picture. turn(deg) goes around the model's up axis."""
    CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, P, up, fwd, aspect, L=None):
        self.aspect = aspect; self.t = np.tan(np.radians(FOV_Y) / 2.0)
        up = np.asarray(up, float); up /= np.linalg.norm(up)
        f = np.asarray(fwd, float); f = f - (f @ up) * up
        if np.linalg.norm(f) < 1e-6: f = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
        self.up = up; self.f0 = f / np.linalg.norm(f)
        Q = np.asarray(P, float)
        if not len(Q): Q = np.zeros((1, 3))
        self.Q = Q; self.c = (Q.min(0) + Q.max(0)) / 2.0
        self.r = max(float(np.linalg.norm(Q - self.c, axis=1).max()), 5.0)
        X = Q - self.c; heights = X @ self.up
        radius = np.linalg.norm(X - heights[:, None] * self.up, axis=1)
        sine, cosine = np.sin(np.radians(ELEVATION)), np.cos(np.radians(ELEVATION))
        tx, ty = self.t * self.aspect * FILL, self.t * FILL
        horizontal = radius * np.sqrt(1 / tx**2 + cosine**2) + heights * sine
        vertical = [np.abs(heights * cosine + q * sine) / ty - q * cosine + heights * sine
                    for q in (-radius, radius)]
        self.fixed_distance = max(float(horizontal.max()), *(float(v.max()) for v in vertical), self.r * 0.5)
        self.dist = {}

    def _axes(self, deg):
        a = np.radians(AZIMUTH + deg); e = np.radians(ELEVATION); u = self.up
        f = self.f0 * np.cos(a) + np.cross(u, self.f0) * np.sin(a)            # turned about up
        d = f * np.cos(e) - u * np.sin(e); d /= np.linalg.norm(d)             # looking down onto the part
        s = np.cross(d, u); s /= np.linalg.norm(s); t = np.cross(s, d)
        return s, t, d

    def _fit(self, deg):
        """How far back so every (kept) point is inside FILL of the picture; the aim stays at the finished bounds' middle."""
        return self.fixed_distance, self.c

    def view(self, deg=0.0):
        key = round(float(deg), 3)
        if key not in self.dist: self.dist[key] = self._fit(deg)
        dist, c = self.dist[key]; s, t, d = self._axes(deg); eye = c - d * dist
        V = np.eye(4); V[0, :3], V[1, :3], V[2, :3] = s, t, -d; V[:3, 3] = -V[:3, :3] @ eye
        return V

    def proj(self, view):
        """Near and far planes about the model (the eye depth of its middle +- its radius)."""
        dc = -(view[2, :3] @ self.c + view[2, 3]); n = max(dc - 1.6 * self.r, 0.02 * self.r); fa = dc + 1.6 * self.r
        r, t = self.t * self.aspect, self.t
        eye=self.Q @ view[:3,:3].T + view[:3,3]
        horizontal=eye[:,0]/(-eye[:,2]*r);vertical=eye[:,1]/(-eye[:,2]*t)
        pan_x=(horizontal.min()+horizontal.max())/2.;pan_y=(vertical.min()+vertical.max())/2.
        return np.array([[1 / r, 0, pan_x, 0], [0, 1 / t, pan_y, 0], [0, 0, (fa + n) / (n - fa), 2 * fa * n / (n - fa)], [0, 0, -1, 0]])


# --------------------------------------------------------------------------------------------------- drawing
class GLRenderer:
    """The 3D view's point drawing (gl_core.SceneGL) on a surfaceless EGL context into a framebuffer, drawn at twice
    the size and scaled down (smooth edges). Needs PyOpenGL on its EGL platform: only in a process that has not
    loaded OpenGL for another platform first (the timelapse child process, the tests)."""
    name = "gpu"
    SS = 2

    def __init__(self):
        import ctypes
        if "OpenGL.GL" not in sys.modules: os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if os.environ.get("PYOPENGL_PLATFORM") != "egl": raise RuntimeError("OpenGL is already loaded for another platform")
        from OpenGL import EGL
        E = EGL; self.E = E
        self.dpy = E.eglGetDisplay(E.EGL_DEFAULT_DISPLAY)
        if not self.dpy or not E.eglInitialize(self.dpy, None, None): raise RuntimeError("no EGL display")
        cfg = E.EGLConfig(); n = E.EGLint()
        attrs = (E.EGLint * 7)(E.EGL_SURFACE_TYPE, E.EGL_PBUFFER_BIT, E.EGL_RENDERABLE_TYPE, E.EGL_OPENGL_BIT, E.EGL_DEPTH_SIZE, 24, E.EGL_NONE)
        if not E.eglChooseConfig(self.dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) or n.value < 1: raise RuntimeError("no EGL config")
        E.eglBindAPI(E.EGL_OPENGL_API)
        ca = (E.EGLint * 7)(E.EGL_CONTEXT_MAJOR_VERSION, 3, E.EGL_CONTEXT_MINOR_VERSION, 3,
                            E.EGL_CONTEXT_OPENGL_PROFILE_MASK, E.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, E.EGL_NONE)
        self.ctx = E.eglCreateContext(self.dpy, cfg, E.EGL_NO_CONTEXT, ca)
        if not self.ctx or not E.eglMakeCurrent(self.dpy, E.EGL_NO_SURFACE, E.EGL_NO_SURFACE, self.ctx): raise RuntimeError("no EGL context")
        from OpenGL import GL
        from app2 import gl_core
        self.GL = GL; self.scene = gl_core.SceneGL(); self.scene.palette = np.tile(SHADED / 255.0, (len(gl_core.PALETTE), 1)).astype(np.float32); self.scene.background = tuple(c / 255.0 for c in BACKGROUND)
        self.fbo = None; self.size = None
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")

    def _target(self, w, h):
        GL = self.GL
        if self.size == (w, h): GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo); return
        if self.fbo is not None: GL.glDeleteFramebuffers(1, [self.fbo]); GL.glDeleteRenderbuffers(2, self.rbo)
        self.fbo = GL.glGenFramebuffers(1); self.rbo = list(GL.glGenRenderbuffers(2))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        for rb, kind, att in ((self.rbo[0], GL.GL_RGBA8, GL.GL_COLOR_ATTACHMENT0), (self.rbo[1], GL.GL_DEPTH_COMPONENT24, GL.GL_DEPTH_ATTACHMENT)):
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb); GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, kind, w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, att, GL.GL_RENDERBUFFER, rb)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE: raise RuntimeError("framebuffer incomplete")
        self.size = (w, h)

    def render(self, P, N, L, view, proj, w, h):
        import cv2
        GL = self.GL; W2, H2 = w * self.SS, h * self.SS
        self._target(W2, H2)
        if getattr(self,"growth",False):self.scene.palette=getattr(self,"palette",GROWTH_PALETTE)/255.
        if len(P): self.scene.set_points(P, N, L)
        else: self.scene.points.count = 0
        self.scene.point_size = POINT_MM
        self.scene.draw(view, proj, W2, H2)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        buf = GL.glReadPixels(0, 0, W2, H2, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, np.uint8).reshape(H2, W2, 3)[::-1]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def close(self):
        try:
            self.scene.points.delete(); self.scene.mesh.delete()
            E = self.E; E.eglMakeCurrent(self.dpy, E.EGL_NO_SURFACE, E.EGL_NO_SURFACE, E.EGL_NO_CONTEXT)
            E.eglDestroyContext(self.dpy, self.ctx); E.eglTerminate(self.dpy)
        except Exception: pass


class CPURenderer:
    """The same picture without GL: points projected and splatted far to near (the Tk scanner's model picture), each
    a square about POINT_MM across, lit from the eye as the GL shader does."""
    name = "cpu"
    renderer = "numpy"

    def render(self, P, N, L, view, proj, w, h):
        img = np.empty((h, w, 3), np.uint8); img[:] = BACKGROUND
        if not len(P): return img
        step = max(1, len(P) // 1500000); P, N, L = P[::step], N[::step], L[::step]
        Pe = P @ view[:3, :3].T.astype(np.float32) + view[:3, 3].astype(np.float32); z = -Pe[:, 2]; ok = z > 1e-3
        Pe, z, Ne, L = Pe[ok], z[ok], N[ok] @ view[:3, :3].T.astype(np.float32), L[ok]
        u = ((Pe[:, 0] * proj[0, 0] / z - proj[0,2]) * 0.5 + 0.5) * w; v = (0.5 - (Pe[:, 1] * proj[1, 1] / z - proj[1,2]) * 0.5) * h
        shade = 0.15 + 0.85 * np.abs(np.sum(Ne * (-Pe / np.linalg.norm(Pe, axis=1, keepdims=True)), 1))
        color=getattr(self,"palette",GROWTH_PALETTE)[L] if getattr(self,"growth",False) else SHADED
        col = (color * shade[:, None]).astype(np.uint8)
        size = POINT_MM * proj[1, 1] * 0.5 * h / np.median(z) * np.sqrt(step)
        k = int(np.clip(np.ceil(size), 1, 12))
        ui = (u - k / 2.0).astype(np.int32); vi = (v - k / 2.0).astype(np.int32)
        inside = (ui >= 0) & (ui < w - k) & (vi >= 0) & (vi < h - k)
        order = np.argsort(-z[inside]); ui, vi, col = ui[inside][order], vi[inside][order], col[inside][order]
        for du in range(k):
            for dv in range(k): img[vi + dv, ui + du] = col
        return img

    def close(self): pass


def make_renderer(kind="auto"):
    """"gpu", "cpu" or "auto" (GL when it starts, else the numpy splats)."""
    if kind == "cpu": return CPURenderer()
    try: return GLRenderer()
    except Exception as e:
        if kind == "gpu": raise
        print("timelapse: no GL (%s), drawing on the CPU" % str(e).splitlines()[0] if str(e) else e, file=sys.stderr, flush=True)
        return CPURenderer()


def camera_picture(side, k, w, h):
    """The Scan step's IR picture of frame k (kept depth tinted green), fitted into w x h."""
    import cv2
    from app2.scan.preview import ir_view
    img = np.empty((h, w, 3), np.uint8); img[:] = BACKGROUND
    irl = side.ir(k); d = side.depth(k)
    if irl is None or d is None: return img
    pic = ir_view(d, irl, side.cal, False, None, None)
    s = min(w / pic.shape[1], h / pic.shape[0]); pw, ph = int(pic.shape[1] * s), int(pic.shape[0] * s)
    x, y = (w - pw) // 2, (h - ph) // 2
    img[y:y + ph, x:x + pw] = cv2.resize(pic, (pw, ph), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------------------------------- the mark
LOGO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "icon.png")
FONTS = ("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/noto/NotoSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")
MARK_H, MARK_ALPHA = 0.06, 0.7


class Watermark:
    """The PointYoink logo and name bottom-left: MARK_H of the frame's height, MARK_ALPHA opaque, with a soft dark
    shadow so it reads on the dark background and on light parts. Made once per frame size; apply() blends it in."""
    def __init__(self, w, h):
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        s = max(12, int(round(h * MARK_H))); m = max(6, int(round(h * 0.03)))
        font = None
        for f in FONTS:
            if os.path.exists(f):
                try: font = ImageFont.truetype(f, int(s * 0.62)); break
                except OSError: pass
        if font is None: font = ImageFont.load_default()
        text = "PointYoink"; tb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
        gap = int(s * 0.25); pad = max(3, s // 5)
        lw, lh = s + gap + (tb[2] - tb[0]) + 2 * pad, s + 2 * pad
        layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        try:
            logo = Image.open(LOGO).convert("RGBA").resize((s, s), Image.LANCZOS); layer.alpha_composite(logo, (pad, pad))
        except OSError: pass
        ImageDraw.Draw(layer).text((pad + s + gap - tb[0], pad + (s - (tb[3] - tb[1])) // 2 - tb[1]), text, font=font, fill=(240, 244, 250, 255))
        a = np.asarray(layer.getchannel("A"))
        sh = Image.fromarray(a).filter(ImageFilter.GaussianBlur(max(1.0, s * 0.06)))
        off = max(1, s // 16); shadow = np.zeros_like(a, dtype=np.float32)
        sa = np.asarray(sh, np.float32); shadow[off:, off:] = sa[:-off, :-off]
        rgb = np.asarray(layer, np.float32)[..., :3]; al = a.astype(np.float32) / 255.0; sh_a = shadow / 255.0 * 0.6
        # the mark over its shadow, then MARK_ALPHA of the whole
        out_a = al + sh_a * (1 - al)
        self.rgb = np.where(out_a[..., None] > 0, rgb * al[..., None] / np.maximum(out_a[..., None], 1e-6), 0)
        self.alpha = (out_a * MARK_ALPHA)[..., None]
        self.x, self.y = m, h - m - lh
        self.w, self.h = lw, lh
        if self.x + lw > w or self.y < 0: self.alpha = None             # too small a frame for it

    def apply(self, img):
        if self.alpha is None: return img
        r = img[self.y:self.y + self.h, self.x:self.x + self.w].astype(np.float32)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = (r * (1 - self.alpha) + self.rgb * self.alpha).round().astype(np.uint8)
        return img


def draw_label(img, text, h):
    """A small side label top-left ("Side 2 of 4")."""
    import cv2
    s = h / 720.0; m = int(round(h * 0.03))
    cv2.putText(img, text, (m + 1, m + int(22 * s) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(img, text, (m, m + int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (200, 208, 220), max(1, int(2 * s)), cv2.LINE_AA)


# --------------------------------------------------------------------------------------------------- encoding
class Encoder:
    """Frames in (HxWx3 uint8) as they are drawn, a file out. MP4: ffmpeg libx264, yuv420p, crf 20. GIF: ffmpeg's
    palettegen/paletteuse (one 128-color palette for the clip, made from what changes: a quarter smaller than 256
    colors with Bayer dither, and the model's few colors hardly show it), else Pillow."""
    def __init__(self, path, fmt_, w, h, fps):
        self.path, self.fmt, self.w, self.h, self.fps = path, fmt_, w, h, fps
        self.part = path + ".part"; self.n = 0; self.frames = None; self.proc = None
        ff = ffmpeg()
        if fmt_ == "mp4" and not ff: raise RuntimeError("MP4 needs ffmpeg; it is not installed (GIF works without it)")
        if ff:
            out = ["-f", "mp4", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart"] if fmt_ == "mp4" else \
                  ["-f", "gif", "-filter_complex",
                   "[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                   "-loop", "0"]
            self.proc = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                          "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-"] + out + [self.part],
                                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else: self.frames = []

    def write(self, img):
        self.n += 1
        if self.proc is not None:
            try: self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
            except BrokenPipeError: raise RuntimeError("ffmpeg stopped: %s" % self.proc.stderr.read().decode(errors="replace")[-300:])
        else: self.frames.append(img.copy())

    def finish(self):
        if self.proc is not None:
            self.proc.stdin.close(); err = self.proc.stderr.read().decode(errors="replace"); self.proc.wait()
            if self.proc.returncode: raise RuntimeError("ffmpeg failed: %s" % err.strip()[-300:])
        else: self._pillow_gif()
        os.replace(self.part, self.path)
        return os.path.getsize(self.path)

    def _pillow_gif(self):
        from PIL import Image
        fr = self.frames
        # one palette from a strip of frames spread over the clip (a palette per frame flickers)
        pick = fr[::max(1, len(fr) // 12)][:12]
        strip = Image.fromarray(np.concatenate([f[::2, ::2] for f in pick], 0))
        pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in fr]
        ims[0].save(self.part, format="GIF", save_all=True, append_images=ims[1:], duration=int(round(1000 / self.fps)),
                    loop=0, optimize=False, disposal=1)

    def abort(self):
        if self.proc is not None:
            try: self.proc.kill(); self.proc.wait(5)
            except Exception: pass
        try: os.remove(self.part)
        except OSError: pass


# --------------------------------------------------------------------------------------------------- the clip
def spread(ns, total):
    """Split total growth frames over sides with ns frames each, in proportion (at least 2 each)."""
    ns = np.asarray(ns, float); g = np.maximum(2, np.floor(ns / ns.sum() * total)).astype(int)
    while g.sum() > total and g.max() > 2: g[np.argmax(g)] -= 1
    while g.sum() < total: g[np.argmax(ns / g)] += 1
    return [int(x) for x in g]


def render_timelapse(project, out=None, sides=None, layout="model", height=None, seconds=8.0, orbit=2.0, fmt_=None,
                     watermark=True, fps=None, progress=None, cancel=None, device=None, renderer="auto", still=None, turns=None, colors="age"):
    """Make the clip. project: a ProjectLive folder; sides: 1-based numbers or folder names (None: all, in order);
    layout "model" or "camera+model"; height 480 / 720 / 1080 (default 720 for MP4, 480 for GIF); seconds: the whole
    clip; turns: camera rotations during growth, one full turn by default; orbit: legacy zero-off switch; fmt_ "mp4" or
    "gif" (default: out's extension, else MP4 when ffmpeg is there); progress(share 0..1, text); cancel() -> True
    stops it (Cancelled; nothing is left behind). still: also save the last growth frame as a PNG there.
    Returns dict(path, bytes, frames, fps, seconds, renderer, sides, secs)."""
    t0 = time.monotonic()
    if fmt_ is None: fmt_ = (os.path.splitext(out)[1][1:].lower() if out else "") or formats()[0]
    if fmt_ not in ("mp4", "gif"): raise ValueError("format must be mp4 or gif")
    if layout not in LAYOUTS: raise ValueError("layout must be one of %s" % ", ".join(LAYOUTS))
    height = int(height or (720 if fmt_ == "mp4" else 480)); fps = int(fps or FPS[fmt_])
    out = out or default_path(project, fmt_)
    w, h = frame_size(height); pw = w // 2 if layout == "camera+model" else w
    say = progress or (lambda share, text: None); stop = cancel or (lambda: False)
    plays = pick_sides(project, sides)
    total = max(4, int(round(float(seconds) * fps)))
    turns = (1.0 if orbit else 0.0) if turns is None else float(turns)
    if not np.isfinite(turns) or turns < 0: raise ValueError('Camera turns must be a finite nonnegative number.')
    for side in plays:
        if side.plane is None: raise ValueError('Build Cut base for this side before exporting the timelapse.')
    n_hold = int(round(HOLD_SECS * fps)) if len(plays) > 1 else 0
    grow = spread([len(s.entries) for s in plays], max(2 * len(plays), total - n_hold * len(plays)))
    fused_total = sum(len(s.entries) for s in plays)
    work = fused_total * 0.08 + total; done = [0.0]   # a frame fused for framing ~ a twelfth of a frame drawn

    def tick(units, text):
        if stop(): raise Cancelled()
        done[0] += units; say(min(1.0, done[0] / work), text)

    if colors not in COLORS: raise ValueError('colors must be one of %s' % ', '.join(COLORS))
    rend = make_renderer(renderer);rend.growth=True;rend.palette=COVERAGE_PALETTE if colors=='coverage' else GROWTH_PALETTE; enc = Encoder(out, fmt_, w, h, fps); mark = Watermark(w, h) if watermark else None
    from app2.scan.tracking import free_gpu
    trk = None; last = None; frame = None
    try:
        for si, side in enumerate(plays):
            label = "Side %d of %d" % (si + 1, len(plays)) if len(plays) > 1 else ""
            n = len(side.entries); pre = "%s: " % label if label else ""
            # 1. Fuse every saved frame to fit the finished model.
            trk = make_tracker(side.cal, device, *side.size); ks = list(range(n))
            for j, k in enumerate(ks): fuse(trk, side, k); tick(0.08, "%sreading frames %d of %d" % (pre, k + 1, n))
            first = model_cloud(trk, side);voxel=float(trk.timelapse_voxel_mm); trk = None; free_gpu()
            if not len(first[0]): raise ValueError("Saved frames did not produce a model. Check the scan depth and saved poses.")
            ages=SurfaceCoverage(first[0],voxel,float(seconds)/4) if colors=='coverage' else SurfaceAge(first[0],voxel,float(seconds))
            cam = ShotCamera(first[0], side.up(), side.pose(0)[:3, 2], pw / h, first[2]); first = None
            V = cam.view(0.0); Pj = cam.proj(V)
            # 2. again, frame by frame, drawing it as it grows
            trk = make_tracker(side.cal, device, *side.size); k = 0; img = None; shown = -1; final = None
            counts = np.clip(np.round(np.linspace(1, n, grow[si])).astype(int), 1, n)
            for j, c in enumerate(counts):
                while k < c: fuse(trk, side, k); k += 1
                if c != shown: final = model_cloud(trk, side); shown = c
                angle = 360.0 * turns * enc.n / max(1, total - 1)
                V = cam.view(angle);P,N,Lc=final
                growth=(P,N,ages.levels(P,Lc,enc.n/fps) if colors=='coverage' else ages.levels(P,enc.n/fps))
                img = rend.render(*growth, V, cam.proj(V), pw, h)
                if layout == "camera+model": img = np.concatenate([camera_picture(side, c - 1, w - pw, h), img], 1)
                if label: draw_label(img, label, h)
                frame = img.copy()
                enc.write(mark.apply(frame) if mark else frame)
                tick(1, "%sframe %d of %d" % (pre, c, n))
            if still: _save_png(still, frame)
            last = (side, cam, final, img)
            trk = None; free_gpu()
            for _ in range(n_hold):
                enc.write(frame); tick(1, "%sdone" % pre)
        size = enc.finish()
    except BaseException:
        enc.abort(); raise
    finally:
        trk = None; rend.close()
        try: free_gpu()
        except Exception: pass
    return dict(path=out, bytes=size, frames=enc.n, fps=fps, seconds=enc.n / fps, renderer=rend.name,
                sides=len(plays), secs=time.monotonic() - t0, width=w, height=h, camera_turns=turns,
                camera_motion='during growth', shading='new surface fades to gray' if colors=='age' else 'coverage: red, yellow, green, then gray', planes=[side.plane_source for side in plays])


def _save_png(path, img):
    from PIL import Image
    Image.fromarray(img).save(path)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m app2.scan.timelapse", description="Export a scan timelapse (GIF or MP4).")
    ap.add_argument("project"); ap.add_argument("-o", "--out")
    ap.add_argument("--side", action="append", help="side number (1-based) or folder name; repeat for more (default: all)")
    ap.add_argument("--layout", choices=LAYOUTS, default="model")
    ap.add_argument("--size", choices=tuple(SIZES), help="default 720p for MP4, 480p for GIF")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS); ap.add_argument("--orbit", type=float, default=2.0, help="legacy switch: zero disables camera motion")
    ap.add_argument("--turns", type=float, help="camera turns during the clip; default is one full turn")
    ap.add_argument("--format", choices=("mp4", "gif")); ap.add_argument("--fps", type=int)
    ap.add_argument("--no-watermark", action="store_true"); ap.add_argument("--renderer", choices=("auto", "gpu", "cpu"), default="auto")
    ap.add_argument("--device", help="Open3D device for the model (default CUDA:0 when there is one)")
    ap.add_argument("--colors", choices=COLORS, default="age", help="age: new surface orange fading to gray; coverage: red, yellow, green, then gray")
    ap.add_argument("--still", help="also save the last growth frame as this PNG")
    a = ap.parse_args(argv)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    sides = [int(s) if s.isdigit() else s for s in a.side] if a.side else None
    last = [0.0]

    def progress(share, text):
        now = time.monotonic()
        if now - last[0] >= 0.2 or share >= 1.0: last[0] = now; print("PROGRESS %.4f %s" % (share, text), flush=True)
    try:
        r = render_timelapse(a.project, a.out, sides, a.layout, SIZES[a.size] if a.size else None, a.seconds, a.orbit,
                             a.format, not a.no_watermark, a.fps, progress, lambda: stopped[0], a.device, a.renderer, a.still, a.turns, a.colors)
    except Cancelled:
        print("CANCELLED", flush=True); return 2
    except Exception as e:
        print("FAILED %s" % (str(e).splitlines()[0] if str(e) else type(e).__name__), flush=True); return 1
    print("DONE " + json.dumps(r), flush=True); return 0


if __name__ == "__main__":
    import app2  # noqa: F401 (puts the repo and scanner/ on sys.path)
    sys.exit(main())
