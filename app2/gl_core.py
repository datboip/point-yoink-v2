# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 3D view's drawing, on whatever OpenGL 3.3 core context is current: the QOpenGLWidget's in the app, a headless EGL
context in the benchmarks and tests. No Qt in here, so the same code is measured on Xvfb (Mesa llvmpipe) and on the
RTX 3080 (EGL device, no window anywhere).

Two things get drawn:
  points  the live model (tracker cloud, 0.3-2 M points, 1 mm apart) with normals and a coverage level each (0 barely
          seen, 1 getting there, 2 well covered; 3 = picked, for Clean). Replaced wholesale about 3 times a second: the
          tracker rebuilds its cloud from the TSDF.
  mesh    the built model (1-5 M triangles) with per-vertex colours, uploaded once, drawn every frame while orbiting.
Each attribute has its own buffer, so the tracker's arrays go up as they are: float32 positions, float32 normals, uint8
levels (the colour comes from a palette in the shader). No packing on the CPU; an interleaved 20-byte vertex was tried
first and packing 2 M points took 66-99 ms of numpy per update, against 0 here.

Units are whatever the caller uses (the project files are in mm); point_size is in the same units.
PyOpenGL checks glGetError after every call by default; that is off here unless POINTYOINK_GL_DEBUG=1.
"""
import ctypes, os
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = os.environ.get("POINTYOINK_GL_DEBUG") == "1"
from OpenGL import GL

# coverage colours of the Tk scanner (track.Tracker.render): barely seen, getting there, well covered; then picked;
# then Touch up's (app2/scan/touchup.py): fine (dimmed gray), fine in the camera's view now, rough (purple), done (green)
# coverage: purple scan here, light blue getting there, deep blue well covered; never the red, yellow and green that
# mean distance on the bar and the dots (seen live: red coverage read as "too close")
PALETTE = np.array([[185, 85, 245], [150, 205, 255], [45, 110, 230], [255, 255, 255],
                    [92, 96, 104], [150, 157, 168], [185, 85, 245], [70, 215, 120]], np.float32) / 255.0
COVERAGE = (PALETTE[:3] * 255).round().astype(np.uint8)

POINT_MM = 1.25   # live splat size in world units: the tracker cloud is 1 mm apart; small round splats, just touching
POINT_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_nrm;
layout(location = 2) in float a_lvl;
uniform mat4 u_mv, u_proj;
uniform vec3 u_palette[8];
uniform bool u_capture_colors;
uniform float u_size, u_vp_h;          // point size in world units, viewport height in pixels
out vec3 v_col;
void main() {
    vec4 p = u_mv * vec4(a_pos, 1.0);
    gl_Position = u_proj * p;
    // lit from the eye, like the Tk scanner's splats: 0.15 ambient + 0.85 |n . view|
    float shade = 0.15 + 0.85 * abs(dot(normalize(mat3(u_mv) * a_nrm), normalize(-p.xyz)));
    // distance colors keep a softer light so the surface still reads as a shape, not flat squares
    v_col = u_palette[int(a_lvl + 0.5) & 7] * (u_capture_colors ? 0.45 + 0.55 * shade : shade);
    // no low cap: zoomed in close, capped points stopped touching and left black lines across the surface
    gl_PointSize = clamp(u_size * u_proj[1][1] * 0.5 * u_vp_h / max(gl_Position.w, 1e-6), 1.0, 256.0);
}
"""
POINT_FS = """#version 330 core
in vec3 v_col;
out vec4 frag;
void main() {
    // round, slightly domed splats instead of flat squares: neighbors overlap into a smooth looking surface
    vec2 c = gl_PointCoord * 2.0 - 1.0; float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    frag = vec4(v_col * (1.0 - 0.10 * r2), 1.0);
}
"""
MESH_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec3 a_col;
uniform mat4 u_mv, u_proj;
out vec3 v_n; out vec3 v_p; out vec3 v_col;
void main() {
    vec4 p = u_mv * vec4(a_pos, 1.0);
    v_p = p.xyz; v_n = mat3(u_mv) * a_nrm; v_col = a_col;
    gl_Position = u_proj * p;
}
"""
MESH_FS = """#version 330 core
in vec3 v_n; in vec3 v_p; in vec3 v_col;
uniform bool u_capture_colors;
out vec4 frag;
void main() {
    vec3 n = normalize(v_n); vec3 v = normalize(-v_p);
    float d = abs(dot(n, v));                                  // two-sided headlight
    float s = pow(max(dot(reflect(-v, n), v), 0.0), 24.0);     // a little shine shows the surface's ripples
    frag = vec4(u_capture_colors ? v_col * (0.45 + 0.55 * d) + 0.06 * s : v_col * (0.18 + 0.82 * d) + 0.12 * s, 1.0);
}
"""
# name, location, components, type, normalized, numpy dtype
POINT_ATTRS = (("pos", 0, 3, GL.GL_FLOAT, False, np.float32), ("nrm", 1, 3, GL.GL_FLOAT, False, np.float32),
               ("lvl", 2, 1, GL.GL_UNSIGNED_BYTE, False, np.uint8))
MESH_ATTRS = (("pos", 0, 3, GL.GL_FLOAT, False, np.float32), ("nrm", 1, 3, GL.GL_FLOAT, False, np.float32),
              ("rgb", 2, 3, GL.GL_UNSIGNED_BYTE, True, np.uint8))


def _program(vs, fs):
    from OpenGL.GL import shaders
    return shaders.compileProgram(shaders.compileShader(vs, GL.GL_VERTEX_SHADER), shaders.compileShader(fs, GL.GL_FRAGMENT_SHADER),
                                  validate=False)


class Drawable:
    """One VAO, one buffer per attribute (+ an index buffer for a mesh). upload() replaces the data; a buffer only grows
    (by half again when too small), so a cloud that grows a little every update does not reallocate every time."""

    def __init__(self, attrs, indexed):
        self.attrs = attrs; self.indexed = indexed; self.count = 0; self.icount = 0
        self.vao = GL.glGenVertexArrays(1); self.vbo = {}; self.cap = {}
        GL.glBindVertexArray(self.vao)
        for name, loc, k, t, norm, _ in attrs:
            self.vbo[name] = GL.glGenBuffers(1); self.cap[name] = 0
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[name])
            GL.glEnableVertexAttribArray(loc); GL.glVertexAttribPointer(loc, k, t, norm, 0, ctypes.c_void_p(0))
        self.ibo = GL.glGenBuffers(1) if indexed else None; self.cap["faces"] = 0
        if indexed: GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)       # the IBO binding lives in the VAO
        GL.glBindVertexArray(0)

    def _put(self, target, buf, key, a, grow):
        GL.glBindBuffer(target, buf)
        if a.nbytes > self.cap[key]:
            self.cap[key] = int(a.nbytes * grow); GL.glBufferData(target, self.cap[key], None, GL.GL_DYNAMIC_DRAW if grow > 1 else GL.GL_STATIC_DRAW)
        GL.glBufferSubData(target, 0, a.nbytes, a)

    def upload(self, faces=None, static=False, **arrays):
        """arrays by attribute name, all the same length; each goes up as it is when it already has the right dtype
        (the tracker's float32 positions / normals, uint8 levels). Pass only some to update just those (a new level
        array recolours the cloud without sending positions again)."""
        grow = 1.0 if static else 1.5
        GL.glBindVertexArray(self.vao)
        for name, loc, k, t, norm, dt in self.attrs:
            if name not in arrays: continue
            a = np.ascontiguousarray(arrays[name], dt)
            self._put(GL.GL_ARRAY_BUFFER, self.vbo[name], name, a, grow)
            self.count = len(a)
        if self.indexed and faces is not None:
            f = np.ascontiguousarray(faces, np.uint32)
            self._put(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo, "faces", f, 1.0); self.icount = f.size
        GL.glBindVertexArray(0)

    def draw(self):
        if not self.count: return
        GL.glBindVertexArray(self.vao)
        if self.indexed: GL.glDrawElements(GL.GL_TRIANGLES, self.icount, GL.GL_UNSIGNED_INT, None)
        else: GL.glDrawArrays(GL.GL_POINTS, 0, self.count)
        GL.glBindVertexArray(0)

    def delete(self):
        GL.glDeleteVertexArrays(1, [self.vao]); GL.glDeleteBuffers(len(self.vbo), list(self.vbo.values()))
        if self.ibo: GL.glDeleteBuffers(1, [self.ibo])
        self.count = self.icount = 0


class SceneGL:
    """Everything the view draws. Create with a current context; call draw() with it current."""

    def __init__(self):
        self.point_prog = _program(POINT_VS, POINT_FS); self.mesh_prog = _program(MESH_VS, MESH_FS)
        self.points = Drawable(POINT_ATTRS, False); self.mesh = Drawable(MESH_ATTRS, True)
        self.point_size = POINT_MM
        self.background = (0.047, 0.055, 0.067)     # the theme's well (#0c0e11), as behind the camera pictures
        self.show_points = True; self.show_mesh = True
        self._u = {p: {n: GL.glGetUniformLocation(p, n) for n in ("u_mv", "u_proj", "u_size", "u_vp_h", "u_palette", "u_capture_colors")}
                   for p in (self.point_prog, self.mesh_prog)}

    def set_points(self, P, N, levels):
        """The live cloud: (n,3) float32 positions and unit normals, (n,) uint8 coverage level. Context must be current."""
        self.points.upload(pos=P, nrm=N, lvl=levels)

    def set_levels(self, levels):
        """Recolour the cloud (coverage changed, or Clean picked some points: level 3) without sending positions."""
        self.points.upload(lvl=levels)

    def set_mesh(self, V, N, C, faces):
        """The built model: float positions and normals, (n,3) uint8 colours (None: plain grey), (m,3) faces."""
        if C is None: C = np.full((len(V), 3), 200, np.uint8)
        self.mesh.upload(pos=V, nrm=N, rgb=C, faces=faces, static=True)

    def draw(self, view, proj, vp_w, vp_h):
        """view: world -> eye (OpenGL convention), proj: eye -> clip, both 4x4 float."""
        GL.glViewport(0, 0, vp_w, vp_h)
        GL.glClearColor(*self.background, 1.0); GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_BLEND); GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        mv = np.ascontiguousarray(view, np.float32); pj = np.ascontiguousarray(proj, np.float32)
        if self.show_mesh and self.mesh.count:
            u = self._u[self.mesh_prog]; GL.glUseProgram(self.mesh_prog)
            GL.glUniform1i(u["u_capture_colors"],int(getattr(self,"capture_colors",False)))
            GL.glUniformMatrix4fv(u["u_mv"], 1, True, mv); GL.glUniformMatrix4fv(u["u_proj"], 1, True, pj)
            self.mesh.draw()
        if self.show_points and self.points.count:
            u = self._u[self.point_prog]; GL.glUseProgram(self.point_prog)
            GL.glUniform1i(u["u_capture_colors"],int(getattr(self,"capture_colors",False)))
            GL.glUniformMatrix4fv(u["u_mv"], 1, True, mv); GL.glUniformMatrix4fv(u["u_proj"], 1, True, pj)
            GL.glUniform1f(u["u_size"], self.point_size); GL.glUniform1f(u["u_vp_h"], float(vp_h))
            GL.glUniform3fv(u["u_palette"], len(getattr(self, "palette", PALETTE)), getattr(self, "palette", PALETTE))
            self.points.draw()
        GL.glUseProgram(0)

    def read_depth(self, x, y, vp_h):
        """Depth buffer value (0..1) under widget pixel x, y (top-left origin); 1.0 = background. Needs the frame
        that was just drawn still bound (inside paintGL, or after makeCurrent in the widget)."""
        d = GL.glReadPixels(int(x), int(vp_h - 1 - y), 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        return float(np.frombuffer(d, np.float32)[0]) if isinstance(d, (bytes, bytearray)) else float(np.asarray(d).reshape(-1)[0])

    def read_depth_image(self, vp_w, vp_h):
        """The whole depth buffer, top row first (vp_h x vp_w float32, 1.0 = background). For box / lasso selection."""
        d = GL.glReadPixels(0, 0, vp_w, vp_h, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        a = np.frombuffer(d, np.float32) if isinstance(d, (bytes, bytearray)) else np.asarray(d, np.float32)
        return a.reshape(vp_h, vp_w)[::-1]


class Camera:
    """Orbit / pan / zoom about a pivot, kept as one world -> eye matrix (no Euler angles: turning over the top works).
    Home is the scanner's own view: set_home(T) with T the tracker pose (OpenCV camera -> world).
    With the depth camera's intrinsics (set_intrinsics) the lens is the scanner's own: the frustum is its picture's,
    off centre where the principal point is (the MIRACO's Pl.bin puts it at x 187 of 800), widened on the short side to
    the view's shape; so at Home the model sits exactly where the IR picture shows the part. Without them: a plain
    symmetric lens of fov_y degrees."""
    CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])    # OpenCV camera (x right, y down, z ahead) <-> OpenGL eye (y up, z back)

    def __init__(self):
        self.view = np.eye(4); self.pivot = np.zeros(3); self.fov_y = 45.0; self.near = 1.0; self.far = 5000.0
        self.home = None; self.bound = None; self.intr = None; self.home_rotation = None      # the user's own Home orientation (3x3), or None
        self.projection_mode = "perspective"

    def set_intrinsics(self, fx, fy, cx, cy, w, h):
        """The scanner's lens: focal lengths and principal point in pixels of its w x h picture."""
        self.intr = (float(fx), float(fy), float(cx), float(cy), float(w), float(h))
        self.fov_y = float(np.degrees(2.0 * np.arctan(h / 2.0 / fy)))

    def tan_rect(self, aspect):
        """The view's edges at distance 1 (left, right, bottom, top; OpenGL y up)."""
        if self.intr is None:
            t = np.tan(np.radians(self.fov_y) / 2.0); return -t * aspect, t * aspect, -t, t
        fx, fy, cx, cy, w, h = self.intr                       # pixel centres at whole numbers: the picture's edges at -0.5
        l, r, b, t = (-0.5 - cx) / fx, (w - 0.5 - cx) / fx, -(h - 0.5 - cy) / fy, (cy + 0.5) / fy
        xc, yc, hw, hh = (l + r) / 2, (b + t) / 2, (r - l) / 2, (t - b) / 2
        if aspect > hw / hh: hw = hh * aspect                  # the view is wider than the picture: more on both sides
        else: hh = hw / aspect                                 # taller: more above and below
        return xc - hw, xc + hw, yc - hh, yc + hh

    def set_home(self, T_cam_to_world, pivot):
        self.home = (np.asarray(T_cam_to_world, float).copy(), np.asarray(pivot, float).copy()); self.go_home()

    def go_home(self):
        if self.home is None: return
        T, c = self.home; self.view = self.CV2GL @ np.linalg.inv(T); self.pivot = c.copy()

    def fit_depth(self, lo, hi):
        """The data's bounding box: near and far planes follow it wherever the camera goes."""
        self.bound = ((np.asarray(lo, float) + hi) / 2.0, float(np.linalg.norm(np.asarray(hi, float) - lo)) / 2.0 or 1.0)

    def frame(self, lo, hi):
        """Look at a bounding box from the front (+z of the data towards the viewer), filling the view."""
        self.fit_depth(lo, hi); c, r = self.bound
        dist = r / np.sin(np.radians(self.fov_y) / 2.0)
        V = np.eye(4); R = self.home_rotation
        if R is not None: V[:3, :3] = np.asarray(R, float)                      # the saved Home orientation, fitted to this result
        V[:3, 3] = -V[:3, :3] @ c; V[2, 3] -= dist; self.view = V; self.pivot = c

    def set_projection_mode(self, mode):
        if mode not in ('perspective', 'orthographic'):
            raise ValueError('Choose Perspective or Orthographic.')
        self.projection_mode = mode

    def projection(self, aspect):
        if self.bound is not None:
            c, r = self.bound; d = -(self.view[2, :3] @ c + self.view[2, 3])           # eye depth of the data's centre
            self.near, self.far = max(d - 1.5 * r, 0.01 * r), max(d + 1.5 * r, 0.02 * r)
        l, r, b, t = self.tan_rect(aspect); n, fa = self.near, self.far
        if self.projection_mode == 'orthographic':
            distance = max(-(self.view[2, :3] @ self.pivot + self.view[2, 3]), 1e-6)
            l, r, b, t = np.asarray([l, r, b, t]) * distance
            return np.array([[2 / (r-l), 0, 0, -(r+l)/(r-l)],
                             [0, 2 / (t-b), 0, -(t+b)/(t-b)],
                             [0, 0, -2 / (fa-n), -(fa+n)/(fa-n)], [0, 0, 0, 1.]])

        return np.array([[2 / (r - l), 0, (r + l) / (r - l), 0], [0, 2 / (t - b), (t + b) / (t - b), 0],
                         [0, 0, (fa + n) / (n - fa), 2 * fa * n / (n - fa)], [0, 0, -1, 0]])

    def _about_pivot(self, R):
        pe = self.view[:3, :3] @ self.pivot + self.view[:3, 3]      # the pivot in eye space stays put
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pe - R @ pe
        self.view = M @ self.view

    def orbit(self, dx, dy, deg_per_px=0.35):
        a, b = np.radians(dx * deg_per_px), np.radians(dy * deg_per_px)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])       # about the screen's up
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])       # about the screen's right
        self._about_pivot(Rx @ Ry)

    def pan(self, dx, dy, vp_h, aspect=1.0):
        pe = self.view[:3, :3] @ self.pivot + self.view[:3, 3]
        l, r, b, t = self.tan_rect(aspect)
        s = max(-pe[2], 1e-6) * (t - b) / vp_h                  # world units per pixel at the pivot
        self.view[0, 3] += dx * s; self.view[1, 3] -= dy * s

    def zoom(self, steps, toward=None):
        """Dolly towards the pivot (or a picked world point), 12% of the distance per wheel step."""
        tgt = self.pivot if toward is None else np.asarray(toward, float)
        te = self.view[:3, :3] @ tgt + self.view[:3, 3]
        if self.projection_mode == 'orthographic':
            te[2] = self.view[2, :3] @ self.pivot + self.view[2, 3]
        self.view[:3, 3] -= te * (1.0 - 0.88 ** steps)

    def unproject(self, x, y, depth, vp_w, vp_h):
        """Widget pixel + depth-buffer value -> world point (for picking)."""
        ndc = np.array([2.0 * (x + 0.5) / vp_w - 1.0, 1.0 - 2.0 * (y + 0.5) / vp_h, 2.0 * depth - 1.0, 1.0])
        w = np.linalg.inv(self.projection(vp_w / vp_h) @ self.view) @ ndc
        return w[:3] / w[3]

    def project(self, P, vp_w, vp_h):
        """World points -> widget pixels (x, y) and eye depth; for lasso/box selection in Clean."""
        M = self.projection(vp_w / vp_h) @ self.view
        Q = P @ M[:3, :3].T + M[:3, 3]; w = P @ M[3, :3] + M[3, 3]
        x = (Q[:, 0] / w * 0.5 + 0.5) * vp_w; y = (0.5 - Q[:, 1] / w * 0.5) * vp_h
        return x, y, -(P @ self.view[2, :3] + self.view[2, 3])

    def select(self, P, mask, depth, tol):
        """Indices of the points of P inside mask (vp_h x vp_w bool, e.g. a filled lasso) that are on the surface as drawn,
        not hidden behind it. depth: SceneGL.read_depth_image of the same frame; tol in world units (about 2 point sizes)."""
        h, w = mask.shape
        x, y, ze = self.project(P, w, h)
        xi, yi = x.astype(np.int32), y.astype(np.int32)
        cand = np.nonzero((ze > 0) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))[0]
        cand = cand[mask[yi[cand], xi[cand]]]
        d = depth[yi[cand], xi[cand]].astype(np.float64); n, f = self.near, self.far
        zb = 2 * n * f / ((f + n) - (2 * d - 1) * (f - n))            # depth buffer -> eye distance
        if self.projection_mode == 'orthographic': zb = n + d * (f - n)
        return cand[(d < 1.0) & (ze[cand] <= zb + tol)]
