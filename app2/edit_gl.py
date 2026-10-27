# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build step 3D drawing: Qt-free, OpenGL 3.3 core, on top of app2.gl_core.

What it adds to gl_core's SceneGL (which stays the Scan step's live view, untouched):
  items   any number of meshes and point clouds, each with its own model matrix (Line-up: one per side; nudge / flip
          previews move the matrix, nothing is sent again) and its own colour (side colour) or per-element colours.
  flags   one byte per element (face of a mesh, point of a cloud) in a buffer of its own:
            bit 0  "would go" (red): a tool's Detect, a box / sphere / lasso selection, the platter cut
            bit 2  hover preview (amber), never a committed selection
            bit 1  "picked" (accent blue): the side picked in Line-up, a hovered piece
          Mesh faces read their byte in the fragment shader by gl_PrimitiveID from a texture buffer, so the index
          buffer is the engine's own face order: flag i IS engine face i (removed.npy of a Detect goes up as it is).
          Changing the selection sends n bytes (10 MB for 10 M faces), never positions.
  shading key light above-left of the eye + headlight fill + a little shine; back faces a dull violet-grey, so an
          inside-out patch or a second skin shows at a glance; optional flat facets (dFdx/dFdy) for Mesh Quality /
          Simplify inspection. Points are round, lit splats of world size.
  widgets lines for the box and sphere tools and the clip plane, drawn twice: solid where in front of the model,
          faint where behind it.
"""
import ctypes, os
import numpy as np
from OpenGL import GL
from app2 import gl_core

def _rgb(hexs): return np.float32([int(hexs[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _tok(name, fallback):
    try:
        from app2.qt.theme import T                      # plain dict, no Qt needed
        return _rgb(T.get(name, fallback))
    except Exception: return _rgb(fallback)


RED = _tok("lost", "#ff5a4f")                     # would go / selected
HOVER = _tok("warn", "#f2b43a")
ACCENT = _tok("accent", "#5b9df0")                # picked
PLATTER = _tok("platter", "#b27cff")              # the platter cut (Side step)
BASE = _tok("mesh", "#c4c8ce")                    # plain model: a light neutral grey, so red and blue read on it
BACK = _tok("backface", "#5c546e")                # back faces
CLOUD = _tok("cloud", "#4f8fe8")                  # points

_LIGHT = """
uniform vec3 u_red, u_accent, u_back, u_hover;
uniform int u_redbit_colour;                         // 0: red, 1: platter violet (the Side step's platter cut)
uniform vec3 u_platter;
vec3 lit(vec3 base, vec3 n, vec3 p) {
    vec3 v = normalize(-p);
    vec3 L = normalize(vec3(-0.35, 0.55, 1.0));      // key: above left of the eye
    float key = max(dot(n, L), 0.0), head = max(dot(n, v), 0.0);
    float s = pow(max(dot(reflect(-L, n), v), 0.0), 32.0);
    return base * (0.14 + 0.52 * key + 0.34 * head) + 0.10 * s;
}
vec3 flagged(vec3 c, uint f) {
    if ((f & 1u) != 0u) return mix(c, (u_redbit_colour == 1 ? u_platter : u_red) * (0.45 + 0.75 * max(max(c.r, c.g), c.b)), 0.85);
    if ((f & 4u) != 0u) return mix(c, u_hover * (0.45 + 0.75 * max(max(c.r, c.g), c.b)), 0.85);
    if ((f & 2u) != 0u) return mix(c, u_accent * (0.45 + 0.75 * max(max(c.r, c.g), c.b)), 0.6);
    return c;
}
"""
MESH_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_nrm;
uniform mat4 u_mv, u_proj;
layout(location = 2) in vec3 a_rgb;
out vec3 v_col;
out vec3 v_n; out vec3 v_p;
void main() {
    vec4 p = u_mv * vec4(a_pos, 1.0);
    v_p = p.xyz; v_n = mat3(u_mv) * a_nrm; v_col = a_rgb;
    gl_Position = u_proj * p;
}
"""
MESH_FS = """#version 330 core
in vec3 v_n; in vec3 v_p; in vec3 v_col;
uniform int u_use_vertex_rgb;
uniform usamplerBuffer u_flags;      // R8UI, one per face (gl_PrimitiveID)
uniform samplerBuffer u_fcol;        // RGBA8, one per face; used when u_use_fcol
uniform int u_use_fcol, u_use_flags, u_flat, u_only_marked;
uniform bool u_capture_colors;
uniform vec3 u_colour;
out vec4 frag;
""" + _LIGHT + """
void main() {
    if (u_only_marked == 1 && (u_use_flags == 0 || (texelFetch(u_flags, gl_PrimitiveID).r & 1u) == 0u)) discard;
    if (u_use_flags == 1 && (texelFetch(u_flags, gl_PrimitiveID).r & 8u) != 0u) discard;      // deleted, rebuild pending
    vec3 n = u_flat == 1 ? normalize(cross(dFdx(v_p), dFdy(v_p))) : normalize(v_n);
    if (u_flat == 0 && !gl_FrontFacing) n = -n;
    if (u_flat == 1 && dot(n, v_p) > 0.0) n = -n;
    vec3 base = u_use_fcol == 1 ? texelFetch(u_fcol, gl_PrimitiveID).rgb : (u_use_vertex_rgb == 1 ? v_col : u_colour);
    if (!gl_FrontFacing && u_use_vertex_rgb == 0 && !u_capture_colors) base = u_back;
    vec3 c = u_capture_colors ? base : lit(base, n, v_p);
    uint f = u_use_flags == 1 ? texelFetch(u_flags, gl_PrimitiveID).r : 0u;
    frag = vec4(flagged(c, f), 1.0);
}
"""
POINT_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec3 a_rgb;
layout(location = 3) in float a_flag;
uniform mat4 u_mv, u_proj;
uniform float u_size, u_vp_h;
uniform int u_use_rgb;
uniform vec3 u_colour;
out vec3 v_n; out vec3 v_p; out vec3 v_col; flat out uint v_flag;
void main() {
    vec4 p = u_mv * vec4(a_pos, 1.0);
    v_p = p.xyz; v_n = mat3(u_mv) * a_nrm; v_col = u_use_rgb == 1 ? a_rgb : u_colour; v_flag = uint(a_flag + 0.5);
    gl_Position = u_proj * p;
    gl_PointSize = clamp(u_size * u_proj[1][1] * 0.5 * u_vp_h / max(gl_Position.w, 1e-6), 1.0, 32.0);
}
"""
POINT_FS = """#version 330 core
in vec3 v_n; in vec3 v_p; in vec3 v_col; flat in uint v_flag;
uniform int u_only_marked;
uniform bool u_capture_colors;
out vec4 frag;
""" + _LIGHT + """
void main() {
    if (u_only_marked == 1 && (v_flag & 1u) == 0u) discard;
    if ((v_flag & 8u) != 0u) discard;                // deleted, rebuild pending
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (dot(q, q) > 1.0) discard;                    // round splats
    vec3 n = normalize(v_n); if (dot(n, v_p) > 0.0) n = -n;
    frag = vec4(flagged(u_capture_colors ? v_col : lit(v_col, n, v_p), v_flag), 1.0);
}
"""
LINE_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_mv, u_proj;
void main() { gl_Position = u_proj * u_mv * vec4(a_pos, 1.0); }
"""
LINE_FS = """#version 330 core
uniform vec4 u_rgba;
out vec4 frag;
void main() { frag = u_rgba; }
"""
EMESH_ATTRS = (("pos", 0, 3, GL.GL_FLOAT, False, np.float32), ("nrm", 1, 3, GL.GL_FLOAT, False, np.float32), ("rgb", 2, 3, GL.GL_UNSIGNED_BYTE, True, np.uint8))
EPOINT_ATTRS = (("pos", 0, 3, GL.GL_FLOAT, False, np.float32), ("nrm", 1, 3, GL.GL_FLOAT, False, np.float32),
                ("rgb", 2, 3, GL.GL_UNSIGNED_BYTE, True, np.uint8), ("flag", 3, 1, GL.GL_UNSIGNED_BYTE, False, np.uint8))
LINE_ATTRS = (("pos", 0, 3, GL.GL_FLOAT, False, np.float32),)


class TexBuf:
    """A texture buffer (GL 3.1 core): one texel per face, fetched by gl_PrimitiveID."""

    def __init__(self, fmt):
        self.fmt = fmt; self.buf = GL.glGenBuffers(1); self.tex = GL.glGenTextures(1); self.cap = 0; self.n = 0

    def upload(self, a):
        a = np.ascontiguousarray(a)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, self.buf)
        if a.nbytes > self.cap:
            self.cap = a.nbytes; GL.glBufferData(GL.GL_TEXTURE_BUFFER, a.nbytes, a, GL.GL_DYNAMIC_DRAW)
        else: GL.glBufferSubData(GL.GL_TEXTURE_BUFFER, 0, a.nbytes, a)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.tex); GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, self.fmt, self.buf)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0); self.n = len(a)

    def update_range(self, a, lo, hi):
        """Send only elements lo..hi (a lasso that touched one corner of a 10 M face mesh sends that span)."""
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, self.buf)
        GL.glBufferSubData(GL.GL_TEXTURE_BUFFER, lo * a.itemsize, (hi - lo) * a.itemsize, np.ascontiguousarray(a[lo:hi]))
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)

    def bind(self, unit):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit); GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.tex)

    def delete(self):
        GL.glDeleteTextures(1, [self.tex]); GL.glDeleteBuffers(1, [self.buf])


class Item:
    """One mesh or cloud: kind "mesh" | "points", its GPU buffers, model matrix (object -> world), colour, flags."""

    def __init__(self, kind):
        self.kind = kind; self.model = np.eye(4); self.colour = BASE.copy(); self.visible = True
        self.n = 0; self.fcol = None; self.flags_on = False
        if kind == "mesh":
            self.d = gl_core.Drawable(EMESH_ATTRS, True); self.flags = TexBuf(GL.GL_R8UI)
        else:
            self.d = gl_core.Drawable(EPOINT_ATTRS, False); self.flags = None
        self.use_rgb = False; self.use_vertex_rgb = False

    def delete(self):
        self.d.delete()
        if self.flags: self.flags.delete()
        if self.fcol: self.fcol.delete()


class EditScene:
    """Everything the Build step's view draws. Create with a context current; draw() with it current."""

    def __init__(self):
        self.mesh_prog = gl_core._program(MESH_VS, MESH_FS); self.point_prog = gl_core._program(POINT_VS, POINT_FS)
        self.line_prog = gl_core._program(LINE_VS, LINE_FS)
        self.lines = gl_core.Drawable(LINE_ATTRS, False); self.line_parts = []     # [(first, count, rgba)]
        self.items = {}; self.point_size = 0.55; self.flat = False; self.redbit_platter = False; self.only_marked = False; self.mesh_points = False
        self.background = (0.047, 0.055, 0.067)
        self._u = {}
        for p in (self.mesh_prog, self.point_prog, self.line_prog):
            n = GL.glGetProgramiv(p, GL.GL_ACTIVE_UNIFORMS)
            self._u[p] = {}
            for i in range(n):
                name = GL.glGetActiveUniform(p, i)[0]; name = name.decode() if isinstance(name, bytes) else name
                self._u[p][name] = GL.glGetUniformLocation(p, name)
        self.max_texbuf = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_BUFFER_SIZE))

    # ---------------- data in (context current) ----------------
    def set_mesh(self, key, V, N, F, face_rgb=None, model=None, colour=None, vertex_rgb=None):
        """V (n,3) float32, N (n,3) float32 vertex normals, F (m,3) uint32 in the engine's face order."""
        if len(F) > self.max_texbuf: raise ValueError("%d faces > GL_MAX_TEXTURE_BUFFER_SIZE %d" % (len(F), self.max_texbuf))
        it = self.items.get(key)
        if it is None or it.kind != "mesh": it = self.items[key] = Item("mesh")
        it.use_vertex_rgb = vertex_rgb is not None
        it.d.upload(pos=V, nrm=N, rgb=np.zeros((len(V),3),np.uint8) if vertex_rgb is None else vertex_rgb, faces=F, static=True); it.n = len(F)
        it.flags.upload(np.zeros(len(F), np.uint8)); it.flags_on = False
        self.set_face_colours(key, face_rgb)
        if model is not None: it.model = np.asarray(model, float)
        if colour is not None: it.colour = np.float32(colour)
        return it

    def set_face_colours(self, key, face_rgb):
        """(m,3) uint8 per face (sides, scan here) or None for the item's plain colour."""
        it = self.items[key]
        if face_rgb is None: it.use_rgb = False; return
        if it.fcol is None: it.fcol = TexBuf(GL.GL_RGBA8)
        a = np.empty((len(face_rgb), 4), np.uint8); a[:, :3] = face_rgb; a[:, 3] = 255
        it.fcol.upload(a); it.use_rgb = True

    def set_points(self, key, P, N, rgb=None, model=None, colour=None):
        it = self.items.get(key)
        if it is None or it.kind != "points": it = self.items[key] = Item("points")
        n = len(P)
        it.d.upload(pos=P, nrm=N, rgb=rgb if rgb is not None else np.zeros((n, 3), np.uint8), flag=np.zeros(n, np.uint8), static=True)
        it.n = n; it.use_rgb = rgb is not None
        if model is not None: it.model = np.asarray(model, float)
        if colour is not None: it.colour = np.float32(colour)
        return it

    def set_flags(self, key, flags, span=None):
        """flags (n,) uint8 in the item's element order (bit 0 would go, bit 1 picked); span=(lo, hi) sends only that
        range. None clears (no fetch at all in the shader)."""
        it = self.items[key]
        if flags is None:
            if it.kind == "points" and it.flags_on: it.d.upload(flag=np.zeros(it.n, np.uint8))    # points read the attribute
            it.flags_on = False; return
        if it.kind == "mesh":
            if span: it.flags.update_range(flags, *span)
            else: it.flags.upload(flags)
        else:
            it.d.upload(flag=flags)
        it.flags_on = True

    def remove(self, key):
        it = self.items.pop(key, None)
        if it: it.delete()

    def set_lines(self, parts):
        """Widget lines: [(segments (k,2,3) float32 world, rgba)]. Drawn over the model: solid in front, faint behind."""
        segs = []; self.line_parts = []; at = 0
        for s, rgba in parts:
            s = np.asarray(s, np.float32).reshape(-1, 3); segs.append(s); self.line_parts.append((at, len(s), rgba)); at += len(s)
        if segs: self.lines.upload(pos=np.vstack(segs))
        else: self.lines.count = 0

    # ---------------- draw ----------------
    def _common(self, prog, mv, pj):
        u = self._u[prog]; GL.glUseProgram(prog)
        GL.glUniformMatrix4fv(u["u_mv"], 1, True, mv); GL.glUniformMatrix4fv(u["u_proj"], 1, True, pj)
        if "u_red" in u:
            g = lambda k: u.get(k, -1)                  # a uniform a shader does not use is optimised away
            GL.glUniform3fv(g("u_red"), 1, RED); GL.glUniform3fv(g("u_accent"), 1, ACCENT); GL.glUniform3fv(g("u_back"), 1, BACK)
            GL.glUniform3fv(g("u_hover"), 1, HOVER); GL.glUniform3fv(g("u_platter"), 1, PLATTER); GL.glUniform1i(g("u_redbit_colour"), 1 if self.redbit_platter else 0)
        if "u_capture_colors" in u: GL.glUniform1i(u["u_capture_colors"],int(getattr(self,"capture_colors",False)))
        if "u_only_marked" in u: GL.glUniform1i(u["u_only_marked"], int(self.only_marked))
        return u

    def draw(self, view, proj, vp_w, vp_h):
        GL.glViewport(0, 0, vp_w, vp_h)
        GL.glClearColor(*self.background, 1.0); GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDepthFunc(GL.GL_LESS); GL.glDisable(GL.GL_BLEND); GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        pj = np.ascontiguousarray(proj, np.float32)
        for it in self.items.values():
            if not it.visible or not it.n: continue
            mv = np.ascontiguousarray(view @ it.model, np.float32)
            if it.kind == "mesh":
                u = self._common(self.mesh_prog, mv, pj)
                it.flags.bind(0); GL.glUniform1i(u["u_flags"], 0)
                if it.fcol: it.fcol.bind(1)
                GL.glUniform1i(u["u_fcol"], 1)
                GL.glUniform1i(u["u_use_vertex_rgb"], 1 if it.use_vertex_rgb else 0)
                GL.glUniform1i(u["u_use_fcol"], 1 if it.use_rgb else 0); GL.glUniform1i(u["u_use_flags"], 1 if it.flags_on else 0)
                GL.glUniform1i(u["u_flat"], 1 if self.flat else 0); GL.glUniform3fv(u["u_colour"], 1, it.colour)
            else:
                u = self._common(self.point_prog, mv, pj)
                GL.glUniform1f(u["u_size"], self.point_size); GL.glUniform1f(u["u_vp_h"], float(vp_h))
                GL.glUniform1i(u["u_use_rgb"], 1 if it.use_rgb else 0); GL.glUniform3fv(u["u_colour"], 1, it.colour)
            if it.kind == "mesh" and self.mesh_points:
                # Keep an opaque depth surface for visible-only selection and picking.
                # Rendering vertices is a display choice; face IDs remain unchanged.
                GL.glColorMask(False, False, False, False)
                try: it.d.draw()
                finally: GL.glColorMask(True, True, True, True)
                GL.glDepthFunc(GL.GL_LEQUAL); GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)
                GL.glDisable(GL.GL_PROGRAM_POINT_SIZE); GL.glPointSize(3.0); GL.glDepthMask(GL.GL_FALSE)
                try: it.d.draw()
                finally:
                    GL.glDepthMask(GL.GL_TRUE); GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
                    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE); GL.glDepthFunc(GL.GL_LESS)
            else: it.d.draw()
        if self.lines.count:
            mv = np.ascontiguousarray(view, np.float32); u = self._common(self.line_prog, mv, pj)
            GL.glEnable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glBindVertexArray(self.lines.vao)
            for func, fade in ((GL.GL_GREATER, 0.3), (GL.GL_LEQUAL, 1.0)):      # behind the model faint, in front solid
                GL.glDepthFunc(func); GL.glDepthMask(GL.GL_FALSE)
                for first, count, rgba in self.line_parts:
                    GL.glUniform4f(u["u_rgba"], rgba[0], rgba[1], rgba[2], rgba[3] * fade)
                    GL.glDrawArrays(GL.GL_LINES, first, count)
            GL.glDepthMask(GL.GL_TRUE); GL.glDepthFunc(GL.GL_LESS); GL.glDisable(GL.GL_BLEND); GL.glBindVertexArray(0)
        GL.glActiveTexture(GL.GL_TEXTURE0); GL.glUseProgram(0)

    read_depth_image = gl_core.SceneGL.read_depth_image


# ---------------- widget geometry (world-space line segments) ----------------
def box_segments(T, lo, hi):
    """The 12 edges of box [lo, hi] in its frame, T box -> world."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    c = c @ T[:3, :3].T + T[:3, 3]
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.array([[c[a], c[b]] for a, b in E], np.float32)


def sphere_segments(c, r, view, k=96):
    """Three great circles plus the outline as seen from the eye (the circle a sphere shows on screen)."""
    t = np.linspace(0, 2 * np.pi, k + 1); cs, sn = np.cos(t), np.sin(t); out = []
    R = np.linalg.inv(view)[:3, :3]                                   # eye axes in world
    for a, b in ((R[:, 0], R[:, 1]), (np.array([1., 0, 0]), np.array([0., 1, 0])), (np.array([0., 1, 0]), np.array([0., 0, 1])),
                 (np.array([1., 0, 0]), np.array([0., 0, 1]))):
        P = c + r * (np.outer(cs, a) + np.outer(sn, b))
        out.append(np.stack([P[:-1], P[1:]], 1))
    return np.vstack(out).astype(np.float32)
