# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rectangle / lasso selection on the GPU: the same test as
app2.builder.select_core.Elements.select_screen, one vertex shader invocation per element, results by transform feedback (GL 3.0
core; no compute shaders in 3.3). Sub-pixel faces are still found: it tests each face's centroid, it does not need
the face to own a pixel.

Per element: project the centroid; inside the viewport and on a mask pixel -> selected (through); visible-only adds
select_core's "on the surface as drawn" (own depth plus local slope, or the 3x3 nearest plus the element's own slope)
and "faces the eye" (normal given). The reference image is made in numpy from the frame's depth (1.6 M pixels, a few
ms) and goes up as an RG32F texture, the mask as R8UI. Elements (centroids float32, normals int8) go up once per shown stage result."""
import ctypes
import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

VS = """#version 330 core
layout(location = 0) in vec3 a_x;
layout(location = 1) in vec3 a_n;
uniform mat4 u_M;
uniform vec3 u_eye;
uniform usampler2D u_mask;
uniform sampler2D u_zmin;
uniform int u_through, u_facing, u_has_n;
uniform float u_tol_mm, u_px_k, u_tol_px, u_reach;
uniform ivec2 u_size;
flat out uint v_sel;
void main() {
    vec4 c = u_M * vec4(a_x, 1.0);
    uint s = 0u;
    if (c.w > 1e-6) {
        float x = (c.x / c.w * 0.5 + 0.5) * float(u_size.x), y = (0.5 - c.y / c.w * 0.5) * float(u_size.y);
        ivec2 p = ivec2(floor(x), floor(y));
        if (p.x >= 0 && p.y >= 0 && p.x < u_size.x && p.y < u_size.y && texelFetch(u_mask, p, 0).r != 0u) {
            s = 1u;
            if (u_through == 0) {
                vec2 zz = texelFetch(u_zmin, p, 0).rg;
                float px = u_px_k * c.w, lim = zz.r;
                if (u_has_n == 1) {
                    vec3 d = u_eye - a_x;
                    float cs = dot(a_n, d) / max(length(a_n) * length(d), 1e-12);
                    float tn = sqrt(max(1.0 - cs * cs, 0.0)) / max(cs, 0.2);
                    lim = max(lim, zz.g + (1.5 * px + u_reach) * tn);
                    if (u_facing == 1 && cs <= 0.0) s = 0u;
                }
                if (c.w > lim + max(u_tol_mm, u_tol_px * px)) s = 0u;
            }
        }
    }
    v_sel = s;
    gl_Position = vec4(0.0);
}
"""


def _tf_program():
    vs = shaders.compileShader(VS, GL.GL_VERTEX_SHADER)
    p = GL.glCreateProgram(); GL.glAttachShader(p, vs)
    name = ctypes.create_string_buffer(b"v_sel")
    arr = (ctypes.POINTER(ctypes.c_char) * 1)(ctypes.cast(name, ctypes.POINTER(ctypes.c_char)))
    GL.glTransformFeedbackVaryings(p, 1, arr, GL.GL_INTERLEAVED_ATTRIBS)
    GL.glLinkProgram(p)
    if GL.glGetProgramiv(p, GL.GL_LINK_STATUS) != GL.GL_TRUE: raise RuntimeError(GL.glGetProgramInfoLog(p))
    return p


class GpuSelector:
    def __init__(self):
        self.prog = _tf_program(); self.n = 0
        self.u = {k: GL.glGetUniformLocation(self.prog, k) for k in
                  ("u_M", "u_eye", "u_mask", "u_zmin", "u_through", "u_facing", "u_has_n", "u_tol_mm", "u_px_k", "u_tol_px", "u_reach", "u_size")}
        self.vao = GL.glGenVertexArrays(1); self.vx, self.vn, self.out = GL.glGenBuffers(3)
        self.tex_mask, self.tex_z = GL.glGenTextures(2); self.has_n = False

    def set_elements(self, X, Nrm=None):
        """X (n,3) element positions in the engine's order (points, or face centroids); Nrm (n,3) unit or None."""
        X = np.ascontiguousarray(X, np.float32); self.n = len(X); self.has_n = Nrm is not None
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vx); GL.glBufferData(GL.GL_ARRAY_BUFFER, X.nbytes, X, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        Nq = np.zeros((self.n, 3), np.int8) if Nrm is None else np.clip(np.round(np.asarray(Nrm, np.float32) * 127), -127, 127).astype(np.int8)
        Nq = np.ascontiguousarray(Nq)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vn); GL.glBufferData(GL.GL_ARRAY_BUFFER, Nq.nbytes, Nq, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1, 3, GL.GL_BYTE, False, 0, ctypes.c_void_p(0))   # raw ints, as numpy
        GL.glBindVertexArray(0); GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, self.out)
        GL.glBufferData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 4 * self.n, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0)

    @staticmethod
    def zmin_image(depth, near, far):
        """The same (h,w,2) reference image as the numpy path (select_core.Elements.ref_image)."""
        from app2.builder.select_core import Elements
        return Elements.ref_image(depth, near, far)

    def _tex(self, tex, unit, a, ifmt, fmt, typ):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit); GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        h, w = a.shape[:2]; GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, typ, np.ascontiguousarray(a))
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

    def select(self, mask, view, proj, near, far, depth=None, through=False, tol_mm=0.35, tol_px=1.5, facing=True, reach_mm=0.0):
        """Same contract as Elements.select_screen -> bool (n,) in element order."""
        h, w = mask.shape; M = (np.asarray(proj, float) @ np.asarray(view, float)).astype(np.float32)
        GL.glUseProgram(self.prog)
        self._tex(self.tex_mask, 0, mask.astype(np.uint8), GL.GL_R8UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_BYTE)
        z = self.zmin_image(depth, near, far) if not through else np.zeros((1, 1, 2), np.float32)
        self._tex(self.tex_z, 1, z, GL.GL_RG32F, GL.GL_RG, GL.GL_FLOAT)
        u = self.u
        GL.glUniformMatrix4fv(u["u_M"], 1, True, M); GL.glUniform3fv(u["u_eye"], 1, np.linalg.inv(view)[:3, 3].astype(np.float32))
        GL.glUniform1i(u["u_mask"], 0); GL.glUniform1i(u["u_zmin"], 1); GL.glUniform1i(u["u_through"], 1 if through else 0)
        GL.glUniform1i(u["u_facing"], 1 if (facing and self.has_n) else 0); GL.glUniform1f(u["u_tol_mm"], tol_mm)
        GL.glUniform1i(u["u_has_n"], 1 if self.has_n else 0); GL.glUniform1f(u["u_reach"], float(reach_mm))
        GL.glUniform1f(u["u_px_k"], 2.0 / (float(proj[1][1]) * h)); GL.glUniform1f(u["u_tol_px"], tol_px); GL.glUniform2i(u["u_size"], w, h)
        GL.glEnable(GL.GL_RASTERIZER_DISCARD)
        GL.glBindVertexArray(self.vao); GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, self.out)
        GL.glBeginTransformFeedback(GL.GL_POINTS); GL.glDrawArrays(GL.GL_POINTS, 0, self.n); GL.glEndTransformFeedback()
        GL.glDisable(GL.GL_RASTERIZER_DISCARD); GL.glBindVertexArray(0)
        out = np.empty(self.n, np.uint32)
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, self.out)
        GL.glGetBufferSubData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0); GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0); GL.glUseProgram(0)
        return out.astype(bool)

    def delete(self):
        GL.glDeleteBuffers(3, [self.vx, self.vn, self.out]); GL.glDeleteTextures(2, [self.tex_mask, self.tex_z])
        GL.glDeleteVertexArrays(1, [self.vao])
