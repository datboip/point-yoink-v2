#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare several versions of one model in ONE window: a button per model (or keys 1-9), and the camera stays where it
is when you switch, so the same spot can be looked at in every version.
usage: compare_view.py "Label A=a.ply" "Label B=b.ply" ...   (or just paths)
Mouse: left-drag tumbles freely (no stop at the poles), right/middle-drag pans, wheel zooms, H = home view.
Light: a headlight a little above-left of the eye, both faces lit (scans have back faces); the Light slider swings it
round the view, Brightness scales it. (pyqtgraph's own 'shaded' lights from (1,-1,-1): below and behind, seen as
"the light is weirdly coming from the bottom".)"""
import math, os, sys
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import numpy as np, open3d as o3d
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QQuaternion, QVector3D, QMatrix4x4
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders as S

VS = """
uniform mat4 u_mvp; uniform mat3 u_normal;
attribute vec4 a_position; attribute vec3 a_normal; attribute vec4 a_color;
varying vec4 v_color; varying vec3 v_normal;
void main() { v_normal = normalize(u_normal * a_normal); v_color = a_color; gl_Position = u_mvp * a_position; }
"""
FS = """
#ifdef GL_ES
precision mediump float;
#endif
uniform float u_light[3]; uniform float u_gain;   // pyqtgraph sets uniforms with glUniform1fv: float arrays only
varying vec4 v_color; varying vec3 v_normal;
void main() {
    vec3 n = normalize(v_normal); vec3 L = normalize(vec3(u_light[0], u_light[1], u_light[2]));
    float d = abs(dot(n, L));          // two-sided: back faces of a scan are lit too
    float h = abs(n.z);                                  // faces turned to the eye a touch brighter
    vec3 r = reflect(-L, n * sign(dot(n, L)));
    float s = pow(max(r.z, 0.0), 24.0) * 0.18;           // a small highlight shows the surface's roughness
    float k = (0.20 + 0.62 * d + 0.12 * h * h) * u_gain;
    gl_FragColor = vec4(v_color.rgb * k + s * u_gain, v_color.a);
}
"""
HEAD = S.ShaderProgram("pyhead", [S.VertexShader(VS), S.FragmentShader(FS)], uniforms={"u_light": [-0.35, 0.45, 1.0], "u_gain": [1.0]})


class TumbleView(gl.GLViewWidget):
    """A trackball camera: the view turns about the screen's own axes, so it never stops at a pole."""
    def __init__(self):
        super().__init__(); self.q = QQuaternion(); self.centre = QVector3D(0, 0, 0); self._last = None; self.home_q = QQuaternion()

    def viewMatrix(self):
        tr = QMatrix4x4(); tr.translate(0, 0, -self.opts["distance"]); tr.rotate(self.q); tr.translate(-self.centre); return tr

    def mousePressEvent(self, e): self._last = e.position()
    def mouseReleaseEvent(self, e): self._last = None

    def mouseMoveEvent(self, e):
        if self._last is None: self._last = e.position(); return
        d = e.position() - self._last; self._last = e.position()
        if e.buttons() & Qt.MouseButton.LeftButton:
            ang = math.hypot(d.x(), d.y()) * 0.4
            if ang: self.q = QQuaternion.fromAxisAndAngle(QVector3D(d.y(), d.x(), 0).normalized(), ang) * self.q
        else:
            k = self.opts["distance"] * 0.0012
            self.centre -= self.q.conjugated().rotatedVector(QVector3D(d.x() * k, -d.y() * k, 0))
        self.update()

    def wheelEvent(self, e): self.opts["distance"] *= 0.999 ** e.angleDelta().y(); self.update()

    def home(self): self.q = QQuaternion(self.home_q); self.centre = QVector3D(0, 0, 0); self.update()


def load(path, centre=None):
    m = o3d.io.read_triangle_mesh(path)
    V = np.asarray(m.vertices, np.float32); F = np.asarray(m.triangles, np.int32)
    if centre is None: centre = (V.min(0) + V.max(0)) / 2
    return V - centre, F, centre


def main():
    specs = [a.split("=", 1) if "=" in a else (os.path.basename(a), a) for a in sys.argv[1:]]
    if not specs: print(__doc__); return 2
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget(); w.setWindowTitle("PointYoink - compare"); w.resize(1400, 900)
    w.setStyleSheet("background:#0e1117; color:#ddd; font-size:14px;")
    lay = QtWidgets.QVBoxLayout(w); lay.setContentsMargins(8, 8, 8, 8)
    row = QtWidgets.QHBoxLayout(); lay.addLayout(row)
    view = TumbleView(); view.setBackgroundColor((14, 17, 23)); lay.addWidget(view, 1)
    bottom = QtWidgets.QHBoxLayout(); lay.addLayout(bottom)
    info = QtWidgets.QLabel(""); bottom.addWidget(info, 1)
    items, buttons, centre = [], [], None
    for k, (label, path) in enumerate(specs):
        V, F, centre = load(path, centre)                       # every version shares the first one's centre
        it = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=F), smooth=True, shader=HEAD, color=(0.62, 0.76, 1.0, 1.0))
        it.setVisible(k == 0); view.addItem(it); items.append((it, label, len(F)))
        b = QtWidgets.QPushButton("%d  %s" % (k + 1, label)); b.setCheckable(True); b.setChecked(k == 0)
        b.setStyleSheet("QPushButton{padding:6px 12px;border-radius:6px;background:#1c2230}"
                        "QPushButton:checked{background:#3a5a9a;color:white}")
        row.addWidget(b); buttons.append(b)
        if k == 0: view.opts["distance"] = float(np.linalg.norm(V.max(0) - V.min(0))) * 1.4
    row.addStretch(1)
    home = QtWidgets.QPushButton("Home (H)"); home.setStyleSheet("padding:6px 12px;border-radius:6px;background:#1c2230"); row.addWidget(home)
    home.clicked.connect(view.home); QtGui.QShortcut(QtGui.QKeySequence("H"), w, activated=view.home)

    def slider(name, lo, hi, val, fn):
        bottom.addWidget(QtWidgets.QLabel(name)); s = QtWidgets.QSlider(Qt.Orientation.Horizontal); s.setRange(lo, hi); s.setValue(val)
        s.setFixedWidth(180); s.valueChanged.connect(fn); bottom.addWidget(s); return s
    def set_light(a):                                       # swing the headlight round the view axis, 45 deg off it
        r = math.radians(a); HEAD.setUniformData("u_light", [math.cos(r) * 0.55, math.sin(r) * 0.55, 1.0]); view.update()
    def set_gain(v): HEAD.setUniformData("u_gain", [v / 100.0]); view.update()
    slider("Light", 0, 359, 128, set_light); slider("Brightness", 50, 180, 100, set_gain); set_light(128)

    def show(k):
        for j, (it, _, _) in enumerate(items): it.setVisible(j == k)
        for j, b in enumerate(buttons): b.setChecked(j == k)
        info.setText("%s   %d triangles     keys 1-%d switch (the camera stays put), H = home" % (items[k][1], items[k][2], len(items)))
    for k, b in enumerate(buttons):
        b.clicked.connect(lambda _=False, k=k: show(k))
        if k < 9: QtGui.QShortcut(QtGui.QKeySequence(str(k + 1)), w, activated=lambda k=k: show(k))
    show(0); w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
