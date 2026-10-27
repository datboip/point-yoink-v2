# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only Build view for models already finished on the scanner."""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app2 import library_dir
from app2.builder import export_model
from app2.qt.build_view import BuildViewPanel
from app2.qt.stage_bridge import StageBridge
from app2.qt.theme import ui_font


def render_model_picture(path):
    """A bounded, unsaved picture for the Build view when OpenGL is unavailable."""
    from app2.builder.view_data import estimate_bytes, load_file
    if estimate_bytes(path) > 256 * 1024 ** 2: return None
    data = load_file(path)
    if data["kind"] == "points":
        from stages.preview import render_points
        points = data["P"]
        if not len(points): return None
        step = max(1, len(points) // 200_000)
        return np.asarray(render_points(points[::step], data["N"][::step], views=1))
    vertices, faces = data["V"], data["F"]
    if not len(vertices) or not len(faces): return None
    vertices = vertices.astype(np.float64, copy=True)
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    vertices *= .85 / max(float(np.abs(vertices).max()), 1e-9)
    faces = faces[::max(1, (len(faces) + 39_999) // 40_000)]
    import shade
    return np.asarray(shade.render(vertices, faces, size=(900, 600), grid=False, gizmo=False))


class ImportedModelsPage(QWidget):
    """The Build viewer with scanner side selection, without a stage host or edit tools."""
    changed = Signal()
    message = Signal(str, str)
    _export_done = Signal(object)
    _picture_done = Signal(object)

    def __init__(self, gl=False, parent=None):
        super().__init__(parent)
        self.gl = gl; self.project = None; self.sides = []; self.selected = None; self.request_id = 0
        self.loader = None; self.bridge = StageBridge(self); self.bridge.stage_view.connect(self._view_ready)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imported-export")
        self.picture_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imported-picture")
        self.export_future = None; self.ask_save = self._ask_save
        self._export_done.connect(self._finish_export)
        self._picture_done.connect(self._finish_picture)
        row = QHBoxLayout(self); row.setContentsMargins(14, 12, 14, 12); row.setSpacing(14)
        left = QFrame(); left.setProperty("role", "panel"); left.setFixedWidth(248)
        lv = QVBoxLayout(left); lv.setContentsMargins(12, 12, 12, 12); lv.setSpacing(8)
        title = QLabel("1  Choose a side"); title.setFont(ui_font(16, QFont.Weight.DemiBold)); lv.addWidget(title)
        note = QLabel("These models were finished on the scanner. Each side is a separate model.")
        note.setWordWrap(True); note.setProperty("role", "hint"); lv.addWidget(note)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_host = QWidget(); self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0); self.list_layout.setSpacing(8)
        scroll.setWidget(self.list_host); lv.addWidget(scroll, 1); row.addWidget(left)
        self.vp = BuildViewPanel(gl); self.vp.tools_w.hide(); self.vp.compare.hide(); self.vp.rebuild.hide(); self.vp.colour_btn.hide()
        self.vp.home.clicked.connect(self.home)
        row.addWidget(self.vp, 1)
        right = QFrame(); right.setProperty("role", "panel"); right.setFixedWidth(290)
        rv = QVBoxLayout(right); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(12)
        heading = QLabel("2  Inspect the model"); heading.setFont(ui_font(17, QFont.Weight.DemiBold)); rv.addWidget(heading)
        guide = QLabel("Drag to rotate, scroll to zoom, or use the corner cube to choose a view." if gl else
                       "This is a rendered preview. OpenGL is needed to rotate the model in 3D.")
        guide.setWordWrap(True); guide.setProperty("role", "hint"); rv.addWidget(guide)
        self.detail = QLabel(""); self.detail.setWordWrap(True); self.detail.setProperty("role", "hint"); rv.addWidget(self.detail)
        export_heading = QLabel("3  Export this side"); export_heading.setFont(ui_font(17, QFont.Weight.DemiBold)); rv.addWidget(export_heading)
        self.export_stl_b = QPushButton("Export STL"); self.export_stl_b.clicked.connect(self.export_stl); rv.addWidget(self.export_stl_b)
        self.export_as_b = QPushButton("Export..."); self.export_as_b.clicked.connect(self.open_export); rv.addWidget(self.export_as_b)
        context = QLabel("Export makes a new file from this scanner model. It does not process raw frames or combine sides. The imported project stays unchanged.")
        context.setWordWrap(True); context.setProperty("role", "hint"); rv.addWidget(context); self.context = context
        rv.addStretch(1); row.addWidget(right)
        self.buttons = []; self._select(None)

    def open_project(self, path, sides):
        self.project = path; self.sides = list(sides)
        from scanner.adopt_device_scan import needs_preparation
        self.context.setText("Export makes a new file from this scanner model. To build from its raw frames, choose Prepare on its project card."
                             if needs_preparation(path) else
                             "Export makes a new file from this scanner model. The imported project stays unchanged.")
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget(): item.widget().hide(); item.widget().deleteLater()
        self.buttons = []
        for i, side in enumerate(self.sides):
            card = QFrame(); card.setProperty("role", "panel")
            cv = QVBoxLayout(card); cv.setContentsMargins(8, 8, 8, 8); cv.setSpacing(5)
            pic = QLabel(); pic.setFixedHeight(90); pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(side["preview"])
            if not pix.isNull(): pic.setPixmap(pix.scaled(210, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else: pic.setText("No preview")
            cv.addWidget(pic)
            button = QPushButton(side["name"]); button.setAccessibleName("Open scanner side " + side["name"])
            button.setCheckable(True); button.clicked.connect(lambda _=False, n=i: self._select(n)); cv.addWidget(button)
            self.buttons.append(button); self.list_layout.addWidget(card)
        self.list_layout.addStretch(1)
        self._select(0 if self.sides else None)

    def clear_project(self):
        self.project = None; self.sides = []; self.request_id += 1; self._select(None)
        if self.loader: self.loader.clear()
        if self.gl: self.vp.view.clear()
        else: self.vp.main.set_image(None)

    def _select(self, index):
        self.request_id += 1
        self.selected = index
        for i, button in enumerate(self.buttons): button.setChecked(i == index)
        side = self.sides[index] if index is not None else None
        self.export_stl_b.setEnabled(bool(side and side["faces"]))
        self.export_as_b.setEnabled(bool(side))
        if not side:
            self.detail.setText(""); self.vp.main.set_chips([], center="Choose a scanner side")
            self.changed.emit(); return
        self.detail.setText(("Selected: " + side["name"] + "\n") + ("Finished mesh, ready to export." if side["faces"] else
                            "Point cloud only. PLY export is available; STL needs a finished mesh."))
        self.vp.main.set_chips([("Scanner model · " + side["name"], None), ("Read only", None)])
        if self.gl:
            if self.loader is None:
                from app2.builder.view_data import ViewLoader
                self.loader = ViewLoader(self.bridge)
            self.request_id += 1
            self.loader.request(dict(id=self.request_id, slot="main", items=[dict(key="main", path=side["path"])],
                                     edit=None, mode="shaded", stage="meshed", path=side["path"], readonly=True, check_mem=True))
        else:
            from PIL import Image
            try:
                with Image.open(side["preview"]) as picture: self.vp.main.set_image(np.asarray(picture.convert("RGB")))
            except (OSError, ValueError): self.vp.main.set_image(None)
            future = self.picture_executor.submit(render_model_picture, side["path"])
            request_id = self.request_id
            future.add_done_callback(lambda done, n=request_id: self._picture_done.emit((n, done)))
        self.changed.emit()

    def _finish_picture(self, result):
        request_id, future = result
        if request_id != self.request_id or self.selected is None: return
        try: picture = future.result()
        except Exception: picture = None
        if picture is not None: self.vp.main.set_image(picture)

    def _view_ready(self, out):
        if out.get("id") != self.request_id or self.selected is None: return
        if out.get("error") or out.get("too_big") or not out.get("items"):
            self.vp.view.clear()
            self.vp.main.set_chips([("Scanner model", None)], center="Could not show this model here.")
            return
        self.vp.view.show_items(out["items"], edit=None, elements=None, frame=True)
        self.vp.view.tools_on = False
        self.home()

    def home(self):
        if self.gl:
            self.vp.view.go_home()
            self.vp.view.cam.orbit(95, -65)
            self.vp.view.cam.zoom(-1.5)
            self.vp.view.camera_moved.emit()
            self.vp.view.update()

    def current_side(self):
        return self.sides[self.selected] if self.selected is not None and self.selected < len(self.sides) else None

    def can_export(self): return bool(self.current_side())

    def _ask_save(self, default, title, filter_text):
        library_dir.ensure_parent(default)
        return QFileDialog.getSaveFileName(self, title, default, filter_text)[0]

    def export_stl(self): self._choose_export("stl")

    def open_export(self):
        if not self.can_export(): return
        from app2.qt.export_sheet import ExportDialog
        dialog = self.__dict__.get("export_dialog")
        if dialog is None:
            dialog = self.export_dialog = ExportDialog(self.window())
            dialog.cancel.connect(dialog.close_sheet)
            dialog.ok.connect(lambda fmt, _zip: (dialog.close_sheet(), self._choose_export(fmt)))
        side = self.current_side()
        formats = [key for key, _label in export_model.formats()] if side["faces"] else ["ply"]
        dialog.present(formats, export_model.last_format() if export_model.last_format() in formats else formats[0], can_zip=False)
        dialog.set_dims("")

    def _choose_export(self, fmt):
        side = self.current_side()
        if not side or self.export_future is not None: return
        if not side["faces"] and fmt != "ply":
            self.message.emit("This side is a point cloud. Export it as PLY or finish the mesh on the scanner.", "warn"); return
        default = library_dir.export_path(side["name"] + "_scanner_model." + fmt)
        dst = self.ask_save(default, "Export scanner model", fmt.upper() + " files (*." + fmt + ")")
        if not dst: return
        if not dst.lower().endswith("." + fmt): dst += "." + fmt
        src = side["path"]
        if os.path.realpath(src) == os.path.realpath(dst):
            self.message.emit("Choose a different file so the scanner's model stays unchanged.", "warn"); return
        export_model.save_format(fmt)
        self.export_stl_b.setEnabled(False); self.export_as_b.setEnabled(False)
        self.message.emit("Exporting " + os.path.basename(dst) + "...", "info")

        def write():
            if not side["faces"] and fmt == "ply":
                temp = os.path.join(os.path.dirname(dst), "." + os.path.basename(dst) + ".part")
                try:
                    shutil.copyfile(src, temp); os.replace(temp, dst)
                finally:
                    if os.path.exists(temp): os.remove(temp)
            else:
                export_model.export(src, dst, fmt)
            return dst

        self.export_future = self.executor.submit(write)
        self.export_future.add_done_callback(lambda future: self._export_done.emit(future))

    def _finish_export(self, future):
        self.export_future = None
        side = self.current_side(); self.export_stl_b.setEnabled(bool(side and side["faces"])); self.export_as_b.setEnabled(bool(side))
        try: dst = future.result()
        except Exception as exc: self.message.emit("Export failed: " + str(exc).splitlines()[0][:160], "warn")
        else: self.message.emit("Saved " + os.path.basename(dst), "info")

    def shutdown(self):
        self.bridge.close()
        if self.loader: self.loader.close(0); self.loader = None
        self.picture_executor.shutdown(wait=False, cancel_futures=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
