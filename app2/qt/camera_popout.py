# SPDX-License-Identifier: AGPL-3.0-or-later
from app2.scan.preview import distance_note
"""A movable, resizable companion window for all three live scanner feeds."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from app2.qt.widgets import ImageTile
from app2.qt.camera_options import CameraPreferences,CameraOptions,apply_frame,camera_notice


class CameraPopout(QWidget):
    closed = Signal()

    def __init__(self, parts, parent=None, prefs=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('PointYoink · Camera views')
        self.setMinimumSize(640, 360)
        self.resize(1100, 500)
        self.setStyleSheet('background:#191d23;')
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 12, 12, 12)
        self.prefs=prefs if prefs is not None else CameraPreferences(self)
        self.options=CameraOptions(self.prefs,turn=True)
        self._frame=None;self._full_frame=None;self._shown_color_token=0;self.prefs.changed.connect(self._refresh)
        self.grid = QGridLayout(); self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setSpacing(10)
        layout.addLayout(self.grid, 1)
        self.ir, self.depth, self.colour = (ImageTile(p) for p in parts)
        self.maximized_camera = None
        from app2.qt.camera_tiles import attach_headers
        attach_headers((self.ir,self.depth,self.colour),self.options,self.maximize_camera,restore=self.restore_cameras)
        layout.addWidget(self.options.edge_key)
        from PySide6.QtGui import QShortcut, QKeySequence
        self.camera_escape=QShortcut(QKeySequence(Qt.Key.Key_Escape),self)
        self.camera_escape.activated.connect(self.restore_cameras)
        self._arrange()

    def maximize_camera(self, kind):
        self.maximized_camera=kind
        self._arrange()

    def restore_cameras(self):
        self.maximized_camera = None
        self._arrange()

    def _arrange(self):
        tiles=dict(IR=self.ir,Depth=self.depth,Color=self.colour)
        chosen=self.maximized_camera or 'IR'
        order=[tiles[chosen]]+[t for k,t in tiles.items() if k!=chosen]
        for tile in tiles.values():
            self.grid.removeWidget(tile)
            tile.header.max_button.setChecked(tile.header.kind==self.maximized_camera)
        self.grid.addWidget(order[0],0,0,1,2)
        self.grid.addWidget(order[1],1,0)
        self.grid.addWidget(order[2],1,1)
        self.grid.setRowStretch(0,2);self.grid.setRowStretch(1,1)
        self.grid.setColumnStretch(0,1);self.grid.setColumnStretch(1,1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()

    def _refresh(self):
        latest=self._frame
        if self._full_frame is not None:self.set_live(self._full_frame,self.ir.parts,sample=False)
        if latest is not None and latest.colour_only:self.set_live(latest,self.ir.parts,sample=False)

    def set_live(self,frame,ir_parts,sample=True):
        self._frame=frame
        if not frame.colour_only:self._full_frame=frame
        self.options.notice.setText(camera_notice(frame,self.prefs))
        self.ir.parts=list(ir_parts) if self.prefs.ir=="Left" else [("IR: "+self.prefs.ir.lower(),None)];self.ir.update()
        self.ir.set_note(distance_note(frame))
        token=getattr(frame,'color_token',None)
        if token is None or token>=self._shown_color_token:
            if token is not None:self._shown_color_token=token
            self.options.turn.set_estimate(getattr(frame,'bed_angle',None) or (getattr(frame,'color_angle',None) if self.prefs.show.get('color_angle',False) else None))
            apply_frame(self.colour,frame,'color',self.prefs,sample=sample)
        if not frame.colour_only:
            apply_frame(self.ir,frame,'ir',self.prefs,sample=sample)
            apply_frame(self.depth,frame,'depth',self.prefs,sample=sample)

    def set_ring(self, ring): self.ir.set_ring(ring)

    def reset(self):
        self._shown_color_token=0
        for tile in (self.ir, self.depth, self.colour):
            tile.set_image(None); tile.set_note(None); tile.set_ring(None)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.closed.emit()
