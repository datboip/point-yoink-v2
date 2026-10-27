# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared camera preferences and compact controls for Scan and its pop-out."""
import math
from PySide6.QtCore import QObject, Signal, Qt, QSize, QRectF
from PySide6.QtGui import QFontMetrics, QPainter, QPen, QColor, QAction
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QToolButton, QMenu
from app2.scan import settings
from app2.qt.theme import ui_font, qc

DEFAULTS = dict(dots=True, box=True, aim=True, part_center=False, depth_edge=True, left_fov=False, right_fov=False)
LABELS = dict(dots='Depth dots', box='Part box', aim='Aim mark', part_center='Part center',
              depth_edge='Depth edge', left_fov='Left IR view', right_fov='Right IR view')


class CameraPreferences(QObject):
    changed = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        saved = settings.load()
        values = saved.get('camera_show', {})
        self.show = {k: values.get(k, v) if isinstance(values.get(k, v), bool) else v for k, v in DEFAULTS.items()} if isinstance(values, dict) else dict(DEFAULTS)
        self.show.update(dot_density=saved.get('dot_density','Normal'),dot_size=saved.get('dot_size','Normal'),safe_palette=bool(saved.get('safe_palette',False)),color_angle=bool(saved.get('color_angle',False)),ir_refine=bool(saved.get('ir_refine',False)),smooth_color=bool(saved.get('smooth_color',False)),view_style=saved.get('view_style','Surface') if saved.get('view_style') in ('Surface','Dots') else 'Surface')
        from app2.scan.range_guide import set_palette
        set_palette(self.show['safe_palette'])
        self.ir = saved.get('ir_camera', 'Left')
        if self.ir not in ('Left', 'Right', 'Both'):
            self.ir = 'Left'

    def set_show(self, name, on):
        if name not in DEFAULTS or self.show[name] == bool(on):
            return
        self.show[name] = bool(on)
        settings.save(camera_show=dict(self.show))
        self.changed.emit()

    def set_option(self,name,value):
        if self.show.get(name)==value:return
        self.show[name]=value;settings.save(**{name:value})
        if name=='safe_palette':
            from app2.scan.range_guide import set_palette
            set_palette(value)
        self.changed.emit()

    def set_ir(self, name):
        if name not in ('Left', 'Right', 'Both') or name == self.ir:
            return
        self.ir = name
        settings.save(ir_camera=name)
        self.changed.emit()


class TurnRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.setToolTip('Kept frames around the learned bed. Gaps need more views. Color is the best distance band reached. Gray means a kept view without saved distance evidence. This is not proof of complete surface coverage.')

    def sizeHint(self):
        side = self.fontMetrics().lineSpacing() * 4
        return QSize(side, side)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        from app2.scan.range_guide import BAND_RGB,UNSEEN_RGB
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        line = self.fontMetrics().lineSpacing(); stroke = line / 3
        r = QRectF(self.rect()); side = min(r.width(), r.height()) - line / 2
        ring = QRectF(r.center().x() - side / 2, r.center().y() - side / 2, side, side)
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(qc('line'), stroke)); p.drawEllipse(ring)
        data = self.data or {}; levels = data.get('levels', [])
        for i, level in enumerate(levels):
            if level == -1:
                continue
            color = QColor(*(UNSEEN_RGB if level==-2 else BAND_RGB[('off','ok','best')[level]]))
            pen = QPen(color, stroke); pen.setCapStyle(Qt.PenCapStyle.FlatCap); p.setPen(pen)
            # A small screen-space gap keeps individual observed slices visible.
            gap = min(360 / len(levels) / 4, stroke / max(side, 1) * 180 / math.pi)
            p.drawArc(ring, round((90 - i * 360 / len(levels) - gap / 2) * 16), -round((360 / len(levels) - gap) * 16))
        quality=data.get('quality_done_share')
        if quality is not None:
            inner=ring.adjusted(stroke*2,stroke*2,-stroke*2,-stroke*2);p.setPen(QPen(QColor(*BAND_RGB['best']),stroke/2));p.drawArc(inner,90*16,-round(360*quality*16))
        degrees = data.get('degrees')
        text = ('Bed not\nseen' if not data.get('bed_seen',degrees is not None) else 'Ready') if degrees is None else '%g°' % round(degrees)
        angle=data.get('live_angle_deg')
        if angle is not None:
            phase=math.radians(angle-(data.get('start_angle_deg') or 0)-90);center=ring.center();radius=side/2
            from PySide6.QtCore import QPointF
            p.setPen(QPen(qc('ink'),stroke/2));p.drawLine(QPointF(center.x()+math.cos(phase)*(radius-stroke*2),center.y()+math.sin(phase)*(radius-stroke*2)),QPointF(center.x()+math.cos(phase)*(radius+stroke),center.y()+math.sin(phase)*(radius+stroke)))
        if data.get('start_angle_deg') is not None:
            p.setPen(QPen(qc('accent'),stroke/2));p.drawLine(ring.center().x(),ring.top()-stroke,ring.center().x(),ring.top()+stroke)
        p.setFont(ui_font()); p.setPen(qc('ink2')); p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


class CameraOptions(QWidget):
    """Preference controls hosted by the tiles, plus the shared line key."""
    def __init__(self, prefs, parent=None, turn=False):
        super().__init__(parent)
        from app2.qt.choices import Choices
        from app2.qt.camera_tiles import MarksPopup, TurnReadout
        self.prefs = prefs
        self.ir = Choices()
        self.ir.addItems(('Left','Right','Both'))
        self.ir.setAccessibleName('IR camera')
        self.ir.setToolTip('Left and Right: each IR camera, green where there is depth. Both: blue where the two cameras agree, orange where both see it but differ, gray where only the left camera can be placed.\nShortcut: I cycles Left, Right, Both.')
        self.ir.currentTextChanged.connect(prefs.set_ir)
        self.show = QToolButton()
        self.show.setText('Show marks')
        self.show.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show.setToolTip('Choose camera marks. Depth edge applies to IR and Color together.')
        self.menu = MarksPopup(prefs,self.show)
        self.actions = self.menu.actions
        self.safe = self.menu.safe
        def open_marks():
            from PySide6.QtCore import QPoint
            area=self.show.screen().availableGeometry()
            self.menu.setMaximumHeight(area.height()-self.fontMetrics().lineSpacing())
            self.menu.adjustSize()
            position=self.show.mapToGlobal(QPoint(0,self.show.height()))
            position.setX(max(area.left(),min(position.x(),area.right()-self.menu.width()+1)))
            position.setY(max(area.top(),min(position.y(),area.bottom()-self.menu.height()+1)))
            self.menu.move(position)
            self.menu.show()
            self.actions['dots'].setFocus()
        self.show.clicked.connect(open_marks)
        self.turn = TurnReadout() if turn else None
        self.notice = QLabel('')
        self.edge_key = QLabel('Dashed line: where depth stops. Pink: no depth there.')
        self.edge_key.setToolTip('Dashed line: where depth stops on the part. Pink: the part is there but has no depth, usually dark, shiny or steep surfaces. Try another angle or more IR gain. These marks are only drawn on screen.')
        self.edge_key.setWordWrap(False)
        self.edge_key.setMinimumWidth(1)
        self.edge_key.setFixedHeight(self.edge_key.fontMetrics().lineSpacing())
        prefs.changed.connect(self.sync)
        self.sync()
        self.hide()

    def sync(self):
        self.ir.blockSignals(True); self.ir.setCurrentText(self.prefs.ir); self.ir.blockSignals(False)
        self.menu.sync(self.prefs)


def color_image(frame, prefs):
    clean=getattr(frame,'colour_clean',None)
    return clean if clean is not None else frame.colour


def apply_frame(tile, frame, kind, prefs, sample=True):
    from app2.scan.stereo_view import select_ir
    if kind=='ir':
        image,guides=select_ir(frame,prefs.ir)
        tile.set_center(None if image is not None else 'Right IR was not saved' if prefs.ir!='Left' else 'Waiting for IR')
        tile.set_image(image,sample=sample);tile.set_guides(guides,prefs.show,ir=True)
    elif kind=='color':
        tile.set_image(color_image(frame,prefs),sample=sample)
        tile.set_guides(getattr(frame,'colour_guides',None),prefs.show)
    else:
        tile.set_image(frame.depth,sample=sample);tile.set_guides(None)


def camera_notice(frame,prefs):
    if prefs.ir=='Left':return 'IR: left'
    if getattr(frame,'ir_right',None) is None:return 'Right IR was not saved'
    if prefs.ir=='Right':
        return 'IR: right' if getattr(frame,'right_guides',None) is not None else 'IR: right, calibration marks unavailable'
    if getattr(frame,'ir_both',None) is None:return 'Both needs depth and stereo calibration'
    return 'Both: blue where the two IR cameras agree, orange where both see it but differ, gray where only the left camera can be placed'
