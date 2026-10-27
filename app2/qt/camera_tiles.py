# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera headers reserve their own room above the complete image."""
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QEvent
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton, QFrame, QVBoxLayout, QCheckBox, QScrollArea
from app2.qt.choices import Choices
from app2.qt.camera_options import LABELS


class TurnReadout(QLabel):
    def __init__(self, parent=None):
        super().__init__('Bed not seen', parent)
        self.data = None
        self.estimate = None
        self.short = False
        self.setFixedWidth(self.fontMetrics().horizontalAdvance('~360°  '))   # stable during fast angle updates

    def set_data(self, data):
        self.data = data
        data = data or {}
        angle = data.get('live_angle_deg')
        if self.estimate is not None:
            self.setText('~%g°' % round(self.estimate['angle_deg']))
            self.setToolTip('Estimated angle from color dots on a recent measured bed plane. Color can guide tracking only after paired IR checks and a depth fit. Keep the camera still.')
            return
        degrees = data.get('degrees')
        self.setText('%g°' % round(angle) if angle is not None and data.get('bed_seen',True) else '')   # nothing, not a bare ?, until dots give the angle
        self.setToolTip(('Bed angle from observed dots.' if angle is not None else 'Bed angle unknown.')+' Kept views: %s. This is not surface coverage.' % ('unknown' if degrees is None else '%g°' % round(degrees)))

    def set_estimate(self, value):
        self.estimate=value
        self.set_data(self.data)


class MarksPopup(QFrame):
    def __init__(self, prefs, parent):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAccessibleName('Camera marks');self.setObjectName('MarksPopup')
        line=self.fontMetrics().lineSpacing()
        # the outline is the popup's own; labels and the scroll area inside stay plain (a bare QFrame rule boxed them all)
        self.setStyleSheet('QFrame#MarksPopup {background:#20252c;border:1px solid #46515e;border-radius:%dpx;} '
                           'QFrame#MarksPopup QLabel {color:#9aa6b4;border:0;background:transparent;} '
                           'QFrame#MarksPopup QCheckBox {color:#e7edf5;border:0;background:transparent;padding:0.2em 0;}' % (line//3))
        outer=QVBoxLayout(self);outer.setContentsMargins(line//2,line//2,line//2,line//2)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame);self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet('QScrollArea, QScrollArea > QWidget > QWidget {background:transparent;}')
        self.content=QWidget();self.scroll.setWidget(self.content);outer.addWidget(self.scroll)
        layout = QVBoxLayout(self.content);layout.setContentsMargins(0,0,0,0);layout.setSpacing(line//6)
        self.actions = {}
        for key, label in LABELS.items():
            b = QCheckBox(label)
            b.toggled.connect(lambda on, key=key: prefs.set_show(key, on))
            self.actions[key] = b
            layout.addWidget(b)
        for key, title, values, tip in (('dot_density', 'Dot density on the camera picture', ('Sparse','Normal','Dense'), 'How many depth dots are drawn over the camera picture. The 3D view is not affected.'),
                                        ('dot_size', 'Dot size (picture and 3D Dots style)', ('Small','Normal','Large'), 'Dot size over the camera picture, and of the 3D view dots when its style is Dots.')):
            layout.addSpacing(line//3);lab=QLabel(title);lab.setToolTip(tip);layout.addWidget(lab)
            choice = Choices();choice.setToolTip(tip)
            choice.addItems(values)
            choice.setCurrentText(prefs.show[key])
            choice.currentTextChanged.connect(lambda value, key=key: prefs.set_option(key,value))
            layout.addWidget(choice)
        layout.addSpacing(line//3)
        self.safe = QCheckBox('Color-blind safe bands')
        self.safe.toggled.connect(lambda on: prefs.set_option('safe_palette',on))
        layout.addWidget(self.safe)
        self.color_angle=QCheckBox('Color angle estimate')
        self.color_angle.setToolTip('Experimental, off by default. Display only. Needs a recently observed bed and a steady camera. Never changes scan poses.')
        self.color_angle.toggled.connect(lambda on:prefs.set_option('color_angle',on))
        layout.addWidget(self.color_angle)
        self.smooth_color=QCheckBox('Smooth color (runs ahead of depth)')
        self.smooth_color.setToolTip('Off: the color picture and the big view update with the depth frames, so the depth dots sit on the right pixels. On: color runs at its own 26 fps and the dots can trail it.')
        self.smooth_color.toggled.connect(lambda on:prefs.set_option('smooth_color',on))
        layout.addWidget(self.smooth_color)
        self.ir_refine=QCheckBox('Refine Both alignment')
        self.ir_refine.setToolTip('Experimental, off by default. A small image correction from matching dots, after three agreeing frames. Never changes depth or calibration.')
        self.ir_refine.toggled.connect(lambda on:prefs.set_option('ir_refine',on))
        layout.addWidget(self.ir_refine)
        prefs.changed.connect(lambda: self.sync(prefs))
        self.sync(prefs)

    def sizeHint(self):
        margins=self.layout().contentsMargins()
        hint=self.content.sizeHint().expandedTo(self.content.minimumSizeHint())
        for child in self.content.findChildren(Choices):   # Dot density and Dot size: every choice in full
            grid=child.layout();need=sum(b.sizeHint().width() for b in child.buttons)+grid.spacing()*(len(child.buttons)-1)
            m=grid.contentsMargins();hint.setWidth(max(hint.width(),need+m.left()+m.right()))
        # a scroll bar only when the screen is too short to show it all
        tall=hint.height()+margins.top()+margins.bottom()>self.maximumHeight()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if tall else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bar=self.scroll.verticalScrollBar().sizeHint().width() if tall else 0
        frame=self.scroll.frameWidth()*2+self.width()-self.contentsRect().width()   # the popup's outline and the scroll area's edge
        return QSize(hint.width()+margins.left()+margins.right()+bar+frame,hint.height()+margins.top()+margins.bottom()+frame)

    def sync(self, prefs):
        for key, b in self.actions.items():
            b.blockSignals(True); b.setChecked(prefs.show[key]); b.blockSignals(False)
        self.safe.blockSignals(True); self.safe.setChecked(prefs.show['safe_palette']); self.safe.blockSignals(False)
        self.color_angle.blockSignals(True);self.color_angle.setChecked(prefs.show.get('color_angle',False));self.color_angle.blockSignals(False)
        self.ir_refine.blockSignals(True);self.ir_refine.setChecked(prefs.show.get('ir_refine',False));self.ir_refine.blockSignals(False)
        self.smooth_color.blockSignals(True);self.smooth_color.setChecked(prefs.show.get('smooth_color',False));self.smooth_color.blockSignals(False)


class CameraHeader(QWidget):
    maximize = Signal()
    restore = Signal()

    def __init__(self, tile, kind, options=None):
        super().__init__(tile)
        self.kind = kind
        self.tile = tile
        self.setObjectName('CameraHeader')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setStyleSheet('#CameraHeader {background:#20262e;color:#e7edf5;} QToolButton {background:#29323d;color:#e7edf5;border:1px solid #46515e;border-radius:4px;padding:0.18em;} QToolButton:focus {border-color:#e7edf5;}')
        self.row = QHBoxLayout(self)
        gap = max(1, self.fontMetrics().lineSpacing()//8)
        side = max(gap, self.fontMetrics().lineSpacing()//3)     # the name and fps sat on the tile edge
        self.row.setContentsMargins(side,gap,side,gap)
        self.row.setSpacing(max(gap, self.fontMetrics().lineSpacing()//4))
        self.name = QLabel(kind)
        self.row.addWidget(self.name)
        self.ir = None
        self.turn = None
        if kind == 'IR':
            self.ir = options.ir
            self.row.addWidget(self.ir)
            self.turn = options.turn
            self.row.addWidget(self.turn)
        if kind == 'Color':
            self.row.addWidget(options.show)
            self.marks = options.show
        self.row.addStretch(1)
        self.rate = QLabel('')
        self.rate.setToolTip('Received camera frames per second')
        self.rate.setMinimumWidth(self.rate.fontMetrics().horizontalAdvance('00.0 fps '))
        self.rate.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        self.row.addWidget(self.rate)
        self.max_button = QToolButton()
        icon=QIcon(str(Path(__file__).with_name('assets')/'camera-maximize.svg'))
        icon.addFile(str(Path(__file__).with_name('assets')/'camera-restore.svg'),QSize(),QIcon.Mode.Normal,QIcon.State.On)
        self.max_button.setIcon(icon)
        self.max_button.setIconSize(QSize(self.fontMetrics().lineSpacing(),self.fontMetrics().lineSpacing()))
        self.max_button.setCheckable(True)
        self.max_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.max_button.setAccessibleName('Maximize '+kind)
        self.max_button.setToolTip('Make '+kind+' the large view. It stays there until you pick another camera. Esc goes back to IR.')
        self.max_button.clicked.connect(self.maximize)
        self.row.addWidget(self.max_button)
        for child in self.findChildren(QWidget):child.installEventFilter(self)
        self._compact = False
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        tile.header = self
        self.show()

    def eventFilter(self, watched, event):
        if watched.window() is not self.window():return super().eventFilter(watched,event)
        if event.type()==QEvent.Type.KeyPress:
            if event.key()==Qt.Key.Key_Escape:
                self.restore.emit();event.accept();return True
            if watched is self.max_button and event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
                self.max_button.click();event.accept();return True
        return super().eventFilter(watched,event)

    def refresh(self):
        rate = self.tile.fps()
        self.rate.setText('' if rate is None else ('%.0f' if self._compact else '%.1f fps') % rate)

    def adapt(self, width):
        # Full controls are used whenever their measured hints fit.
        if self.ir is not None:self.ir.abbreviate(False)
        self.name.setText(self.kind)
        if hasattr(self,'marks'):self.marks.setText('Show marks')
        if self.turn is not None:
            self.turn.short=True
            self.turn.set_data(self.turn.data)
            self.turn.show()
        self._compact = False
        self.rate.setFixedWidth(self.rate.fontMetrics().horizontalAdvance('00.0 fps '))
        self.refresh()
        compact = self.row.sizeHint().width() > width
        self._compact = compact
        if compact:
            self.rate.setFixedWidth(self.rate.fontMetrics().horizontalAdvance('00 '))
            self.name.setText(self.kind[:1])
            if self.ir is not None:self.ir.abbreviate(True)
            if hasattr(self,'marks'):self.marks.setText('M')
        self.refresh()
        height = max(self.max_button.sizeHint().height(), self.fontMetrics().lineSpacing()) + self.row.contentsMargins().top()+self.row.contentsMargins().bottom()
        from app2.qt.widgets import RING_WIDTH   # inside the state ring, so its top edge stays visible
        self.setGeometry(RING_WIDTH,RING_WIDTH,width-2*RING_WIDTH,height)


def attach_headers(tiles, options, maximize, popout=None, restore=None):
    for tile, kind in zip(tiles, ('IR','Depth','Color')):
        h = CameraHeader(tile,kind,options)
        h.maximize.connect(lambda kind=kind: maximize(kind))
        if restore is not None:h.restore.connect(restore)
    if popout is not None:
        # Pop out stays with the camera controls, without a separate row.
        tiles[0].header.row.insertWidget(tiles[0].header.row.count()-2,popout)
        popout.setText('')
        popout.setIcon(QIcon(str(Path(__file__).with_name('assets')/'camera-popout.svg')))
        line=popout.fontMetrics().lineSpacing()
        popout.setIconSize(QSize(line,line))
