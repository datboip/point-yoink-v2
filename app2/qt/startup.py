# SPDX-License-Identifier: AGPL-3.0-or-later
"""The startup splash: icon.png and the ring artwork inside one window."""
from pathlib import Path
import math
from app2.version import VERSION
from PySide6.QtCore import Qt, QEvent, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QPushButton

ICON = Path(__file__).resolve().parents[2]/'icon.png'


def splash_background():
    """Original _make_splash_bg color formula and 480 px dimensions."""
    import numpy as np
    yy, xx = np.mgrid[0:480, 0:480]
    d = np.sqrt((xx-240)**2+(yy-144)**2)
    t = np.clip(1-d/(480*.9), 0, 1)**1.5
    rgb = np.dstack((13+t*22, 16+t*24, 22+t*33)).astype(np.uint8)
    return QImage(rgb.data,480,480,1440,QImage.Format.Format_RGB888).copy()


class StartupOverlay(QWidget):
    """Real status with indeterminate progress; ready() hides without a delay."""
    dismissed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.background = splash_background(); self.logo = QPixmap(str(ICON))
        self.angle = 0; self.error = False
        self.status = QLabel('Opening project', self)
        self.status.setFixedSize(424,40); self.status.setWordWrap(False)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet('color:#8290a6;background:transparent;border:0;')
        self.skip = QPushButton('Open workspace', self); self.skip.setFixedSize(150,28)
        self.skip.clicked.connect(self.ready)
        self.timer = QTimer(self); self.timer.setInterval(40); self.timer.timeout.connect(self._tick)
        parent.installEventFilter(self); self.relayout()

    def relayout(self):
        self.setGeometry(self.parentWidget().rect())
        self.panel = QRectF((self.width()-480)/2,(self.height()-480)/2,480,480)
        x,y = int(self.panel.x()),int(self.panel.y())
        self.status.move(x+28,y+394); self.skip.move(x+165,y+441); self.update()

    def eventFilter(self, obj, event):
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize: self.relayout()
        return super().eventFilter(obj,event)

    def showEvent(self,event):
        self.relayout(); self.timer.start(); super().showEvent(event)

    def hideEvent(self,event):
        self.timer.stop(); super().hideEvent(event)

    def set_status(self, text):
        self.error = False; self.status.setText(self.status.fontMetrics().elidedText(str(text),Qt.TextElideMode.ElideRight,420)); self.status.setToolTip(str(text))
        if self.isVisible(): self.timer.start()
        self.update()

    def fail(self, text):
        """Open workspace always works, even if a device or project failed."""
        self.error = True; self.timer.stop()
        self.status.setText(self.status.fontMetrics().elidedText(str(text),Qt.TextElideMode.ElideRight,420)); self.status.setToolTip(str(text)); self.update()

    def ready(self):
        self.hide(); self.dismissed.emit()

    def _tick(self):
        self.angle = (self.angle+9)%360; self.update()

    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(),QColor('#0e1117')); p.drawImage(self.panel,self.background)
        p.save(); p.translate(self.panel.topLeft()); p.fillRect(QRectF(0,0,480,3),QColor('#4aa3ff'))
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor('#1b2230'),3))
        circle=QRectF(142,46,196,196); p.drawEllipse(circle)
        if not self.error:
            p.setPen(QPen(QColor('#4aa3ff'),3)); p.drawArc(circle,int((90-self.angle)*16),95*16)
        if not self.logo.isNull(): p.drawPixmap(QRectF(165,69,150,150),self.logo,QRectF(self.logo.rect()))
        font=QFont('Ubuntu'); font.setPixelSize(40); font.setBold(True); p.setFont(font); p.setPen(QColor('#e6edf6'))
        p.drawText(QRectF(20,245,440,50),Qt.AlignmentFlag.AlignCenter,'PointYoink')
        font.setPixelSize(12); font.setBold(False); p.setFont(font); p.setPen(QColor('#8290a6'))
        p.drawText(QRectF(20,294,440,30),Qt.AlignmentFlag.AlignCenter,'Y O I N K .   C L E A N .   K E E P .')
        p.fillRect(QRectF(115,384,250,5),QColor('#0c0f15'))
        if not self.error: p.fillRect(QRectF(115+(self.angle/360)*190,384,60,5),QColor('#4aa3ff'))
        font.setPixelSize(11); p.setFont(font); p.setPen(QColor("#8290a6"))
        p.drawText(QRectF(330,450,130,20),Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,"v" + VERSION)
        p.restore()


class EmptyViewBackground(QWidget):
    """Original draw_empty_state fading rings and _illustration preview cube."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(),QColor('#0a0c10')); p.setBrush(Qt.BrushStyle.NoBrush)
        cx,cy=self.width()/2,self.height()*.40
        scale=min(1.5,max(1.,1+(min(self.width(),self.height())-420)/700))
        far=max(math.hypot(cx-x,cy-y) for x in (0,self.width()) for y in (0,self.height()))
        r=26*scale*.9
        while r<far:
            t=max(0.,1-r/far)**1.8
            color=QColor(*(round(a+(b-a)*t) for a,b in zip((10,12,16),(30,38,52))))
            p.setPen(QPen(color,1)); p.drawEllipse(QPointF(cx,cy),r,r); r+=26*scale
        p.translate(cx-65*scale,cy-55*scale); p.scale(scale,scale)
        T,UR,LR,B,LL,UL,C=[QPointF(*v) for v in ((65,9),(105,32),(105,78),(65,101),(25,78),(25,32),(65,55))]
        mid=lambda a,b:(a+b)/2
        p.setPen(QPen(QColor('#2a3140'),1))
        for a,b,c,d in ((T,UL,UR,C),(T,UR,UL,C),(UL,LL,C,B),(UL,C,LL,B),(UR,LR,C,B),(UR,C,LR,B)):
            p.drawLine(mid(a,b),mid(c,d))
        p.setPen(QPen(QColor('#3a4556'),3)); corners=(T,UR,LR,B,LL,UL,T)
        for a,b in zip(corners,corners[1:]): p.drawLine(a,b)
        for vertex in (UL,UR,B): p.drawLine(C,vertex)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor('#4aa3ff')); p.drawEllipse(C,4.5,4.5)
