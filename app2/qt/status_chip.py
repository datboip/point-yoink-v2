# SPDX-License-Identifier: AGPL-3.0-or-later
"""One steady line for live status, with details on hover."""
from PySide6.QtCore import Qt,QSize,QRectF
from PySide6.QtGui import QPainter,QColor
from PySide6.QtWidgets import QLabel,QSizePolicy


def state_color(state):return dict(ok='#4cc86a',warn='#f2b43a',problem='#ff5a4f',idle='#a3aab3').get(state,'#a3aab3')


def paint_status(p,rect,metrics,state,text):
    line=metrics.lineSpacing();color=QColor(state_color(state));bg=QColor(color);bg.setAlpha(22);p.setPen(Qt.PenStyle.NoPen);p.setBrush(bg);p.drawRoundedRect(rect,line/4,line/4);p.setPen(color)
    mark={'ok':'✓','warn':'!','problem':'!','idle':'○'}.get(state,'○');p.drawText(QRectF(rect.left(),rect.top(),line*1.5,rect.height()),Qt.AlignmentFlag.AlignCenter,mark)
    words=QRectF(rect.left()+line*1.5,rect.top(),max(0,rect.width()-line*2),rect.height())
    lines=max(1,int(rect.height()//line))
    if lines<2:p.drawText(words,Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft,metrics.elidedText(text,Qt.TextElideMode.ElideRight,int(words.width())));return
    p.drawText(words,Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft,'\n'.join(wrap(metrics,text,words.width(),lines)))


def wrap(metrics,text,width,lines):
    """Text in at most `lines` lines no wider than `width`, split evenly: never one lonely word on the last line.
    The narrowest width that still fits in that many lines sets the breaks; only text that cannot fit is cut short."""
    def fill(limit):
        out=[];rest=text.split()
        while rest:
            cur=rest.pop(0)
            while rest and metrics.horizontalAdvance(cur+' '+rest[0])<=limit:cur+=' '+rest.pop(0)
            out.append(cur)
        return out
    out=fill(width)
    if len(out)>lines:
        out=out[:lines];out[-1]=metrics.elidedText(out[-1]+' '+' '.join(fill(width)[lines:]),Qt.TextElideMode.ElideRight,int(width));return out
    lo,hi=0,int(width)
    while lo<hi:                                      # the narrowest limit that keeps the same number of lines
        mid=(lo+hi)//2
        if len(fill(mid))<=len(out):hi=mid
        else:lo=mid+1
    return fill(hi)


class LiveStatus(QLabel):
    def __init__(self,parent=None,lines=1):
        # lines: the text rows it always keeps room for, so longer words wrap instead of being cut and nothing jumps
        super().__init__(parent);self.state='idle';self.setMinimumWidth(1);self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        line=self.fontMetrics().lineSpacing();self.setFixedHeight(round(line*(lines+.7)))
    def sizeHint(self):return QSize(self.fontMetrics().horizontalAdvance('Show the part'),self.height())
    def minimumSizeHint(self):return QSize(1,self.height())
    def set_status(self,state,text,details=''):
        if not hasattr(self,'steady'):
            from app2.qt.steady import Steady
            self.steady=Steady(self._show_status,self)
        self.steady.offer((state,text),state,text,details,urgent=state=='problem')
    def _show_status(self,state,text,details):
        self.state=state;self.setText(text);self.setToolTip(details or text);self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);paint_status(p,QRectF(self.rect()),self.fontMetrics(),self.state,self.text())
