# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capture-distance key shared by the Scan and Build model views."""
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QFontMetrics,QColor,QPainter
from app2.qt.theme import ui_font,qc
from app2.scan.range_guide import BAND_RGB,UNSEEN_RGB


def layout(width,height,font=None,bottom_inset=0):
    font=font or ui_font();fm=QFontMetrics(font);line=fm.lineSpacing();pad=line/2
    entries=[('Off',BAND_RGB['off']),('OK',BAND_RGB['ok']),('Best',BAND_RGB['best']),('No saved view',UNSEEN_RGB)]
    rows=[];x=pad;y=line+pad
    for text,color in entries:
        w=line+fm.horizontalAdvance(text)+pad
        if x>pad and x+w>width-pad:x=pad;y+=line
        rows.append((text,color,QRectF(x,y,w,line)));x+=w
    h=y+line+pad;top=max(0,height-bottom_inset-h-pad)
    return QRectF(pad,top,max(0,width-2*pad),h),[(t,c,r.translated(pad,top)) for t,c,r in rows]


def paint(widget,painter=None,pending=False):
    own=painter is None;p=painter or QPainter(widget);font=ui_font();p.setFont(font);fm=QFontMetrics(font);line=fm.lineSpacing()
    box,entries=layout(widget.width(),widget.height(),font,getattr(widget,"capture_key_inset",0))
    p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(10,12,15,240));p.drawRoundedRect(box,line/3,line/3)
    p.setPen(qc('ink'));p.drawText(box.adjusted(line/2,0,0,0),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop,'Distance'+(' (updating)' if pending else ''))
    for text,color,rect in entries:
        p.fillRect(QRectF(rect.left(),rect.top()+line/4,line/2,line/2),QColor(*color));p.setPen(qc('ink'))
        p.drawText(rect.adjusted(line,0,0,0),Qt.AlignmentFlag.AlignVCenter,text)
    if own:p.end()
