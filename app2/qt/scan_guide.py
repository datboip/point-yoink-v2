# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paint suggested scanner bodies, view cones and selected surface spots in the Build view."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFontMetrics


def lines(pose,lens):
    T=np.asarray(pose['T'],float); eye=T[:3,3]; R=T[:3,:3]
    corners=[]
    z=min(float(lens.get('far',400)),float(pose['distance'])+50)
    for u,v in ((0,0),(lens['width'],0),(lens['width'],lens['height']),(0,lens['height'])):
        corners.append(eye+R@np.array([(u-lens['cx'])*z/lens['fx'],(v-lens['cy'])*z/lens['fy'],z]))
    body=np.array([[x,y,z] for x,y,z in ((-55,-25,-25),(55,-25,-25),(55,25,-25),(-55,25,-25),(-55,-25,0),(55,-25,0),(55,25,0),(-55,25,0))])@R.T+eye
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(body[a],body[b],False) for a,b in edges]+[(eye,c,True) for c in corners]+[(corners[i],corners[(i+1)%4],True) for i in range(4)]


def display_points(view,points):
    points=np.asarray(points,float)
    if not view.items:return points
    M=next(iter(view.items.values())).get("model")
    if M is None:return points
    M=np.asarray(M,float)
    return points@M[:3,:3].T+M[:3,3]


def paint(view):
    guide=getattr(view,'scan_guide',None)
    if not guide: return
    p=QPainter(view.overlay);p.setRenderHint(QPainter.RenderHint.Antialiasing)
    width,height=max(1,view.width()),max(1,view.height())
    for pose in guide['poses']:
        col=QColor(pose['color'])
        for a,b,cone in (lines(pose,guide['lens']) if getattr(view, "guide_ghosts", True) else []):
            x,y,z=view.cam.project(display_points(view,[a,b]),width,height)
            if not (z>0).all() or not np.isfinite(x).all():continue
            p.setPen(QPen(col,1 if cone else 2,Qt.PenStyle.DashLine if cone else Qt.PenStyle.SolidLine))
            p.drawLine(QPointF(x[0],y[0]),QPointF(x[1],y[1]))
        # Highlight the exact vertices reached by this suggestion.
        if view.items:
            it=next(iter(view.items.values())); V=it.get('V',it.get('P'))
            idx=np.asarray(pose['indices'],int)
            idx=idx[idx<len(V)][::max(1,len(idx)//500)]
            if len(idx):
                x,y,z=view.cam.project(display_points(view,np.asarray(V)[idx]),width,height);p.setPen(QPen(col,3))
                for xx,yy,zz in zip(x,y,z):
                    if zz>0 and np.isfinite(xx):p.drawPoint(QPointF(xx,yy))
    p.setPen(QColor('#f0f2f5'))
    p.drawText(QPointF(15,75),'What is left: '+('Scanner moves' if guide.get('mode')=='handheld' else 'Table turns'))
    row_y=95
    for color,text in (('#ff4632','Not scanned yet (holes)'),('#b45ae6','Rough or doubled, scan again'),('#46d778','Done')):
        p.setPen(QColor(color));p.drawText(QPointF(15,row_y),text);row_y+=20
    row_y+=5
    for i,pose in enumerate(guide['poses']):
        p.setPen(QColor(pose['color']));text=f'{pose["rank"]}. {pose["text"]}'
        box=p.boundingRect(QRectF(15,row_y,width-30,10000),Qt.TextFlag.TextWordWrap.value,text)
        p.drawText(QRectF(15,row_y,width-30,box.height()+2),Qt.TextFlag.TextWordWrap.value,text);row_y+=box.height()+8
    if guide.get('reason'):
        p.setPen(QColor('#ffbe55'));p.drawText(QPointF(15,55),guide['reason'])
    if guide.get('needs_flip'):
        p.setPen(QColor('#ffbe55'));p.drawText(QPointF(15,55),'Bed side needs the part flipped')
    p.end()


def show(view,guide,frame=False):
    view.scan_guide=guide
    view._guide_part_box = view._box
    view._guide_all_box = view._box
    if guide and guide.get('poses') and view._box is not None:
        points=np.array([p for pose in guide['poses'] for a,b,_ in lines(pose,guide['lens']) for p in (a,b)])
        points=display_points(view,points)
        lo,hi=view._box
        view._guide_all_box=(np.minimum(lo,points.min(0)),np.maximum(hi,points.max(0)))
        if frame:view.cam.frame(*view._guide_part_box)
        # Keep all cones within the camera's depth range without changing zoom.
        view.cam.fit_depth(*view._guide_all_box)
    view.overlay.update();view.update()


def frame_view(view, everything=False):
    box = getattr(view, "_guide_all_box" if everything else "_guide_part_box", None)
    if box is not None:
        view.cam.frame(*box)
        depth = getattr(view, "_guide_all_box", None)
        if depth is not None: view.cam.fit_depth(*depth)
        view.overlay.update(); view.update()


def touch_legend_height(overlay):
    from app2.qt.theme import ui_font
    return 6*(QFontMetrics(ui_font(12)).height()+5)+26


def guidance_rect(overlay):
    from app2.qt.theme import ui_font
    h=2*QFontMetrics(ui_font(14)).height()+6
    legend=touch_legend_height(overlay) if getattr(overlay,'touch',None) is not None else 0
    return QRectF(180 if not legend else 14,overlay.height()-legend-h-12,max(40,overlay.width()-(194 if not legend else 28)),h)
