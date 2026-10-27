# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera marks painted in screen pixels, after the camera image is scaled."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen, QPolygonF


_SPRITES = {}
_LAYERS = {}   # per painted widget: (key, dot layer image, the dots array it was drawn from)


def dot_sprite(rgb, width, stroke):
    """A round dot with a dark rim, drawn once per color and size."""
    key = (tuple(rgb), round(width, 1), round(stroke, 1))
    if key not in _SPRITES:
        from PySide6.QtGui import QImage, QPainter
        outer = (width + stroke) / 2
        side = int(np.ceil(outer * 2)) + 2
        img = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        q = QPainter(img); q.setRenderHint(QPainter.RenderHint.Antialiasing); q.setPen(Qt.PenStyle.NoPen)
        c = QPointF(side / 2, side / 2)
        q.setBrush(QColor(24, 28, 34)); q.drawEllipse(c, outer, outer)
        q.setBrush(QColor(*rgb)); q.drawEllipse(c, width / 2, width / 2)
        q.end()
        _SPRITES[key] = img
    return _SPRITES[key]


def paint(p, image_rect, data, options, font_height, ir=False):
    if not data:
        return
    w, h = data['size']
    if w <= 0 or h <= 0:
        return
    r = QRectF(image_rect)
    def point(xy):
        return QPointF(r.x() + float(xy[0]) * r.width() / w, r.y() + float(xy[1]) * r.height() / h)
    stroke = max(1., font_height / 12)
    dark = QColor(24, 28, 34)
    color = QColor(*data.get('color', (255, 220, 0)))
    p.save()
    p.setClipRect(r, Qt.ClipOperation.IntersectClip)
    p.setBrush(Qt.BrushStyle.NoBrush)
    def path(points, c, dashed=False, faint=False):
        if points is None:
            return
        values = np.asarray(points, float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all() or not len(values):
            return
        q = QPainterPath(point(values[0]))
        for xy in values[1:]:
            q.lineTo(point(xy))
        p.save()
        if faint:
            p.setOpacity(.65)
        pen = QPen(dark, stroke * 3)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(q)
        pen = QPen(c, stroke)
        if dashed:
            # Dash lengths follow the screen font, independent of image scale.
            pen.setDashPattern([font_height / (3 * stroke)] * 2)
        p.setPen(pen)
        p.drawPath(q)
        p.restore()
    if options.get('box', True) and not ir:
        path(data.get('box'), color)
    if options.get('depth_edge', True):
        # only where depth really stops: the camera's theoretical edge never matched, since depth thins out well inside it
        path(data.get('depth_outline'), QColor(235, 238, 242), dashed=True, faint=ir)
    if not ir and options.get('dots',True) and data.get('dots_xy') is not None:
        from app2.scan.range_guide import BAND_RGB
        from app2.scan.preview import REMOVED_RGB
        xy=np.asarray(data['dots_xy']);uv=np.asarray(data['dots_uv']);levels=np.asarray(data['dots_levels']);removed=np.asarray(data['dots_removed'])
        cell={'Sparse':12,'Normal':6,'Dense':3}.get(options.get('dot_density'),6)
        selected=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        selected &= np.arange(len(xy))%(cell//3)**2==0 if data.get('dots_random') else (uv[:,0]%cell==0)&(uv[:,1]%cell==0)
        width=font_height/{'Small':12,'Normal':8,'Large':5}.get(options.get('dot_size'),8)
        width=min(width,0.8*cell*r.width()/max(w,1))   # never wider than the gap to the next dot: small tiles drew dark blobs
        # Dots change with depth (about 3 a second) but the tile repaints on every color frame: draw them once
        # into a layer and reuse it until the depth, the tile size or the dot options change.
        key=(id(data['dots_xy']),round(r.width()),round(r.height()),round(width,2),round(stroke,2),options.get('dot_density'),bool(options.get('safe_palette')))
        layer=_LAYERS.get(id(p.device()))
        if layer is None or layer[0]!=key:
            from PySide6.QtGui import QImage
            W_,H_=max(1,round(r.width())),max(1,round(r.height()));canvas=np.zeros((H_,W_,4),np.uint8)
            for level,c in [(0,BAND_RGB['off']),(1,BAND_RGB['ok']),(2,BAND_RGB['best']),(3,REMOVED_RGB)]:
                take=selected&(removed if level==3 else (levels==level)&~removed)
                if not take.any():continue
                # one pre-drawn dot per color, stamped into the layer by numpy in one go: a drawImage call per dot in
                # Python took most of the window's time (profiled live)
                sprite=dot_sprite(c,width,min(stroke,width/3));side=sprite.width()
                px=np.frombuffer(sprite.constBits(),np.uint8,count=side*sprite.bytesPerLine()).reshape(side,sprite.bytesPerLine()//4,4)[:,:side]
                oy,ox=np.nonzero(px[:,:,3]);rgba=px[oy,ox]
                x0=np.rint(xy[take,0]*r.width()/w-side/2).astype(np.int64);y0=np.rint(xy[take,1]*r.height()/h-side/2).astype(np.int64)
                X=x0[:,None]+ox[None,:];Y=y0[:,None]+oy[None,:];inside=(X>=0)&(X<W_)&(Y>=0)&(Y<H_)
                canvas[Y[inside],X[inside]]=np.broadcast_to(rgba,X.shape+(4,))[inside]
            img=QImage(canvas.data,W_,H_,W_*4,QImage.Format.Format_ARGB32_Premultiplied).copy()
            layer=(key,img,data['dots_xy']);_LAYERS[id(p.device())]=layer   # keep the array so its id stays unique
        p.drawImage(r.topLeft(),layer[1])
    if not ir and data.get('missing_xy') is not None:
        points=QPolygonF([point(q) for q in data['missing_xy']])
        pen=QPen(QColor(255,140,225),stroke*2);pen.setCapStyle(Qt.PenCapStyle.RoundCap);p.setPen(pen);p.drawPoints(points)
    if not ir:
        if options.get('left_fov', False):
            path(data.get('left_fov'), QColor(160, 228, 248), dashed=True)
        if options.get('right_fov', False):
            path(data.get('right_fov'), QColor(255, 209, 163), dashed=True)
    aim = data.get('aim')
    if aim is not None and (ir or options.get('aim', True)) and np.isfinite(aim).all():
        c = point(aim)
        arm = font_height / 3
        for a, b in ((QPointF(c.x() - arm, c.y()), QPointF(c.x() + arm, c.y())),
                     (QPointF(c.x(), c.y() - arm), QPointF(c.x(), c.y() + arm))):
            p.setPen(QPen(dark, stroke * 3)); p.drawLine(a, b)
            p.setPen(QPen(color, stroke)); p.drawLine(a, b)
    center = data.get('part_center')
    if not ir and options.get('part_center', False) and center is not None and aim is not None:
        center = np.asarray(center, float)
        box = np.asarray(data.get('box'), float)
        if np.isfinite(center).all() and box.ndim == 2:
            near = np.linalg.norm(center - aim) <= .1 * np.min(np.ptp(box, axis=0))
            if near:
                c = point(aim); radius = font_height / 2
                p.setPen(QPen(dark, stroke * 3)); p.drawEllipse(c, radius, radius)
                p.setPen(QPen(color, stroke)); p.drawEllipse(c, radius, radius)
            else:
                path([center, aim], color)
                p.setPen(QPen(dark, stroke * 3)); p.setBrush(color)
                p.drawEllipse(point(center), font_height / 7, font_height / 7)
    p.restore()
