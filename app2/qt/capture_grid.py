# SPDX-License-Identifier: AGPL-3.0-or-later
"""The screenshot and recording grid on the Import page: clean tiles (picture, tick circle, time and size) and a viewer
that opens a picture large, with arrows to go through them."""
from PySide6.QtCore import Qt, QRect, QRectF, QSize, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap, QFont, QPainterPath
from PySide6.QtWidgets import QListWidget, QStyledItemDelegate, QStyle, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
from app2.qt.theme import T

CAP_W, CAP_H = 416, 234            # the size pictures are kept at (16:9); tiles scale them to fit
MIN_TILE = 236                      # the narrowest a tile gets before the grid drops a column
TEXT_H = 62                         # room under a tile's picture for the time and size


def _on(v): return getattr(v, 'value', v) == 2      # Qt.Checked, however the binding hands it back


class CapDelegate(QStyledItemDelegate):
    """One tile. The tick circle in the picture's corner is the only thing that ticks; a click anywhere else opens the viewer."""
    def __init__(self, parent=None):
        super().__init__(parent); self.ticked = False; self.tile_w = MIN_TILE

    def tile_h(self): return (self.tile_w - 24) * 9 // 16 + TEXT_H
    def sizeHint(self, option, index): return QSize(self.tile_w, self.tile_h())

    def _pic(self, rect):
        w = rect.width() - 16; return QRect(rect.x() + 8, rect.y() + 8, w, w * 9 // 16)

    def check_rect(self, rect):
        pic = self._pic(rect.adjusted(4, 4, -4, -4)); return QRect(pic.x() + 6, pic.y() + 6, 22, 22)

    def paint(self, p, opt, idx):
        r = opt.rect.adjusted(4, 4, -4, -4); on = _on(idx.data(Qt.CheckStateRole)); hover = bool(opt.state & QStyle.State_MouseOver)
        p.save(); p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor('#5b9df0') if on else QColor('#2e353f'), 1.4 if on else 1)); p.setBrush(QColor('#22344a') if on else QColor('#1f2a37') if hover else QColor('#181d23'))
        p.drawRoundedRect(QRectF(r).adjusted(.5, .5, -.5, -.5), 9, 9)
        pic = self._pic(r); pm = idx.data(Qt.DecorationRole)
        pm = pm.pixmap(CAP_W, CAP_H) if pm is not None and hasattr(pm, 'pixmap') else None
        if pm is not None and not pm.isNull(): pm = pm.scaled(pic.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        clip = QPainterPath(); clip.addRoundedRect(QRectF(pic), 6, 6); p.setClipPath(clip)
        if pm and not pm.isNull(): p.drawPixmap(pic.x() + (pic.width() - pm.width()) // 2, pic.y() + (pic.height() - pm.height()) // 2, pm)
        p.setClipping(False)
        if idx.data(Qt.UserRole + 1) == 'video':                          # a play mark over a recording's picture
            m = QRectF(pic.center().x() - 22, pic.center().y() - 22, 44, 44); p.setPen(Qt.NoPen); p.setBrush(QColor(13, 17, 22, 170)); p.drawEllipse(m)
            tri = QPainterPath(); tri.moveTo(m.x() + 17, m.y() + 12); tri.lineTo(m.x() + 17, m.y() + 32); tri.lineTo(m.x() + 33, m.y() + 22); tri.closeSubpath()
            p.setBrush(QColor('white')); p.drawPath(tri)
        c = self.check_rect(opt.rect)
        p.setPen(QPen(QColor('#dceaff') if on else QColor(255, 255, 255, 190), 1.6)); p.setBrush(QColor('#5b9df0') if on else QColor(13, 17, 22, 170)); p.drawEllipse(c)
        if on:
            p.setPen(QPen(QColor('white'), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            x, y = c.x(), c.y(); p.drawLine(x + 6, y + 11, x + 10, y + 15); p.drawLine(x + 10, y + 15, x + 16, y + 7)
        when, _, size = (idx.data(Qt.DisplayRole) or '').partition('\n')
        f = QFont(p.font()); f.setPointSize(10); f.setBold(True); p.setFont(f); p.setPen(QColor(T['ink']))
        p.drawText(QRect(r.x(), pic.bottom() + 6, r.width(), 18), Qt.AlignCenter, when)
        f.setBold(False); f.setPointSize(9); p.setFont(f); p.setPen(QColor(T['ink3']))
        p.drawText(QRect(r.x(), pic.bottom() + 24, r.width(), 16), Qt.AlignCenter, size)
        p.restore()

    def editorEvent(self, event, model, opt, idx):
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.check_rect(opt.rect).contains(event.position().toPoint()):
            now = _on(idx.data(Qt.CheckStateRole))
            model.setData(idx, Qt.Unchecked if now else Qt.Checked, Qt.CheckStateRole); self.ticked = True; return True
        return False


class CapList(QListWidget):
    """The grid: as many columns as fit at least MIN_TILE wide, tiles stretched to use the whole width (no blank strip)."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setViewMode(QListWidget.IconMode); self.setResizeMode(QListWidget.Adjust); self.setMovement(QListWidget.Static)
        self.setSpacing(0); self.setUniformItemSizes(True); self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.look = CapDelegate(self); self.setItemDelegate(self.look); self.setMouseTracking(True); self._fit()

    def _fit(self):
        w = max(MIN_TILE, self.viewport().width() - 14)          # a little kept back for the scrollbar
        cols = max(1, w // MIN_TILE); self.look.tile_w = w // cols
        self.setGridSize(QSize(self.look.tile_w, self.look.tile_h())); self.doItemsLayout()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._fit()


class CaptureViewer(QDialog):
    """A screenshot big, on its own: left and right go through them, space ticks it for import, Esc closes."""
    def __init__(self, items, start, get_bytes, is_ticked, set_ticked, parent=None):
        super().__init__(parent); self.setWindowTitle('Screenshot'); self.resize(1180, 760)
        self.items, self.i, self.get_bytes, self.is_ticked, self.set_ticked = items, start, get_bytes, is_ticked, set_ticked
        self.setStyleSheet('QDialog {background:#0d1116;} QLabel {color:%s; background:transparent;} QPushButton {padding:7px 16px;} QCheckBox {color:%s; spacing:8px;}' % (T['ink'], T['ink']))
        v = QVBoxLayout(self); v.setContentsMargins(14, 12, 14, 12); v.setSpacing(8)
        self.pic = QLabel(); self.pic.setAlignment(Qt.AlignCenter); self.pic.setMinimumSize(320, 200); v.addWidget(self.pic, 1)
        self.caption = QLabel(); self.caption.setAlignment(Qt.AlignCenter); v.addWidget(self.caption)
        row = QHBoxLayout(); prev, nxt, close = QPushButton('◀  Previous'), QPushButton('Next  ▶'), QPushButton('Close')
        self.tick = QCheckBox('Import this one')
        prev.clicked.connect(lambda: self.go(-1)); nxt.clicked.connect(lambda: self.go(1)); close.clicked.connect(self.accept); self.tick.toggled.connect(self._ticked)
        row.addWidget(prev); row.addWidget(nxt); row.addStretch(1); row.addWidget(self.tick); row.addStretch(1); row.addWidget(close); v.addLayout(row)
        self.pm = None; self.show_it()

    def show_it(self):
        rel, caption = self.items[self.i]; data = self.get_bytes(rel); pm = QPixmap(); pm.loadFromData(data) if data else None
        self.pm = pm if not pm.isNull() else None; self.caption.setText('%s   (%d of %d)' % (caption, self.i + 1, len(self.items)))
        self.tick.blockSignals(True); self.tick.setChecked(self.is_ticked(rel)); self.tick.blockSignals(False); self.fit()

    def fit(self):
        if self.pm: self.pic.setPixmap(self.pm.scaled(self.pic.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.pic.setText('Could not read this picture from the scanner.')

    def resizeEvent(self, e): super().resizeEvent(e); self.fit()

    def go(self, d):
        self.i = (self.i + d) % len(self.items); self.show_it()

    def _ticked(self, on): self.set_ticked(self.items[self.i][0], on)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left: self.go(-1)
        elif e.key() == Qt.Key_Right: self.go(1)
        elif e.key() == Qt.Key_Space: self.tick.toggle()
        else: super().keyPressEvent(e)
