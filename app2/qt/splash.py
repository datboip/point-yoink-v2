# SPDX-License-Identifier: AGPL-3.0-or-later
"""The start-up splash: the logo in a turning ring, the wordmark, a progress bar and what is loading, like the Tk app's.
Shown before the heavy imports (Open3D, the GPU view, the main window); main() calls step() between them, which also
turns the ring, since the event loop is not running yet."""
import os, time
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QRadialGradient, QFont, QGuiApplication, QCursor
from PySide6.QtWidgets import QWidget, QApplication
from app2.qt.theme import T, ui_font

SIZE = 480
MIN_SHOW_S = 2.0          # shortest time on screen
MIN_SPIN_S = 1.2          # smooth turning after loading
TAGLINE = "Y O I N K .   C L E A N .   K E E P ."


class Splash(QWidget):
    def __init__(self, icon_path, version):
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(SIZE, SIZE); self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.logo = QPixmap(icon_path) if os.path.exists(icon_path) else QPixmap()
        self.version, self.status, self.frac, self.t0 = version, "Getting things ready...", 0.0, time.monotonic()
        scr = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()      # the monitor the pointer is on
        g = scr.availableGeometry(); self.move(g.center().x() - SIZE // 2, g.center().y() - SIZE // 2)

    def step(self, status, frac):
        """Show what is loading now (frac 0..1 of the bar) and let the splash repaint."""
        self.status, self.frac = status, max(self.frac, frac)
        self.update(); QApplication.processEvents()

    def hold(self, min_s=MIN_SHOW_S, spin_s=MIN_SPIN_S):
        """Keep the splash up, the ring turning smoothly, for spin_s after loading and min_s in all (the ring cannot turn
        while the imports hold the thread, so a fast PC would otherwise see a still ring for a blink)."""
        end = max(self.t0 + min_s, time.monotonic() + spin_s)
        while time.monotonic() < end:
            self.update(); QApplication.processEvents(); time.sleep(1 / 60)

    def finish(self, window):
        window.raise_(); window.activateWindow(); self.close()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W = H = SIZE; cx, cy = W / 2, H * 0.30
        g = QRadialGradient(QPointF(cx, cy), W * 0.9)                        # soft glow behind the logo
        g.setColorAt(0, QColor(35, 40, 55)); g.setColorAt(1, QColor(T["window"])); p.fillRect(self.rect(), g)
        p.fillRect(QRectF(0, 0, W, 3), QColor(T["accent"]))                  # accent hairline
        R = 98; ring = QRectF(cx - R, cy - R, 2 * R, 2 * R)
        p.setPen(QPen(QColor("#1b2230"), 3)); p.drawEllipse(ring)             # faint full track
        a = (time.monotonic() - self.t0) * 360.0                            # one turn a second, from the clock
        pen = QPen(QColor(T["accent"]), 3); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
        p.drawArc(ring, int(-a * 16), int(95 * 16))
        if not self.logo.isNull(): p.drawPixmap(QRectF(cx - 75, cy - 75, 150, 150), self.logo, QRectF(self.logo.rect()))
        p.setPen(QColor(T["ink"])); f = ui_font(30, QFont.Weight.Bold); p.setFont(f)
        p.drawText(QRectF(0, H * 0.52, W, 44), Qt.AlignmentFlag.AlignCenter, "PointYoink")
        p.setPen(QColor(T["ink2"])); p.setFont(ui_font(10))
        p.drawText(QRectF(0, H * 0.615, W, 20), Qt.AlignmentFlag.AlignCenter, TAGLINE)
        bx, by, bw, bh = W / 2 - 125, H * 0.80, 250, 5
        p.fillRect(QRectF(bx, by, bw, bh), QColor("#0c0f15"))
        p.fillRect(QRectF(bx, by, max(1, bw * self.frac), bh), QColor(T["accent"]))
        p.setPen(QColor(T["ink2"])); p.setFont(ui_font(11))
        p.drawText(QRectF(0, H * 0.845, W, 20), Qt.AlignmentFlag.AlignCenter, self.status)
        p.setPen(QColor("#4a5260")); p.setFont(ui_font(10))
        p.drawText(QRectF(0, H - 30, W - 22, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "v" + self.version)
