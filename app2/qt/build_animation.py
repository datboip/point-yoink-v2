# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build progress card's animation: a point globe turns, a target locks onto the logo's dots one by one, the logo sharpens,
then bursts back into a globe. One loop is 18 seconds. Drawing and timing come from the approved design (orbit_build_v6);
the card's own stage, percent, elapsed time and Stop are driven by the build, not by this."""
import math
import random
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

LOGO_PATH = Path(__file__).resolve().parents[2] / "icon.png"
BLUE = QColor("#70b7ff")
CYAN = QColor("#19daea")
ORANGE = QColor("#ff941e")
TARGET = QPointF((920.6 / 1024 - .5) * 240, (255 / 1024 - .5) * 240)
BOTTOM_DOT = QPointF((411.0 / 1024 - .5) * 240, (862.5 / 1024 - .5) * 240)
LEFT_DOT = QPointF((82.4 / 1024 - .5) * 240, (325.9 / 1024 - .5) * 240)
TOP_DOT = QPointF((412.0 / 1024 - .5) * 240, (155.0 / 1024 - .5) * 240)
# Each logo dot has its own arrival time. Nearby dots start soon after a lock.
DOT_WAVE = (
    (411, 863, .315), (295, 817, .385), (520, 817, .405),
    (410, 735, .46), (188, 760, .49), (80, 700, .56),
    (80, 577, .535), (80, 450, .49), (82, 326, .445),
    (193, 266, .505), (191, 376, .53), (301, 433, .555),
    (410, 490, .575), (410, 612, .525),
    (303, 205, .585), (412, 155, .625), (523, 205, .66),
    (629, 260, .69), (522, 430, .625), (629, 742, .665),
    (738, 700, .69), (738, 597, .675),
)
ROUTE = (
    (.26, QPointF(-63, -122)), (.28, QPointF(0, -145 * .93)), (.33, QPointF(134, -83)), (.385, QPointF(75, 85)),
    (.418, QPointF(BOTTOM_DOT.x() + 16, BOTTOM_DOT.y() + 8)), (.43, BOTTOM_DOT), (.455, QPointF(BOTTOM_DOT.x() - 18, BOTTOM_DOT.y() + 7)),
    (.49, QPointF(-125, 43)), (.52, QPointF(LEFT_DOT.x() - 12, LEFT_DOT.y() + 14)), (.545, LEFT_DOT),
    (.57, QPointF(LEFT_DOT.x() + 4, LEFT_DOT.y() - 17)), (.62, QPointF(-106, -108)),
    (.665, QPointF(TOP_DOT.x() - 19, TOP_DOT.y() - 12)), (.695, TOP_DOT), (.725, QPointF(TOP_DOT.x() + 19, TOP_DOT.y() - 9)),
    (.785, QPointF(115, -108)), (.815, QPointF(TARGET.x() + 14, TARGET.y() - 15)), (.85, TARGET), (.89, TARGET),
)
LOOP_TICKS = 360           # 18 s at one tick per 50 ms


def smooth(v):
    v = max(0.0, min(1.0, v))
    return v * v * (3 - 2 * v)


def mix(a, b, amount):
    return a + (b - a) * amount


def curved_route(t, stops):
    """Move through timed points with a continuous tangent at every join."""
    for index in range(1, len(stops) - 2):
        start_time, start = stops[index]
        end_time, end = stops[index + 1]
        if t > end_time:
            continue
        before_time, before = stops[index - 1]
        after_time, after = stops[index + 2]
        duration = end_time - start_time
        u = max(0.0, min(1.0, (t - start_time) / duration))

        def axis(a, b, c, d):
            start_speed = (c - a) / (end_time - before_time) * duration
            end_speed = (d - b) / (after_time - start_time) * duration
            return ((2 * u ** 3 - 3 * u ** 2 + 1) * b + (u ** 3 - 2 * u ** 2 + u) * start_speed
                    + (-2 * u ** 3 + 3 * u ** 2) * c + (u ** 3 - u ** 2) * end_speed)

        return QPointF(axis(before.x(), start.x(), end.x(), after.x()), axis(before.y(), start.y(), end.y(), after.y()))
    return stops[-1][1]


_shared = {}


def _assets():
    """The logo, its light-pass copy and the point data, made once for every card."""
    if _shared: return _shared
    logo = QImage(str(LOGO_PATH)).scaled(480, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    logo = logo.convertToFormat(QImage.Format.Format_ARGB32)
    a = np.frombuffer(logo.constBits(), np.uint8).reshape(logo.height(), logo.bytesPerLine() // 4, 4)[:, :logo.width()].copy()   # B G R A
    shine = np.zeros_like(a)
    warm = (a[..., 2] > 210) & (a[..., 1] < 190)
    shine[..., 0] = np.where(warm, 174, 255); shine[..., 1] = np.where(warm, 227, 250); shine[..., 2] = np.where(warm, 255, 185)
    shine[..., 3] = a[..., 3]
    shine = (shine.astype(np.uint16) * (a[..., 3:4].astype(np.uint16) + 1) // 256).astype(np.uint8)       # premultiplied by its alpha
    shine_img = QImage(shine.tobytes(), logo.width(), logo.height(), logo.width() * 4, QImage.Format.Format_ARGB32_Premultiplied).copy()
    rng = random.Random(20261001)
    sample = logo.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    points = []
    for _ in range(1450):
        for _attempt in range(100):
            x = rng.randrange(sample.width()); y = rng.randrange(sample.height()); color = sample.pixelColor(x, y)
            if color.alpha() > 180: break
        else:
            continue
        z = rng.uniform(-1, 1); azimuth = rng.uniform(0, math.tau); tx, ty = x - 120, y - 120
        orange = color.red() > 210 and color.green() < 190 and color.blue() < 100
        if orange: order = .745 + rng.random() * .008                      # the orange target arrives last
        elif color.red() > 210 and color.green() > 200 and color.blue() > 200: order = .60 + (tx + 25) / 150 * .09 + rng.random() * .008   # the arrow grows to it
        else:
            nearest = min(DOT_WAVE, key=lambda dot: (tx - (dot[0] / 1024 - .5) * 240) ** 2 + (ty - (dot[1] / 1024 - .5) * 240) ** 2)
            order = nearest[2] + rng.random() * .012
        points.append((tx, ty, z, azimuth, color, order, orange))
    burst = []
    for _ in range(95):
        angle = rng.uniform(0, math.tau); reach = rng.uniform(86, 175); delay = rng.uniform(0, .22); size = rng.uniform(1.1, 2.8)
        color = CYAN if rng.random() < .72 else (ORANGE if rng.random() < .35 else BLUE)
        burst.append((angle, reach, delay, size, color))
    _shared.update(logo=logo, shine=shine_img, points=points, burst=burst)
    return _shared


class GlobeBuild(QWidget):
    """Transparent, fixed-size, runs only while it is shown."""

    def __init__(self, size=150, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size); self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.phase = 0.0; self._assets = None
        self.timer = QTimer(self); self.timer.setInterval(50); self.timer.timeout.connect(self._tick)

    def _tick(self):
        self.phase = (self.phase + 1.0 / LOOP_TICKS) % 1.0; self.update()

    def showEvent(self, e):
        if self._assets is None: self._assets = _assets()
        self.timer.start(); super().showEvent(e)

    def hideEvent(self, e):
        self.timer.stop(); super().hideEvent(e)

    def paintEvent(self, e):
        if self._assets is None: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(self.width() / 2, self.height() / 2); k = min(self.width(), self.height()) / 340.0; p.scale(k, k)
        self.draw_scene(p)

    def draw_scene(self, p):
        A = self._assets; logo = A["logo"]; t = self.phase
        spin = math.tau * t
        reset = smooth((t - .969) / .031)
        glow = QRadialGradient(QPointF(0, 0), 174); glow.setColorAt(0, QColor(25, 60, 82, 80)); glow.setColorAt(1, QColor(25, 60, 82, 0))
        p.setPen(Qt.PenStyle.NoPen); p.fillRect(QRectF(-174, -174, 348, 348), glow)
        sharpen = smooth((t - .865) / .075); sharp_x = mix(-120, 120, sharpen)
        logo_alpha = 1 - smooth((t - .943) / .025)
        point_visibility = max(1 - smooth((t - .941) / .025), smooth((t - .975) / .023))
        for tx, ty, z, azimuth, color, order, orange in A["points"]:
            a = azimuth + spin; latitude = math.sqrt(1 - z * z)
            gx = 137 * latitude * math.cos(a); gy = 137 * z * .92; depth = latitude * math.sin(a)
            amount = smooth((t - order) / (.085 if orange else .11)) * (1 - reset)
            x = mix(gx, tx, amount); y = mix(gy, ty, amount)
            dot = QColor(color); sharpened = smooth((sharp_x - tx) / 16) * (1 - reset)
            dot.setAlpha(min(255, int((90 + 150 * amount + 30 * max(0, depth)) * (1 - .85 * sharpened) * point_visibility)))
            p.setBrush(dot); size = 2.15 + .55 * max(0, depth); p.drawEllipse(QPointF(x, y), size, size)
        rect = QRectF(-120, -120, 240, 240)
        if t >= .865 and sharp_x > -120:                     # a vertical pass sharpens the points it has passed
            if sharpen >= 1:
                p.setOpacity(logo_alpha); p.drawImage(rect, logo); p.setOpacity(1)
            else:
                solid_right = max(-120, sharp_x - 20)
                if solid_right > -120:
                    p.save(); p.setClipRect(QRectF(-120, -120, solid_right + 120, 240)); p.setOpacity(logo_alpha); p.drawImage(rect, logo); p.restore()
                for band in range(4):
                    x0 = max(-120, sharp_x - 20 + band * 5); x1 = min(120, sharp_x - 15 + band * 5)
                    if x1 <= x0: continue
                    p.save(); p.setClipRect(QRectF(x0, -120, x1 - x0, 240)); p.setOpacity(logo_alpha * (4 - band) / 5); p.drawImage(rect, logo); p.restore()
        beam_alpha = smooth((t - .865) / .015) * (1 - smooth((t - .925) / .015))
        if beam_alpha:
            p.save(); p.translate(sharp_x, 0); p.scale(.15, 1)
            beam = QRadialGradient(QPointF(0, 0), 138)
            beam.setColorAt(0, QColor(112, 183, 255, int(46 * beam_alpha))); beam.setColorAt(.72, QColor(112, 183, 255, int(20 * beam_alpha)))
            beam.setColorAt(1, QColor(112, 183, 255, 0))
            p.setPen(Qt.PenStyle.NoPen); p.fillRect(QRectF(-138, -138, 276, 276), beam); p.restore()
            for offset, width, opacity in ((-10, 7, .15), (-3, 6, .48), (3, 5, .88), (8, 6, .24)):
                p.save(); p.setClipRect(QRectF(sharp_x + offset, -120, width, 240)); p.setOpacity(beam_alpha * opacity); p.drawImage(rect, A["shine"]); p.restore()
        self.draw_burst(p, t, A["burst"])
        self.draw_search(p, t)

    @staticmethod
    def draw_burst(p, t, burst):
        progress = (t - .966) / .034
        if progress <= 0 or progress >= 1: return
        glow_alpha = smooth(progress / .16) * (1 - smooth((progress - .34) / .45))
        if glow_alpha:
            glow = QRadialGradient(QPointF(0, 0), 82); glow.setColorAt(0, QColor(208, 251, 255, int(115 * glow_alpha)))
            glow.setColorAt(.36, QColor(25, 218, 234, int(55 * glow_alpha))); glow.setColorAt(1, QColor(25, 218, 234, 0))
            p.setPen(Qt.PenStyle.NoPen); p.fillRect(QRectF(-82, -82, 164, 164), glow)
        for angle, reach, delay, size, base in burst:
            motion = max(0.0, min(1.0, (progress - delay) / (1 - delay)))
            if motion <= 0: continue
            radius = reach * (1 - (1 - motion) ** 2.3); alpha = math.sin(math.pi * motion) ** .8 * (1 - smooth((progress - .78) / .22))
            if alpha <= 0: continue
            x, y = math.cos(angle) * radius, math.sin(angle) * radius; tail = min(16, radius * .22)
            color = QColor(base); color.setAlpha(int(100 * alpha)); p.setPen(QPen(color, 1.1))
            p.drawLine(QPointF(x - math.cos(angle) * tail, y - math.sin(angle) * tail), QPointF(x, y))
            color.setAlpha(int(220 * alpha)); p.setPen(Qt.PenStyle.NoPen); p.setBrush(color); p.drawEllipse(QPointF(x, y), size, size)

    def draw_search(self, p, t):
        visible = smooth(t / .04) * (1 - smooth((t - .865) / .10))
        if not visible: return
        if t < .28:                                          # one orbit while every point is still in the raw globe
            angle = -.5 * math.pi + math.tau * t / .28
            locator = QPointF(145 * math.cos(angle), 145 * .93 * math.sin(angle))
        else:
            locator = curved_route(t, ROUTE)
        pulse = max(smooth((t - hit) / .012) * (1 - smooth((t - hit - .020) / .025)) for hit in (.43, .545, .695, .85))
        self.draw_locator(p, locator, visible, t, .25, pulse)

    @staticmethod
    def draw_locator(p, position, opacity, t, lock_duration, pulse=0):
        orange = QColor(ORANGE); orange.setAlpha(int(235 * opacity))
        radius = (21 - 6 * pulse) if pulse else mix(17, 24, smooth((t - .50) / lock_duration))
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(orange, 2.7)); p.drawEllipse(position, radius, radius)
        p.drawLine(QPointF(position.x() - radius - 7, position.y()), QPointF(position.x() - radius + 1, position.y()))
        p.drawLine(QPointF(position.x() + radius - 1, position.y()), QPointF(position.x() + radius + 7, position.y()))
        p.drawLine(QPointF(position.x(), position.y() - radius - 7), QPointF(position.x(), position.y() - radius + 1))
        p.drawLine(QPointF(position.x(), position.y() + radius - 1), QPointF(position.x(), position.y() + radius + 7))
