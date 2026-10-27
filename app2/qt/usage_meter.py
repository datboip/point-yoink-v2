# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the app and the PC are using while scanning or building: CPU, GPU and RAM tiles in the ribbon, each with its value,
temperature where the PC reports one, and a small graph of the last two minutes. Sampled on a worker thread every 2 s
(nvidia-smi takes tens of ms); painting only reads the last samples."""
import os, shutil, subprocess, threading, time
from collections import deque
from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient, QFont
from PySide6.QtWidgets import QWidget, QPushButton
from app2.qt.theme import T, ui_font
from app2.disk_usage import nice
from app2.qt import link_status
from app2.scan.usb_mode import usb_mode

try: import psutil
except ImportError: psutil = None

HISTORY = 60                                     # samples kept per graph (2 minutes at one every 2 s)
GREEN, AMBER, BLUE = QColor("#3fd16f"), QColor(T["warn"]), QColor(T["accent"])


def _gib(n): return n / (1 << 30)


def _cpu_temp():
    """The CPU package temperature (AMD Tctl / Intel Package), or None."""
    if not psutil or not hasattr(psutil, "sensors_temperatures"): return None
    try: t = psutil.sensors_temperatures()
    except Exception: return None
    for chip, label in (("k10temp", "Tctl"), ("coretemp", "Package id 0"), ("zenpower", "Tdie")):
        for x in t.get(chip, []):
            if x.label == label or not label: return x.current
    return None


def _cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except OSError: pass
    return "CPU"


class Sampler:
    """Background sampling: one thread for the whole app, started on first use, stopped with stop()."""
    def __init__(self, every=2.0):
        self.every, self.last, self._stop = every, {}, threading.Event()
        self.hist = {k: deque(maxlen=HISTORY) for k in ("cpu", "gpu", "ram")}
        self._smi = shutil.which("nvidia-smi"); self.gpu_name = ""; self.cpu_name = _cpu_name()
        self.build_usage = BuildUsage()
        self.samples = deque(maxlen=HISTORY)
        self._processes = {}
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if psutil: psutil.cpu_percent(None); self._proc.cpu_percent(None)        # first call primes the counters
        self._t = threading.Thread(target=self._run, name="usage-meter", daemon=True); self._t.start()

    def stop(self): self._stop.set()

    def _run(self):
        first = True
        while first or not self._stop.wait(self.every):
            if first: time.sleep(.3); first = False          # a short first window so the CPU figures mean something
            s = {}
            if psutil:
                vm = psutil.virtual_memory(); ncpu = psutil.cpu_count() or 1
                app_cpu, rss, pids, memory_kind = self._owned_usage(ncpu)
                s.update(cpu=psutil.cpu_percent(None), app_cpu=app_cpu, cpu_temp=_cpu_temp(),
                         ram_used=vm.total - vm.available, ram_total=vm.total, app_ram=rss,
                         app_ram_kind=memory_kind)
                self.hist["cpu"].append(s["cpu"]); self.hist["ram"].append(100 * s["ram_used"] / s["ram_total"])
            if self._smi:
                try:
                    out = subprocess.run([self._smi, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout
                    f = [x.strip() for x in out.splitlines()[0].split(",")]
                    s.update(gpu=float(f[0]), vram_used=float(f[1]) * (1 << 20), vram_total=float(f[2]) * (1 << 20), gpu_temp=float(f[3]))
                    self.gpu_name = f[4].replace("NVIDIA GeForce ", ""); self.hist["gpu"].append(s["gpu"])
                    try:
                        xml = subprocess.run([self._smi, '-q', '-x'], capture_output=True, text=True, timeout=2).stdout
                        s['app_vram'] = gpu_process_memory(xml, pids if psutil else {os.getpid()})
                    except (OSError, ValueError, subprocess.SubprocessError): s['app_vram'] = None
                except (OSError, ValueError, IndexError, subprocess.SubprocessError): self._smi = None     # no GPU readout, stop asking
            s["t"] = time.time(); self.last = s
            self.samples.append(dict(s))
            self.build_usage.add(s)

    def _owned_usage(self, ncpu):
        """Only this process and its children; another app instance belongs to the rest of the PC."""
        try: processes = [self._proc] + self._proc.children(recursive=True)
        except psutil.Error: processes = [self._proc]
        owned = set(); cpu = 0.; memory = 0; kind = 'PSS'
        for item in processes:
            owned.add(item.pid)
            process = self._processes.setdefault(item.pid, item)
            try:
                cpu += process.cpu_percent(None) / ncpu
                try: memory += process.memory_full_info().pss
                except (AttributeError, psutil.Error): memory += process.memory_info().rss; kind = 'RSS estimate'
            except psutil.Error: continue
        self._processes = {pid: proc for pid, proc in self._processes.items() if pid in owned}
        return cpu, memory, owned, kind


def gpu_process_memory(xml, owned):
    """NVIDIA XML includes graphics and compute memory; utilization cannot be split reliably."""
    from xml.etree import ElementTree
    try: gpu = ElementTree.fromstring(xml).find('gpu')
    except ElementTree.ParseError as exc: raise ValueError('GPU process memory is unavailable.') from exc
    if gpu is None or gpu.find('processes') is None: return None
    total = 0
    for row in gpu.findall('./processes/process_info'):
        try: pid = int(row.findtext('pid', '-1'))
        except ValueError: continue
        if pid not in owned: continue
        value = row.findtext('used_memory', '')
        try: total += float(value.split()[0]) * (1 << 20)
        except (ValueError, IndexError): return None
    return total


class BuildUsage:
    """Sampled usage during this app's last build, retained across restarts."""
    def __init__(self):
        from scanner.stages import resources
        import json
        self.path = resources.config_dir() / 'last-build-usage.json'
        try: self.average = json.loads(self.path.read_text())
        except (OSError, ValueError): self.average = {}
        self.running = False; self.rows = []; self.started = None
        self.lock = threading.Lock()

    def set_building(self, on):
        import json, tempfile
        with self.lock:
            if bool(on) == self.running: return
            self.running = bool(on)
            if on: self.rows = []; self.started = time.monotonic(); return
            keys = ('cpu', 'app_cpu', 'gpu', 'vram_used', 'app_vram', 'ram_used', 'ram_total', 'app_ram')
            self.average = {'samples': len(self.rows), 'seconds': round(time.monotonic()-self.started, 1)}
            for key in keys:
                values = [r[key] for r in self.rows if r.get(key) is not None]
                if values: self.average[key] = sum(values)/len(values)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as f:
                    json.dump(self.average, f); name = f.name
                os.replace(name, self.path)
            except OSError: pass

    def add(self, sample):
        with self.lock:
            if self.running: self.rows.append(dict(sample))


_shared = None
def sampler():
    global _shared
    if _shared is None: _shared = Sampler()
    return _shared


class UsageMeter(QWidget):
    """Three tiles: CPU (use, temperature), GPU (use, temperature, video memory), RAM (PC use, this app). Each graphs the last
    two minutes and turns amber when busy or hot (the desktop was killed by the out-of-memory guard once)."""
    tile_requested = Signal(str)
    TILE_W, H = 124, 54

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("UsageMeter")
        self.setFixedSize(4 * self.TILE_W + 3 * 8, self.H)
        self._timer = QTimer(self); self._timer.timeout.connect(self.refresh); self._timer.start(2000)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName('Scanner and PC resources')
        self.panel = None
        self.tile_requested.connect(self.open_panel)
        self.tile_buttons = {}
        for index, tile in enumerate(('SCANNER', 'CPU', 'GPU', 'RAM')):
            control = QPushButton(self)
            control.setObjectName('ResourceTile_' + tile)
            control.setAccessibleName(tile + ' resource details')
            control.setToolTip(tile + ' use and limits')
            control.setGeometry(index * (self.TILE_W + 8), 0, self.TILE_W, self.H)
            control.setStyleSheet('QPushButton {background:transparent;border:none;} QPushButton:focus {border:1px solid #4a9dff;border-radius:6px;}')
            control.clicked.connect(lambda _checked=False, name=tile: self.tile_requested.emit(name))
            self.tile_buttons[tile] = control
        sampler(); QTimer.singleShot(600, self.refresh)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = int(event.position().x()) // (self.TILE_W + 8)
            if 0 <= index < 4: self.tile_requested.emit(('SCANNER', 'CPU', 'GPU', 'RAM')[index])
            event.accept(); return
        super().mousePressEvent(event)

    def open_panel(self, tile):
        from app2.qt.resource_panel import ResourcePanel
        if self.panel is not None: self.panel.close()
        self.panel = ResourcePanel(tile, self.window(), sampler=sampler())
        self.panel.show(); self.panel.raise_()

    def refresh(self):
        link_status.sample()
        s, sm = sampler().last, sampler()
        if not s: return
        tip = []
        if "cpu" in s: tip.append("%s: %d%% (this app %d%%)%s" % (sm.cpu_name, round(s["cpu"]), round(s["app_cpu"]), " · %.0f°C" % s["cpu_temp"] if s.get("cpu_temp") else ""))
        if "gpu" in s: tip.append("%s: %d%% · %.0f°C · video memory %.1f of %.0f GB" % (sm.gpu_name or "GPU", round(s["gpu"]), s["gpu_temp"], _gib(s["vram_used"]), _gib(s["vram_total"])))
        if "ram_used" in s: tip.append("RAM: %.1f of %.0f GB in use · PointYoink %s" % (_gib(s["ram_used"]), _gib(s["ram_total"]), nice(s["app_ram"])))
        self.setToolTip("\n".join(tip) + "\nUpdates every 2 seconds; graphs show the last 2 minutes.")
        self.update()

    def _scanner_tile(self):
        """How the scanner is linked: Wi-Fi receiver state and speed first, else the USB mode; graph = transfer MB/s."""
        st = link_status.state; mode = usb_mode()
        if st["wifi"] == "receiving": val, extra = "Wi-Fi", "%.1f MB/s" % (st["rate"] / (1 << 20))
        elif st["wifi"] == "waiting": val, extra = "Wi-Fi", "waiting · code %s" % st["code"]
        elif mode == "pc": val, extra = "USB", "PC mode"
        elif mode == "transfer": val, extra = "USB", ("File Transfer · %.1f MB/s" % (st["rate"] / (1 << 20))) if st["rate"] else "File Transfer"
        elif mode: val, extra = "USB", "other mode"
        else: val, extra = None, "not connected"
        return ("SCANNER", val, extra, link_status.rates, False)

    def paintEvent(self, e):
        s, sm = sampler().last, sampler()
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        tiles = [self._scanner_tile(), ("CPU", s.get("cpu"), ("%.0f°C" % s["cpu_temp"]) if s.get("cpu_temp") else "", sm.hist["cpu"], (s.get("cpu_temp") or 0) > 85),
                 ("GPU", s.get("gpu"), ("%.0f°C · %.1fG" % (s["gpu_temp"], _gib(s["vram_used"]))) if "gpu" in s else "", sm.hist["gpu"], (s.get("gpu_temp") or 0) > 83),
                 ("RAM", (100 * s["ram_used"] / s["ram_total"]) if "ram_used" in s else None,
                  ("%.0f/%.0fG" % (_gib(s["ram_used"]), _gib(s["ram_total"]))) if "ram_used" in s else "", sm.hist["ram"], False)]
        for i, (name, val, extra, hist, hot) in enumerate(tiles):
            x = i * (self.TILE_W + 8); r = QRectF(x + .5, .5, self.TILE_W - 1, self.H - 1)
            if name == "SCANNER": scale = max(1.0, max(hist, default=0)); shown = val or "--"; c = BLUE if val else QColor(T["ink3"])
            else:
                scale = 100.0; shown = "--" if val is None else "%d%%" % round(val)
                busy = hot or (val or 0) > 85; c = AMBER if busy else GREEN
            p.setPen(QPen(QColor("#2e353f"), 1)); p.setBrush(QColor("#161a1f")); p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor(T["ink2"])); p.setFont(ui_font(10, QFont.Weight.DemiBold)); p.drawText(QRectF(x + 8, 4, 60, 14), Qt.AlignLeft | Qt.AlignVCenter, name)
            p.setPen(c); p.setFont(ui_font(12, QFont.Weight.Bold))
            p.drawText(QRectF(x + 8, 4, self.TILE_W - 16, 14), Qt.AlignRight | Qt.AlignVCenter, shown)
            p.setPen(QColor(T["ink3"])); p.setFont(ui_font(9)); p.drawText(QRectF(x + 8, 18, self.TILE_W - 16, 12), Qt.AlignLeft | Qt.AlignVCenter, extra)
            g = QRectF(x + 6, 32, self.TILE_W - 12, self.H - 38)        # the graph: filled area under the last 2 minutes
            if len(hist) > 1:
                n = HISTORY - 1; pts = [QPointF(g.right() - (len(hist) - 1 - j) * g.width() / n, g.bottom() - min(scale, v) / scale * g.height()) for j, v in enumerate(hist)]
                path = QPainterPath(QPointF(pts[0].x(), g.bottom()))
                for q in pts: path.lineTo(q)
                path.lineTo(pts[-1].x(), g.bottom()); path.closeSubpath()
                fill = QLinearGradient(0, g.top(), 0, g.bottom()); top = QColor(c); top.setAlpha(170); bot = QColor(c); bot.setAlpha(25)
                fill.setColorAt(0, top); fill.setColorAt(1, bot); p.fillPath(path, fill)
                line = QPainterPath(pts[0])
                for q in pts[1:]: line.lineTo(q)
                p.setPen(QPen(c, 1.2)); p.setBrush(Qt.NoBrush); p.drawPath(line)
            else:
                p.setPen(QPen(QColor("#2e353f"), 1)); p.drawLine(QPointF(g.left(), g.bottom()), QPointF(g.right(), g.bottom()))
