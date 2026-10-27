# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scanner project import inside the unified app."""
import json
import shutil
import time
import subprocess
import re
from datetime import datetime
import threading
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QProcess, QTimer, QSize, QUrl
from PySide6.QtGui import QPixmap, QBrush, QColor, QDesktopServices, QIcon, QPainter, QFont
from app2.scan.usb_mode import usb_mode
from app2.qt.icons import icon
from app2.qt import link_status
from app2 import disk_usage
from app2.qt.widgets import StableLabel, ElidedLabel, keep_room
from app2.library_dir import set_default, default_dir, capture_dir, set_capture_dir
from app2.qt.capture_grid import CapList, CaptureViewer, CAP_W, CAP_H
from app2.project_preview import rename_project
from scanner.adopt_device_scan import adopt_project, issue_summary
from app2.qt.theme import T
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QProgressBar, QComboBox, QMessageBox, QWidget,
                               QCheckBox, QFileDialog, QInputDialog, QMenu, QTabBar, QListWidget, QListWidgetItem, QScrollArea, QLayout)

from app2 import REPO


PREVIEW = 120                  # side preview size in the strip under the table (px)
LONG_COPY_S = 30 * 60             # past this, suggest Wi-Fi
BYTES_PER_SECOND = 15 << 20       # big files (meshes) stream at ~15-19 MB/s over MTP
SECONDS_PER_FILE = 1.0          # USB File Transfer pays about a second per file (the raw frames are thousands of files)


def nice(n):
    from app2.disk_usage import nice as fmt
    return fmt(n)


def eta(files, size=0):
    """About how long copying this many files (size bytes) over USB File Transfer takes, in words."""
    s = files * SECONDS_PER_FILE + size / BYTES_PER_SECOND
    return "< 1 min" if s < 60 else "%d min" % round(s / 60) if s < 5400 else "%.1f hours" % (s / 3600)


def scanned_at(name):
    """The scan time the scanner writes into the project name (ProjectMMDDYYYYHHMMSS), as a timestamp, or None."""
    m = re.fullmatch(r'Project(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})', name)
    if not m: return None
    mo, d, y, h, mi, se = map(int, m.groups())
    try: return datetime(y, mo, d, h, mi, se).timestamp()
    except ValueError: return None


def capture_time(name):
    """A screenshot or recording is named for when it was taken (MMDDYYYYHHMMSS.png), as a datetime, or None."""
    m = re.fullmatch(r'(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})', Path(name).stem)
    if not m: return None
    mo, d, y, h, mi, se = map(int, m.groups())
    try: return datetime(y, mo, d, h, mi, se)
    except ValueError: return None


def capture_text(name):
    d = capture_time(name)
    return d.strftime('%b %d, %Y, %I:%M:%S %p').replace(' 0', ' ') if d else Path(name).stem


def scanned_text(name):
    """'Sep 10, 10:29 PM' this year, 'Sep 10, 2025' before (the full time is on hover)."""
    t = scanned_at(name)
    if not t: return ''
    d = datetime.fromtimestamp(t)
    return (d.strftime('%b %d, %I:%M %p') if d.year == datetime.now().year else d.strftime('%b %d, %Y')).replace(' 0', ' ')


def scan_details(side):
    """Settings from this scan's public project file, with missing fields stated plainly."""
    side=Path(side)
    try: info=json.loads((side/'property.rvproj').read_text())
    except (OSError,ValueError): info={}
    params=info.get('scan_param') or {}
    accuracy={1:'High',0:'Standard',2:'High-speed'}.get(params.get('accuracy_type'),'Unknown')
    object_type={0:'General',4:'Dark',3:'Body'}.get(params.get('scan_object'),'Unknown')
    color='Color on' if params.get('color_type')==1 else 'Color off' if params.get('color_type')==0 else 'Color unknown'
    w,h=params.get('depth_width'),params.get('depth_height')
    depth='%s x %s depth' % (w,h) if w and h else 'Depth size unknown'
    try: frames=len(list((side/'cache').glob('*.dph')))
    except OSError: frames=0
    if not frames: frames=info.get('vf_count')
    count=('%s frame%s' % (frames,'' if frames==1 else 's')) if frames is not None else 'Frames unknown'
    stamp=scanned_text('Project'+side.name) or side.name
    return '%s | %s | %s | %s | %s | %s' % (stamp,accuracy,object_type,color,depth,count)


def _names_path():
    config = Path(os.environ.get('POINTYOINK_CONFIG') or Path.home() / '.config/pointyoink/config.json')
    return config.parent / 'device-names.json'


def load_names():
    """Human names for scanner projects, by the scanner's own project name (kept on this PC)."""
    try:
        d = json.loads(_names_path().read_text()); return d if isinstance(d, dict) else {}
    except (OSError, ValueError): return {}


def save_names(names):
    p = _names_path(); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(names, indent=1)); tmp.replace(p)


def clock(seconds):
    """1:07 or 1:02:07."""
    s = int(max(0, seconds)); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return '%d:%02d:%02d' % (h, m, s) if h else '%d:%02d' % (m, s)


def no_widow(text):
    """Glue the last two words so a wrapped line never ends with one word alone."""
    head, sep, last = text.rpartition(' ')
    return head + '\u00a0' + last if sep and head else text


def cropped(data, dark=24, pad=0.08):
    """The scanner's preview is a small model on a big black picture: crop to the model (plus a margin) so it fills
    the thumbnail. Returns a QPixmap (empty when the picture can't be read)."""
    pm = QPixmap(); pm.loadFromData(data)
    if pm.isNull(): return pm
    try:
        import io, numpy as np
        from PIL import Image
        a = np.asarray(Image.open(io.BytesIO(data)).convert('L'))
        ys, xs = np.nonzero(a > dark)
        if len(xs) < 20: return pm
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); m = int(pad * max(x1 - x0, y1 - y0)) + 2
        x0, y0 = max(0, x0 - m), max(0, y0 - m); x1, y1 = min(a.shape[1] - 1, x1 + m), min(a.shape[0] - 1, y1 + m)
        side = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2          # square, centered on the model
        return pm.copy(max(0, cx - side // 2), max(0, cy - side // 2), side, side)
    except Exception: return pm


def video_frame(path):
    """A picture from a third of the way into a recording (PNG bytes), or b'' when it cannot be read."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if n > 3: cap.set(cv2.CAP_PROP_POS_FRAMES, int(n / 3))
            ok, frame = cap.read()
        finally: cap.release()
        if not ok: return b''
        ok, png = cv2.imencode('.png', frame); return png.tobytes() if ok else b''
    except Exception: return b''


class Thumb(QLabel):
    """A side picture; hovering shows it bigger in a popup beside the pointer."""
    BIG = 420

    def __init__(self, pm, tip, parent=None):
        super().__init__(parent); self.full = pm; self.setToolTip(tip); self._pop = None
        self.setFixedSize(PREVIEW, PREVIEW); self.setAlignment(Qt.AlignCenter); self.setStyleSheet('background:#0b0d10; border-radius:6px;')
        if not pm.isNull(): self.setPixmap(pm.scaled(PREVIEW - 8, PREVIEW - 8, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def enterEvent(self, e):
        if self.full.isNull(): return
        self._pop = QLabel(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self._pop.setStyleSheet('background:#0b0d10; border:1px solid #3a414b; border-radius:8px; padding:6px;')
        self._pop.setPixmap(self.full.scaled(self.BIG, self.BIG, Qt.KeepAspectRatio, Qt.SmoothTransformation)); self._pop.adjustSize()
        g = self.mapToGlobal(self.rect().topRight()); self._pop.move(g.x() + 12, max(0, g.y() - self.BIG // 3)); self._pop.show()

    def leaveEvent(self, e):
        if self._pop: self._pop.close(); self._pop = None


class WayCard(QFrame):
    """One way to connect: icon, bold title, a dim line under it, and an optional live status line; the whole card clicks."""
    clicked = Signal()

    def __init__(self, icon_name, title, sub, parent=None):
        super().__init__(parent); self.setObjectName('ImportWay'); self.setCursor(Qt.PointingHandCursor)
        h = QHBoxLayout(self); h.setContentsMargins(12, 10, 12, 10); h.setSpacing(12)
        ic = QLabel(); ic.setPixmap(icon(icon_name).pixmap(28, 28)); ic.setFixedSize(32, 32); h.addWidget(ic, 0, Qt.AlignTop)
        v = QVBoxLayout(); v.setSpacing(2)
        t = QLabel(title); t.setStyleSheet('font-size:15px; font-weight:600;'); v.addWidget(t)
        d = QLabel(no_widow(sub)); d.setObjectName('ImportHint'); d.setWordWrap(True); v.addWidget(d)
        v.addSpacing(6)
        row = QHBoxLayout(); row.setSpacing(6)                  # status: the dot in its own gutter, the words beside it
        self.dot = QLabel('\u25cf'); self.dot.setFixedWidth(14); self.dot.setAlignment(Qt.AlignTop | Qt.AlignHCenter); row.addWidget(self.dot, 0, Qt.AlignTop)
        self.state = QLabel(''); self.state.setWordWrap(True); self.state.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.state.setStyleSheet('font-size:12px;')
        self.state.setMinimumHeight(2 * self.state.fontMetrics().lineSpacing() + 2); row.addWidget(self.state, 1)   # 2 lines kept: no jump
        v.addLayout(row)
        h.addLayout(v, 1); self.setChecked(False)
        # Keep the title, two subtitle lines and two status lines readable.
        # The containing connection panel scrolls when these need more room.
        self.setMinimumHeight(20 + t.sizeHint().height() + 2 * d.fontMetrics().lineSpacing()
                              + self.state.minimumHeight() + 12)

    def setChecked(self, on):
        self._on = on
        self.setStyleSheet('QFrame#ImportWay {border:%s; border-radius:10px; background:%s;}'
                           % (('2px solid %s' % T['accent'], '#1c2a3d') if on else ('1px solid #3a414b', '#22272e')))

    def set_state(self, color, text): self.dot.setStyleSheet('color:%s; font-size:12px;' % color); self.state.setText(no_widow(text))

    def mousePressEvent(self, e):
        if self.isEnabled() and e.button() == Qt.LeftButton: self.clicked.emit()


class ProjectItem(QTreeWidgetItem):
    """A row that sorts by value, not by its text: sides, sizes and copy time as numbers, the name as text."""
    KEYS = {0: 'name', 1: 'what', 2: 'sides', 3: 'models', 4: 'mfiles', 5: 'full', 6: 'files', 7: 'mesh', 8: 'exists', 9: 'scanned', 10: 'state'}

    def __lt__(self, other):
        tree = self.treeWidget(); col = tree.sortColumn() if tree else 0
        ra = bool((self.data(0, Qt.UserRole + 1) or {}).get('sized')); rb = bool((other.data(0, Qt.UserRole + 1) or {}).get('sized'))
        if ra != rb:                         # sized rows above rows still loading, in either sort direction
            up = tree is None or tree.header().sortIndicatorOrder() == Qt.AscendingOrder
            return ra if up else rb
        key = self.KEYS.get(col)
        if key is None: return self.text(col).lower() < other.text(col).lower()
        a = (self.data(0, Qt.UserRole + 1) or {}).get(key); b = (other.data(0, Qt.UserRole + 1) or {}).get(key)
        if a is None or b is None: return (a is None) < (b is None)     # rows still being read sort last
        return a < b


class DeviceProjects(QFrame):
    open_requested = Signal(str)             # a received/imported project to open (right-click Open)
    _capsListed = Signal(list)                 # the scanner's screenshots and recordings, from the reader thread
    _capThumb = Signal(str, bytes)            # (path on the scanner, png bytes) for a screenshot's row picture
    _previewed = Signal(str, list)            # (project name, [(side, png bytes)]) from the preview reader thread
    connect_requested = Signal(str)
    dismissed = Signal()
    imported = Signal(str)

    def __init__(self, library, parent=None):
        super().__init__(parent)
        self.setObjectName('DeviceProjects')
        self.setStyleSheet('''
            #DeviceProjects {background:#1a1d22;border:none;border-radius:0;color:#e6edf6;}
            #ImportConnection, #ImportList {background:#20252c;border:1px solid #35414e;border-radius:9px;}
            #ImportEyebrow {color:#8ebdff;font-weight:600;}
            #ImportHint {color:#a4b0bf;}
            #ImportPrimary {background:#5b9df0;color:#101923;border:0;border-radius:6px;font-weight:600;padding:9px 16px;}
            #ImportPrimary:disabled {background:#354355;color:#8290a2;}
            QTreeWidget {background:#14191f;border:1px solid #36414f;border-radius:6px;color:#e6edf6;}
            QTreeWidget::indicator {width:15px;height:15px;border:1px solid #8290a6;border-radius:3px;background:#1d232b;}
            QTreeWidget::indicator:checked {background:#5b9df0;border:1px solid #dceaff;}
            QTreeWidget::item {padding:9px 6px;}
            QHeaderView::section {background:#29323d;color:#c4d2e3;padding:7px;border:0;}
            QProgressBar {background:#11161c;border:1px solid #38434f;border-radius:5px;text-align:center;height:10px;}
            QProgressBar::chunk {background:#5b9df0;border-radius:4px;}
        ''')
        self.library = str(Path(library or default_dir()).expanduser().resolve())
        self.mountpoint = str(Path.home()/'.cache/pointyoink/device-projects-mtp')
        self.root = None; self.receipt = None; self.proc = None; self.mode = None
        self.buffer = ''; self.failed = False; self.paused = False; self.incomplete = False; self.skipped = []
        self._prepare_notes = []; self._prepared_sides = 0
        box = QVBoxLayout(self); box.setContentsMargins(24, 20, 24, 20); box.setSpacing(10)
        title = QLabel('Import from scanner'); title.setStyleSheet('font-size:22px;font-weight:600;'); box.addWidget(title); self.page_title = title
        subtitle = QLabel('Bring complete scanner projects into this PC workspace. Raw scans and finished results stay together.')
        subtitle.setObjectName('ImportHint'); subtitle.setWordWrap(True); box.addWidget(subtitle); self.page_sub = subtitle
        content = QHBoxLayout(); content.setSpacing(12); box.addLayout(content, 1)
        connection = QFrame(); connection.setObjectName('ImportConnection'); connection.setFixedWidth(280)
        connection_layout = QVBoxLayout(connection); connection_layout.setContentsMargins(0, 0, 0, 0)
        self.connection_scroll = QScrollArea(); self.connection_scroll.setWidgetResizable(True)
        self.connection_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.connection_scroll.setStyleSheet('QScrollArea {border:none; background:transparent;}')
        body = QWidget(); body.setObjectName('ImportConnectionBody'); body.setStyleSheet('#ImportConnectionBody {background:transparent;}')
        self.connection_scroll.setWidget(body); connection_layout.addWidget(self.connection_scroll)
        left = QVBoxLayout(body); left.setContentsMargins(18, 18, 18, 18); left.setSpacing(12)
        left.setSizeConstraint(QLayout.SetMinimumSize)
        eyebrow = QLabel('STEP 1 · CONNECT SCANNER'); eyebrow.setObjectName('ImportEyebrow'); left.addWidget(eyebrow)
        # two plain choices instead of a dropdown and a paragraph; the (hidden) combo still carries which one is picked
        self.transport = QComboBox(); self.transport.addItems(['USB File Transfer', 'Wi-Fi Share to PC']); self.transport.hide()
        self.transport.currentIndexChanged.connect(self.transport_changed)
        self.way_b = {}
        for i, (ic, title, sub) in enumerate((('usb', 'USB cable', 'File Transfer on the scanner'), ('wifi', 'Wi-Fi', 'Share to PC on the scanner'))):
            b = WayCard(ic, title, sub); b.clicked.connect(lambda k=i: self.transport.setCurrentIndex(k)); left.addWidget(b); self.way_b[i] = b
        self.help = StableLabel('', 2, ['Same Wi-Fi network for both. Enter the code shown below on the scanner.']); self.help.setObjectName('ImportHint'); left.addWidget(self.help)
        self.connect_b = QPushButton('Connect USB'); self.connect_b.setObjectName('ImportPrimary'); self.connect_b.clicked.connect(self.connect)
        btn_row = QHBoxLayout(); btn_row.setSpacing(6); btn_row.addWidget(self.connect_b, 1); left.addLayout(btn_row)   # [Stop | New code] while receiving
        self.code_box = QFrame(); self.code_box.setObjectName('ImportCode'); self.code_box.hide()
        self.code_box.setStyleSheet('QFrame#ImportCode {border:none; background:transparent;}')     # plain, right under the button
        cb = QVBoxLayout(self.code_box); cb.setContentsMargins(0, 4, 0, 0); cb.setSpacing(2)
        self.code = QLabel(''); self.code.setTextFormat(Qt.RichText); self.code.setWordWrap(True); self.code.setAlignment(Qt.AlignCenter); self.code.setStyleSheet('background:transparent; border:none;'); cb.addWidget(self.code)
        self.new_code_b = QPushButton(); self.new_code_b.setIcon(icon('refresh')); self.new_code_b.setToolTip('New code: restart the receiver with a fresh code')
        self.new_code_b.setFixedWidth(40); self.new_code_b.clicked.connect(self.new_code); btn_row.addWidget(self.new_code_b); self.new_code_b.hide()
        self.code.setText(self._code_html('0000', '000.000.000.000')); self.code_box.setFixedHeight(self.code_box.sizeHint().height()); self.code.setText('')   # room for a code
        left.addWidget(self.code_box); keep_room(self.code_box)       # its room is kept: nothing below moves when the code shows
        left.addStretch(1)
        dest_title = QLabel('SAVE TO'); dest_title.setObjectName('ImportEyebrow'); left.addWidget(dest_title)
        box_ = QFrame(); box_.setObjectName('ImportDest'); box_.setStyleSheet('QFrame#ImportDest {border:1px solid #3a414b; border-radius:8px; background:#1a1e24;}')
        dest_row = QHBoxLayout(box_); dest_row.setContentsMargins(10, 6, 6, 6); dest_row.setSpacing(8)
        fi = QLabel(); fi.setPixmap(icon('projects').pixmap(20, 20)); dest_row.addWidget(fi)
        self.location = ElidedLabel(); self.location.setToolTip(self.library); dest_row.addWidget(self.location, 1)
        left.addWidget(box_); self._show_library()
        opts = QHBoxLayout(); self.change_b = QPushButton('Change folder'); self.change_b.clicked.connect(self.pick_library); opts.addWidget(self.change_b)
        self.default_c = QCheckBox('Default'); self.default_c.setToolTip('New and imported projects go here from now on')
        self.default_c.toggled.connect(lambda on: set_default(self.library) if on else None); opts.addWidget(self.default_c); opts.addStretch(1); left.addLayout(opts)
        content.addWidget(connection); self.connection = connection
        listing = QFrame(); listing.setObjectName('ImportList')
        right = QVBoxLayout(listing); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(9)
        list_header = QHBoxLayout(); self.list_title = QLabel('STEP 2 · CHOOSE PROJECTS'); list_header.addWidget(self.list_title)
        self.selection = QLabel('Connect to see scanner projects'); self.selection.setObjectName('ImportHint'); list_header.addStretch(1)
        self.refresh_b = QPushButton(' Refresh'); self.refresh_b.setIcon(icon('refresh')); self.refresh_b.setToolTip('Read the scanner again; reconnects if it stopped answering')
        self.refresh_b.clicked.connect(self.refresh)
        right.addLayout(list_header)
        self.tabs = QTabBar(self); self.tabs.setDrawBase(False); self.tabs.setExpanding(False)
        for t in ('Projects', 'Screenshots', 'Screen recordings'): self.tabs.addTab(t)
        self.tabs.setStyleSheet('QTabBar::tab {background:#20252c; color:#a4b0bf; padding:7px 18px; border:1px solid #36414f; border-radius:6px; margin-right:6px;}'
                                'QTabBar::tab:selected {background:#2b3a4d; color:#e6edf6; border-color:#5b9df0;} QTabBar::tab:hover {color:#e6edf6;}')
        self.tabs.currentChanged.connect(self._tab)           # the shell moves these into the header bar, next to the page name
        self._caps_rows = None; self._thumbs = 0
        # the scanner itself: a small card with its name, serial and storage (shown once connected)
        self.dev_card = QFrame(); self.dev_card.setObjectName('ImportDevice')    # always shown: its room never jumps the table
        self.dev_card.setStyleSheet('QFrame#ImportDevice {border:1px solid #3a414b; border-radius:8px; background:#1a1e24;}')
        dc = QHBoxLayout(self.dev_card); dc.setContentsMargins(12, 8, 12, 8); dc.setSpacing(12)
        di = QLabel(); di.setPixmap(icon('usb').pixmap(24, 24)); dc.addWidget(di)
        dv = QVBoxLayout(); dv.setSpacing(4); top = QHBoxLayout()
        name = QLabel('MIRACO'); name.setStyleSheet('font-weight:600;'); top.addWidget(name); top.addStretch(1)
        self.eye_b = QPushButton(); self.eye_b.setIcon(icon('visible-only')); self.eye_b.setIconSize(QSize(16, 16)); self.eye_b.setCheckable(True); self.eye_b.setFixedSize(26, 22); self.eye_b.setToolTip('Show the serial number (hidden so it stays out of screenshots)')
        self.eye_b.setStyleSheet('QPushButton {border:none; background:transparent; border-radius:4px;} QPushButton:hover {background:#2a3038;} QPushButton:checked {background:#1c2a3d;}')
        self.eye_b.toggled.connect(lambda _on: self._show_serial()); top.addWidget(self.eye_b)
        self.dev_serial = QLabel(''); self.dev_serial.setObjectName('ImportHint')
        self.dev_serial.setFixedWidth(self.dev_serial.fontMetrics().horizontalAdvance('serial WWWWWWWWWWWWWWWWW') )   # same width shown or hidden
        top.addWidget(self.dev_serial); dv.addLayout(top); self._serial = ''
        self.dev_bar = QProgressBar(); self.dev_bar.setFixedHeight(8); self.dev_bar.setTextVisible(False); dv.addWidget(self.dev_bar)
        self.dev_info = QLabel(''); self.dev_info.setObjectName('ImportHint'); dv.addWidget(self.dev_info)
        self.dev_bar.setRange(0, 1000); self.dev_bar.setValue(0)
        dc.addLayout(dv, 1); self.dev_card.setFixedWidth(440)
        keep_room(self.dev_card); self.dev_card.hide()          # its room is kept while empty: nothing jumps when it fills in
        pick_row = QHBoxLayout(); pick_row.setSpacing(8)
        self.all_b = QPushButton('Select all'); self.all_b.clicked.connect(lambda: self.select_all(True)); pick_row.addWidget(self.all_b)
        self.none_b = QPushButton('Clear'); self.none_b.clicked.connect(lambda: self.select_all(False)); pick_row.addWidget(self.none_b)
        pick_row.addWidget(self.refresh_b); pick_row.addStretch(1)
        picks = QVBoxLayout(); picks.setSpacing(6); picks.addLayout(pick_row); picks.addWidget(self.selection)   # the count under the buttons
        top_row = QHBoxLayout(); top_row.addLayout(picks, 1); top_row.addWidget(self.dev_card, 0, Qt.AlignBottom); right.addLayout(top_row)
        self.files = QTreeWidget(); self.files.setHeaderLabels(['Scanner project', 'Copy', 'Sides', 'Models', 'Models copy', 'Full', 'Full copy', 'Mesh', 'On this PC', 'Scanned', 'Status'])
        self.files.setRootIsDecorated(True); self.files.setIndentation(20); self.files.setExpandsOnDoubleClick(False); self.files.setColumnWidth(0, 280)
        for col, w in ((1, 118), (2, 55), (3, 115), (4, 105), (5, 115), (6, 105), (7, 60), (8, 90), (9, 150)): self.files.setColumnWidth(col, w)   # room for '1.8 GB', 'under a minute'
        self.files.setSortingEnabled(True); self.files.sortByColumn(0, Qt.AscendingOrder)   # click a header to sort by it
        self.files.header().setToolTip('Click a column to sort by it; click again to reverse')
        fm = self.files.header().fontMetrics()                 # every column fits its title plus the sort arrow
        for col in range(self.files.columnCount()):
            need = fm.horizontalAdvance(self.files.headerItem().text(col)) + 40
            if self.files.columnWidth(col) < need: self.files.setColumnWidth(col, need)
        self.files.setStyleSheet('QHeaderView::section {border:none; border-right:1px solid #3a414b; padding-left:8px; padding-right:18px;}'   # room for the sort arrow
                                 'QHeaderView::section:hover {background:#2f3640;}'
                                 'QScrollBar:vertical {background:#14191f; width:10px; margin:0; border:none;}'
                                 'QScrollBar::handle:vertical {background:#4a525d; border-radius:3px; min-height:30px; margin:2px;}'
                                 'QScrollBar::handle:vertical:hover {background:#5b6470;}'
                                 'QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {height:0;}'
                                 'QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {background:none;}'
                                 )
        cap = QWidget(); cap.setFixedHeight(self.files.header().sizeHint().height()); cap.setStyleSheet('background:#2a2f37;')
        self.files.addScrollBarWidget(cap, Qt.AlignTop)          # the scrollbar starts under the header row, not beside it
        self.files.header().geometriesChanged.connect(lambda c=cap: c.setFixedHeight(self.files.header().height()))   # its real height, after styling
        QTimer.singleShot(0, lambda c=cap: c.setFixedHeight(self.files.header().height()))
        self.names = load_names(); self.files.itemDoubleClicked.connect(self._rename)
        self.files.setContextMenuPolicy(Qt.CustomContextMenu); self.files.customContextMenuRequested.connect(self._row_menu)
        self.files.itemExpanded.connect(self._expanded); self.files.itemCollapsed.connect(self._collapsed)
        self.files.itemClicked.connect(self._toggle_copy); self.files.itemChanged.connect(self._ticked); self._ticking = False
        self.files.setColumnHidden(1, True)                     # the choice is the tick on Models or Full
        self.files.itemChanged.connect(self.update_selection); self.files.currentItemChanged.connect(self.update_selection)
        right.addWidget(self.files, 1)
        self.caps = CapList()
        self.caps.setIconSize(QSize(CAP_W, CAP_H)); self.caps.setWordWrap(True)
        self.caps.setStyleSheet('QListWidget {background:#14191f; border:1px solid #36414f; border-radius:6px;} QListWidget::item {background:transparent; border:none;}'
                                'QScrollBar:vertical {background:#14191f; width:10px; margin:0; border:none;} QScrollBar::handle:vertical {background:#4a525d; border-radius:3px; min-height:30px; margin:2px;}'
                                'QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {height:0;}')
        self.caps_look = self.caps.look
        self.caps.itemClicked.connect(self._cap_clicked); self._cap_data = {}
        self.caps_bar = QWidget(); cb2 = QHBoxLayout(self.caps_bar); cb2.setContentsMargins(0, 0, 0, 0); cb2.setSpacing(8)
        self.caps_dest = ElidedLabel(); self.caps_change = QPushButton('Change folder'); self.caps_change.clicked.connect(self.pick_caps_dir)
        cb2.addWidget(QLabel('Saves to')); cb2.addWidget(self.caps_dest, 1); cb2.addWidget(self.caps_change); self.caps_bar.hide()
        right.addWidget(self.caps_bar)
        self.caps.itemChanged.connect(self.update_selection); self.caps.hide(); right.addWidget(self.caps, 1)
        self._capsListed.connect(self._caps_ready); self._capThumb.connect(self._cap_thumb)
        self.detail = QLabel('Full project import includes raw scan files and any finished results.')
        self.detail.setObjectName('ImportHint'); self.detail.setWordWrap(True); right.addWidget(self.detail)
        self.slow_note = QLabel(''); self.slow_note.setWordWrap(True); self.slow_note.setStyleSheet('color:%s;' % T['warn']); right.addWidget(self.slow_note)
        # the picked project's side previews (the scanner's own data/<side>/preview.png), read off the device in the
        # background so a slow USB read never holds the page
        strip = QWidget(); self.previews = QHBoxLayout(strip); strip.setVisible(False); self.previews.setContentsMargins(0, 4, 0, 0); self.previews.setSpacing(8)
        self.previews.addStretch(1); strip.setFixedHeight(PREVIEW + 8); strip.hide(); right.addWidget(strip); self.preview_strip = strip   # shown once a project is picked
        self._preview_for = None; self._previewed.connect(self._show_previews)
        content.addWidget(listing, 1)
        transfer_header = QHBoxLayout(); transfer_header.addStretch(1)          # the bar and buttons speak for themselves: no heading
        box.addLayout(transfer_header)
        # no "0%" while nothing runs: the percent shows only once a copy reports progress
        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setTextVisible(False); box.addWidget(self.progress)
        self.status = QLabel(''); self.status.setWordWrap(True); box.addWidget(self.status)
        actions = QHBoxLayout()
        self.copy_b = QPushButton('Import selected projects'); self.copy_b.setObjectName('ImportPrimary'); self.copy_b.clicked.connect(self.copy); actions.addWidget(self.copy_b)
        self.stop_b = QPushButton('Cancel transfer'); self.stop_b.clicked.connect(self.cancel); actions.addWidget(self.stop_b)
        actions.addStretch(); self.close_b = QPushButton('Back'); self.just_imported = False; self.close_b.clicked.connect(self.dismiss); actions.addWidget(self.close_b)
        box.addLayout(actions)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.timed_out)
        self.transport_changed(); self._buttons(); self.hide()
        self._usb_timer = QTimer(self); self._usb_timer.timeout.connect(self.refresh_usb_state); self._usb_timer.start(2000)   # live USB mode

    def present(self):
        self.show(); self.raise_()

    def pick_library(self):
        d = QFileDialog.getExistingDirectory(self, 'Save imported projects to', self.library)
        if not d: return
        self.library = str(Path(d).resolve()); self._show_library()
        if self.default_c.isChecked(): set_default(self.library)

    def _show_library(self):
        home = str(Path.home()); t = self.library.replace(home, '~', 1) if self.library.startswith(home) else self.library
        self.location.setText(t); self.location.setToolTip(self.library)

    def refresh_usb_state(self):
        mode = usb_mode(); wifi = self.transport.currentIndex() == 1
        text = {'transfer': ('#6ccf8e', 'Scanner connected in File Transfer mode'), 'pc': ('#f2b43a', 'Scanner is in PC mode: choose File Transfer on it'),
                'other': ('#f2b43a', 'Scanner found: choose File Transfer on it'), None: (T['ink3'], 'No scanner on USB')}[mode]
        self.way_b[0].set_state(*text)
        if not (self.proc and self.mode == 'wifi'): self.way_b[1].set_state(T['ink3'], 'Receiver off')

    def transport_changed(self):
        wifi = self.transport.currentIndex() == 1
        for k, b in self.way_b.items(): b.setChecked(k == self.transport.currentIndex())
        self.refresh_usb_state()
        self.help.setText('Same Wi-Fi network for both. Enter the code shown below on the scanner.' if wifi else
                          'Copies go to this PC. The scanner keeps its originals.')
        self.connect_b.setText('Start Wi-Fi receiver' if wifi else 'Connect USB')
        self.status.setText('Start the receiver, then send projects from the scanner.' if wifi else 'Choose File Transfer on the scanner, then connect USB.')
        self.code.setText(''); self.files.clear(); self.caps.clear(); self._caps_rows = None; self._cap_data = {}; self.root = None; self.incomplete = False; self._side()
        self.tabs.setVisible(not wifi)
        if wifi: self.tabs.setCurrentIndex(0)             # screenshots and recordings come over the cable only
        self._buttons(); self.update_selection()

    def _buttons(self):
        busy = self.proc is not None
        self.transport.setEnabled(not busy and not self.receipt)
        for c in getattr(self, 'way_b', {}).values(): c.setEnabled(not busy and not self.receipt)
        receiving = busy and self.mode == 'wifi'
        self.connect_b.setEnabled(not busy or receiving)
        if self.transport.currentIndex() == 1: self.connect_b.setText('Stop Wi-Fi receiver' if receiving else 'Start Wi-Fi receiver')
        self.connect_b.setStyleSheet('QPushButton {background:#d9534f; color:white;} QPushButton:hover {background:#e06561;}' if receiving else '')   # red while it stops something
        if hasattr(self, 'code_box'): self.code_box.setVisible(receiving and bool(self.code.text())); self.new_code_b.setVisible(receiving)
        if hasattr(self, 'refresh_b'): self.refresh_b.setEnabled(not busy)
        self.files.setEnabled(not busy or receiving); self.caps.setEnabled(not busy); self.tabs.setEnabled(not busy)
        tree = self.caps if self.tabs.currentIndex() else self.files
        self.all_b.setEnabled(not busy and (tree.count() if tree is self.caps else tree.topLevelItemCount()) > 0)
        self.none_b.setEnabled(not busy and (tree.count() if tree is self.caps else tree.topLevelItemCount()) > 0)
        selected = bool(self.caps_picked()) if self.tabs.currentIndex() else any(self.picked(self.files.topLevelItem(i)) for i in range(self.files.topLevelItemCount()))
        self.copy_b.setEnabled(not busy and bool(self.root) and selected and not self.incomplete)
        self.stop_b.setEnabled(busy and self.mode != 'unmount')
        self.close_b.setEnabled(not busy)

    @staticmethod
    def _code_html(code, ip):
        return ('<div style="font-size:13px;color:#a3aab3;">Enter this code on the scanner</div>'
                '<div style="font-size:34px;font-weight:700;letter-spacing:6px;color:#91c0fa;">%s</div>'
                '<div style="font-size:12px;color:#a3aab3;">This PC: %s</div>' % (code, ip))

    def _place_received(self, stage):
        """Wi-Fi already brought whole projects to this PC (into a holding folder in the library): move each into the
        library as is (a rename on the same drive, instant), then empty the holding folder. No Models/Full step: it's here."""
        stage = Path(stage); source = stage / 'Projects' if (stage / 'Projects').is_dir() else stage
        placed = []; attention = []
        for proj in sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink() and (p / 'data').is_dir()):
            from app2.transfer.incoming import install_received, received_target
            target = received_target(proj, Path(self.library) / proj.name)
            try: install_received(proj, target)
            except (OSError, ValueError):
                attention.append(proj.name + ': could not update this copy. Send the project again.'); continue
            try:
                report = adopt_project(target)
                if report['issues']: attention.append(proj.name + ': ' + issue_summary(report))
            except (OSError, ValueError) as exc:
                attention.append(proj.name + ': could not prepare raw scan frames: ' + str(exc))
            if self.names.get(proj.name):
                try: rename_project(str(target), self.names[proj.name])
                except (OSError, ValueError): pass
            placed.append((proj.name, target)); self.imported.emit(str(target))
        try:
            if stage.name != '.wifi-incoming' and '.wifi-incoming' in stage.parts and not any(source.iterdir()):
                shutil.rmtree(stage)   # keep any failed incoming project for a later retry
        except OSError: pass
        self.files.clear(); self._received_view(True)
        for name, target in placed:
            try: sides = sum(1 for d in (target / 'data').iterdir() if d.is_dir())
            except OSError: sides = 0
            size = disk_usage.folder_bytes(str(target))
            it = ProjectItem([name, '', str(sides), '', '', nice(size), '', '', 'Yes', scanned_text(name), 'Received ' + datetime.now().strftime('%I:%M %p').lstrip('0')])
            it.setData(0, Qt.UserRole, name); it.setData(0, Qt.UserRole + 1, dict(name=name, sized=True, sides=sides, full=size, path=str(target)))
            it.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            self.files.addTopLevelItem(it);self._relabel(it)
        self.root = None; self.just_imported = bool(placed)
        if placed: self.close_b.setText('Show in Your projects')
        self.status.setText(('Received %d project%s over Wi-Fi in %s. They are in Your projects.' % (len(placed), 's' if len(placed) != 1 else '', clock(time.monotonic() - self._t0)))
                            if placed else 'Nothing complete arrived. Send the project again from the scanner.')
        if attention: self.status.setText(self.status.text() + ' ' + ' '.join(attention))
        self.update_selection()

    def _received_view(self, on):
        """Wi-Fi lists what already arrived: no Models/Full choice or USB copy times, just the project, sides, size, status."""
        for col in (3, 4, 6, 7): self.files.setColumnHidden(col, on)
        self.files.headerItem().setText(0, 'Received project' if on else 'Scanner project'); self.files.headerItem().setText(5, 'Size' if on else 'Full')

    def _row_menu(self, pos):
        """Right-click a project: Rename, and for one on this PC, Open and Show in folder."""
        item = self.files.itemAt(pos)
        if item is None or item.parent() is not None: return
        row = item.data(0, Qt.UserRole + 1) or {}; path = row.get('path') or os.path.join(self.library, item.data(0, Qt.UserRole))
        m = QMenu(self); here = os.path.isdir(os.path.join(path, 'data'))
        m.addAction('Rename...', lambda: self._rename(item, 0))
        a = m.addAction('Open', lambda: self.imported.emit(path) or self.open_requested.emit(path)); a.setEnabled(here)
        b = m.addAction('Show in folder', lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path))); b.setEnabled(here)
        m.exec(self.files.viewport().mapToGlobal(pos))

    def new_code(self):
        """Restart the Wi-Fi receiver: the scanner then shows a fresh code to enter."""
        if self.proc and self.mode == 'wifi': self._restart_wifi = True; self.cancel()

    def connect(self):
        if self.proc and self.mode == 'wifi': self.cancel(); return          # the button is Stop while the receiver waits
        if self.proc: return
        self.connect_requested.emit('wifi' if self.transport.currentIndex() == 1 else 'usb')

    def start_transport(self):
        if self.transport.currentIndex() == 1:
            self.root = None; self.files.clear()
            self.run('wifi', ['--dest', self.library])
        elif self.root and self._root_ok():
            self.scan()
        else:
            self.root = None; self.receipt = None; self._side()
            self.run('mount', ['--mountpoint', self.mountpoint])

    def _root_ok(self):
        """The scanner still answers through our mount; a dead one (I/O error, or gone) is released so we can mount again."""
        try: os.listdir(self.root); return True
        except OSError:
            subprocess.run(['fusermount', '-u', self.mountpoint], capture_output=True, timeout=10)   # our own mount only
            return False

    def refresh(self):
        """Read the list again. A mount that stopped answering (I/O errors) is released first, then connected again."""
        if self.proc: return
        self.stop_sizes()
        if self.root and self.transport.currentIndex() == 0 and not self._root_ok():
            self.status.setText('The scanner stopped answering. Reconnecting...')
        self.connect()

    def scan(self):
        self._received_view(False); self._caps_rows = None; self._cap_data = {}
        self.files.clear(); self.run('list', ['--root', self.root, '--dest', self.library])

    KINDS = {1: 'image', 2: 'video'}
    PAGES = (('Import from scanner', 'Bring complete scanner projects into this PC workspace. Raw scans and finished results stay together.'),
             ('Import screenshots', "Copy pictures of the scanner's screen to this PC. Click one to see it larger. The scanner keeps its originals."),
             ('Import screen recordings', "Copy the scanner's screen recordings to this PC. Click one to play it. The scanner keeps its originals."))
    def _tab(self, i):
        """Projects / Screenshots / Recordings: one list shown at a time; the scanner and its mount are shared."""
        self.files.setVisible(i == 0); self.caps.setVisible(i > 0); self.caps_bar.setVisible(i > 0)
        self.detail.setVisible(i == 0); self.slow_note.setVisible(i == 0)
        if i: self._show_caps_dest()
        if i: self.preview_strip.hide()
        self._side()
        self.list_title.setVisible(i == 0)                       # the steps belong to importing projects; pictures and videos are their own, one-step job
        title, sub = self.PAGES[i]; self.page_title.setText(title); self.page_sub.setText(sub)
        if i and self.root and self._caps_rows is None and not self.proc: self._list_caps()
        elif i and self._caps_rows is not None: self._fill_caps()
        elif i == 0 and self.root and self.files.topLevelItemCount() and not getattr(self, 'sizer', None): self.start_sizes()
        if i and not self.root:
            self.caps.clear(); word = ('screenshots', 'screen recordings')[i - 1]; self.copy_b.setText('Import selected ' + word); self._buttons()
            if not self.proc and self.transport.currentIndex() == 0 and usb_mode() == 'transfer':
                self.selection.setText('Connecting to the scanner...'); self.connect()          # File Transfer is on: no separate Connect step needed
            else: self.selection.setText('Plug in the scanner and choose File Transfer on it, then press Refresh to see its ' + word)
        else: self.update_selection()

    def _side(self):
        """Screenshots and Recordings only need the scanner already connected: the connect column steps aside for the grid."""
        self.connection.setVisible(not self.tabs.currentIndex())

    def _kind(self): return ('screenshots', 'recordings')[self.tabs.currentIndex() - 1]

    def _show_caps_dest(self):
        d = capture_dir(self._kind()); self.caps_dest.setText(d); self.caps_dest.setToolTip(d + '\nChange it here, or for all of them in File > Preferences.')

    def pick_caps_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Save %s to' % self._kind(), capture_dir(self._kind()))
        if d: set_capture_dir(self._kind(), d); self._show_caps_dest(); self._fill_caps()

    def _cap_clicked(self, item):
        if self.caps_look.ticked: self.caps_look.ticked = False; return          # that click was the tick circle
        rel = item.data(Qt.UserRole)
        if self.tabs.currentIndex() == 2:                                       # a recording plays in the PC's video player
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(self.root, rel))); return
        rows = [(self.caps.item(i).data(Qt.UserRole), self.caps.item(i).toolTip()) for i in range(self.caps.count())]
        start = next((i for i, (r, _) in enumerate(rows) if r == rel), 0)
        def get(r):
            if r not in self._cap_data:
                try:
                    with open(os.path.join(self.root, r), 'rb') as f: self._cap_data[r] = f.read()
                except OSError: return b''
            return self._cap_data[r]
        def find(r): return next((self.caps.item(i) for i in range(self.caps.count()) if self.caps.item(i).data(Qt.UserRole) == r), None)
        CaptureViewer(rows, start, get, lambda r: (find(r) is not None and find(r).checkState() == Qt.Checked),
                      lambda r, on: find(r).setCheckState(Qt.Checked if on else Qt.Unchecked), self).exec()

    def _list_caps(self):
        self.stop_sizes(); self.caps.clear(); self.selection.setText('Reading the scanner...'); root = self.root
        def read():
            from app2.transfer.captures import list_captures
            try: rows = list_captures(root)
            except (OSError, ValueError): rows = []
            self._capsListed.emit(rows)
        threading.Thread(target=read, name='import-captures', daemon=True).start()

    def _caps_ready(self, rows):
        self._caps_rows = sorted(rows, key=lambda r: (capture_time(r['name']) or datetime.fromtimestamp(r['modified'])), reverse=True)
        if self.tabs.currentIndex(): self._fill_caps()

    def _fill_caps(self):
        kind = self.KINDS.get(self.tabs.currentIndex()); self.caps.blockSignals(True); self.caps.clear(); todo = []
        dest = Path(capture_dir(self._kind()))
        for r in self._caps_rows or []:
            if r['kind'] != kind: continue
            d = capture_time(r['name']); when = (d.strftime('%b %d, %I:%M %p') if d.year == datetime.now().year else d.strftime('%b %d, %Y')).replace(' 0', ' ') if d else r['name']
            it = QListWidgetItem(QIcon(self._blank(kind)), '%s\n%s%s' % (when, nice(r['size']), ' · on this PC' if (dest / r['name']).exists() else ''))
            it.setCheckState(Qt.Unchecked); it.setData(Qt.UserRole, r['relpath']); it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable); it.setToolTip(capture_text(r['name'])); self.caps.addItem(it)
            it.setData(Qt.UserRole + 1, r['kind']); todo.append(r['relpath'])
        self.caps.blockSignals(False); self.update_selection()
        if todo and self.root:                            # pictures fill in one at a time behind the list; a copy stops them
            self._thumbs += 1; gen = self._thumbs; root = self.root
            def read():
                for rel in todo:
                    if gen != self._thumbs: return
                    try:
                        if rel.lower().endswith(('.png', '.jpg', '.jpeg')):
                            with open(os.path.join(root, rel), 'rb') as f: data = f.read()
                            self._cap_data[rel] = data
                        else: data = video_frame(os.path.join(root, rel))
                    except OSError: continue
                    if data: self._capThumb.emit(rel, data)
            threading.Thread(target=read, name='import-thumbs', daemon=True).start()

    @staticmethod
    def _blank(kind):
        """The picture before a screenshot or recording loads: a dark tile."""
        pm = QPixmap(CAP_W, CAP_H); pm.fill(QColor('#0d1116')); p = QPainter(pm)
        p.end(); return pm

    def _cap_thumb(self, rel, data):
        pm = QPixmap(); pm.loadFromData(data)
        if pm.isNull(): return
        pm = pm.scaled(CAP_W, CAP_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        for i in range(self.caps.count()):
            it = self.caps.item(i)
            if it.data(Qt.UserRole) == rel: it.setIcon(QIcon(pm)); break

    def caps_picked(self):
        return [self.caps.item(i).data(Qt.UserRole) for i in range(self.caps.count()) if self.caps.item(i).checkState() == Qt.Checked]

    def select_all(self, selected):
        if self.tabs.currentIndex():
            for i in range(self.caps.count()): self.caps.item(i).setCheckState(Qt.Checked if selected else Qt.Unchecked)
            return
        for index in range(self.files.topLevelItemCount()):
            it = self.files.topLevelItem(index); it.setCheckState(3, Qt.Checked if selected else Qt.Unchecked)
            if not selected: it.setCheckState(5, Qt.Unchecked)
        self.update_selection()

    def update_selection(self, *_args):
        if self.tabs.currentIndex():
            n = len(self.caps_picked()); total = self.caps.count(); word = ('screenshot', 'screen recording')[self.tabs.currentIndex() - 1]
            self.selection.setText((f'{n} selected' if n else f'{total} {word}' + ('s' if total != 1 else '') + ' on the scanner') if total else 'Nothing of this kind on the scanner')
            self.copy_b.setText(f'Import {n} {word}' + ('s' if n != 1 else '') if n else f'Import selected {word}s'); self._buttons(); return
        count = 0; models = full = files = mfiles = nfull = 0; known = True
        for index in range(self.files.topLevelItemCount()):
            item = self.files.topLevelItem(index)
            if self.picked(item):
                row = item.data(0, Qt.UserRole + 1) or {}; count += 1; known = known and bool(row.get('sized'))
                if row.get('what') == 'full': full += row.get('full') or 0; files += row.get('files') or 0; nfull += 1
                else: models += row.get('models') or 0; mfiles += row.get('mfiles') or 0
        secs = files * SECONDS_PER_FILE + full / BYTES_PER_SECOND + mfiles * SECONDS_PER_FILE + models / BYTES_PER_SECOND
        sizes = (f' · {count - nfull} models only, {nfull} full · {nice(models + full)}, about {eta(files + mfiles, models + full)}' if known else ' · sizes still loading')
        long_ = known and count and secs > LONG_COPY_S and self.transport.currentIndex() == 0
        self.slow_note.setText('Over USB this takes about %s. Wi-Fi Share to PC may be faster for full projects.' % eta(files + mfiles, models + full) if long_ else '')
        self.selection.setText(f'{count} selected' + sizes if count else
                               (f'{self.files.topLevelItemCount()} project' + ('s' if self.files.topLevelItemCount() != 1 else '') + (' received' if self.files.isColumnHidden(3) else ' on the scanner')) if self.files.topLevelItemCount() else 'Connect to see scanner projects')
        self.copy_b.setText(f'Import {count} project' + ('s' if count != 1 else '') if count else 'Import selected projects')
        current = self.files.currentItem()
        row = current.data(0, Qt.UserRole + 1) if current else None
        if row:
            self.detail.setText(f"{row['name']} · {row['sides']} side" + ('s' if row['sides'] != 1 else '') +
                                (f" · {row['files']} files" if row.get('files') is not None else "") + f" · {'Finished mesh included' if row['mesh'] else 'No finished mesh yet'}")
        else: self.detail.setText('Full project import includes raw scan files and any finished results.')
        self._buttons()

    def _show_device(self):
        """Scanner storage from the mounted file system, and its serial from the USB ids (nothing sent to the device)."""
        serial = ''
        try:
            for d in os.scandir('/sys/bus/usb/devices'):
                try:
                    if open(os.path.join(d.path, 'idVendor')).read().strip() == '2207': serial = open(os.path.join(d.path, 'serial')).read().strip(); break
                except OSError: continue
        except OSError: pass
        try:
            st = os.statvfs(self.root); total = st.f_blocks * st.f_frsize; free = st.f_bavail * st.f_frsize
        except (OSError, TypeError): total = free = 0
        self._serial = serial; self._show_serial()
        if total:
            self.dev_info.setText('%s used · %s free of %s' % (nice(total - free), nice(free), nice(total)))
            self.dev_bar.setRange(0, 1000); self.dev_bar.setValue(int(1000 * (total - free) / total))
        else: self.dev_info.setText('Storage not reported')
        self.dev_card.show()

    def _relabel(self, item):
        """The project cell: open/closed arrow, the human name (with the scanner's name after it) and a warning mark."""
        row = item.data(0, Qt.UserRole + 1) or {}; name = item.data(0, Qt.UserRole)
        human = self.names.get(name)
        text = (f'{human}   ({name})' if human else name) + ('  \u26a0' if row.get('bad') else '')
        item.setText(0, text)
        item.setToolTip(0, (f'{human}\nScanner name: {name}\n' if human else '') + 'Double-click to give it a name')

    def _rename(self, item, col):
        if col != 0 or item.parent() is not None: return
        name = item.data(0, Qt.UserRole)
        text, ok = QInputDialog.getText(self, 'Name this project', f'A name for {name} (kept on this PC; the scanner name stays):', text=self.names.get(name, ''))
        if not ok: return
        text = text.strip()[:100]
        if text: self.names[name] = text
        else: self.names.pop(name, None)
        save_names(self.names); self._relabel(item)

    @staticmethod
    def picked(item):
        """What this row imports: 'full', 'models' or None (the ticks sit on the Models and Full cells)."""
        if item.checkState(5) == Qt.Checked: return 'full'
        if item.checkState(3) == Qt.Checked: return 'models'
        return None

    def _ticked(self, item, col):
        """Models and Full are either/or on a row: Full already holds the models."""
        if col not in (3, 5) or item.parent() is not None or self._ticking: return
        self._ticking = True
        try:
            if item.checkState(col) == Qt.Checked: item.setCheckState(5 if col == 3 else 3, Qt.Unchecked)
            row = dict(item.data(0, Qt.UserRole + 1) or {}); row['what'] = self.picked(item) or 'models'; item.setData(0, Qt.UserRole + 1, row)
        finally: self._ticking = False
        self.update_selection()

    def _set_copy(self, item, what):
        row = dict(item.data(0, Qt.UserRole + 1) or {}); row['what'] = what
        item.setData(0, Qt.UserRole + 1, row); self.update_selection()

    def _show_serial(self):
        s = self._serial
        if s and not self.eye_b.isChecked(): s = '\u2022' * len(s)     # fully hidden by default: nothing to match in a screenshot
        self.dev_serial.setText(('serial ' + s) if s else ''); self.eye_b.setVisible(bool(self._serial))

    def _toggle_copy(self, item, col): pass

    def _expanded(self, item):
        """The arrow opened a project: read its side pictures (the placeholder child keeps the arrow showing until then)."""
        if item.parent() is None and not getattr(self, '_opening', False): self._preview_for = None; self._load_previews(item)

    def _collapsed(self, item):
        if item.parent() is None and self._preview_for == item.data(0, Qt.UserRole): self._preview_for = None

    def _load_previews(self, item=None):
        item = item or self.files.currentItem(); name = item.data(0, Qt.UserRole) if item else None
        if name == self._preview_for: return
        self._preview_for = name; self._show_previews(name, None)
        if not name:return
        row=item.data(0,Qt.UserRole+1) or {}
        project_path=Path(row['path']) if row.get('path') else None
        if project_path is None:
            if not self.root:return
            base=Path(self.root)/'Projects' if (Path(self.root)/'Projects').is_dir() else Path(self.root)
            project_path=base/name
        def read():
            got = []
            try:
                for side in sorted(p for p in (project_path/'data').iterdir() if p.is_dir()):
                    try: data=(side/'preview.png').read_bytes()
                    except OSError: data=b''
                    got.append((side.name,data,scan_details(side)))
            except OSError: pass
            self._previewed.emit(name, got)
        threading.Thread(target=read, name='import-previews', daemon=True).start()

    def _show_previews(self, name, shots):
        """The picked project's side pictures in a row that opens right under it (one open at a time)."""
        if name != self._preview_for: return                     # the pick moved on while this was read
        for i in range(self.files.topLevelItemCount()):          # close any open preview row
            it = self.files.topLevelItem(i)
            while it.childCount(): it.removeChild(it.child(0))
            self._relabel(it)
        item = next((self.files.topLevelItem(i) for i in range(self.files.topLevelItemCount())
                     if self.files.topLevelItem(i).data(0, Qt.UserRole) == name), None)
        if item is None: return
        box=QWidget();h=QVBoxLayout(box);h.setContentsMargins(28,6,6,8);h.setSpacing(8)
        if shots is None or not shots:
            lab = QLabel('Loading previews...' if shots is None else 'No preview pictures on the scanner for this project.')
            lab.setObjectName('ImportHint'); h.addWidget(lab)
        else:
            for shot in shots:
                side,data=shot[:2];details=shot[2] if len(shot)>2 else side
                row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
                if data: layout.addWidget(Thumb(cropped(data),'Scan '+side))
                else:
                    image=QLabel('Preview not received');image.setFixedSize(PREVIEW,PREVIEW);layout.addWidget(image)
                label=QLabel(details);label.setWordWrap(True);layout.addWidget(label,1);h.addWidget(row)
        h.addStretch(1)
        child = QTreeWidgetItem(item); child.setFirstColumnSpanned(True); child.setFlags(Qt.ItemIsEnabled)
        tall = (PREVIEW+14)*len(shots)+16 if shots else 34                     # the row's real height (a widget's size hint is 0 before it is shown)
        child.setSizeHint(0, QSize(0, tall)); box.setFixedHeight(tall); self.files.setItemWidget(child, 0, box)
        self._opening = True; item.setExpanded(True); self._opening = False; self._relabel(item)
        self.files.scrollToItem(item)  # keep the project's count and sizes visible above its previews

    def copy(self):
        if self.tabs.currentIndex():
            paths = self.caps_picked()
            if not paths: self.status.setText('Select at least one.'); return
            self._thumbs += 1                                  # stop the picture reader: the scanner takes one reader at a time
            self._cap_paths = paths; self._cap_dest = capture_dir(self._kind()); self.run('capcopy', ['--root', self.root, '--dest', self._cap_dest, '--flat', '--paths-json', json.dumps(paths)]); return
        names = [self.files.topLevelItem(i).data(0, Qt.UserRole)
                 for i in range(self.files.topLevelItemCount()) if self.picked(self.files.topLevelItem(i))]
        if not names: self.status.setText('Select at least one project.'); return
        existing = [n for n in names if os.path.lexists(os.path.join(self.library, n))]
        policy = 'skip'
        if existing:
            policy = self.ask_existing(existing)
            if policy is None: return
        what = {}
        for i in range(self.files.topLevelItemCount()):
            it = self.files.topLevelItem(i)
            if self.picked(it): what[it.data(0, Qt.UserRole)] = self.picked(it)
        self.run('import', ['--root', self.root, '--dest', self.library, '--names-json', json.dumps(names), '--existing', policy, '--what-json', json.dumps(what)])

    def ask_existing(self, existing):
        """Some picked projects are already on this PC: skip them (default), import another copy, or stop."""
        n = len(existing)
        box = QMessageBox(self); box.setWindowTitle('Already on this PC'); box.setIcon(QMessageBox.Information)
        box.setText((f'{existing[0]} is' if n == 1 else f'{n} of the selected projects are') + ' already on this PC.')
        box.setInformativeText('Skip keeps the copy you have. Import again adds a second copy with a number after its name.')
        skip = box.addButton('Skip', QMessageBox.AcceptRole); again = box.addButton('Import again', QMessageBox.DestructiveRole)
        box.addButton('Cancel', QMessageBox.RejectRole); box.setDefaultButton(skip); box.exec()
        return 'skip' if box.clickedButton() is skip else 'copy' if box.clickedButton() is again else None

    def start_sizes(self):
        """Read every project's models-only and full size in a second worker, filling the Size column as it goes. The
        list stays usable; any other job stops it first (the scanner takes one reader at a time)."""
        self.stop_sizes()
        if not self.root: return
        p = QProcess(self); self.sizer = p; self._size_buf = ''
        p.setWorkingDirectory(REPO); p.setProgram(sys.executable)
        p.setArguments(['-m', 'app2.transfer.projects', 'sizes', '--root', self.root])
        def read():
            if self.sizer is not p: return
            self._size_buf += bytes(p.readAllStandardOutput()).decode('utf8', 'replace')
            while '\n' in self._size_buf:
                line, self._size_buf = self._size_buf.split('\n', 1)
                try: event = json.loads(line)
                except ValueError: continue
                if event.get('event') == 'size': self.worker_event(event)
        p.readyReadStandardOutput.connect(read)
        p.finished.connect(lambda *_: (read(), setattr(self, 'sizer', None) if self.sizer is p else None, p.deleteLater()))
        p.start()

    def stop_sizes(self):
        p = getattr(self, 'sizer', None); self.sizer = None
        if not p: return
        p.terminate()
        def kill():
            try:
                if p.state() != QProcess.NotRunning: p.kill()
            except RuntimeError: pass                  # already finished and deleted
        QTimer.singleShot(1500, kill)

    def run(self, mode, args):
        self._t0 = time.monotonic()
        if mode == 'import': self._import_names = json.loads(args[args.index('--names-json') + 1])
        self.stop_sizes(); self.mode = mode; self.buffer = ''; self.failed = False; self.skipped = []; self._prepare_notes = []; self._prepared_sides = 0
        process = QProcess(self); self.proc = process
        process.setWorkingDirectory(REPO); process.setProgram(sys.executable)
        process.setArguments(['-m', 'app2.transfer.captures', 'copy'] + args if mode == 'capcopy' else ['-m', 'app2.transfer.projects', mode] + args)
        process.readyReadStandardOutput.connect(self.read)
        process.finished.connect(self.finished); process.errorOccurred.connect(self.process_error)
        self.status.setText({'mount':'Connecting to USB File Transfer...', 'list':'Reading scanner projects...',
                             'import':'Copying complete projects...', 'wifi':'Waiting for scanner Wi-Fi transfer...',
                             'unmount':'Releasing USB...', 'capcopy':'Copying from the scanner...'}[mode])
        self.progress.setRange(0,100 if mode=='wifi' else 0);self.progress.setValue(0)
        self.progress.setVisible(mode!='wifi');self.progress.setTextVisible(False);self._buttons()
        self.timer.start(60000 if mode != 'wifi' else 1800000); process.start()

    def read(self):
        if not self.proc: return
        self.timer.start(60000 if self.mode != 'wifi' else 1800000)
        self.buffer += bytes(self.proc.readAllStandardOutput()).decode('utf8', 'replace')
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            try: self.worker_event(json.loads(line))
            except (ValueError, TypeError): continue

    def worker_event(self, event):
        kind = event.get('event')
        if kind == 'mounted': self.root = event['root']; self.receipt = event; self._show_device(); self._side()
        elif kind == 'project':
            event = dict(event, what='models', scanned=scanned_at(event['name']) or event.get('modified') or 0)
            item = ProjectItem([event['name'], '', str(event['sides']), 'reading', 'reading', 'reading', 'reading', 'Yes' if event['mesh'] else 'No',
                                    'Yes' if event.get('exists') else 'Not yet', scanned_text(event['name']), ''])
            item.setData(0, Qt.UserRole, event['name']); item.setData(0, Qt.UserRole + 1, event)
            if event.get('scanned'): item.setToolTip(9, datetime.fromtimestamp(event['scanned']).strftime('%A, %B %d, %Y, %I:%M:%S %p'))
            if self.mode != 'wifi':
                item.setCheckState(3, Qt.Unchecked); item.setCheckState(5, Qt.Unchecked)     # USB Models or Full choice
            else:
                for column in (1,4,6): self.files.setColumnHidden(column,True)
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)                 # the arrow opens its side pictures
            if self.mode != 'wifi':
                item.setToolTip(3, 'Tick to import the finished models and previews (fast)'); item.setToolTip(5, 'Tick to import the full project with every raw frame (slow over USB)')
            self.files.addTopLevelItem(item); self._relabel(item); self.update_selection()
        elif kind == 'size':
            for i in range(self.files.topLevelItemCount()):
                item = self.files.topLevelItem(i); row = item.data(0, Qt.UserRole + 1) or {}
                if row.get('name') != event['name']: continue
                row.update(models=event['models'], full=event['full'], files=event['files'], mfiles=event.get('mfiles'), sized=event['done'])
                item.setData(0, Qt.UserRole + 1, row)
                item.setText(3, nice(event['models'])); item.setText(4, eta(event.get('mfiles') or 0, event['models']))
                item.setText(5, nice(event['full'])); item.setText(6, eta(event['files'], event['full']))
                if event.get('bad'):
                    row['bad'] = event['bad']; item.setData(0, Qt.UserRole + 1, row); self._relabel(item)
                    item.setToolTip(4, '%d file%s in this project report a broken size on the scanner. They are skipped when importing.' % (event['bad'], 's' if event['bad'] != 1 else ''))
                for c in (3, 4, 5, 6): item.setForeground(c, QBrush(QColor(T['ink'] if event['done'] else T['ink3'])))   # dim while still counting
                if event.get('state'):
                    row['state'] = event['state']; item.setData(0, Qt.UserRole + 1, row)
                    item.setText(10, {'seen': 'Seen before', 'changed': 'Changed on scanner', 'new': 'New'}[event['state']] if event['done'] else 'Counting...')
                    if event.get('first_seen'): item.setToolTip(10, 'First seen here ' + datetime.fromtimestamp(event['first_seen']).strftime('%b %d, %Y, %I:%M %p'))
                for c in (3, 4, 5, 6): item.setToolTip(c, f"Finished models {nice(event['models'])}; full project {nice(event['full'])} in {event['files']} files, about {eta(event['files'])} to copy over USB" + ('' if event['done'] else ' (still counting)'))
            self.update_selection()
        elif kind == 'listening':
            self.code.setText(self._code_html(event['code'], event['ip']))
            self.way_b[1].set_state('#6ccf8e', 'Waiting for the scanner'); link_status.set_wifi('waiting', event['code']); self.code_box.show(); self._stage = event.get('stage')      # the code arrives after the buttons were set
            self.status.setText('Waiting for the scanner');self.progress.hide()
        elif kind=='connected': self.status.setText('Waiting for the scanner');self.progress.hide()
        elif kind=='received_file':
            self.root=event.get('stage') or self.root
            name=event.get('name');scan=event.get('scan');self._current_file=event.get('relpath','')
            item=next((self.files.topLevelItem(i) for i in range(self.files.topLevelItemCount())
                       if self.files.topLevelItem(i).data(0,Qt.UserRole)==name),None)
            if name and item is None:
                self.worker_event(dict(event='project',name=name,sides=0,mesh=False))
                item=self.files.topLevelItem(self.files.topLevelItemCount()-1)
            if item is not None:
                from app2.transfer.incoming import project_path, side_names
                row=item.data(0,Qt.UserRole+1) or {}
                incoming = project_path(event.get('stage') or self.root, name)
                row['path'] = str(incoming)
                scans=set(row.get('received_scans',[])); scans.update(side_names(incoming))
                if scan: scans.add(scan)
                row['received_scans']=sorted(scans); item.setText(2,str(len(scans)))
                sizes = dict(row.get('received_sizes', {})); rel = event.get('relpath', '')
                if rel:
                    count = event.get('file_bytes')
                    if count is None:
                        try: count = (Path(event.get('stage') or self.root) / rel).stat().st_size
                        except OSError: count = 0
                    sizes[rel] = int(count)
                row['received_sizes'] = sizes
                row['full'] = sum(sizes.values()); row['models'] = sum(n for path,n in sizes.items() if 'cache' not in Path(path).parts)
                row['sized'] = False; item.setData(0,Qt.UserRole+1,row)
                item.setText(3, nice(row['models']) + ' received'); item.setText(5, nice(row['full']) + ' received')
                item.setText(4, ''); item.setText(6, '')
                for column in (3,5):
                    item.setToolTip(column, 'Files received so far. The project is incomplete until the transfer ends.')
                    self.files.setColumnWidth(column, max(self.files.columnWidth(column), self.files.sizeHintForColumn(column)))
                now=time.monotonic()
                if item.isExpanded() and now-getattr(self,'_received_preview_at',0)>=1:
                    self._received_preview_at=now;self._preview_for=None;self._load_previews(item)
            self.progress.show()
            self.status.setText('Receiving '+self._current_file)
        elif kind == 'progress':
            self.progress.show()
            if event.get('relpath'): self._current_file=event['relpath']
            total = event.get('total') or 0; done = event.get('bytes') or 0
            if total: self.progress.setRange(0,100); self.progress.setTextVisible(True); self.progress.setValue(min(99,int(done*100/total)))
            rate = event.get('rate') or 0; names = getattr(self, '_import_names', []); link_status.set_rate(rate)
            if self.mode == 'wifi': link_status.set_wifi('receiving', link_status.state.get('code'))
            if self.mode == 'capcopy':
                paths = getattr(self, '_cap_paths', []); k = paths.index(event['relpath']) + 1 if event.get('relpath') in paths else 0
                self.progress.setRange(0, 100); self.progress.setValue(int(100 * (k - 1) / max(1, len(paths))))
                self.status.setText('Copying %d of %d · %s elapsed' % (k, len(paths), clock(time.monotonic() - self._t0))); return
            which = (' %d of %d' % (names.index(event['name']) + 1, len(names))) if event.get('name') in names else ''
            eta_total=event.get('transfer_total') if self.mode=='wifi' else total
            left=(' · about %s left' % clock((eta_total-done)/rate)) if eta_total and rate else ' · time left unknown'
            self.progress.setFormat('Known files: %p%' if self.mode=='wifi' else '%p%')
            self.status.setText(f"{'Copying' + which if self.mode == 'import' else 'Receiving'} · {clock(time.monotonic() - self._t0)} elapsed{left} · "
                                f"{done/1048576:.1f} of {total/1048576:.1f} MB · {rate/1048576:.1f} MB/s" if total else
                                f"{done/1048576:.1f} MB received · {rate/1048576:.1f} MB/s · time left unknown")
            current=event.get('relpath') or getattr(self,'_current_file','')
            if current: self.status.setText(current+' · '+self.status.text())
        elif kind == 'done' and self.mode == 'wifi':
            self.root = event.get('stage'); self.incomplete = bool(event.get('incomplete'))
            if self.incomplete: self.status.setText('Transfer has incomplete files. Send the projects again before importing.')
        elif kind == 'badfiles':
            self.status.setText('%s: skipped %d file%s the scanner reports with a broken size.' % (event['name'], event['count'], 's' if event['count'] != 1 else ''))
        elif kind == 'saved' and self.mode == 'capcopy': pass
        elif kind == 'saved':
            if self.names.get(event['name']):
                try: rename_project(event['path'], self.names[event['name']])
                except (OSError, ValueError): pass
            prepared = event.get('prepared') or {}
            if prepared.get('issues'):
                self._prepare_notes.append(event['name'] + ': ' + issue_summary(prepared))
            elif prepared.get('status') == 'prepared':
                self._prepared_sides += len(prepared['sides'])
            self.imported.emit(event['path']); self.status.setText('Imported ' + event['name'] + ' into this PC library.')
            for i in range(self.files.topLevelItemCount()):
                if self.files.topLevelItem(i).data(0, Qt.UserRole) == event['name']: self.files.topLevelItem(i).setText(8, 'Yes')
        elif kind == 'exists':
            self.skipped.append(event['name']); self.status.setText('Skipped ' + event['name'] + ': it is already on this PC.')
        elif kind == 'badcode': self.status.setText('Incorrect code entered on scanner. Restart receiver if locked.' if event.get('locked') else 'Incorrect scanner code; try the code above.')
        elif kind == 'error': self.failed = True; self.status.setText(event.get('message', 'Transfer failed.'))

    def finished(self, code, _status):
        process = self.proc
        if not process: return
        self.read(); mode = self.mode; error = bytes(process.readAllStandardError()).decode('utf8','replace')
        self.timer.stop(); self.proc = None; process.deleteLater()
        if mode == 'wifi':                                       # the receiver ended: nothing left on screen from it
            link_status.set_wifi(None); self.code.setText(''); self.code_box.hide(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setTextVisible(False)
            self.way_b[1].set_state(T['ink3'], 'Receiver off')
            st = getattr(self, '_stage', None)
            if st and os.path.isdir(st) and not os.listdir(st) and '.wifi-incoming' in Path(st).parts: os.rmdir(st)   # nothing came: no empty folder left
            if code or self.failed: self.status.setText('Wi-Fi receiver stopped.'); self.failed = True
        if getattr(self, '_restart_wifi', False) and mode == 'wifi':
            self._restart_wifi = False; self.code.setText(''); self.failed = False; self.run('wifi', ['--dest', self.library]); return
        self.progress.setRange(0,100); self._buttons()
        if code or self.failed:
            if not self.failed: self.status.setText('Transfer stopped. ' + (error.strip()[-300:] or 'Try connecting again.'))
            return
        if mode in ('import', 'wifi', 'capcopy'): self.progress.setTextVisible(True); self.progress.setValue(100)
        else: self.progress.setTextVisible(False); self.progress.setValue(0)
        if mode == 'mount': self.scan()
        elif mode == 'wifi':
            if self.incomplete: self._buttons()
            elif self.root: self._place_received(self.root)
            else: self.status.setText('No projects received. Start the receiver and try again.')
        elif mode == 'list':
            self.status.setText(f'{self.files.topLevelItemCount()} projects found. Sizes fill in as they are read. Select those to import.')
            if self.tabs.currentIndex(): self._list_caps()
            else: self.start_sizes()
        elif mode == 'capcopy':
            n = len(getattr(self, '_cap_paths', [])); self._fill_caps()
            self.status.setText('Copied %d to %s in %s. The scanner kept its originals.' % (n, self._cap_dest, clock(time.monotonic() - self._t0)))
        elif mode == 'import':
            skipped = ('Skipped %d already on this PC. ' % len(self.skipped)) if self.skipped else ''
            self.just_imported = True; self.close_b.setText('Show in Your projects')
            self.status.setText(skipped + 'Import complete in %s. Open a project from Workspace. Scanner originals were kept.' % clock(time.monotonic() - self._t0))
            if self._prepared_sides: self.status.setText(self.status.text() + ' Prepared %d raw side%s for Build.' %
                                                         (self._prepared_sides, 's' if self._prepared_sides != 1 else ''))
            if self._prepare_notes: self.status.setText(self.status.text() + ' ' + ' '.join(self._prepare_notes))
        elif mode == 'unmount': self.receipt = None; self.root = None; self._dismissed()

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.timer.stop(); self.proc.deleteLater(); self.proc = None; self._buttons()
            self.status.setText('Could not start the transfer worker.')

    def cancel(self):
        if self.proc:
            self.failed = True; self.status.setText('Canceling transfer. Partial projects will not appear in Workspace.')
            process = self.proc; process.terminate()
            QTimer.singleShot(1500, lambda: process.kill() if self.proc is process else None)

    def timed_out(self):
        self.cancel(); self.status.setText('The scanner stopped responding. Check the connection and try again.')

    def dismiss(self):
        if self.proc: return
        if self.receipt and self.receipt.get('owned'):
            self.run('unmount', ['--mountpoint', self.receipt['mountpoint'], '--token', self.receipt['token']])
        else: self._dismissed()

    def _dismissed(self):
        self.hide(); self.dismissed.emit()
