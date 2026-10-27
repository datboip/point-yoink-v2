# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Export scan timelapse sheet (File > Export scan timelapse..., and a link in the Export sheet): which sides, model
alone or camera and model, size, length, MP4 or GIF, camera motion during growth and the mark. Export runs app2/scan/timelapse.py as
a child process (its own GPU model and EGL context; the window never waits) and shows its progress; Stop ends it and
nothing is left behind. The clip is saved in the project folder as <project>_timelapse.<ext> (_2, _3 ... when taken).
A sheet over the whole window, like the Export sheet."""
import json, os, sys
from PySide6.QtCore import Qt, Signal, QEvent, QProcess, QProcessEnvironment, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QProgressBar, QGridLayout, QLineEdit
from app2.qt.theme import ui_font
from app2.qt.widgets import button, StableLabel, Scrim
from app2.qt.tool_panel import big_button
from app2.qt import build_words as W

WIDTH, HEIGHT = 560, 520
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECK_QSS = ("QCheckBox {spacing:8px;} QCheckBox::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;"
             "background:#151a21;} QCheckBox::indicator:checked {background:#5b9df0;border:2px solid #dceaff;}")


class TimelapseSheet(QFrame):
    done = Signal(dict)                    # the child's result: path, bytes, frames, seconds ...
    failed = Signal(str)
    stopped = Signal()
    closed = Signal()

    def __init__(self, parent):
        super().__init__(parent); self.setObjectName("KeysSheet"); E = W.TIMELAPSE_SHEET
        self.scrim = Scrim(parent); self.scrim.hide(); self.scrim.clicked.connect(self.cancel)
        parent.installEventFilter(self)
        v = QVBoxLayout(self); v.setContentsMargins(26, 20, 26, 16); v.setSpacing(8)
        title = QLabel(E["title"]); title.setFont(ui_font(20, QFont.Weight.DemiBold)); title.setFixedHeight(34)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(title)
        intro = StableLabel(E["intro"], 2); intro.setProperty("role", "hint"); v.addWidget(intro)
        g = QGridLayout(); g.setHorizontalSpacing(12); g.setVerticalSpacing(8); self.seg = {}
        self.sides = QComboBox(); self.sides.setFixedHeight(30)
        g.addWidget(QLabel(E["sides"]), 0, 0); g.addWidget(self.sides, 0, 1, Qt.AlignmentFlag.AlignLeft)
        for r, (key, label, items) in enumerate((("layout", E["layout"], E["layouts"]), ("size", E["size"], E["sizes"]),
                                                  ("length", E["length"], E["lengths"]), ("format", E["format"], E["formats"])), 1):
            row = QHBoxLayout(); row.setSpacing(6); self.seg[key] = {}
            for val, text in items:
                b = button(text, "seg"); b.setCheckable(True); b.setFixedHeight(30)
                b.clicked.connect(lambda checked=False, k=key, x=val: self.pick(k, x)); row.addWidget(b); self.seg[key][val] = b
            row.addStretch(1); g.addWidget(QLabel(label), r, 0); g.addLayout(row, r, 1)
        v.addLayout(g)
        self.hint = StableLabel("", 2, alts=list(E["hints"].values()) + [E["no_mp4"]]); self.hint.setProperty("role", "hint"); v.addWidget(self.hint)
        self.orbit = QCheckBox(E["orbit"]); self.mark = QCheckBox(E["mark"])
        for c in (self.orbit, self.mark): c.setFixedHeight(28); c.setStyleSheet(CHECK_QSS); c.setChecked(True); v.addWidget(c)
        from app2 import general_prefs; self.mark.setChecked(bool(general_prefs.get("timelapse_watermark")))
        turn_row=QHBoxLayout();turn_row.addWidget(QLabel("Camera turns"))
        self.turns=QLineEdit();self.turns.setAccessibleName("Camera turns");self.turns.setFixedHeight(30)
        self._auto_turns=True;self.turns.textEdited.connect(lambda _text:setattr(self,'_auto_turns',False))
        turn_row.addWidget(self.turns,1);v.addLayout(turn_row)
        self.where = StableLabel("", 1); self.where.setProperty("role", "hint"); v.addWidget(self.where)
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(10); self.bar.setRange(0, 1000)
        self.status = StableLabel("", 1); self.status.setFont(ui_font(13, QFont.Weight.DemiBold))
        v.addWidget(self.bar); v.addWidget(self.status)
        v.addStretch(1)
        foot = QHBoxLayout(); self.cancel_b = big_button(E["cancel"], "quiet"); self.ok_b = big_button(E["ok"], "build")
        self.cancel_b.setFixedHeight(44); self.ok_b.setFixedHeight(44); foot.addWidget(self.cancel_b); foot.addWidget(self.ok_b)
        v.addLayout(foot)
        self.cancel_b.clicked.connect(self.cancel); self.ok_b.clicked.connect(self.run)
        self.choice = dict(layout="model", size="720p", length=8, format="mp4")
        self.turns.setText("1")
        self.project = None; self.formats = ("mp4", "gif"); self.proc = None; self.out = None; self._buf = ""
        self._result = None; self._why = ""; self._stopping = False
        self.hide()

    # ---------------- the choices
    def present(self, project, sides, formats=None):
        """Show the sheet for project (sides: how many saved sides it has). formats: what this machine writes
        (timelapse.formats(); MP4 needs ffmpeg)."""
        from app2.scan import timelapse as TL
        E = W.TIMELAPSE_SHEET; self.project = project; self.formats = tuple(formats or TL.formats())
        self.sides.clear(); self.sides.addItem(E["all_sides"], None)
        for n in range(1, sides + 1): self.sides.addItem(E["side"].format(n=n), n)
        self.seg["format"]["mp4"].setEnabled("mp4" in self.formats)
        fmt_ = self.choice["format"] if self.choice["format"] in self.formats else self.formats[0]
        for k in ("layout", "size", "length"): self.pick(k, self.choice[k])
        self.pick("format", fmt_, sized=False)
        self._running(False); self.bar.setValue(0); self.status.setText("")
        self._place(); self.scrim.show(); self.scrim.raise_(); self.show(); self.raise_()

    def pick(self, key, val, sized=True):
        """A segment button: its row shows val. Picking GIF goes to 480p (a smaller file), MP4 back to 720p."""
        E = W.TIMELAPSE_SHEET; self.choice[key] = val
        for x, b in self.seg[key].items(): b.setChecked(x == val)
        if key == "length" and self._auto_turns:self.turns.setText("1")
        if key == "format":
            if sized: self.pick("size", "480p" if val == "gif" else "720p")
            self.hint.setText(E["hints"][val] if "mp4" in self.formats else E["no_mp4"])
            self._where()

    def _where(self):
        from app2.scan import timelapse as TL
        if self.project: self.where.setText(W.TIMELAPSE_SHEET["save_to"].format(name=os.path.basename(TL.default_path(self.project, self.choice["format"]))))

    def args(self):
        """The child's command line for the choices made (the output path is picked now)."""
        from app2.scan import timelapse as TL
        c = self.choice; self.out = TL.default_path(self.project, c["format"])
        a = ["-m", "app2.scan.timelapse", self.project, "-o", self.out, "--format", c["format"], "--size", c["size"],
             "--layout", c["layout"], "--seconds", str(c["length"]), "--turns", self.turns.text() if self.orbit.isChecked() else "0"]
        side = self.sides.currentData()
        if side is not None: a += ["--side", str(side)]
        if not self.mark.isChecked(): a.append("--no-watermark")
        return a

    # ---------------- the job
    @property
    def busy(self): return self.proc is not None

    def run(self):
        if self.busy or not self.project: return
        import math
        try: turns=float(self.turns.text())
        except ValueError: turns=float('nan')
        if not math.isfinite(turns) or turns<0:
            self.status.setText("Enter a nonnegative number of camera turns.");return
        E = W.TIMELAPSE_SHEET; self._result = None; self._why = ""; self._buf = ""; self._stopping = False
        p = self.proc = QProcess(self); p.setWorkingDirectory(REPO)
        env = QProcessEnvironment.systemEnvironment(); env.insert("PYOPENGL_PLATFORM", "egl")
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        p.readyReadStandardOutput.connect(self._read); p.finished.connect(self._finished)
        p.errorOccurred.connect(lambda err: self._finished(-1, None) if err == QProcess.ProcessError.FailedToStart else None)
        self._running(True); self.bar.setValue(0); self.status.setText(E["starting"])
        p.start(sys.executable, self.args())

    def _read(self):
        E = W.TIMELAPSE_SHEET
        if self.proc is not None: self._buf += bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            if line.startswith("PROGRESS "):
                try: share = float(line.split()[1])
                except (IndexError, ValueError): continue
                self.bar.setValue(int(share * 1000)); self.status.setText(E["working"].format(pct=int(share * 100)))
            elif line.startswith("DONE "):
                try: self._result = json.loads(line[5:])
                except ValueError: pass
            elif line.startswith("FAILED "): self._why = line[7:].strip()

    def _finished(self, code, _status=None):
        if self.proc is None: return
        p = self.proc
        if code != -1:
            self._read(); self._buf += "\n"; self._read()          # what is left, the last line too
            if not self._why:
                err = bytes(p.readAllStandardError()).decode(errors="replace").strip().splitlines()
                self._why = err[-1] if err else ""
        self.proc = None; p.deleteLater(); self._running(False)
        if self._result is not None and code == 0: self.close_sheet(); self.done.emit(self._result)
        elif self._stopping or code == 2: self.status.setText(W.TIMELAPSE_SHEET["stopped"]); self.bar.setValue(0); self.stopped.emit()
        else:
            why = self._why or ("it could not start" if code == -1 else "it stopped with code %s" % code)
            for private, label in ((self.project, "the project"), (REPO, "the app folder")):
                if private: why = why.replace(private, label)
            self.status.setText(W.TIMELAPSE_SHEET["failed"].format(why=why[:160])); self.failed.emit(why)

    def stop(self):
        """Stop the child: SIGTERM (it removes its half-made file and exits), killed if it has not gone in 5 s."""
        if self.proc is None: return
        self._stopping = True; p = self.proc; p.terminate()
        QTimer.singleShot(5000, lambda: p.kill() if self.proc is p else None)

    def cancel(self):
        if self.busy: self.stop()
        else: self.close_sheet()

    def _running(self, on):
        E = W.TIMELAPSE_SHEET
        self.cancel_b.setText(E["stop"] if on else E["cancel"]); self.ok_b.setEnabled(not on)
        for w in [self.sides, self.orbit, self.mark, self.turns] + [b for row in self.seg.values() for b in row.values()]:
            w.setEnabled(not on and (w is not self.seg["format"]["mp4"] or "mp4" in self.formats))

    def close_sheet(self):
        if self.busy: return
        was = self.isVisible(); self.hide(); self.scrim.hide()
        if was: self.closed.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape: self.cancel(); return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter): self.run(); return
        super().keyPressEvent(e)

    def _place(self):
        p = self.parentWidget(); w = min(WIDTH, max(340, p.width() - 24)); h = min(max(HEIGHT,self.layout().sizeHint().height()), max(360, p.height() - 24))
        self.setFixedSize(w, h); self.move((p.width() - w) // 2, max(10, (p.height() - h) // 2))
        self.scrim.setGeometry(p.rect())

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.Resize and self.isVisible(): self._place()
        return False
