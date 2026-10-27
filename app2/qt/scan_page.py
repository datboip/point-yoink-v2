# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Scan step's page: this scan session only. Left to right: what the scanner sees now (IR big, depth and colour under
it, the controls, the log, the big Start / Pause button), the model panel (the side being scanned, live, or the picked
side's own model; nothing picked: the color camera), and the project's sides with Build combined model (when it is
done the window goes to the Build step, where the combined model lives). Under all three the
status line; a banner floats over the top of the model only when frames stop, the scanner is gone, a replay ended or
tracking got too slow (floating: the pictures never jump when it comes and goes). With the bed adapter a Bed card
sits on top of the sides; clicking it shows the learned bed big over the model panel (no side picked meanwhile). It only shows what the session
emits and sends commands back; app2.scan.ui_model decides every text and enabled flag, and the confirmations go
through self.confirm (the shell's dialog; tests swap it)."""
from PySide6.QtCore import QTimer, QEvent, Signal, Qt, QProcess, QProcessEnvironment
import os
from PySide6.QtWidgets import QWidget, QComboBox, QToolButton, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QFileDialog, QApplication
from app2.scan.preview import distance_note
from app2.qt.camera_options import CameraPreferences,CameraOptions,apply_frame,color_image,camera_notice
from app2.scan import ui_model
from app2.scan.object_modes import label as object_label
from app2.qt.widgets import (ImageTile, Controls, CameraSwitch, BigButton, DistanceBar, SidesPanel, LogPanel, LogPopover, StatusLine,
                             Banner, fit_overlay_in_parent, button)
from app2.qt.model_panel import ModelPanel
from app2.qt.bed_card import BedCard, BedView

GAP = 12
CAMERA_PARTS = ([("IR", None), ("in range", "kept"), ("platter", "platter")], [("Depth", None)], [("Color", None)])
IR_NO_PLATTER = CAMERA_PARTS[0][:2]                        # the platter's legend only while its tint is drawn
IR_CUT = ("below bed", "ink3")                             # while Cut below bed takes pixels (dimmed in the picture)


class RewindBar(QFrame):
    """While a side with frames is paused: Back 30 / 10 / 5 and Forward, floating at the foot of the model panel
    (floating: the pictures never jump when it comes and goes). back(n) / forward are the page's; ui_model says the
    texts and what is enabled."""
    back = Signal(int)
    forward = Signal()

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("RewindBar")
        self.setStyleSheet("QFrame#RewindBar { background: rgba(10, 12, 15, 215); border: 1px solid #343941; border-radius: 10px; }"
                           " QLabel { color: #c6d4e4; border: 0; }")
        h = QHBoxLayout(self); h.setContentsMargins(12, 6, 8, 6); h.setSpacing(6)
        self.note = QLabel(); h.addWidget(self.note); h.addSpacing(6)
        self.backs = {}
        for k in ui_model.BACK_STEPS:
            b = button("Back %d" % k, "plain", ui_model.REWIND_TIP); b.clicked.connect(lambda _=False, k=k: self.back.emit(k))
            self.backs[k] = b; h.addWidget(b)
        self.fwd = button("Forward", "plain", "Bring back the frames the last Back took."); self.fwd.clicked.connect(self.forward)
        h.addWidget(self.fwd); self.hide()

    def set_buttons(self, b):
        """b: ui_model.buttons(). Returns whether the bar is to be shown."""
        self.note.setText(b["rewind_note"])
        for k, w in self.backs.items(): w.setText(b["back%d" % k][0]); w.setEnabled(b["back%d" % k][2])
        self.fwd.setText(b["forward"][0]); self.fwd.setEnabled(b["forward"][2])
        return b["rewind_visible"]


class ScanPage(QWidget):
    build_done = Signal()           # Build combined model, pressed here, finished and made a model: show the Build step
    guide_changed = Signal(bool)    # the first scan guide shown or hidden: the top bar's Guide button follows

    def __init__(self, bridge, gl=False, parent=None):
        super().__init__(parent)
        self.bridge = bridge; self.session = None; self.state = None; self.btns = None; self.cards = []; self.view = None
        self._last_live = None; self._last_full_live = None
        self._shown_color_token=0
        self.confirm = lambda title, text: True                  # the shell puts its dialog here
        self._pending_range = None; self._fov_set = None; self._size = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        body = QVBoxLayout(); body.setContentsMargins(GAP, GAP, GAP, GAP); body.setSpacing(GAP)
        row = QHBoxLayout(); row.setSpacing(GAP)
        self.left = QWidget(); lv = QVBoxLayout(self.left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(GAP)
        self.ir = ImageTile(CAMERA_PARTS[0]); self.depth = ImageTile(CAMERA_PARTS[1]); self.colour = ImageTile(CAMERA_PARTS[2])
        self.popout_b = QToolButton()
        self.popout_b.setText('↗')
        self.popout_b.setAccessibleName('Pop out cameras')
        self.popout_b.setToolTip('Pop out cameras into a separate window. Click again to return.')
        self.popout_b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.camera_bar = QWidget()  # retained as a hidden owner for old sessions
        self.camera_bar.hide()
        self.maximized_camera = None
        self.camera_prefs=CameraPreferences(self);self.camera_options=CameraOptions(self.camera_prefs,turn=True)
        self.camera_prefs.changed.connect(self._camera_preferences_changed)
        self.camera_window = None
        self.camera_content = QWidget()
        self.camera_grid = QGridLayout(self.camera_content); self.camera_grid.setContentsMargins(0, 0, 0, 0); self.camera_grid.setSpacing(GAP); self.camera_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        from app2.qt.camera_tiles import attach_headers
        attach_headers((self.ir,self.depth,self.colour),self.camera_options,self.maximize_camera,self.popout_b,restore=self.restore_cameras)
        self._arrange_cameras()
        from PySide6.QtGui import QShortcut, QKeySequence
        self.camera_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape),self)
        self.camera_escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.camera_escape.activated.connect(self.restore_cameras)
        self.camera_scroll = QScrollArea(); self.camera_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.camera_scroll.setWidgetResizable(True); self.camera_scroll.setWidget(self.camera_content)
        self.camera_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.camera_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.camera_scroll.verticalScrollBar().setSingleStep(60)
        self.camera_detached = QFrame(); self.camera_detached.setFixedHeight(108)
        self.camera_detached.setStyleSheet('QFrame {background:#20252c;border:1px solid #38434f;border-radius:8px;} QLabel {border:0;color:#c6d4e4;}')
        detached_layout = QVBoxLayout(self.camera_detached); detached_layout.setContentsMargins(14, 12, 14, 12)
        detached_note = QLabel('IR, Depth, and Color are open in the camera window.'); detached_note.setWordWrap(True)
        detached_layout.addWidget(detached_note)
        self.return_cameras_b = QPushButton('Return cameras to Scan'); self.return_cameras_b.clicked.connect(self.return_cameras)
        detached_layout.addWidget(self.return_cameras_b)
        self.camera_detached.hide()
        self.switch = CameraSwitch(); self.switch.hide()
        self.controls = Controls(); self.log = LogPanel(); self.go = BigButton()
        lv.addWidget(self.camera_scroll,1);lv.addWidget(self.camera_options.edge_key);lv.addWidget(self.camera_detached)
        self.controls_scroll=QScrollArea();self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setWidgetResizable(True);self.controls_scroll.setWidget(self.controls)
        self.controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.log.hide()
        self.model = ModelPanel(gl)
        self.model.reserve_header()
        self.camera_match=QPushButton(self.model);self.camera_match.setProperty('kind','primary');self.camera_match.hide()
        self.camera_match.clicked.connect(lambda:self._cmd('switch_camera_match'))
        from app2.qt.build_view import ProgressCard
        self.build_card = ProgressCard(self.model); self.build_card.hide()
        self.build_card.setStyleSheet("QFrame#ProgressCard { background: #1a1d22; border: 1px solid #343941; border-radius: 12px; }")
        self.build_card.stop.connect(self.stop_build)
        self.sides = SidesPanel();self.sides.set_settings(self.controls_scroll)
        self.sides.tabs.tabBarClicked.connect(lambda i:setattr(self,'_open_on_sides',False))   # the user's own tab choice wins
        self.bed_card = BedCard(); self.sides.pin(self.bed_card); self.bed_card.hide()
        from app2.qt.live_strip import LiveStrip
        self.live_strip = LiveStrip(self)
        lv.insertWidget(lv.indexOf(self.camera_options.edge_key), self.live_strip)   # under the camera views, in the left column
        self.sides.action_row.insertWidget(0, self.sides.new, 1)   # + New side shares a row with the Bed button, above Start
        lay = self.sides.layout(); lay.insertWidget(lay.indexOf(self.sides.buildbox), self.go)   # Start/Pause above Build
        from app2.qt import ui_level
        from app2.qt.first_scan_guide import GuideCard, Ring
        self.guide = GuideCard(); self.guide.hidden_by_user.connect(lambda: self._show_guide(False))
        self.sides.side_layout.insertWidget(0, self.guide)   # Advanced: top of Sides. Simple: under the cameras (ui_level)
        self.guide_ring = Ring(self)
        row.addWidget(self.left); row.addWidget(self.model, 1); row.addWidget(self.sides)
        body.addLayout(row, 1)
        outer.addLayout(body, 1)
        self.status = StatusLine(); outer.addWidget(self.status)
        self.banner = Banner(self)                                  # floats over the model panel (place_banner)
        self.rewind_bar = RewindBar(self)                           # floats at the foot of the model panel (place_rewind)
        self.distance_bar = DistanceBar(self); self.distance_bar.hide()   # floats at the foot of the model panel too
        self.bedview = BedView(self)                                # the bed big, over the model panel (place_bed)
        self.bed_shown = False; self._bed_from = None
        self.ask_text = self._ask_text                              # (title, label, default) -> str or None; tests put a stand-in here
        self.ask_int = self._ask_int                                # (title, label, default, lo, hi) -> int or None
        self.model.installEventFilter(self)
        self._learn_text = ""; self._learn_tip = self.sides.learn.toolTip(); self.sides.learn.installEventFilter(self)
        self.popover = LogPopover(self.log, self)
        self.one_pic = False; self.ended = False; self.platter_shown = False; self.cut_shown = False
        self.streamed = False                                       # a live frame came in this session (the scanner is there)
        self._build_asked = False; self._scan_build = False; self._last_prog = None
        self._wire()
        self.controls.layout().activate(); self.full_controls_min = self.controls.minimumSizeHint().width()   # every row shown, before Simple hides some
        self.level = ui_level.saved(); ui_level.apply(self, self.level)   # the switch itself is in the top bar (shell)
        view=getattr(self.model,'view',None)
        if view is not None and hasattr(view,'set_point_scale'):   # the saved Dot size for the 3D dots from the start
            view.set_point_scale({'Small':.75,'Normal':1.0,'Large':1.5}.get(self.camera_prefs.show.get('dot_size'),1.0))
        self._guide_target = None; self._update_guide()

    def _show_guide(self, on):
        from app2.scan import settings
        settings.save(first_scan_guide=bool(on)); self._update_guide()

    def suppress_guide(self, on):
        """Hide the step card while the Getting started page is open (the setting itself is not touched), show it again after."""
        if bool(on) != getattr(self, '_guide_suppressed', False): self._guide_suppressed = bool(on); self._update_guide()

    def _update_guide(self):
        from app2.scan import first_scan, settings
        step = first_scan.step(self.state) if settings.load().get('first_scan_guide', True) and not getattr(self, '_guide_suppressed', False) else None
        shown = (step, self.guide.parentWidget() is self.live_strip)
        if shown != getattr(self, '_guide_shown', None):   # new words wrap to a new height: the column is measured again
            self._guide_shown = shown; self.guide.setVisible(step is not None); self.guide_changed.emit(step is not None)
            if step is not None: self.guide.set_step(step, first_scan.TOTAL)
            self.controls.layout_dirty = True; self._controls_changed()
        targets = dict(learn=self.sides.learn, go=self.go, new=self.sides.new, build=self.sides.buildbox.build)
        self._guide_target = targets.get(step[3]) if step is not None else None
        QTimer.singleShot(0, lambda: self.guide_ring.around(self._guide_target))   # after the layout settles

    def set_level(self, level):
        from app2.qt import ui_level
        from app2.scan import settings
        if level == self.level: return
        self.level = level; settings.save(ui_level=level); ui_level.apply(self, level)

    # ------------------------------------------------------------------ wiring
    def _wire(self):
        b = self.bridge
        b.state.connect(self.on_state); b.status.connect(self.on_status); b.sides.connect(self.on_sides)
        b.live.connect(self.on_live); b.cloud.connect(self.model.on_cloud); b.pose.connect(self.model.on_pose)
        b.model_image.connect(self.model.on_image); b.side_mesh.connect(self.model.on_side_mesh); b.turn.connect(self.on_turn); b.range_moved.connect(self.on_range)
        b.gain_moved.connect(self.on_gain); b.log.connect(self.log.append); b.build.connect(self.on_build)
        self.go.clicked.connect(self.key_space)
        self.rewind_bar.back.connect(self.key_back); self.rewind_bar.forward.connect(self.key_forward)
        self.sides.new.clicked.connect(self.key_new); self.sides.learn.clicked.connect(self.ask_learn)
        self.sides.touch.clicked.connect(lambda: self._cmd("touch_up") if self.sides.touch.isEnabled() else None)
        self.sides.pick_asked.connect(self.pick); self.sides.delete_asked.connect(self.ask_delete)
        self.bed_card.clicked.connect(self.toggle_bed)
        self.bed_card.profile_picked.connect(lambda name: self._cmd("pick_bed", name))
        self.bed_card.learn_new.connect(self.ask_learn_new)
        bb = self.sides.buildbox
        bb.build.clicked.connect(self.key_build); bb.stop.clicked.connect(self.stop_build)
        c = self.controls
        c.capture_mode.currentIndexChanged.connect(lambda i: self._cmd("set_capture_mode", c.capture_mode.itemData(i)))
        c.surface.currentTextChanged.connect(lambda name:self._cmd('set_surface',name))
        c.light_during_scans.toggled.connect(lambda on:self._cmd('set_light_during_scans',on))
        c.live_build.changed.connect(lambda detail,support,share,angle:self._cmd('set_live_build',detail,support,share,angle))
        c.live_build.export.connect(self.export_live_model)
        c.live_build.undo.connect(lambda:self._cmd('undo_live_build'));c.live_build.next_pass.connect(lambda:self._cmd('next_live_pass'))
        c.scan_mode.currentIndexChanged.connect(lambda i: self._cmd("set_scan_mode", c.scan_mode.itemData(i)))
        self._far_process=None
        c.calibration_setup.toggled.connect(lambda _on: self._controls_changed())
        c.choose_far.clicked.connect(self.choose_far_calibration)
        c.capture_tools.overlay_changed.connect(lambda values:self._cmd('set_overlay',values))
        c.capture_tools.overlay_auto.connect(lambda:self._cmd('auto_overlay'))
        c.capture_tools.overlay_reset.connect(lambda:self._cmd('reset_overlay'))
        c.capture_tools.range_test.connect(lambda:self._cmd('range_test'))
        c.capture_tools.range_reset.connect(lambda:self._cmd('reset_range_test'))
        c.capture_tools.range_apply.connect(lambda:self._cmd('apply_range_test'))
        self.distance_bar.defaults_asked.connect(lambda:self._cmd('reset_range_test'))
        c.calibration_setup.toggled.connect(lambda on:c.capture_tools.nudge.setVisible(True))
        c.rebuild_far.clicked.connect(self.rebuild_far_copy)
        c.frame_range.clicked.connect(lambda:self._cmd('use_frame_range'));c.range_undo.clicked.connect(lambda:self._cmd('undo_frame_range'))
        c.range_changed.connect(self._range_moving); c.range.released.connect(lambda v: self._send_range())
        c.gain_changed.connect(lambda g: self._cmd("set_gain", g))
        c.auto_range_toggled.connect(lambda on: self._cmd("set_auto_range", on))
        c.auto_gain_toggled.connect(lambda on: self._cmd("set_auto_gain", on))
        c.save_color_toggled.connect(lambda on:self._cmd("set_save_color",on))
        c.led_toggled.connect(lambda on:self._cmd("set_led",False if getattr(self.state,"led_status",None) in ("pending","unverified") and getattr(self.state,"led_requested",False) else on))
        c.below_bed_toggled.connect(lambda on: self._cmd("set_below_bed", on))
        c.remove_base_toggled.connect(lambda on: self._cmd("set_remove_base", on))
        c.outside_below_changed.connect(lambda mm:self._cmd('set_outside_below',mm))
        c.bed_lift_changed.connect(lambda mm:self._cmd('set_bed_lift',mm))
        self.model.home_asked.connect(lambda: self._cmd("home"))
        self.model.dragged.connect(lambda dx, dy: self._cmd("orbit_drag", dx, dy))
        self.model.wheeled.connect(lambda s: self._cmd("zoom", s))
        self.switch.picked.connect(self._switch_camera)
        self.popout_b.clicked.connect(self.pop_out_cameras)
        self.status.log_btn.clicked.connect(self.toggle_log)
        self._range_timer = QTimer(self); self._range_timer.setSingleShot(True); self._range_timer.setInterval(60)
        self._range_timer.timeout.connect(self._send_range)
        self._clock = QTimer(self); self._clock.setInterval(1000); self._clock.timeout.connect(self.sides.buildbox.tick); self._clock.timeout.connect(self._show_build_card); self._clock.start()

    def _cmd(self, name, *a):
        if self.session is not None: getattr(self.session, name)(*a)

    def attach(self, session):
        """A new session (start, New project, Open project): forget what the old one showed."""
        self.session = session; self.state = None; self.btns = None; self.cards = []; self._last_live = None; self._last_full_live = None
        if hasattr(session,'set_ir_view'): session.set_ir_view(self.camera_prefs.ir)
        if hasattr(session,'set_smooth_color'): session.set_smooth_color(self.camera_prefs.show.get('smooth_color',False))
        self._shown_color_token=0
        self.sides.tabs.setCurrentIndex(1);self._setup_problem_seen=False;self._open_on_sides=True   # no sides yet: start on Settings; a project with sides flips to Sides once they arrive
        self.ended = False; self.streamed = False; self.controls.setEnabled(True); self.sides.buildbox.stop.setEnabled(True)
        self.log.clear(); self.sides.set_sides([]); self.controls.set_auto_msg("")
        self.model.home_seq = 0; self.model.live_img = None
        self.model.on_cloud(dict(P=[], N=None, L=None, covered=None, pivot=None, seq=0))
        self.model.on_image(dict(img=None, kind="live", seq=0)); self.model.cloud_seq = self.model.img_seq = -1
        for t in (self.ir, self.depth, self.colour): t.set_image(None); t.set_ring(None); t.set_note(None)
        if self.camera_window is not None: self.camera_window.reset()
        self.sides.buildbox.set_progress(None); self.banner.show_msg(None); self._rewind_wanted = False; self.rewind_bar.hide()
        self.bed_shown = False; self._bed_from = None; self.bedview.hide(); self.bed_card.hide()
        self._build_asked = self._scan_build = False; self._last_prog = None; self.model.set_context(False, None, 0)
        self._refresh_touch_up()

    def export_live_model(self):
        path,_=QFileDialog.getSaveFileName(self,'Export live model','live-model.ply','PLY model (*.ply)')
        if path:self._cmd('export_live_model',path)

    def choose_far_calibration(self):
        mode=getattr(self.state,'scan_mode','near') or 'near'
        folder=QFileDialog.getExistingDirectory(self,"Choose the calibration folder from a "+object_label(mode)+" scanner export")
        if folder:self._cmd('set_'+mode+'_calibration',folder)

    def rebuild_far_copy(self):
        if self._far_process is not None:
            self._far_process.terminate();return
        state=self.state
        if state is None or not state.picked or not getattr(state,'far_calibration_path',None):return
        parent=QFileDialog.getExistingDirectory(self,"Save a new Large part project copy")
        if not parent:return
        from pathlib import Path
        from app2 import REPO
        import sys,json
        target=Path(parent)/(Path(state.project).name+'_LargePart');number=1
        while target.exists():target=Path(parent)/(Path(state.project).name+'_LargePart ('+str(number)+')');number+=1
        process=self._far_process=QProcess(self);process.setWorkingDirectory(REPO)
        env=QProcessEnvironment.systemEnvironment();env.insert('POINTYOINK_NO_DEVICE','1');env.remove('POINTYOINK_DRIVER')
        process.setProcessEnvironment(env);process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.controls.rebuild_far.setText('Stop copy rebuild');self.controls.far_copy_note.setText('Rebuilding a new copy. The source stays unchanged.')
        self.controls.far_copy_note.show();self.controls.layout_dirty=True;self._controls_changed()
        def finish(code,_status=None):
            text=bytes(process.readAllStandardOutput()).decode(errors='replace')
            self._far_process=None;self.controls.rebuild_far.setText('Rebuild Large part copy')
            if code==0:self.controls.far_copy_note.setText('Large part copy saved: '+target.name+'. Open it to review.')
            else:self.controls.far_copy_note.setText('The copy did not finish. The source was not changed.')
            self.controls.far_copy_note.show();self.controls.layout_dirty=True;self._controls_changed()
            process.deleteLater()
        process.finished.connect(finish)
        process.errorOccurred.connect(lambda error:finish(-1) if error==QProcess.ProcessError.FailedToStart else None)
        QApplication.instance().aboutToQuit.connect(process.terminate)
        process.start(sys.executable,['-m','app2.scan.recalibrate',state.project,'--side',str(state.picked),
                      '--calibration',state.far_calibration_path,'--output',str(target)])

    def set_ended(self, why=None):
        """The session stopped on an error (or never started: why says so): nothing here does anything any more, and
        the page says so (Projects opens the project again or starts a new one)."""
        self.ended = True; self.session = None; self._rewind_wanted = False; self.rewind_bar.hide()
        self._refresh_touch_up()
        self.go.set_action("Scanning stopped", "primary", False)
        for w in (self.sides.new, self.sides.learn, self.sides.buildbox.build, self.sides.buildbox.stop):
            w.setEnabled(False)
        for card in self.sides.cards: card.delete.setEnabled(False)
        self.bed_shown = False; self.bedview.hide(); self.bed_card.clickable = False
        self.bed_card.setCursor(self.cursor()); self.bed_card.setToolTip("")
        self.controls.setEnabled(False)
        self.banner.show_msg(("lost", why or "The scan session stopped on an error. Every side it had is saved.",
                              "Go to Projects to open this project again or start a new one. The log says what went wrong."))
        self.place_banner()
        self.status.set_view(ui_model.StatusView(kind="gone", pill="Scan session stopped", stats=[], hint="", banner=None, auto_msg=""))

    # ------------------------------------------------------------------ session -> page
    def on_state(self, s):
        if self.ended: return
        was = self.state is not None and self.state.building
        self.state = s; b = self.btns = ui_model.buttons(s)
        self._update_guide()
        mode = self.controls.capture_mode
        mode.blockSignals(True); mode.setCurrentIndex(max(0, mode.findData(s.capture_mode or "continuous"))); mode.blockSignals(False)
        mode.setEnabled(not (s.recording or s.busy or s.building or s.learning))
        distance = self.controls.scan_mode
        distance.blockSignals(True); distance.setCurrentIndex(max(0, distance.findData(getattr(s, "scan_mode", "near")))); distance.blockSignals(False)
        distance.setEnabled(mode.isEnabled() and not getattr(s, "picked_has_frames", False))
        distance.setToolTip("Start a new side before changing the object setting." if getattr(s, "picked_has_frames", False) else
                            "Each object choice needs its matching calibration. The camera check compares the stream.")
        source=getattr(s,'calibration_source',None) or {}
        c=self.controls
        old_source_text=c.calibration_info.text()
        c.calibration_info.setText(getattr(s,'calibration_error',None) or getattr(s,'mode_notice',None) or
            ("Large part calibration: scanner export, side " + str(source.get('source_side','')) if source.get('declared_mode')=='far'
             else "Saved calibration; physical mode not verified"))
        c.calibration_setup.setToolTip(c.calibration_info.text()+"\nChoose calibration files from a scanner export only when setting up a camera.")
        c.choose_far.setText('Choose calibration files...')
        far_tools=True
        far_path=getattr(s,'far_calibration_path',None);far_ok=bool(far_path) and os.path.exists(os.path.join(far_path,'Pl.bin'))
        near_ok=getattr(s,'calibration_missing',None) is not True and not (getattr(s,'calibration_error','') or '').startswith('Camera calibration')
        c.setup_status.setText("Small part camera: %s<br>Large part camera: %s" % (
            "<b>set up</b>" if near_ok else "<span style='color:#ff8a80'>not set up yet</span>",
            "<b>set up</b>" if far_ok else "<span style='color:#ff8a80'>not set up yet</span>: scan with Large part on the scanner and import it"))
        missing=getattr(s,'calibration_missing',None)
        needs_setup=bool(getattr(s,'calibration_error',None)) if missing is None else bool(missing)
        dragging=c.range.dragging or c.gain.dragging
        if needs_setup and not getattr(self,'_setup_problem_seen',False) and not dragging:self.sides.tabs.setCurrentIndex(1)
        if s.recording and not getattr(self,'_recording_seen',False) and not dragging:self.sides.tabs.setCurrentIndex(0)   # a side starts: show Sides once; Settings stays one click away
        self._recording_seen=bool(s.recording)
        self._setup_problem_seen=needs_setup
        if needs_setup and not c.calibration_setup.isChecked():c.calibration_setup.setChecked(True)
        c.calibration_setup.setText('Camera setup')
        for tool,on in ((c.choose_far,far_tools),
                        (c.rebuild_far,bool(s.picked_has_frames)),(c.far_copy_note,bool(c.far_copy_note.text()))):
            if tool.isHidden()==on:tool.setVisible(on);c.layout_dirty=True
        c.choose_far.setEnabled(mode.isEnabled())
        c.rebuild_far.setEnabled(mode.isEnabled() and bool(getattr(s,'far_calibration_path',None)) and bool(s.picked_has_frames))
        if getattr(self,'_far_process',None) is not None:c.rebuild_far.setEnabled(True);c.choose_far.setEnabled(False)
        if s.building and not was: self._scan_build = self._build_asked; self._build_asked = False
        if was and not s.building and self._scan_build:
            self._scan_build = False; prog = self._last_prog
            if prog is not None and prog.done: QTimer.singleShot(0, self.build_done.emit)
        self._show_build_card()
        self._model_context()
        if not (s.recording and s.auto_gain): self.controls.set_auto_msg("")      # what Auto gain does, only while it does it
        self._apply_go(); self._build_step()
        self.sides.new.setEnabled(b["new"][2])
        self.sides.learn.setVisible(b["learn_visible"]); self.sides.learn.setEnabled(b["learn"][2])
        self._learn_text = b["learn"][0]; self._fit_learn()
        self._show_bed()
        self.sides.buildbox.set_buttons(b)
        for card in self.sides.cards: card.delete.setEnabled(b["delete"][2])
        self._rewind_wanted = self.rewind_bar.set_buttons(b)
        self.place_rewind()
        self.controls.set_state(s, b)
        self.live_strip.refresh_tips(self)
        matched=(getattr(s,'mode_evidence',None) or {}).get('mode');mismatch=bool(matched and matched!=s.scan_mode)
        self.camera_match.setVisible(mismatch);self.camera_match.setEnabled(not (s.recording or s.busy or s.building))
        if mismatch:
            from app2.scan.object_modes import label
            self.camera_match.setText('Use '+label(matched));self.camera_match.adjustSize()
        self.model.set_camera_notice(self.camera_match if mismatch else None)
        self.controls.capture_tools.set_state(s,self.controls.calibration_setup.isChecked())
        rim=(getattr(s,'framing',None) or {}).get('rim')
        if rim:self.ir.setToolTip('Measured rim: %.2f mm radius, %.2f mm spread (%d frames)' % (rim['radius_mm'],rim['spread_mm'],rim['frames']))
        hint=(getattr(s,'framing',None) or {}).get('hint','') if (getattr(s,'framing',None) or {}).get('state')!='framed' else ''
        if not hasattr(self,'_frame_steady'):
            from app2.qt.steady import Steady
            self._frame_steady=Steady(lambda h:(setattr(self.distance_bar,'frame_hint',h),self.distance_bar.update()),self)
        self._frame_steady.offer(hint,hint)
        r=self.controls.range;self.distance_bar.set_guide(None if s.building else getattr(s,'range_guide',None),(s.near,s.far),(r.lo,r.hi))
        self._controls_changed();self.place_distance()
        self._refresh_touch_up()
        if s.cal is not None and s.cal != self._fov_set: self._fov_set = s.cal; self.model.set_lens(s.cal)

    def _refresh_touch_up(self):
        s = self.state
        built = bool(s and getattr(s, "project", None) and os.path.isfile(os.path.join(s.project, "model.ply")))
        connected = bool(s and not self.ended and self.streamed and s.cam not in ("gone", "replay_end") and getattr(s, "source_kind", "live") == "live")
        busy = bool(s and (s.recording or s.building or s.busy or s.learning))
        ready = built and connected and not busy and not self.ended and self.session is not None
        reason = "Connect the scanner to touch up" if not connected else "Wait for the current work to finish" if not ready else ""
        self.sides.touch.setVisible(built); self.sides.touch.setEnabled(ready)
        self.sides.touch_reason.setText(reason if built else ""); self.sides.touch_reason.setVisible(built and bool(reason))

    def on_status(self, v):
        if self.ended: return
        if v.kind == "gone" and not self.streamed and getattr(self.state, "source_kind", "live") == "live":   # never streamed: not "disconnected"
            v = ui_model.StatusView(**{k: getattr(v, k) for k in v.__slots__}); v.pill = __import__("app2.scan.usb_mode", fromlist=["pill_text"]).pill_text()
        self.status.set_view(v, show_auto=self.controls.compact)
        ring = ui_model.RING.get(v.kind)
        self.ir.set_ring(ring); self.model.set_ring(ring)
        if self.camera_window is not None: self.camera_window.set_ring(ring)
        if v.banner != self.banner.shown: self.banner.show_msg(v.banner); self.place_banner()

    def _model_context(self):
        s = self.state
        if s is None: return
        frames = next((c["frames"] for c in self.cards if c["no"] == s.picked), 0)
        if hasattr(self.model,"set_touch_context"):self.model.set_touch_context(bool(getattr(s,"picked_touch",False)))
        self.model.set_context(s.recording,s.picked,frames)

    def on_sides(self, cards):
        self.cards = cards; self._model_context()
        if cards and getattr(self,'_open_on_sides',False):self._open_on_sides=False;self.sides.tabs.setCurrentIndex(0)
        if self.sides.set_sides(cards): QTimer.singleShot(0, self.sides.show_picked)
        if self.state is not None and not self.ended: self._show_bed()        # a provisional bed names its side
        if self.btns is not None:
            for card in self.sides.cards: card.delete.setEnabled(self.btns["delete"][2])

    def _ir_parts(self):
        if self.camera_prefs.ir!="Left":return [("IR: "+self.camera_prefs.ir.lower(),None)]
        parts = CAMERA_PARTS[0] if self.platter_shown else IR_NO_PLATTER
        return parts + [IR_CUT] if self.cut_shown else parts

    NO_SCANNER_TIP = "Connect the scanner in PC mode to scan. Saved sides can still be built, edited and exported."

    def _apply_go(self):
        """The big button, greyed out until frames come in from a scanner (a replay source needs none)."""
        s = self.state; b = self.btns
        if s is None or b is None: return
        text, kind, enabled = b["go"]
        waiting = b.get("go_action")!="switch_camera_match" and enabled and not s.recording and s.source_kind != "replay" and (not self.streamed or s.cam == "gone")
        self.go.set_action(text, kind, enabled and not waiting)
        self.go.setToolTip((self.NO_SCANNER_TIP if waiting else b.get("hint",""))+'\nShortcut: Space')

    def _color_tile(self, v, kind, prefs, sample=True):
        """The small Color tile repeats the big view while the big view shows the live color camera: it stays empty
        then (each color frame was scaled and marked twice) unless Color is the maximized camera."""
        token=getattr(v,'color_token',None)
        if token is not None and token<self._shown_color_token:return
        if self.model.page == 'live' and self.maximized_camera != 'Color':
            if self.colour.img is not None: self.colour.set_image(None)
            self.colour.set_center('Shown in the big view'); return
        self.colour.set_center(None)
        apply_frame(self.colour, v, kind, prefs, sample=sample)

    def _show_model_color(self,v):
        token=getattr(v,'color_token',None)
        if token is not None:
            if token<self._shown_color_token:return
            self._shown_color_token=token
        self.model.show_live(color_image(v,self.camera_prefs),getattr(v,'colour_guides',None),self.camera_prefs.show)

    def _camera_preferences_changed(self):
        if self.session is not None and hasattr(self.session,'set_ir_view'): self.session.set_ir_view(self.camera_prefs.ir)
        view=getattr(self.model,'view',None)
        if view is not None and hasattr(view,'set_point_scale'):
            view.set_point_scale({'Small':.75,'Normal':1.0,'Large':1.5}.get(self.camera_prefs.show.get('dot_size'),1.0))
        self._cmd('set_color_angle',self.camera_prefs.show.get('color_angle',False))
        self._cmd('set_smooth_color',self.camera_prefs.show.get('smooth_color',False))
        self._cmd('set_stereo_refine',self.camera_prefs.show.get('ir_refine',False))
        if not self.camera_prefs.show.get('color_angle',False):self.camera_options.turn.set_estimate(None)
        # Repaint the latest frame without inventing new camera samples.
        self.model._distance_mode_changed();self.distance_bar.update();self.camera_options.turn.update()
        v=self._last_full_live or self._last_live
        if v is not None:self._paint_cameras(v,sample=False)
        latest=self._last_live
        if latest is not None:
            self._color_tile(latest,'color',self.camera_prefs,sample=False)
            self._show_model_color(latest)
        if self.camera_window is not None and v is not None:
            self.camera_window.set_live(v,self._ir_parts(),sample=False)
            if latest is not None and latest.colour_only:self.camera_window.set_live(latest,self._ir_parts(),sample=False)

    def _paint_cameras(self,v,sample=True):
        self.camera_options.notice.setText(camera_notice(v,self.camera_prefs))
        self.ir.parts=self._ir_parts()
        if self.one_pic:
            self._switch_camera(self.switch.current,sample=sample)
        else:
            apply_frame(self.ir,v,'ir',self.camera_prefs,sample=sample)
        apply_frame(self.depth,v,'depth',self.camera_prefs,sample=sample)
        self._color_tile(v,'color',self.camera_prefs,sample=sample)
        if self.camera_prefs.ir!='Left':
            from app2.scan.stereo_view import select_ir
            image,_=select_ir(v,self.camera_prefs.ir)
            self.camera_options.ir.setToolTip('Right IR was not saved in this side.' if getattr(v,'ir_right',None) is None else
                'Both needs a supported stereo calibration.' if image is None else
                'Both aligns supported pixels using measured depth and nearby depth. Unmatched pixels stay left only.')

    def on_live(self, v):
        from app2.scan import timing; timing.mark('gui_color' if v.colour_only else 'gui_depth')
        self.camera_options.turn.set_estimate(getattr(v,'bed_angle',None) or (getattr(v,'color_angle',None) if self.camera_prefs.show.get('color_angle',False) else None))
        was=self.streamed
        self.streamed=True;self._last_live=v;self.platter_shown=bool(v.platter);self.cut_shown=bool(getattr(v,'cut',0))
        self._refresh_touch_up()
        if not v.colour_only:self._last_full_live=v
        if not was:self._apply_go()
        if v.colour is not None:self._show_model_color(v)
        if self.camera_window is not None and self.camera_window.isVisible():
            self.camera_window.set_live(v,self._ir_parts())
            return
        self.ir.set_note(distance_note(v))
        if v.colour_only:
            self._color_tile(v,'color',self.camera_prefs)
            if self.one_pic and self.switch.current==2:self._switch_camera(2)
            return
        self._paint_cameras(v)

    def pop_out_cameras(self):
        if self.camera_window is not None and self.camera_window.isVisible():
            self.return_cameras(); return
        if self.camera_window is None:
            from app2.qt.camera_popout import CameraPopout
            self.camera_window = CameraPopout(CAMERA_PARTS, self,prefs=self.camera_prefs)
            self.camera_window.closed.connect(self._cameras_returned)
        if self._last_full_live is not None: self.camera_window.set_live(self._last_full_live, self._ir_parts())
        if self._last_live is not None and self._last_live.colour_only: self.camera_window.set_live(self._last_live, self._ir_parts())
        self.camera_window.set_ring(self.ir.ring)
        self.camera_scroll.show();self.camera_detached.setParent(self.camera_scroll.viewport());self.camera_detached.setFixedWidth(max(1,self.camera_scroll.viewport().width()-self.fontMetrics().lineSpacing()));self.camera_detached.move(self.fontMetrics().lineSpacing()//2,self.fontMetrics().lineSpacing()//2);self.camera_detached.show();self.camera_detached.raise_()
        for tile in (self.ir, self.depth, self.colour): tile.set_image(None)
        self.model.show_live(None)
        self.camera_window.show(); self.camera_window.raise_(); self.camera_window.activateWindow()
        self.popout_b.setToolTip('Return cameras to Scan')

    def return_cameras(self):
        if self.camera_window is not None and self.camera_window.isVisible(): self.camera_window.close()

    def _cameras_returned(self):
        self.camera_detached.hide();self.camera_scroll.show();self.switch.setVisible(self.one_pic)
        self.popout_b.setToolTip('Pop out cameras')
        self._camera_preferences_changed()
        if self._size is not None:self.relayout(*self._size)

    def _switch_camera(self,k,sample=True):
        """Focus one uses the same IR camera and color marks as the other views."""
        self.ir.parts=(CAMERA_PARTS[k] if k else self._ir_parts()) if self.one_pic else self._ir_parts()
        v=self._last_full_live or self._last_live
        if v is None:return
        latest=self._last_live
        kind=('ir','depth','color')[k] if self.one_pic else 'ir'
        chosen=latest if kind=='color' and latest is not None else v
        apply_frame(self.ir,chosen,kind,self.camera_prefs,sample=sample)

    def maximize_camera(self, kind):
        self.maximized_camera = kind      # the last camera picked stays on top; picking the top one again keeps it
        self._arrange_cameras()
        if self._size is not None:
            size = self._size; self._size = None; self.relayout(*size)

    def restore_cameras(self):
        self.maximized_camera = None
        self._arrange_cameras()
        if self._size is not None:
            size = self._size; self._size = None; self.relayout(*size)

    def _camera_order(self):
        tiles = dict(IR=self.ir, Depth=self.depth, Color=self.colour)
        chosen = self.maximized_camera or 'IR'
        return [tiles[chosen]] + [t for name,t in tiles.items() if name != chosen]

    def _arrange_cameras(self):
        for tile in (self.ir,self.depth,self.colour):
            self.camera_grid.removeWidget(tile)
            tile.header.max_button.setChecked(tile.header.kind == self.maximized_camera)
        large, left, right = self._camera_order()
        for tile in (self.ir,self.depth,self.colour):tile.header.row.removeWidget(self.popout_b)
        large.header.row.insertWidget(large.header.row.count()-2,self.popout_b)
        self.popout_b.show()
        self.camera_grid.addWidget(large,0,0,1,2)
        self.camera_grid.addWidget(left,1,0)
        self.camera_grid.addWidget(right,1,1)

    def on_turn(self, t):
        self.camera_options.turn.set_data(t.get("coverage"))
        if self.camera_window is not None:self.camera_window.options.turn.set_data(t.get("coverage"))
        self.model.on_turn(t); self.controls.set_auto_msg(t.get("msg") or ""); self._controls_changed()

    def _controls_changed(self):
        """Fill LED's text (one row or two for the chips) or the Auto gain line changed the controls' height: the
        column's sizes are worked out again."""
        if self.controls.layout_dirty:
            if self._size is not None:
                size=self._size;self._size=None;self.relayout(*size)
            else:self.controls.layout_dirty=False

    def on_range(self, r):
        if not self.controls.range.dragging: self.controls.range.set_values(r); self.controls._show_range(*self.controls.range.values)

    def on_gain(self, g):
        if not self.controls.gain.dragging: self.controls.gain.set_values((g,)); self.controls.gain_row.value.setText(str(g))

    def on_build(self, prog):
        self._last_prog = prog; self.sides.buildbox.set_progress(prog); self._build_step(); self._show_build_card()

    def _show_build_card(self):
        prog = self._last_prog
        if not self.state or not self.state.building or (prog is not None and prog.finished):
            stopped = prog is not None and prog.failed == "Build stopped"               # a Stop is not a failure: the card goes
            if prog is not None and prog.finished and prog.failed and not stopped and not self.build_card.holding and self.build_card.isVisible():
                self.build_card.show_failed(prog.failed, prog.elapsed()); self.place_rewind(); return
            self.build_card.hide(); self.place_rewind(); return
        card = self.build_card
        from app2.qt.build_step import run_overview
        overall, doing = run_overview(prog) if prog else ("", "")
        card.show_state("Building model", prog.step if prog else "Starting", "",
                        prog.frac if prog else None, prog.elapsed() if prog else 0,
                        (getattr(prog, "plan", None) or {}).get("cached", 0) if prog else 0, overall=overall, doing=doing, nodes=getattr(prog, "nodes", None))
        from app2.qt.build_view import CARD_W, globe_side
        g = globe_side(self.model.height()); card.setFixedWidth(min(CARD_W, max(240, self.model.width()-24))); card.anim.setFixedSize(g, g)
        card.adjustSize(); self._place_card(); self.place_rewind()

    def _card_top(self):
        """Where the build card starts: under the chip row, and under the banner while one is showing (they used to sit on each other)."""
        top = 52
        if self.banner.isVisible(): top += self.banner.height() + 10
        return top

    def _place_card(self):
        card = self.build_card
        if card.isVisible():
            card.move((self.model.width() - card.width()) // 2, self._card_top()); card.raise_()

    def _build_step(self):
        """Keep the scan action short; the progress card carries the running step."""
        prog = self.sides.buildbox.prog
        if prog is not None and self.state is not None and self.state.building and not self.ended:
            self.go.set_action("Building model...", "primary", False)

    # ------------------------------------------------------------------ page -> session
    def _range_moving(self, near, far):
        self._pending_range = (near, far)
        if not self._range_timer.isActive(): self._range_timer.start()

    def _send_range(self):
        if self._pending_range is not None: self._cmd("set_range", *self._pending_range); self._pending_range = None

    def pick(self, no):
        """A card clicked: picks that side; clicking the picked card again un-picks it (the live camera comes back).
        With the bed shown big: picks that side (the bed goes)."""
        s = self.state
        if self.bed_shown:
            self.bed_shown = False; self._show_bed(); self._cmd("pick_side", no); return
        self._cmd("pick_side", None if (s is not None and s.picked == no) else no)

    # ------------------------------------------------------------------ the Bed card
    def _show_bed(self):
        """The Bed card from the state (hidden without the bed adapter), and the bed big while it is shown. Showing it
        stops when a side gets picked, a side records, Learn bed or a build starts, or the bed is gone."""
        s = self.state
        v = ui_model.bed_card(s, [c["name"] for c in self.cards]) if s is not None else None
        if v is None:
            self.bed_card.hide(); self.bed_shown = False; self.bedview.hide(); return
        if self.bed_shown and (s.bed is None or s.recording or s.learning or s.building
                               or (s.picked is not None and s.picked != self._bed_from)):
            self.bed_shown = False
        if s.picked is None: self._bed_from = None
        self.bed_card.set_view(v, s.bed, self.bed_shown)
        self.bed_card.set_profiles(s.bed_profiles, s.bed_profile, not (s.recording or s.busy or s.building or s.learning))
        self.bed_card.show()
        if s.learning: self.bedview.set_bed(s.learn_map, v["sub"], v["deg"] or 0)     # the map being built, big, while Learn bed runs
        else: self.bedview.set_bed(s.bed, v["sub"])
        show = self.bed_shown or bool(s.learning)
        if show != self.bedview.isVisible():
            self.bedview.setVisible(show)
            if show: self.place_bed()

    def toggle_bed(self):
        """The Bed card clicked: the bed big (any side un-picked); clicked again: the live camera again."""
        s = self.state
        if s is None or self.ended: return
        if self.bed_shown: self.bed_shown = False; self._show_bed(); return
        v = ui_model.bed_card(s, [c["name"] for c in self.cards])
        if not (v and v["clickable"]): return
        self.bed_shown = True; self._bed_from = s.picked
        if s.picked is not None: self._cmd("pick_side", None)
        self._show_bed()

    def place_bed(self):
        if not self.bedview.isVisible(): return
        self.bedview.setGeometry(self.model.geometry()); self.bedview.raise_()
        if self.banner.isVisible(): self.banner.raise_()
        if self.popover.isVisible(): self.popover.raise_()

    def eventFilter(self, obj, e):
        if obj is self.model and e.type() in (QEvent.Type.Resize, QEvent.Type.Move): self.place_bed(); self.place_rewind()
        elif obj is self.sides.learn and e.type() == QEvent.Type.Resize: self._fit_learn()
        return super().eventFilter(obj, e)

    def _fit_learn(self):
        """Learn bed's text, or a short one when the sides panel is narrow ("Bed: 29 dots (learn again)" did not fit
        at 1280 px; the Bed card above says the rest). The whole text is then its tooltip."""
        b = self.sides.learn; new = self.sides.new; full = self._learn_text; s = self.state
        room = new.width() + b.width() - new.minimumSizeHint().width()       # the row, less what + New side needs
        if b.fontMetrics().horizontalAdvance(full) + 28 <= room or s is None or s.learning: text = full
        else: text = "Learn bed" if s.bed_size is None else "Learn again"
        if b.text() != text: b.setText(text)
        b.setToolTip(self._learn_tip if text == full else "%s. %s" % (full, self._learn_tip))

    def key_unpick(self):
        if self.bed_shown: self.toggle_bed(); return            # Esc: the live camera again
        s = self.state
        if s is not None and s.picked is not None and not (s.recording or s.busy or s.building): self._cmd("pick_side", None)

    def ask_delete(self, no=None):
        s = self.state
        if s is None or not (self.btns and self.btns["delete"][2]): return
        no = s.picked if no is None else no
        if no is None: return
        frames = next((c["frames"] for c in self.cards if c["no"] == no), 0)
        from app2.project_preview import backup_kind, BACKUP_WORDS
        proj = getattr(self.state, "project", None); note = BACKUP_WORDS[backup_kind(proj)][1] if proj else ""
        if self.confirm("Delete side", "Delete side %d (%d frames)? This can't be undone." % (no, frames) + ("\n\n" + note if note else "")): self._cmd("delete_side", no)

    def ask_learn(self):
        s = self.state
        if s is None or not s.bed_available: return
        if s.bed_size is not None and not s.learning and \
                not self.confirm("Learn bed", "Learn the bed again? The old one is kept as bed_prev.json."): return
        self._cmd("learn_bed")

    def ask_learn_new(self):
        """ "Learn a new bed..." in the Bed card's list: its name, then Learn bed (empty table, one turn) for it."""
        s = self.state
        if s is None or not s.bed_available or s.learning or s.recording or s.busy or s.building: return
        names = tuple(s.bed_profiles or ())
        k = len(names) + 1; default = "Bed %d" % k
        while default in names: k += 1; default = "Bed %d" % k
        name = self.ask_text("Learn a new bed", "Name for the new bed:", default)
        name = (name or "").strip()
        if not name: return
        if name in names and not self.confirm("Learn a new bed", "%s is already a bed. Learn it again?" % name): return
        want = self.ask_int("Learn a new bed", "How many dots are on the bed? Count them once; 0 if you do not know.\nThe learning line then reads 18 of 32 dots, and the result says whether it got them all.", 0, 0, 500)
        if want is None: return
        self._cmd("learn_bed", name, int(want) or None)

    def _ask_int(self, title, label, default, lo, hi):
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getInt(self, title, label, default, lo, hi)
        return value if ok else None

    def _ask_text(self, title, label, default):
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, title, label, text=default)
        return text if ok else None

    def stop_build(self): self._cmd("cancel_build")

    # ------------------------------------------------------------------ keys (the shell's shortcuts call these)
    def key_space(self):
        if self.state is not None and self.state.learning: self._cmd("learn_bed"); return        # the button reads Cancel learning
        if self.btns and self.btns["go"][2]: self._cmd(self.btns.get('go_action','toggle_scan'))


    def key_back(self, n):
        """Back 5 / 10 / 30 (the bar under the model): take back the paused side's newest frames."""
        if self.btns and self.btns.get("back%d" % n, ("", "", False))[2]: self._cmd("rewind", n)

    def key_forward(self):
        if self.btns and self.btns["forward"][2]: self._cmd("forward")

    def key_new(self):
        if self.btns and self.btns["new"][2]: self._cmd("new_side")

    def key_step_side(self, d):
        """Up / Down: pick the side above or below the picked one."""
        s = self.state
        if s is None or not s.n_sides or s.recording or s.busy or s.building: return
        no = (1 if d > 0 else s.n_sides) if s.picked is None else min(s.n_sides, max(1, s.picked + d))
        if no != s.picked: self._cmd("pick_side", no)

    def key_home(self): self.model.home()

    def key_auto_range(self):
        if self.state is not None and not self.state.busy: self._cmd("set_auto_range", not self.state.auto_range)

    def key_auto_gain(self):
        if self.state is not None and not self.state.busy: self._cmd("set_auto_gain", not self.state.auto_gain)

    def key_led(self):
        if self.state is not None and self.state.has_led and not self.state.busy: self._cmd("set_led",not (getattr(self.state,"led_requested",False) if getattr(self.state,"led_status",None) in ("pending","unverified") else self.state.led))

    def key_gain_down(self): self._key_gain(-self.controls.gain.step)
    def key_gain_up(self): self._key_gain(self.controls.gain.step)
    def _key_gain(self, delta):
        if self.state is not None and not self.state.busy:
            self._cmd('set_gain', max(self.controls.gain.lo, min(self.controls.gain.hi, self.state.gain + delta)))
    def key_view_color(self):
        choices = self.model.distance_mode
        for offset in range(1, choices.count() + 1):
            index = (choices.currentIndex() + offset) % choices.count()
            if choices.buttons[index].isEnabled(): choices.setCurrentIndex(index); break
    def key_ir_camera(self):
        names = ('Left', 'Right', 'Both')
        self.camera_prefs.set_ir(names[(names.index(self.camera_prefs.ir) + 1) % len(names)])

    def key_build(self):
        """Build combined model (the button, Ctrl+B): one model from every side; the Build step opens on it when done."""
        if self.btns and self.btns["build"][2]:
            self._build_asked = True
            request=getattr(self,'build_request',None)
            if request is not None:request()
            else:self._cmd("build")

    def toggle_log(self):
        if self.popover.isVisible(): self.popover.close_popover(); return
        if not self.popover._user_placed:
            w = min(self.width() - 2 * GAP, self.left.width() + self.model.width() // 2)
            h = max(200, int(self.height() * 0.5))
            self.popover.setGeometry(GAP, self.height() - 36 - h - 6, w, h)
        fit_overlay_in_parent(self.popover); self.popover.open_popover()

    # ------------------------------------------------------------------ sizes
    def relayout(self, page_w, page_h):
        """Size Scan to its actual page area inside the shared shell.

        Three square cameras fit as one over two; short pages keep a larger focus view.
        """
        page_w, page_h = max(1, page_w), max(1, page_h)
        if self._size == (page_w, page_h) and not self.controls.layout_dirty: return
        resized=getattr(self,"_layout_extent",None)!=(page_w,page_h);self._layout_extent=(page_w,page_h)
        self._size = (page_w, page_h)
        self.controls.set_compact(page_h < 680); self.live_strip.set_compact(page_h < 680)
        sides_w = 352 if page_w >= 1600 else (300 if page_w >= 1280 else 248)
        camera_min=2*(self.fontMetrics().horizontalAdvance("I L R B 360° 30")+self.ir.header.max_button.sizeHint().width()+self.fontMetrics().lineSpacing())+GAP
        col = max(camera_min,page_w//3)
        one_pic=False
        ratio=3/4;widest=col
        # A narrower column wraps the strip taller, which leaves less room for the cameras: settle both together.
        for _ in range(4):
            self.live_strip.fit_width(col);chrome=self.status.height()+self.live_strip.height()+self.camera_options.edge_key.sizeHint().height()+5*GAP+sum(t.header.height() for t in self._camera_order()[:2])
            fitting_col=int((max(0,page_h-chrome)-GAP*(1-ratio/2))/(ratio*1.5))
            settled=min(widest,max(camera_min,fitting_col))
            if settled==col:break
            col=settled
        side_m=self.sides.layout().contentsMargins();control_m=self.controls.layout().contentsMargins()
        # Resolve the setup button rows before their minimum width sets the
        # sidebar. Otherwise its first camera click shrinks the sidebar.
        self.controls.fit_width(sides_w-side_m.left()-side_m.right()-self.controls_scroll.verticalScrollBar().sizeHint().width())
        self.controls.layout().activate();self.controls.updateGeometry()
        framing_tools=(self.controls.frame_range,self.controls.range_undo)
        framing_min=sum(max(tool.sizeHint().width(),tool.minimumWidth()) for tool in framing_tools)+self.controls.framing_row.layout().spacing()*(len(framing_tools)-1)+control_m.left()+control_m.right()
        # Advanced's setup rows count while Simple hides them, so switching never resizes the column (or the 3D view)
        sides_w=max(sides_w,max(framing_min,getattr(self,'full_controls_min',0),self.controls.minimumSizeHint().width())+self.controls_scroll.verticalScrollBar().sizeHint().width()+side_m.left()+side_m.right())
        control_w=sides_w-side_m.left()-side_m.right()-self.controls_scroll.verticalScrollBar().sizeHint().width()
        self.controls.fit_width(control_w);self.controls.setMinimumHeight(self.controls.height_for(control_w))
        self.controls.layout_dirty=False
        ir_h=int(col*ratio);small_w=(col-GAP)//2
        depth_h=colour_h=int(small_w*ratio)
        natural=ir_h if one_pic else ir_h+GAP+depth_h
        camera_room=min(natural,max(ir_h,page_h-chrome))
        self.controls_scroll.setMinimumHeight(0)
        self.left.setFixedWidth(col); self.live_strip.fit_width(col)
        if one_pic!=self.one_pic:self.one_pic=one_pic;self._switch_camera(self.switch.current if one_pic else 0)
        self.camera_bar.hide();self.switch.hide()
        self.depth.setVisible(not one_pic);self.colour.setVisible(not one_pic)
        large,small_left,small_right=self._camera_order()
        large_h=ir_h+large.header.sizeHint().height()
        small_h=depth_h+max(small_left.header.sizeHint().height(),small_right.header.sizeHint().height())
        for tile,width,height in ((large,col,large_h),(small_left,small_w,small_h),(small_right,col-GAP-small_w,small_h)):
            tile.setFixedSize(width,height)
            tile.header.adapt(width)
        natural=large_h+GAP+small_h
        self.camera_content.setMinimumHeight(natural)
        self.camera_scroll.setMinimumHeight(0)
        self.log.hide()
        self.sides.setFixedWidth(sides_w); self.go.setFixedHeight(self.fontMetrics().lineSpacing()*3)
        self.sides.fit(sides_w, page_h)
        QTimer.singleShot(0, self.place_banner)
        if getattr(self, '_guide_target', None) is not None: QTimer.singleShot(0, lambda: self.guide_ring.around(self._guide_target))
        if resized:QTimer.singleShot(0,self.sides.show_picked)

    def place_banner(self):
        """The banner over the top of the model panel, below its title chip, as wide as the text needs (at most the
        panel less its margins)."""
        if not self.banner.isVisible(): self._place_card(); return           # no banner any more: the build card goes back up
        m = self.model.geometry(); w = min(m.width() - 24, 760)
        h = self.banner.heightForWidth(w) if self.banner.hasHeightForWidth() else self.banner.sizeHint().height()
        self.banner.setGeometry(m.x() + (m.width() - w) // 2, m.y() + 52, w, max(h, self.banner.sizeHint().height()))
        self.banner.raise_(); self._place_card()                              # and the build card moves under it

    def place_rewind(self):
        """The rewind bar centred at the foot of the model panel; lifted over the coverage key (bottom left, about 280 px
        wide) and the platter dial (bottom right) when it would reach them."""
        self.place_distance()
        show = bool(getattr(self, "_rewind_wanted", False)) and not self.build_card.isVisible()      # the build card has the floor
        if show != self.rewind_bar.isVisible(): self.rewind_bar.setVisible(show)
        if not show: return
        m = self.model.geometry(); self.rewind_bar.adjustSize()
        w = min(self.rewind_bar.sizeHint().width(), max(120, m.width() - 24)); h = self.rewind_bar.sizeHint().height()
        lift = 12 if m.width() - w >= 2 * 300 else 112
        self.rewind_bar.setGeometry(m.x() + (m.width() - w) // 2, m.y() + m.height() - h - lift, w, h)
        self.rewind_bar.raise_()

    def place_distance(self):
        """Reserve a real footer for image and GL viewports. Rewind has priority."""
        bar=self.distance_bar;m=self.model.geometry();h=bar.sizeHint().height()
        visible=not self.rewind_bar.isVisible() and bar.guide is not None and m.height()>=h+self.model.image.minimumHeight()+2*GAP
        if not visible:
            bar.hide();self.model.set_footer(0);return
        self.model.set_footer(h+2*GAP)
        bar.setGeometry(m.x()+GAP,m.bottom()-h-GAP,m.width()-2*GAP,h);bar.show();bar.raise_()
        toast=getattr(self.window(),'toast',None)
        if toast is not None and toast.isVisible():toast.place()   # a toast shown before the bar must move above it

    def resizeEvent(self, e):
        super().resizeEvent(e); self.place_banner(); self.place_rewind()
        self.place_distance()
        if self.popover.isVisible(): fit_overlay_in_parent(self.popover)
