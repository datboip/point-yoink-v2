# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 2.0 window: six steps (Scan, Build, Review, Clean, Measure, Export) down the left rail, one page each, a header
with the project and where the frames come from. Scan and Build are live; the other four are placeholders.

The window owns the scan session's lifecycle: it makes one (make_session), hands it to the pages, and on New project,
Open project, Finish or the window's close button asks it to close. The session saves every side on its own thread;
the window waits for its "closed" signal and never blocks (the first close is ignored until then). New project and
Open project start the next session in the same process; a Finish, Ctrl+Q or kill that comes while one of them is
saving closes the window once the save is done instead. A session that stops on an error leaves the window open with
the Scan page greyed out, until a project is opened or started again from Projects.

No scanner is needed to open a saved project: the session runs without frames and the Build step builds, edits and
exports it; only scanning a side needs the scanner (and a calibration)."""
import os
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import Qt, QUrl, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QIcon, QDesktopServices
from app2 import REPO, library_dir
from app2.version import APP_TITLE
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QStackedLayout, QMessageBox, QFileDialog, QApplication, QLineEdit, QAbstractSpinBox
from app2.scan import fmt
from app2.imported_projects import RAW_ONLY_MESSAGE, imported_project
from scanner.adopt_device_scan import adopt_project, issue_summary, needs_preparation
from app2.qt.bridge import SessionBridge
from app2.qt.widgets import Rail, Header, Scrim, KeysSheet, Toast, STEPS
from app2.qt.scan_page import ScanPage
from app2.qt.build_step import BuildStepPage, BuildKeysSheet
from app2.qt.placeholders import PlaceholderPage
from app2.qt.workspace_home import WorkspaceHome

LIVE_STEPS = (0, 1, 2)                     # Scan, Build, Edit: the steps with a page (Export is an action, not a page)
MSG = dict(busy_open="Wait for the current save or build to finish, then open the project.",
           busy_new="Wait for the current save or build to finish, then start a new project.",
           not_project="{name} is not a scan project (it has no data folder).",
           import_running="A transfer is running. Stop it or let it finish before leaving Import.",
           already_open="{name} is already open.")
SCAN_KEYS = (("Space", "key_space"), ("N", "key_new"), ("Up", lambda p: p.key_step_side(-1)), ("Down", lambda p: p.key_step_side(1)),
             ("Del", "ask_delete"), ("H", "key_home"), ("R", "key_auto_range"), ("G", "key_auto_gain"), ("L", "key_led"),
             ("[", "key_gain_down"), ("]", "key_gain_up"), ("V", "key_view_color"), ("I", "key_ir_camera"))


SEP = (None, None)                                              # a separator in the menu tables
ALWAYS_ON = ("close", "prefs", "guide", "keys", "report", "about", "fullscreen")   # work on every page, the Import page too

class MainWindow(QMainWindow):
    _prepare_done = Signal(object)
    uncaught = Signal(str)

    def __init__(self, make_session, gl=False, projects_dir=None, defer_new_project=False):
        """make_session(bridge, **over) -> a new, not yet started ScanSession emitting to bridge; over: project=...,
        last=... for New project / Open project."""
        super().__init__(); self.setWindowTitle(APP_TITLE); self.setWindowIcon(QIcon(os.path.join(REPO, "icon.png"))); self.make_session = make_session
        self.projects_dir = projects_dir; self.session = None; self.state = None
        self.imported_project = None
        self._prepare_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepare-imported")
        self._prepare_future = None; self._prepare_done.connect(self._finish_prepare_imported)
        self._closing = False; self._really = False; self._crashed = False
        self._close_after = False                                  # closing was asked while New / Open project was saving
        self._streaming = False                                    # a live frame came in from this session's scanner
        self._after_import = None                                  # where to go once Import's clean-up (Done) is over
        self.confirm = self._ask                                   # (title, text) -> bool; tests put a stand-in here
        self._project_name_path = None; self._project_name = ""
        self.project_picker = None
        self.pick_folder = self._pick_folder                       # () -> a folder or None
        self.bridge = SessionBridge()
        shell = QWidget(); shell.setObjectName("Shell"); self.setCentralWidget(shell)
        shell_layout = QVBoxLayout(shell); shell_layout.setContentsMargins(0, 0, 0, 0); shell_layout.setSpacing(0)
        h = QHBoxLayout(); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0); shell_layout.addLayout(h, 1)
        self.rail = Rail(ready=(1, 2, 3)); h.addWidget(self.rail)
        right = QVBoxLayout(); right.setContentsMargins(0, 0, 0, 0); right.setSpacing(0)
        self.header = Header(); right.addWidget(self.header)
        self.stack = QStackedWidget()
        from app2.qt.live_rates import LiveRates
        self.stream_rates = LiveRates()
        self.scan = ScanPage(self.bridge, gl); self.build = BuildStepPage(self.bridge, gl)
        self.scan.build_request=self._scan_build_checked
        self._orientation_project = None
        bar = self.build.vp.rebuild; self.stack.currentChanged.connect(self._sync_rebuild_pill); bar.changed.connect(self._sync_rebuild_pill); self.header.pill_btn.clicked.connect(bar.clicked.emit)
        self._view_cubes = [view.nav_cube for view in (self.scan.model.view, self.build.vp.main.view, self.build.vp.before.view) if view is not None] if gl else []
        for cube in self._view_cubes: cube.basis_changed.connect(self._save_view_orientation)
        self.scan.confirm = lambda title, text: self.confirm(title, text)
        from app2.qt.edit_page import EditPage
        from app2.qt.imported_models import ImportedModelsPage
        self.build_slot = QWidget(); self.build_layout = QStackedLayout(self.build_slot)
        self.build_layout.setContentsMargins(0, 0, 0, 0); self.build_layout.addWidget(self.build)
        self.imported_models = ImportedModelsPage(gl)
        self.build_layout.addWidget(self.imported_models)
        self.imported_models.message.connect(self.say); self.imported_models.changed.connect(self._refresh_export)
        self.edit = EditPage()
        self.stack.addWidget(self.scan); self.stack.addWidget(self.build_slot); self.stack.addWidget(self.edit)
        self.placeholders = [PlaceholderPage(s) for s in STEPS[3:]]
        for p in self.placeholders: self.stack.addWidget(p)
        self.workspace = WorkspaceHome(self.projects_dir)
        cards = (self.build.vp.card, self.scan.build_card)      # one build, two cards (Build page, Scan step): minimized together, into the top bar's strip
        self._docked_from = cards[0]
        def docked(on, me):
            if on: self._docked_from = me
            for o in cards:
                if o is not me: o.set_docked(on)
            self.workspace.mini.setVisible(on)
        for c in cards:
            c.status.connect(self.workspace.mini.set_status); c.docked_changed.connect(lambda on, me=c: docked(on, me))
        self.workspace.mini.restore.connect(lambda: [c.undock() for c in cards])
        self.workspace.mini.stop.connect(lambda: self._docked_from.stop.emit())
        self.workspace.new_requested.connect(self._workspace_new)
        self.workspace.open_requested.connect(self._workspace_open)
        self.workspace.continue_requested.connect(self._workspace_continue)
        self.workspace.project_requested.connect(self._workspace_project)
        self.workspace.prepare_requested.connect(self._prepare_imported)
        self.workspace.folder_requested.connect(self._workspace_folder)
        self.workspace.mode_requested.connect(self._workspace_mode)
        self.workspace.import_requested.connect(lambda: self._menu_action('projects_import'))
        self.workspace.layout().removeWidget(self.workspace.ribbon_scroll)
        shell_layout.insertWidget(0, self.workspace.ribbon_scroll)
        self.stack.addWidget(self.workspace)
        right.addWidget(self.stack, 1)
        right.addWidget(self.stream_rates)
        self.scan.layout().removeWidget(self.scan.status)
        right.addWidget(self.scan.status)
        h.addLayout(right, 1)
        self.toast = Toast(self)
        self.scrim = Scrim(self); self.scrim.hide(); self.keys_scan = KeysSheet(self); self.keys_build = BuildKeysSheet(self)
        self.keys = self.keys_scan                                # the ? sheet of the step shown
        self.scrim.clicked.connect(self.hide_sheets)
        self.rail.step_asked.connect(self.show_step); self.rail.keys.clicked.connect(self.show_keys)
        self.header.new.clicked.connect(self.ask_new_project); self.header.open.clicked.connect(self.ask_open_project)
        self.header.finish.clicked.connect(self.close)
        ws = self.workspace; ws.level.setCurrentText(self.scan.level)
        ws.level.currentTextChanged.connect(self.scan.set_level)
        ws.guide_button.toggled.connect(lambda on: ((self.guide_card.dismiss() if on and self.guide_card.isVisible() else None), (self.scan._show_guide(on) if on != self.scan.guide.isVisible() else None)))
        self.scan.guide_changed.connect(lambda on: (ws.guide_button.blockSignals(True), ws.guide_button.setChecked(on), ws.guide_button.blockSignals(False)))
        ws.guide_button.setChecked(self.scan.guide.isVisible())
        self.header.project_named.connect(self.rename_active_project)
        self.bridge.closed.connect(self.on_closed); self.bridge.error.connect(self.on_error); self.bridge.state.connect(self.on_state)
        self.bridge.status.connect(self.on_status); self.bridge.live.connect(self._on_live)
        self.scan.build_done.connect(self.open_built)
        self.build.message.connect(self._build_message); self.build.status_changed.connect(self._refresh_export)
        self.build.edit_requested.connect(lambda: self.show_step(2))          # Clean up in the stage list
        self.build.timelapse_requested.connect(lambda: self._menu_action("timelapse"))
        from app2.qt.light_prompt import LightPrompt
        self.light_prompt = LightPrompt(self)
        self.light_prompt.extend.connect(lambda: self.session.extend_led() if self.session else None)
        self.light_prompt.turn_off.connect(lambda: self.session.set_led(False) if self.session else None)
        from app2.qt.telemetry_panel import TelemetryPanel
        self.motion_panel = TelemetryPanel(self); self.motion_panel.hide()
        self.scan.model.motion_provider=self.motion_panel.client.snapshot
        from app2.qt.motion_link import MotionLink
        self.motion_link = MotionLink(self.motion_panel, lambda: self._streaming and not self._crashed, self)
        self.motion_link.need_address.connect(self.show_motion)
        self.header.motion.hide()                                   # Motion lives in the scanner panel now
        from app2.qt.app_menu import AboutCard, GuideCard
        self.about_card = AboutCard(self)
        self.guide_card = GuideCard(self)
        self.guide_card.dismissed.connect(lambda: self._focus_changed(None, QApplication.focusWidget()))
        self.guide_card.dismissed.connect(lambda: self.scan.suppress_guide(False))
        self.about_card.dismissed.connect(lambda: self._focus_changed(None, QApplication.focusWidget()))
        from app2.qt.device_captures import DeviceCaptures
        self.captures = DeviceCaptures(self); self._captures_paused=False; self._captures_quit=False
        self.captures.connect_requested.connect(self.connect_captures)
        self.captures.dismissed.connect(self.captures_done)
        from app2.qt.device_projects import DeviceProjects
        self.device_projects = DeviceProjects(self.projects_dir)
        self.stack.addWidget(self.device_projects)
        self._projects_paused = False; self._projects_quit = False
        self.device_projects.connect_requested.connect(self.connect_projects)
        self.device_projects.dismissed.connect(self.projects_done)
        self.device_projects.imported.connect(lambda _path: self.workspace.refresh_projects())
        self.device_projects.open_requested.connect(lambda path: self._import_holds(lambda: self._workspace_project(path)) or self._workspace_project(path))
        self._setup_menus()
        for button in (self.header.new, self.header.open, self.header.finish, self.header.motion): button.hide()
        self._shortcuts()
        self.setMinimumSize(1024, 640); self.resize(1688, 1044)
        from app2.qt.startup import StartupOverlay
        self.startup = StartupOverlay(self)
        self.startup.dismissed.connect(lambda: self._focus_changed(None, QApplication.focusWidget()))
        self.stack.setCurrentWidget(self.workspace)
        self.workspace.set_current_step(None)
        self.rail.hide(); self.header.hide()
        self.scan.status.hide()
        self.workspace.refresh_projects()
        from app2.memory_watch import Watch
        self.memory_watch = Watch()                     # the 36 GB runaway of 2026-10-22 said nothing; now the window does
        self._error_said = 0.0; self.uncaught.connect(self._uncaught_toast)
        from app2.diagnostics import log as _log; _log.LISTENERS.append(lambda kind, exc: self.uncaught.emit(str(exc)))
        self._memory_timer = QTimer(self); self._memory_timer.setInterval(5000); self._memory_timer.timeout.connect(self._check_memory); self._memory_timer.start()
        if not defer_new_project:
            self.start_session()
        else:
            self._splashed = True                       # no session yet: the first project you open must not flash the splash
            self.startup.ready()                        # and the startup screen has nothing to wait for: it was only hidden when a session reported in

    # ------------------------------------------------------------------ keys
    def _shortcuts(self):
        self._keys = []
        def add(seq, fn):
            s = QShortcut(QKeySequence(seq), self); s.setContext(Qt.ShortcutContext.WindowShortcut); s.activated.connect(lambda: fn() if self.project_picker is None or seq == "Esc" else None)
            s._allow_edit = seq == "Esc"
            self._keys.append(s)
        for seq, what in SCAN_KEYS:
            add(seq, lambda s=seq, w=what: self._page_key(s, w))
        add("Ctrl+B", self.key_build)
        for seq, action in (("Ctrl+N", "new"), ("Ctrl+O", "open"), ("F2", "rename"), ("Ctrl+E", "export"), ("Ctrl+Shift+E", "export_as"), ("F1", "guide")):
            add(seq, lambda key=action:self._menu_action(key))
        add("Ctrl+0", self.show_workspace)
        for i in LIVE_STEPS: add("Ctrl+%d" % (i + 1), lambda k=i: self.show_step(k))
        add("F11", self.toggle_full_screen); add("Ctrl+Q", self.close); add("?", self.show_keys); add("Shift+?", self.show_keys)
        add("Esc", self.hide_sheets)
        QApplication.instance().focusChanged.connect(self._focus_changed)

    def _focus_changed(self, old, current):
        editing = (self.project_picker is not None or isinstance(current, (QLineEdit, QAbstractSpinBox))
                   or (hasattr(self, "startup") and not self.startup.isHidden())
                   or (hasattr(self, "about_card") and self.about_card.isVisible())
                   or (hasattr(self, "guide_card") and self.guide_card.isVisible())
                   or (hasattr(self, "captures") and self.captures.isVisible())
                   or (hasattr(self, "device_projects") and self.device_projects.isVisible()))
        for shortcut in self._keys: shortcut.setEnabled(not editing or shortcut._allow_edit)

    def _scan_key(self, what):
        if self.keys.isVisible() or self.stack.currentWidget() is not self.scan: return
        if callable(what): what(self.scan)
        else: getattr(self.scan, what)()

    def _page_key(self, seq, what):
        """A window-wide key: the Scan step's action on Scan, else the shown page's own page_key (the Build step)."""
        if self.keys.isVisible(): return
        page = self.stack.currentWidget()
        if page is self.scan: self._scan_key(what)
        elif page is self.build_slot and self.imported_project:
            if seq == "H": self.imported_models.home()
        elif page in (self.build_slot, self.edit):
            if page is self.edit and seq in ("Up", "Down"): return
            self.build.page_key(seq)
        elif hasattr(page, "page_key"): page.page_key(seq)

    def key_build(self):
        if self.imported_project: return
        if self.stack.currentWidget() in (self.build_slot, self.edit): self.build.start_build()
        else: self.scan.key_build()

    def _setup_menus(self):
        bar = self.menuBar(); bar.setNativeMenuBar(False); bar.setFixedHeight(28)
        bar.setStyleSheet("QMenuBar {background:#20252c;color:#e6edf6;} QMenuBar::item:selected {background:#344254;} QMenu {background:#20252c;color:#e6edf6;} QMenu::item:selected {background:#344254;} QMenu::item:disabled {color:#687583;}")
        groups = {
            "File": (("workspace","Workspace"),("new","New scan"),("open","Open project"),("rename","Rename project"),("folder","Open project folder"),SEP,
                     ("projects_import","Import scanner projects"),("captures","Import screenshots and screen recordings"),SEP,
                     ("export","Export STL"),("export_as","Export..."),("timelapse","Export scan timelapse..."),SEP,
                     ("prefs","Preferences..."),("cleanup","Clean up temporary files"),SEP,("close","Quit")),
            "Edit": (("undo","Undo"),("redo","Redo"),SEP,("edit","Edit model"),("delete_selection","Delete selected")),
            "View": (("compare","Compare"),("home","Home view"),SEP,("perspective","Perspective"),("orthographic","Orthographic"),SEP,("front","Set this view as Front"),("top","Set this view as Top"),SEP,("motion","Scanner motion"),("fullscreen","Full screen")),
            "Help": (("guide","Getting started"),("scan_steps","First scan steps on the Scan page"),("keys","Keyboard shortcuts"),SEP,("report","Report a problem..."),("about","About PointYoink")),
        }
        self.menu_actions = {}; self.menus = {}
        for name, entries in groups.items():
            menu = bar.addMenu(name); self.menus[name] = menu; menu.aboutToShow.connect(self._refresh_menus)
            for key, text in entries:
                if key is None: menu.addSeparator(); continue
                action = menu.addAction(text); action.triggered.connect(lambda checked=False,k=key:self._menu_action(k))
                self.menu_actions[key] = action
        for key in ("perspective", "orthographic", "scan_steps"): self.menu_actions[key].setCheckable(True)
        shortcuts = dict(workspace="Ctrl+0", new="Ctrl+N", open="Ctrl+O", rename="F2", export="Ctrl+E", export_as="Ctrl+Shift+E", close="Ctrl+Q",
                         undo="Ctrl+Z", redo="Ctrl+Y", edit="Ctrl+3", delete_selection="Del",
                         compare="C", home="H", fullscreen="F11", guide="F1", keys="?")
        for key, sequence in shortcuts.items():
            action = self.menu_actions[key]; action.setText(action.text()+"\t"+sequence)
        steps = self.menus["View"].addMenu("Go to step")
        for index in LIVE_STEPS:
            action = steps.addAction(STEPS[index]+"\tCtrl+"+str(index+1))
            action.triggered.connect(lambda checked=False, i=index:self.show_step(i))

    def _menu_states(self):
        s = self.state
        busy = bool(self._closing or self.build.busy or self.build.busy_call or self.imported_models.export_future is not None
                    or self._prepare_future is not None
                    or (s is not None and (s.busy or s.building or s.recording or s.learning)))
        imported = bool(self.imported_project)
        project = self.imported_project or getattr(s, "project", None)
        editing = self.stack.currentWidget() in (self.build_slot, self.edit)
        can_export = self.imported_models.can_export() if imported else self.build.can_export()
        return dict(report=True, captures=not busy, projects_import=not busy, cleanup=not busy, prefs=not busy, new=not self._closing and not self.build.busy and not self.build.busy_call and
                    (s is None or not (s.busy or s.building or s.learning)),
                    open=not busy and (imported or self.header.open.isEnabled()),
                    folder=bool(project and os.path.isdir(project)),
                    export=not busy and can_export, close=not busy,
                    export_as=not busy and can_export,
                    timelapse=not imported and not busy and bool(project and os.path.isdir(project)),
                    undo=not imported and not busy and self.build.undo_b.isEnabled(),
                    redo=not imported and not busy and self.build.redo_b.isEnabled(), edit=not imported and not busy,
                    delete_selection=not imported and not busy and self.stack.currentWidget() is self.edit and self.build.vp.sel_delete.isEnabled(),
                    rename=not imported and not busy and bool(project) and self.header.folder.isVisible(),
                    workspace=True, compare=editing and not imported and not busy, home=True,
                    front=self._active_cube() is not None, top=self._active_cube() is not None,
                    perspective=self._active_view() is not None, orthographic=self._active_view() is not None,
                    motion=True, keys=True, about=True, guide=True, scan_steps=not imported, fullscreen=True)

    def _refresh_menus(self):
        for key, enabled in self._menu_states().items(): self.menu_actions[key].setEnabled(enabled)
        view = self._active_view()
        mode = view.cam.projection_mode if view is not None else None
        for key in ("perspective", "orthographic"): self.menu_actions[key].setChecked(key == mode)
        self.menu_actions["scan_steps"].setChecked(self.workspace.guide_button.isChecked())

    def _menu_action(self, action):
        if self.device_projects.isVisible() and action in ("workspace", "new", "open", "export", "export_as", "timelapse", "edit", "captures"):
            self._import_holds(lambda: self._menu_action(action)); return        # leave Import as Done does, then go on
        if (self.captures.isVisible() or self.device_projects.isVisible()) and action not in ALWAYS_ON:return    # 2026-10-24: Preferences and Help open from the Import page too
        if not self._menu_states().get(action, False): return
        if action == "workspace": self.show_workspace()
        elif action == "new": self.ask_new_project()
        elif action == "open": self._workspace_open()
        elif action == "captures": self.show_import(); self.device_projects.tabs.setCurrentIndex(1)     # screenshots and screen recordings live on the Import page now
        elif action == "projects_import": self.show_import()
        elif action == "folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.imported_project or self.state.project)))
        elif action == "export": (self.imported_models if self.imported_project else self.build).export_stl()
        elif action == "export_as": (self.imported_models if self.imported_project else self.build).open_export()
        elif action == "timelapse": self.open_timelapse()
        elif action == "prefs":
            from app2.qt.preferences_dialog import PreferencesDialog
            PreferencesDialog(self).exec()
            if self.device_projects.tabs.currentIndex(): self.device_projects._show_caps_dest(); self.device_projects._fill_caps()
        elif action == "cleanup":
            from app2.qt.cleanup_dialog import CleanupDialog
            CleanupDialog(self.projects_dir or library_dir.default_dir(), self).exec()
        elif action == "close": self.close()
        elif action == "undo": self.build.undo(-1)
        elif action == "redo": self.build.undo(1)
        elif action == "edit": self.show_step(2)
        elif action == "delete_selection": self.build.vp.sel_delete.click()
        elif action == "keys": self.show_keys()
        elif action == "fullscreen": self.toggle_full_screen()
        elif action == "rename": self.header.folder.setFocus(); self.header.folder.selectAll()
        elif action == "compare": self.build.toggle_compare()
        elif action in ("perspective", "orthographic"):
            if self.imported_project:
                self.imported_models.vp.set_projection_mode(action)
            elif self.stack.currentWidget() in (self.build_slot, self.edit):
                self.build.vp.set_projection_mode(action)
            else:
                view = self._active_view()
                if view is not None: view.set_projection_mode(action)
        elif action in ("front", "top"):
            cube = self._active_cube()
            if cube is not None: cube.set_reference(action)
        elif action == "home":
            self.imported_models.home() if self.imported_project else self.scan.key_home() if self.stack.currentWidget() is self.scan else self.build.home()
        elif action == "motion": self.show_motion()
        elif action == "scan_steps": self.workspace.guide_button.toggle()
        elif action == "guide":
            self.about_card.dismiss(); self.keys.hide(); self.scrim.hide()
            self.scan.suppress_guide(True); self.guide_card.show(); self.guide_card.raise_(); self._focus_changed(None, QApplication.focusWidget())
        elif action == "report":
            from app2.diagnostics.report_window import ReportWindow
            self._report_win = ReportWindow(self._report_data, self); self._report_win.show(); self._report_win.raise_()
        elif action == "about":
            self.guide_card.dismiss()
            self.keys.hide(); self.scrim.hide()
            self.about_card.show(); self.about_card.raise_(); self._focus_changed(None, QApplication.focusWidget())

    def _report_data(self):
        """What the Report a problem window can include, as the app is right now (the raw text: the window scrubs a copy)."""
        import platform
        from app2.diagnostics import log as L, report as R
        from app2.version import VERSION
        lines = L.tail(300); snap = self.build.snap; rv = self.build._recipe_value
        settings = {"preset": ((snap.recipe if snap else None) or {}).get("preset"), "fusion method": rv("fused.method"), "point spacing mm": rv("fused.point_distance"),
                    "cut margin mm": rv("side.platter.margin"), "cut max tilt": rv("side.platter.max_tilt"), "isolation share": rv("side.keep.min_share"),
                    "line-up base": rv("lineup.place.base"), "line-up tolerance": rv("lineup.place.tol"), "line-up hints": rv("lineup.place.hints"),
                    "refine rounds": rv("lineup.refine.rounds"), "mesh method": rv("meshed.method")} if snap else {}
        sides = list(snap.sides) if snap else []
        project = {"sides": len(sides), "left out": sum(1 for x in sides if not x["active"]),
                   "frames": sum(int(((snap.node("raw@" + x["id"]) or {}).get("stats") or {}).get("frames") or 0) for x in sides)} if snap else {}
        failed = getattr(getattr(self.build, "prog", None), "failed", None) or ""
        step = (failed + "\n" if failed else "") + "\n".join(self.build.log_pop.log.toPlainText().splitlines()[-40:])
        data = dict(system=R.system_lines(lines) or ["PointYoink %s" % VERSION, platform.platform()], settings=settings, project=project,
                    popover=self.scan.log.toPlainText(), applog=lines, error=R.last_error(lines), steplog=step.strip())
        paths = [p for p in (getattr(self.state, "project", None), self.build.host_root, self.imported_project) if p]
        return data, dict(roots=[self.projects_dir] if self.projects_dir else [], projects=paths)

    def connect_captures(self):
        if self._closing or self.captures.proc:return
        if self._captures_paused:self.captures.connect_usb();return
        self._capture_return_project=getattr(self.state,'project',None)
        if self.session is None or self._crashed:
            self._captures_paused=not bool(self.imported_project);self.captures.connect_usb();return
        self.captures.status.setText('Saving the scan and releasing the camera...')
        self.captures.connect_b.setEnabled(False);self.captures.close_b.setEnabled(False)
        self._closing=True;self.session.close(then='captures')

    def captures_done(self):
        self._focus_changed(None,QApplication.focusWidget())
        if self._captures_quit:
            self.build.shutdown();self._really=True;self.close();return
        if self._captures_paused:
            self._captures_paused=False
            self.start_session(project=getattr(self,'_capture_return_project',None),last=False)

    def connect_projects(self, transport):
        if self._closing or self.device_projects.proc: return
        if transport == 'wifi':                               # Wi-Fi Share to PC: the camera stays with the scan session
            self.device_projects.start_transport(); return
        if self._projects_paused:
            self.device_projects.start_transport(); return
        self._projects_return_project = getattr(self.state, 'project', None)
        if self.session is None or self._crashed:
            self._projects_paused = not bool(self.imported_project); self.device_projects.start_transport(); return
        self.device_projects.status.setText('Saving the scan and releasing the camera...')
        self.device_projects.connect_b.setEnabled(False); self.device_projects.close_b.setEnabled(False)
        self._closing = True; self.session.close(then='projects_import')

    def projects_done(self):
        """Import's Done (or leaving Import any other way, after the same clean-up): the camera back to the scan
        session, then Projects, or where the user was going."""
        self._focus_changed(None, QApplication.focusWidget())
        then = self._after_import; self._after_import = None
        self.workspace.refresh_projects()
        if self._projects_quit:
            self.build.shutdown(); self._really = True; self.close(); return
        if self._projects_paused:
            self._projects_paused = False
            self.start_session(project=getattr(self, '_projects_return_project', None), last=False)
        (then or self.show_workspace)()
        dp = self.device_projects
        if dp.just_imported and then is None:                    # after an import: straight to the new projects
            dp.just_imported = False; dp.close_b.setText('Back')
            QTimer.singleShot(0, self.workspace.toggle_projects_popover)

    def _import_holds(self, then):
        """Leaving Import by the ribbon, a key or a menu runs Done's clean-up first (USB released, the scan session
        back). True while that holds the move: a transfer running keeps the user on Import; otherwise then() runs
        once the clean-up is over."""
        dp = self.device_projects
        if self.stack.currentWidget() is not dp or dp.isHidden(): return False
        if dp.proc: self.say(MSG["import_running"]); return True
        self._after_import = then; dp.dismiss(); return True

    def show_motion(self):
        self.guide_card.dismiss()
        self.about_card.dismiss()
        self.workspace.usage.open_panel('SCANNER', tab='Motion')     # the scanner panel's second tab

    def toggle_full_screen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def show_keys(self):
        self.guide_card.dismiss()
        self.about_card.dismiss()
        self.keys.hide()
        self.keys = self.keys_build if self.stack.currentWidget() in (self.build_slot, self.edit) else self.keys_scan
        self.scrim.setGeometry(self.rect()); self.scrim.show(); self.scrim.raise_()
        self.keys.adjustSize(); self.keys.move((self.width() - self.keys.width()) // 2, (self.height() - self.keys.height()) // 2)
        self.keys.show(); self.keys.raise_()

    def hide_sheets(self):
        from app2.qt.resource_panel import ResourcePanel
        for panel in self.findChildren(ResourcePanel):
            if panel.isVisible():panel.close();return
        if hasattr(self, 'device_projects') and self.device_projects.isVisible(): self.device_projects.dismiss(); return
        if hasattr(self,"captures") and self.captures.isVisible(): self.captures.dismiss(); return
        if self.guide_card.isVisible(): self.guide_card.dismiss(); return
        if self.about_card.isVisible(): self.about_card.dismiss(); return
        if self.startup.isVisible(): self.startup.ready(); return
        if getattr(self, "_help_card", None) is not None and self._help_card.isVisible(): self._help_card.hide(); return
        if self.project_picker is not None: self.project_picker.reject(); return
        if getattr(self.build, "export_dlg", None) is not None and self.build.export_dlg.isVisible(): self.build.close_export(); return
        if getattr(self, "timelapse_dlg", None) is not None and self.timelapse_dlg.isVisible(): self.timelapse_dlg.cancel(); return
        if self.header.folder.hasFocus():
            self.header.folder.setText(self._project_name); self.header.folder.clearFocus(); return
        open_ = self.keys.isVisible() or self.scrim.isVisible() or self.scan.popover.isVisible()
        self.keys.hide(); self.scrim.hide()
        if self.scan.popover.isVisible(): self.scan.popover.close_popover()
        page = self.stack.currentWidget()
        if not open_ and page in (self.build_slot, self.edit): self.build.page_key("Esc")
        elif not open_ and page is not self.scan and hasattr(page, "page_key"): page.page_key("Esc")
        elif not open_ and page is self.scan: self.scan.key_unpick()          # Esc: no side picked, live camera back

    def _scan_build_checked(self):
        session=self.session;state=self.state
        if session is None or state is None or state.recording or state.building:return
        from app2.builder.stagehost import StagesRun
        backend=getattr(getattr(session,'cfg',None),'build_backend',None) or StagesRun()
        self.build.check_build(backend,lambda checked:self.scan._cmd('build',checked) if self.session is session else None)

    def open_built(self):
        """Build combined model (pressed on the Scan step) is done: the Build step, open on the built model. Only from
        the Scan or Build step (a user who went elsewhere meanwhile is not pulled away)."""
        if self.stack.currentWidget() not in (self.scan, self.build_slot) or self._closing: return
        if self.keys.isVisible(): self.hide_sheets()
        self.show_step(1); self.build.show_built()

    def _sync_rebuild_pill(self, *_):
        """Build's "will rebuild" state as a pill in the header: "3 to rebuild, about 4 min" and the button."""
        import re
        bar = self.build.vp.rebuild; text = bar.text.text() if not bar.isHidden() else ""
        on = bool(text) and self.stack.currentWidget() is self.build_slot
        if on:
            m = re.search(r"(?:^|\. )([^.]*?) will rebuild \(about ([^)]*)\)", text)
            if m:
                names = re.split(r", | and ", m.group(1)); label = "%d to rebuild \u00b7 about %s" % (len(names), m.group(2))
            else: label = text
            self.header.set_rebuild_text(label, text)
            if bar.btn is not None: self.header.pill_btn.setText(bar.btn.text())
            self.header.pill_btn.setEnabled(bar.btn.isEnabled() if bar.btn is not None else True)
        self.header.pill.setVisible(on)

    def show_step(self, k):
        if k not in LIVE_STEPS:                                  # the placeholder steps have no page: Export is its action
            if k == 5: self._menu_action("export_as")
            return
        if self._import_holds(lambda:self.show_step(k)):return
        if k!=0 and self.stack.currentWidget() is self.scan and self.session is not None:
            getattr(self.session,"leave_touch_up",lambda:None)()
        if self.imported_project:
            if k == 1: self._show_imported_build()
            else: self.say("This scanner project has a finished model to view and export. Start a new scan to use Scan or Edit.")
            return
        if self.stack.currentWidget() is not self.build_slot or k != 1: self.build.close_onetap()
        self.rail.hide(); self.header.show(); self._header_for_import(False)
        if k != 0: self.scan.popover.close_popover()
        if k != 1: self.build.log_pop.close_popover()
        if k != 2 and self.edit.build is not None:
            self.build_layout.addWidget(self.edit.detach()); self.build.show()
        if k == 2 and self.edit.build is None: self.edit.attach(self.build)
        if k == 1: self.build_layout.setCurrentWidget(self.build)
        self.stack.setCurrentIndex(k); self.rail.show_step(k); self.header.step.setText(STEPS[k])
        self.workspace.set_current_step(k)
        self._refresh_export()
        hints = {1: "Choose a build stage to inspect or adjust.", 2: "Select faces to remove. Undo restores deleted faces."}
        self.scan.status.set_context(None if k == 0 else hints.get(k, ""))
        self.scan.status.setVisible(k in (0, 2))
        self.stream_rates.active = k == 1
        self.stream_rates.refresh()
        self.scan.status.log_btn.setVisible(k == 0)
        self.build.set_log_page_active(k in (1, 2))   # Edit has the Log too ("no log on edit?", 2026-10-25)
        if k == 0: QTimer.singleShot(0, self._relayout_scan)

    def _relayout_scan(self):
        self.scan.relayout(self.scan.width(), self.scan.height())

    def _refresh_export(self):
        active = self.imported_project or (getattr(self.state, "project", None) if not self._crashed else None)
        self.workspace.set_project_open(bool(active), export=self._menu_states()["export"])
        if self.imported_project:
            for name in ("Edit", "Cut base", "Smooth", "Simplify", "Line-up"):
                self.workspace.ribbon_tools[name].setEnabled(False)
                self.workspace.ribbon_tools[name].setToolTip("This scanner model is view and export only")

    def _show_imported_build(self):
        self.rail.hide(); self.header.show(); self._header_for_import(False)
        self.build_layout.setCurrentWidget(self.imported_models)
        self.stack.setCurrentWidget(self.build_slot); self.header.step.setText("Models")
        self.workspace.set_current_step(1); self.workspace.build_tools.hide()
        self.scan.status.hide(); self.build.set_log_page_active(False)
        self._refresh_export()

    def _header_for_import(self, on):
        """Import is about the scanner's projects, not the open one: no project name, no scanner pill."""
        self.header.folder.setVisible(not on); self.header.source.setVisible(not on); self.header.project.setVisible(not on)
        self.header.original.hide()
        tabs = self.device_projects.tabs
        if tabs.parent() is not self.header.slot: self.header.slot.layout().addWidget(tabs)      # Projects / Screenshots / Recordings live in the header bar
        self.header.slot.setVisible(on)

    def _uncaught_toast(self, why):
        """An error nothing caught: one toast a minute that points at Report a problem (the log has the traceback)."""
        import time
        now = time.monotonic()
        if now - self._error_said < 60.0: return
        self._error_said = now
        self.say("Something went wrong. The app is still running, but if it acts strangely, send a report.", "warn", secs=15, action=("Report a problem", lambda: self._menu_action("report")))

    def _check_memory(self):
        """Every 5 s: the app and its workers against the PC's RAM; a warning toast (once a minute) and a log line."""
        import time
        now = time.monotonic(); text = self.memory_watch.check(now); u = self.memory_watch.last
        logged = getattr(self, "_memory_logged", None)
        if u and (text or logged is None or now - logged >= 60.0):    # a line a minute in app.log: which phase grows
            self._memory_logged = now
            print("memory: ours %.1f GB, available %.1f GB of %.1f GB" % (u[0] / 1024 ** 3, u[2] / 1024 ** 3, u[1] / 1024 ** 3), flush=True)
        if text: self.say(text, "warn", secs=30)
        self._check_disk(now)

    DISK_LOW_GB = 5.0

    def _check_disk(self, now):
        """The projects folder's disk under 5 GB free: a toast (once every 10 minutes). Scans and builds write GBs."""
        import shutil
        last = getattr(self, "_disk_said", None)
        if last is not None and now - last < 600.0: return
        root = self.projects_dir or library_dir.default_dir()
        try: free = shutil.disk_usage(root).free / 1024 ** 3
        except OSError: return
        if free < self.DISK_LOW_GB:
            self._disk_said = now
            self.say("Only %.1f GB free on the disk with your projects. Scanning and building need room: free some space or clear temporary files (File > Clean up temporary files)." % free, "warn", secs=30)

    def say(self, text, level="warn", secs=8, action=None):
        """A short message on whatever page is shown."""
        if text and str(text).startswith('Opening project'):
            self.scan.status.setToolTip(text);self.scan.status.hint.setText(text);return
        self.toast.say(text, level, secs, action)

    def open_timelapse(self):
        """File > Export scan timelapse... (and the Export sheet's link): the sheet, for the open project's saved sides."""
        from app2.qt import build_words as W
        from app2.qt.timelapse_sheet import TimelapseSheet
        project = getattr(self.state, "project", None); n = len(fmt.saved_sides(project)) if project else 0
        if not n: self.say(W.TIMELAPSE_SHEET["no_sides"], "info"); return
        d = getattr(self, "timelapse_dlg", None)
        if d is None:
            d = self.timelapse_dlg = TimelapseSheet(self); d.done.connect(self._timelapse_done)
            d.closed.connect(lambda: self._focus_changed(None, QApplication.focusWidget()))
            for signal in (d.done, d.failed, d.stopped): signal.connect(self._timelapse_closed)
        d.present(project, n); d.setFocus()

    def _timelapse_closed(self, *_args):
        if not getattr(self, "_timelapse_close_pending", False): return
        self._timelapse_close_pending = False
        QTimer.singleShot(0, self.close)

    def _timelapse_done(self, r):
        from app2.qt import build_words as W
        E = W.TIMELAPSE_SHEET; path = r["path"]; folder = os.path.dirname(path)
        self.say(E["done"].format(name=os.path.basename(path), mb="%.1f" % (r["bytes"] / 1e6)), "info", 12,
                 action=(E["open_folder"], lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder))))

    def _build_message(self, text, level):
        if self.stack.currentWidget() not in (self.build_slot, self.edit): self.say(text, level, 12)

    def _on_live(self, _v):
        self.stream_rates.add(_v)
        if not self._streaming:
            self._streaming = True
            if self.state is not None: self._source(self.state)

    def show_workspace(self):
        """Show the project-first surface without interrupting an active session."""
        if self._import_holds(self.show_workspace): return
        self.build.close_onetap()
        self.hide_sheets()
        self.scan.popover.close_popover(); self.build.log_pop.close_popover()
        self.scan.status.log_btn.hide(); self.build.set_log_page_active(False)
        project = self.imported_project or getattr(self.state, "project", None)
        self.workspace.set_active_project(project if not self._crashed and project and os.path.isdir(os.path.join(project, "data")) else None,
                                          view_only=bool(self.imported_project))
        self._refresh_export()
        self.workspace.refresh_projects()
        self.stack.setCurrentWidget(self.workspace)
        self.workspace.set_current_step(None)
        self.rail.hide(); self.header.hide(); self.header.step.setText("Workspace")
        self.scan.status.set_context(None)
        self.scan.status.hide()

    def show_import(self):
        """Keep project transfer inside the same ribbon and content area."""
        if self.stack.currentWidget() is self.device_projects and not self.device_projects.isHidden(): return   # already here
        self.hide_sheets(); self.build.close_onetap()
        self.scan.popover.close_popover(); self.build.log_pop.close_popover()
        self.scan.status.log_btn.hide(); self.build.set_log_page_active(False)
        self.stack.setCurrentWidget(self.device_projects)
        self.device_projects.present()
        self.workspace.set_current_step('import')
        self.rail.hide(); self.header.show(); self.header.step.setText('Import'); self._header_for_import(True)
        self.scan.status.set_context(None)
        self.scan.status.hide()

    @staticmethod
    def project_sides(path):
        """How many sides a project folder has on disk (a folder under data/, as the project cards count them)."""
        data = os.path.join(path or "", "data")
        try: return sum(1 for x in os.listdir(data) if os.path.isdir(os.path.join(data, x)))
        except OSError: return 0

    def landing_step(self, path):
        """Where an opened project lands: Build when it has a side to build, else Scan."""
        return 1 if self.project_sides(path) else 0

    def _is_open(self, path):
        cur = self.imported_project or getattr(self.state, "project", None)
        return bool(cur and path and (self.imported_project or (self.session is not None and not self._crashed)) and not self._closing
                    and os.path.realpath(cur) == os.path.realpath(path))

    def _open_imported(self, path, sides=None):
        if sides is None:
            kind, sides = imported_project(path)
            if kind != "models":
                self.session = None; self.state = None; self.show_workspace()
                self.say("The scanner model is no longer available. Refresh Projects and try again."); return
        if self.edit.build is not None:
            self.build_layout.addWidget(self.edit.detach()); self.build.show()
        self.session = None; self.state = None; self._crashed = False; self.imported_project = path
        self.build.attach(None); self.imported_models.open_project(path, sides)
        self._orientation_project = None
        from app2.project_preview import project_name
        self._project_name = project_name(path); self._project_name_path = path
        self.header.set_project(fmt.project_when(path), os.path.basename(path), self._project_name)
        self.header.set_source("Imported scanner model", "ink3")
        self.header.new.setEnabled(True); self.header.open.setEnabled(True)
        self.workspace.set_active_project(path, view_only=True)
        self._show_imported_build()

    def _workspace_new(self):
        if self.ask_new_project(): self.show_step(0)

    def _workspace_open(self):
        path = self.ask_open_project()
        if path: self.show_step(self.landing_step(path))

    def _workspace_continue(self):
        if self.imported_project: self._show_imported_build(); return
        self.show_step(self.landing_step(getattr(self.state, "project", None)))

    def _workspace_project(self, path):
        """Open a concrete card from the Workspace rail using normal save safety."""
        if not path or not os.path.isdir(os.path.join(path, "data")):
            self.say(MSG["not_project"].format(name=os.path.basename(os.path.normpath(path or "")) or "That folder")); return
        kind, sides = imported_project(path)
        if kind == "raw_only":
            self.show_workspace(); self.say(RAW_ONLY_MESSAGE, secs=16); return
        if self._is_open(path): self.show_step(self.landing_step(path)); return        # already open: no new session
        if kind == "models":
            if self.session is None or self._crashed: self._open_imported(path, sides); return
            s = self.state
            if s is None or s.busy or s.building or self._closing or self.build.busy or self.build.busy_call:
                self.say(MSG["busy_open"]); return
            self._closing = True; self.session.close(then=("imported", path)); return
        s = self.state
        if self.session is None or self._crashed:
            self.start_session(project=path, last=False); self.show_step(self.landing_step(path)); return
        if s is None or s.busy or s.building or self._closing: self.say(MSG["busy_open"]); return
        self._closing = True; self.session.close(then=("open", path)); self.show_step(self.landing_step(path))

    def _prepare_imported(self, path):
        if self._prepare_future is not None:
            self.say("An imported scan is already being prepared."); return
        if not needs_preparation(path):
            self.workspace.refresh_projects(); self.say("This project has no unprepared raw scan sides."); return
        self.say("Checking the scanner poses and preparing the raw sides for Build...", secs=12)
        future = self._prepare_future = self._prepare_executor.submit(adopt_project, path)
        future.add_done_callback(lambda done: self._prepare_done.emit((path, done)))

    def _finish_prepare_imported(self, result):
        path, future = result; self._prepare_future = None
        try: report = future.result()
        except (OSError, ValueError) as exc:
            self.say("Could not prepare this scan: " + str(exc), secs=16); return
        self.workspace.refresh_projects()
        if report["issues"]:
            self.say(issue_summary(report), secs=20); return
        if report["status"] != "prepared":
            self.say("This project has no raw scan sides to prepare."); return
        self.say("Prepared %d side%s and %d frames for Build." %
                 (len(report["sides"]), "s" if len(report["sides"]) != 1 else "", report["frames"]), secs=10)
        if self.imported_project == path:
            self.imported_project = None; self.imported_models.clear_project()
            self.build_layout.setCurrentWidget(self.build)
        self._workspace_project(path)

    def _workspace_folder(self):
        if self.projects_dir and os.path.isdir(self.projects_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.projects_dir)))

    def _workspace_mode(self, mode):
        """Map the 1.0 ribbon vocabulary to the live 2.0 page that owns it today."""
        if mode == 'Import':
            self.show_import(); return
        if self._import_holds(lambda: self._workspace_mode(mode)): return
        if mode == "Projects":
            self.show_workspace(); return
        if mode == 'Import':
            self.show_import(); return
        if mode == "Export":
            self._menu_action("export_as"); return
        if self.imported_project and mode not in ("Projects", "Build"):
            self.say("This scanner model is view and export only. Start a new scan to use these tools."); return
        stages = {"Scan": (0, None, None), "Build": (1, None, None), "Edit": (2, None, None),
                  "Cut base": (1, "side", "Platter"), "Smooth": (1, "fused", "Smooth"),
                  "Simplify": (1, "meshed", "Simplify"), "Align": (1, "lineup", None)}
        if mode not in stages: return
        step, stage, tab = stages[mode]
        if mode == "Scan" and self.session is None and not self.imported_project:
            self.start_session(project=None, last=False)
        self.show_step(step)
        if stage is not None:
            self.build.show_stage(stage)
            if tab is not None: self.build.panel[stage].show_tab(tab)

    # ------------------------------------------------------------------ session lifecycle
    def start_session(self, **over):
        if self.imported_project:
            self.imported_project = None; self.imported_models.clear_project()
            self.build_layout.setCurrentWidget(self.build)
        self.startup.set_status("Opening project")
        if getattr(self, "_splashed", False): self.say("Opening project...", "info", 4)     # later opens: no splash flash
        else:
            self._splashed = True; self.startup.show(); self.startup.raise_()                # the first start shows the splash
        self._focus_changed(None, QApplication.focusWidget())
        try: self.session = self.make_session(self.bridge, **over)
        except Exception as e:
            self.session = None; self.scan.log.append("could not start: %s" % e)
            self.startup.fail("Could not open the project. See Log for details.")
            self.startup.ready()
            if str(e): self.scan.set_ended(str(e)); self.build.set_ended(); self.say(str(e), secs=12)
            return
        self.state = None; self._crashed = False; self._close_after = False; self._streaming = False
        if hasattr(self.session, "set_motion_provider"):
            client = self.motion_panel.client
            self.session.set_motion_provider(getattr(client, "snapshot_at", client.snapshot))
        self.scan.attach(self.session); self.build.attach(self.session)
        self.startup.set_status("Connecting camera")
        try: self.session.start()
        except Exception as e:
            self._crashed = True
            self.on_error("Could not start the camera: %s" % e)

    def ask_new_project(self):
        s = self.state
        if self.imported_project:
            self.start_session(project=None, last=False); self.show_step(0); return True
        if self.session is None or self._crashed: self.start_session(project=None, last=False); return True   # nothing left to close
        if s is None or s.busy or s.building or self._closing: self.say(MSG["busy_new"]); return
        if s.recording:
            if not self.confirm("New scan","Stop recording, save this project, and start a new scan?"): return
        elif s.n_sides:
            choice = self.ask_keep_or_discard("New project", "Save this project and start a new one?")
            if choice is None: return
            if choice == "discard": self._closing = True; self.session.close(then=("discard", s.project, "new")); return True
        self._closing = True; self.session.close(then="new"); return True

    def ask_keep_or_discard(self, title, text):
        """Save / Discard / Cancel. Returns "keep", "discard" or None. Tests put a stand-in here like confirm."""
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self)
        keep = box.addButton("Save and start new", QMessageBox.ButtonRole.AcceptRole)
        drop = box.addButton("Discard this project", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel); box.setDefaultButton(keep)
        drop.setToolTip("Deletes this project's folder and every side in it. This can't be undone.")
        box.exec(); b = box.clickedButton()
        return "keep" if b is keep else "discard" if b is drop else None

    def ask_open_project(self):
        """Pick a project folder and open it. Returns the folder opened (or already open), else None."""
        s = self.state
        if self.session is not None and not self._crashed and (s is None or s.busy or s.building or self._closing):
            self.say(MSG["busy_open"]); return
        path = self.pick_folder()
        if not path: return
        if not os.path.isdir(os.path.join(path, "data")):
            self.scan.log.append("Open project: %s has no data folder, so it is not a scan project" % path)
            self.say(MSG["not_project"].format(name=os.path.basename(os.path.normpath(path)))); return
        kind, sides = imported_project(path)
        if kind == "raw_only":
            self.show_workspace(); self.say(RAW_ONLY_MESSAGE, secs=16); return
        if self._is_open(path): return path                                  # already open: keep its session
        if kind == "models":
            if self.session is None or self._crashed: self._open_imported(path, sides); return
            if self.build.busy or self.build.busy_call: self.say(MSG["busy_open"]); return
            self._closing = True; self.session.close(then=("imported", path)); return
        if self.session is None or self._crashed: self.start_session(project=path, last=False); return path
        self._closing = True; self.session.close(then=("open", path)); return path

    def _ask(self, title, text):
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _pick_folder(self):
        from app2.qt.project_picker import ProjectPicker
        picker = ProjectPicker(self, self.projects_dir or library_dir.default_dir())
        self.project_picker = picker
        try: return picker.run()
        finally:
            self.project_picker = None; self._project_name_path = None; picker.deleteLater()
            self._focus_changed(None, QApplication.focusWidget())

    def closeEvent(self, e):
        if hasattr(self, 'device_projects') and self.device_projects.isVisible():
            e.ignore()
            if self.device_projects.proc:
                self.device_projects.cancel()
                self.device_projects.status.setText('Transfer canceled. Click Done to release USB, then close the app.')
                return
            self._projects_quit = True; self.device_projects.dismiss(); return
        if hasattr(self,'captures') and self.captures.isVisible():
            e.ignore()
            if self.captures.proc:
                self.captures.cancel();self.captures.status.setText('Transfer canceled. Click Done to release USB, then close the app.');return
            self._captures_quit=True;self.captures.dismiss();return
        self.about_card.dismiss()
        if self.project_picker is not None: self.project_picker.reject()
        if getattr(self, "timelapse_dlg", None) is not None and self.timelapse_dlg.busy:
            e.ignore(); self._timelapse_close_pending = True
            self.timelapse_dlg.stop(); return
        if self.imported_project and self.imported_models.export_future is not None:
            e.ignore(); self.say("Wait for the scanner model export to finish."); return
        if self._prepare_future is not None:
            e.ignore(); self.say("Wait for the imported scan preparation to finish."); return
        if self.imported_project: self.build.shutdown()
        if self._really or self.session is None or self._crashed:
            self.build.shutdown()
            try: QApplication.instance().focusChanged.disconnect(self._focus_changed)
            except (RuntimeError, TypeError): pass
            self.imported_models.shutdown()
            self._prepare_executor.shutdown(wait=False, cancel_futures=True)
            self.motion_link.shutdown(); self.motion_panel.shutdown(); e.accept(); return
        e.ignore()
        if self._closing: self._close_after = True; return        # the session is saving for New / Open project
        s = self.state
        if s is not None and s.building and not getattr(self, "force_close", False) and \
                not self.confirm("Build running", "A build is running. Stop it and close?"): return
        if not (s is not None and s.building) and self.build.busy and not getattr(self, "force_close", False) and \
                not self.confirm("Build step busy", "The Build step is still working on a change. Stop it and close?"): return
        self.motion_link.shutdown(); self.motion_panel.shutdown()
        self.build.shutdown()
        self._closing = True; self._close_after = True; self.scan.status.hint.setText("Saving every side...")
        self.session.close()

    def on_closed(self, info):
        self._closing = False
        if info.get("crashed"):
            self._crashed = True; self.scan.set_ended(); self.build.set_ended()
            self.header.new.setEnabled(True); self.header.open.setEnabled(True)      # the way out, even if it stopped busy
            self.scan.log.append("The scan session stopped on an error; every side it had is saved.")
            if self.state is not None: self._source(self.state)
            self.workspace.set_active_project(None)
            self.workspace.refresh_projects()
        if self._close_after: self._really = True; self.close(); return
        if info.get("crashed") and info.get("then") is None: return      # stopped on an error mid-scan: the window stays
        then = info.get("then")
        if then == "captures":
            self.session=None;self._captures_paused=True;self.scan.set_ended();self.build.set_ended()
            self.header.set_source('USB File Transfer','accent');self.captures.connect_usb();return
        if then == 'projects_import':
            self.session = None; self._projects_paused = True; self.scan.set_ended(); self.build.set_ended()
            self.header.set_source('Scanner transfer', 'accent'); self.device_projects.start_transport(); return
        if then == "new": self.start_session(project=None, last=False); return
        if isinstance(then, (tuple, list)) and then and then[0] == "discard":      # the closed project's folder goes, then on to the next thing
            path = then[1]
            if path and os.path.isdir(path):
                import shutil
                try: shutil.rmtree(path)
                except OSError as e: self.scan.log.append("Could not delete %s: %s" % (path, e))
                self.workspace.set_active_project(None); self.workspace.refresh_projects()
            then = then[2]
            if then == "new": self.start_session(project=None, last=False); return
        if isinstance(then, (tuple, list)) and then and then[0] == "imported": self._open_imported(then[1]); return
        if isinstance(then, (tuple, list)) and then and then[0] == "open": self.start_session(project=then[1], last=False); return
        self._really = True; self.close()

    def on_error(self, text):
        if not self.startup.isHidden():
            self.startup.fail("Startup failed. See Log for details."); self.startup.ready()
        print("session error:", text[-600:], flush=True)
        for ln in text.strip().splitlines()[-2:]: self.scan.log.append(ln)

    def on_state(self, s):
        if self.imported_project: return
        if not self.startup.isHidden(): self.startup.ready()
        self.light_prompt.update_time(s.light_remaining)
        self.state = s
        from app2.qt.usage_meter import sampler
        sampler().build_usage.set_building(bool(s.building))
        self.workspace.set_project_open(bool(getattr(self.state, "project", None)) and not self._crashed, export=self._menu_states()["export"])
        if self.stack.currentWidget() is self.workspace:
            self.workspace.set_active_project(s.project)
        if self._orientation_project != s.project:
            from app2.view_orientation import load_orientation
            self._orientation_project = s.project
            self._apply_view_basis(load_orientation(s.project), reset_bed=True)
            self.workspace.refresh_projects()
        if self._project_name_path != s.project:
            from app2.project_preview import project_name
            self._project_name = project_name(s.project); self._project_name_path = s.project
        self.header.set_project(fmt.project_when(s.project), os.path.basename(s.project), self._project_name)
        b = self.scan.btns or {}
        ok = lambda k: self._crashed or (bool(b.get(k, (0, 0, True))[2]) and not self._closing)
        self.header.new.setEnabled(ok("new_project")); self.header.open.setEnabled(ok("open_project"))
        self._source(s)

    def _active_view(self):
        if self.imported_project and self.stack.currentWidget() is self.build_slot:
            return self.imported_models.vp.view
        if not self._view_cubes: return None
        if self.stack.currentWidget() is self.scan: return self.scan.model.view
        if self.stack.currentWidget() in (self.build_slot, self.edit): return self.build.vp.view
        return None

    def _active_cube(self):
        if self.imported_project and self.stack.currentWidget() is self.build_slot:
            view = self.imported_models.vp.view
            return view.nav_cube if view is not None and view.isVisible() else None
        if not self._view_cubes: return None
        if self.stack.currentWidget() is self.scan:
            view = self.scan.model.view
            return view.nav_cube if view is not None and view.isVisible() else None
        if self.stack.currentWidget() in (self.build_slot, self.edit): return self.build.vp.view.nav_cube
        return None

    def _save_view_orientation(self, basis):
        from app2.view_orientation import save_orientation, load_orientation
        if self._orientation_project:
            try: save_orientation(self._orientation_project, basis)
            except (OSError, ValueError) as exc:
                self.scan.log.append("Could not save view directions: %s" % exc)
                self._apply_view_basis(load_orientation(self._orientation_project))
                return
        self._apply_view_basis(basis)

    def _apply_view_basis(self, basis, reset_bed=False):
        self.scan.model.set_project_basis(basis, reset_bed=reset_bed)
        for pane in (self.build.vp.main, self.build.vp.before):
            if pane.view is not None: pane.view.nav_cube.set_basis(basis)

    def rename_active_project(self, name):
        if self.state is None or not self.state.project or name == self._project_name: return
        from app2.project_preview import rename_project
        try:
            self._project_name = rename_project(self.state.project, name)
            self._project_name_path = self.state.project
            self.header.folder.setText(self._project_name)
        except (OSError, ValueError) as e:
            self.scan.log.append("Could not save project name: %s" % e)
            self.header.folder.setText(self._project_name)

    def on_status(self, v):
        self._source(self.state)

    def _source(self, s):
        from app2.scan.usb_mode import pill_text as usb_pill
        """The header's scanner pill: connected only once frames are coming in."""
        if self._crashed: self.header.set_source("Session stopped", "lost")
        elif s is None: self.header.set_source(usb_pill(), "ink3")
        elif s.source_kind == "replay": self.header.set_source(s.source_label, "accent")
        elif s.cam == "gone": self.header.set_source("Scanner disconnected" if self._streaming else usb_pill(), "warn" if self._streaming else "ink3")
        elif s.cam == "noframes": self.header.set_source("Scanner not answering", "warn")
        elif not self._streaming: self.header.set_source(usb_pill(), "ink3")
        else: self.header.set_source(s.source_label or "Scanner in PC mode", "kept")

    # ------------------------------------------------------------------ sizes
    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        self.rail.set_compact(w < 1280)
        self.header.folder.setVisible(self.stack.currentWidget() is not self.device_projects)
        QTimer.singleShot(0, self._relayout_scan)
        if self.toast.isVisible(): self.toast.place()
        if self.project_picker is not None: self.project_picker.relayout()
        self.scrim.setGeometry(self.rect())
        if self.keys.isVisible():
            self.keys.adjustSize(); self.keys.move((w - self.keys.width()) // 2, (h - self.keys.height()) // 2)
        if hasattr(self,"captures") and self.captures.isVisible(): self.captures.present()
        super().resizeEvent(e)
