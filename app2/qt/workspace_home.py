# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project-first Workspace surface for PointYoink 2.0.

This is deliberately a small navigation layer over the live 2.0 session. It
does not own scanner state or project storage: New and Open are supplied by
the shell so they always use the normal save/close safeguards. "Open project
from folder" picks a folder here and hands it to the shell as project_requested,
the same safe path a project card uses.

Ribbon words match the Build stage list: "Line-up" (emitted to the shell as its
older mode key "Align") and "Cut base" (opens the Side stage's Cut base tab).
"""
import os
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QSize
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
                               QProgressBar, QToolButton, QVBoxLayout, QWidget)

from app2.project_preview import project_name, project_rows, preview_bytes, BACKUP_WORDS
from scanner.adopt_device_scan import needs_preparation
from app2 import disk_usage
from app2.qt.usage_meter import UsageMeter
from app2.qt.widgets import Mark
from app2.qt.theme import ui_font
from app2.qt.icons import icon


# Build subtools: (label shown, mode key the shell routes on, icon). The shell's router keys predate the ribbon words.
BUILD_TOOLS = (("Cut base", "Cut base", "cut-base"), ("Smooth", "Smooth", "brush"), ("Simplify", "Simplify", "prepare"),
               ("Line-up", "Align", "compare"))
START_COPY = "Start a new scan, or open one you already have."
PROJECT_TOOLS = ("Build", "Edit", "Export")      # need an open project; Projects, Import and Scan never do
CARDS_SHOWN = 20                                 # project cards before "Show all"


class MiniProgress(QFrame):
    """A build that was minimized: one line, a thin bar and a small Stop in the top bar. Click the line to bring the card back."""
    restore = Signal()
    stop = Signal()

    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName("MiniProgress"); self.setFixedHeight(46); self.setMinimumWidth(160); self.setMaximumWidth(600)
        self.setStyleSheet("QFrame#MiniProgress {background:#1d2229;border:1px solid #2f6fb5;border-radius:8px;} "
                           "QLabel {color:#d8e0ea;} QPushButton {padding:2px 10px;}")
        v = QVBoxLayout(self); v.setContentsMargins(10, 4, 8, 5); v.setSpacing(3)
        row = QHBoxLayout(); row.setSpacing(6)
        self.text = QLabel(""); self.text.setFont(ui_font(13, QFont.Weight.DemiBold)); self.text.setCursor(Qt.CursorShape.PointingHandCursor)
        self.text.setToolTip("Click to bring the build card back"); self.text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.stop_b = QPushButton("Stop"); self.stop_b.setFixedHeight(22); self.stop_b.clicked.connect(self.stop.emit)
        row.addWidget(self.text, 1); row.addWidget(self.stop_b); v.addLayout(row)
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(6); self.bar.setRange(0, 1000); v.addWidget(self.bar)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setToolTip("Click to bring the build card back")
        self._full = ""; self.hide()

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(getattr(self, "_want", 380), self.height())

    def _elide(self):
        """The whole line when it fits; squeezed, the progress part alone ("8 of 24 steps done, about 45 s left") before
        the title, since "Building …" says nothing."""
        fm = self.text.fontMetrics(); room = max(40, self.width() - self.stop_b.sizeHint().width() - 40)
        text = self._full
        if fm.horizontalAdvance(text) > room and "\u00b7" in text: text = text.split("\u00b7", 1)[1].strip()
        self.text.setText(fm.elidedText(text, Qt.TextElideMode.ElideRight, room))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._elide()

    def mousePressEvent(self, e):
        """Anywhere on the strip but its Stop button brings the card back (a per-instance override of the label's handler never ran:
        PySide dispatches events to the class, not to an attribute set on the object)."""
        if e.button() == Qt.MouseButton.LeftButton: self.restore.emit(); e.accept()
        else: super().mousePressEvent(e)

    def set_status(self, text, frac):
        """The whole line when the bar has room: the strip asks for the width that fits (up to its maximum), the layout may
        give it less (a fixed width rode over the Advanced and Guide buttons, 2026-10-22), and the text is elided to what it got."""
        self._full = text; fm = self.text.fontMetrics()
        need = fm.horizontalAdvance(text) + self.stop_b.sizeHint().width() + 64
        self._want = max(380, min(600, need)); self.setMinimumWidth(160); self.setMaximumWidth(self._want); self.updateGeometry()
        self._elide()
        self.text.setToolTip(text + "\nClick to bring the build card back")
        if frac <= -2: self.bar.setRange(0, 1000); self.bar.setValue(self.bar.value())          # paused: the bar stays where it stopped
        elif frac < 0: self.bar.setRange(0, 0)
        else: self.bar.setRange(0, 1000); self.bar.setValue(int(1000 * min(1.0, frac)))


class WorkspaceHome(QWidget):
    new_requested = Signal()
    open_requested = Signal()
    continue_requested = Signal()
    project_requested = Signal(str)
    prepare_requested = Signal(str)
    folder_requested = Signal()
    mode_requested = Signal(str)
    import_requested = Signal()
    _sized = Signal(str, object)                   # (project path, bytes) from the disk-usage thread
    _thumb_ready = Signal(str, object)             # (project path, PNG bytes or None) from a preview thread
    _deleted = Signal(str, str)                    # (project path, error or '') from the delete thread
    project_deleted = Signal(str)

    def __init__(self, projects_dir=None, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkspaceHome")
        self.projects_dir = projects_dir
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        tools = QFrame(); tools.setObjectName("WorkspaceTools"); tools.setMinimumWidth(700)
        tool_rows = QVBoxLayout(tools); tool_rows.setContentsMargins(0, 0, 0, 0); tool_rows.setSpacing(0)
        pages = QWidget(); tl = QHBoxLayout(pages); tl.setContentsMargins(10, 6, 10, 6); tl.setSpacing(8)
        tl.addSpacing(6); self.mark = Mark(size=54); self.mark.setToolTip('PointYoink'); tl.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignVCenter); tl.addSpacing(14)   # same room both sides
        ribbon = (("Projects", "projects"), ("Import", "import"), ("Scan", "live-view"), ("Build", "build"),
                  ("Edit", "edit"), ("Export", "export"))
        self.ribbon_tools = {}
        for text, asset in ribbon:
            tool = QToolButton(); tool.setObjectName("WorkspaceToolPrimary" if text == "Projects" else "WorkspaceTool")
            tool.setText(text); tool.setIcon(icon(asset)); tool.setIconSize(QSize(25, 25))
            tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon); tool.setFixedSize(94, 66)
            tool.setEnabled(text not in PROJECT_TOOLS)         # Build, Edit, Export wait for a project (they looked clickable on the first paint)
            tool.clicked.connect(lambda _=False, name=text: self.mode_requested.emit(name)); tl.addWidget(tool); self.ribbon_tools[text] = tool
        tl.addStretch(1)
        self.page_tools = tuple(self.ribbon_tools)
        tool_rows.addWidget(pages)
        build_tools = QFrame(); build_tools.setObjectName("WorkspaceBuildTools"); build_tools.setFixedHeight(66)
        subrow = QHBoxLayout(build_tools); subrow.setContentsMargins(7, 5, 7, 5); subrow.setSpacing(4)
        for text, mode, asset in BUILD_TOOLS:
            tool = QToolButton(); tool.setObjectName("WorkspaceSubTool")
            tool.setText(text); tool.setIcon(icon(asset)); tool.setIconSize(QSize(18, 18))
            tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon); tool.setFixedSize(84, 54)
            tool.clicked.connect(lambda _=False, name=mode: self.mode_requested.emit(name))
            subrow.addWidget(tool); self.ribbon_tools[text] = tool
        tl.insertWidget(7, build_tools); self.build_tools = build_tools   # after the logo, its spacing, Projects, Import, Scan, Build
        self.mini = MiniProgress(); tl.addWidget(self.mini, 2, Qt.AlignmentFlag.AlignVCenter); tl.addStretch(1)       # centered in the bar's free space
        # Simple or Advanced, and the first scan guide: page-wide choices, so they sit at the right beside the meters
        from app2.qt.choices import Choices
        from app2.qt import ui_level
        self.level = Choices(); self.level.addItems(ui_level.LEVELS); self.level.hide()   # holds the choice; the button shows it
        self.level_button = QToolButton(); self.level_button.setObjectName("WorkspaceTool"); self.level_button.setIcon(icon("settings"))
        self.guide_button = QToolButton(); self.guide_button.setObjectName("WorkspaceTool"); self.guide_button.setIcon(icon("check"))
        self.guide_button.setText("Guide"); self.guide_button.setCheckable(True); self.guide_button.setToolTip("Show or hide the step-by-step first scan guide")
        self.guide_button.hide()                                  # 2026-10-24: off the bar (it only did anything on the Scan page); Help > First scan steps instead
        for tool in (self.level_button,):
            tool.setIconSize(QSize(25, 25)); tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon); tool.setFixedSize(94, 66)
            tl.addWidget(tool, 0, Qt.AlignmentFlag.AlignVCenter)
        tl.addSpacing(12)
        def show_level(level):
            self.level_button.setText(level)
            self.level_button.setToolTip("%s controls. Click for %s. %s" % (level, [l for l in ui_level.LEVELS if l != level][0], ui_level.TIP))
        self.level.currentTextChanged.connect(show_level); show_level(self.level.currentText())
        self.level_button.clicked.connect(lambda: self.level.setCurrentText([l for l in ui_level.LEVELS if l != self.level.currentText()][0]))
        self.usage = UsageMeter(); tl.addWidget(self.usage, 0, Qt.AlignmentFlag.AlignVCenter); tl.addSpacing(8)   # right end, every page
        ribbon_scroll = QScrollArea(); ribbon_scroll.setFrameShape(QFrame.Shape.NoFrame)
        ribbon_scroll.setWidgetResizable(True)
        ribbon_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ribbon_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ribbon_scroll.setWidget(tools); ribbon_scroll.setFixedHeight(88); root.addWidget(ribbon_scroll); self.ribbon_scroll = ribbon_scroll
        page_header = QFrame(); page_header.setObjectName("WorkspacePageHeader"); page_header.setFixedHeight(52)
        page_header_row = QHBoxLayout(page_header); page_header_row.setContentsMargins(18, 0, 14, 0)
        title = QLabel("Workspace"); title.setObjectName("WorkspaceTitle"); title.setFont(ui_font(18, QFont.Weight.Bold))
        page_header_row.addWidget(title)
        self.projects_dropdown = QPushButton('Your projects  ▾'); self.projects_dropdown.setObjectName('WorkspaceProjectDropdown')
        self.projects_dropdown.setFixedHeight(34)
        self.projects_dropdown.setStyleSheet('QPushButton {background:#252c35;border:1px solid #3a4858;border-radius:6px;color:#dbe7f5;padding:4px 12px;} QPushButton:hover {background:#304156;}')
        self.projects_dropdown.clicked.connect(self.toggle_projects_popover)
        page_header_row.addSpacing(18); page_header_row.addWidget(self.projects_dropdown); page_header_row.addStretch(1)
        root.addWidget(page_header); self.page_header = page_header
        outer = QHBoxLayout(); outer.setContentsMargins(0, 24, 0, 32); outer.setSpacing(0); root.addLayout(outer, 1)

        main = QVBoxLayout(); main.setSpacing(18)
        main.addStretch(1)
        start = QFrame(); start.setObjectName("WorkspaceStart"); start.setProperty("role", "panel"); self.start = start
        start.setMinimumHeight(270); start.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sv = QVBoxLayout(start); sv.setContentsMargins(30, 28, 30, 28); sv.setSpacing(12)
        heading = QLabel("Start a project"); heading.setFont(ui_font(28, QFont.Weight.Bold)); sv.addWidget(heading); self.start_heading = heading
        copy = QLabel(START_COPY)
        copy.setWordWrap(True); copy.setProperty("role", "hint"); copy.setFont(ui_font(15)); sv.addWidget(copy); self.start_copy = copy
        actions = QHBoxLayout(); actions.setSpacing(12)
        self.continue_b = QPushButton("Go to Build"); self.continue_b.setObjectName("WorkspacePrimary"); self.continue_b.setFixedHeight(50)
        self.continue_b.setVisible(False); actions.addWidget(self.continue_b)
        self.new = QPushButton("New scan"); self.new.setObjectName("WorkspacePrimary"); self.new.setFixedHeight(50)
        self.open = QPushButton("Open project"); self.open.setObjectName("WorkspaceSecondary"); self.open.setFixedHeight(50)
        actions.addWidget(self.new); actions.addWidget(self.open); sv.addLayout(actions)
        note = QLabel("Build and edit become available after you capture or open a scan.")
        note.setWordWrap(True); note.setProperty("role", "hint"); sv.addWidget(note); self.start_note = note
        card_row = QHBoxLayout(); card_row.setContentsMargins(0, 0, 0, 0); card_row.setSpacing(0)
        card_row.addStretch(1); card_row.addWidget(start, 2); card_row.addStretch(1)
        main.addLayout(card_row)
        main.addStretch(1)
        outer.addLayout(main, 1)

        self.new.clicked.connect(self.new_requested); self.open.clicked.connect(self._open_clicked); self.continue_b.clicked.connect(self.continue_requested)
        film = QFrame(); film.setObjectName("WorkspaceFilmstrip"); film.setFixedHeight(86)
        fl = QHBoxLayout(film); fl.setContentsMargins(16, 10, 16, 10); fl.setSpacing(10)
        self.film_note = QLabel("No project selected · Start a scan or open one to begin."); self.film_note.setProperty("role", "hint"); fl.addWidget(self.film_note, 1)
        root.addWidget(film); self.filmstrip = film
        popover = QFrame(self); popover.setObjectName('WorkspaceProjectsPopover'); popover.setProperty('role', 'panel')
        popover.setStyleSheet('#WorkspaceProjectsPopover {background:#20252c;border:1px solid #38434f;border-radius:0;}')
        pv = QVBoxLayout(popover); pv.setContentsMargins(18, 18, 18, 18); pv.setSpacing(10)
        label = QLabel('Your projects'); label.setFont(ui_font(18, QFont.Weight.Bold)); pv.addWidget(label)
        hint = QLabel('Choose a project to open its scans.'); hint.setProperty('role', 'hint'); hint.setWordWrap(True); pv.addWidget(hint)
        scroll = QScrollArea(); scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cards = QWidget(); self.project_list = QVBoxLayout(cards); self.project_list.setContentsMargins(0, 0, 14, 0); self.project_list.setSpacing(8)
        scroll.setWidget(cards); pv.addWidget(scroll, 1)
        self.count_note = QLabel(''); self.count_note.setProperty('role', 'hint'); self.count_note.setWordWrap(True); pv.addWidget(self.count_note)
        self.show_all_b = QPushButton('Show all'); self.show_all_b.clicked.connect(self._toggle_show_all); pv.addWidget(self.show_all_b)
        self._show_all = False
        self.space_note = QLabel(''); self.space_note.setProperty('role', 'hint'); pv.addWidget(self.space_note)
        self._size_labels, self._sizes = {}, {}; self._sized.connect(self._on_sized)
        self._thumbs = {}; self._thumb_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='card-preview'); self._thumb_ready.connect(self._on_thumb)
        self._deleted.connect(self._on_deleted); self._deleting = set()
        browse = QPushButton('Open project from folder…'); browse.clicked.connect(self.browse_project_folder); pv.addWidget(browse)
        import_b = QPushButton('Import from scanner…'); import_b.clicked.connect(lambda: self._popover_action(self.import_requested)); pv.addWidget(import_b)
        folder = QPushButton('Open save folder'); folder.clicked.connect(lambda: self._popover_action(self.folder_requested)); pv.addWidget(folder)
        self.projects_popover = popover; self.projects_popover.hide()
        self.set_current_step(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.start.setMaximumWidth(600)
        if self.projects_popover.isVisible(): self._place_projects_popover()

    def set_current_step(self, step):
        if step is not None and hasattr(self, 'projects_popover'): self.projects_popover.hide()
        selected = {None: "Projects", 'import': 'Import', 0: "Scan", 1: "Build", 2: "Edit"}.get(step)
        for name in self.page_tools:
            tool = self.ribbon_tools[name]
            role = "WorkspaceToolPrimary" if name == selected else "WorkspaceTool"
            if tool.objectName() != role:
                tool.setObjectName(role)
                tool.style().unpolish(tool); tool.style().polish(tool)
        self.build_tools.setVisible(False)      # Cut base, Smooth, Simplify and Line-up are rows in Build's rail now

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.start.mapTo(self, self.start.rect().center())
        center_x, center_y = center.x(), center.y()
        pen = QPen(QColor("#31516a")); pen.setWidth(1)
        painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
        for radius in (70, 130, 195, 265, 345, 435, 535, 645, 765, 895):
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

    def _place_projects_popover(self):
        top = self.page_header.geometry().bottom() + 1
        bottom = self.filmstrip.geometry().top()
        width = min(520, max(460, int(self.width() * .30)))          # room for the thumbnail, the words and Open
        self.projects_popover.setGeometry(0, top, width, max(0, bottom - top))

    def toggle_projects_popover(self):
        if self.projects_popover.isVisible(): self.projects_popover.hide(); return
        self.refresh_projects(); self._place_projects_popover()
        self.projects_popover.show(); self.projects_popover.raise_()

    def _open_clicked(self):
        """Open project: the list of your projects (one place to open from; its "Open project from folder..." is the file dialog).
        No projects yet: the folder dialog straight away."""
        try: has = bool(project_rows(self.projects_dir)) if self.projects_dir else False
        except (OSError, ValueError): has = False
        if not has: self.open_requested.emit(); return
        if not self.projects_popover.isVisible(): self.toggle_projects_popover()

    def _popover_action(self, signal):
        self.projects_popover.hide(); signal.emit()

    def _choose_project(self, path):
        self.projects_popover.hide(); self.project_requested.emit(path)

    def _prepare_project(self, path):
        self.projects_popover.hide(); self.prepare_requested.emit(path)

    def set_project_open(self, is_open, export=None):
        """Build, Edit, Export and the Build subtools need an open project. export: the shell's own Export state
        (something built to save); None follows is_open."""
        is_open = bool(is_open)
        for name in PROJECT_TOOLS + tuple(text for text, _mode, _asset in BUILD_TOOLS):
            on = is_open and (bool(export) if (name == "Export" and export is not None) else True)
            self.ribbon_tools[name].setEnabled(on)
            self.ribbon_tools[name].setToolTip("" if on else "Capture or open a scan first")

    def browse_project_folder(self):
        """A real folder dialog; a scan project is a folder with a data folder in it."""
        self.projects_popover.hide()
        start = self.projects_dir if self.projects_dir and os.path.isdir(self.projects_dir) else os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Open project from folder", start)
        if not path: return
        if not os.path.isdir(os.path.join(path, "data")):
            QMessageBox.information(self, "Not a scan project",
                                    "That folder has no data folder inside, so it is not a scan project. Pick the project's own folder.")
            return
        self.project_requested.emit(path)

    def _toggle_show_all(self):
        self._show_all = not self._show_all; self.refresh_projects()

    def set_active_project(self, active, view_only=False):
        try:                                   # the empty project a plain launch opens is not "your project" until it has a side
            if active and not any(d.is_dir() for d in os.scandir(os.path.join(active, "data"))): active = None
        except OSError: active = None
        built = bool(active) and os.path.isfile(os.path.join(active, "model.ply"))
        if hasattr(self, "_active_project") and (active, view_only, built) == (self._active_project, getattr(self, "_view_only", False), getattr(self, "_built", None)): return
        self._active_project = active; self._view_only = view_only; self._built = built
        self.continue_b.setVisible(bool(active))           # it lands on Build: say what happens there, not "scan"
        self.continue_b.setText("View scanner model" if view_only else "Open model" if built else "Go to Build")
        self.start_heading.setText("Your project" if active else "Start a project")
        self.start_copy.setText("View its scanner models, start a new scan, or open a different project." if view_only else
                                "Open its model, start another scan, or open a different project." if built else
                                "Build a model from this scan, start another, or open a different project." if active
                                else START_COPY)
        self.start_note.setText("Scanner models open read only. Export the side you need." if view_only else
                                "Your scan stays in its project folder while you work." if active
                                else "Build and edit become available after you capture or open a scan.")
        role = "WorkspaceSecondary" if active else "WorkspacePrimary"
        if self.new.objectName() != role:
            self.new.setObjectName(role)
            self.new.style().unpolish(self.new); self.new.style().polish(self.new)
        if active:
            name = project_name(active) or os.path.basename(active)
            self.film_note.setText("Current project: %s" % name)
        else:
            self.film_note.setText("No project selected · Start a scan or open one to begin.")
        self.update()

    def refresh_projects(self):
        self._size_labels = {}; self._thumbs = {}
        while self.project_list.count():
            item = self.project_list.takeAt(0)
            if item.widget(): item.widget().hide(); item.widget().deleteLater()
        try: rows = project_rows(self.projects_dir) if self.projects_dir else []
        except (OSError, ValueError): rows = []
        rows = [r for r in rows if r.get('sides')]          # an empty project (a plain launch makes one) is not listed until it has a side
        self.projects_dropdown.setText(f'Your projects ({len(rows)})  ▾')
        more = len(rows) > CARDS_SHOWN
        shown = rows if (self._show_all or not more) else rows[:CARDS_SHOWN]
        self.count_note.setText(f'Showing the newest {CARDS_SHOWN} of {len(rows)}.' if more and not self._show_all else
                                f'Showing all {len(rows)}.' if more else '')
        self.count_note.setVisible(more)
        self.show_all_b.setText(f'Show newest {CARDS_SHOWN}' if self._show_all else f'Show all {len(rows)}')
        self.show_all_b.setVisible(more)
        for row in shown:
            card = QFrame(); card.setObjectName('WorkspaceProjectCard')
            ch = QHBoxLayout(card); ch.setContentsMargins(12, 10, 12, 10); ch.setSpacing(10)   # words left, Open right: text wraps before the button
            thumb = QLabel('Loading...'); thumb.setFixedSize(92, 62); thumb.setAlignment(Qt.AlignmentFlag.AlignCenter); thumb.setWordWrap(True)
            thumb.setStyleSheet('background:#12161b;border:1px solid #2c343e;border-radius:4px;')      # a model preview comes in when it is ready
            ch.addWidget(self._delete_control(row)); ch.addWidget(thumb); self._thumbs[row['path']] = thumb
            thumb.setProperty('role', 'hint'); self._thumb_pool.submit(self._make_thumb, row)
            card.delete_box = ch.itemAt(0).widget()
            cv = QVBoxLayout(); cv.setSpacing(3); ch.addLayout(cv, 1)
            name = QLabel(row['name'] if row['name'].startswith('Scan · ') or not row['unnamed'] else 'Unnamed scan'); name.setFont(ui_font(13, QFont.Weight.DemiBold)); name.setWordWrap(True); cv.addWidget(name)
            when = datetime.fromtimestamp(row["date"]).strftime("%b %d, %Y")
            first = row['folder'] if row['folder'] != row['name'] else when      # the scanner's own name, to match it on the scanner
            folder_l = QLabel(first); folder_l.setProperty('role','hint'); folder_l.setFont(ui_font(11)); folder_l.setMinimumWidth(0); cv.addWidget(folder_l)
            detail = QLabel(); detail.setProperty('base', f'{row["sides"]} ' + ('side' if row['sides'] == 1 else 'sides') + (' · Models only' if row.get('models_only') else '') + (' · ' + BACKUP_WORDS[row.get('backup', 'unknown')][0] if BACKUP_WORDS[row.get('backup', 'unknown')][0] else ''))
            self._size_labels[row['path']] = detail; self._show_size(row['path'])
            detail.setProperty('role','hint'); detail.setFont(ui_font(11)); cv.addWidget(detail)
            actions = QVBoxLayout(); actions.setSpacing(5)
            open_b = QPushButton('Open'); open_b.setObjectName('WorkspaceProjectOpen'); open_b.setFixedHeight(28)
            open_b.clicked.connect(lambda _=False, path=row['path']: self._choose_project(path))
            actions.addWidget(open_b)
            if needs_preparation(row['path']):
                prepare_b = QPushButton('Prepare'); prepare_b.setFixedHeight(28)
                prepare_b.setToolTip('Prepare imported scan: use the scanner poses so this full import can open in Build')
                prepare_b.clicked.connect(lambda _=False, path=row['path']: self._prepare_project(path))
                actions.addWidget(prepare_b)
            ch.addLayout(actions)
            card.delete_box.confirm_host = cv; cv.addWidget(card.delete_box.confirm); card.delete_box.confirm.hide()
            self.project_list.addWidget(card)
        if not rows:
            empty = QLabel('No local projects yet. Start a scan or import one from the scanner.')
            empty.setWordWrap(True); empty.setProperty('role','hint'); self.project_list.addWidget(empty)
        self.project_list.addStretch(1)
        self._sizes = {p: disk_usage.project_bytes(p) for p in (r['path'] for r in rows)}; self._show_total()
        disk_usage.measure([r['path'] for r in rows], lambda p, n: self._sized.emit(p, n))

    def _delete_control(self, row):
        """A small trash button at the card's left edge; it asks in the card's text area: "Delete 2.7 GB? Not undoable.  Delete  Keep". The open project's is
        off."""
        path = row['path']; box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0)
        ask = QToolButton(); ask.setObjectName('WorkspaceProjectTrash'); ask.setIcon(icon('delete')); ask.setIconSize(QSize(16, 16)); ask.setFixedSize(24, 24)
        ask.setAutoRaise(True); ask.setFocusPolicy(Qt.FocusPolicy.NoFocus); ask.setCursor(Qt.CursorShape.PointingHandCursor); lay.addWidget(ask, 0, Qt.AlignmentFlag.AlignVCenter)
        confirm = QWidget(); cv2 = QVBoxLayout(confirm); cv2.setContentsMargins(0, 2, 0, 0); cv2.setSpacing(4)
        q = QLabel(); q.setStyleSheet('color:#ff8a80;'); q.setFont(ui_font(11)); q.setWordWrap(True); cv2.addWidget(q)
        cl = QHBoxLayout(); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(6); cv2.addLayout(cl)
        yes = QPushButton('Delete'); yes.setObjectName('WorkspaceProjectDeleteYes'); yes.setFixedHeight(24); cl.addWidget(yes)
        keep = QPushButton('Keep'); keep.setFixedHeight(24); cl.addWidget(keep); cl.addStretch(1)
        if self._is_active(path) or path in self._deleting:
            ask.setEnabled(False); ask.setToolTip('This project is open. Open another one first.' if self._is_active(path) else 'Deleting...')
        else: ask.setToolTip('Delete this project and all its scans from disk. It asks first.')
        def arm():
            n = self._sizes.get(path); q.setText('Delete%s? Not undoable.' % ((' ' + disk_usage.nice(n)) if n else ''))
            confirm.show(); ask.setEnabled(False)
        def disarm(): confirm.hide(); ask.setEnabled(True)
        ask.clicked.connect(arm); keep.clicked.connect(disarm)
        yes.clicked.connect(lambda: self._delete_project(path, confirm))
        box.ask, box.confirm = ask, confirm
        return box

    def _is_active(self, path):
        a = getattr(self, '_active_project', None)
        return bool(a) and os.path.realpath(a) == os.path.realpath(path)

    def _delete_project(self, path, box=None):
        if self._is_active(path) or path in self._deleting: return
        self._deleting.add(path)
        if box is not None: box.setEnabled(False)
        lab = self._size_labels.get(path)
        if lab is not None: lab.setText('Deleting...')
        import shutil, threading
        def work():
            try: shutil.rmtree(path); err = ''
            except OSError as e: err = str(e)
            try: self._deleted.emit(path, err)
            except RuntimeError: pass
        threading.Thread(target=work, daemon=True, name='project-delete').start()

    def _on_deleted(self, path, err):
        self._deleting.discard(path)
        if err: QMessageBox.warning(self, 'Delete project', 'Could not delete %s:\n%s' % (os.path.basename(path), err))
        else: self.project_deleted.emit(path)
        self.refresh_projects()

    def _show_size(self, path):
        lab = self._size_labels.get(path)
        if lab is None: return
        try: lab.setText('%s · %s' % (lab.property('base'), disk_usage.nice(disk_usage.project_bytes(path))))
        except RuntimeError: self._size_labels.pop(path, None)       # card deleted by a refresh

    def _make_thumb(self, row):
        try: raw = preview_bytes(row)
        except Exception: raw = None                       # no picture: the card keeps its empty frame
        try: self._thumb_ready.emit(row['path'], raw or ('Too big to preview' if row['model'] else 'No model yet'))
        except RuntimeError: pass                          # the window closed first

    def _on_thumb(self, path, raw):
        label = self._thumbs.get(path)
        if label is None or not raw: return
        if isinstance(raw, str):                           # nothing to draw: say why
            label.setText(raw); label.setStyleSheet(label.styleSheet() + 'color:#7f8b99;font-size:11px;'); return
        pix = QPixmap()
        try:
            if pix.loadFromData(raw): label.setPixmap(pix.scaled(label.size() - QSize(2, 2), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except RuntimeError: self._thumbs.pop(path, None)  # card deleted by a refresh

    def _on_sized(self, path, n):
        if path in self._sizes: self._sizes[path] = n; self._show_total()
        self._show_size(path)

    def _show_total(self):
        known = [n for n in self._sizes.values() if n is not None]
        self.space_note.setVisible(bool(self._sizes))
        self.space_note.setText('All projects use %s on disk%s.' % (disk_usage.nice(sum(known)), '' if len(known) == len(self._sizes) else ' so far'))
