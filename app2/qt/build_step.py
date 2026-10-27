# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Build step: the MIRACO app's processing flow on the staged engine (scanner/stages). Every stage in a list on the
left (Raw frames, Side, Line-up, Fused, Meshed, Scan here), its result in the 3D view, the tools with their sliders on
the right (Apply, or Detect then Delete), YOLO Yoink, Rectangle and Lasso deletes, Applied lists whose steps can be
edited, turned off or removed, undo / redo / History, Compare and Versions.

Threads: the GUI never waits. Engine calls go to one StageHost ("stages-edit" thread, in-process Engine); stage results
are read by a ViewLoader ("stages-load"); runs are a `cli.py run` subprocess through the scan session's BuildRunner
(session.build(StagesRun(...))), so the Scan step's Build button and this page share one path and one lock.

The shell needs from this page: attach(session), set_ended(), start_build() (Ctrl+B = Rebuild the rest),
page_key(seq) for the window-wide keys (Up, Down, Del, H, R, L, Esc), busy, shutdown(), and show_built() (the Scan
step's Build combined model is done: the page opens on the built model).

The view's label (top left) says exactly what it shows: "Raw frames · side 3 of 6", "Side 3 of 6 · platter cut",
"Line-up · 6 sides, a color per side", "Fused · 6 sides", "Built model · 6 sides", "Finished for print · 250k
triangles", "Scan here · 3 places to scan"."""
import math, os, time
TRACE = bool(os.environ.get("POINTYOINK_TRACE"))       # developer: print how long host calls, status and view loads take
import numpy as np
from PySide6.QtCore import Qt, QTimer, QEvent, QFileSystemWatcher, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QScrollArea, QStackedWidget, QSizePolicy,
                               QApplication)
from app2.qt.theme import T, ui_font
from app2.qt.widgets import button, Scrim, KeysSheet, fit_overlay_in_parent
from app2 import library_dir
from app2.qt import build_words as W
from app2.qt.stage_bridge import StageBridge
from app2.qt.stage_list import InfoBlock, StageList, Pager
from app2.qt.tool_panel import EXTRA_QSS, AppliedList, big_button, CurrentStack
from app2.qt.stage_panels import StagePanel
from app2.qt.lineup_panel import LineupPanel
from app2.qt.report_panel import ReportPanel
from app2.qt.build_view import BuildViewPanel
from app2.qt.onetap import OneTapDialog
from app2.qt.history import HistoryPopover, VersionsPopover, BuildLogPopover
from app2.builder.stagehost import StageHost, StagesRun, tool_words, mins, running_elsewhere, elsewhere_words
from app2.builder.orient_math import bottom_rotation, front_rotation

STAGE_IDS = ["raw", "side", "lineup", "fused", "meshed", "report"]
LABEL = {s["id"]: s["row"] for s in W.STAGES}
TABS_WORDS = {t["name"]: t for t in W.PANELS["lineup"]["tabs"]}


def fix_words(o, pos, applied=False):
    """A Line-up flip, point match or turn op in plain words (the Flips / Nudges lists, or the Applied list)."""
    n = pos.get(o.get("side"), "?")
    if o.get("tool") == "turn":
        if applied: return "Turn side %s (%g deg%s)" % (n, o.get("deg", 0), ", kept as placed" if o.get("keep") else "")
        return TABS_WORDS["Nudge"]["turn_row_keep" if o.get("keep") else "turn_row"].format(n=n, deg="%g" % o.get("deg", 0))
    if o.get("tool") == "match":
        k = len(o.get("pairs") or []); to = pos.get(o.get("to"), "?")
        return ("Match side %s to side %s (%d points)" if applied else "Side %s on side %s, %d points") % (n, to, k)
    ax = o.get("axis", "auto"); ax = "auto: %s" % o["chose"] if ax == "auto" and o.get("chose") else ax
    return ("Flip side %s (%s axis)" if applied else "Side %s, %s axis") % (n, ax)


TOOL_TAB ={"isolation": "Isolation", "overlap": "Overlap Detection", "smooth": "Smooth", "simplify": "Simplify",
            "fill_holes": "Hole Filling", "slivers": "Remove slivers", "finish": "Finish for print", "fin": "Add fin", "slice_gap": "Cut gap", "drill": "Cut shape"}
TOOL_LABEL = {"overlap": "Overlaps", "fill_holes": "Fill holes", "isolation": "Loose pieces"}   # shown text where it differs from the tab key
FINISH = "Finish for print"                # the Mesh panel's block (build_words PANELS["meshed"]["blocks"])


def hole_mm(edges, extent_mm, depth, spacing_mm=0.0):
    """About how far across (mm) a round hole of `edges` rim edges is. PyMeshLab's screened Poisson follows the input's
    sample spacing and never goes finer than its grid (extent x 1.1 / 2^depth): its median edge came out at the input
    spacing at depths 7 and 8 on a 0.7 mm sphere (1x and 2x the cell), so an edge is about max(cell, spacing) and N rim
    edges close a round hole about N x edge / pi across."""
    from stages import params as PR
    cell = float(extent_mm) * PR.POISSON_SCALE / 2 ** int(depth)
    return float(edges) * max(cell, float(spacing_mm or 0)) / math.pi
DONE_STATES = ("fresh", "adopted", "warn", "check", "done", "cached", "retry", "off", "n/a")
FUSED_NAMES = {"tsdf": "Standard", "points": "Points", "plus": "Plus"}
MESH_NAMES = {"auto": "Auto", "as_fused": "As scanned", "poisson": "Watertight", "bpa": "Joined points"}
# why YOLO Yoink, Apply and Rebuild are off (under the YOLO Yoink button); the export fall-back's warning
OFF = dict(no_sides="No sides yet. Scan a side on the Scan step, then build it here.",
           ended="The scan session stopped. Open this project again from Projects to keep building.",
           elsewhere="This project is being built somewhere else. Tools come back when that is done.")
EXPORT_STALE = "Meshed is out of date, so this exports the last built model.ply. Rebuild to include your latest changes."
EXPORT_REBUILD = "Rebuild Meshed before exporting. The saved model does not include your current edits."
MANUAL_SIDE = W.CUT["all_sides"]
EXPORT_CHECKING = "Checking the built model..."


def tri_text(n):
    """250000 -> "250k", 1250000 -> "1.25M", 900 -> "900"."""
    n = int(n)
    if n >= 1_000_000: return "%gM" % round(n / 1e6, 2)
    if n >= 1000: return "%dk" % round(n / 1000.0)
    return str(n)


def join_words(xs):
    xs = list(xs)
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def run_overview(prog):
    """("Step 6 of 14, about 2 min left", "Running: Side, Line-up, Fused...") from the run's plan: only the steps it does, not the reused ones."""
    plan = (getattr(prog, "plan", None) or {}) if prog else {}
    todo = [n["id"] for n in plan.get("nodes", []) if n.get("status") in ("stale", "waiting") or n.get("retry")]
    if not todo: return "", ""
    from app2.builder.stagehost import mins, stage_word
    nodes = getattr(prog, "nodes", None) or {}
    done = sum(1 for i in todo if (nodes.get(i) or {}).get("state") in ("done", "warn", "check", "cached", "off"))
    active = sum(1 for i in todo if (nodes.get(i) or {}).get("state") == "start")
    k = min(len(todo), done + 1)
    if active > 1: text = W.PROGRESS["overall_many"].format(done=done, n=len(todo), active=active)
    else: text = W.PROGRESS["overall"].format(k=k, n=len(todo))
    frac = float(getattr(prog, "frac", 0) or 0)                      # side by side, the plan's serial estimate runs long: go by the pace so far
    left = prog.elapsed() * (1 - frac) / frac if .02 < frac < 1 else float(plan.get("est_s") or 0) - prog.elapsed()
    if plan.get("est_s") and left > 5: text += ", " + W.PROGRESS["left"].format(eta=mins(left))
    stages = []
    for i in todo:
        w = stage_word(i)
        if w not in stages and w != i: stages.append(w)
    names = stages[0] if len(stages) == 1 else (", ".join(stages[:-1]) + " and " + stages[-1]) if stages else ""
    return text, (W.PROGRESS["doing"].format(stages=names) if names else "")


class BuildKeysSheet(KeysSheet):
    KEYS = W.KEYS

    def __init__(self, parent=None):
        super().__init__(parent)
        t = self.findChild(QLabel)
        if t is not None: t.setText("Keys on the Build step")


class BuildStepPage(QWidget):
    timelapse_requested = Signal()
    coverage_result = Signal(object)
    closed_counted = Signal(object)     # (key, open edges) from the open-edges thread, for the Export sheet
    project_sized = Signal(str, object) # (project path, bytes) from the disk-usage thread, for the info block
    message = Signal(str, str)          # (text, level) of every notice: the shell shows it on the page the user is on
    status_changed = Signal()           # a new engine status (the shell's Export turns on when a built model is there)
    edit_requested = Signal()           # the Clean up row under Meshed: the shell opens the Edit page

    def __init__(self, bridge, gl=False, parent=None):
        super().__init__(parent)
        self.bridge = bridge; self.gl = gl; self.session = None; self.state = None; self.prog = None
        self.host = None; self.host_root = None; self.snap = None; self.loader = None
        self.sb = None; self.engine_factory = None                # sb: the open host's own bridge (dropped with it)
        self.vb = StageBridge(self)                                     # the view loader's bridge (lives with the page)
        self._shut = False; self._applied_ids = []
        self._run_watchdog=QTimer(self);self._run_watchdog.setSingleShot(True)
        self._run_watchdog.timeout.connect(self._run_check_timeout);self._run_watchdog_backend=None
        self.stage = "fused"; self.side_k = 1; self.tickets = {}; self.run_mine = None; self._was_building = False
        self.detected = None; self.editing = None; self.comparing = False; self.peek = False; self._b_down = False
        self.view_id = 0; self.shown = {}; self.before_shown = {}; self.sel_path = None; self.notice_until = 0
        self._pts = None; self._flip_ask = 0
        self.lineup = {}; self.last_changed = None; self.rebuild_text = None; self.undo_text = None; self.stopping = False
        self.side_note_shown = False; self.busy_call = None; self.pending_view = None; self.before_kind = "tool"; self.change_baseline = None
        self.colour_mode = {}; self.platter_cut = None; self.platter_share = 0.0; self._cut_msg_on = False
        self.want_built = False; self.published = None      # show_built(): open on the model; model.ply when Meshed is not built
        self._ended = False; self._export_after_status = False
        self.ask_save = self._ask_save                              # Export STL's file dialog (a test puts its own)
        self.setStyleSheet(EXTRA_QSS); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        outer = QHBoxLayout(self); outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(14)
        # ---- left
        self.left = QWidget(); lv = QVBoxLayout(self.left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(8)
        self.info = InfoBlock(); self.stages = StageList(); self.pager = Pager()
        self.rail_scroll = QScrollArea(); self.rail_scroll.setWidgetResizable(True); self.rail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.rail_scroll.setWidget(self.stages)
        self.rail_scroll.setStyleSheet("QScrollArea {background: transparent;} QScrollArea > QWidget > QWidget {background: transparent;}"
                                       "QScrollBar:vertical {background: transparent; width: 10px; margin: 2px 0 2px 4px; border: none;}"
                                       "QScrollBar::handle:vertical {background: #3a434e; border-radius: 3px; min-height: 30px;}"
                                       "QScrollBar::handle:vertical:hover {background: #4b5663;}"
                                       "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {height: 0;}"
                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {background: none;}")
        sb = self.rail_scroll.verticalScrollBar()                # its room stays reserved: the rail keeps one width with or without the bar
        keep = lambda *_: self.rail_scroll.setViewportMargins(0, 0, 0 if sb.maximum() > 0 else sb.sizeHint().width(), 0)
        sb.rangeChanged.connect(keep); keep()
        lv.addWidget(self.info); lv.addWidget(self.rail_scroll, 1); lv.addWidget(self.pager)       # the rail scrolls up and down when it is taller than the window
        outer.addWidget(self.left)
        # ---- centre
        self.vp = BuildViewPanel(gl); outer.addWidget(self.vp, 1)
        self.vp.card.words = lambda nid: self._what(nid)[0]          # the log says "Side 1: Cut base", not "Side 1 platter"
        # ---- right
        self.right = QWidget(); rv = QVBoxLayout(self.right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(8)
        self.onetap = big_button(W.ONETAP["button"], "onetap", 50); self.onetap.setToolTip(W.ONETAP["tip"] + " (Ctrl+Shift+B)")
        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QSize
        from pathlib import Path
        self.onetap.setIcon(QIcon(str(Path(__file__).with_name("assets") / "yoink-arrow.svg")))
        self.onetap.setIconSize(QSize(30, 24)); self.onetap.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        rv.addWidget(self.onetap)
        from PySide6.QtWidgets import QMenu
        self.recipe_b = button("", "quiet", W.RECIPE["tip"]); self.recipe_b.setObjectName("RecipeChip")   # what Build uses, always in sight
        self.recipe_menu = QMenu(self.recipe_b); self.recipe_b.setMenu(self.recipe_menu); self.recipe_menu.aboutToShow.connect(self._fill_recipe_menu)
        # 2026-10-25: a real button, outlined in the YOLO green, not a dim line of text ("this recipe is actually very hidden")
        self.recipe_b.setStyleSheet("QPushButton#RecipeChip {background:#1d2b22;color:#cfeeda;border:1px solid #5fbf7a;border-radius:6px;padding:4px 12px;text-align:center;}"
                                    " QPushButton#RecipeChip:hover {background:#24382c;} QPushButton#RecipeChip::menu-indicator {image:none;width:0;}"
                                    " QPushButton#RecipeChip:disabled {color:#77818f;border-color:#3a4a40;}")
        self.recipe_b.setFixedHeight(32)
        rv.addWidget(self.recipe_b)
        self.off_note = QLabel(""); self.off_note.setWordWrap(True); self.off_note.setProperty("role", "hint"); self.off_note.hide()
        self.off_note.setObjectName("ToolsOffNote"); rv.addWidget(self.off_note)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); iv = QVBoxLayout(inner); iv.setContentsMargins(0, 0, 4, 0); iv.setSpacing(8); iv.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.panels = CurrentStack(); self.panel = {}
        extra = self._method_chips()
        for sid in ("raw", "side", "fused", "meshed"):
            p = StagePanel(sid, "mesh", method_extra=extra if sid == "fused" else ()); self.panel[sid] = p; self.panels.addWidget(p)
            p.button_clicked.connect(self.on_button); p.second_clicked.connect(self.on_second); p.moved.connect(self.on_moved)
            p.extra_clicked.connect(self.on_extra); p.sides_delete.connect(self.delete_sides); p.tab_changed.connect(lambda _n: self._after_tab())
        self._build_spot_ui(); self._build_orient_ui(); self._build_cut_ui(); self._build_clone_ui(); self._build_fix_ui(); self._build_holes_ui(); self._build_mirror_ui()
        pdc = next((c for c in self.panel["fused"].bodies["Fusion"].controls if c.c.get("path") == "fused.point_distance"), None)
        if pdc is not None: pdc.reserve_warn(W.tab_of("fused", "Fusion")["low_memory_warn"])    # comes and goes as it slides
        self.panel["lineup"] = LineupPanel(); self.panels.addWidget(self.panel["lineup"])
        self.panel["report"] = ReportPanel(); self.panels.addWidget(self.panel["report"])
        self.mesh_alias=QLabel("Same as Fused (As scanned). Pick Watertight for a closed surface.")
        self.mesh_alias.setWordWrap(True);self.mesh_alias.hide();iv.addWidget(self.mesh_alias)
        self.job_state=QLabel("");self.job_state.setWordWrap(True);self.job_state.hide();iv.addWidget(self.job_state)
        self.coverage_state=QLabel(""); self.coverage_state.setWordWrap(True); self.coverage_state.hide(); iv.addWidget(self.coverage_state)
        from PySide6.QtWidgets import QProgressBar
        self.job_progress=QProgressBar();self.job_progress.setRange(0,0);self.job_progress.setTextVisible(False);self.job_progress.hide();iv.addWidget(self.job_progress)
        self._one_shot_jobs={}
        self.panels.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        iv.addWidget(self.panels, 0, Qt.AlignmentFlag.AlignTop); self.applied = AppliedList(); iv.addWidget(self.applied, 1)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); self.scroll.setWidget(inner); rv.addWidget(self.scroll, 1)
        from app2.qt.widgets import quiet_scrollbar
        quiet_scrollbar(self.scroll.verticalScrollBar())
        self.utilities = QWidget(); utility_layout = QVBoxLayout(self.utilities)
        utility_layout.setContentsMargins(0, 0, 0, 0); utility_layout.setSpacing(4)
        foot = QHBoxLayout(); foot.setSpacing(4)
        undo_row = QHBoxLayout(); undo_row.setSpacing(4)
        self.hist_b = button(W.BOTTOM["history"], "quiet", "Every change, newest first"); self.vers_b = button("Versions", "quiet", "Stored results of this stage")
        self.undo_b = button("↶", "quiet", W.BOTTOM["nothing_undo"]); self.redo_b = button("↷", "quiet", W.BOTTOM["nothing_redo"])
        for b in (self.undo_b, self.redo_b): b.setFont(ui_font(22)); b.setFixedSize(50, 40)
        self.log_b = button("Log", "quiet", "The last build's full output, for when something goes wrong")
        self._log_page_active = False
        for b in (self.log_b, self.hist_b, self.vers_b, self.undo_b, self.redo_b):
            b.setProperty("kind", "footer"); b.setFixedHeight(34); b.setFont(ui_font(12))
            b.setStyleSheet('QPushButton {background:#282e36;border:1px solid #414b58;border-radius:6px;padding:0 6px;} QPushButton:hover {background:#344254;} QPushButton:disabled {color:#77818f;background:#20252c;border-color:#343c46;}')
            (undo_row if b in (self.undo_b, self.redo_b) else foot).addWidget(b, 1)
        self.undo_b.setAccessibleName("Undo"); self.redo_b.setAccessibleName("Redo")
        from app2.qt.icons import icon
        from PySide6.QtCore import QSize
        for b, name in ((self.log_b, "captures"), (self.hist_b, "history"), (self.vers_b, "save-version"), (self.undo_b, "undo"), (self.redo_b, "redo")):
            b.setIcon(icon(name)); b.setIconSize(QSize(16,16))
        self.undo_b.setText("Undo"); self.redo_b.setText("Redo")
        for b in (self.undo_b, self.redo_b): b.setMinimumWidth(0); b.setMaximumWidth(16777215)
        utility_layout.addLayout(foot); utility_layout.addLayout(undo_row)
        lv.addWidget(self.utilities)
        outer.addWidget(self.right)
        # ---- sheets
        self.hist_pop = HistoryPopover(self); self.vers_pop = VersionsPopover(self); self.log_pop = BuildLogPopover(self)
        # the last build (any backend, the Scan step's too), as the one-shot Build page showed it
        lp_ = self.log_pop; self.bar, self.pct, self.result, self.engine, self.log = lp_.bar, lp_.pct, lp_.result, lp_.engine, lp_.log
        self.scrim = Scrim(self); self.scrim.hide(); self.onetap_dlg = OneTapDialog(self)
        # ---- wiring
        self.stages.picked.connect(self._pick_stage); self.pager.step.connect(self.step_side)
        self.onetap.clicked.connect(self.open_onetap)
        self.onetap_dlg.preset_changed.connect(self._ask_onetap); self.onetap_dlg.cancel.connect(self.close_onetap)
        self.onetap_dlg.size_changed.connect(self._center_onetap)
        self.onetap_dlg.custom_changed.connect(self._ask_custom_onetap)
        self.onetap_dlg.custom_ok.connect(self.do_custom_onetap)
        self.onetap_dlg.ok.connect(self.do_onetap); self.scrim.clicked.connect(self.close_onetap)
        self.undo_b.clicked.connect(lambda: self.undo(-1)); self.redo_b.clicked.connect(lambda: self.undo(1))
        self.hist_b.clicked.connect(self.open_history); self.vers_b.clicked.connect(self.open_versions)
        self.log_b.clicked.connect(lambda: self.log_pop.close_popover() if self.log_pop.isVisible() else
                                   self.log_pop.place(self.log_b.mapToGlobal(self.log_b.rect().topLeft()), max(160, self.height() - 60)))
        bridge.build_line.connect(self.log.appendPlainText)
        self.log.textChanged.connect(self._update_log_button)
        self._update_log_button()
        self.hist_pop.go.connect(lambda s: (self.hist_pop.close_popover(), self.undo(s)))
        self.vers_pop.keep.connect(self.do_pin); self.vers_pop.compare.connect(self.compare_version)
        self.applied.edit.connect(self.edit_op); self.applied.toggle.connect(lambda i: self.row_call("toggle_op", i))
        self.applied.remove.connect(lambda i: self.row_call("drop_op", i))
        self.applied.pending_changed.connect(self._plan_pending_edits)
        self.applied.apply_pending.connect(self._confirm_pending_edits)
        vp = self.vp
        vp.compare_clicked.connect(self.toggle_compare); vp.home_clicked.connect(self.home); vp.home_set_clicked.connect(self.set_home_view); vp.home_reset_clicked.connect(self.reset_home_view); vp.stop_clicked.connect(self.stop)
        vp.rebuild_clicked.connect(self.rebuild_rest); vp.pause_clicked.connect(self.pause_build); vp.tool_picked.connect(self.set_tool); vp.through_toggled.connect(self.set_through)
        vp.delete_clicked.connect(self.delete_selection); vp.invert_clicked.connect(self.invert); vp.clear_clicked.connect(self.clear_selection)
        vp.before_picked.connect(self.pick_before); vp.before_version_picked.connect(self.compare_version); vp.colour_picked.connect(self.pick_colour)
        if gl:
            vp.main.view.selection_changed.connect(self.on_selection); vp.main.view.camera_moved.connect(lambda: self._sync_cam(vp.main, vp.before))
            vp.before.view.camera_moved.connect(lambda: self._sync_cam(vp.before, vp.main))
        lp = self.panel["lineup"]
        lp.apply_sides.connect(lambda ch, lp=lp: self.apply_sides(ch, lp.pending_hints() if lp.hints_dirty() else None)); lp.show_side.connect(self.show_side_toggle); lp.flip_apply.connect(self.do_flip)
        lp.flip_preview.connect(self.preview_flip); lp.nudge_preview.connect(self.preview_nudge); lp.nudge_apply.connect(self.do_nudge)
        lp.remove_op.connect(self._lineup_remove); lp.settings_apply.connect(lambda: self.on_button("Settings"))
        lp.refit.connect(lambda: self.start_run(StagesRun(to="lineup.adjust", force=("lineup.pairs",), publish=False)))
        lp.tab_changed.connect(lambda _n: self._after_tab()); lp.picked_side.connect(lambda _s: self._lineup_colours())
        lp.points_changed.connect(self._points_view); lp.points_apply.connect(self.do_points); lp.auto_points.connect(self.suggest_dots); lp.checked_changed.connect(self._save_checked)
        if gl:
            vp.pa.view.surface_picked.connect(lambda w: self._points_pick_pane(0, w)); vp.pb.view.surface_picked.connect(lambda w: self._points_pick_pane(1, w))
            vp.main.view.surface_picked.connect(self._surface_pick); vp.main.view.pick_missed.connect(self._pick_missed)
            vp.main.view.selected_nothing.connect(lambda tool: self.notice(W.NOTICE["nothing_click" if tool == "piece" else "nothing_box"], level="info"))
            vp.main.view.marker_clicked.connect(self._goto_side)
            vp.main.view.ring_turn.connect(lambda k, deg, done: self._nudge_on() and self.panel["lineup"].ring_turn(k, deg, done))
            vp.main.view.arrow_move.connect(lambda k, mm, done: self._nudge_on() and self.panel["lineup"].arrow_move(k, mm, done))
            vp.main.view.arrow_move.connect(self._spot_arrow_move)
        rp = self.panel["report"]
        rp.place_clicked.connect(self.turn_to_place); rp.scan_new_side.connect(self.scan_new_side); rp.settings_apply.connect(lambda: self.on_button("Settings"))
        rp.tab_changed.connect(lambda _n: self._after_tab())
        self.vb.stage_view.connect(self.on_view)
        bridge.state.connect(self.on_state); bridge.build.connect(self.on_build)
        from app2.qt.fin_tool import FinTool
        self.fin = FinTool(self)
        from app2.qt.slice_tool import SliceTool
        self.slice_tool = SliceTool(self)
        from app2.qt.drill_tool import DrillTool
        self.drill = DrillTool(self)
        self._keys()
        self.coverage_result.connect(self._coverage_finished)
        self.clock = QTimer(self); self.clock.setInterval(1000); self.clock.timeout.connect(self._tick); self.clock.start()
        self.poll = QTimer(self); self.poll.setInterval(5000); self.poll.timeout.connect(self._poll); self.poll.start()
        self.debounce = QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(300)
        self.debounce.timeout.connect(lambda: self.host.status() if (self.host and not self._shut) else None)
        self.watch = QFileSystemWatcher(self); self.watch.fileChanged.connect(lambda _p: (self.debounce.start(), self._rewatch()))
        QApplication.instance().installEventFilter(self)
        self.stages.set_current(self._rail_current(self.stage)); self._show_panel(self.stage)
        self._refresh_all()

    @staticmethod
    def _method_chips():
        """A Fusion Method chip for every registered fused method beyond Standard and Points (R10: no help text yet)."""
        out = []
        try:
            from stages import params as PR
            from stages.methods import get_method
            for m in PR.METHODS["fused"]:
                if m in FUSED_NAMES: continue
                try: lab = get_method("fused", m).label or m
                except Exception: lab = m
                out.append((m, lab, ""))
        except Exception: pass
        return out

    # ================================================================ shell hooks
    def set_log_page_active(self, active):
        self._log_page_active = bool(active)
        if not active: self.log_pop.close_popover()
        self._update_log_button()

    def _update_log_button(self):
        has_output = (not self.log.document().isEmpty() or
                      bool(self.state and self.state.building) or
                      bool(self.result.text().strip()))
        self.log_b.setVisible(self._log_page_active and has_output)

    def attach(self, session):
        BuildStepPage._cancel_run_check(self)
        self._cancel_memory_choice()
        self._cancel_measurements()
        self._coverage_attempted = set()
        colours = getattr(self, "colour_mode", {})
        for stage, mode in list(colours.items()):
            if mode in ("scan_guide", "scan_guide_handheld"): colours.pop(stage)
        self._scan_guides = {}; self._scan_guide_pending = set()
        self._reset_orientation_prompt()
        self._clear_compare_change()
        self.session = session; self.state = None; self.prog = None; self.run_mine = None; self._ended = False
        self._export_after_status = False; self.close_onetap(); self.close_export()
        self.log.clear(); self.result.clear(); self._update_log_button()
        self.want_built = False; self.published = None
        self._spots = {}; self._spots_built = set(); self._cspots = {"fused": [], "meshed": []}; self._orient_spots = []
        self._drop_host(); self.snap = None; self._refresh_all()

    def set_ended(self):
        self.session = None; self._ended = True; self.close_onetap(); self._refresh_tools()

    @property
    def busy(self):
        return bool((self.host is not None and self.host.busy) or (self.state is not None and self.state.building and self.run_mine is not None))

    def shutdown(self, timeout=3.0):
        """The window is closing: no host is made again (the session still sends states while it saves), no timers,
        tools off. Waits (up to timeout) only for an engine call that is running, so a history.json write finishes."""
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
            if getattr(self, "_build_focus_connected", False):
                app.focusChanged.disconnect(self._shortcut_focus_changed)
                self._build_focus_connected = False
        BuildStepPage._cancel_run_check(self)
        self._cancel_memory_choice()
        self._cancel_measurements()
        self._reset_orientation_prompt()
        self._shut = True
        for t in (self.clock, self.poll, self.debounce): t.stop()
        busy = self.host is not None and self.host.busy
        self._drop_host(timeout if busy else 0)
        self.vb.close()
        if self.loader is not None: self.loader.close(0); self.loader = None
        self._refresh_tools()

    def _drop_host(self, timeout=0):
        """Forget the open project's host. Nothing it still says reaches the page: its bridge is disconnected, and a call
        still running finishes on its own (the GUI does not wait, unless shutdown asks)."""
        if self.sb is not None: self.sb.close()
        if self.host is not None:
            h = self.host; self.host = None; self.host_root = None
            try: h.close(timeout)
            except Exception: pass
        if self.sb is not None:
            for sig in (self.sb.stages_status, self.sb.stages_result, self.sb.stages_busy):
                try: sig.disconnect()
                except (RuntimeError, TypeError): pass
            self.sb.deleteLater(); self.sb = None
        if self.loader is not None: self.loader.clear()
        self.tickets = {}; self.detected = None; self.editing = None; self.shown = {}; self.before_shown = {}; self.busy_call = None
        getattr(self,"_one_shot_jobs",{}).clear()
        if hasattr(self,"job_progress"): self.job_progress.hide();self.job_state.clear();self.job_state.hide()
        self.colour_mode.pop("report",None)
        for f in self.watch.files(): self.watch.removePath(f)

    def start_build(self): self.rebuild_rest()

    def show_built(self):
        """The Scan step's Build combined model is done: open on the built model (Meshed, where Finish for print and
        Export STL are) once the engine's status is in. When the Meshed stage is not built (a one-shot build from the
        Scan step), the published model.ply is shown, read only."""
        self.want_built = True
        if self.snap is not None and self.host is not None and not self._shut: self.host.status()
        elif self.snap is not None: self._open_built()

    def _scanner_model(self, side):
        """A side the scanner made carries its own model of that scan: shown on Raw frames until our fusion exists."""
        d = os.path.join(self.host_root or "", "data", side["id"])
        for name in ("fuse_mesh_rgb.ply", "fuse_mesh.ply", "fuse.ply"):
            if os.path.exists(os.path.join(d, name)): return os.path.join(d, name)
        return None

    def _never_built(self):
        s = self.snap
        return s is not None and not (s.out(s.final("fused")) or s.out(s.final("meshed")) or self._has_model())

    def _has_model(self):
        return bool(self.host_root) and os.path.exists(os.path.join(self.host_root, "model.ply"))

    def _open_built(self):
        self.want_built = False; s = self.snap
        if self.notice_until: self.notice_until = 0; self.vp.notice.show_text(None)     # "a build ... is running": over
        n = s.node(s.final("meshed")) or s.node("meshed") or {}
        m = os.path.join(self.host_root or "", "model.ply")
        self.published = m if (n.get("status") not in DONE_STATES and self.host_root and os.path.exists(m)) else None
        self.show_stage("meshed", keep_published=True)
        QTimer.singleShot(0, self._finish_in_sight)

    def _finish_in_sight(self):
        """Finish for print (and its Export STL) scrolled into sight: the next thing to do with a built model."""
        b = self.panel["meshed"].bodies.get(FINISH)
        if b is None or self.stage != "meshed": return
        sb = self.scroll.verticalScrollBar(); y = b.mapTo(self.scroll.widget(), b.rect().topLeft()).y()
        sb.setValue(min(sb.maximum(), max(0, y - 8)))

    def page_key(self, seq):
        """The window-wide keys, passed on by the shell while this page is shown."""
        if hasattr(self, "edit_confirm") and self.edit_confirm.isVisible():
            if seq == "Esc": self._close_edit_confirm()
            return
        if self.onetap_dlg.isVisible():
            if seq == "Esc": self.close_onetap()
            return
        if seq == "Up": self.step_stage(-1)
        elif seq == "Down": self.step_stage(1)
        elif seq == "Del": self.delete_selection()
        elif seq == "H": self.home()
        elif seq == "R": self.set_tool("orbit" if self.tool == "rect" else "rect")
        elif seq == "L": self.set_tool("orbit" if self.tool == "lasso" else "lasso")
        elif seq == "Esc": self.escape()

    def _keys(self):
        from PySide6.QtWidgets import QApplication
        self._build_shortcuts = []
        def add(seq, fn):
            s = QShortcut(QKeySequence(seq), self); s.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut); s.activated.connect(lambda: fn() if not (hasattr(self, "edit_confirm") and self.edit_confirm.isVisible()) else None); self._build_shortcuts.append(s)
        QApplication.instance().focusChanged.connect(self._shortcut_focus_changed)
        self._build_focus_connected = True
        add("Left", lambda: self.step_side(-1)); add("Right", lambda: self.step_side(1))
        add("Ctrl+Return", self.main_button); add("Ctrl+Enter", self.main_button)
        add("Ctrl+Z", lambda: self.undo(-1)); add("Ctrl+Y", lambda: self.undo(1)); add("Ctrl+Shift+Z", lambda: self.undo(1))
        add("T", lambda: self.vp.through.toggle() if getattr(self.vp, "edit_workspace", False) else None); add("C", self.toggle_compare); add("Ctrl+Shift+B", self.open_onetap)

    def _shortcut_focus_changed(self, old, widget):
        from PySide6.QtWidgets import QLineEdit, QAbstractSpinBox
        enabled = not isinstance(widget, (QLineEdit, QAbstractSpinBox))
        for shortcut in self._build_shortcuts: shortcut.setEnabled(enabled)

    def showEvent(self, e):
        super().showEvent(e); self.setFocus()
        if self._shut: return
        self._ensure_host()
        if self.host is not None: self.host.status()
        QTimer.singleShot(0, self._layout_sizes)

    def eventFilter(self, obj, e):
        """B held = see Before (a shortcut gives no release)."""
        from PySide6.QtWidgets import QLineEdit, QAbstractSpinBox
        if isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox)): return super().eventFilter(obj, e)
        t = e.type()
        if t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and e.key() == Qt.Key.Key_B and not e.isAutoRepeat() \
                and not (e.modifiers() & ~Qt.KeyboardModifier.KeypadModifier) and self.isVisible() and self.window().isActiveWindow():
            down = t == QEvent.Type.KeyPress
            if down != self._b_down: self._b_down = down; self.set_peek(down)
            return True
        return super().eventFilter(obj, e)

    # ================================================================ engine host
    def _ensure_host(self, force=False):
        """The open project's engine host. Made when the page shows, or (force) before that, so Export knows the built
        model before Build has been opened."""
        s = self.state
        root = s.project if s is not None else None
        if not root or not (force or self.isVisible()) or self._shut: return
        if self.host is not None and self.host_root == root: return
        self._drop_host(); self.host_root = root
        sb = self.sb = StageBridge(self)
        sb.stages_status.connect(self.on_status); sb.stages_result.connect(self.on_result); sb.stages_busy.connect(self.on_busy)
        self.host = StageHost(root, sb, engine_factory=self.engine_factory)
        if self.snap is None: self.vp.set_empty(W.VIEW["opening"])     # the first status takes a while on a big scan: not a blank page
        if self.gl and self.loader is None:
            from app2.builder.view_data import ViewLoader
            self.loader = ViewLoader(self.vb)
        self._rewatch()

    def _rewatch(self):
        if not self.host_root: return
        for f in ("history.json", "recipe.json"):
            p = os.path.join(self.host_root, "stages", f)
            if os.path.exists(p) and p not in self.watch.files(): self.watch.addPath(p)

    def call(self, name, *a, then=None, fail=None, **kw):
        if self._shut: return None
        if self.host is None: self.notice(W.NOTICE["no_sides"]); return None
        if name in ("set", "h_apply", "h_apply_custom", "h_mask_delete", "h_history_walk", "preset", "flip", "match", "add_op", "set_op", "drop_op", "toggle_op", "undo", "redo", "clear_ops", "delete", "include", "leave_out"):
            self._cancel_measurements()
        t=self.host.call(name,*a,**kw);self.tickets[t]=(name,then,fail)
        if name in ("h_quality","h_scan_guide","h_fill_preview","h_detect","h_side_detect"):
            purpose={"h_scan_guide":"Finding useful scanner positions and spots still needing frames.","h_quality":"Measuring scan quality. Finding spots that need more frames.","h_fill_preview":"Making the patch preview.","h_detect":"Checking the surface.","h_side_detect":"Checking this side."}[name]
            self._one_shot_jobs[t]=(purpose,time.monotonic())
            self.job_state.setText(purpose);self.job_state.show();self.job_progress.show();self._refresh_progress()
        return t

    def on_result(self, r):
        name,then,fail=self.tickets.pop(r["ticket"],(r["name"],None,None))
        job=getattr(self,"_one_shot_jobs",{}).pop(r["ticket"],None)
        if job:
            self.job_progress.setVisible(bool(self._one_shot_jobs))
            self.job_state.setText((next(iter(self._one_shot_jobs.values()))[0]) if self._one_shot_jobs else
                ("Done: " if r["ok"] else "Failed: ")+job[0]+" %.1f s" % (time.monotonic()-job[1]))
            self._refresh_progress()
        if TRACE: print("trace call %s %.2fs" % (r["name"], r.get("secs") or 0), flush=True)
        if r["name"] == "open" and not r["ok"]:
            if self._export_after_status: self._export_after_status = False; self.tell(W.EXPORT["failed"].format(why=r["error"])); return
            if not self._needs_rebuild_first(r["error"]): self.notice(W.NOTICE["engine_error"].format(why=r["error"]))
            return
        if r["ok"]:
            if then: then(r["value"])
        else:
            if fail: fail(r["error"])
            elif r["error"] != "cancelled" and not self._needs_rebuild_first(r["error"]): self.notice(W.NOTICE["engine_error"].format(why=str(r["error"]).splitlines()[0][:200]))
        self._refresh_tools()

    def on_busy(self, b):
        self.busy_call = None if b and b.get("name") == "h_model_dims" else b      # measuring the model locks nothing
        self._refresh_progress(); self._refresh_tools()

    def _poll(self):
        if self.isVisible() and self.host is not None and not self._shut: self.host.status()

    # ================================================================ status
    def on_status(self, snap):
        if TRACE: print("trace status %.2fs" % (getattr(snap, "secs", 0) or 0), flush=True)
        if snap.error:
            if self._export_after_status: self._export_after_status = False; self.tell(W.EXPORT["failed"].format(why=snap.error)); return
            self.notice(W.NOTICE["engine_error"].format(why=snap.error)); return
        old = self.snap; self.snap = snap
        if not self.running() and self.run_mine is None: self._resume_measurements()
        if self.editing and not any(o.get("id") == self.editing[2] for _lk, _j, o in (snap.chains or {}).get(self.editing[0], [])):
            b = self.panel[self.stage].body(self.editing[1]) if hasattr(self.panel[self.stage], "body") else None
            self.editing = None
            if b is not None:
                b.set_mode("idle"); b.set_msg("")
                for c in b.controls:
                    if c.c.get("tool"): c.engine_value = None            # back to a new step's start (refresh below)
        if self.run_mine is None and snap.running and not (self.state and self.state.building):
            pass                                            # a CLI run somewhere else: rows say so through their states
        if self.published and ((snap.node(snap.final("meshed")) or {}).get("status") in DONE_STATES or self.stage != "meshed"):
            self.published = None; self.shown = {}                   # the stage's own result takes over
        if not self.running() and not getattr(self, "_shut", False): QTimer.singleShot(3000, self._prefetch_quality)     # Quality ready before it is asked for
        if self.want_built: self._open_built(); self.status_changed.emit(); return
        if old is None and self.stage == "fused" and (snap.node(snap.final("meshed")) or {}).get("status") in DONE_STATES:
            self.show_stage("meshed")                       # a built project opens on the finished model, not the Fused step's crumbs
        elif old is None and self.stage in ("fused", "meshed") and not any(
                (snap.node(snap.final(k)) or {}).get("status") in DONE_STATES for k in ("fused", "meshed")):
            if self._has_model(): self._open_built(); self.status_changed.emit(); return    # built before (now out of date): open on that model
            self.show_stage("raw")                          # a project with nothing built yet opens on its scanned frames, not an empty page
        if snap.sides and 1 <= self.side_k <= len(snap.sides):  # "base cut by hand" goes when Auto takes the side back
            self.pager.set_page(self.side_k, len(snap.sides), self.stage in ("raw", "side"), self._pager_note())
        self._refresh_all()
        if self.isVisible() and self.detected is None and not self.peek and (old is None or self._display_path() != self.shown.get("path")): self.request_view()
        self.status_changed.emit()
        if self._export_after_status: self._export_after_status = False; QTimer.singleShot(0, getattr(self, "_export_next", self.export_stl))

    def on_state(self, s):
        was = self._was_building; self.state = s; self._was_building = bool(s.building)
        if s.building and not was:self._cancel_memory_choice()
        if s.building and not was: self._cancel_measurements()
        if s.build_engine: self.engine.setText(s.build_engine)
        if s.building and not was: self.log.clear()
        self._update_log_button()
        if s.build_msg and not s.building:
            self.result.setText(s.build_msg)
            self.result.setStyleSheet("color: %s;" % (T["kept"] if s.build_msg.startswith(("Model built", "Fused built", "Meshed built", "Side built",
                                                                                             "Line-up built", "Coverage built")) else T["lost"]))
        if not self._shut and (self.isVisible() or (s.n_sides and not s.recording)): self._ensure_host(force=True)
        if was and not s.building: self._run_ended()
        if s.building and not was and self.run_mine is None: self.notice(W.NOTICE["scan_build"])
        self._refresh_tools(); self._refresh_progress()

    def on_build(self, prog):
        self.bar.setValue(int(prog.frac * 1000)); self.pct.setText(prog.elapsed_text())
        self.log_pop.step.setText(("Done." if prog.done else (prog.failed or "Stopped.")) if prog.finished else prog.step)
        if not prog.finished: self.result.setText("")
        self.prog = prog
        self._update_log_button()
        if self.run_mine is None: return
        ev = getattr(prog, "nodes", None) or {}
        if any(v.get("state") in ("done", "warn", "check", "failed", "cached") for v in ev.values()): self.debounce.start()
        self._refresh_rows(); self._refresh_progress()

    def _run_ended(self):
        prog = self.prog; mine = self.run_mine; self.run_mine = None; self.stopping = False; was_stopped = self._stopped
        if mine is not None and prog is not None:
            if prog.done: self.notice(mine.done_text(prog).replace(os.path.dirname(prog.model or "") + os.sep, "") if prog.model else mine.done_text(prog), level="info")
            elif self._stopped: self.notice(W.NOTICE["stopped"])
            elif prog.failed: self.notice(prog.failed); self.vp.card.show_failed(prog.failed, prog.elapsed())
        self._stopped = False; self.prog = None
        if prog is not None and prog.done:
            if getattr(prog, "model", None) or getattr(mine, "publish", False): self._orientation_prompt_ready = True
            self._drop_host(); self._ensure_host(force=True)
        elif self.host is not None: self.host.status()
        self._refresh_all()
        if getattr(self, "_ultra", None) and getattr(mine, "what", None) == "onetap": self._ultra_done(bool(prog is not None and prog.done and not was_stopped))
        if getattr(mine, "onetap_phase", 0) == 1 and prog is not None and prog.done and not was_stopped: self._onetap_check()
        elif (getattr(mine, "what", None) == "onetap" or getattr(mine, "land", False)) and prog is not None and prog.done and not was_stopped: self.show_built()   # YOLO / Build it done: the model, not side 1's raw frames
        if prog is not None and prog.done and getattr(prog, "model", None):
            self._quality = {}
            self.show_built()
            if isinstance(prog.done, dict) and prog.done.get("coverage_pending"):
                self._start_coverage(prog.done.get("generation"))

    def _cancel_measurements(self):
        import threading
        job = self.__dict__.pop("_coverage_job", None)
        if job:
            job["cancel"].set()
            if job.get("engine"):
                threading.Thread(target=job["engine"].cancel, daemon=True, name="coverage-cancel").start()
        quality = self.__dict__.pop("_quality_job", None)
        if quality:
            quality["cancel"].set()
            process = quality.get("process")
            if process is not None and process.poll() is None: process.terminate()
        if hasattr(self, "coverage_state"): self.coverage_state.hide()

    def _start_coverage(self, generation=None):
        import json, threading
        root = self.host_root
        if not root or self._shut or self.run_mine is not None: return
        try:
            with open(os.path.join(root, "stages", "published.json")) as stream: pub = json.load(stream)
        except (OSError, ValueError): return
        generation = generation or pub.get("generation")
        if not generation or generation != pub.get("generation"): return
        attempts = self.__dict__.setdefault("_coverage_attempted", set())
        identity = (root, generation)
        if identity in attempts: return
        attempts.add(identity)
        self._cancel_measurements()
        job = {"root": root, "generation": generation, "cancel": threading.Event()}
        self._coverage_job = job
        self.coverage_state.setText("Measuring coverage..."); self.coverage_state.show()
        def work():
            try:
                from stages.engine import Engine
                from stages.project import Project
                engine = job["engine"] = Engine(Project.open(root), out=lambda *_: None)
                if job["cancel"].is_set(): return
                result = engine.measure_published(generation)
                if not job["cancel"].is_set(): self.coverage_result.emit(dict(job=job, result=result))
            except Exception as error:
                if not job["cancel"].is_set(): self.coverage_result.emit(dict(job=job, error=str(error)))
        threading.Thread(target=work, daemon=True, name="coverage-follow-on").start()

    def _resume_measurements(self):
        import json
        if not self.host_root or self._shut or getattr(self, "_coverage_job", None): return
        try:
            with open(os.path.join(self.host_root, "stages", "published.json")) as stream: publication = json.load(stream)
        except (OSError, ValueError): return
        if publication.get("coverage_pending"): self._start_coverage(publication.get("generation"))
        elif publication.get("coverage_error"):
            self.coverage_state.setText("Coverage could not finish. Rebuild Coverage to retry."); self.coverage_state.show()

    def _coverage_finished(self, event):
        job = event["job"]
        if self.__dict__.get("_coverage_job") is not job or self.host_root != job["root"] or self._shut: return
        self._coverage_job = None
        if event.get("error") or event.get("result", {}).get("error"):
            self.coverage_state.setText("Coverage could not finish. Rebuild Coverage to retry."); self.coverage_state.show()
            return
        if event.get("result", {}).get("discarded"): self.coverage_state.hide(); return
        self.coverage_state.setText("Coverage measured."); self.coverage_state.setVisible(self.stage != "raw")
        if self.host: self.host.status()
        self._prefetch_quality()

    _stopped = False

    @property
    def last_changed(self): return self.__dict__.get("_last_changed")

    @last_changed.setter
    def last_changed(self, v):                 # a new change: the Rebuild bar names it, not an earlier undo
        self.__dict__["_last_changed"] = v; self.undo_text = None

    # ================================================================ refresh
    def _refresh_all(self):
        self._refresh_rows(); self._refresh_info(); self._refresh_panel(); self._refresh_applied(); self._refresh_footer()
        self._refresh_rebuild(); self._refresh_tools(); self._refresh_progress(); self._refresh_chips(); self._refresh_recipe()

    def _nodes(self, group):
        g = self.snap.group(group) if self.snap else None
        return g["nodes"] if g else []

    def _pager_note(self, k=None):
        """"left out" beside the side number when the shown side is not part of the model."""
        sides = (self.snap.sides if self.snap else None) or []
        k = self.side_k if k is None else k
        if 1 <= k <= len(sides) and not sides[k - 1]["active"]: return "left out"
        if 1 <= k <= len(sides) and sides[k - 1]["id"] in self._hand_cut_ids(): return W.SPOTS["hand_done"]
        return ""

    def _hand_cut_ids(self):
        """Ids of the sides whose base was cut by hand (recipe: side platter method manual)."""
        import json
        try:
            with open(os.path.join(self.host_root or "", "stages", "recipe.json")) as f: pl = json.load(f)["sides"].get("platter", {})
        except (OSError, ValueError, KeyError, TypeError): return set()
        return {k for k, v in pl.items() if isinstance(v, dict) and v.get("method") == "manual"}

    def _side(self):
        sides = (self.snap.sides if self.snap else None) or []
        if not sides: return None
        self.side_k = max(1, min(self.side_k, len(sides))); return sides[self.side_k - 1]

    def _row_state(self, sid):
        """(text, colour, tip, dim) of a stage row, from the run's events when a run of ours is going, else status."""
        if sid == "raw": return "", None, None, False
        nodes = [n for n in self._nodes(sid) if n["status"] not in ("parked",)]
        if sid == "fused": nodes = [n for n in nodes if n["status"] != "parked"]
        run = getattr(self.prog, "nodes", None) if (self.run_mine is not None and self.prog is not None) else None
        if run:
            ids = [n["id"] for n in nodes]
            ev = [run.get(i) for i in ids if run.get(i)]
            if any(e.get("state") == "failed" for e in ev):
                e = next(e for e in ev if e.get("state") == "failed")
                return W.ROW_STATE["failed"]["text"], "lost", (e.get("error") or "")[-300:], False
            cur = [e for e in ev if e.get("state") == "start"]
            if cur:
                fr = [float(e.get("frac") or 0) for e in cur]
                done = sum(1 for e in ev if e.get("state") in DONE_STATES)
                pct = int(100 * (done + sum(fr)) / max(1, len([e for e in ev if e.get("state") not in ("off", "n/a")])))
                return "%d%%" % min(99, pct), "accent", "How far this step has got. The progress card shows what runs right now.", False
            if any(e.get("state") in ("waiting", "stale") for e in ev): return W.ROW_STATE["waiting"]["text"], "ink3", W.ROW_STATE["waiting"]["tip"], False
        st = [n["status"] for n in nodes]
        if not st: return "", None, None, True
        if "failed" in st: return W.ROW_STATE["failed"]["text"], "lost", None, False
        if any(n.get("retry") for n in nodes): return W.ROW_STATE["built_retry"]["text"], "warn", " ".join(W.plain_warning(x) for x in next(n for n in nodes if n.get("retry")).get("warn") or []), False
        if "check" in st:
            n = next(n for n in nodes if n["status"] == "check"); s = n.get("stats") or {}
            if int(s.get("removed") or 0) == 0: return W.ROW_STATE["gone"]["text"], "ink3", W.ROW_STATE["gone"]["tip"], False
            return W.ROW_STATE["check"]["text"], "warn", W.ROW_STATE["check"]["tip"].format(now=s.get("removed", "?"), was=(n.get("params") or {}).get("recorded", "?")), False
        if "stale" in st and self._never_built(): return W.ROW_STATE["waiting"]["text"], "ink3", W.ROW_STATE["new"]["tip"], False   # never built: nothing to "rebuild"
        if "stale" in st: return W.ROW_STATE["stale"]["text"], "warn", W.ROW_STATE["stale"]["tip"], False
        if "waiting" in st: return W.ROW_STATE["waiting"]["text"], "ink3", W.ROW_STATE["waiting"]["tip"], True
        if all(n.get("adopted") for n in nodes if n["status"] == "adopted") and "adopted" in st: return "", None, W.ROW_STATE["adopted"]["tip"], False
        return "", None, None, False

    def _stage_sub(self, sid):
        s = self.snap
        if s is None: return ""
        if sid == "raw":
            raw = self._nodes("raw"); fr = sum(int((n.get("stats") or {}).get("frames") or 0) for n in raw)
            return "%d side%s, %d frames" % (len(raw), "" if len(raw) == 1 else "s", fr)
        if sid == "side":
            return "Per side, before the line-up"
        if sid == "lineup":
            act = sum(1 for x in s.sides if x["active"]); base = self._base_n(); out = [str(x["n"]) for x in s.sides if not x["active"]]
            return "%d side%s in%s%s" % (act, "" if act == 1 else "s", (", %d left out (side %s)" % (len(out), ", ".join(out))) if out else "",
                                         (", base side %s" % base) if base else "")
        if sid == "fused":
            n = s.node("fused") or {}; pd = (n.get("params") or {}).get("point_distance")
            ntools = sum(1 for lk, j, o in (s.chains or {}).get("fused", []) if o.get("on", True))
            return "%s %s mm, %d tool%s" % (self._fused_name(), ("%g" % pd) if pd else "?", ntools, "" if ntools == 1 else "s")
        if sid == "meshed":
            ntools = sum(1 for lk, j, o in (s.chains or {}).get("meshed", []) if o.get("on", True))
            if s.meshed_method=="as_fused" or (s.meshed_method=="auto" and s.fused_kind=="mesh"):
                return "Same as Fused (As scanned). Pick Watertight for a closed surface."
            return "%s, %s" % (MESH_NAMES.get(s.meshed_method, s.meshed_method), "no tools" if not ntools else "%d tool%s" % (ntools, "" if ntools == 1 else "s"))
        if sid == "report":
            st = (s.node("report") or {}).get("stats") or {}; k = len(st.get("places") or [])
            guide=getattr(self,"_scan_guides",{}).get((self._display_path(),self.colour_mode.get("report")))
            if guide:return "%d missing or rough spots" % guide["spots"]
            return "Checking missing and rough spots"
        return ""

    def _fused_name(self):
        s = self.snap; m = s.fused_method if s else "tsdf"
        return FUSED_NAMES.get(m) or dict((s.methods or {}).get("fused", [])).get(m, m)

    def _base_n(self):
        s = self.snap; b = ((s.node("lineup.adjust") or {}).get("stats") or {}).get("base") if s else None
        return next((x["n"] for x in s.sides if x["id"] == b), None) if b else None

    def _fill_spot_list(self, box, pts, remove):
        """One row per picked spot: its colored number, where it is, and a Remove."""
        lay = box.layout()
        while lay.count():
            it = lay.takeAt(0); w = it.widget()
            if w is not None: w.setParent(None); w.deleteLater()
        for i, x in enumerate(pts):
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)
            dot = QLabel("%d" % (i + 1)); dot.setFixedSize(22, 22); dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet("background:%s;color:#0b0e12;border-radius:11px;font-weight:700;" % W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)])
            txt = QLabel("Spot %d    %.0f, %.0f, %.0f mm" % (i + 1, x[0], x[1], x[2])); txt.setFont(ui_font(12))
            rm = button("Remove", "quiet"); rm.setFixedHeight(26); rm.clicked.connect(lambda _=False, k=i: remove(k))
            h.addWidget(dot); h.addWidget(txt, 1); h.addWidget(rm); lay.addWidget(row)

    # ---------------- Cut base on Fused / Meshed: click spots, Remove base cuts everything below them; each press is a stacked layer
    def _build_cut_ui(self):
        self._cspots = {"fused": [], "meshed": []}; self.cut_ui = {}
        for stage in ("fused", "meshed"):
            body = self.panel[stage].bodies.get("Cut base")
            if body is None: continue
            box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
            info = QLabel(W.SPOTS["cut_info"]); info.setWordWrap(True); info.setProperty("role", "hint"); v.addWidget(info)
            count = QLabel(""); count.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(count)
            lst = QWidget(); QVBoxLayout(lst).setContentsMargins(0, 0, 0, 0); lst.layout().setSpacing(3); v.addWidget(lst)
            row = QHBoxLayout(); row.setSpacing(6)
            undo = button(W.SPOTS["undo"], None); clear = button(W.SPOTS["clear"], None)
            for b_ in (undo, clear): b_.setMinimumHeight(32); row.addWidget(b_)
            v.addLayout(row)
            go = big_button(W.SPOTS["setbase"], "apply"); v.addWidget(go)
            msg = QLabel(""); msg.setWordWrap(True); msg.setProperty("role", "hint"); v.addWidget(msg)
            undo.clicked.connect(lambda _=False, st=stage: self._cut_undo(st)); clear.clicked.connect(lambda _=False, st=stage: self._cut_clear(st))
            go.clicked.connect(lambda _=False, st=stage: self._cut_remove(st))
            body.v.insertWidget(max(0, body.v.count() - 1), box)
            for c in body.controls: c.moved.connect(lambda _v=None: self._cut_view())
            self.cut_ui[stage] = dict(box=box, count=count, list=lst, undo=undo, clear=clear, go=go, msg=msg)

    def _cut_on(self):
        return bool(self.gl and self.stage in ("fused", "meshed") and self.stage in getattr(self, "cut_ui", {})
                    and self.panel[self.stage].current == "Cut base" and self.vp.view.items)

    def _cut_local(self, world):
        """A click in the view as a point of the model itself (the view may show it turned by the saved orientation)."""
        key, it = self._orient_item()
        if key is None or it.get("model") is None: return [float(x) for x in world]
        M = np.asarray(it.get("model"), float); Mi = np.linalg.inv(M)
        return [float(x) for x in (Mi[:3, :3] @ np.asarray(world, float) + Mi[:3, 3])]

    def _cut_margin(self, stage):
        c = self._ctl(self.panel[stage].bodies["Cut base"], param="margin")
        return float(c.value()) if c is not None else 1.5

    def _cut_plane(self, stage):
        """{nx, ny, nz, d, margin} for the base the spots sit on, the bigger side of the model kept; None below 3 spots or in a line.
        The plane through the spots is then refitted to the flat layer around them and put at the top of that layer, so a click that
        landed a little low or high, and scan noise, still take the whole base."""
        pts = np.asarray(self._cspots.get(stage, []), float)
        if len(pts) < 3: return None
        c = pts.mean(0); u, sv, vt = np.linalg.svd(pts - c)
        if sv[1] < 1e-3 * max(sv[0], 1e-9) or sv[1] < 2.0: return None
        n = vt[2] / np.linalg.norm(vt[2]); v = self.vp.view; d = float(n @ c)
        if v.elements is not None and len(v.elements[0]):
            E = np.asarray(v.elements[0], float); S = E[::max(1, len(E) // 400000)]; h = (S - c) @ n
            if (h > 0).sum() < (h < 0).sum(): n = -n; h = -h; d = float(n @ c)
            reach = 1.4 * float(np.linalg.norm(pts - c, axis=1).max()) + 10.0
            along = S - c - np.outer(h, n)
            band = (np.abs(h) <= 6.0) & (np.linalg.norm(along, axis=1) <= reach)
            if int(band.sum()) >= 200:
                B = S[band]; cb = B.mean(0); n2 = np.linalg.svd(B - cb, full_matrices=False)[2][2]
                if float(n2 @ n) < 0: n2 = -n2
                if float(n2 @ n) > 0.97:                       # a real flat layer: use its own tilt
                    n = n2 / np.linalg.norm(n2); hb = (B - c) @ n
                    d = float(n @ c) + float(np.percentile(hb, 95))      # the top of the layer
        from stages.tools import plane_cut as PC
        return {"nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2]), "d": float(d), "margin": self._cut_margin(stage),
                "pad": 4.0, "poly": PC.outline(pts, n)}

    def _cut_view(self):
        ui = self.cut_ui.get(self.stage); v = self.vp.view
        for st, u in self.cut_ui.items(): u["box"].setVisible(st == self.stage and self._cut_on())
        if v is None: return False
        if not self._cut_on() or ui is None:
            if v.tool == "pairpick" and not self._points_on() and not self._spots_on() and not self._orient_on() and not self._clone_on() and not self._fix_on() and not self._mirror_on() and not self._mirror_on(): v.set_tool("orbit"); v.set_markers([])
            return False
        pts = self._cspots[self.stage]; n = len(pts)
        if v.tool != "pairpick": v.set_tool("pairpick")
        v.set_markers([(x, W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)], i + 1) for i, x in enumerate(self._cut_display(pts))])
        ui["count"].setText(W.SPOTS["count"].format(n=n, s="" if n == 1 else "s"))
        self._fill_spot_list(ui["list"], pts, lambda k, st=self.stage: self._cut_drop(st, k))
        ui["undo"].setEnabled(n > 0); ui["clear"].setEnabled(n > 0); ui["go"].setEnabled(n >= 3 and getattr(self, "_tools_on", False) and v.tools_on)
        plane = self._cut_plane(self.stage); ui["msg"].setText("")
        if plane is None or v.elements is None: v.clear_would_go()
        else:
            from stages.tools import plane_cut as PC
            cut = PC.below(v.elements[0], plane)
            v.show_would_go(cut); share = float(np.mean(cut)) if len(cut) else 0.0
            if share > 0.4: ui["msg"].setText(W.SPOTS["cut_big"].format(pct=int(round(100 * share))))
            elif share > 0.02 and self._cut_in_the_part(v.elements[0], cut, plane): ui["msg"].setText(W.SPOTS["cut_part"])
        return True

    def _cut_in_the_part(self, E, cut, plane):
        """A cut along a base removes a thin layer just under its plane. When what goes reaches deep under the plane (more than a
        fifth of the model's height), the plane runs through the part: warn."""
        if not cut.any(): return False
        E = np.asarray(E, np.float64); n = np.array([plane["nx"], plane["ny"], plane["nz"]])
        h = E[::max(1, len(E) // 200000)] @ n; span = float(h.max() - h.min()) or 1.0
        return float(plane["d"] - np.percentile(E[cut] @ n, 5)) > 0.2 * span

    def _cut_display(self, pts):
        """Model points as the view shows them (turned by the saved orientation when there is one)."""
        key, it = self._orient_item()
        if key is None or it.get("model") is None: return pts
        M = np.asarray(it.get("model"), float)
        return [[float(x) for x in (M[:3, :3] @ np.asarray(p, float) + M[:3, 3])] for p in pts]

    def _cut_add(self, world):
        self._cspots[self.stage].append(self._cut_local(world)); self._cut_view()

    def _cut_drop(self, stage, k):
        if 0 <= k < len(self._cspots[stage]): self._cspots[stage].pop(k); self._cut_view()

    def _cut_undo(self, stage):
        if self._cspots[stage]: self._cspots[stage].pop(); self._cut_view()

    def _cut_clear(self, stage):
        self._cspots[stage] = []; self._cut_view()

    def _cut_remove(self, stage):
        if (self.running() or self.busy_call or not getattr(self, "_tools_on", False)
                or self.stage != stage or not self._cut_on() or not self.vp.view.tools_on): return
        plane = self._cut_plane(stage)
        if plane is None: self.notice(W.SPOTS["need"]); return
        n = len(self._cspots[stage]); tok = self._list_token(stage); self.last_changed = stage
        k = 1 + sum(1 for c in ((getattr(self.snap, "chains", None) or {}).get(tok) or []) if isinstance(c[2], dict) and c[2].get("tool") == "cut_plane")
        self.call("add_op", tok, {"tool": "cut_plane", "params": plane},
                  then=lambda _r: (self.notice(W.SPOTS["cut_done"].format(n=n, k=k), level="info"),
                                   self.start_run(self._edit_run())))
        self._cspots[stage] = []; self._cut_view()

    # ---------------- Clone on Fused / Meshed: a shape at a source spot is copied over the same shape at a destination spot
    def _build_clone_ui(self):
        self._clone = {"fused": {"src": None, "dst": None, "next": "src"}, "meshed": {"src": None, "dst": None, "next": "src"}}; self.clone_ui = {}
        for stage in ("fused", "meshed"):
            body = self.panel[stage].bodies.get("Clone")
            if body is None: continue
            box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
            info = QLabel(W.SPOTS["clone_info"]); info.setWordWrap(True); info.setProperty("role", "hint"); v.addWidget(info)
            row = QHBoxLayout(); row.setSpacing(6)
            b_src = button("Set source", None); b_dst = button("Set destination", None)
            for b_ in (b_src, b_dst): b_.setCheckable(True); b_.setMinimumHeight(32); row.addWidget(b_)
            v.addLayout(row)
            state = QLabel(""); state.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(state)
            clear = button(W.SPOTS["clear"], None); clear.setMinimumHeight(30); v.addWidget(clear)
            go = big_button("Apply clone", "apply"); v.addWidget(go)
            b_src.clicked.connect(lambda _=False, st=stage: self._clone_next(st, "src")); b_dst.clicked.connect(lambda _=False, st=stage: self._clone_next(st, "dst"))
            clear.clicked.connect(lambda _=False, st=stage: self._clone_clear(st)); go.clicked.connect(lambda _=False, st=stage: self._clone_apply(st))
            body.v.insertWidget(0, box)              # the main buttons first: Apply clone is never below the fold
            for c in body.controls: c.moved.connect(lambda _v=None: self._clone_view())
            self.clone_ui[stage] = dict(box=box, src=b_src, dst=b_dst, state=state, clear=clear, go=go)

    def _clone_on(self):
        return bool(self.gl and self.stage in ("fused", "meshed") and self.stage in getattr(self, "clone_ui", {})
                    and self.panel[self.stage].current == "Clone" and self.vp.view.items)

    def _clone_params(self, stage):
        body = self.panel[stage].bodies["Clone"]; out = {}
        for name in ("shape", "diameter", "width", "height", "depth", "rotation", "mirror"):
            c = self._ctl(body, param=name)
            if c is not None: out[name] = c.value()
        st = self._clone[stage]
        if st["src"] is not None: out["src"] = st["src"]
        if st["dst"] is not None: out["dst"] = st["dst"]
        return out

    def _clone_normal(self, local):
        """The surface direction at a picked point, from the shown points around it, turned to face the camera."""
        from scipy.spatial import cKDTree
        v = self.vp.view; E = np.asarray(v.elements[0], np.float64) if v.elements is not None else None
        if E is None or not len(E): return None
        key = (id(v.elements[0]), len(E))
        if getattr(self, "_clone_tree", (None,))[0] != key: self._clone_tree = (key, cKDTree(E[::max(1, len(E) // 400000)]), E[::max(1, len(E) // 400000)])
        _k, tree, S = self._clone_tree; idx = tree.query_ball_point(np.asarray(local, float), 6.0)
        if len(idx) < 6: return None
        Q = S[idx]; n = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][2]
        V = np.asarray(v.cam.view, float); eye = -(V[:3, :3].T @ V[:3, 3])                 # camera in the shown frame
        key2, it = self._orient_item(); M = it.get("model") if it else None
        if M is not None: M = np.asarray(M, float); eye = np.linalg.inv(M)[:3, :3] @ eye + np.linalg.inv(M)[:3, 3]
        return n if (eye - np.asarray(local, float)) @ n > 0 else -n

    def _clone_pick(self, world):
        import json
        local = self._cut_local(world); n = self._clone_normal(local); st = self._clone[self.stage]
        if n is None: self.notice(W.SPOTS["missed"], level="info"); return
        st[st["next"]] = json.dumps([[float(x) for x in local], [float(x) for x in n]])
        st["next"] = "dst" if st["next"] == "src" else "src" if st["dst"] is None else "dst"
        self._clone_view()

    def _clone_next(self, stage, which): self._clone[stage]["next"] = which; self._clone_view()

    def _clone_clear(self, stage): self._clone[stage] = {"src": None, "dst": None, "next": "src"}; self._clone_view()

    def _clone_view(self):
        """The clone tab on: the pick tool armed, S and D dots, the area the copy will replace in red."""
        import json
        for st_, u in self.clone_ui.items(): u["box"].setVisible(st_ == self.stage and self._clone_on())
        v = self.vp.view
        if v is None: return False
        if not self._clone_on():
            if v.tool == "pairpick" and not self._points_on() and not self._spots_on() and not self._orient_on() and not self._cut_on() and not self._fix_on() and not self._mirror_on(): v.set_tool("orbit"); v.set_markers([])
            if getattr(self, "_clone_outlined", False): v.set_plane_marks([]); self._clone_outlined = False
            self._spot_arrows_off("clone")
            return False
        ui = self.clone_ui[self.stage]; st = self._clone[self.stage]
        which = st["next"] if st.get(st["next"]) else ("dst" if st["dst"] else "src")       # the arrows ride the picked spot
        if st.get(which): self._spot_arrows(st[which], "clone", float(self._clone_params(self.stage).get("diameter", 10.0)) * 0.7)
        else: self._spot_arrows_off("clone")
        if v.tool != "pairpick": v.set_tool("pairpick")
        marks = []
        for which, col, lab in (("src", "#2fd06a", "S"), ("dst", "#f0a030", "D")):
            if st[which] is not None: marks.append((json.loads(st[which])[0], col, lab))
        v.set_markers(self._clone_marks_display(marks))
        ui["src"].setChecked(st["next"] == "src"); ui["dst"].setChecked(st["next"] == "dst"); ui["go"].setEnabled(st["src"] is not None and st["dst"] is not None)
        ui["state"].setText("Source %s   Destination %s" % ("set" if st["src"] else "not set", "set" if st["dst"] else "not set"))
        v.clear_would_go(); v.set_plane_marks(self._clone_outlines(st)); self._clone_outlined = True
        if st["dst"] is not None and v.elements is not None:
            from stages.tools import clone as CL
            cd, nd = json.loads(st["dst"]); cs = json.loads(st["src"])[0] if st["src"] else None
            F = CL.frame(np.asarray(nd, float), (np.asarray(cd, float) - np.asarray(cs, float)) if cs else None)
            mask = CL.inside(np.asarray(v.elements[0], np.float64), np.asarray(cd, float), F, self._clone_params(self.stage))
            if mask.any(): v.show_would_go(mask)
        return True

    def _clone_outlines(self, st):
        """The shape at each spot, drawn on the surface: green at the source, orange at the destination."""
        import json
        from stages.tools import clone as CL
        p = self._clone_params(self.stage); out = []
        if st["src"] is None and st["dst"] is None: return out
        along = None
        if st["src"] is not None and st["dst"] is not None: along = np.asarray(json.loads(st["dst"])[0], float) - np.asarray(json.loads(st["src"])[0], float)
        key, it = self._orient_item(); M = it.get("model") if it else None
        for which, col, text in (("src", "#2fd06a", "Copy from here"), ("dst", "#f0a030", "Copy goes here")):
            if st[which] is None: continue
            c, n = json.loads(st[which]); c = np.asarray(c, float); F = CL.frame(np.asarray(n, float) / np.linalg.norm(n), along)
            if p.get("shape") == "box": w, h = float(p["width"]) / 2, float(p["height"]) / 2; loc = [(-w, -h), (w, -h), (w, h), (-w, h)]
            else:
                r = float(p["diameter"]) / 2; loc = [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 28, endpoint=False)]
            pts = np.array([c + a * F[:, 0] + b * F[:, 1] + 0.3 * F[:, 2] for a, b in loc])
            if M is not None: M_ = np.asarray(M, float); pts = pts @ M_[:3, :3].T + M_[:3, 3]
            out.append(dict(corners=pts.tolist(), colour=col, label=which, text=text))
        return out

    # ---- move arrows on a tool's spot (Smooth area, Clone): drag one to slide the spot along a world axis
    def _spot_arrows(self, spot_json, owner, size=8.0):
        import json
        v = self.vp.view; c = json.loads(spot_json)[0]
        v.set_gizmo(self._clone_marks_display([(c, "", "")])[0][0], max(6.0, float(size)), rings=False, arrows=True); v.gizmo["owner"] = owner

    def _spot_arrows_off(self, owner):
        v = self.vp.view
        if v is not None and v.gizmo is not None and v.gizmo.get("owner") == owner: v.set_gizmo(None)

    def _spot_arrow_move(self, k, mm, done):
        """A drag on the arrows of the Smooth area spot or the picked Clone spot: the spot slides along world axis k,
        its surface direction found again where it lands."""
        import json
        from app2.qt.rotate_gizmo import AXES
        v = self.vp.view
        if v is None or v.gizmo is None: return
        owner = v.gizmo.get("owner")
        if owner == "fix" and self._fix_on(): get = lambda: self._fix[self.stage]["spot"]; put = lambda j: self._fix[self.stage].__setitem__("spot", j); redraw = self._fix_view
        elif owner == "clone" and self._clone_on():
            st = self._clone[self.stage]; which = st["next"] if st.get(st["next"]) else ("dst" if st["dst"] else "src")
            get = lambda: st[which]; put = lambda j: st.__setitem__(which, j); redraw = self._clone_view
        else: return
        spot = get()
        if spot is None: return
        c, n = json.loads(spot)
        base = getattr(self, "_spot_arrow_base", None)
        if base is None: base = self._spot_arrow_base = np.asarray(c, float)
        key, it = self._orient_item(); M = np.asarray(it.get("model"), float) if it and it.get("model") is not None else np.eye(4)
        c2 = base + np.linalg.solve(M[:3, :3], AXES[k] * mm)
        n2 = self._clone_normal(c2) if done else None
        put(json.dumps([[float(x) for x in c2], [float(x) for x in (n2 if n2 is not None else n)]]))
        if done: self._spot_arrow_base = None
        redraw()

    def _clone_marks_display(self, marks):
        key, it = self._orient_item(); M = it.get("model") if it else None
        if M is None: return marks
        M = np.asarray(M, float); return [([float(x) for x in (M[:3, :3] @ np.asarray(p, float) + M[:3, 3])], c, l) for p, c, l in marks]

    def _clone_apply(self, stage):
        st = self._clone[stage]
        if st["src"] is None or st["dst"] is None: self.notice(W.SPOTS["clone_need"]); return
        params = self._clone_params(stage); tok = self._list_token(stage); self.last_changed = stage
        self.call("add_op", tok, {"tool": "clone", "params": params},
                  then=lambda _r: (self.notice(W.SPOTS["clone_done"].format(n="the"), level="info"), self.start_run(self._edit_run())))
        self._clone[stage] = {"src": None, "dst": None, "next": "src"}; self._clone_view()

    # ---------------- Smooth area on Fused / Meshed: smooth only inside a shape at one spot
    def _build_fix_ui(self):
        self._fix = {"fused": {"spot": None}, "meshed": {"spot": None}}; self.fix_ui = {}
        for stage in ("fused", "meshed"):
            body = self.panel[stage].bodies.get("Smooth area")
            if body is None: continue
            box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
            info = QLabel(W.SPOTS["fix_info"]); info.setWordWrap(True); info.setProperty("role", "hint"); v.addWidget(info)
            state = QLabel(""); state.setWordWrap(True); state.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(state)
            row = QHBoxLayout(); row.setSpacing(6); clear = button(W.SPOTS["clear"], None); clear.setMinimumHeight(30); row.addWidget(clear)
            preview = button("Preview patch", None); row.addWidget(preview); v.addLayout(row)
            preview.clicked.connect(lambda _=False, st=stage: self._fix_preview(st))
            go = big_button(W.SPOTS["fix_go"], "apply"); v.addWidget(go)
            clear.clicked.connect(lambda _=False, st=stage: self._fix_clear(st)); go.clicked.connect(lambda _=False, st=stage: self._fix_apply(st))
            body.v.insertWidget(0, box)
            for c in body.controls: c.moved.connect(lambda _v=None: self._fix_view())
            self.fix_ui[stage] = dict(box=box, state=state, clear=clear, go=go, info=info, preview=preview)

    # ---------------- Fill holes on Fused / Meshed: the holes it would close, outlined before Apply
    def _build_holes_ui(self):
        self.holes_ui = {}; self._holes_cache = {}
        for stage in ("fused", "meshed"):
            body = self.panel[stage].bodies.get("Hole Filling")
            if body is None: continue
            state = QLabel(""); state.setWordWrap(True); state.setFont(ui_font(13, QFont.Weight.DemiBold))
            body.v.insertWidget(0, state)
            for c in body.controls: c.moved.connect(lambda _v=None: self._holes_view())
            self.holes_ui[stage] = dict(state=state)

    def _holes_on(self):
        return bool(self.gl and self.stage in ("fused", "meshed") and self.stage in getattr(self, "holes_ui", {})
                    and self.panel[self.stage].current == "Hole Filling" and self.vp.view.items)

    def _holes_view(self):
        """Outline every hole of the shown mesh: the ones Largest hole closes in blue, the ones it leaves in red, and
        say how many."""
        v = self.vp.view
        if v is None: return False
        if not self._holes_on():
            if getattr(self, "_holes_drawn", False): v.set_lines([]); self._holes_drawn = False
            return False
        from app2.builder import holes as H
        key = v.edit_key; it = v.items.get(key) or {}
        ui = self.holes_ui[self.stage]
        if it.get("kind") != "mesh":
            ui["state"].setText(W.HOLES["no_mesh"]); return True
        F = it["F"]; sig = (key, id(F), len(F))
        loops = self._holes_cache.get(sig)
        if loops is None: self._holes_cache = {sig: H.boundary_loops(it["V"], F)}; loops = self._holes_cache[sig]
        c = self._ctl(self.panel[self.stage].bodies["Hole Filling"], param="hole_mm")
        mm = float(c.value()) if c is not None else 0.0
        auto = not mm
        if auto: mm = H.auto_mm(it["V"])
        filled, kept = H.split(loops, mm)
        M = it.get("model")
        def world(segs):
            if M is None or not len(segs): return segs
            Mx = np.asarray(M, np.float32); return segs @ Mx[:3, :3].T + Mx[:3, 3]
        parts = []
        if filled: parts.append((world(H.segments(filled)), (0.36, 0.62, 0.94, 1.0)))
        if kept: parts.append((world(H.segments(kept)), (1.0, 0.35, 0.31, 1.0)))
        v.set_lines(parts); self._holes_drawn = True
        if not loops: text = W.HOLES["closed"]
        else:
            big = max(l[1] for l in kept) * 2 if kept else 0.0
            text = W.HOLES["count"].format(n=len(filled), m=len(loops), mm="%g" % round(mm, 1), auto=" (Auto)" if auto else "")
            if kept: text += " " + W.HOLES["left"].format(k=len(kept), big="%g" % round(big, 1))
        ui["state"].setText(text)
        return True

    def _fix_on(self):
        return bool(self.gl and self.stage in ("fused", "meshed") and self.stage in getattr(self, "fix_ui", {})
                    and self.panel[self.stage].current == "Smooth area" and self.vp.view.items)

    def _fix_params(self, stage):
        body = self.panel[stage].bodies["Smooth area"]; out = {}
        for name in ("action", "shape", "diameter", "width", "height", "depth", "feather", "strength", "passes", "reach"):
            c = self._ctl(body, param=name)
            if c is not None: out[name] = c.value()
        if self._fix[stage]["spot"] is not None: out["spot"] = self._fix[stage]["spot"]
        return out

    def _fix_pick(self, world):
        import json
        local = self._cut_local(world); n = self._clone_normal(local)
        if n is None: self.notice(W.SPOTS["missed"], level="info"); return
        self._fix[self.stage]["spot"] = json.dumps([[float(x) for x in local], [float(x) for x in n]]); self._fix_view()

    def _fix_clear(self, stage): self._fix[stage]["spot"] = None; self._fix_view()

    def _fix_view(self):
        import json
        for st_, u in self.fix_ui.items(): u["box"].setVisible(st_ == self.stage and self._fix_on())
        v = self.vp.view
        if v is None: return False
        if not self._fix_on():
            if v.tool == "pairpick" and not self._points_on() and not self._spots_on() and not self._orient_on() and not self._cut_on() and not self._clone_on() and not self._fix_on() and not self._mirror_on() and not self._mirror_on(): v.set_tool("orbit"); v.set_markers([])
            if getattr(self, "_fix_outlined", False): v.set_plane_marks([]); self._fix_outlined = False
            v.set_fill_preview(None); self._spot_arrows_off("fix")
            return False
        ui = self.fix_ui[self.stage]; spot = self._fix[self.stage]["spot"]
        if v.tool != "pairpick": v.set_tool("pairpick")
        fill = self._fix_params(self.stage).get("action") == "fill"
        signature = self._fix_signature(self.stage)
        ready = self._fix[self.stage].get("preview") == signature
        if not fill or not ready: v.set_fill_preview(None)
        ui["preview"].setVisible(fill); ui["preview"].setEnabled(fill and spot is not None and not self.running())
        ui["go"].setEnabled(spot is not None and (not fill or ready)); ui["state"].setText("Spot set" if spot else "Spot not set")
        ui["go"].setText(W.SPOTS["fill_go"] if fill else W.SPOTS["fix_go"]); ui["info"].setText(W.SPOTS["fill_info"] if fill else W.SPOTS["fix_info"])
        v.clear_would_go(); v.set_markers([]); v.set_plane_marks([]); self._fix_outlined = True
        if spot is not None and v.elements is not None:
            from stages.tools import local_fix as LF
            from stages.tools import clone as CL
            p = self._fix_params(self.stage); c, n = json.loads(spot); F = CL.frame(np.asarray(n, float) / np.linalg.norm(n))
            if p.get("action") != "fill":                        # filling adds surface: only the outline shows where
                w = LF.weights(np.asarray(v.elements[0], np.float64), np.asarray(c, float), F, p)
                if (w > 0).any(): v.show_would_go(w > 0)
            v.set_markers(self._clone_marks_display([(c, "#4fa8ff", "S")]))
            v.set_plane_marks(self._clone_outlines_for(spot, p, F))
            self._spot_arrows(spot, "fix", float(p.get("diameter") or p.get("width") or 8.0) * 0.7)
        else: self._spot_arrows_off("fix")
        return True

    def _clone_outlines_for(self, spot, p, F):
        import json
        c = np.asarray(json.loads(spot)[0], float)
        if p.get("shape") == "box": w, h = float(p["width"]) / 2, float(p["height"]) / 2; loc = [(-w, -h), (w, -h), (w, h), (-w, h)]
        else: r = float(p["diameter"]) / 2; loc = [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 28, endpoint=False)]
        pts = np.array([c + a * F[:, 0] + b * F[:, 1] + 0.3 * F[:, 2] for a, b in loc])
        key, it = self._orient_item(); M = it.get("model") if it else None
        if M is not None: M = np.asarray(M, float); pts = pts @ M[:3, :3].T + M[:3, 3]
        return [dict(corners=pts.tolist(), colour="#4fa8ff", label="fix", text="Filled here" if p.get("action") == "fill" else "Smoothed here")]

    def _fix_signature(self, stage):
        import json
        return (stage, self._display_path(), json.dumps(self._fix_params(stage), sort_keys=True))


    def _fix_preview(self, stage):
        if self._fix[stage]["spot"] is None: return
        signature = self._fix_signature(stage); params = self._fix_params(stage)
        self.fix_ui[stage]["state"].setText("Making the patch preview...")
        def got(result):
            if stage != self.stage or signature != self._fix_signature(stage) or not self._fix_on(): return
            if not result["stats"].get("added_faces"):
                self.fix_ui[stage]["state"].setText("No gap to fill here."); return
            self._fix[stage]["pending_preview"] = signature
            self._fix[stage]["preview_stats"] = result["stats"]
            self.loader.request(dict(id=self.view_id, slot="fill-preview", items=[dict(key="patch", path=result["path"])],
                                     mode="shaded", signature=signature, stage=stage))
        self.call("h_fill_preview", self._display_path(), params, then=got, fail=lambda e: self.notice(str(e)))


    def _on_fill_preview(self, out):
        req = out.get("req") or {}; stage = req.get("stage")
        if out.get("id") != self.view_id or stage != self.stage or tuple(req.get("signature", ())) != self._fix_signature(stage): return
        if out.get("error"): self.notice(out["error"]); return
        if not self._fix_on() or not out.get("items"): return
        item = out["items"][0]; _, original = self._orient_item()
        item["model"] = original.get("model") if original else None
        self._fix[stage]["preview"] = tuple(req["signature"])
        self.vp.view.set_fill_preview(item); self._fix_view()
        self.fix_ui[stage]["state"].setText("Purple is the proposed patch. " + W.operation_stats("local_fix", self._fix[stage]["preview_stats"], "mesh"))


    def _fix_apply(self, stage):
        if self._fix[stage]["spot"] is None: self.notice(W.SPOTS["fix_need"]); return
        params = self._fix_params(stage)
        if params.get("action") == "fill" and self._fix[stage].get("preview") != self._fix_signature(stage):
            self.notice("Press Preview patch first, then apply it."); return
        tok = self._list_token(stage); self.last_changed = stage
        self.call("add_op", tok, {"tool": "local_fix", "params": params},
                  then=lambda _r: (self.notice(W.SPOTS["fill_done"] if params.get("action") == "fill" else W.SPOTS["fix_done"], level="info"), self.start_run(self._edit_run())))
        self._fix[stage] = {"spot": None}; self._fix_view()

    # ---------------- Mirror on Fused / Meshed: the other half of a symmetric part from a mirrored copy
    def _build_mirror_ui(self):
        self._mspots = {"fused": [], "meshed": []}; self._mplane = {"fused": None, "meshed": None}; self.mirror_ui = {}
        for stage in ("fused", "meshed"):
            body = self.panel[stage].bodies.get("Mirror")
            if body is None: continue
            box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
            info = QLabel(W.SPOTS["mirror_info"]); info.setWordWrap(True); info.setProperty("role", "hint")
            info.setMinimumWidth(0); info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum); v.addWidget(info)
            row0 = QHBoxLayout(); row0.setSpacing(6); auto = button(W.SPOTS["mirror_auto"], None); auto.setMinimumHeight(32); row0.addWidget(auto); v.addLayout(row0)
            count = QLabel(""); count.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(count)
            lst = QWidget(); QVBoxLayout(lst).setContentsMargins(0, 0, 0, 0); lst.layout().setSpacing(3); v.addWidget(lst)
            row = QHBoxLayout(); row.setSpacing(6)
            undo = button(W.SPOTS["undo"], None); clear = button(W.SPOTS["clear"], None)
            for b_ in (undo, clear): b_.setMinimumHeight(32); row.addWidget(b_)
            v.addLayout(row)
            state = QLabel(""); state.setWordWrap(True); state.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(state)
            go = big_button(W.SPOTS["mirror_go"], "apply"); v.addWidget(go)
            msg = QLabel(""); msg.setWordWrap(True); msg.setProperty("role", "hint"); v.addWidget(msg)
            auto.clicked.connect(lambda _=False, st=stage: self._mirror_auto(st))
            undo.clicked.connect(lambda _=False, st=stage: self._mirror_undo(st)); clear.clicked.connect(lambda _=False, st=stage: self._mirror_clear(st))
            go.clicked.connect(lambda _=False, st=stage: self._mirror_apply(st))
            body.v.insertWidget(0, box)
            for c in body.controls: c.moved.connect(lambda _v=None: self._mirror_view())
            self.mirror_ui[stage] = dict(box=box, auto=auto, count=count, list=lst, undo=undo, clear=clear, state=state, go=go, msg=msg)

    def _mirror_on(self):
        return bool(self.gl and self.stage in ("fused", "meshed") and self.stage in getattr(self, "mirror_ui", {})
                    and self.panel[self.stage].current == "Mirror" and self.vp.view.items)

    def _mirror_params(self, stage):
        body = self.panel[stage].bodies["Mirror"]; out = {}
        for name in ("fit", "gap_mm"):
            c = self._ctl(body, param=name)
            if c is not None: out[name] = c.value()
        return out

    def _mirror_plane(self, stage):
        """(n, d) in model space: from the spots when there are three or more, else the plane Auto-find set; None without either."""
        from stages.tools import mirror as MI
        pts = self._mspots.get(stage, [])
        if len(pts) >= 3:
            pl = MI.fit_plane(pts)
            if pl is not None: return pl
        return self._mplane.get(stage)

    def _mirror_auto(self, stage):
        """Auto-find: the model's principal plane its mirror image matches best, refined. Replaces the spots."""
        from stages.tools import mirror as MI
        v = self.vp.view
        if v.elements is None or not self._mirror_on(): return
        E = np.asarray(v.elements[0], np.float64)
        key, it = self._orient_item(); M = np.asarray(it.get("model"), float) if (it and it.get("model") is not None) else None
        if M is not None: E = (E - M[:3, 3]) @ M[:3, :3]                 # the view's elements are in the shown (oriented) frame: back to the model's
        n, d, score = MI.auto_plane(E)
        self._mspots[stage] = []; self._mplane[stage] = (n, d)
        self._mirror_view()
        if score < 0.6: self.notice(W.SPOTS["mirror_low"].format(pct=int(round(100 * score))))

    def _mirror_view(self):
        ui = self.mirror_ui.get(self.stage); v = self.vp.view
        for st, u in self.mirror_ui.items(): u["box"].setVisible(st == self.stage and self._mirror_on())
        if v is None: return False
        if not self._mirror_on() or ui is None:
            if v.tool == "pairpick" and not self._points_on() and not self._spots_on() and not self._orient_on() and not self._cut_on() and not self._clone_on() and not self._fix_on(): v.set_tool("orbit"); v.set_markers([])
            if getattr(self, "_mirror_outlined", False): v.set_plane_marks([]); self._mirror_outlined = False
            return False
        pts = self._mspots[self.stage]; n_ = len(pts)
        if v.tool != "pairpick": v.set_tool("pairpick")
        v.set_markers([(x, W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)], i + 1) for i, x in enumerate(self._cut_display(pts))])
        ui["count"].setText(W.SPOTS["count"].format(n=n_, s="" if n_ == 1 else "s"))
        self._fill_spot_list(ui["list"], pts, lambda k, st=self.stage: self._mirror_drop(st, k))
        ui["undo"].setEnabled(n_ > 0); ui["clear"].setEnabled(n_ > 0 or self._mplane.get(self.stage) is not None)
        plane = self._mirror_plane(self.stage); ui["msg"].setText("")
        ui["go"].setEnabled(plane is not None and getattr(self, "_tools_on", False) and v.tools_on)
        if plane is None or v.elements is None:
            ui["state"].setText(W.SPOTS["mirror_need"] if plane is None else ""); v.clear_would_go(); v.set_plane_marks([]); self._mirror_outlined = True; return True
        from stages.tools import mirror as MI
        from scipy.spatial import cKDTree
        n, d = plane; E = np.asarray(v.elements[0], np.float64)
        key, it = self._orient_item(); M = np.asarray(it.get("model"), float) if (it and it.get("model") is not None) else None
        if M is not None: E = (E - M[:3, 3]) @ M[:3, :3]
        ck = (id(v.elements[0]), len(E))
        if getattr(self, "_mirror_tree", (None,))[0] != ck: self._mirror_tree = (ck, cKDTree(E))
        tree = self._mirror_tree[1]; p = self._mirror_params(self.stage)
        gap = max(float(p.get("gap_mm", 1.5)), 0.05)
        dist, _ = tree.query(MI.reflect(E, n, d), k=1, distance_upper_bound=gap, workers=-1)
        add = ~np.isfinite(dist); agree = MI.agreement(E, n, d, tree=tree)
        v.show_would_go(add)                                          # red: what the mirror image of these spots would fill in
        ui["state"].setText(W.SPOTS["mirror_state"].format(pct=int(round(100 * agree)), n=format(int(add.sum()), ",")))
        if agree < 0.6: ui["msg"].setText(W.SPOTS["mirror_low"].format(pct=int(round(100 * agree))))
        # the plane as a square through the model, sized to it
        c = E.mean(0); c = c - (c @ n - d) * n; e1, e2 = MI._basis(n); r = 0.55 * float(np.linalg.norm(E.max(0) - E.min(0)))
        corners = np.array([c + a * e1 + b * e2 for a, b in ((-r, -r), (r, -r), (r, r), (-r, r))])
        if M is not None: corners = corners @ M[:3, :3].T + M[:3, 3]
        v.set_plane_marks([dict(corners=corners.tolist(), colour="#c77dff", label="mirror", text=W.SPOTS["mirror_plane"])]); self._mirror_outlined = True
        return True

    def _mirror_add(self, world):
        self._mspots[self.stage].append(self._cut_local(world)); self._mplane[self.stage] = None; self._mirror_view()

    def _mirror_drop(self, stage, k):
        if 0 <= k < len(self._mspots[stage]): self._mspots[stage].pop(k); self._mirror_view()

    def _mirror_undo(self, stage):
        if self._mspots[stage]: self._mspots[stage].pop(); self._mirror_view()

    def _mirror_clear(self, stage):
        self._mspots[stage] = []; self._mplane[stage] = None; self._mirror_view()

    def _mirror_apply(self, stage):
        if (self.running() or self.busy_call or not getattr(self, "_tools_on", False)
                or self.stage != stage or not self._mirror_on() or not self.vp.view.tools_on): return
        plane = self._mirror_plane(stage)
        if plane is None: self.notice(W.SPOTS["mirror_need"]); return
        n, d = plane; params = dict(self._mirror_params(stage), nx=float(n[0]), ny=float(n[1]), nz=float(n[2]), d=float(d))
        tok = self._list_token(stage); self.last_changed = stage
        self.call("add_op", tok, {"tool": "mirror", "params": params},
                  then=lambda _r: (self.notice(W.SPOTS["mirror_done"], level="info"), self.start_run(self._edit_run())))
        self._mspots[stage] = []; self._mplane[stage] = None; self._mirror_view()

    # ---------------- Cut base: By hand (click spots on the base)
    def _build_spot_ui(self):
        body = self.panel["side"].bodies["Platter"]; box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
        self.spot_count = QLabel(""); self.spot_count.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(self.spot_count)
        self.spot_list = QWidget(); QVBoxLayout(self.spot_list).setContentsMargins(0, 0, 0, 0); self.spot_list.layout().setSpacing(3); v.addWidget(self.spot_list)
        row = QHBoxLayout(); row.setSpacing(6)
        self.spot_undo = button(W.SPOTS["undo"], None); self.spot_clear = button(W.SPOTS["clear"], None)
        for b_ in (self.spot_undo, self.spot_clear): b_.setMinimumHeight(32); row.addWidget(b_)
        v.addLayout(row)
        self.spot_set = big_button(W.SPOTS["setbase"], "apply"); v.addWidget(self.spot_set)
        nrow = QHBoxLayout(); nrow.setSpacing(6)
        self.spot_prev = button(W.SPOTS["prev"], None); self.spot_next = button(W.SPOTS["nextb"], None)
        for b_ in (self.spot_prev, self.spot_next): b_.setMinimumHeight(30); nrow.addWidget(b_, 1)
        v.addLayout(nrow)
        self.spot_allh = button(W.SPOTS["allh"], None, W.SPOTS["allh_tip"]); self.spot_allh.setMinimumHeight(30); v.addWidget(self.spot_allh)
        self.spot_prev.clicked.connect(lambda: self._spot_next(-1)); self.spot_next.clicked.connect(lambda: self._spot_next(1)); self.spot_allh.clicked.connect(self._spot_height_all)
        self.spot_undo.clicked.connect(self._spot_undo); self.spot_clear.clicked.connect(self._spot_clear); self.spot_set.clicked.connect(self._spot_set)
        ctl = self._ctl(body, "platter.method")                   # right under Auto / Manual, where the Manual hint says what to click
        body.v.insertWidget(body.v.indexOf(ctl) + 1 if ctl is not None else max(0, body.v.count() - 1), box)
        self.spot_box = box; self._spots = {}; self._spots_built = set(); box.hide()
        if ctl is not None: ctl.moved.connect(lambda val: QTimer.singleShot(0, lambda: self._cut_auto_apply(val)))   # after the panel's own refresh

    def _cut_auto_apply(self, val):
        """Auto, a Where to look chip or Keep the base cuts at once: the result shows, not a pending violet from the last cut.
        Manual waits for the spots and Remove base."""
        if val == "manual" or self.stage != "side" or self.panel["side"].current != "Platter": return
        ctl = self._ctl(self.panel["side"].bodies["Platter"], "platter.method")
        if ctl is None or not ctl.changed() or ctl.value() != val: return
        if self.running() or self.busy_call or not getattr(self, "_tools_on", False):
            # a status or line-up ask in flight: try again shortly (giving up left the violet of the last cut on screen); a build
            # running leaves it to Apply, as before
            tries = getattr(self, "_auto_cut_tries", 0)
            if not self.running() and tries < 20:
                self._auto_cut_tries = tries + 1; QTimer.singleShot(250, lambda: self._cut_auto_apply(val))
            else: self._auto_cut_tries = 0
            return
        self._auto_cut_tries = 0
        self.on_button("Platter")

    def _ctl(self, body, suffix=None, param=None):
        """A control of a tab by what it sets (the end of its recipe path, or its tool param), never by its position."""
        for c in body.controls:
            if suffix and str(c.c.get("path") or "").endswith(suffix): return c
            if param and c.c.get("param") == param: return c
        return None

    def _spots_on(self):
        if not (self.gl and self.stage == "side" and self.panel["side"].current == "Platter" and self._side() is not None): return False
        seg = self._ctl(self.panel["side"].bodies["Platter"], "platter.method")
        return bool(seg is not None and seg.value() == "manual" and not self._all_mode())

    def _solo_wanted(self):
        """Cut base shows the picked side alone, Auto or Manual: the base to check (or click) is this side's, and ghosted sides
        with their own bases hide it. Every other tab shows the side among the others."""
        return bool(self.gl and self.stage == "side" and self.panel["side"].current == "Platter" and not self._all_mode())

    def _spots_view(self):
        if not self.gl or self.vp.view is None: return
        """By hand on: the pick tool armed and the spots shown as numbered dots; otherwise the box is hidden."""
        if not hasattr(self, "spot_box"): return
        if hasattr(self, "cut_ui") and self._cut_view(): return
        if hasattr(self, "clone_ui") and self._clone_view(): return
        if hasattr(self, "fix_ui") and self._fix_view(): return
        if hasattr(self, "holes_ui") and self._holes_view(): return
        if hasattr(self, "mirror_ui") and self._mirror_view(): return
        on = self._spots_on(); self.spot_box.setVisible(on)
        main = self.panel["side"].bodies["Platter"].main
        seg = self._ctl(self.panel["side"].bodies["Platter"], "platter.method")
        manual_all = self._all_mode() and seg is not None and seg.value() == "manual"
        if main is not None: main.setVisible(not on and not manual_all)   # By hand has its own Remove base
        if manual_all: self.panel["side"].bodies["Platter"].set_note(MANUAL_SIDE)
        if not self.gl: return
        v = self.vp.view
        if on:
            side = self._side()["id"]; pts = self._spots.get(side, [])
            if v.tool != "pairpick": v.set_tool("pairpick")
            v.set_markers([(x, W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)], i + 1) for i, x in enumerate(self._cut_display(pts))])
            n = len(pts); self.spot_count.setText((W.SPOTS["count_need"] if n < 3 else W.SPOTS["count"]).format(n=n, s="" if n == 1 else "s"))
            self._fill_spot_list(self.spot_list, pts, self._spot_remove)
            self.spot_undo.setEnabled(n > 0); self.spot_clear.setEnabled(n > 0); self.spot_set.setEnabled(n >= 3 and getattr(self, "_tools_on", False))
            self.spot_allh.setEnabled(n >= 3 and getattr(self, "_tools_on", False))
            sd = self._side()
            if (not self._display_path() and sd["active"] and sd["id"] not in self._spots_built
                    and not self.running() and not self.busy_call):
                self._spots_built.add(sd["id"])            # nothing to click on yet: fuse this side's frames once, then show it
                self.notice(W.SPOTS["building"].format(side=sd["n"]), level="info"); self.start_run(StagesRun(to="side.fuse@" + sd["id"], publish=False))
            if self._display_path() != self.shown.get("path") or not getattr(self, "_view_solo", False): self.request_view()   # the ghosts go: the base must be clickable
            self._platter_flags()
        else:
            if v.tool == "pairpick" and not self._points_on(): v.set_tool("orbit"); v.set_markers([]); self._ghost_legend()
            if self.stage == "side" and getattr(self, "_view_solo", False) != self._solo_wanted(): self.request_view()   # Cut base left: the other sides come back around this one

    def _spot_add(self, world):
        if not self._spots_on(): return
        side = self._side()
        if side is not None: self._spots.setdefault(side["id"], []).append(self._cut_local(world)); self._spots_view()

    def _spot_next(self, d):
        """Go on to the next (or previous) side without leaving By hand: 12 sides one after the other."""
        if not self.snap or not self.snap.sides: return
        n = len(self.snap.sides); self.side_k = ((self.side_k - 1 + d) % n) + 1
        done = self._hand_cut_ids()
        for _ in range(n):                         # skip the sides already cut by hand (and left-out ones), unless all are done
            sd = self.snap.sides[self.side_k - 1]
            if sd["id"] not in done and sd["active"]: break
            self.side_k = ((self.side_k - 1 + d) % n) + 1
        else: self.notice(W.SPOTS["all_done"], level="info")
        self._clear_detect(); self.editing = None; self.pager.set_page(self.side_k, n, True, self._pager_note(self.side_k)); self._refresh_all()
        ctl = self._ctl(self.panel["side"].bodies["Platter"], "platter.method")
        if ctl is not None and ctl.value() != "manual": ctl.pick("manual")           # a new side's own method must not drop By hand
        self.request_view(); self._spots_view()

    def _spot_height_all(self):
        """The plate's thickness above the turntable, from this side's spots, as Margin for every side."""
        import json
        if self.running() or self.busy_call or not getattr(self, "_tools_on", False): return
        if not self._spots_on():
            if self._all_mode(): self.notice(MANUAL_SIDE)
            return
        side = self._side(); plane = self._spots_plane(quiet=False)
        if side is None or plane is None: return
        pp = self.snap.out("side.platter@" + side["id"], "plane") if self.snap else None
        try:
            with open(pp) as f: pl = json.load(f)
        except (OSError, TypeError, ValueError): self.notice(W.SPOTS["allh_bad"]); return
        if float(np.dot(plane["n"], pl["n"])) < 0.9: self.notice(W.SPOTS["allh_bad"]); return
        h = float(plane["d"]) - float(pl["d"])
        if h <= 0: self.notice(W.SPOTS["allh_bad"]); return
        m = round(min(40.0, h + 0.3), 1)
        self.notice(W.SPOTS["allh_done"].format(mm="%.1f" % h, m="%.1f" % m), level="info")
        self._spots[side["id"]] = []; self.last_changed = "side"
        self._set_then_run({"side.platter.margin": m}, "side", StagesRun(to="sides", publish=False))

    def _spot_remove(self, k):
        side = self._side()
        if side is not None and 0 <= k < len(self._spots.get(side["id"], [])): self._spots[side["id"]].pop(k); self._spots_view()

    def _spot_undo(self):
        side = self._side()
        if side is not None and self._spots.get(side["id"]): self._spots[side["id"]].pop(); self._spots_view()

    def _spot_clear(self):
        side = self._side()
        if side is not None: self._spots[side["id"]] = []; self._spots_view()

    def _spots_plane(self, quiet=True):
        """The base plane {n, d, margin, keep_above} fitted through the picked spots (the side's own frame), or None while there are
        fewer than 3 or they are almost in a line. The part's side is the bigger side of the plane."""
        side = self._side(); pts = np.asarray(self._spots.get(side["id"], []) if side else [], float)
        if len(pts) < 3: return None
        c = pts.mean(0); u, sv, vt = np.linalg.svd(pts - c)
        if sv[1] < 1e-3 * max(sv[0], 1e-9) or sv[1] < 2.0:
            if not quiet: self.notice(W.SPOTS["flat"])
            return None
        n = vt[2] / np.linalg.norm(vt[2]); V = self.vp.view.elements[0] if self.vp.view.elements else None
        if V is not None and len(V):
            S = np.asarray(V[::max(1, len(V) // 50000)], float); h = (S - c) @ n
            if (h > 0).sum() < (h < 0).sum(): n = -n
        mc = next((x for x in self.panel["side"].bodies["Platter"].controls if x.c.get("path") == "side.platter.margin"), None)
        return {"n": [float(x) for x in n], "d": float(n @ c), "margin": float(mc.value()) if mc is not None else 1.0, "keep_above": True}

    def _spot_set(self):
        """Remove base: cut at the plane through the spots, then run the Side steps for this side."""
        if self.running() or self.busy_call or not getattr(self, "_tools_on", False): return
        if not self._spots_on():
            if self._all_mode(): self.notice(MANUAL_SIDE)
            return
        side = self._side(); plane = self._spots_plane(quiet=False)
        if side is None or plane is None:
            if plane is None and len(self._spots.get(side["id"], []) if side else []) < 3: self.notice(W.SPOTS["need"])
            return
        n_spots = len(self._spots[side["id"]]); sid = side["id"]; self.last_changed = "side"
        self.call("h_platter_plane", sid, plane, then=lambda _r: (self.notice(W.SPOTS["done"].format(n=n_spots, side=side["n"]), level="info"),
                                                              self.start_run(StagesRun(to="side.keep@" + sid, publish=False)),
                                                              self._spot_next(1)))        # on to the next side without a cut
        self._spots[sid] = []; self._spots_view()                # the spots did their job: the view moves on to the cut result

    # ---------------- Orient for export (bottom by spots, front by the view); saved in the project, used at view and export
    def _reset_orientation_prompt(self):
        self._orientation_prompt_ready = False
        prompt = getattr(self, '_orientation_prompt', None)
        if prompt is not None:
            prompt.finished.disconnect()
            prompt.close()
            prompt.deleteLater()
        self._orientation_prompt = None

    def _maybe_prompt_orientation(self):
        from app2.view_orientation import orientation_prompt_pending, dismiss_orientation_prompt
        if not getattr(self, '_orientation_prompt_ready', False) or not self.host_root or self.stage != 'meshed' or not self.gl:
            return
        if not self.vp.view.items or not orientation_prompt_pending(self.host_root): return
        self._orientation_prompt_ready = False
        from PySide6.QtWidgets import QMessageBox
        prompt = QMessageBox(self)
        prompt.setWindowTitle('Bottom and front')
        prompt.setText('Is the model sitting the right way up?')
        prompt.setInformativeText('Right now the bed from the scan is the bottom and where the scanner started is the front. '
                                  'That sets how the model is shown and how it is exported, nothing about its shape.\n\n'
                                  'If it looks wrong, "Fix it" opens the Orient tool: click three spots on the face that sits '
                                  'on the table and press Set bottom, then turn the model to face you and press Use this view as front. '
                                  'You can come back to Orient any time.')
        choose = prompt.addButton('Fix it', QMessageBox.ButtonRole.AcceptRole)
        prompt.addButton('Looks right', QMessageBox.ButtonRole.RejectRole)
        project = self.host_root
        self._orientation_prompt = prompt
        def answered(_result):
            dismiss_orientation_prompt(project)
            if self.host_root == project and prompt.clickedButton() is choose:
                self._pick_stage('orient')
            prompt.deleteLater()
            self._orientation_prompt = None
        prompt.finished.connect(answered)
        prompt.open()

    def _orient_path(self): return os.path.join(self.host_root or "", "orient.json") if self.host_root else None

    def _orient_load(self):
        import json
        try:
            with open(self._orient_path()) as f: return np.asarray(json.load(f)["R"], float).reshape(3, 3)
        except (OSError, ValueError, KeyError, TypeError): return None

    def _orient_save(self, R):
        import json
        path = self._orient_path()
        if not path: return
        if R is None:
            try: os.remove(path)
            except OSError: pass
            return
        with open(path, "w") as f: json.dump({"R": [[float(x) for x in row] for row in R]}, f)

    def _orient_kw(self):
        R = self._orient_load()
        return {"orient": R.tolist()} if R is not None else {}

    def _build_orient_ui(self):
        body = self.panel["meshed"].bodies["Orient"]; box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 4, 0, 0); v.setSpacing(6)
        info = QLabel("Click the face that sits on the bed. Pick three or more spots on that face."); info.setWordWrap(True); info.setProperty("role", "hint"); v.addWidget(info)
        self.orient_count = QLabel(""); self.orient_count.setFont(ui_font(13, QFont.Weight.DemiBold)); v.addWidget(self.orient_count)
        self.orient_list = QWidget(); QVBoxLayout(self.orient_list).setContentsMargins(0, 0, 0, 0); self.orient_list.layout().setSpacing(3); v.addWidget(self.orient_list)
        row = QHBoxLayout(); row.setSpacing(6)
        self.orient_undo = button(W.ORIENT["undo"], None); self.orient_clear = button(W.ORIENT["clear"], None)
        for b_ in (self.orient_undo, self.orient_clear): b_.setMinimumHeight(32); row.addWidget(b_)
        v.addLayout(row)
        self.orient_bottom = big_button(W.ORIENT["setbottom"], "apply"); v.addWidget(self.orient_bottom)
        self.orient_front = big_button("Use this view as front", None); v.addWidget(self.orient_front)
        self.orient_reset = button(W.ORIENT["reset"], None); self.orient_reset.setMinimumHeight(32); v.addWidget(self.orient_reset)
        self.orient_state = QLabel(""); self.orient_state.setWordWrap(True); self.orient_state.setProperty("role", "hint"); v.addWidget(self.orient_state)
        self.orient_undo.clicked.connect(self._orient_undo); self.orient_clear.clicked.connect(self._orient_clear)
        self.orient_bottom.clicked.connect(self._orient_set_bottom); self.orient_front.clicked.connect(self._orient_set_front)
        self.orient_reset.clicked.connect(self._orient_reset)
        body.v.insertWidget(max(0, body.v.count() - 1), box); self.orient_box = box; self._orient_spots = []; self._rail_orient = False

    def _orient_item(self):
        v = self.vp.view
        return next(iter(v.items.items()), (None, None)) if self.gl and v.items else (None, None)

    def _orient_on(self):
        return bool(self.gl and getattr(self, "_rail_orient", False) and self.stage == "meshed" and self._orient_item()[0] is not None)

    def _orient_matrix(self, key):
        """The 4x4 that shows the item turned by the saved orientation about its centre (identity when none is saved)."""
        R = self._orient_load(); it = self.vp.view.items.get(key)
        if R is None or it is None: return np.eye(4)
        M = np.eye(4); M[:3, :3] = R; c = self._orient_centre(it); M[:3, 3] = c - R @ c; return M

    def _orient_centre(self, it):
        V = it["V"] if it["kind"] == "mesh" else it["P"]; return np.asarray(V[::max(1, len(V) // 20000)], float).mean(0)

    def _orient_before(self):
        """Compare / peek: the Before pane's model turned the same way, about the main model's centre, so the two sit
        alike under the linked cameras (it showed the model unturned beside the turned one, 2026-10-22)."""
        if not self.gl or self.stage not in ("meshed", "report"): return
        bv = self.vp.before.view; key, it = self._orient_item()
        R = self._orient_load()
        for bkey, bit in list(bv.items.items()):
            M = np.eye(4)
            if R is not None: c = self._orient_centre(it if it is not None else bit); M[:3, :3] = R; M[:3, 3] = c - R @ c
            bv.set_item_model(bkey, M)

    def _orient_view(self):
        """Meshed stage: the model is shown turned by the saved orientation; on the Orient block the pick tool is armed."""
        if not self.gl or not hasattr(self, "orient_box") or self.stage not in ("meshed", "report"): return      # the orientation belongs to the finished model only: in Line-up the first item is a SIDE
        v = self.vp.view; key, it = self._orient_item()
        if key is None: return
        M = self._orient_matrix(key); v.set_item_model(key, M); self._orient_before()
        n = len(self._orient_spots); on = self._orient_on()
        self.orient_count.setText(W.ORIENT["count"].format(n=n, s="" if n == 1 else "s"))
        self._fill_spot_list(self.orient_list, self._orient_spots, self._orient_remove)
        self.orient_undo.setEnabled(n > 0); self.orient_clear.setEnabled(n > 0); self.orient_bottom.setEnabled(n >= 3)
        self.orient_state.setText(W.ORIENT["state_set"] if self._orient_load() is not None else W.ORIENT["state_none"])
        if on:
            if v.tool != "pairpick": v.set_tool("pairpick")
            v.set_markers([(M[:3, :3] @ np.asarray(x) + M[:3, 3], W.PAIR_COLOURS[i % len(W.PAIR_COLOURS)], i + 1) for i, x in enumerate(self._orient_spots)])
        elif v.tool == "pairpick" and not self._points_on() and not self._spots_on() and not self._cut_on() and not self._clone_on() and not self._fix_on() and not self._mirror_on():
            v.set_tool("orbit"); v.set_markers([])

    def _orient_add(self, world):
        key, it = self._orient_item()
        if key is None: return
        M = np.asarray(it.get("model"), float); Mi = np.linalg.inv(M)
        self._orient_spots.append([float(x) for x in (Mi[:3, :3] @ np.asarray(world, float) + Mi[:3, 3])]); self._orient_view()

    def _orient_remove(self, k):
        if 0 <= k < len(self._orient_spots): self._orient_spots.pop(k); self._orient_view()

    def _orient_undo(self):
        if self._orient_spots: self._orient_spots.pop(); self._orient_view()

    def _orient_clear(self):
        self._orient_spots = []; self._orient_view()

    def _orient_set_bottom(self):
        key, it = self._orient_item(); pts = np.asarray(self._orient_spots, float)
        if key is None: self.notice(W.ORIENT["nomodel"]); return
        if len(pts) < 3: self.notice(W.ORIENT["need"]); return
        V = it["V"] if it["kind"] == "mesh" else it["P"]
        R = bottom_rotation(pts, V[::max(1, len(V) // 50000)])
        if R is None: self.notice(W.ORIENT["flat"]); return
        M0 = self._orient_matrix(key); self._orient_save(R); self._orient_spots = []; self._orient_view(); self._orient_follow(key, M0)
        self.notice(W.ORIENT["saved_bottom"].format(n=len(pts)))

    def _orient_set_front(self):
        """Turn about the up axis so the way the camera looks now becomes the model's front (looking toward +y)."""
        key, it = self._orient_item()
        if key is None: self.notice(W.ORIENT["nomodel"]); return
        R0 = self._orient_load(); R0 = np.eye(3) if R0 is None else R0
        fw = -np.asarray(self.vp.view.cam.view, float)[2, :3]               # the world direction the camera looks along
        M0 = self._orient_matrix(key); self._orient_save(front_rotation(fw, R0)); self._orient_view(); self._orient_follow(key, M0)
        self.notice(W.ORIENT["saved_front"])

    def _orient_follow(self, key, M0):
        """The model was just turned by the new orientation (M0 -> its matrix now): turn the camera the same way, so the
        picture stays as the user set it up and the cube now reads Front. Before this the model swung away under a camera
        that stayed put ("it moves it but does not set it", 2026-10-22)."""
        v = self.vp.view; M1 = self._orient_matrix(key); D = M1 @ np.linalg.inv(M0)
        if np.allclose(D, np.eye(4)): return
        cam = v.cam; cam.view = np.asarray(cam.view, float) @ np.linalg.inv(D)
        cam.pivot = D[:3, :3] @ np.asarray(cam.pivot, float) + D[:3, 3]
        v.camera_moved.emit(); v.update()

    def _orient_reset(self):
        self._orient_save(None); self._orient_spots = []; self._orient_view(); self.notice(W.ORIENT["cleared"])

    ROW_TABS = {s["id"]: (s["stage"], s["tab"]) for s in W.STAGES if s.get("tab")}          # a rail row -> (stage, its tab)

    def _pick_stage(self, sid):
        """A rail row was clicked. A step row opens its stage's panel on that tab; Orient jumps to its block."""
        self._rail_orient = sid == "orient"
        if sid == "ms_cleanup":                                   # Clean up: the Edit page on the finished model
            self.show_stage("meshed"); self.stages.set_current(sid); self.edit_requested.emit(); return
        if sid in self.ROW_TABS:
            stage, tab = self.ROW_TABS[sid]
            self.show_stage(stage); self.panel[stage].show_tab(tab); self._after_tab(); self.stages.set_current(sid); return
        if sid in ("side", "fused", "meshed"):
            first = next(k for k, (st, _t) in self.ROW_TABS.items() if st == sid); self._pick_stage(first); return
        if sid == "orient":
            self.show_stage("meshed"); self._rail_orient = True; self.stages.set_current("orient"); self._orient_view()
            self.scroll.ensureWidgetVisible(self.panel["meshed"].bodies["Orient"], 0, 40); return
        self.show_stage(sid)

    def _more_tools_menu(self):
        """Meshed > More tools: the tools that were behind the tab strip's "...", as a menu."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QCursor
        menu = QMenu(self)
        for t in W.MORE_TOOLS.values():
            act = menu.addAction(t["name"]); act.triggered.connect(lambda _=False, n=t["name"]: self._open_tool_tab(n))
        menu.exec(QCursor.pos())

    def _open_tool_tab(self, name):
        self.show_stage("meshed"); self.panel["meshed"].show_tab(name); self._after_tab()

    def _rail_current(self, sid):
        """Which rail row is lit: on a stage with step rows, the row whose tab is shown."""
        if sid == "meshed" and getattr(self, "_rail_orient", False): return "orient"
        if sid in ("side", "fused", "meshed") and getattr(self, "panel", None) and sid in self.panel:
            tab = self.panel[sid].current
            return next((k for k, (st, t) in self.ROW_TABS.items() if st == sid and t == tab), "ms_mesh" if sid == "meshed" else sid)
        return sid

    def _base_state(self):
        """Count only current successful cuts, never merely configured cut methods."""
        pl = [n for n in self._nodes("side") if n["id"].startswith("side.platter@")]
        if not pl: return "", "", None, None
        no_base=[n for n in pl if (n.get("stats") or {}).get("base_cut_skipped")]
        if len(no_base)==len(pl):return "No base in view: nothing to cut","",None,None
        pl=[n for n in pl if n not in no_base]
        enabled=[n for n in pl if n.get("method")!="none"]
        built = [n for n in enabled if n.get("status") in ("fresh", "adopted", "done", "cached", "warn", "check")
                 and (n.get("stats") or {}).get("source") != "failed" and self.snap.out(n["id"], "plane")]
        cut = len(built); sub = "Cut applied on %d of %d" % (cut, len(pl))
        def uncut(n):                                     # a fallback that still cut (plate rejected, turntable used) is fine, not a failure
            src = str((n.get("stats") or {}).get("source") or "")
            return src == "failed" or src.startswith("uncut") or n.get("status") in ("failed", "error")
        bad = [n.get("n") for n in enabled if uncut(n)]
        if bad:                                           # which sides, and what to do there: a first-timer can't act on "Check cuts"
            ns = [str(x) for x in bad if x]; one = len(bad) == 1
            who = ("side " + ns[0]) if one and ns else ("%d side%s" % (len(bad), "" if one else "s"))
            return sub, "Check " + who, "warn", ("%s: Auto left the base on. If you still see base there, open %s and use Manual."
                                                 % (("Side%s %s" % ("" if one else "s", ", ".join(ns))) if ns else "A side", "it" if one else "each"))
        if len(built) < len(enabled): return sub, "Needs build", "warn", "Press Apply or Rebuild: some base cuts are out of date."
        if cut == 0: return sub, "Not removed", "warn", "Pick how to cut the base, then press Apply. No base cut is in use yet."
        if cut < len(pl): return sub, "Some sides", "warn", "Pick the sides to cut, then press Apply. Some sides still have their base."
        return sub, "", None, "Every side has its base cut. Check the result before going on."

    def _refresh_rows(self):
        if "orient" in self.stages.rows and self.snap is not None and self.snap.sides:
            self.stages.rows["orient"].set_info("Set" if self._orient_load() is not None else "Not set, optional", dim=self._orient_load() is None)
        if self.snap is not None and not self.snap.sides:       # nothing scanned yet: every row blank and gray, no states
            for sid, r in self.stages.rows.items():
                if sid != "textured": r.set_info("", "", None, None, dim=True)
            return
        if "base" in self.stages.rows:
            sub, text, colour, tip = self._base_state(); self.stages.rows["base"].set_info(sub, text, colour, tip)
            keep = [n for n in self._nodes("side") if n["id"].startswith("side.keep@")]
            share = (keep[0].get("params") or {}).get("min_share") if keep else None
            self.stages.rows["isolation"].set_info("Drops pieces under %g%%" % (100 * float(share)) if share is not None else "Drops loose pieces")
            fuse = [n for n in self._nodes("side") if n["id"].startswith("side.fuse@")]
            vox = (fuse[0].get("params") or {}).get("voxel") if fuse else None
            self.stages.rows["sidefusion"].set_info("Point spacing %g mm" % float(vox) if vox else "Point spacing")
        for sid in STAGE_IDS:
            r = self.stages.rows[sid]; text, colour, tip, dim = self._row_state(sid)
            r.set_info(self._stage_sub(sid), text, colour, tip, dim=dim)

    def _refresh_info(self):
        s = self.snap
        if s is None: self.info.set_lines([W.NOTICE["loading"] if self.host else ""]); return
        raw = self._nodes("raw"); fr = sum(int((n.get("stats") or {}).get("frames") or 0) for n in raw)
        l1 = W.INFO["line1"].format(sides=len(raw), frames=fr)
        if len(raw) == 1: l1 = l1.replace("1 sides", "1 side")
        mod = self._camera_module(); l2 = W.INFO["line2"].format(module=mod) if mod else ""
        dims = self._model_dims(self._export_src()) if self.stage == "meshed" and self.isVisible() else None
        if dims: l2 = (l2 + "    " if l2 else "") + W.INFO["size"].format(dims=dims)
        gb = self._project_size(); sizes = ([W.INFO["project"].format(size=gb)] if gb else []) + self._file_line()
        self.info.set_lines([l1, l2, self._count_line(), "    ".join(sizes)])

    def _project_size(self):
        """The project's size on disk, measured once off the thread and
        shown when it lands; '' until then."""
        from app2 import disk_usage
        root = self.host_root
        if not root: return ""
        n = disk_usage.project_bytes(root)
        if n is None:
            if not getattr(self, "_project_sized_wired", False):
                self._project_sized_wired = True; self.project_sized.connect(lambda p, b: p == self.host_root and self._refresh_info())
            if getattr(self, "_sizing", None) != root:
                self._sizing = root; disk_usage.measure([root], lambda p, b: self.project_sized.emit(p, b))
            return ""
        return disk_usage.nice(n)

    def _file_line(self):
        """The model file on screen and about what its STL would weigh (binary STL: 50 bytes a triangle), as pieces."""
        from app2 import disk_usage
        path = self.shown.get("path") if self.shown.get("kind") == "mesh" else None
        if not path or not os.path.isfile(path): return []
        try: n = os.path.getsize(path)
        except OSError: return []
        faces = self.shown.get("n") or 0
        out = [W.INFO["file"].format(size=disk_usage.nice(n))]
        if faces: out.append(W.INFO["stl"].format(size=disk_usage.nice(84 + 50 * int(faces))))
        return out

    def _camera_module(self):
        """"Near" / "Far" from the project's property.rvproj when there is one (hidden when unknown)."""
        if not self.host_root: return None
        if getattr(self, "_module_for", None) == self.host_root:
            from app2.scan.object_modes import label
            return label(self._module) if self._module is not None else None
        self._module_for = self.host_root; self._module = None
        import glob, re
        for p in glob.glob(os.path.join(self.host_root, "property.rvproj")) + glob.glob(os.path.join(self.host_root, "data", "*", "property.rvproj"))[:1]:
            try:
                txt = open(p, errors="replace").read(20000)
                m = re.search(r'"?(?:camera_?module|cameraModule)"?\s*[:=]\s*"?(\w+)', txt, re.I)
                if m: self._module = m.group(1).capitalize(); break
            except OSError: pass
        from app2.scan.object_modes import label
        return label(self._module) if self._module is not None else None

    def _count_line(self):
        s = self.snap; sid = self.stage
        if sid == "raw":
            side = self._side(); n = s.node("raw@" + side["id"]) if side else None
            return W.INFO["raw"].format(frames=int(((n or {}).get("stats") or {}).get("frames") or 0))
        if sid == "side":
            side = self._side()
            if not side: return ""
            # the list end's own count (an adopted or tool step has no "faces" stat: that printed Faces: 0), else the file's
            f = self._faces_of(self._view_path()) or ((s.node("side.keep@" + side["id"]) or {}).get("stats") or {}).get("faces")
            return W.INFO["side"].format(n=side["n"], sides=len(s.sides), faces=f) if f else W.INFO["side_head"].format(n=side["n"], sides=len(s.sides))
        if sid == "lineup":
            ag = ((s.node("lineup.refine") or {}).get("stats") or {}).get("agreement") or {}
            active = sum(1 for x in s.sides if x["active"])
            if ag.get("after") is None: return W.INFO["lineup_head"].format(active=active, sides=len(s.sides))
            return W.INFO["lineup"].format(active=active, sides=len(s.sides), agree="%.2f" % ag["after"])
        if sid == "meshed" and self.shown.get("published"):           # the published model.ply on screen: its own counts
            f, v = self.shown.get("n"), self.shown.get("nv")
            return (W.INFO["mesh"].format(verts=v, faces=f) if v else W.INFO["faces"].format(faces=f)) if f else ""
        if sid in ("fused", "meshed"):
            n = s.node(s.final(sid)) or {}; st = n.get("stats") or {}
            here = self.shown.get("path") == self._view_path() and self.shown.get("n")      # the result on screen
            if sid == "fused" and s.fused_kind == "points":
                pts = st.get("points") or st.get("out") or (self.shown.get("n") if here else None)
                return W.INFO["points"].format(points=pts) if pts else ""
            # a mesh: the step's own counts, else the result's on screen (a Poisson step's stats name only its input points)
            faces = st.get("faces") or (self.shown.get("n") if here and self.shown.get("kind") == "mesh" else None) or self._faces_of(self._view_path())
            verts = st.get("verts") or (self.shown.get("nv") if here else None)
            if not faces: return ""
            return W.INFO["mesh"].format(verts=verts, faces=faces) if verts else W.INFO["faces"].format(faces=faces)
        if sid == "report":
            st = (s.node("report") or {}).get("stats") or {}
            return W.INFO["report"].format(edge_mm=int(round(st.get("open_edge_mm", 0))), places=len(st.get("places") or [])) if st else ""
        return ""

    def _faces_of(self, path):
        """Polygons of a mesh file: the loaded result's when it is on screen, else its PLY header (read once per path)."""
        if not path: return None
        if self.shown.get("path") == path and self.shown.get("kind") == "mesh" and self.shown.get("n"): return int(self.shown["n"])
        c = self.__dict__.setdefault("_face_counts", {})
        if path not in c:
            try:
                from stages import plyio
                c[path] = int(plyio.counts(path).get("face") or 0) if path.endswith(".ply") and os.path.exists(path) else None
            except Exception: c[path] = None
        return c[path]

    # ---------------- the panel
    def _recipe_value(self, path):
        from stages import params as PR
        r = (self.snap.recipe if self.snap else None) or {}
        try: return PR.get_path(r, path)
        except Exception: return None

    def _list_token(self, sid=None):
        sid = sid or self.stage
        if sid == "side":
            side = self._side(); return "side@" + side["id"] if side else None
        return {"fused": "fused", "meshed": "meshed"}.get(sid)

    def _list_kind(self, sid=None):
        sid = sid or self.stage
        return (self.snap.fused_kind if self.snap else "mesh") if sid == "fused" else "mesh"

    def _refresh_panel(self):
        s = self.snap; sid = self.stage; p = self.panel[sid]
        if s is None: return
        self.mesh_alias.setVisible(sid=="meshed" and (s.meshed_method=="as_fused" or (s.meshed_method=="auto" and s.fused_kind=="mesh")))
        if sid in ("side", "fused", "meshed", "raw"):
            kind = self._list_kind(sid)
            tags = {kind}
            if sid == "fused":
                fm = self._ctl(p.bodies["Fusion"], "fused.method").value() if self._ctl(p.bodies["Fusion"], "fused.method") else s.fused_method
                tags |= {fm}
                p.set_more_tools(W.PANELS["fused"]["more_tools_mesh"] if kind == "mesh" else [])
                ov = p.bodies["Overlap Detection"]; on = kind == "points"
                p.tabs.set_tab_enabled("Overlap Detection", on, None if on else W.tab_of("fused", "Overlap Detection")["off_note"])
                ov.set_note("" if on else W.tab_of("fused", "Overlap Detection")["off_note"])
                if ov.main is not None: ov.main.setProperty("off", not on)
            if sid == "meshed":
                mm = self._ctl(p.bodies["Mesh"], "meshed.method").value() if self._ctl(p.bodies["Mesh"], "meshed.method") else s.meshed_method
                eff = mm if mm != "auto" else ("as_fused" if s.fused_kind == "mesh" else "poisson")
                tags |= {eff}
                seg = self._ctl(p.bodies["Mesh"], "meshed.method")
                seg.set_choice("as_fused", enabled=s.fused_kind == "mesh", tip=None if s.fused_kind == "mesh" else W.NOTICE["as_fused_points"])
                p.set_more_tools(W.PANELS["meshed"]["more_tools_mesh"])
            p.show_if(tags)
            if sid == "meshed": self._poisson_in_sight(p, "poisson" in tags)
            for name, b in p.bodies.items():
                for c in b.controls:
                    if c.c.get("path"):
                        path = c.c["path"]
                        if "{id}" in path or path.startswith("side["):
                            side = self._side(); path = path.replace("{id}", side["id"] if side else "x")
                        if b.apply_to is not None and b.apply_to.value() == "this" and path.startswith("side.") and self._side() \
                                and path not in ("side.platter.margin", "side.platter.max_tilt"):
                            path = "side[%s].%s" % (self._side()["id"], path[5:])
                        if path == "side[%s].platter.method" % (self._side() or {}).get("id") or path.endswith("platter.method"):
                            v = self._platter_method()
                        else: v = self._recipe_value(path)
                        if v is not None and (c.engine_value is None or not self._same(v, c.engine_value)): c.set_engine(v)
                    elif c.c.get("tool") and c.engine_value is None and not (self.editing and self.editing[1] == name):
                        c.set_engine(self._tool_start(c, kind))
            self._panel_extras(sid)
        elif sid == "lineup": self._refresh_lineup_panel()
        elif sid == "report":
            rp = self.panel["report"]; n = s.node("report") or {}
            rp.set_report(n.get("stats") if n.get("status") in DONE_STATES else None, self._recipe_value("report.min_mm") or 8)
            rp.set_guide(getattr(self,"_scan_guides",{}).get((self._display_path(),self.colour_mode.get("report"))))
            for c in rp.settings.controls:
                v = self._recipe_value(c.c["path"])
                if v is not None and (c.engine_value is None or not self._same(v, c.engine_value)): c.set_engine(v)

    def _poisson_in_sight(self, p, on):
        """The Mesh tab has Quality and Auto hole filling: keep them in sight even when the mesh is taken as fused,
        off, with a line saying how to use them. Their values are not sent then (shown stays False)."""
        b = p.bodies["Mesh"]
        for c in b.controls:
            if c.c.get("show_if") == "poisson" and not c.c.get("advanced"):
                c.setEnabled(on)
                if on: c.setGraphicsEffect(None)
                elif c.graphicsEffect() is None:
                    from PySide6.QtWidgets import QGraphicsOpacityEffect
                    fx = QGraphicsOpacityEffect(c); fx.setOpacity(0.4); c.setGraphicsEffect(fx)
                if not on: c.setVisible(True)
                if getattr(c, "sep", None) is not None: c.sep.setVisible(True)
        b.set_note("" if on else W.tab_of("meshed", "Mesh")["poisson_off"])

    @staticmethod
    def _same(a, b):
        try: return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError): return a == b

    def _platter_method(self):
        side = self._side(); r = (self.snap.recipe if self.snap else None) or {}
        ov = ((r.get("sides") or {}).get("platter") or {}).get(side["id"]) if side else None
        if ov and ov.get("method"): return ov["method"]
        return self._recipe_value("side.platter.method") or "auto"

    def _tool_start(self, ctl, kind):
        """Where a new tool step's slider starts: the factory value when the scanner has the tool, else the engine's default."""
        if ctl.c.get("revo") is not None and getattr(ctl, "start", None) is not None and ctl.start != getattr(ctl, "default", None):
            return ctl.start
        return self._tool_default(ctl.c, kind)

    def _tool_default(self, c, kind):
        from stages import params as PR
        try: return PR.resolve_tool(c["tool"], kind, {}).get(c["param"])
        except Exception:
            d = c["default"]; return d[kind] if isinstance(d, dict) else d

    def _panel_extras(self, sid):
        """Hints that depend on the result: the grid size, Simplify's counts, the low-memory warning, Raw frames' rows."""
        s = self.snap; p = self.panel[sid]
        if sid == "fused":
            b = p.bodies["Fusion"]; seg = self._ctl(b, "fused.method"); pdc = next((c for c in b.controls if c.c.get("path") == "fused.point_distance"), None)
            if pdc is not None:
                low = seg.value() == "tsdf" and pdc.value() < 0.25
                pdc.set_warn(W.tab_of("fused", "Fusion")["low_memory_warn"] if low else "")
        if sid == "meshed":
            b = p.bodies["Mesh"]; q = next((c for c in b.controls if c.c.get("path") == "meshed.poisson.quality"), None)
            if q is not None:
                from stages import params as PR
                bb = ((s.node("fused") or {}).get("stats") or {}).get("bbox_mm")
                t = W.tab_of("meshed", "Mesh")
                if bb:
                    pl = PR.poisson_plan(max(bb), int(round(q.value())))
                    q.set_extra(t["grid_same"].format(grid=pl["grid_mm"], same="Q%d" % pl["same_as"]) if pl.get("same_as") else t["grid"].format(grid=pl["grid_mm"]))
                else: q.set_extra(t["grid_unknown"])
            self._finish_extras(p, s)
            sb = p.bodies.get("Simplify")
            keep = self._ctl(sb, param="keep_pct") if sb is not None else None
            if keep is not None:
                st = (s.node(s.final("meshed")) or {}).get("stats") or {}; f = st.get("faces")
                if not f and self.stage == "meshed" and self.shown.get("path") == self._view_path() and self.shown.get("kind") == "mesh":
                    f = self.shown.get("n")
                t2 = next(c for c in W.tab_of("meshed", "Simplify")["controls"] if c.get("param") == "keep_pct")
                keep.set_hint(t2["hint"].format(before=f, after=int(f * keep.value() / 100)) if f else t2["hint_plain"])
        if sid == "raw":
            import datetime, statistics
            def started(sid_):
                try: return datetime.datetime.strptime(sid_, "%m%d%Y%H%M%S")      # a side's folder name is the time it started
                except ValueError: return None
            rows = []; lines = []; apart = {}; times = [started(x["id"]) for x in s.sides]
            known = [t for t in times if t]
            mid = statistics.median(t.timestamp() for t in known) if len(known) >= 3 else None     # fewer than 3 sides: no "the others"
            for x, t in zip(s.sides, times):
                n = s.node("raw@" + x["id"]) or {}; st = n.get("stats") or {}
                plane = os.path.exists(os.path.join(self.host_root or "", "data", x["id"], "table_plane.json"))
                when = t.strftime("%b %d %H:%M").replace(" 0", " ") if t else "time unknown"
                gap = abs(t.timestamp() - mid) if (t and mid is not None) else 0
                rows.append((x["n"], W.PANELS["raw"]["tabs"][0]["rows"].format(n=x["n"], when=when, frames=st.get("frames", 0),
                                                                            lost=("    %d not tracked" % st["lost"]) if st.get("lost") else "") + ("" if x["active"] else "    left out"), gap > 6 * 3600))
                if gap > 6 * 3600:
                    apart.setdefault(("%d days apart" % round(gap / 86400)) if gap >= 36 * 3600 else ("%d hours apart" % round(gap / 3600)), []).append(x["n"])
            for gap_text, ns in apart.items():                    # one note per gap, runs of sides as "13-19"
                runs = []
                for n in sorted(ns):
                    if runs and n == runs[-1][1] + 1: runs[-1][1] = n
                    else: runs.append([n, n])
                names = ", ".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in runs)
                lines.append(W.PANELS["raw"]["tabs"][0]["apart" if len(ns) == 1 else "apart_many"].format(n=names, ns=names, gap=gap_text))
            p.body("").set_sides(rows); p.body("").set_note("\n".join(lines))
        if sid == "side":
            side = self._side(); b = p.bodies["Platter"]
            self._refresh_cut_status()
            seg = self._ctl(b, "platter.method")
            if self._all_mode() and seg is not None and seg.value() == "manual": b.set_note(MANUAL_SIDE)

    def _finish_extras(self, p, s):
        """Finish for print: the STL size next to Target triangles, the model's triangles now, Fill holes in mm."""
        b = p.bodies.get(FINISH)
        if b is None: return
        t = next(b_ for b_ in W.PANELS["meshed"]["blocks"] if b_["name"] == "Finish for print"); ctl = {c.c.get("param"): c for c in b.controls}
        fc = ctl.get("faces")
        if fc is not None:
            from app2.builder.export_stl import stl_size
            fc.set_extra(t["stl"].format(mb="%.1f" % (stl_size(fc.value()) / 1e6)))
            f = ((s.node(s.final("meshed")) or {}).get("stats") or {}).get("faces") if not self.shown.get("published") else self.shown.get("n")
            if not f and self.stage == "meshed" and self.shown.get("path") == self._view_path() and self.shown.get("kind") == "mesh":
                f = self.shown.get("n")
            fc.set_hint(fc.c["hint"].format(now=t["now"].format(faces=f)) if f else fc.c["hint_plain"])
        hc = ctl.get("hole_edges"); dc = ctl.get("depth")
        if hc is not None:
            fn = s.node("fused") or {}; bb = (fn.get("stats") or {}).get("bbox_mm")
            depth = dc.value() if dc is not None else 9; pd = (fn.get("params") or {}).get("point_distance") or 0
            if bb:
                ext = float(max(bb))
                per = hole_mm(1, ext, depth, pd)            # whole mm once a step is a mm or more ("100 mm", not "99.90 mm")
                hc.set_numeric_units(per, " mm", 0 if per * hc.step >= 1 else 1)
                hc.set_shown_as(lambda e, x=ext, d=depth, sp=pd: t["holes_off"] if not e else
                                t["holes_mm"].format(mm=round(hole_mm(e, x, d, sp), 1 if hole_mm(e, x, d, sp) < 10 else 0)))
            else:
                hc.set_numeric_units(1.0, " edges", 0)
                hc.set_shown_as(lambda e: t["holes_off"] if not e else t["holes_edges"].format(edges=int(e)))

    def _refresh_lineup_panel(self):
        s = self.snap; lp = self.panel["lineup"]
        sc = ((s.node("lineup.place") or {}).get("stats") or {}).get("score") or {}
        ag = ((s.node("lineup.refine") or {}).get("stats") or {}).get("agreement")
        base = ((s.node("lineup.adjust") or {}).get("stats") or {}).get("base")
        pos = {x["id"]: x["n"] for x in s.sides}
        fx = list(enumerate((s.chains or {}).get("lineup.fix", [])))
        flips = [(fix_words(o, pos), "lineup.fix", i) for i, (lk, j, o) in fx if o.get("tool") != "turn"]
        nudges = [(fix_words(o, pos), "lineup.fix", i) for i, (lk, j, o) in fx if o.get("tool") == "turn"]
        nudges += [("Side %s: %s mm, %s deg" % (pos.get(o.get("side"), "?"), ",".join("%g" % x for x in o.get("t", [0, 0, 0])), ",".join("%g" % x for x in o.get("r", [0, 0, 0]))), "lineup.adjust", i)
                   for i, (lk, j, o) in enumerate((s.chains or {}).get("lineup.adjust", []))]
        lp.set_sides(s.sides, sc, base, (ag["before"], ag["after"]) if ag else None, flips, nudges, W.SIDE_COLOURS)
        lp.set_auto_out(self._auto_out_load())
        tags = {}
        for _lk, _j, o in (s.chains or {}).get("lineup.fix", []):
            if not o.get("on", True): continue
            sid_, tool = o.get("side"), o.get("tool")
            if tool == "match": tags.setdefault(sid_, []).append("matched by points to side %s" % pos.get(o.get("to"), "?"))
            elif tool == "flip": tags.setdefault(sid_, []).append("flipped")
            elif tool == "turn": tags.setdefault(sid_, []).append("turned")
        for _lk, _j, o in (s.chains or {}).get("lineup.adjust", []):
            if o.get("on", True) and o.get("side"): tags.setdefault(o["side"], []).append("nudged")
        for sid_, words in self._auto_disagreements(base, pos).items(): tags.setdefault(sid_, []).append(words)
        chk = self._load_checked()
        for sid_, against in chk.items():
            if sid_ in pos and against in pos: tags.setdefault(sid_, []).append("fine against side %s" % pos[against])
        lp.set_tags(tags)
        lp.set_checked(chk)
        hints = {}
        for part in str(self._recipe_value("lineup.place.hints") or "").split(","):
            sid_, _, how = part.strip().rpartition(":")
            if sid_ and how: hints[sid_] = how
        lp.set_hints(hints)
        for c in lp.settings.controls:
            v = self._recipe_value(c.c["path"])
            if v is not None and (c.engine_value is None or not self._same(v, c.engine_value)): c.set_engine(v)

    # ---------------- Applied list
    def shown_pieces(self):
        """How many separate pieces the stage's current result has (its last op's stats, else the stage's), or None."""
        s = self.snap; sid = self.stage
        if s is None or sid not in ("side", "fused", "meshed"): return None
        chain = self._chain()
        node = s.node("%s#%s" % (chain[-1][0], chain[-1][2].get("id"))) if chain else s.node(sid if sid != "side" else "side.keep@%s" % (self._side() or {}).get("id"))
        st = (node or {}).get("stats") or {}
        return int(st["pieces"]) if st.get("pieces") is not None else None

    def _chain(self, sid=None):
        tok = self._list_token(sid)
        if tok is None: return []
        return (self.snap.chains or {}).get(tok, []) if self.snap else []

    def _refresh_applied(self):
        s = self.snap; sid = self.stage
        if s is None or sid in ("raw", "report"): self.applied.hide(); return
        self.applied.show(); rows = []; sections = []; extra = []; ids = []
        if sid == "lineup":
            pos = {x["id"]: x["n"] for x in s.sides}
            for tok in ("lineup.fix", "lineup.adjust"):
                for lk, j, o in s.chains.get(tok, []):
                    rows.append(dict(label=fix_words(o, pos, True) if o["tool"] in ("flip", "match", "turn")
                                     else "Nudge side %s" % pos.get(o.get("side"), "?"), on=o.get("on", True), editable=False))
                    ids.append((tok, o.get("id")))
            self._applied_ids = ids; self.applied.set_rows(LABEL[sid], rows); return
        kind = self._list_kind(); what = "points" if kind == "points" else "polygons"; tok = self._list_token()
        for k, (lk, j, o) in enumerate(self._chain()):
            ids.append((tok, o.get("id")))
            if lk == "fused" and not any(x == k for x, _t in sections) and not any(r.get("_del") for r in rows): sections.append((k, W.APPLIED["deletes"]))
            node = s.node("%s#%s" % (lk, o.get("id"))) or {}; st = node.get("stats") or {}
            label = tool_words(o["tool"], o.get("params"), kind) if o["tool"] != "delete" else "Delete %s" % o.get("shape", "")
            stats = W.operation_stats(o["tool"], st, kind) if node.get("status") in DONE_STATES else ""
            chk = (W.APPLIED["gone"] if not int(st.get("removed") or 0) else W.APPLIED["check"]).format(now=st.get("removed", "?"), what=what, was=(o.get("recorded") or {}).get("count", "?")) if node.get("status") == "check" else None
            rows.append(dict(id=o.get("id"), recipe_key=repr((self.host_root, o)), cached=node.get("status") in DONE_STATES and o.get("on", True), label=label, stats=stats, on=o.get("on", True), editable=o["tool"] not in ("delete", "cut_plane"), check=chk,
                             editing=bool(self.editing and self.editing[2] == o.get("id")), _del=lk == "fused"))
        if sid == "fused" and s.parked:
            by = {}
            for lk, j, o in s.parked: by.setdefault(lk.split(".", 1)[1], 0); by[lk.split(".", 1)[1]] += 1
            for m, n in by.items(): extra.append(W.APPLIED["parked"].format(method=FUSED_NAMES.get(m, m), n=n, s="" if n == 1 else "s"))
        stage_name = LABEL[sid] if sid != "side" else "Side %d" % self._side()["n"] if self._side() else "Side"
        self._applied_ids = ids; self.applied.set_rows(stage_name, rows, extra, sections)

    def _refresh_footer(self):
        s = self.snap
        ent = (s.entries if s else None) or []; cur = s.cursor if s else -1
        can_undo = s is not None and cur is not None and cur > 0; can_redo = s is not None and cur is not None and cur < len(ent) - 1
        self.undo_b.setToolTip(W.BOTTOM["undo_tip"].format(note=ent[cur]["note"]) if can_undo else W.BOTTOM["nothing_undo"])
        self.redo_b.setToolTip(W.BOTTOM["redo_tip"].format(note=ent[cur + 1]["note"]) if can_redo else W.BOTTOM["nothing_redo"])
        self._can = (can_undo, can_redo)
        self.vers_b.setEnabled(self.stage in ("side", "lineup", "fused", "meshed", "report"))    # off, not gone: the row stays put

    def _refresh_rebuild(self):
        s = self.snap
        if s is None or self.running() or not self._nodes("raw"): self.vp.rebuild.show_text(None); return
        n = getattr(self, "pending_edits", 0)
        if n:                                                                 # deletes wait for one rebuild: the strip says so, in Edit too
            if self.vp.rebuild.btn is not None: self.vp.rebuild.btn.setText(W.REBUILD_BAR["pending_button"])
            self.vp.rebuild.show_text(W.REBUILD_BAR["pending_text"].format(n=n, s="" if n == 1 else "s")); return
        stale = [sid for sid in STAGE_IDS[1:] if self._row_state(sid)[0] in (W.ROW_STATE["stale"]["text"], W.ROW_STATE["waiting"]["text"])]
        if not stale: self.vp.rebuild.show_text(None); self.undo_text = None; return
        eta = mins(s.est_s)
        self.fresh = self._never_built()                                     # "Build it", not "Rebuild the rest"
        if self.vp.rebuild.btn is not None:
            self.vp.rebuild.btn.setText(W.RECIPE["build" if self.fresh else "rebuild"].format(name=self._recipe_text()))
        if self.fresh: text = W.REBUILD_BAR["fresh_text"].format(eta=eta)
        elif s.all_stale and len(stale) >= 4: text = W.NOTICE["scripts_changed"].format(eta=eta)
        elif self.undo_text: text = W.REBUILD_BAR["undo_text"].format(rest=join_words(LABEL[x] for x in stale), eta=eta)
        elif self.last_changed and STAGE_IDS.index(self.last_changed) < STAGE_IDS.index(stale[0]):
            text = W.REBUILD_BAR["text"].format(changed=LABEL[self.last_changed], rest=join_words(LABEL[x] for x in stale), eta=eta)
        else: text = W.REBUILD_BAR["plain_text"].format(rest=join_words(LABEL[x] for x in stale), eta=eta)
        self.vp.rebuild.show_text(text)

    # ---------------- tools on / off
    def running(self): return bool(self.state is not None and self.state.building)

    def _refresh_tools(self):
        run = self.running(); host_busy = self.busy_call is not None
        elsewhere = bool(self.snap is not None and self.snap.running and not run       # a CLI run holds the project's lock
                         and running_elsewhere(os.path.join(self.host_root or "", "stages")))   # snap lags our own run's end
        no_sides = self.snap is not None and not self._nodes("raw")
        on = self.snap is not None and not run and self.session is not None and not elsewhere and not self._shut and not no_sides
        self._tools_on = on and not host_busy
        off = None if (self._shut or run) else OFF["ended"] if self._ended else OFF["no_sides"] if no_sides else OFF["elsewhere"] if elsewhere else None
        if off == OFF["elsewhere"]:
            words = elsewhere_words(os.path.join(self.host_root or "", "stages"))
            if words: off = words + ". Tools come back when it is done."
        self.off_note.setText(off or ""); self.off_note.setVisible(bool(off))
        self.onetap.setToolTip(off or (W.ONETAP["tip"] + " (Ctrl+Shift+B)"))
        for sid, p in self.panel.items(): p.set_enabled_tools(on and not host_busy)
        self.applied.set_tools_enabled(on and not host_busy)
        for tool in (getattr(self, 'fin', None), getattr(self, 'slice_tool', None), getattr(self, 'drill', None)):
            if tool is not None:
                body = tool.body()
                if body is not None and body.main is not None:
                    body.main.setEnabled(on and not host_busy and not tool.pending and
                                         len(tool.anchors) >= tool.minimum_picks and self.gl and
                                         tool.preview_key in self.vp.view.items)
        can = getattr(self, "_can", (False, False))
        self.undo_b.setEnabled(on and can[0] and not host_busy); self.redo_b.setEnabled(on and can[1] and not host_busy)
        self.hist_b.setEnabled(self.snap is not None and not run)
        self.onetap.setEnabled(on and not host_busy)
        # Rectangle / Lasso
        why = None; show = self.stage in ("side", "fused", "meshed")
        if not self.gl: why = W.NOTICE["no_gl"]
        elif run: why = W.NOTICE["tools_off_run"]
        elif elsewhere: why = W.NOTICE["locked"]
        elif self.comparing: why = W.NOTICE["compare_off"]
        elif self.shown.get("published"): why = W.NOTICE["published"]
        elif self.shown.get("readonly"): why = W.NOTICE["too_big"]
        elif not self._view_path(): why = W.NOTICE["run_first"]
        elif self.shown.get("path") != self._view_path(): why = W.NOTICE["latest_only"]
        elif self.detected is not None: why = " "
        if why in (W.NOTICE["latest_only"], W.NOTICE["published"], W.NOTICE["run_first"]) and (self._spots_on() or self._orient_on()):
            why = None                                  # picking spots on the base is for a picture that is not the latest result (the side before its cut, a published model)
        self.vp.set_tools(why is None and on, (why or "").strip(), show)
        if self.gl: self.vp.view.tools_on = why is None and on and not host_busy
        for stage, ui in getattr(self, "cut_ui", {}).items():
            ui["go"].setEnabled(self._tools_on and self.stage == stage and why is None and len(self._cspots.get(stage, [])) >= 3)
        if hasattr(self, "spot_set"):
            hand_ready = self._tools_on and self._spots_on() and len(self._spots.get((self._side() or {}).get("id"), [])) >= 3
            self.spot_set.setEnabled(hand_ready); self.spot_allh.setEnabled(hand_ready)
        if (why is not None or not show) and self.tool not in ("orbit", "pairpick"): self.set_tool("orbit")
        if self.gl: self._points_view()
        self.vp.rebuild.btn.setEnabled(on and not host_busy)
        self.vp.rebuild.changed.emit()                          # the header pill mirrors the button's state

    @property
    def tool(self): return self.vp.view.tool if self.gl else "orbit"

    # ---------------- progress card
    def _refresh_progress(self):
        vp = self.vp
        if self.running():
            if self.run_mine is None:
                prog = self.prog                       # a build started from the Scan step: the same card as there
                overall, doing = run_overview(prog) if prog else ("", "")
                vp.card.show_state("Building model", prog.step if prog else W.PROGRESS["starting"], "", prog.frac if prog else None,
                                   prog.elapsed() if prog else 0, (getattr(prog, "plan", None) or {}).get("cached", 0) if prog else 0, self.stopping,
                                   overall=overall, doing=doing, nodes=getattr(prog, "nodes", None))
                return
            prog = self.prog
            active = [k for k, v in (getattr(prog, "nodes", None) or {}).items() if v.get("state") == "start"] if prog else []
            cur = active[0] if active else getattr(prog, "current", None) if prog else None
            what, _settings = self._what(active if len(active) > 1 else cur)      # several sides at once: "Sides 3 and 4: Fusion, Side 2: Cut base"
            if not what: what = getattr(prog, "step", "") or W.PROGRESS["starting"]
            live = ((getattr(prog, "nodes", None) or {}).get(cur or "", {}) if prog else {})
            params = (live.get("text") or "") if len(active) <= 1 else ""   # under the headline: what the step is doing now ("side 4 frame 635 of 635")
            title = W.PROGRESS["onetap"] if self.run_mine.what == "onetap" else W.PROGRESS["fusing"] if (cur or "").split("#")[0] == "fused" else W.PROGRESS["processing"]
            cached = (getattr(prog, "plan", None) or {}).get("cached", 0) if prog else 0
            frac = None; percent_label = "Build"
            if prog is not None:
                node = (getattr(prog, "nodes", None) or {}).get(cur or "", {})
                frac = prog.frac if live.get("frac") is not None or prog.frac > 0 else None
            overall, doing = self._run_overview(prog)
            vp.card.show_state(title, what, params, frac, prog.elapsed() if prog else 0, cached, self.stopping, percent_label=percent_label,
                               overall=overall, doing=doing, nodes=getattr(prog, "nodes", None))
        elif self.busy_call and self.busy_call.get("name") in ("h_detect", "h_side_detect", "h_mask_delete", "delete", "h_onetap_", "preview",
                                                               "h_export_stl", "h_export"):
            vp.card.show_state(W.PROGRESS["processing"], {"h_detect": "Detect", "h_side_detect": "Detect", "h_mask_delete": "Delete",
                                                          "delete": "Delete", "preview": "Preview",
                                                          "h_export_stl": W.EXPORT["title"], "h_export": W.EXPORT_SHEET["title"]}.get(self.busy_call["name"], ""), "", None,
                               time.time() - self.busy_call["since"], 0)
        elif getattr(self, "_one_shot_jobs", None):               # What is left, quality, a preview: the card in the view, not only the side line
            purpose, since = next(iter(self._one_shot_jobs.values()))
            vp.card.show_state(W.PROGRESS["processing"], purpose, "", None, time.monotonic() - since, 0)
        else: vp.card.hide()
        vp._place()

    def _run_overview(self, prog): return run_overview(prog)

    def _what(self, nid):
        """The progress lines "Now: Side 3: Fusion" / "Settings: 0.4 mm" for a node id."""
        if not nid: return "", ""
        if isinstance(nid, (list, tuple)):                 # several steps running side by side: one grouped line
            from app2.builder.progress_words import headline
            live = getattr(self.prog, "nodes", None) or {}
            nodes = {k: dict(live.get(k) or (self.snap.node(k) if self.snap else None) or {}, state="start") for k in nid}
            return headline(nodes, lambda k: self._what(k)[0]), ""
        s = self.snap; node = (s.node(nid) if s else None) or (getattr(self.prog, "nodes", None) or {}).get(nid)
        if "#" in nid:
            tool = (node or {}).get("tool") or (node or {}).get("method") or ""; key = "op." + tool
        else: key = nid.split("@")[0]
        what, ptmpl = W.PROGRESS_WHAT.get(key, (nid, ""))
        prm = dict((node or {}).get("params") or {})
        n = (node or {}).get("n") or ""
        prm.setdefault("min_share_pct", "%g" % (100 * float(prm.get("min_share", 0) or 0)))
        prm.setdefault("hole_mm_or_auto", "Auto" if not prm.get("hole_mm") else "%g mm" % prm["hole_mm"])
        prm.setdefault("count", len(prm.get("flips") or prm.get("nudges") or []))
        if key == "meshed" and (node or {}).get("method") != "poisson": return "Mesh", "as fused"
        what = what.replace("pair {k} of {n}", "pairs")
        try: w = what.format(n=n, k="", **{}); p = ptmpl.format(**{k: (("%g" % v) if isinstance(v, float) else v) for k, v in prm.items()}) if ptmpl else ""
        except (KeyError, ValueError, IndexError): w, p = what.replace("{n}", str(n)), ""
        if key == "meshed" and "quality" in prm: p = "%.1f" % float(prm["quality"])
        if key == "op.local_fix": w = "Filling the gap" if prm.get("action") == "fill" else "Smoothing the area"
        return w.replace("pair {k} of {n}", "pairs"), p

    def _tick(self):
        if self.running() or self.busy_call: self._refresh_progress()
        if self.snap is not None and self.snap.running and not self.running(): self._refresh_tools()      # a build elsewhere: its progress line
        if getattr(self,"_one_shot_jobs",None):
            purpose,started=next(iter(self._one_shot_jobs.values()))
            self.job_state.setText(purpose+" %.0f s elapsed" % (time.monotonic()-started))
        if self.notice_until and time.time() > self.notice_until: self.notice_until = 0; self.vp.notice.show_text(None)

    # ---------------- notices
    def notice(self, text, level="warn", secs=8):
        n = self.vp.notice
        if n.property("level") != level:
            n.setProperty("level", level); n.style().unpolish(n); n.style().polish(n)
        n.show_text(text); self.notice_until = time.time() + secs if text else 0
        n.text.setStyleSheet("color: %s;" % (T["warn"] if level == "warn" else T["accent_hi"]))

    def tell(self, text, level="warn", secs=8):
        """A notice the user must see wherever they are (Export's result): here, and on the shell's current page."""
        self.notice(text, level, secs); self.message.emit(text, level)

    # ================================================================ stages and sides
    def show_stage(self, sid, keep_published=False):
        if hasattr(self, "fin"): self.fin.clear(); self.slice_tool.clear()
        if hasattr(self, "drill"): self.drill.clear()
        if sid not in STAGE_IDS: return
        self.vp.guide_controls.hide()
        if self.gl:
            self.vp.view.scan_guide = None; self.vp.view.overlay.update()
        if sid != self.stage: self._clear_compare_change()
        if not keep_published: self.published = None
        self._clear_detect(); self.editing = None
        if self.gl: self.vp.view.clear_selection()
        if sid != "meshed": self._rail_orient = False
        if sid=="report":
            from app2.scan.guide_mode import selected_mode
            self.colour_mode["report"]="scan_guide_handheld" if selected_mode(self.host_root)=="handheld" else "scan_guide"
        self.stage = sid; self.stages.set_current(self._rail_current(sid)); self._show_panel(sid); self.scroll.verticalScrollBar().setValue(0)
        self.pager.set_page(self.side_k, len(self.snap.sides) if self.snap else 0, sid in ("raw", "side"), self._pager_note())
        if self.comparing: self.toggle_compare()
        self._refresh_all(); self.request_view()

    def _show_panel(self, sid):
        """Only the shown panel counts for the right column's size (a wide Line-up panel must not widen Fused's)."""
        if self.coverage_state.text() == "Coverage measured.": self.coverage_state.setVisible(sid != "raw")
        for k, p in self.panel.items():
            on = k == sid
            p.setSizePolicy(QSizePolicy.Policy.Preferred if on else QSizePolicy.Policy.Ignored,
                            QSizePolicy.Policy.Preferred if on else QSizePolicy.Policy.Ignored)
        self.panels.setCurrentWidget(self.panel[sid]); self.panels.updateGeometry()

    def step_stage(self, d):
        k = STAGE_IDS.index(self.stage) + d
        if 0 <= k < len(STAGE_IDS): self.show_stage(STAGE_IDS[k])

    def step_side(self, d):
        if self.stage not in ("raw", "side") or not self.snap: return
        n = len(self.snap.sides); k = max(1, min(n, self.side_k + d))
        if k == self.side_k: return
        self.side_k = k; self._clear_detect(); self.editing = None
        self.pager.set_page(k, n, True, self._pager_note(k)); self._refresh_all()
        self.request_view()

    def _after_tab(self):
        if hasattr(self, "fin"): self.fin.clear(); self.slice_tool.clear()
        if hasattr(self, "drill"): self.drill.clear()
        self._clear_detect(); self._refresh_panel(); self._refresh_tools(); self._points_view(); self._nudge_view()
        self.stages.set_current(self._rail_current(self.stage)); self._spots_view()
        if self.stage == "side" and self.snap is not None and (self._display_path() != self.shown.get("path")
                                                               or getattr(self, "_view_solo", False) != self._solo_wanted()): self.request_view()

    # ================================================================ the view
    def _view_path(self):
        """The file the current stage shows (the list end its tools edit)."""
        s = self.snap
        if s is None: return None
        side = self._side()
        if self.stage == "raw": return (s.out("side.fuse@" + side["id"]) or self._scanner_model(side)) if side else None
        if self.stage == "side": return self._built_end("side.keep@" + side["id"]) if side else None
        if self.stage in ("fused", "meshed"): return self._built_end(self.stage)
        if self.stage == "report": return s.out(s.final("meshed"))
        if self.stage == "lineup": return s.out("lineup.adjust", "lineup")
        return None

    def _built_end(self, token):
        """The list end's file; while deletes wait for their rebuild (2026-10-25) the newest built file before them, so
        the picture with the faces hidden stays instead of a blank "not built yet"."""
        s = self.snap; path = s.out(s.final(token))
        if path or not getattr(self, "pending_edits", 0): return path
        for step in reversed(list((s.lists or {}).get(token) or [])):
            path = s.out(step)
            if path: return path
        return None

    def _platter_pending(self):
        """Is a base cut waiting to be applied? Then the view shows the side before the cut with the violet (what would go); otherwise
        it shows the cut result, so the base is really gone after Remove base."""
        side = self._side()
        if side is None or self.snap is None: return False
        node = self.snap.node("side.platter@" + side["id"]) or {}
        edited = bool(self.panel["side"].bodies["Platter"].changed())
        # Manual: the base must stay on screen until the spots are picked and the cut is run, or there is nothing to click on;
        # once the spots have cut it, the result shows (a new spot brings the base back)
        if self._spots_on():
            return edited or node.get("status") not in DONE_STATES or \
                (len(self._spots.get(side["id"], [])) < 3 and (node.get("stats") or {}).get("source") != "manual")
        # Auto / Keep: the object as clean as it gets, never the base (the last cut while the side rebuilds). Only a Margin
        # being set shows the violet; Auto itself cuts at once (_cut_auto_apply)
        return edited

    def _cut_status(self, node, method, spots=0):
        """The Cut base status line: what the last build found on this side (Manual: whether the spots mark it yet), and how much
        the pending cut takes."""
        w = W.CUT; st = (node or {}).get("stats") or (node or {}).get("last_stats") or {}; src = st.get("source") or ""
        if st.get("base_cut_skipped") and method not in ("none","manual"):return "No base in view: nothing to cut",None
        if method == "none": return w["kept"], None
        if method == "manual": text, level = (w["hand_ready"] if spots >= 3 else w["found"]["manual"] if src == "manual" else w["hand_wait"]), None
        elif not st: return w["unbuilt"], None
        elif src == "failed": text, level = w["failed"], "warn"
        elif src.startswith("uncut"): text, level = w["crossed"], "warn"
        else: text, level = w["found"].get(src, w["found"]["auto"]), None
        share = getattr(self, "platter_share", 0.0) or 0.0
        if self._platter_preview() and share > 0:
            text += (w["share_big"] if share > 0.4 else w["share"]).format(pct=int(round(100 * share)))
            if share > 0.4: level = "warn"
        return text, level

    def _refresh_cut_status(self):
        side = self._side() if self.snap else None
        if side is None: return
        ctl = self._ctl(self.panel["side"].bodies["Platter"], "platter.method")
        if ctl is None or not hasattr(ctl, "set_status"): return
        ctl.set_status(*self._cut_status(self.snap.node("side.platter@" + side["id"]), ctl.value(), len(self._spots.get(side["id"], []))))

    def _cut_share(self, cut, body):
        """How much of this side the pending cut removes: in the chip, and a warning in the panel above 40%."""
        self.platter_share = float(np.mean(cut)) if len(cut) else 0.0
        self._refresh_cut_status()
        if self.platter_share > 0.4:
            body.set_msg(W.VIEW["cut_big"].format(pct=int(round(100 * self.platter_share))), "warn"); body.msg.setVisible(True); self._cut_msg_on = True
        elif self._cut_msg_on: body.set_msg(""); body.msg.setVisible(body.uses_second); self._cut_msg_on = False

    def _platter_preview(self):
        """Side stage on its Platter tab with a cut pending: the side before the cut, with what the cut takes in violet."""
        return self.stage == "side" and self.snap is not None and self._side() is not None and self.panel["side"].current == "Platter" \
            and bool(self.snap.out("side.fuse@" + self._side()["id"])) and self._platter_pending()

    def _display_path(self):
        if self._all_mode(): return "all-sides"
        if self.published and self.stage == "meshed": return self.published
        if self._platter_preview(): return self.snap.out("side.fuse@" + self._side()["id"])
        return self._view_path()

    def _platter_flags(self):
        """n . x < d + margin (keep_above) is cut: from the plane the last run used (R22) and the margin being set."""
        if not self.gl: return
        if not self._platter_preview():
            if getattr(self.vp.view, "would_go", None) is not None and getattr(self.vp.view, "_platter", False): self.vp.view.clear_would_go(); self._refresh_chips()
            return
        if self.shown.get("path") != self._display_path(): return
        import json
        v = self.vp.view; b = self.panel["side"].bodies["Platter"]; self.platter_cut = None
        def none():                                           # no violet: the share and the status line say so too, not the last cut's
            v.clear_would_go(); self._cut_share(np.zeros(0, bool), b); self._refresh_chips()
        if self._spots_on():                                  # Manual: the violet follows the spots as they are clicked
            plane = self._spots_plane()
            if plane is None or v.elements is None: none(); return
            cut = (np.asarray(v.elements[0], np.float32) @ np.asarray(plane["n"], np.float32)) < np.float32(plane["d"] + plane["margin"])
            v.show_would_go(cut, platter=True); self.platter_cut = int(cut.sum()); self._cut_share(cut, b); self._refresh_chips(); return
        pp = self.snap.out("side.platter@" + self._side()["id"], "plane")
        seg = self._ctl(b, "platter.method")
        if not pp or v.elements is None or (seg is not None and seg.value() == "none"): none(); return
        try:
            with open(pp) as f: pl = json.load(f)
        except (OSError, ValueError): none(); return
        mc = next((c for c in b.controls if c.c.get("path") == "side.platter.margin"), None)
        margin = float(mc.value()) if mc is not None else float(pl.get("margin", 1.0))
        key = (id(v.elements[0]), len(v.elements[0]), pp, tuple(pl["n"]))         # the heights over the plane, once per shown result and plane
        if getattr(self, "_platter_s", (None,))[0] != key:
            self._platter_s = (key, np.asarray(v.elements[0], np.float32) @ np.asarray(pl["n"], np.float32))
        s = self._platter_s[1]
        cut = s < np.float32(float(pl["d"]) + margin) if pl.get("keep_above", True) else s > np.float32(float(pl["d"]) - margin)
        v.show_would_go(cut, platter=True); self.platter_cut = int(cut.sum()); self._cut_share(cut, b); self._refresh_chips()

    def pick_colour(self, mode):
        self.vp.guide_controls.hide()
        if self.gl:
            self.vp.view.scan_guide = None; self.vp.view.overlay.update()
        if mode in ("scan_guide","scan_guide_handheld"):
            from app2.scan.guide_mode import select_mode
            try:select_mode(self.host_root,"handheld" if mode=="scan_guide_handheld" else "turntable")
            except OSError:self.notice("The scanner guide mode could not be saved.")
        self.colour_mode[self.stage] = mode
        if self.gl:
            self.vp.view.set_flat(mode == "flat")
            self.vp.before.view.set_flat(mode == "flat")
            self.vp.view.set_mesh_points(mode == "points")
            self.vp.before.view.set_mesh_points(mode == "points")
        self._refresh_colours()
        if mode in ("scan_guide", "scan_guide_handheld"): self._show_scan_guide(mode); return
        if mode == "capture_distance": self._show_capture_distance(); return
        if mode == "quality": self._show_quality(); return
        if mode in ("sides", "shaded", "scan_color"): self.request_view()

    def _show_scan_guide(self, mode="scan_guide"):
        path=self._display_path()
        if not path or not self.gl: return
        memo=getattr(self,"_scan_guides",None)
        if memo is None: memo=self._scan_guides={}
        key=(path,mode)
        if key in memo: self.request_view(); return
        pending=getattr(self,"_scan_guide_pending",set()); self._scan_guide_pending=pending
        if key in pending:return
        pending.add(key)
        self.notice("Checking shadows and useful scanner positions...", "info", 40)
        def got(result):
            pending.discard(key); memo[key]=result
            if self.colour_mode.get(self.stage)==mode: self.request_view()
        def fail(error):
            pending.discard(key)
            self.colour_mode[self.stage]="shaded"; self._refresh_colours()
            self.notice(str(error).splitlines()[0][:180])
        self.call("h_scan_guide",path,"handheld" if mode=="scan_guide_handheld" else "turntable",then=got,fail=fail)

    def _show_capture_distance(self):
        path=self._display_path()
        if not path or not self.gl:return
        side=self._side()['id'] if self.stage=='side' and self._side() else None
        def got(result):
            self.__dict__.setdefault('_capture_distance',{})[path]=result['rgb']
            c=result['counts'];self.notice('Scan the yellow and red spots again. Capture distance: %d best, %d OK, %d off, %d not measured (gray).' % (c['best'],c['ok'],c['off'],c['unmeasured']),'info')
            if self.colour_mode.get(self.stage)=='capture_distance':self.request_view()
        def fail(error):
            self.colour_mode[self.stage]='shaded';self._refresh_colours();self.notice(str(error).splitlines()[0])
        self.call('h_capture_distance',path,side,then=got,fail=fail)

    def _show_quality(self):
        """Color > Quality: each spot by how many frames saw it (red few: rough, blue many: smooth). Measured once per result, about 30 s."""
        path = self._display_path()
        if not path or not self.gl: return
        q = getattr(self, "_quality", None)
        if q is None: q = self._quality = {}
        if path in q: self.request_view(); return
        def got(r):
            q[path] = r["rgb"]; lo, mid, hi = r["share"]
            self.notice(W.VIEW["quality_done"].format(lo=round(100 * lo), mid=round(100 * mid), hi=round(100 * hi)), "info", 25)
            if self.colour_mode.get(self.stage) == "quality": self.request_view()
        def fail(e):
            self.colour_mode[self.stage] = "shaded"; self._refresh_colours(); self.notice(W.VIEW["quality_fail"].format(why=str(e).splitlines()[0][:160]))
        self.call("h_quality", path, then=got, fail=fail)

    def _prefetch_quality(self):
        """After a build: measure the finished model's quality map in the background (the same file h_quality reads, so
        Color > Quality opens at once). Its own thread and process at low priority; never while a build runs, never twice."""
        s = self.snap
        if s is None or self.running() or not self.host_root: return
        src = s.out(s.final("meshed")); sides = s.out("lineup.adjust", "sides")
        if not src or not sides or not os.path.exists(src) or not os.path.exists(sides): return
        import hashlib, subprocess, sys, threading
        if getattr(self, "_coverage_job", None): return
        st = os.stat(src); placed = os.stat(sides)
        key = "%s|%s|%s|%s|%s|%s" % (src, st.st_mtime_ns, st.st_size, sides, placed.st_mtime_ns, placed.st_size)
        busy = self.__dict__.setdefault("_quality_prefetch", set())
        if key in busy: return
        busy.add(key); root = self.host_root
        job = {"cancel": threading.Event()}; self._quality_job = job
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        def work():
            out = None
            try:
                from app2.builder.quality_cache import identity
                out = os.path.join(root, "build", "quality-%s.npy" % identity(src, sides))
                if os.path.exists(out): return
                os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".pre.npy"
                env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(here, "scanner"), here]))
                if job["cancel"].is_set(): return
                process = job["process"] = subprocess.Popen(["nice", "-n", "15", sys.executable, os.path.join(here, "scanner", "frame_quality.py"), root, sides, src,
                                    "--out", out + ".pre.q.npz", "--rgb", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                if job["cancel"].is_set(): process.terminate()
                process.communicate()
                if process.returncode == 0 and not job["cancel"].is_set() and os.path.exists(tmp) and not os.path.exists(out): os.replace(tmp, out)
            except Exception: pass
            finally:
                busy.discard(key)
                if out and os.path.exists(out + ".pre.npy"): os.remove(out + ".pre.npy")
        threading.Thread(target=work, daemon=True, name="quality-prefetch").start()

    def _refresh_colours(self):
        kind = self.shown.get("kind"); sid = self.stage
        if sid == "lineup" or not self.gl: self.vp.set_colours([], None); return
        names = dict(W.VIEW["colours"])
        modes = [("shaded", names["shaded"]), ("sides", names["sides"])] if kind == "points" else [("shaded", names["shaded"]), ("flat", names["flat"]), ("points", "Points")]
        if sid in ("fused","meshed","report") and kind=="mesh" and self._display_path():
            modes.extend((("scan_guide","What is left: table turns"),("scan_guide_handheld","What is left: scanner moves")))
        if sid in ("side","fused","meshed","report") and self._display_path():modes.extend((("quality","Frame support"),("capture_distance","Capture distance")))
        from stages.color import has_colors
        path=self._display_path()
        if path and has_colors(path):
            modes.insert(1,("scan_color","Scan color"))
            if __import__("app2.general_prefs",fromlist=["get"]).get("default_look")=="scan_color":self.colour_mode.setdefault(sid,"scan_color")   # else shaded (2026-10-24)
        self.vp.set_colours(modes, self.colour_mode.get(sid, "shaded"))

    def _node_for_view(self):
        s = self.snap; side = self._side()
        if self.stage == "raw": return s.node("side.fuse@" + side["id"]) if side else None
        if self.stage == "side": return s.node(s.final("side.keep@" + side["id"])) if side else None
        if self.stage == "lineup": return s.node("lineup.adjust")
        return s.node(s.final(self.stage)) or s.node(self.stage)

    def request_view(self, path=None, then=None):
        """Load the current stage's result into the view (GL), or show its preview picture (no GL)."""
        s = self.snap
        if s is None: return
        if self.gl: self._apply_home_view()
        self.view_id += 1
        path = path or self._display_path()
        if self.stage=="report" and path and self.gl:
            from app2.scan.guide_mode import selected_mode
            self.colour_mode.setdefault("report","scan_guide_handheld" if selected_mode(self.host_root)=="handheld" else "scan_guide")
            selected=self.colour_mode["report"]
            if selected in ("scan_guide","scan_guide_handheld") and (path,selected) not in getattr(self,"_scan_guides",{}):
                self._show_scan_guide(selected); return
        pub = bool(self.published) and self.stage == "meshed" and path == self.published
        if not self.gl:
            n = self._node_for_view() or {}; pv = n.get("preview")
            if pub: pv = os.path.join(self.host_root or "", "model.png")
            if pv and os.path.exists(pv):
                from PIL import Image
                try: self.vp.main.set_image(np.asarray(Image.open(pv).convert("RGB")))
                except Exception: self.vp.main.set_image(None)
            else: self.vp.main.set_image(None)
            self.shown = {"path": path, "readonly": True, "published": pub}; self._refresh_chips(); self._refresh_tools(); return
        if self.loader is None: return
        if self.stage == "lineup": self._request_lineup(); return
        if not path:
            self.view_id+=1
            self.vp.view.clear(); self.shown = {}; self._refresh_chips(); self._refresh_tools()
            self.vp.set_empty(W.VIEW["empty_busy"] if self.running() else None); return        # the view draws its own "not built yet" line
        self.vp.set_empty(None)
        self.view_id += 1
        if pub:
            if self.gl:
                self.vp.view.capture_colors=False;self.vp.view.overlay.update()
            self.pending_view = then
            self.loader.request(dict(id=self.view_id, slot="main", items=[dict(key="main", path=path)], edit=None, mode="shaded",
                                     stage=self.stage, path=path, readonly=True, published=True)); return
        editable = self.stage in ("side", "fused", "meshed")
        mode = self.colour_mode.get(self.stage,"shaded")
        colour = np.float32([int(T["cloud"][i:i + 2], 16) for i in (1, 3, 5)]) / 255.0 if self.stage == "raw" else None
        qrgb = None
        if self.colour_mode.get(self.stage) == "quality" and path in (getattr(self, "_quality", None) or {}):
            try: qrgb = np.load(self._quality[path]); mode = "quality"
            except OSError: qrgb = None
        if self.colour_mode.get(self.stage)=='capture_distance' and path in getattr(self,'_capture_distance',{}):
            qrgb=np.load(self._capture_distance[path]);mode='capture_distance'
        guide=(getattr(self,"_scan_guides",{}) or {}).get((path,self.colour_mode.get(self.stage)))
        if guide:
            qrgb=np.load(guide["rgb"]); mode="quality"
        if self.gl:
            self.vp.view.capture_colors=mode=='capture_distance';self.vp.view.overlay.update()
        self.pending_view = then
        if self.stage in ("raw", "side") and qrgb is None and len(self.snap.sides) > 1: self._request_side_in_place(path, mode); return
        self.loader.request(dict(id=self.view_id, slot="main", items=[dict(key="main", path=path, colour=colour, rgb=qrgb)],
                                 edit="main" if editable else None, mode=mode, stage=self.stage, path=path))

    def _request_side_in_place(self, path, mode):
        """The Side stage's one view: the picked side where the line-up put it, solid and editable, and every other side ghosted
        (dim, lined up) around it, so a side is seen in its place in the whole and the pager or a click on a dot picks another
        without the picture jumping. Before the first line-up, or when the picked side has no place yet, the side alone."""
        vid = self.view_id; side = self._side(); solo = self._solo_wanted(); self._view_solo = solo
        def build(lu):
            if self.view_id != vid or side is None: return
            T_ = (lu or {}).get("T") or {}; items = []; ghost = np.float32([0.30, 0.31, 0.33])
            mine = dict(key=side["id"], path=path); placed = T_.get(side["id"]) is not None
            if placed: mine["model"] = np.asarray(T_[side["id"]], float)
            items.append(mine)                                       # first: the editable one (its model turns the picks local)
            for x in self.snap.sides:                                # a side with no place yet is shown alone: ghosts would sit anywhere
                if solo or not placed or x["id"] == side["id"] or not x["active"] or T_.get(x["id"]) is None: continue   # Cut base: this side alone
                p = self.snap.out(self.snap.final("side.keep@" + x["id"]))
                if p: items.append(dict(key=x["id"], path=p, model=np.asarray(T_[x["id"]], float), colour=ghost))
            self._ghosts = [it["key"] for it in items[1:]]
            node = getattr(self.snap, "node", None)
            self._ghosts_old = [k for k in self._ghosts if node and (node(self.snap.final("side.keep@" + k)) or {}).get("status") not in DONE_STATES]
            self.loader.request(dict(id=vid, slot="main", items=items, edit=side["id"] if self.stage == "side" else None, mode=mode, stage=self.stage, path=path, check_mem=len(items) > 1))
        memo = getattr(self, "_lineup_memo", None)                # the line-up only changes with the snapshot: paging asks the host once
        if memo and memo[0] == getattr(self.snap, "t", None): build(memo[1]); return
        def keep(lu): self._lineup_memo = (getattr(self.snap, "t", None), lu); build(lu)
        self.call("h_lineup", then=keep, fail=lambda e: build(None))

    def _ghost_legend(self):
        """Side stage: a numbered dot in each ghosted side's color, on the part of that side farthest from the picked one (what it
        alone saw: lined up, the middles all coincide, so a dot at the middle would say nothing); a click on one picks that side."""
        from scipy.spatial import cKDTree
        v = self.vp.view; marks = []
        if self.stage not in ("raw", "side"):
            if getattr(self, "_ghost_marks", False): v.set_markers([]); self._ghost_marks = False      # left over on the model after YOLO
            return
        if not getattr(self, "_ghosts", None) or any(k not in v.items for k in self._ghosts): v.set_markers([]); self._ghost_marks = False; return   # a side alone: the last view's dots go
        def placed(it):
            A = it["V"] if it["kind"] == "mesh" else it["P"]; S = np.asarray(A[::max(1, len(A) // 20000)], float)
            M = it.get("model")
            return S if M is None or not len(S) else S @ np.asarray(M, float)[:3, :3].T + np.asarray(M, float)[:3, 3]
        mine = next((placed(it) for key, it in v.items.items() if key not in self._ghosts), None)
        tree = cKDTree(mine) if mine is not None and len(mine) else None
        dots = []                                    # placed so far: each new dot also keeps away from them (crowded scans piled them up)
        for k, x in enumerate(self.snap.sides):
            it = v.items.get(x["id"])
            if it is None or x["id"] not in self._ghosts: continue
            S = placed(it)
            if not len(S): continue
            if tree is None: c = S.mean(0)
            else:
                score = tree.query(S, workers=-1)[0]
                if dots: score = np.minimum(score, 0.7 * np.min(np.linalg.norm(S[:, None, :] - np.asarray(dots)[None], axis=2), axis=1))
                c = S[int(np.argmax(score))]
            dots.append(c)
            marks.append(([float(t) for t in c], W.SIDE_COLOURS[k % len(W.SIDE_COLOURS)], x["n"]))
        v.set_markers(marks); self._ghost_marks = bool(marks)

    def _all_mode(self):
        return False



    def _should_frame(self, req):
        """Re-frame when the stage changes; changing a placed side keeps the camera."""
        return self.shown.get("stage") != req.get("stage")



    def _goto_side(self, n):
        """A click on a ghosted side's number in the Side view: that side becomes the picked one (as the pager would)."""
        sides = (self.snap.sides if self.snap else None) or []
        if not (self.stage in ("raw", "side") and isinstance(n, int) and 1 <= n <= len(sides) and n != getattr(self, "side_k", None)): return
        self.side_k = n
        self._clear_detect(); self.editing = None; self.pager.set_page(self.side_k, len(sides), True, self._pager_note(self.side_k)); self._refresh_all()
        self.request_view(); self._spots_view()


    def _request_lineup(self):
        def got(lu):
            self.lineup = lu or {}; items = []
            for k, x in enumerate(self.snap.sides):
                if not x["active"]: continue
                p = self.snap.out(self.snap.final("side.keep@" + x["id"]))
                T_ = (lu.get("T") or {}).get(x["id"])
                if p and T_ is not None:
                    c = W.SIDE_COLOURS[k % len(W.SIDE_COLOURS)]
                    items.append(dict(key=x["id"], path=p, model=np.asarray(T_, float), colour=np.float32([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0))
            self.view_id += 1
            self.loader.request(dict(id=self.view_id, slot="main", items=items, edit=None, mode="shaded", stage="lineup",
                                     path=self._view_path(), check_mem=True))
        self.call("h_lineup", then=got)

    def on_view(self, out):
        if TRACE: print("trace view id=%s/%s %.2fs" % (out.get("id"), self.view_id, out.get("secs") or 0), flush=True)
        if out.get("slot") == "fill-preview": self._on_fill_preview(out); return
        if out.get("slot") == "before": self._on_before_view(out); return
        if out.get("id") != self.view_id: return
        if out.get("error"): self.notice(W.NOTICE["engine_error"].format(why=out["error"])); return
        req = out.get("req") or {}
        if out.get("too_big") and req.get("published"):
            self.published = None; self.shown = {}; self.request_view(); return      # too big to show here: the stage as it is
        if out.get("too_big"):
            node = {"side": "side@%s" % (self._side() or {}).get("id"), "fused": "fused", "meshed": "meshed"}.get(self.stage, self.stage)
            vid = out.get("id")                  # the light copy answers THIS request; a stage or side change since drops it

            def light(vp):
                if vp and self.loader is not None and self.view_id == vid:
                    self.loader.request(dict(req, id=vid, items=[dict(req["items"][0], path=vp)], readonly=True, check_mem=False, edit=None))
            self.call("h_view_file", node, then=light)
            return
        self.fin.clear(); self.slice_tool.clear()
        if hasattr(self, "drill"): self.drill.clear()
        v = self.vp.view
        previous_guide = getattr(v, "scan_guide", None)
        v.show_items(out["items"], edit=out.get("edit"), elements=out.get("elements"), frame=self._should_frame(req))
        guide=(getattr(self,"_scan_guides",{}) or {}).get((req.get("path"),self.colour_mode.get(self.stage)))
        from app2.qt.scan_guide import show as show_guide
        self._ghost_legend(); self._orient_view(); self._spots_view()     # ghosts' dots first (cleared off other stages), then the picked tool's marks
        show_guide(v,guide,frame=self._should_frame(req) or bool(guide and previous_guide is not guide))
        self.vp.guide_controls.setVisible(bool(guide)); self.vp._place()
        if self.stage == "report": self.panel["report"].set_guide(guide)
        self._maybe_prompt_orientation()
        it0 = out["items"][0] if out["items"] else {}
        self.shown = {"path": req.get("path"), "stage": req.get("stage"), "readonly": out.get("readonly"), "published": bool(req.get("published")),
                      "n": out.get("n") or sum(int(it.get("n") or 0) for it in out["items"]),
                      "kind": it0.get("kind"), "nv": len(it0["V"]) if it0.get("kind") == "mesh" and len(out["items"]) == 1 else None}
        self.sel_path = req.get("path")
        if self.stage == "lineup": self._lineup_colours(); self._pts = None; self._points_view()
        self._nudge_view()
        v.set_flat(self.colour_mode.get(self.stage) == "flat"); v.set_mesh_points(self.colour_mode.get(self.stage) == "points"); self._refresh_colours()
        if self._platter_preview(): self._platter_flags()
        then, self.pending_view = self.pending_view, None
        if then: then(out)
        self._refresh_info()
        if self.stage in ("fused", "meshed") and self.snap is not None: self._panel_extras(self.stage)
        self._refresh_chips(); self._refresh_tools()

        if self.stage == "meshed" and self.state is not None and out.get("items"):
            project = self.state.project; view_id = self.view_id
            QTimer.singleShot(250, lambda: self._save_project_preview(project, view_id))

    def _save_project_preview(self, project, view_id):
        if self._shut or not self.gl or self.stage != "meshed" or self.view_id != view_id:
            return
        if self.state is None or self.state.project != project or self.detected is not None:
            return
        from app2.qt.project_picker import save_scene_preview
        save_scene_preview(project, self.vp.view)

    def _finished(self):
        """The Meshed stage ends in Finish for print (turned on) and is built: its triangles (or 0 when not known yet),
        else None."""
        s = self.snap
        if s is None or not any(o.get("tool") == "finish" and o.get("on", True) for lk, j, o in self._chain("meshed")): return None
        n = s.node(s.final("meshed")) or {}
        if n.get("status") not in DONE_STATES: return None
        f = (n.get("stats") or {}).get("faces")
        if not f and self.shown.get("path") == self._view_path() and self.shown.get("kind") == "mesh": f = self.shown.get("n")
        return int(f or 0)

    def view_label(self):
        """What the view shows, for its label (top left): [(text, colour or None)], the first part the title. A stage
        with nothing built says so ("Meshed · not built yet")."""
        s = self.snap; sid = self.stage
        if s is None: return [(LABEL.get(sid, ""), None)]
        if sid == "raw" and self._side() and self.shown.get("path") and self.shown.get("path") == self._scanner_model(self._side()) \
                and (self._node_for_view() or {}).get("status") not in DONE_STATES:
            return [(LABEL["raw"] + " \u00b7 side %d \u00b7 the scanner's own model" % self._side()["n"], None)]
        if sid != "lineup" and not self.shown.get("published") and (self._node_for_view() or {}).get("status") not in DONE_STATES:
            side = self._side() if sid in ("raw", "side") else None
            name = ("Side %d" % side["n"] if sid == "side" else LABEL["raw"] + " \u00b7 side %d" % side["n"]) if side else LABEL.get(sid, "")
            return [(W.VIEW["chip"].format(stage=name, summary=W.VIEW["showing_old"] if self.shown.get("path") else W.VIEW["nothing_yet"]), None)]
        n_in = sum(1 for x in s.sides if x["active"]); sides = "%d side%s" % (n_in, "" if n_in == 1 else "s")
        side = self._side()
        if sid == "raw":
            return [(LABEL["raw"] + " \u00b7 side %d of %d" % (side["n"], len(s.sides)), None)] if side else [(LABEL["raw"], None)]
        if sid == "side":
            if not side: return [("Side", None)]
            head = "Side %d of %d" % (side["n"], len(s.sides))
            if self._platter_preview(): return [(head + " \u00b7 before the platter cut", None)]
            return [(head + (" \u00b7 platter cut" if self._platter_method() != "none" else ""), None)]
        if sid == "lineup": return [(W.VIEW["chip"].format(stage="Line-up", summary=W.VIEW["lineup_chip"].format(n=n_in)), None)]
        if sid == "fused":
            n = s.node("fused") or {}
            stats=n.get("stats") or {}
            warning=next((w for w in n.get("warn",[]) if w.startswith("Built the classic way")),None)
            if warning:return [(warning.split(" (",1)[0],"warn")]
            spacing=stats.get("spacing_mm",(n.get("params") or {}).get("point_distance",0.4))
            return [(W.VIEW["chip"].format(stage="Fused", summary=sides), None),
                    ("%s %g mm" % (self._fused_name(),spacing), None)]
        if sid == "meshed":
            if self.shown.get("published"): return [(W.VIEW["chip"].format(stage=W.VIEW["built"], summary=sides), None), (W.VIEW["published"], None)]
            f = self._finished()
            if f is not None: return [(W.VIEW["chip"].format(stage=W.VIEW["finished"], summary=W.VIEW["triangles"].format(n=tri_text(f)) if f else sides), None)]
            return [(W.VIEW["chip"].format(stage=W.VIEW["built"], summary=sides), None), (self._stage_sub("meshed"), None)]
        if sid == "report": return [(W.VIEW["chip"].format(stage="What is left", summary=self._stage_sub("report") or sides), None)]
        return [(LABEL.get(sid, ""), None)]

    def _refresh_chips(self):
        s = self.snap; sid = self.stage
        parts = self.view_label()
        baseline = self._sync_compare_change()
        if baseline and self.comparing and self.before_kind == "change":
            ready = (baseline["changed"] and not self.running() and not self.busy_call
                     and (self._node_for_view() or {}).get("status") in DONE_STATES
                     and self.shown.get("path") == self._view_path())
            parts = [(baseline["after_label"] if ready else "Rebuilding change" if baseline["changed"] else "Change pending", None)]
        if self.detected is not None and self.detected.get("count"): parts.append((W.DETECTED["chip"], "lost" if not self.detected.get("platter") else "platter"))
        if self._platter_preview() and getattr(self, "platter_cut", None):
            share = getattr(self, "platter_share", 0.0)
            parts.append(("%s (%d%% of this side)" % (W.VIEW["platter_cut"], int(round(100 * share))), "lost" if share > 0.4 else "platter"))
        rs = self._row_state(sid)[0]
        if rs == W.ROW_STATE["stale"]["text"] and s is not None and self.shown.get("path") and not self.shown.get("published"):
            parts.append((W.NOTICE["out_of_date_chip"], "warn"))
        elif sid in ("raw", "side") and getattr(self, "_ghosts_old", None) and not getattr(self, "_view_solo", False):
            parts.append((W.VIEW["ghosts_old"], "warn"))           # the gray sides around this one are their last build, not what a build would make now
        note = None; center = None
        if s is not None:
            if self.shown.get("n"): note = ("Points: %d" if self.shown.get("kind") == "points" else "Polygons: %d") % self.shown["n"]
            node = self._node_for_view() or {}
            built = node.get("status") in DONE_STATES or self.shown.get("published")
            side_now = self._side() if sid in ("raw", "side") else None
            if side_now is not None and not side_now["active"] and not self.shown.get("path"):
                center = W.VIEW["left_out"].format(n=side_now["n"])                   # a left-out side is never built: say so, not "not built yet"
            elif not built and sid != "lineup" and not self.shown.get("path"):         # nothing to say over a picture that is showing
                center = (W.VIEW["not_built_edit"] if getattr(self.vp, "edit_workspace", False) else
                          W.VIEW["not_built_raw"] if sid == "raw" else W.VIEW["not_built"])
            elif not self.gl and not node.get("preview") and not self.shown.get("published"):
                center = W.NOTICE["no_picture"]
                self._ask_preview(node)
        if self.peek: parts = [(W.VIEW["before"], None)] + parts[1:]
        self.vp.main.setToolTip(" · ".join(part[0] for part in parts))
        self.vp.main.set_chips(parts, note, center)

    def _ask_preview(self, node):
        """No GL: have the engine draw a picture of a built stage that has none yet (once per result)."""
        key = node.get("key"); asked = self.__dict__.setdefault("_previews_asked", set())
        if not key or key in asked or self.host is None or self.running(): return
        asked.add(key); tok = node.get("id")
        if not tok or tok.startswith("raw@") or tok == "textured": return
        self.call("preview", tok, then=lambda _p: self.host.status() if self.host else None, fail=lambda _e: None)

    # ---- the user's own Home view: an orientation saved with the project, used by every view (Build, Edit, Compare, Points panes)
    def _home_path(self): return os.path.join(self.host_root or "", "home_view.json")

    def _home_cams(self):
        v = self.vp
        return [p.view.cam for p in (v.main, v.before, getattr(v, "pa", None), getattr(v, "pb", None)) if p is not None and getattr(p, "view", None) is not None]

    def _apply_home_view(self):
        import json
        try:
            with open(self._home_path()) as f: R = np.asarray(json.load(f)["R"], float).reshape(3, 3)
        except (OSError, ValueError, KeyError, TypeError): R = None
        for cam in self._home_cams(): cam.home_rotation = R

    def set_home_view(self):
        """The view as it is now becomes the Home view for this project: Home, and every new framing, look from here."""
        import json
        if not self.gl or not self.host_root: return
        R = np.asarray(self.vp.view.cam.view[:3, :3], float).copy()
        try:
            with open(self._home_path(), "w") as f: json.dump({"R": R.tolist()}, f)
        except OSError: self.notice(W.NOTICE["home_not_saved"]); return
        self._apply_home_view(); self.notice(W.NOTICE["home_saved"], level="info")

    def reset_home_view(self):
        if not self.gl: return
        try: os.remove(self._home_path())
        except OSError: pass
        self._apply_home_view(); self.notice(W.NOTICE["home_reset"], level="info"); self.home()

    def home(self):
        if self.gl: self.vp.view.go_home()

    def _sync_cam(self, src, dst):
        if not self.gl or not self.comparing or dst is None: return
        dst.view.cam.view = src.view.cam.view.copy(); dst.view.cam.pivot = src.view.cam.pivot.copy()
        if src.view.cam.bound is not None: dst.view.cam.bound = src.view.cam.bound
        dst.view.update()

    # ================================================================ panel buttons
    def main_button(self):
        p = self.panel[self.stage]; b = p.body() if hasattr(p, "body") else None
        if b is not None and b.main is not None and b.main.isEnabled(): b.main.click()

    def on_moved(self, tab):
        if tab in ("Add fin", "Cut gap", "Cut shape"): (self.fin if tab == "Add fin" else self.slice_tool if tab == "Cut gap" else self.drill).preview(); return
        if self.detected is not None and self.detected.get("tab") == tab:
            self._clear_detect(); self.panel[self.stage].body(tab).set_msg(W.DETECTED["moved"])
        self._refresh_panel()
        if self.stage == "side" and tab == "Platter": self._platter_flags(); self._refresh_tools(); self._spots_view()

    def on_second(self, tab):
        if tab in ("Add fin", "Cut gap", "Cut shape"): (self.fin if tab == "Add fin" else self.slice_tool if tab == "Cut gap" else self.drill).clear()
        """Cancel: after a Detect, or while editing an applied step (the tool's sliders go back to a new step's start)."""
        was_edit = self.editing is not None; self._clear_detect(); self.editing = None
        b = self.panel[self.stage].body(tab)
        if b is not None:
            b.set_mode("idle"); b.set_msg("")
            for c in b.controls:
                if was_edit and c.c.get("tool"): c.engine_value = None
                elif c.engine_value is not None: c.set_engine(c.engine_value)
        self._refresh_applied(); self._refresh_panel()

    def on_extra(self, tab, k):
        if tab in ("Add fin", "Cut gap", "Cut shape"):
            self.fin.clear(); self.slice_tool.clear()
            if hasattr(self, "drill"): self.drill.clear()
            (self.fin if tab == "Add fin" else self.slice_tool if tab == "Cut gap" else self.drill).begin(); return
        if self.stage == "meshed" and tab == FINISH: self.export_stl(); return
        side = self._side()
        if self.stage == "raw" and side:
            if k == 0: self.resume_scan(side["n"])
            elif k == 1: self.start_run(StagesRun(to="side@" + side["id"], force=("side.fuse@" + side["id"],), publish=False))

    def on_button(self, tab):
        sid = self.stage; p = self.panel[sid]; b = p.body(tab)
        if b is None: return
        if sid == "lineup":                                   # Settings
            ch = {c.c["path"]: c.value() for c in b.controls if c.changed() and c.property("shown") is not False}
            self._set_then_run(ch, "lineup", StagesRun(to="lineup.adjust", publish=False)); return
        if sid == "report":
            ch = {c.c["path"]: c.value() for c in b.controls if c.changed()}
            self._set_then_run(ch, "report", StagesRun()); return
        vals = p.tab_values(tab); t = b.t; run_to = self._run_to(t)
        if tab in ("Add fin", "Cut gap", "Cut shape"):
            tool = self.fin if tab == "Add fin" else self.slice_tool if tab == "Cut gap" else self.drill
            try: vals["params"] = tool.params()
            except ValueError as e: b.set_msg(str(e)); return
            tool.clear()
        if b.mode == "detected": self._confirm_detect(tab, vals, run_to); return
        if b.mode == "editing" and self.editing:
            tok, _tab, k = self.editing                                   # k: the op's id
            self.editing = None; b.set_mode("idle"); b.set_msg("")
            for c in b.controls: c.engine_value = None            # a new step starts from the defaults again
            self.last_changed = sid
            self.call("set_op", tok, k, vals["params"], then=lambda _r: self.start_run(self._edit_run())); return
        if t.get("button") == W.DETECT: self._detect(tab, vals); return
        if tab == FINISH and vals["tool"] == "finish":
            self.last_changed = sid
            self.call("h_finish", vals["params"],
                      then=lambda _r: self.start_run(self._edit_run()))
            return
        if vals["tool"]:                                       # Apply on a tool tab: one new step at the list's end
            tok = self._list_token(); self.last_changed = sid
            self.call("add_op", tok, {"tool": vals["tool"], "params": vals["params"]},
                      then=lambda _r: self.start_run(self._edit_run()))
            for c in b.controls: c.engine_value = None        # tool controls start from the defaults again
            return
        ch = dict(vals["paths"])
        if sid == "side" and tab == "Platter":
            side = self._side(); m = ch.pop("side[{id}].platter.method", None)
            if m is None: m = next((ch.pop(k) for k in list(ch) if k.endswith("platter.method")), None)
            if m == "manual": self.notice(W.SPOTS["need"]); return              # a by-hand base is set with Set base from spots
            if m is not None and b.apply_to is not None and b.apply_to.value() == "this":
                self.last_changed = "side"
                self.call("platter", side["id"], m, then=lambda _r: self._set_then_run(self._side_paths(b, ch), "side", StagesRun(to=run_to, publish=False)))
                return
            if m is not None: ch["side.platter.method"] = m
        if sid == "side": ch = self._side_paths(b, ch)
        self._set_then_run(ch, sid, StagesRun(to=run_to, publish=False))

    def _side_paths(self, b, ch):
        """Side tab changes as per-side overrides (This side) or shared values (All sides). Margin and tilt: All sides (R8)."""
        side = self._side(); out = {}
        for k, v in ch.items():
            k = k.replace("side[{id}].", "side.")
            if b.apply_to is not None and b.apply_to.value() == "this" and side and k.startswith("side.") and \
                    k not in ("side.platter.margin", "side.platter.max_tilt", "side.platter.method"):
                k = "side[%s].%s" % (side["id"], k[5:])
            out[k] = v
        return out

    def _edit_run(self):
        """An applied edit rebuilds its descendants and publishes the finished model."""
        self.published = None
        return StagesRun()

    def _run_to(self, t):
        r = t.get("run_to")
        side = self._side()
        if r and "{id}" in r: r = r.replace("{id}", side["id"] if side else "")
        return r or {"side": "side@%s" % (side or {}).get("id"), "fused": "fused", "meshed": "meshed"}.get(self.stage, self.stage)

    def _set_then_run(self, ch, stage, run):
        self.last_changed = stage
        if not ch: self.start_run(run); return
        self.call("set", ch, then=lambda _r: self.start_run(run))

    # ---------------- detect
    def _detect(self, tab, vals):
        sid = self.stage; b = self.panel[sid].body(tab)
        if sid == "side":
            side = self._side(); rate = 100.0 * float(vals["paths"].get("side.keep.min_share", self._ctl(b, "keep.min_share").value()))
            self.call("h_side_detect", side["id"], rate, then=lambda r: self._show_detect(tab, r, "polygons"),
                      fail=lambda e: b.set_msg(str(e).splitlines()[0], "warn"))
        else:
            self.call("h_detect", self._list_token(), {"tool": vals["tool"], "params": vals["params"]},
                      then=lambda r: self._show_detect(tab, r, "points" if self._list_kind() == "points" else "polygons"),
                      fail=lambda e: b.set_msg(str(e).splitlines()[0], "warn"))
        b.set_msg("")

    def _show_detect(self, tab, r, what):
        """A Detect's result in the tool's two result lines, the red on the view. Side Isolation looks at the finished
        side (side.ply, the view): pieces the side's rate already removed are said to be gone ("Already removed at 3%"),
        never offered as a Delete that changes nothing."""
        b = self.panel[self.stage].body(tab)
        n = int(r.get("n") or 0); cnt = int(r.get("count") or 0); back = int(r.get("back") or 0); pc = r.get("pieces")
        mask = r.get("mask"); inp = r.get("input"); now = r.get("now_pct", 0)
        gone_pc = 0
        if r.get("view_mask") is not None or r.get("view_count") is not None:        # Side: counted on the finished side
            n = int(r.get("view_n") or n); gone_pc = max(0, int(pc or 0) - int(r.get("view_pieces") or 0))
            cnt = int(r.get("view_count") or 0); pc = r.get("view_pieces"); mask = r.get("view_mask"); inp = None
            if back and not cnt: mask = r.get("mask"); inp = r.get("input")                 # what comes back is only on the cut
        pct = 100.0 * cnt / max(1, n); mode = "detected"
        if cnt == 0 and back: text = W.DETECTED["back"].format(count=back, what=what, now=now)
        elif cnt == 0 and gone_pc:
            text = W.DETECTED["already"].format(now=now, pieces=gone_pc, s="" if gone_pc == 1 else "s", rate=float(r.get("rate") or 0)); mode = "none"
        elif cnt == 0: text = W.DETECTED["none"]; mode = "none"
        elif tab == "Overlap Detection": text = W.DETECTED["overlap_count"].format(count=cnt, n=n, pct=pct)
        elif pc and gone_pc: text = W.DETECTED["also"].format(pieces=pc, s="" if pc == 1 else "s", count=cnt, what=what, pct=pct, gone=gone_pc, now=now)
        elif pc: text = W.DETECTED["pieces"].format(pieces=pc, s="" if pc == 1 else "s", count=cnt, n=n, what=what, pct=pct)
        else: text = W.DETECTED["count"].format(count=cnt, n=n, what=what, pct=pct)
        if cnt and back: text = text.split("\n")[0] + "\n" + W.DETECTED["back"].format(count=back, what=what, now=now).split("\n")[0]
        b.set_msg(text); b.set_mode(mode)
        if back and not cnt and b.main is not None: b.main.setText(W.APPLY); b.main.setProperty("kind", "apply")    # lowering the rate brings pieces back
        self.detected = dict(tab=tab, count=cnt or back, n=n, mask=mask if (cnt or back) else None, input=inp, step=r.get("step"))
        if mode == "none": self._refresh_chips(); self._refresh_tools(); return
        if not self.gl:
            if r.get("img") is not None: self.vp.main.set_image(r["img"])
        elif mask is not None:
            if inp and inp != self.shown.get("path"):          # the mask indexes the list end before the deletes
                def after(out, m=mask):
                    if self.detected is None: return
                    if out.get("n") == len(m): self.vp.view.show_would_go(m)
                    else: self.notice(W.NOTICE["out_of_date"]); self._clear_detect()
                self.request_view(path=inp, then=after); self.shown["path"] = inp
            elif self.shown.get("n") == len(mask): self.vp.view.show_would_go(mask)
            else: self.notice(W.NOTICE["out_of_date"]); self._clear_detect(); self.request_view()
        self._refresh_chips(); self._refresh_tools()

    def _confirm_detect(self, tab, vals, run_to):
        """Delete after a Detect: the tool becomes a step of the list (Side Isolation: the side's rate)."""
        b = self.panel[self.stage].body(tab); self._clear_detect(); b.set_mode("idle"); b.set_msg("")
        self.last_changed = self.stage
        if self.stage == "side":
            ch = self._side_paths(b, {"side.keep.min_share": self._ctl(b, "keep.min_share").value()})
            self._set_then_run(ch, "side", StagesRun(to=run_to, publish=False)); return
        self.call("add_op", self._list_token(), {"tool": vals["tool"], "params": vals["params"]},
                  then=lambda _r: self.start_run(self._edit_run()))

    def _clear_detect(self):
        if self.detected is None: return
        d = self.detected; self.detected = None
        b = self.panel[self.stage].body(d.get("tab")) if self.stage in ("side", "fused", "meshed") else None
        if b is not None: b.set_mode("idle")
        if self.gl:
            self.vp.view.clear_would_go()
            if d.get("input") and self.shown.get("path") != self._view_path(): self.request_view()
        else: self.request_view()
        self._refresh_chips(); self._refresh_tools()

    def escape(self):
        if hasattr(self, "edit_confirm") and self.edit_confirm.isVisible(): self._close_edit_confirm(); return
        if self.hist_pop.isVisible(): self.hist_pop.close_popover(); return
        if self.log_pop.isVisible(): self.log_pop.close_popover(); return
        if self.vers_pop.isVisible(): self.vers_pop.close_popover(); return
        if self.detected is not None:
            tab = self.detected.get("tab"); self._clear_detect()
            b = self.panel[self.stage].body(tab) if hasattr(self.panel[self.stage], "body") else None
            if b is not None: b.set_msg("")
            return
        if self.gl and self.vp.view.selection is not None and self.vp.view.selection.any(): self.clear_selection(); return
        if self.editing: self.on_second(self.editing[1])

    # ---------------- applied list
    def edit_op(self, k):
        ch = self._chain()
        if not (0 <= k < len(ch)): return
        lk, j, o = ch[k]; tab = TOOL_TAB.get(o["tool"])
        p = self.panel[self.stage]
        if tab is None or tab not in p.bodies: return
        p.show_tab(tab); b = p.body(tab)
        if tab in getattr(p, "blocks", {}): self.scroll.ensureWidgetVisible(b)
        for c in b.controls:
            if c.c.get("tool") == o["tool"] and c.property("shown") is not False:
                v = (o.get("params") or {}).get(c.c["param"], self._tool_default(c.c, self._list_kind()))
                c.set_engine(v)
        if tab in ("Add fin", "Cut gap", "Cut shape"):
            import json
            tool = self.fin if tab == "Add fin" else self.slice_tool if tab == "Cut gap" else self.drill
            tool.anchors = json.loads(o.get("params", {}).get("anchors", "[]")); tool.preview()
        self.editing = (self._list_token(), tab, o.get("id"))              # by id: an undo or a CLI edit meanwhile cannot move it
        b.set_mode("editing"); b.set_msg(W.APPLIED["editing"].format(k=k + 1), "accent")
        if b.t.get("button") == W.DETECT and k != len(ch) - 1: b.set_note(W.NOTICE["detect_last"])   # R1
        self._refresh_applied()

    def _plan_pending_edits(self, changes):
        self._pending_plan_ticket = getattr(self, "_pending_plan_ticket", 0) + 1
        ticket = self._pending_plan_ticket
        if not changes:
            self.applied.set_pending_plan(""); return
        self.applied.set_pending_plan("Checking saved results...")
        token, stage = self._list_token(), self.stage
        to = {"lineup.fix": "lineup.adjust", "lineup.adjust": "lineup.adjust"}.get(token, stage)
        def show(plan):
            if ticket != self._pending_plan_ticket: return
            count = plan["to_run"]
            text = "All steps already built" if count == 0 else "%d step%s need building" % (count, "" if count == 1 else "s")
            self.applied.set_pending_plan(text, cached=count == 0)
        def ask():
            if ticket != self._pending_plan_ticket: return
            self.call("h_toggle_plan", token, dict(changes), to=to, then=show,
                      fail=lambda e: self.applied.set_pending_plan("Could not check saved results") if ticket == self._pending_plan_ticket else None)
        QTimer.singleShot(180, ask)

    def _confirm_pending_edits(self, changes):
        if not changes or self.snap is None or self.running() or self.busy_call: return
        # Reuse the same in-window confirmation and stale-recipe guard as a row action.
        first = next((i for i, (_tok, oid) in enumerate(self._applied_ids) if oid in changes), None)
        if first is None: return
        self.row_call("toggle_op", first)
        if getattr(self, "_pending_edit", None) is None: return
        _name, token, _oid, recipe, stage = self._pending_edit
        self._pending_edit = ("h_toggle_many", token, dict(changes), recipe, stage)
        self.edit_confirm.title.setText("Apply these edit changes?")
        self.edit_confirm.text.setText("%d edit%s will change. The model rebuilds once, using saved results where possible. The checkboxes alone do not change your model. Undo restores the whole group." % (len(changes), "" if len(changes) == 1 else "s"))
        self.edit_confirm.go.setText("Apply changes")

    def row_call(self, name, k):
        """Explain the downstream rebuild before changing an applied edit."""
        if not (0 <= k < len(self._applied_ids)) or self.running() or self.busy_call: return
        tok, oid = self._applied_ids[k]
        if tok is None or oid is None or self.snap is None: return
        import copy
        from app2.qt.rebuild_confirm import RebuildConfirm
        chain = self.snap.chains.get(tok, [])
        found = next(((i, o) for i, (_lk, _j, o) in enumerate(chain) if o.get("id") == oid), None)
        if found is None: return
        index, op = found
        action = "Remove" if name == "drop_op" else "Turn off" if op.get("on", True) else "Turn on"
        label = TOOL_LABEL.get(op.get("tool")) or TOOL_TAB.get(op.get("tool"), "Edit %d" % (k + 1))
        recipe, stage = copy.deepcopy(self.snap.recipe), self.stage
        if not hasattr(self, "edit_confirm"):
            self.edit_confirm_scrim = Scrim(self); self.edit_confirm_scrim.hide()
            self.edit_confirm = RebuildConfirm(self)
            self.edit_confirm.cancelled.connect(self._close_edit_confirm)
            self.edit_confirm_scrim.clicked.connect(self._close_edit_confirm)
            self.edit_confirm.accepted.connect(self._accept_edit_confirm)
        self._pending_edit = (name, tok, oid, recipe, stage)
        self.edit_confirm.describe(action, label, sum(o.get("on", True) for _lk, _j, o in chain[index+1:]))
        self.edit_confirm_scrim.setGeometry(self.rect()); self.edit_confirm_scrim.show(); self.edit_confirm_scrim.raise_()
        self.edit_confirm.move((self.width()-self.edit_confirm.width())//2, max(8,(self.height()-self.edit_confirm.height())//2))
        self.edit_confirm.show(); self.edit_confirm.raise_()

    def _close_edit_confirm(self):
        self._pending_edit = None
        self.edit_confirm.hide(); self.edit_confirm_scrim.hide()

    def _accept_edit_confirm(self):
        pending = self._pending_edit
        self._close_edit_confirm()
        if pending is None: return
        name, tok, oid, recipe, stage = pending
        if self.running() or self.busy_call or self.snap is None or self.stage != stage or self.snap.recipe != recipe:
            self.notice("Open the edit menu again: the edits changed."); return
        import copy
        expected = copy.deepcopy(recipe)
        chain = self.snap.chains.get(tok, [])
        self._clear_compare_change()
        if name == "h_toggle_many" and isinstance(oid, dict):
            valid = True
            for lk, j, op in chain:
                if op.get("id") not in oid: continue
                ops = expected.setdefault("ops", {}).get(lk)
                if ops is None or j >= len(ops) or ops[j].get("id") != op.get("id"):
                    valid = False; break
                ops[j]["on"] = bool(oid[op["id"]])
            if valid: self._remember_compare_change(expected, "Before edit changes", "After edit changes")
            self.op_call(name, oid, token=tok); return
        hit = next(((i, lk, j, o) for i, (lk, j, o) in enumerate(chain) if o.get("id") == oid), None)
        if hit is not None:
            i, lk, j, op = hit
            ops = expected.setdefault("ops", {}).get(lk)
            if ops is not None and j < len(ops) and ops[j].get("id") == oid:
                if name == "drop_op":
                    ops.pop(j); before_label = "Before removing edit %d" % (i+1); after_label = "Now: edit %d removed" % (i+1)
                else:
                    ops[j]["on"] = not op.get("on", True)
                    action = "on" if ops[j]["on"] else "off"
                    before_label = "Before turning %s edit %d" % (action, i+1); after_label = "Now: edit %d %s" % (i+1, action)
                self._remember_compare_change(expected, before_label, after_label)
            else: self._clear_compare_change()
        self.op_call(name, oid, token=tok)

    def op_call(self, name, k, token=None):
        tok = token or self._list_token()
        if tok is None: return
        run_to = {"lineup.fix": "lineup.adjust", "lineup.adjust": "lineup.adjust"}.get(tok) or self._run_to(self.panel[self.stage].body().t if hasattr(self.panel[self.stage], "body") and self.panel[self.stage].body() else {})
        self.last_changed = self.stage
        self.call(name, tok, k, then=lambda _r: self.start_run(
            StagesRun(to=run_to, publish=False) if tok.startswith('lineup.') else self._edit_run()))

    # ---------------- Rectangle / Lasso
    def set_tool(self, t):
        if not self.gl: return
        if t != "orbit" and not getattr(self.vp, "edit_workspace", False): t = "orbit"
        self.fin.clear(); self.slice_tool.clear()
        if hasattr(self, "drill"): self.drill.clear()
        if t != "orbit" and not self.vp.view.tools_on: t = "orbit"
        if t != "orbit" and self.stage == "side" and not self.side_note_shown:
            self.side_note_shown = True; self.notice(W.PANELS["side"]["delete_note"], level="info")
        self.vp.view.set_tool(t); self.vp.set_tool(t)

    def set_through(self, on):
        if self.gl: self.vp.view.through = bool(on)

    def on_selection(self, n):
        what = "points" if self.shown.get("kind") == "points" else "polygons"
        self.vp.set_selection(n if self.detected is None else 0, what)

    def clear_selection(self):
        if self.gl: self.vp.view.clear_selection()

    def invert(self):
        if self.gl: self.vp.view.invert_selection()

    def delete_selection(self):
        if not self.gl or not getattr(self.vp, "edit_workspace", False) or self.running(): return
        v = self.vp.view
        if v.selection is None or not v.selection.any(): return
        if self.sel_path != self._view_path() or self.shown.get("path") != self._view_path():
            self.notice(W.NOTICE["changed_under"]); v.clear_selection(); return
        pos = v.mask_points(); want = int(v.selection.sum()); sid = self.stage
        node = {"side": "side@%s" % (self._side() or {}).get("id"), "fused": "fused", "meshed": "meshed"}.get(sid)
        if node is None: return
        run_to = self._run_to({"run_to": {"side": "side.keep@{id}", "fused": "fused", "meshed": "meshed"}[sid]})
        self.last_changed = sid

        def done(r):
            # 2026-10-25: the faces leave the picture now; the one rebuild waits until Rebuild, a shape tool, another
            # stage or leaving Edit ("don't force a rebuild till you're done editing all of them")
            v.hide_selected(); self.pending_edits = getattr(self, "pending_edits", 0) + 1
            if int(r.get("got") or 0) != want: self.notice(W.NOTICE["took_other"].format(got=r.get("got"), want=want))
            else: self.notice(W.NOTICE["delete_pending"].format(n=self.pending_edits, s="" if self.pending_edits == 1 else "s"), level="info")
            self.pending_changed()
        self.call("h_mask_delete", node, pos, want, then=done)

    def pending_changed(self):
        cb = getattr(self, "on_pending_edits", None)
        if cb: cb(getattr(self, "pending_edits", 0))
        try: self._refresh_rebuild()
        except Exception: pass

    def flush_pending_edits(self):
        """The rebuild the waiting deletes need. True when one was started."""
        if not getattr(self, "pending_edits", 0): return False
        if self.gl: self.set_tool("orbit")
        started = bool(self.start_run(self._edit_run()))
        if started: self.pending_edits = 0; self.pending_changed()  # refused (busy, recording): the deletes keep waiting
        return started

    def _needs_rebuild_first(self, why):
        """An engine refusal because the result is not built: say what to do, and start the waiting rebuild when there is
        one."""
        if "not built yet" not in str(why): return False
        if getattr(self, "pending_edits", 0) and self.flush_pending_edits(): self.notice(W.NOTICE["rebuild_now"], level="info", secs=8)
        else: self.notice(W.NOTICE["rebuild_first"], secs=10)
        return True

    # ---------------- Line-up
    def apply_sides(self, changes, hints=None):
        """Use / leave out for the sides changed, and the "how scanned" hints when they changed; then the line-up runs again."""
        if not changes and hints is None: return
        now = {x["id"]: x["active"] for x in (self.snap.sides if self.snap else [])}
        now.update(dict(changes))
        if now and not any(now.values()): self.notice(W.NOTICE["one_side"]); return      # refused before anything is left out
        self.last_changed = "lineup"
        run = lambda: self.start_run(StagesRun(to="lineup.adjust", publish=False))
        after = (lambda _r=None: self._set_then_run({"lineup.place.hints": hints}, "lineup", StagesRun(to="lineup.adjust", publish=False))) if hints is not None else (lambda _r=None: run())
        if not changes: after(); return
        calls = [("include" if use else "leave_out", (sid,), None) for sid, use in changes]
        self.call("h_apply", calls, then=after,
                  fail=lambda e: self.notice(W.NOTICE["one_side"] if "at least one side" in str(e) else str(e).splitlines()[0]))

    def _lineup_remove(self, tok, i):
        ch = (self.snap.chains or {}).get(tok, []) if self.snap else []
        if 0 <= i < len(ch) and ch[i][2].get("id"): self.op_call("drop_op", ch[i][2]["id"], token=tok)

    def show_side_toggle(self, sid, on):
        if self.gl and sid in self.vp.view.items: self.vp.view.items[sid]["visible"] = on; self._lineup_visible()

    def _lineup_visible(self):
        v = self.vp.view
        if v.scene is None: return
        for k, it in v.items.items():
            if k in v.scene.items: v.scene.items[k].visible = it.get("visible", True)
        v.update()

    def _lineup_colours(self):
        if not self.gl or self.stage != "lineup": return
        lp = self.panel["lineup"]; pick = lp.flip_side.value() if lp.current == "Flip" else lp.nudge_side.value() if lp.current == "Nudge" else None
        for k, x in enumerate(self.snap.sides if self.snap else []):
            c = W.SIDE_COLOURS[k % len(W.SIDE_COLOURS)]; rgb = np.float32([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
            if pick and x["id"] != pick: rgb = rgb * 0.35 + 0.1
            self.vp.view.set_item_colour(x["id"], rgb)

    def _base_T(self, sid):
        T_ = (self.lineup.get("T") or {}).get(sid)
        return np.asarray(T_, float) if T_ is not None else None

    def preview_flip(self, sid, axis):
        """Preview only the item's model matrix. The pose comes from the engine's own flip fit (flip preview=True), the
        same code Apply stores, so the preview is where the side lands (auto included)."""
        if not self.gl or sid not in self.vp.view.items: return
        T0 = self._base_T(sid); self._flip_ask += 1; ask = self._flip_ask
        if T0 is None: return
        self.vp.view.set_item_model(sid, T0)
        if not axis or self.host is None or self.busy_call: return
        self.notice(TABS_WORDS["Flip"]["preview_wait"], level="info")
        lp = self.panel["lineup"]

        def got(r):
            if ask == self._flip_ask and self.stage == "lineup" and lp.current == "Flip" and sid in self.vp.view.items and r:
                self.vp.view.set_item_model(sid, np.asarray(r["T"], float))
        self.call("flip", sid, axis, preview=True, then=got, fail=lambda e: None)

    def do_flip(self, sid, axis):
        if not sid: return
        self.last_changed = "lineup"
        self.call("flip", sid, axis, then=lambda _r: self.start_run(StagesRun(to="lineup.adjust", publish=False)))

    # ---------------- Line-up Points
    def _points_on(self):
        return bool(self.gl and self.stage == "lineup" and self.panel["lineup"].current == "Points" and self.vp.view.items)

    def _item_ball(self, key, M):
        it = self.vp.view.items[key]; V = it["V"] if it["kind"] == "mesh" else it["P"]
        P = np.asarray(V[::max(1, len(V) // 20000)], float) @ M[:3, :3].T + M[:3, 3]
        c = P.mean(0); return c, float(np.linalg.norm(P - c, axis=1).max())

    def _pts_candidate(self):
        """Moving side's own frame -> base side's own frame from the clicked pairs, or None until there are 3 good ones."""
        return self.panel["lineup"].picks.matrix()

    def _points_view(self):
        """Points tab: the base (top left) and the moving side (top right) each alone, where you click matching spots; below, the two
        lined up as they are now, or with your dots once there are 3 pairs. Other tabs and stages: the line-up as built."""
        if not self.gl: return
        v = self.vp.view; lp = self.panel["lineup"]
        if not self._points_on():
            if self._pts is not None or v.only is not None or self.vp.top_row.isVisible():
                self._pts = None; v.set_only(None); v.set_markers([]); self.vp.set_points_mode(False)
                for pane in (self.vp.pa, self.vp.pb): pane.view.clear(); pane.view.set_markers([])
                if v.tool == "pairpick": v.set_tool("orbit")
                for k in list(v.items):
                    T0 = self._base_T(k)
                    if T0 is not None and self.stage == "lineup": v.set_item_model(k, T0)
            return
        sid, to = lp.points_sides()
        if sid not in v.items or to not in v.items or sid == to or self._base_T(sid) is None or self._base_T(to) is None:
            v.set_only(None); v.set_markers([]); self._pts = None; self.vp.set_points_mode(False); return
        Tt, Ts = self._base_T(to), self._base_T(sid)
        first = self._pts is None or self._pts["pair"] != (sid, to)
        if first:
            self._pts = {"pair": (sid, to)}; self.vp.set_points_mode(True)
            for pane, key, label in ((self.vp.pa, to, "Base"), (self.vp.pb, sid, "Moving")):
                it = dict(v.items[key]); it["model"] = None; it["visible"] = True
                pane.view.show_items([it], edit=None, elements=None, frame=True); pane.view.set_tool("pairpick")
                pane.set_chips([("%s: %s" % (label, lp.side_name(key)), None)])
            v.set_only([sid, to])
        cand = self._pts_candidate()
        Ms = (Tt @ cand) if cand is not None else Ts          # where the moving side sits in the result
        v.set_item_model(to, Tt); v.set_item_model(sid, Ms)
        if first:
            (c1, r1), (c2, r2) = self._item_ball(to, Tt), self._item_ball(sid, Ms); r = max(r1, r2)
            v.cam.frame(np.minimum(c1, c2) - r, np.maximum(c1, c2) + r); v.camera_moved.emit()
        if v.tool != "orbit": v.set_tool("orbit")
        self.vp.main.set_chips([("Lined up with your dots" if cand is not None else "Lined up as it is now", None)])
        dots = lp.picks.dots(); col = lambda n: W.PAIR_COLOURS[(n - 1) % len(W.PAIR_COLOURS)]
        self.vp.pa.view.set_markers([(x, col(n), n) for w, x, n in dots if w == 0])
        self.vp.pb.view.set_markers([(x, col(n), n) for w, x, n in dots if w == 1])
        mods = {0: Tt, 1: Ms}
        v.set_markers([(mods[w][:3, :3] @ np.asarray(x) + mods[w][:3, 3], col(n), n) for w, x, n in dots])

    def _near_model(self, world, tol=8.0):
        """Is the picked point on the model? The view's depth can hand back a point in empty space (a hole in a point cloud, the far
        plane), so a pick must lie within tol mm of a shown element, else it is dropped."""
        v = self.vp.view; E = v.elements
        if E is None or not len(E[0]): return True
        key = (id(E[0]), len(E[0]))
        if getattr(self, "_pick_tree", (None,))[0] != key:
            from scipy.spatial import cKDTree
            A = np.asarray(E[0], np.float64); self._pick_tree = (key, cKDTree(A[::max(1, len(A) // 400000)]))
        loc = self._cut_local(world) if self.stage in ("fused", "meshed", "side") else [float(x) for x in world]
        dist = float(self._pick_tree[1].query(np.asarray(loc, float))[0])
        if dist > tol:
            from app2.diagnostics import log as _dl; import logging; logging.getLogger(_dl.NAME).info("pick dropped: %.1f mm from the model at %s", dist, [round(x) for x in loc])
        return dist <= tol

    def _pick_missed(self):
        """A pick click on empty space: the first is forgiven, the second and later say to click on the model."""
        if not (self._cut_on() or self._clone_on() or self._fix_on() or self._mirror_on() or self._spots_on() or self._orient_on()): return
        self._missed = getattr(self, "_missed", 0) + 1
        if self._missed >= 2: self.notice(W.SPOTS["missed"], level="info")

    def _surface_pick(self, world):
        """A click on the model in the main view: a spot for By hand (Cut base) or for Orient, whichever is on."""
        if self._cut_on() or self._clone_on() or self._fix_on() or self._mirror_on() or self._spots_on() or self._orient_on():
            if not self._near_model(world):
                self._missed = getattr(self, "_missed", 0) + 1
                if self._missed >= 2: self.notice(W.SPOTS["missed"], level="info")        # the first blank click is forgiven
                return
            self._missed = 0
        if self._fix_on(): self._fix_pick(world)
        elif self._mirror_on(): self._mirror_add(world)
        elif self._clone_on(): self._clone_pick(world)
        elif self._cut_on(): self._cut_add(world)
        elif self._spots_on(): self._spot_add(world)
        elif self._orient_on(): self._orient_add(world)

    def _points_pick_pane(self, which, world):
        """A click in the base pane (0) or the moving pane (1): that side's own frame (each pane shows its side as scanned)."""
        if not self._points_on() or self._pts is None: return
        self.panel["lineup"].add_pick(which, [float(x) for x in world])        # its points_changed redraws every pane

    # ---------------- Line-up: sides marked as checked, and the automatic fit's matches as dots
    def _auto_disagreements(self, base, pos):
        """{side id: words} for sides whose placed pose is far from where their own automatic fit to the base puts them (over 20 degrees or 30 mm).
        A near-symmetric part can be placed flipped and still overlap well, so the score alone cannot show it."""
        import json
        out = {}
        T = (getattr(self, "lineup", None) or {}).get("T") or {}
        path = self.snap.out("lineup.pairs") if self.snap else None
        if not base or base not in T or not path: return out
        try:
            with open(path) as f: entries = json.load(f)["pairs"]
        except (OSError, ValueError, KeyError, TypeError): return out
        Tb = np.asarray(T[base], float)
        for sid, Ts in T.items():
            if sid == base: continue
            e = next((x for x in entries if x["base"] == base and x["moving"] == sid), None); inv = False
            if e is None: e = next((x for x in entries if x["base"] == sid and x["moving"] == base), None); inv = e is not None
            if e is None: continue
            M = np.asarray(e["matrix"], float).reshape(4, 4); M = np.linalg.inv(M) if inv else M
            D = np.linalg.inv(Tb @ M) @ np.asarray(Ts, float)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))); mm = float(np.linalg.norm(D[:3, 3]))
            if ang > 20 or mm > 30: out[sid] = "differs from its own auto fit to the base (%d deg, %d mm)" % (round(ang), round(mm))
        return out

    def _checked_path(self): return os.path.join(self.host_root, "lineup_checked.json") if self.host_root else None

    def _load_checked(self):
        import json
        try:
            with open(self._checked_path()) as f: d = json.load(f)
        except (OSError, ValueError, TypeError): return {}
        return {k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(d, dict) else {}      # the old list format is dropped

    def _save_checked(self, checked):
        import json
        path = self._checked_path()
        if not path: return
        try:
            with open(path, "w") as f: json.dump(dict(checked), f)
        except OSError: pass
        self._refresh_lineup_panel()                                # the Sides rows say which side it was checked against

    def suggest_dots(self, sid, to, n):
        """The automatic fit's matches for a pair, as n well-spread editable pairs: sample the moving side, carry it by the pair's fit onto
        the base side, keep the points that land within the landing tolerance, spread them out, pair each with its nearest base point."""
        import json
        from scipy.spatial import cKDTree
        v = self.vp.view
        if not self.gl or sid not in v.items or to not in v.items or sid == to: self.notice("Pick two different sides first."); return
        path = self.snap.out("lineup.pairs") if self.snap else None
        try:
            with open(path) as f: entries = json.load(f)["pairs"]
        except (OSError, TypeError, ValueError, KeyError): self.notice("The line-up has not been built yet, so there are no automatic matches to show."); return
        e = next((x for x in entries if x["base"] == to and x["moving"] == sid), None); inv = False
        if e is None: e = next((x for x in entries if x["base"] == sid and x["moving"] == to), None); inv = e is not None
        if e is None: self.notice("The automatic fit has no result for this pair."); return
        M = np.asarray(e["matrix"], float).reshape(4, 4); M = np.linalg.inv(M) if inv else M        # moving frame -> base frame
        def cloud(key, k):
            it = v.items[key]; V = np.asarray(it["V"] if it["kind"] == "mesh" else it["P"], float)
            return V[::max(1, len(V) // k)]
        Sm = cloud(sid, 60000); Sb = cloud(to, 200000); Q = Sm @ M[:3, :3].T + M[:3, 3]
        tree = cKDTree(Sb); d, idx = tree.query(Q); tol = float(self._recipe_value("lineup.place.tol") or 1.5)
        ok = np.nonzero(d < tol)[0]
        if len(ok) < 3: self.notice("The automatic fit for this pair has almost nothing in common to show. Pick points yourself, or try another side."); return
        chosen = [int(ok[0])]; far = np.linalg.norm(Q[ok] - Q[chosen[0]], axis=1)
        while len(chosen) < min(n, len(ok)):                         # farthest point sampling: spread out over the shared surface
            k = int(np.argmax(far)); chosen.append(int(ok[k])); far = np.minimum(far, np.linalg.norm(Q[ok] - Q[ok[k]], axis=1))
        self.panel["lineup"].set_pairs([[Sb[idx[i]], Sm[i]] for i in chosen])

    def do_points(self, sid, to, pairs):
        if not sid or not to or len(pairs) < 3: return
        self.last_changed = "lineup"
        self.call("match", sid, to, pairs, then=lambda _r: self.start_run(StagesRun(to="lineup.adjust", publish=False)))

    # ---------------- Line-up Nudge (Move steppers, rotate rings)
    def _nudge_on(self):
        return bool(self.gl and self.stage == "lineup" and self.panel["lineup"].current == "Nudge" and self.vp.view.items)

    def _side_ball(self, sid):
        """(centre in the side's own frame, radius mm): the vertex mean, the engine's turn centre (stages turn_side_matrix)."""
        it = self.vp.view.items[sid]; V = it["V"] if it["kind"] == "mesh" else it["P"]
        got = self.__dict__.setdefault("_balls", {}).get(sid)
        if got is None or got[0] is not V:
            c0 = np.asarray(V, float).mean(0) if len(V) else np.zeros(3)
            S = np.asarray(V[::max(1, len(V) // 20000)], float); got = (V, c0, float(np.linalg.norm(S - c0, axis=1).max()) if len(S) else 10.0)
            self._balls[sid] = got
        return got[1], got[2]

    def _nudge_view(self):
        """Nudge tab: the rings on the picked side (its preview through the panel); elsewhere no rings, the line-up as built."""
        if not self.gl: return
        v = self.vp.view
        if not self._nudge_on():
            if v.gizmo is not None:
                v.set_gizmo(None)
                for k in list(v.items):
                    T0 = self._base_T(k)
                    if T0 is not None and self.stage == "lineup": v.set_item_model(k, T0)
            return
        self.panel["lineup"]._nudge_prev()

    def preview_nudge(self, sid, t, R):
        """Only the item's model matrix: turn R (world axes) about the side's centre, then move by t. No rebuild."""
        if not self.gl: return
        v = self.vp.view; T0 = self._base_T(sid) if sid in v.items else None
        if T0 is None or not self._nudge_on(): v.set_gizmo(None); return
        c0, rad = self._side_ball(sid); c = T0[:3, :3] @ c0 + T0[:3, 3]; R = np.asarray(R, float); t = np.asarray(t, float)
        D = np.eye(4); D[:3, :3] = R; D[:3, 3] = c + t - R @ c
        v.set_item_model(sid, D @ T0); v.set_gizmo(c + t, rad * 1.1)

    def do_nudge(self, sid, t, R, keep=False):
        """Apply: the turn as a Line-up fix op (side frame, local fit unless keep) and the move as a nudge, as one history
        entry (one undo)."""
        R = np.asarray(R, float); turn = np.abs(R - np.eye(3)).max() > 1e-6
        if not sid or (not any(t) and not turn): return
        calls = [("turn", (sid, R.tolist(), bool(keep), list(t) if any(t) else None), None)] if turn else [("nudge", (sid, list(t), [0, 0, 0]), None)]
        self.last_changed = "lineup"; lp = self.panel["lineup"]

        def done(_r):
            lp.reset_nudge(emit=False); self.start_run(StagesRun(to="lineup.adjust", publish=False))
        self.call("h_apply", calls, then=done, fail=lambda e: self.notice(str(e).splitlines()[0]))

    # ---------------- Scan here
    def turn_to_place(self, i):
        st = ((self.snap.node("report") or {}).get("stats") or {}) if self.snap else {}
        pl = (st.get("places") or [])[i:i + 1]
        if not pl or not self.gl: return
        v = self.vp.view; c = np.asarray(pl[0]["at_mm"], float)
        pe = v.cam.view[:3, :3] @ v.cam.pivot + v.cam.view[:3, 3]
        v.cam.pivot = c; v.cam.view[:3, 3] = pe - v.cam.view[:3, :3] @ c; v.camera_moved.emit(); v.update()
        self.turned_to = c

    def scan_new_side(self):
        w = self.window()
        if hasattr(w, "show_step"): w.show_step(0)
        if self.session is not None: self.session.new_side()

    def delete_sides(self, ns):
        """Delete the ticked sides' frames from the project (the Scan step's Delete, asked for from Raw frames): for sides from
        the wrong scan. Highest number first, so the numbers of the rest do not move under it."""
        s = self.state
        if not ns: return
        if self.session is None or s is None: self.notice(OFF["ended"] if self._ended else W.NOTICE["no_sides"]); return
        if s.recording or s.building or s.busy or self.run_mine is not None:
            self.notice(W.NOTICE["recording"] if s.recording else W.NOTICE["tools_off_run"]); return
        names = ", ".join("side %d" % n for n in ns)
        what = ("%s (%d sides)" % (names, len(ns))) if len(ns) > 1 else names
        from PySide6.QtWidgets import QMessageBox
        from app2.project_preview import backup_kind, BACKUP_WORDS
        note = BACKUP_WORDS[backup_kind(self.host_root or "")][1] if self.host_root else ""
        ask = W.PANELS["raw"]["tabs"][0]["delete_ask"].format(what=what) + ("\n\n" + note if note else "")
        if QMessageBox.question(self, "Delete sides", ask) != QMessageBox.StandardButton.Yes: return
        for n in sorted(ns, reverse=True): self.session.pick_side(n); self.session.delete_side(n)

    def resume_scan(self, n):
        w = self.window()
        if hasattr(w, "show_step"): w.show_step(0)
        if self.session is not None: self.session.pick_side(n)

    # ================================================================ runs
    def start_run(self, backend):
        s = self.state
        if self.session is None or s is None: self.notice(OFF["ended"] if self._ended else W.NOTICE["no_sides"]); return False
        if s.recording: self.notice(W.NOTICE["recording"]); return False
        if s.building or self.run_mine is not None:        # a run asked for and not started yet counts as running
            self.notice(W.NOTICE["scan_build"] if (s.building and self.run_mine is None) else W.NOTICE["tools_off_run"]); return False
        if self.snap is not None and running_elsewhere(os.path.join(self.host_root or "", "stages")): self.notice(W.NOTICE["locked"]); return False
        if not s.any_frames: self.notice(W.NOTICE["no_sides"]); return False
        return self.check_build(backend,self._launch_run)

    def check_build(self, backend, launch):
        if getattr(self,'_memory_pending',None):return False
        self._ensure_host(force=True)
        if self.host is None:return False
        token=object();session=self.session;root=self.host_root
        self._memory_pending=token
        def ready(plan):
            if self._memory_pending is not token or self.session is not session or self.host_root!=root or self._shut:return
            if not plan.get('low'):
                self._memory_pending=None;launch(backend);return
            from app2.qt.build_memory import BuildMemoryDialog
            dialog=self._memory_dialog=BuildMemoryDialog(plan,self.window())
            def choose(accepted):
                if self._memory_pending is not token:return
                self._memory_pending=None;self._memory_dialog=None
                if not accepted:
                    if getattr(self,'_ultra',None):self._ultra_done(False)
                    return
                import copy
                limited=copy.copy(backend)
                limited.env=dict(getattr(backend,'env',{}),POINTYOINK_BUILD_WORKERS='1',POINTYOINK_BUILD_GPU_SLOTS='1')
                if self.session is session and self.host_root==root and not self._shut:launch(limited)
            dialog.choice.connect(choose);dialog.open()
        def failed(error):
            if self._memory_pending is token:
                self._memory_pending=None;self.notice('Build could not be checked: '+str(error).splitlines()[0])
                if getattr(self,'_ultra',None):self._ultra_done(False)
        self.call('h_build_memory',getattr(backend,'to',None),getattr(backend,'force',()),getattr(backend,'from_',None),getattr(backend,'with_params',None),then=ready,fail=failed)
        return True

    def _cancel_memory_choice(self):
        self._memory_pending=None
        dialog=getattr(self,'_memory_dialog',None)
        if dialog is not None:dialog.reject();self._memory_dialog=None

    def _launch_run(self, backend):
        s=self.state
        if self._shut or self.session is None or s is None or s.recording or s.building or self.run_mine is not None:return False
        self._cancel_measurements()
        self.run_mine = backend; self.prog = None; self.undo_text = None; self.stopping = False
        self.session.build(backend)
        self._run_watchdog_backend=backend;self._run_watchdog.start(4000)
        self._refresh_all(); return True

    def _cancel_run_check(self):
        timer=getattr(self,"_run_watchdog",None)
        if timer is not None:timer.stop()
        self._run_watchdog_backend=None

    def _run_check_timeout(self):
        backend=self._run_watchdog_backend
        if backend is not None:self._run_check(backend)

    def _run_check(self, backend):
        """A run asked for that never showed as building (refused, or so quick the state flipped back unseen): end it
        here, so the next Apply is not refused as "a build is running"."""
        if self._shut or self.run_mine is not backend or self.running(): return
        if self.prog is None:
            self.run_mine = None; self.notice(W.NOTICE["locked"] if running_elsewhere(os.path.join(self.host_root or "", "stages")) else
                                              W.NOTICE["not_started"])
            self._refresh_all()
        elif getattr(self.prog, "finished", False): self._run_ended()
        else:self._run_watchdog_backend=backend;self._run_watchdog.start(2000)

    def _with_pending_sides(self, then):
        """Line-up's Use and How scanned choices wait for that panel's own Apply. A build started from somewhere else (YOLO Yoink,
        Rebuild the rest) must not ignore them, so they are applied first (no run of their own), then `then` goes on."""
        lp = self.panel.get("lineup") if hasattr(self, "panel") else None
        ch = lp.pending_sides() if lp is not None and hasattr(lp, "pending_sides") else []
        hints = lp.pending_hints() if lp is not None and ch is not None and hasattr(lp, "hints_dirty") and lp.hints_dirty() else None
        if not ch and hints is None: then(); return
        now = {x["id"]: x["active"] for x in (self.snap.sides if self.snap else [])}; now.update(dict(ch))
        if now and not any(now.values()): self.notice(W.NOTICE["one_side"]); return
        calls = [("include" if use else "leave_out", (sid,), None) for sid, use in ch]
        if hints is not None: calls.append(("set", ({"lineup.place.hints": hints},), None))
        self.call("h_apply", calls, then=lambda _r: then(),
                  fail=lambda e: self.notice(W.NOTICE["one_side"] if "at least one side" in str(e) else str(e).splitlines()[0]))

    def rebuild_rest(self):
        if self.running(): return
        if getattr(self, "pending_edits", 0): self.flush_pending_edits(); return   # the strip's Rebuild while deletes wait
        run = StagesRun(); run.land = self._never_built()                    # "Build it": open on the model after
        self._with_pending_sides(lambda: self.start_run(run))

    def pause_build(self, on):
        """Pause / Resume the running build (its processes are frozen, nothing is lost)."""
        if self.running() and self.session is not None: self.session.pause_build(on)

    def stop(self):
        if self.running() and self.session is not None:
            self.stopping = True; self._stopped = True; self.session.cancel_build(); self._refresh_progress()
        elif self.busy_call and self.host is not None: self.host.cancel()

    # ---------------- undo / redo / history / versions
    def undo(self, steps):
        """Ctrl+Z / Ctrl+Shift+Z / History / the arrows: the same rules as the arrows (never while a run, a CLI run or an
        engine call is going, never with nothing to undo)."""
        if self.running() or self.host is None or self.busy_call or not getattr(self, "_tools_on", False): return
        can_undo, can_redo = getattr(self, "_can", (False, False))
        if (steps < 0 and not can_undo) or (steps > 0 and not can_redo) or not steps: return
        if self.snap is not None and running_elsewhere(os.path.join(self.host_root or "", "stages")): self.notice(W.NOTICE["locked"]); return
        if getattr(self, "pending_edits", 0): self.pending_edits = 0; self.pending_changed()     # the run after the undo rebuilds the waiting deletes too
        def after(notes):
            if not notes: return                             # nothing moved: nothing to rebuild or publish
            self.notice(("Undo: " if steps < 0 else "Redo: ") + str(notes[-1]), level="info", secs=4)
            self.undo_text = steps < 0
            self.call("h_plan", then=self._after_undo)
        if steps in (-1, 1): self.call("undo" if steps < 0 else "redo", then=lambda n: after([n] if n else []))
        else: self.call("h_history_walk", steps, then=after)

    def _after_undo(self, pl):
        self.published = None
        self.start_run(StagesRun())  # Restore the visible result and the published model together.

    def open_history(self):
        if self.hist_pop.isVisible(): self.hist_pop.close_popover(); return          # the button again closes it
        s = self.snap
        if s is None: return
        self.hist_pop.set_entries(s.entries or [], s.cursor if s.cursor is not None else -1)
        self.hist_pop.place(self.hist_b.mapToGlobal(self.hist_b.rect().topLeft()), max(160, self.height() - 60))

    def _versions_node(self):
        side = self._side()
        return {"side": "side.keep@%s" % (side or {}).get("id"), "lineup": "lineup.adjust", "fused": "fused", "meshed": "meshed", "report": "report"}.get(self.stage)

    def open_versions(self):
        if self.vers_pop.isVisible(): self.vers_pop.close_popover(); return          # the button again closes it
        node = self._versions_node()
        if node is None: return
        cur = (self._node_for_view() or {}).get("key")
        def got(vs):
            self.vers_pop.set_versions(LABEL[self.stage], vs or [], cur)
            self.vers_pop.place(self.vers_b.mapToGlobal(self.vers_b.rect().topLeft()), max(160, self.height() - 60))
        self.call("versions", node, then=got)

    def do_pin(self, k16, on):
        node = self._versions_node(); self.call("pin", node, k16, on, then=lambda _r: self.open_versions())

    # ---------------- Compare / Before
    def _clear_compare_change(self):
        self.change_baseline = None
        if self.before_kind == "change": self.before_kind = "tool"
        if hasattr(self, "vp"): self.vp.before_change_action.setEnabled(False)

    def _remember_compare_change(self, expected_recipe, before_label, after_label):
        """Keep an immutable artifact path, never the mutable published model.ply.

        expected_recipe is the recipe after the confirmed change. Batch edits use
        this same hook before posting their engine call.
        """
        import copy
        self._clear_compare_change()
        path = self._view_path(); root = self.host_root or getattr(self.state, "project", None)
        if not path or not root or self.snap is None: return False
        cache = os.path.realpath(os.path.join(root, "stages", "nodes")) + os.sep
        if not os.path.realpath(path).startswith(cache) or not os.path.isfile(path): return False
        node = self._node_for_view() or {}
        if node.get("status") not in DONE_STATES: return False
        self.change_baseline = dict(path=path, preview=node.get("preview"), root=root,
                                    stage=self.stage, token=self._list_token(),
                                    before_recipe=copy.deepcopy(self.snap.recipe),
                                    after_recipe=copy.deepcopy(expected_recipe), changed=False,
                                    before_label=before_label, after_label=after_label)
        self.before_kind = "change"; self.vp.before_change_action.setEnabled(True)
        if self.comparing: self._load_before()
        return True

    def _sync_compare_change(self):
        baseline = self.change_baseline
        if baseline is None: return None
        root = self.host_root or getattr(self.state, "project", None)
        if (self.snap is None or baseline["root"] != root or baseline["stage"] != self.stage
                or baseline["token"] != self._list_token() or not os.path.isfile(baseline["path"])):
            self._clear_compare_change()
            if self.comparing: self._load_before()
            return None
        recipe = self.snap.recipe
        if recipe == baseline["after_recipe"]: baseline["changed"] = True
        elif baseline["changed"] or recipe != baseline["before_recipe"]:
            self._clear_compare_change()
            if self.comparing: self._load_before()
            return None
        self.vp.before_change_action.setEnabled(True)
        return baseline

    def toggle_compare(self):
        self.comparing = not self.comparing; self.vp.set_comparing(self.comparing)
        if self.comparing:
            if self.gl: self.set_tool("orbit")
            self._load_before()
        self._refresh_tools(); self._refresh_chips()

    def pick_before(self, what):
        if what == "version": self.open_versions(); return
        if what == "change" and self._sync_compare_change() is None: return
        self.before_kind = what; self._load_before(); self._refresh_chips()

    def compare_version(self, k16):
        self.vers_pop.close_popover()
        self.before_kind = "version"
        self._refresh_chips()
        if not self.comparing: self.comparing = True; self.vp.set_comparing(True)
        node = self._versions_node()
        when = getattr(self, "_before_versions", {}).get(k16)
        self.call("output", node, None, k16, then=lambda p: self._load_before(path=p, label=W.VIEW["version_from"].format(when=when) if when else "Version %s" % k16[:6]))

    def _refresh_before_versions(self):
        """The Before menu lists this stage's last few versions by date, so one is a click away."""
        node = self._versions_node()
        if node is None: self.vp.set_before_versions([]); return
        cur = (self._node_for_view() or {}).get("key")
        def got(vs):
            from app2.qt.history import version_row
            rows = []; self._before_versions = {}
            for v in (vs or [])[:6]:
                if cur and str(cur).startswith(v["key16"]): continue
                when, text = version_row(v); rows.append((v["key16"], text)); self._before_versions[v["key16"]] = when
            self.vp.set_before_versions(rows)
        self.call("versions", node, then=got)

    def _before_path(self):
        """Before the last tool (the parent of the chain's last step) or before all tools (the stage's base)."""
        s = self.snap
        if s is None: return None, None
        if self.before_kind == "change":
            baseline = self._sync_compare_change()
            if baseline: return baseline["path"], baseline["before_label"]
        ch = [entry for entry in self._chain() if entry[2].get("on", True)]; base = {"side": "side.keep@%s" % (self._side() or {}).get("id"), "fused": "fused", "meshed": "meshed"}.get(self.stage)
        if base is None: return None, None
        steps = ["%s#%s" % (lk, o["id"]) for lk, j, o in ch]
        if self.before_kind == "stage" or len(steps) <= 1: return s.out(base), W.VIEW["compare_menu"][1][1]
        return s.out(steps[-2]), W.VIEW["compare_menu"][0][1]

    def _load_before(self, path=None, label=None):
        if path is None: path, label = self._before_path()
        self.before_label = label or W.VIEW["before"]
        short = W.VIEW["before_short"].get(self.before_kind, self.before_kind)
        if self.before_kind == "version" and label: short = label.replace(W.VIEW["version_from"].split("{")[0], "", 1)
        self.vp.set_before_choice(self.before_kind, short); self._refresh_before_versions()
        self.vp.before.setToolTip(self.before_label)
        if not self.gl:
            s = self.snap; ch = [entry for entry in self._chain() if entry[2].get("on", True)]
            nid = ("%s#%s" % (ch[-2][0], ch[-2][2]["id"])) if (len(ch) > 1 and self.before_kind == "tool") else {"fused": "fused", "meshed": "meshed"}.get(self.stage)
            baseline = self._sync_compare_change() if self.before_kind == "change" else None
            pv = baseline.get("preview") if baseline else ((s.node(nid) if (s and nid) else None) or {}).get("preview")
            if pv and os.path.exists(pv):
                from PIL import Image
                self.vp.before.set_image(np.asarray(Image.open(pv).convert("RGB")))
            else: self.vp.before.set_image(None)
            self.vp.before.setToolTip(self.before_label)
            self.vp.before.set_chips([(self.before_label, None)]); return
        if not path or self.loader is None:
            if self.vp.before.view is not None: self.vp.before.view.clear()
            self.vp.before.set_chips([(W.VIEW["before"], None)], center=W.VIEW["not_built"]); return
        self.view_id += 1; self.before_id = self.view_id
        self.loader.request(dict(id=self.view_id, slot="before", items=[dict(key="main", path=path)], edit=None, mode="shaded", path=path))

    def _on_before_view(self, out):
        if out.get("id") != getattr(self, "before_id", None) or out.get("error") or out.get("too_big"): return
        bv = self.vp.before.view if self.comparing else None
        self.before_payload = out
        if self.comparing and bv is not None:
            bv.show_items(out["items"], edit=None, frame=False); self._orient_before(); bv.set_flat(self.colour_mode.get(self.stage) == "flat"); bv.set_mesh_points(self.colour_mode.get(self.stage) == "points"); self._sync_cam(self.vp.main, self.vp.before)
            self.vp.before.set_chips([(self.before_label, None)],
                                     ("Polygons: %d" if out["items"] and out["items"][0]["kind"] == "mesh" else "Points: %d") % (out["items"][0]["n"] if out["items"] else 0))
        if self.peek: self._peek_show()

    def set_peek(self, on):
        """B held: the single view shows Before."""
        if self.comparing or not self.snap: return
        self.peek = on
        if on:
            if not self.gl:
                self._load_before(); self.vp.main.hide(); self.vp.before.show(); return
            self._load_before()
        else:
            if not self.gl: self.vp.before.hide(); self.vp.main.show(); return
            self.vp.before.hide(); self.vp.main.show()
        self._refresh_chips()

    def _peek_show(self):
        if not self.peek or not self.gl: return
        out = getattr(self, "before_payload", None)
        if not out: return
        bv = self.vp.before.view; bv.show_items(out["items"], edit=None, frame=False); self._orient_before()
        bv.cam.view = self.vp.view.cam.view.copy(); bv.cam.pivot = self.vp.view.cam.pivot.copy(); bv.cam.bound = self.vp.view.cam.bound
        self.vp.before.set_chips([(self.before_label, None)]); self.vp.main.hide(); self.vp.before.show()

    # ---------------- Export STL
    def _export_needs_rebuild(self):
        """A legacy model is not a substitute for pending staged edits or cuts."""
        s = self.snap
        if s is None: return False
        end = s.final("meshed"); node = s.node(end) or s.node("meshed") or {}
        if node.get("status") in DONE_STATES:
            current = s.out(end, "mesh")
            return not (current and os.path.exists(current))
        if s.out(end, "mesh") or s.out("fused"):
            return True
        nodes = [n for group in (s.groups or []) for n in group["nodes"]]
        if any(n.get("status") in ("stale", "failed", "error") for n in nodes): return True
        if any(n["id"].startswith("side.platter@") and n.get("method") != "none"
               and (s.out(n["id"], "plane") or n.get("method") == "manual") for n in nodes): return True
        return any(op.get("tool") == "cut_plane" and op.get("on", True)
                   for chain in (s.chains or {}).values() for _key, _index, op in chain)

    def _export_src(self):
        """Use current Meshed; a legacy model is allowed only when no staged edit is pending."""
        s = self.snap; root = self.host_root or getattr(self.state, "project", None)
        if s is not None:
            end = s.final("meshed"); n = s.node(end) or {}
            p = s.out(end, "mesh") if n.get("status") in DONE_STATES else None
            if p and os.path.exists(p): return p
        if self._export_needs_rebuild(): return None
        m = os.path.join(root or "", "model.ply")
        return m if root and os.path.exists(m) else None

    def can_export(self):
        """A built model of the open project is on disk (the shell's File > Export and the ribbon's Export)."""
        return not self._shut and bool(self._export_src())

    def _export_stale(self, src):
        """Export falls back to model.ply while Meshed has a newer, not yet built state."""
        s = self.snap
        if s is None or not src or os.path.basename(src) != "model.ply": return False
        n = s.node(s.final("meshed")) or s.node("meshed") or {}
        return bool(n) and n.get("status") not in DONE_STATES

    def _ask_save(self, default, title=None, filt=None):
        from PySide6.QtWidgets import QFileDialog
        library_dir.ensure_parent(default)
        path, _f = QFileDialog.getSaveFileName(self, title or W.EXPORT["title"], default, filt or W.EXPORT["filter"])
        return path

    def export_stl(self):
        """Export STL: a file dialog (default <Exports folder>/<project name>_print.stl), then the write on the engine host's
        thread (the GUI never waits); the size shows under the button and in the notice."""
        if getattr(self, "pending_edits", 0) and self.flush_pending_edits(): self.tell(W.NOTICE["rebuild_now"], level="info", secs=8); return   # the deletes first
        src = self._export_src(); b = self.panel["meshed"].body(FINISH)
        if not src: self.tell(EXPORT_REBUILD if self._export_needs_rebuild() else W.EXPORT["nothing"]); return
        if not self._shut: self._ensure_host(force=True)
        if self.snap is None and self.host is not None:  # Build not opened yet: the engine's status first (is Meshed newer?)
            self._export_next = self.export_stl
            self._export_after_status = True; self.host.status(); self.tell(EXPORT_CHECKING, level="info", secs=4); return
        stale = self._export_stale(src)
        if stale: self.tell(EXPORT_STALE, secs=12)
        root = self.host_root or os.path.dirname(src)
        dst = self.ask_save(library_dir.export_path(os.path.basename(os.path.normpath(root)) + W.EXPORT["suffix"]))
        if not dst: return
        if not dst.lower().endswith(".stl"): dst += ".stl"
        name = os.path.basename(dst)
        b.set_note(W.EXPORT["saving"].format(name=name))

        def done(r):
            text = W.EXPORT["done"].format(name=name, mb="%.1f" % (r["bytes"] / 1e6), faces=r["faces"])
            if stale: text += " It is the last built model.ply: Meshed is out of date."
            b.set_note(text); self.tell(text, level="warn" if stale else "info", secs=12)

        def failed(e):
            text = W.EXPORT["failed"].format(why=str(e).splitlines()[0][:160]); b.set_note(text); self.tell(text)
        self.call("h_export_stl", src, dst, **self._orient_kw(), then=done, fail=failed)

    # ---------------- Export sheet (File > Export..., the ribbon's Export): a format, or a ZIP with the scan data
    def _model_dims(self, path, then=None):
        """The model's "W x D x H mm" when measured (per path and change time), else None: asked of the engine host once
        (it reads the vertices there), then the info line and `then` get it ("" when it could not be measured)."""
        if not path or self._shut or self.host is None: return None
        try: key = (path, os.path.getmtime(path))
        except OSError: return None
        cache = self.__dict__.setdefault("_dims", {})
        if key in cache: return cache[key]
        waiting = self.__dict__.setdefault("_dims_then", {})
        if key in waiting:
            if then: waiting[key].append(then)
            return None
        waiting[key] = [then] if then else []

        def got(text):
            cache[key] = text
            for f in waiting.pop(key, ()): f(text)
            self._refresh_info()
        self.call("h_model_dims", path, then=lambda r: got(r["text"]), fail=lambda e: got(""))
        return None

    def export_sheet(self):
        if getattr(self, "pending_edits", 0) and self.flush_pending_edits(): self.tell(W.NOTICE["rebuild_now"], level="info", secs=8); return   # the deletes first
        from app2.qt.export_sheet import ExportDialog
        d = self.__dict__.get("export_dlg")
        if d is None:
            d = self.export_dlg = ExportDialog(self.window())
            d.cancel.connect(self.close_export); d.ok.connect(self._export_as)
            d.timelapse.connect(lambda: (self.close_export(), self.timelapse_requested.emit()))
            d.watertight.connect(self._export_make_watertight)
        return d

    def _export_closed_check(self, src, d):
        """Count the model's open edges off the window's thread; the sheet's line says closed or not."""
        import threading
        from app2.builder.export_model import open_edges
        key = (src, os.path.getmtime(src)) if os.path.exists(src) else None
        cache = self.__dict__.setdefault("_closed_cache", {})
        if key in cache: d.set_closed(cache[key], can_fix=self._can_make_watertight()); return
        d.set_closed(None)
        def work():
            n = open_edges(src)
            if not self._shut: self.closed_counted.emit((key, -1 if n is None else n))
        if not self.__dict__.get("_closed_wired"):
            self._closed_wired = True; self.closed_counted.connect(self._export_closed_got)
        threading.Thread(target=work, daemon=True, name="open-edges").start()

    def _export_closed_got(self, item):
        key, n = item; self.__dict__.setdefault("_closed_cache", {})[key] = n
        d = self.__dict__.get("export_dlg")
        if d is not None and d.isVisible(): d.set_closed(n, can_fix=self._can_make_watertight())

    def _can_make_watertight(self):
        s = self.snap
        return bool(s is not None and self.host is not None and getattr(s, "meshed_method", None) != "poisson" and not self.running())

    def _export_make_watertight(self):
        """The sheet's Make watertight and rebuild: Meshed to Watertight, run to meshed; Export again when it is done."""
        self.close_export()
        self.show_stage("meshed")
        self.notice(W.EXPORT_SHEET["make_watertight"] + ": Meshed is set to Watertight and building. Export again when it is done.", level="info", secs=12)
        self._set_then_run({"meshed.method": "poisson"}, "meshed", StagesRun(to="meshed", publish=True))

    def close_export(self):
        d = self.__dict__.get("export_dlg")
        if d is not None: d.close_sheet()

    def open_export(self):
        """File > Export... and the ribbon's Export: the sheet (last format, the model's size), then a file dialog."""
        from app2.builder import export_model as X
        src = self._export_src()
        if not src: self.tell(EXPORT_REBUILD if self._export_needs_rebuild() else W.EXPORT["nothing"]); return
        if not self._shut: self._ensure_host(force=True)
        if self.snap is None and self.host is not None:  # Build not opened yet: the engine's status first (is Meshed newer?)
            self._export_next = self.open_export
            self._export_after_status = True; self.host.status(); self.tell(EXPORT_CHECKING, level="info", secs=4); return
        d = self.export_sheet(); root = self.host_root or os.path.dirname(src)
        d.present([k for k, _l in X.formats()], X.last_format(), can_zip=bool(X.side_folders(root)))
        dims = self._model_dims(src, then=lambda t: d.set_dims(t) if d.isVisible() else None)
        if dims is not None: d.set_dims(dims)
        v, f, col = X.ply_counts(src); d.set_has_color(col); est = X.size_estimates(v, f, col)
        scan = X.folders_bytes(X.side_folders(root)) if X.side_folders(root) else 0
        d.set_files(est, scan)
        self._export_closed_check(src, d)

    def _export_as(self, fmt, with_scan):
        from app2.builder import export_model as X
        E = W.EXPORT_SHEET; self.close_export(); X.save_format(fmt)
        src = self._export_src()
        if not src: self.tell(EXPORT_REBUILD if self._export_needs_rebuild() else W.EXPORT["nothing"]); return
        stale = self._export_stale(src)
        if stale: self.tell(EXPORT_STALE, secs=12)
        root = self.host_root or os.path.dirname(src); stem = os.path.basename(os.path.normpath(root))
        ext = "zip" if with_scan else fmt
        dst = self.ask_save(library_dir.export_path(stem + (E["zip_suffix"] if with_scan else E["suffix"]) + "." + ext),
                            E["save_title"].format(fmt=ext.upper()),
                            E["zip_filter"] if with_scan else E["filter"].format(fmt=fmt.upper(), ext=fmt))
        if not dst: return
        if not dst.lower().endswith("." + ext): dst += "." + ext
        name = os.path.basename(dst); b = self.panel["meshed"].body(FINISH)
        b.set_note(W.EXPORT["saving"].format(name=name))

        def done(r):
            mb = "%.1f" % (r["bytes"] / 1e6); dims = r.get("dims") or ""
            if with_scan: text = E["zip_done"].format(name=name, mb=mb, faces=r["faces"], dims=dims, sides=r.get("sides", 0), s="" if r.get("sides") == 1 else "s")
            else: text = E["done"].format(name=name, mb=mb, faces=r["faces"], dims=dims)
            if stale: text += " It is the last built model.ply: Meshed is out of date."
            b.set_note(text); self.tell(text, level="warn" if stale else "info", secs=12)

        def failed(e):
            text = E["failed"].format(why=str(e).splitlines()[0][:160]); b.set_note(text); self.tell(text)
        self.call("h_export", src, dst, fmt, root if with_scan else None, **self._orient_kw(), then=done, fail=failed)

    # ---------------- YOLO Yoink
    # ---------------- recipe chip
    def _recipe_info(self):
        """(key, display name, edited): the recipe Build uses. edited: its Fused / Meshed settings or tools differ from
        what the preset sets (a slider moved since)."""
        r = (self.snap.recipe if self.snap else None) or {}
        key = r.get("preset") or "today"
        names = {k: label for k, label, _h in W.ONETAP["presets"]}
        name = names.get(key, "Custom" if key == "custom" else key)
        edited = False
        try:
            from stages import params as PR
            if key in PR.PRESETS:
                ref = PR.apply_preset(r, key)
                part = lambda x: ({k: v for k, v in (x.get("params") or {}).items() if k.split(".")[0] in ("fused", "meshed")},
                                  {k: v for k, v in (x.get("ops") or {}).items() if k.split(".")[0] in ("fused", "meshed")})
                edited = part(ref) != part(r)
        except Exception: pass
        return key, name, edited

    def _recipe_text(self):
        key, name, edited = self._recipe_info()
        return W.RECIPE["edited"].format(name=name) if edited else name

    def _refresh_recipe(self):
        if not hasattr(self, "recipe_b"): return
        self.recipe_b.setText(W.RECIPE["chip"].format(name=self._recipe_text()) + "  \u25be")
        self.recipe_b.setEnabled(self.snap is not None and not self.running() and self.busy_call is None)

    def _fill_recipe_menu(self):
        m = self.recipe_menu; m.clear(); key, _n, _e = self._recipe_info()
        for k, label, _h in W.ONETAP["presets"]:
            a = m.addAction("%s   \u00b7  %s" % (label, W.RECIPE["menu_hint"].get(k, ""))); a.setCheckable(True); a.setChecked(k == key)
            a.triggered.connect(lambda _c=False, kk=k: self.pick_recipe(kk))
        m.addSeparator()
        m.addAction(W.RECIPE["ultra"]).triggered.connect(self.open_ultra)
        m.addAction(W.RECIPE["all_options"]).triggered.connect(self.open_onetap)

    def pick_recipe(self, key):
        """Switch the recipe Build uses; nothing builds until Build (the bar then says what will rebuild)."""
        if self.running() or self.busy_call is not None or self.snap is None: return
        names = {k: label for k, label, _h in W.ONETAP["presets"]}
        self.call("preset", key, then=lambda _r: (self.notice(W.RECIPE["picked"].format(name=names.get(key, key)), level="info", secs=4),
                                                   self.host.status() if self.host is not None else None))

    def ultra_running(self): return bool(getattr(self, "_ultra", None))

    def start_ultra(self, keys, sheet):
        """Build each picked recipe in turn (preset, then a full run); results go to the sheet as they finish."""
        if self.running() or self.busy_call is not None or self.ultra_running(): return
        self._ultra = dict(queue=list(keys), all=list(keys), results=[], sheet=sheet, cur=None)
        self._ultra_next()

    def _ultra_next(self):
        u = getattr(self, "_ultra", None)
        if not u: return
        names = {k: label for k, label, _h in W.ONETAP["presets"]}
        if not u["queue"]:
            self._ultra = None; u["sheet"].show_results(u["results"]); return
        k = u["queue"].pop(0); u["cur"] = (k, time.time())
        u["sheet"].show_running(len(u["results"]), len(u["all"]), names.get(k, k))
        self.call("preset", k, then=lambda _r: self.start_run(StagesRun(what="onetap")),
                  fail=lambda e: self._ultra_done(False))

    def _ultra_done(self, ok):
        """A recipe's run ended: note what it made, then the next one (a stop or failure ends Ultra Yoink)."""
        u = getattr(self, "_ultra", None)
        if not u or not u["cur"]: return
        k, t0 = u["cur"]; u["cur"] = None
        m = os.path.join(self.host_root or "", "model.ply"); faces = None; size = 0
        if ok and os.path.exists(m):
            size = os.path.getsize(m)
            try:
                with open(m, "rb") as f: head = f.read(4096).decode("latin-1")
                faces = next((int(l.split()[2]) for l in head.splitlines() if l.startswith("element face")), None)
            except (OSError, ValueError, IndexError): pass
        u["results"].append(dict(preset=k, ok=ok, secs=time.time() - t0, faces=faces, bytes=size))
        if ok: QTimer.singleShot(0, self._ultra_next)
        else: self._ultra = None; u["sheet"].show_results(u["results"], stopped=True)

    def show_recipe(self, key):
        """Ultra Yoink's Show: make it the recipe and run (its results are stored, so this only re-shows them)."""
        if self.running() or self.busy_call is not None: return
        self.call("preset", key, then=lambda _r: self.start_run(StagesRun(what="onetap")))

    def open_ultra(self):
        from app2.qt.ultra import UltraSheet
        UltraSheet.open_for(self)

    def open_onetap(self):
        if not self.onetap.isEnabled() or not self.isVisible(): return
        self.scrim.setGeometry(self.rect()); self.scrim.show(); self.scrim.raise_()
        d = self.onetap_dlg; d.fix_size(); d.move((self.width() - d.width()) // 2, max(10, (self.height() - d.height()) // 2))
        d.show(); d.raise_()
        preset = ((self.snap.recipe or {}).get("preset") if self.snap else None) or "auto"
        if preset == "today": preset = "auto"            # the same settings: Auto "Let the app choose" is the one a first-timer should see picked
        if preset not in d.btns: preset = next(iter(d.btns))
        d.pick(preset, ask=True)

    def _preview_onetap(self, name, args):
        self._preset_preview = getattr(self, "_preset_preview", 0) + 1
        ticket = self._preset_preview
        def accept(value, failed=False):
            if ticket != self._preset_preview: return
            if failed: self.onetap_dlg.show_error(str(value).splitlines()[0])
            else: self.onetap_dlg.show_summary(value)
            self._center_onetap()
        if self.host is None:
            accept("Open a project with scans to preview these settings.", True); return
        self.call(name, *args, then=accept, fail=lambda e: accept(e, True))

    def _ask_onetap(self, preset):
        self._preview_onetap("h_onetap", (preset,))

    def _ask_custom_onetap(self, payload):
        self._preview_onetap("h_custom_onetap", (payload["fragment"], payload["preset"]))

    def do_custom_onetap(self, payload):
        self.close_onetap(); self.last_changed = "fused"
        self._with_pending_sides(lambda: self.call("h_apply_custom", payload["fragment"], payload["preset"],
                                                    then=lambda _r: self._start_onetap_run()))

    def _start_onetap_run(self):
        """YOLO Yoink in two parts: first up to the line-up, where a side that does not line up with the others is found and left out
        (with a warning); then the rest. Every result of the first part is reused by the second."""
        run = StagesRun(what="onetap", to="lineup.adjust", publish=False); run.onetap_phase = 1
        self.start_run(run)

    def _onetap_check(self):
        go = lambda: self.start_run(StagesRun(what="onetap"))
        if self.host is None: go(); return
        self.call("h_side_agreement", then=lambda r: self._onetap_flagged(r, go), fail=lambda e: go())

    def _onetap_flagged(self, r, go):
        ids = list((r or {}).get("bad") or []); ns = list((r or {}).get("bad_n") or [])
        if not ids: go(); return
        sides = (r or {}).get("sides") or {}
        calls = [("leave_out", (sid,), None) for sid in ids]
        def done(_r=None):
            info = {sid: {"n": n, "agree": round(100 * sides.get(str(n), {}).get("agree", 0)), "alone": round(100 * sides.get(str(n), {}).get("alone", 0), 1)}
                    for sid, n in zip(ids, ns)}
            self._auto_out_save(info)
            who = "Side %d was" % ns[0] if len(ns) == 1 else "Sides %s were" % join_words(str(n) for n in ns)
            low = min(v["agree"] for v in info.values())
            self.notice(W.NOTICE["auto_out"].format(who=who, agree=low), level="warn", secs=45); go()
        self.call("h_apply", calls, then=done, fail=lambda e: go())

    def _auto_out_path(self): return os.path.join(self.host_root or "", "auto_left_out.json")

    def _auto_out_load(self):
        """{side id: {n, agree, alone}} for sides YOLO left out and that are still out (one turned back on is dropped)."""
        import json
        try:
            with open(self._auto_out_path()) as f: d = json.load(f)
        except (OSError, ValueError): return {}
        active = {x["id"]: x["active"] for x in ((self.snap.sides if self.snap else None) or [])}
        return {k: v for k, v in d.items() if active.get(k) is False}

    def _auto_out_save(self, info):
        import json
        d = self._auto_out_load(); d.update(info)
        try:
            with open(self._auto_out_path(), "w") as f: json.dump(d, f)
        except OSError: pass

    def _center_onetap(self):
        d = self.onetap_dlg; d.fix_size(); d.move((self.width() - d.width()) // 2, max(10, (self.height() - d.height()) // 2))

    def close_onetap(self):
        self.onetap_dlg.hide(); self.scrim.hide()

    def hideEvent(self, e):
        super().hideEvent(e)
        if not e.spontaneous(): self.close_onetap()         # another page or step: the sheet does not wait behind it

    def do_onetap(self, preset):
        self.close_onetap(); self.last_changed = "fused"
        self._with_pending_sides(lambda: self.call("preset", preset, then=lambda _r: self._start_onetap_run()))

    # ================================================================ sizes
    def resizeEvent(self, e):
        super().resizeEvent(e); self._layout_sizes()
        if hasattr(self, "log_pop") and self.log_pop.isVisible(): fit_overlay_in_parent(self.log_pop)
        if hasattr(self, "edit_confirm") and self.edit_confirm.isVisible():
            self.edit_confirm_scrim.setGeometry(self.rect())
            self.edit_confirm.move((self.width()-self.edit_confirm.width())//2, max(8,(self.height()-self.edit_confirm.height())//2))

    def _layout_sizes(self):
        narrow = self.window().width() < 1280 if self.window() is not None else self.width() < 1100
        self.left.setFixedWidth(200 if narrow else 248); self.right.setFixedWidth(330 if narrow else 400)
        self.info.set_narrow(narrow)
        self.stages.set_compact(self.window().height() <= 900)
        from app2.qt.icons import icon
        from PySide6.QtGui import QIcon
        for b, name in ((self.log_b, "captures"), (self.hist_b, "history"), (self.vers_b, "save-version")):
            b.setIcon(QIcon() if narrow else icon(name))
        if self.scrim.isVisible(): self.scrim.setGeometry(self.rect()); self._center_onetap()
        self.vp._place()
