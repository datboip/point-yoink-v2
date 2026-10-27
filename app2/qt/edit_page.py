# SPDX-License-Identifier: AGPL-3.0-or-later
"""Edit uses the existing Build scene, history, and stage host in the same window."""
from app2.qt.help_text import HelpButton
from app2.qt.theme import ui_font
from PySide6.QtCore import QTimer, Qt, QSize, QRect
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QSpinBox, QScrollArea, QGridLayout, QSizePolicy


class EditPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.build = None; self.tools_expanded = False
        self.box = QVBoxLayout(self); self.box.setContentsMargins(0, 0, 0, 0)
        self.edge_button = QPushButton("‹"); self.edge_button.setFixedSize(14, 56); self.edge_button.setToolTip("Show edit settings"); self.edge_button.setAccessibleName("Show edit settings"); self.edge_button.clicked.connect(lambda: self.set_tools_expanded(not self.tools_expanded)); self.edge_button.hide()
        self.palette = QScrollArea(); self.palette.setWidgetResizable(True); self.palette.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        from app2.qt.icons import icon
        content = QWidget(); root = QVBoxLayout(content); root.setContentsMargins(4,8,12,8); root.setSpacing(8)      # the right gutter is the scroll bar's, so it never covers a button
        class _Target:
            """Where the palette's widgets go: the page's root column until the first step, then the current step's card
            (2026-10-25: the steps are cards with a numbered badge, not three gray labels in one long column)."""
            def __init__(t): t.cur = root; t.all = [root]
            def addWidget(t, *a, **k): t.cur.addWidget(*a, **k)
            def addLayout(t, *a, **k): t.cur.addLayout(*a, **k)
            def addSpacing(t, n): t.cur.addSpacing(n)
            def addStretch(t, *a): root.addStretch(*a)
            def setSpacing(t, n):
                for l in t.all: l.setSpacing(n)
        layout = _Target()
        self.tools = []; self.tool_buttons = {}; self.action_buttons = []
        self.tool_layout = layout
        def action(text, name=None, check=False):
            button = QPushButton(text); button.setFixedHeight(30); button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed); button.setFont(ui_font(12))
            button.setStyleSheet("QPushButton {padding: 3px 2px;} QPushButton:checked {background: #274a78; border: 1px solid #639feb; color: white; font-weight: normal;}")
            button.setCheckable(check); self.action_buttons.append(button)
            if name: button.setIcon(icon(name)); button.setIconSize(QSize(14,14))
            return button
        from app2.qt.widgets import ElidedLabel
        title_row = QHBoxLayout(); title_row.setSpacing(4); title=ElidedLabel("Edit model"); title.setFixedHeight(24); title_row.addWidget(title,1); self.title = title
        self.options_button=QPushButton('Options');self.options_button.setObjectName('EditMeshOptions');self.options_button.setFixedSize(74,26)
        self.options_button.setStyleSheet('QPushButton {padding:2px;}');self.options_button.clicked.connect(lambda:self.set_tools_expanded(not self.tools_expanded));title_row.addWidget(self.options_button)
        title_row.addWidget(HelpButton("Select faces", "Lasso draws a loop. Rectangle draws a box. Brush paints faces. Magic follows similar surface angles. Check the red selection before deleting.")); layout.addLayout(title_row)
        self.stage_row = QHBoxLayout(); self.stage_row.setSpacing(4); self.stage_buttons = {}
        for key, text in (("side", "Side"), ("fused", "Fused"), ("meshed", "Meshed")):
            b = action(text, check=True); b.setToolTip("Edit the %s" % {"side": "side you are on", "fused": "Fused result", "meshed": "finished model"}[key])
            b.clicked.connect(lambda checked=False, k=key: self.pick_stage(k)); self.stage_row.addWidget(b); self.stage_buttons[key] = b
        layout.addLayout(self.stage_row)
        self.carry = QLabel(""); self.carry.setWordWrap(True); self.carry.setFixedHeight(32); self.carry.setFont(ui_font(11)); self.carry.setProperty("role", "hint"); layout.addWidget(self.carry)
        self.reason=QLabel(""); self.reason.setWordWrap(True); self.reason.setMinimumHeight(22); layout.addWidget(self.reason)
        from PySide6.QtWidgets import QFrame
        def section(number, text):
            card = QFrame(); card.setObjectName("EditStep"); card.setStyleSheet("#EditStep {background:#1b2027; border:1px solid #2c343f; border-radius:8px;}")
            col = QVBoxLayout(card); col.setContentsMargins(3, 6, 3, 6); col.setSpacing(6)
            head = QHBoxLayout(); head.setSpacing(8)
            badge = QLabel(str(number)); badge.setFixedSize(22, 22); badge.setAlignment(Qt.AlignmentFlag.AlignCenter); badge.setFont(ui_font(12, QFont.Weight.Bold))
            badge.setStyleSheet("background:#274a78; color:white; border-radius:11px;")
            title = QLabel(text); title.setFont(ui_font(12, QFont.Weight.DemiBold)); title.setStyleSheet("color:#e7edf5;"); title.setMinimumWidth(0); title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            head.addWidget(badge); head.addWidget(title, 1); col.addLayout(head)
            root.addWidget(card); layout.cur = col; layout.all.append(col); return card
        self.sections = [section(1, "Select faces")]
        orbit=action("Orbit", "view-3d", True); orbit.clicked.connect(lambda:self.set_tool("orbit")); layout.addWidget(orbit); self.tools.append(orbit); self.tool_buttons["orbit"]=orbit
        grid=QGridLayout(); grid.setSpacing(6)
        piece=action("Piece", "magic", True); piece.clicked.connect(lambda:self.set_tool("piece")); piece.setToolTip("Click a piece of the model: the whole connected piece is selected. The quickest way to take table scraps and floaters off.")
        layout.addWidget(piece); self.tools.append(piece); self.tool_buttons["piece"]=piece
        for index,(label,tool,icon_name) in enumerate((("Lasso","lasso","lasso"),("Box","rect","box"),("Brush","brush","brush"),("Magic","magic","magic"))):
            button=action(label,icon_name,True); button.clicked.connect(lambda checked=False,t=tool:self.set_tool(t)); button.setToolTip("Magic previews faces in amber. Click to select them red. A thin bridge may still connect to the main part." if tool=="magic" else "Select faces with "+label.lower()+". Ctrl subtracts.")
            grid.addWidget(button,index//2,index%2); self.tools.append(button); self.tool_buttons[tool]=button
        layout.addLayout(grid)
        # 2026-10-25: Delete sits right under the tools ("move the delete up where the tools are, if the window is small you won't see it")
        self.delete_button=action("Delete selected","delete"); self.delete_button.setFixedHeight(34); self.delete_button.setToolTip("Delete the red selection. Undo restores it.")
        self.delete_button.clicked.connect(lambda:self.build.delete_selection() if self.build else None); layout.addWidget(self.delete_button)
        self.rebuild_button=action("Rebuild"); self.rebuild_button.setToolTip("Deletes leave the picture at once; this runs the one rebuild they wait for. It also runs by itself when you open a shape tool, another stage, or leave Edit.")
        self.rebuild_button.clicked.connect(lambda:self.build.flush_pending_edits() if self.build else None); self.rebuild_button.hide(); layout.addWidget(self.rebuild_button)
        self.how = QLabel(""); self.how.setWordWrap(True); self.how.setFont(ui_font(11)); self.how.setProperty("role", "hint"); self.how.setMinimumHeight(30); layout.addWidget(self.how)   # how the active tool is used
        def narrow(lab):
            """A label that never widens the column: it wraps or clips instead (the palette was pushed past its column, 2026-10-25)."""
            lab.setMinimumWidth(0); lab.setWordWrap(True); lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred); return lab
        self.how = narrow(self.how)
        mode_lab = narrow(QLabel("A new click or drag")); mode_lab.setToolTip("Shift while clicking or dragging adds, Ctrl takes out, whatever is picked here.")
        mode_title=QHBoxLayout(); mode_title.addWidget(mode_lab,1)
        self.mode_help=HelpButton("Selection mode", "Replace starts a new selection. Add keeps the old selection and adds faces. Subtract takes faces out. These choices do not change the model.")
        mode_title.addWidget(self.mode_help); layout.addLayout(mode_title)
        self.mode=QComboBox(content); self.mode.addItems(["Replace","Add","Subtract"]); self.mode.hide(); self.mode.currentTextChanged.connect(self.set_mode)
        self.mode_buttons={}; mode_row=QHBoxLayout(); mode_row.setSpacing(2)
        for name, text in (("Replace", "New"), ("Add", "Add"), ("Subtract", "Remove")):           # short: three across a narrow card
            button=action(text,check=True); button.setToolTip({"Replace": "Each new click or drag starts a fresh selection.", "Add": "Each new click or drag adds to the red selection.", "Subtract": "Each new click or drag takes faces out of the selection. Ctrl while dragging does the same at any time."}[name])
            button.clicked.connect(lambda checked=False,n=name:self.mode.setCurrentText(n)); mode_row.addWidget(button); self.mode_buttons[name]=button
        layout.addLayout(mode_row)
        reach_title = QHBoxLayout(); reach_title.addWidget(narrow(QLabel("Reach")), 1); reach_title.addWidget(HelpButton("Reach", "Visible only takes the faces you can see. Through also takes the faces hidden behind them, right through the part.")); layout.addLayout(reach_title)
        reach=QHBoxLayout(); reach.setSpacing(2)
        self.visible_button=action("Visible only",check=True); self.through_button=action("Through",check=True)
        self.visible_button.clicked.connect(lambda:self._set_through(False)); self.through_button.clicked.connect(lambda:self._set_through(True))
        self.through_button.setToolTip("Also select faces hidden on the back. The selection goes right through the part.")
        reach.addWidget(self.visible_button); reach.addWidget(self.through_button); layout.addLayout(reach)
        self.brush, self.brush_row = self._number(layout,"Brush radius (px)",2,100,24); self.angle, self.angle_row = self._number(layout,"Magic angle (deg)",1,85,49)   # shown with their tool only
        self.brush.valueChanged.connect(lambda value:self._view_setting("brush_px",value)); self.angle.valueChanged.connect(lambda value:self._view_setting("magic_angle",value))
        self.sections.append(section(2, "Adjust the selection"))
        select_row=QHBoxLayout(); self.invert_button=action("Invert","invert-selection"); self.clear_button=action("Clear","clear-selection")
        self.invert_button.clicked.connect(lambda:self.build.vp.sel_invert.click() if self.build else None); self.clear_button.clicked.connect(lambda:self.build.clear_selection() if self.build else None)
        self.clear_button.setToolTip("Clears the selection. The model is not changed.")
        select_row.addWidget(self.invert_button); select_row.addWidget(self.clear_button); layout.addLayout(select_row)
        grow_row=QHBoxLayout(); self.topology_tools = []
        for label,op,icon_name in (("Grow","grow","select-through"),("Shrink","shrink","clear-selection"),("Connected piece","connected","magic")):
            button=action(label,icon_name); button.clicked.connect(lambda checked=False,o=op:self.topology(o))
            button.setToolTip("Adds one ring of joined faces. It can cross a thin bridge into the main part." if op=="grow" else "Selects the entire joined piece, including thin bridges." if op=="connected" else "Removes one boundary ring from the selection.")
            self.tools.append(button)
            self.topology_tools.append(button)
            if op=="connected": layout.addLayout(grow_row); layout.addWidget(button)
            else:grow_row.addWidget(button)
        self.sections.append(section(3, "Change the model"))
        undo_row=QHBoxLayout(); undo_row.setContentsMargins(0, 0, 0, 0); self.undo_button=action("Undo","undo"); self.redo_button=action("Redo","redo")
        self.undo_button.clicked.connect(lambda:self.build.undo(-1) if self.build else None); self.redo_button.clicked.connect(lambda:self.build.undo(1) if self.build else None)
        undo_row.addWidget(self.undo_button); undo_row.addWidget(self.redo_button)
        self.undo_row_widget=QWidget(content); self.undo_row_widget.setLayout(undo_row); layout.addWidget(self.undo_row_widget)   # 2026-10-25: in the palette, under Delete, where the change happened
        self.mesh_tools = []; self.points_ok = []
        spots = narrow(QLabel("Click spots on the model (the right panel opens with the steps):")); spots.setFont(ui_font(11)); spots.setProperty("role", "hint"); layout.addWidget(spots)
        grid2 = QGridLayout(); grid2.setSpacing(6)
        for i, (label, tab, tip, ic) in enumerate((
                ("Cut shape", "Cut shape", "Click a spot: a cylinder, box or sphere there is cut away. Arrows move it. Preview before Apply.", "box"),
                ("Add fin", "Add fin", "Click the fin's two ends, then a third spot for which way it faces. Preview before Apply.", None),
                ("Cut gap", "Cut gap", "Click the two ends of a slot to cut through an unwanted join. Preview before Apply.", None),
                ("Fill gap", "Smooth area", "Click beside a gap: a guess that continues the surface fills it (finished model only). Preview before Apply.", None),
                ("Smooth area", "Smooth area", "Click a spot: a round or square area around it is smoothed, fading at its edge. Preview before Apply.", None),
                ("Clone", "Clone", "Click the spot to copy from, then the spot to paste onto. Preview before Apply.", None),
                ("Mirror", "Mirror", "A symmetric part: click spots on the symmetry plane, the other side is filled from a mirrored copy. Preview before Apply.", None))):
            button = action(label, ic); button.setToolTip(tip); button.clicked.connect(lambda checked=False, n=tab, lb=label: self.open_tool(n, "fill" if lb == "Fill gap" else "smooth" if lb == "Smooth area" else None))
            grid2.addWidget(button, i // 2, i % 2); self.tools.append(button)
            if label in ("Clone", "Smooth area", "Mirror"): self.points_ok.append(button)             # these work on the fused points as well as the finished mesh
            else: self.mesh_tools.append(button)
        layout.addLayout(grid2)
        whole = narrow(QLabel("Whole model, no clicking:")); whole.setFont(ui_font(11)); whole.setProperty("role", "hint"); layout.addWidget(whole)
        grid3 = QGridLayout(); grid3.setSpacing(6)
        for i, (label, tab, tip) in enumerate((("Fill holes", "Hole Filling", "Close the holes in the model. Blue outlines the ones it closes. Preview before Apply."),
                                               ("Slivers", "Remove slivers", "Remove long, thin, spiky triangles."),
                                               ("Simplify", "Simplify", "Fewer triangles, smaller file, same shape."))):
            button = action(label); button.setToolTip(tip); button.clicked.connect(lambda checked=False, n=tab: self.open_tool(n))
            grid3.addWidget(button, i // 2, i % 2); self.tools.append(button); self.mesh_tools.append(button)
        layout.addLayout(grid3)
        self.tool_grids = [grid, grid2, grid3]
        layout.addStretch()
        self.palette.setWidget(content); self.palette.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        try:
            from app2.qt.widgets import quiet_scrollbar
            quiet_scrollbar(self.palette.verticalScrollBar())
        except Exception: pass
        self.timer = QTimer(self); self.timer.setInterval(150); self.timer.timeout.connect(self.sync)

    def _fit_carry(self):
        width = max(40, self.carry.width())
        height = self.carry.fontMetrics().boundingRect(QRect(0, 0, width, 10000), Qt.TextFlag.TextWordWrap, self.carry.text()).height()
        self.carry.setFixedHeight(max(self.carry.fontMetrics().height(), height) + 2)

    def _fit_palette(self):
        compact = self.window().height() < 900
        for button in self.action_buttons:
            button.setFixedHeight(28 if compact else 30)
        self.delete_button.setFixedHeight(32 if compact else 34)
        self.tool_layout.setSpacing(4 if compact else 6)
        for grid in self.tool_grids:
            grid.setSpacing(4 if compact else 6)
        self._fit_carry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "tool_grids"):
            self._fit_palette()

    @staticmethod
    def _number(layout, text, low, high, value):
        """A labelled spin box in its own widget (so a tool can show or hide its setting). Returns (spin box, row widget)."""
        holder = QWidget(); row = QHBoxLayout(holder); row.setContentsMargins(0, 0, 0, 0); label = QLabel(text); label.setFont(ui_font(12)); label.setWordWrap(True); label.setMinimumWidth(0); label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed); number = QSpinBox(); number.setRange(low, high); number.setValue(value)
        number.setFixedSize(70,30); row.addWidget(label,1); row.addWidget(number); layout.addWidget(holder); return number, holder

    def attach(self, build):
        """Move the one existing Build page here. Shell restores it after detach()."""
        if self.build is build: return
        self.build = build; self.box.addWidget(build)
        if hasattr(build, "vp"): build.vp.set_edit_workspace(True)
        if hasattr(build, "right"):
            self.set_tools_expanded(False)
        build.left.layout().insertWidget(1, self.palette, 1)
        if hasattr(build, "rail_scroll"): build.rail_scroll.hide()      # the whole rail box, or its empty box shares the column with the tools
        build.stages.hide(); self.palette.show()
        # Keep the result selected in Build: side, fused and meshed all support
        # selection edits, and switching to Meshed silently edits the wrong one.
        if getattr(build, 'stage', None) not in ("side", "fused", "meshed"):
            build.show_stage("meshed",keep_published=False)
        if getattr(build,"published",None):
            build.published=None;build.request_view()
        self.title.setText({"side": "Side surface", "fused": "Fused surface", "meshed": "Finished model"}.get(getattr(build, 'stage', None), "Edit model"))
        self.set_mode(self.mode.currentText()); self._view_setting("brush_px", self.brush.value()); self._view_setting("magic_angle", self.angle.value())
        build.on_pending_edits = self._pending
        self._pending(getattr(build, "pending_edits", 0))
        self.timer.start(); self.sync(); build.show()
        if build.gl and getattr(build.vp.view, "tools_on", False) and build.vp.view.tool == "orbit": build.set_tool("piece")   # 2026-10-25: Edit opens ready to click pieces

    def detach(self):
        """Return the shared page for the shell to put back in the Build slot."""
        build = self.build
        if build is None: return None
        self.timer.stop(); build.on_pending_edits = None
        try: build.flush_pending_edits()                           # the deletes made here get their rebuild on the way out
        except Exception: pass
        if hasattr(build, "vp"): build.vp.set_edit_workspace(False)
        if hasattr(build, "right"):
            build.layout().removeWidget(self.edge_button); self.edge_button.hide(); self.edge_button.setParent(self); build.right.show()
        if hasattr(build, "fin"): build.fin.clear()
        if hasattr(build, "slice_tool"): build.slice_tool.clear()
        if hasattr(build, "drill"): build.drill.clear()
        if getattr(build, "panel", None):
            for stage, panel in build.panel.items():
                if stage in ("side", "fused", "meshed"): panel.show_tab({"side": "Platter", "fused": "Fusion", "meshed": "Mesh"}[stage])
        build.set_tool("orbit"); self._view_setting("selection_mode", "add")
        build.left.layout().removeWidget(self.palette); self.palette.hide(); self.palette.setParent(self)
        if hasattr(build, "rail_scroll"): build.rail_scroll.show()
        build.stages.show(); self.box.removeWidget(build); build.setParent(None)
        self.build = None
        return build

    def set_tools_expanded(self, expanded):
        self.tools_expanded = bool(expanded)
        if self.build is not None and hasattr(self.build, "right"): self.build.right.setVisible(self.tools_expanded)
        self.edge_button.hide()
        self.options_button.setEnabled(self.build is not None)
        self.options_button.setText('Hide' if expanded else 'Options')
        self.edge_button.setText("›" if self.tools_expanded else "‹")
        label = "Hide edit options" if self.tools_expanded else "Show edit options"
        self.edge_button.setToolTip(label); self.edge_button.setAccessibleName(label)
        self.options_button.setToolTip(label);self.options_button.setAccessibleName(label)

    def _pending(self, n):
        self.rebuild_button.setVisible(n > 0); self.rebuild_button.setText("Rebuild (%d delete%s waiting)" % (n, "" if n == 1 else "s"))

    def open_tool(self, name, action=None):
        if self.build is None: return
        if self.build.flush_pending_edits(): self.build.notice("Rebuilding first, then the tool opens.", level="info"); return
        self.set_tools_expanded(True)
        if action is not None and getattr(self.build, "panel", None):         # Smooth area / Fill gap share one tab: the button picks its action
            for st in ("fused", "meshed"):
                body = self.build.panel.get(st) and self.build.panel[st].bodies.get(name)
                c = self.build._ctl(body, param="action") if body is not None else None
                if c is not None and hasattr(c, "box"): c.box.setCurrentIndex(max(0, c.box.findData(action)))
        stage = getattr(self.build, "stage", "meshed"); self.build.panel[stage if stage in ("fused", "meshed") else "meshed"].show_tab(name)
        if hasattr(self.build, "scroll"): self.build.scroll.verticalScrollBar().setValue(0)
        if name in ("Add fin", "Cut gap", "Cut shape"): self.build.on_extra(name, 0)

    def _view_setting(self, name, value):
        if self.build is not None and self.build.gl: setattr(self.build.vp.view, name, value)

    def set_mode(self, text):
        self._view_setting("selection_mode", text.lower())
        for name, button in self.mode_buttons.items(): button.setChecked(name == text)

    def _set_through(self, on):
        if self.build: self.build.vp.through.setChecked(on)
        self.through_button.setChecked(on); self.visible_button.setChecked(not on)

    def set_tool(self, tool):
        if self.build is not None: self.build.set_tool(tool)

    def topology(self, operation):
        if self.build is None or not self.build.gl or self.build.running(): return
        if not self.build.vp.view.topology_selection(operation): self.build.notice("Selection did not change.")

    def pick_stage(self, key):
        """Edit another level: the side you are on, the fused points, or the finished mesh."""
        if self.build is None or self.build.running(): self.sync(); return
        if self.build.flush_pending_edits(): return
        self.build.show_stage(key,keep_published=False);self.sync()

    HOW = {"orbit": "Drag turns the model, the wheel zooms, right-drag pans. Pick a tool below to select faces.",
           "piece": "Click a piece: the whole connected piece turns red. Shift-click adds more, Ctrl-click takes one out. Then Delete selected.",
           "rect": "Drag a box over the faces you want. Shift-drag adds to the selection, Ctrl-drag takes out.",
           "lasso": "Draw a loop around the faces you want. Shift adds to the selection, Ctrl takes out.",
           "brush": "Paint over the faces. Shift adds, Ctrl paints them out. The radius is below.",
           "magic": "Hovering shows in amber what one click would take: faces at a similar angle. Click to take them. The angle is below."}
    CARRY = {"side": "Side %s only. Deletes carry into Fused and Meshed.",
             "fused": "Meshed is built from this, so it follows.",
             "meshed": "Last step: nothing builds from this."}

    def sync(self):
        build = self.build
        stage = getattr(build, 'stage', None) if build else None
        side = (build._side() or {}).get("n") if build is not None and stage == "side" and hasattr(build, "_side") else None
        self.title.setText({"side": "Side surface", "fused": "Fused surface", "meshed": "Finished model"}.get(stage, "Edit model").replace("side  ", "side "))
        for key, b in self.stage_buttons.items(): b.setChecked(key == stage)
        carry=self.CARRY.get(stage,"").replace("%s",str(side or "")) if stage else ""
        snap=getattr(build,"snap",None)
        if stage=="meshed" and snap and (snap.meshed_method=="as_fused" or (snap.meshed_method=="auto" and snap.fused_kind=="mesh")):
            carry="Same as Fused (As scanned). Pick Watertight for a closed surface."
        if build and stage in ("side","fused","meshed") and hasattr(build,"_display_path") and not build._display_path():
            carry="%s is not built yet. Build this stage to edit it." % stage.capitalize()
        self.carry.setText(carry);self._fit_carry()
        ready = bool(build and build.gl and not build.running() and build.vp.view.tools_on and stage in ("side", "fused", "meshed"))
        for control in self.tools + [self.mode, self.brush, self.angle, self.through_button, self.visible_button, self.invert_button] + list(self.mode_buttons.values()): control.setEnabled(ready)
        for control in self.mesh_tools: control.setEnabled(ready and stage == 'meshed')
        for control in self.points_ok: control.setEnabled(ready and stage in ('fused', 'meshed'))
        if build and getattr(build, 'shown', {}).get('kind') == 'points':
            for control in self.topology_tools: control.setEnabled(False)
        if build and hasattr(getattr(build,"vp",None),"through"):
            through=build.vp.through.isChecked(); self.through_button.setChecked(through); self.visible_button.setChecked(not through)
        self.set_mode(self.mode.currentText())
        selection = getattr(build.vp.view, "selection", None) if build and build.gl else None
        count = int(selection.sum()) if selection is not None else 0
        self.delete_button.setEnabled(ready and count > 0); self.clear_button.setEnabled(ready and count > 0)
        armed = ready and count > 0                                  # 2026-10-25: red when there is something to delete
        if armed != getattr(self, "_delete_armed", None):
            self._delete_armed = armed
            self.delete_button.setStyleSheet("QPushButton {padding: 3px 2px; background: #7a2a2a; border: 1px solid #d9534f; color: white;} QPushButton:hover {background: #943333;}" if armed
                                             else "QPushButton {padding: 3px 2px;} QPushButton:checked {background: #274a78; border: 1px solid #639feb; color: white; font-weight: normal;}")
        kind = 'points' if build and getattr(build, 'shown', {}).get('kind') == 'points' else 'faces'
        self.reason.setText(("%s %s selected" % (format(count, ","), kind) if count else "Red marks the selection.") if ready else "Wait for the build to finish." if build and getattr(build, "running", lambda:False)() else "Build the result first.")
        for target, original in ((self.undo_button, "undo_b"), (self.redo_button, "redo_b")):
            source = getattr(build, original, None)
            target.setEnabled(bool(source and source.isEnabled()))
            if source is not None: target.setToolTip(source.toolTip())
        active = build.vp.view.tool if build and build.gl else "orbit"
        for tool, button in self.tool_buttons.items(): button.setChecked(tool == active)
        how = self.HOW.get(active, "") if ready else ""
        pieces = build.shown_pieces() if (ready and hasattr(build, "shown_pieces")) else None
        if pieces and pieces > 50: how += "  This model is in %s pieces: Loose pieces under Meshed takes all the small ones off at once." % format(pieces, ",")   # 2026-10-25: 9,266 table specks, one click each is hopeless
        self.how.setText(how)
        self.brush_row.setVisible(active == "brush"); self.angle_row.setVisible(active == "magic")
        for control in (self.invert_button,) + tuple(self.topology_tools): control.setEnabled(control.isEnabled() and count > 0)   # nothing to grow, shrink, invert or follow without a selection
