# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pass controls under the camera views (light, IR gain, Range, 3D color)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from app2.qt.choices import Choices


class LiveStrip(QFrame):
    def __init__(self, page):
        super().__init__()
        self.setObjectName('LiveStrip')
        self.setProperty('role', 'panel')
        self.setAccessibleName('Live scan controls')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        c = page.controls
        line = self.fontMetrics().lineSpacing()
        layout.setContentsMargins(line//2,line//3,line//2,line//3)
        layout.setSpacing(line//3)
        from app2.qt.theme import T
        self.setStyleSheet('QFrame#LiveStrip { background: %s; border: 1px solid %s; border-radius: %dpx; }' % (T['panel'],T['line'],line//2))
        for widget in (c.gain_row, c.range_row):
            widget.g.setHorizontalSpacing(line//3)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            layout.addWidget(widget)
        # This is the original model switch, so pending distance evidence and
        # live-build coverage use the same state as the model renderer.
        page.model.external_distance_mode = True
        self.page = page; self.color_group = self.group('3D view', page.model.distance_mode)
        layout.addWidget(self.color_group)
        page.model.distance_mode.show()
        page.model.distance_mode.grid.setSpacing(line // 2)   # room between Distance, Coverage and Plain
        # The IR camera picker lives in the IR tile's own header, beside the picture it changes; this hidden twin
        # only keeps the I shortcut and older callers in step.
        self.ir = Choices(self); self.ir.hide()
        self.ir.addItems(('Left', 'Right', 'Both'))
        self.ir.setAccessibleName('IR camera in live controls')
        self.ir.currentTextChanged.connect(page.camera_prefs.set_ir)
        page.camera_prefs.changed.connect(lambda: self.sync_ir(page.camera_prefs.ir))
        self.sync_ir(page.camera_prefs.ir)
        self.tall = (c.led, c.auto_gain, c.auto_range, *page.model.distance_mode.buttons, *self.style.buttons)
        self.compact = None; self.set_compact(False)
        self.setToolTip('Controls for the current pass. Setup stays in Settings.')
        self.refresh_tips(page)

    def set_compact(self, compact):
        """Short windows: the camera pictures need the height, so buttons drop to their natural size."""
        if compact == self.compact: return
        self.compact = compact
        if getattr(self.page, 'guide', None) is not None:
            from app2.qt import ui_level
            ui_level.place_guide(self.page)   # a short window has no room under the cameras: the guide goes to Sides
        line = self.fontMetrics().lineSpacing()
        height=max(max(control.sizeHint().height() for control in self.tall),line if compact else line*2)
        for control in self.tall:control.setFixedHeight(height)
        self.page.model.distance_mode.setMinimumHeight(0 if compact else height)   # its row as tall as its buttons: the bottom edge was cut off
        for track in (self.page.controls.gain,self.page.controls.range):track.set_compact(compact)
        for row in (self.page.controls.gain_row,self.page.controls.range_row):
            row.name.setFixedHeight(height);row.value.setFixedHeight(height)
            row.g.setVerticalSpacing(0 if compact else line//4)
        # Light goes under + New side (that row is too narrow for three) and 3D color returns beside Home view, so
        # the strip keeps the two sliders.
        page, layout = self.page, self.layout()
        led, model, mode = page.controls.led, page.model, page.model.distance_mode
        if compact:
            model.external_distance_mode = False; mode.setParent(model)
        else:
            model.external_distance_mode = True; self.color_group.layout().addWidget(mode)
        mode.setVisible(not compact)   # in the 3D panel it shows once there is a model to color
        model._refresh_overlay()
        self.place_light()

    def place_light(self):
        """Light sits beside 3D view color; under + New side in short windows (that row is too narrow for three);
        on its own row in Simple, where 3D view color is hidden."""
        page, layout, led = self.page, self.layout(), self.page.controls.led
        advanced = getattr(page, 'advanced', True)
        if self.compact:
            side = page.sides.layout()
            row = next(i for i in range(side.count()) if side.itemAt(i).layout() is page.sides.action_row)
            side.insertWidget(row + 1, led)
        elif advanced: self.color_header.addWidget(led)
        else: layout.addWidget(led, 0, Qt.AlignmentFlag.AlignRight)
        led.show(); self.color_group.setVisible(not self.compact and advanced)
        layout.invalidate()

    def group(self, title, widget):
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self.fontMetrics().lineSpacing()//4)
        self.color_header=QHBoxLayout();self.color_header.setContentsMargins(0,0,0,0)
        label=self.color_label=QLabel(title);label.setFont(self.page.controls.gain_row.name.font())
        self.color_header.addWidget(label);self.color_header.addStretch(1)
        # dot size beside the title while Dots is picked (it lived in Show marks, where the user could not find it)
        self.dot_size = Choices(self, compact=True); self.dot_size.addItems(('Small', 'Normal', 'Large'))
        for b, text in zip(self.dot_size.buttons, ('S', 'M', 'L')): b.setText(text)
        self.dot_size.setToolTip('Dot size in the 3D view: small, medium or large.'); self.dot_size.setAccessibleName('3D view dot size')
        prefs = self.page.camera_prefs
        self.dot_size.setCurrentIndex(('Small', 'Normal', 'Large').index(prefs.show.get('dot_size', 'Normal')) if prefs.show.get('dot_size') in ('Small', 'Normal', 'Large') else 1)
        self.dot_size.currentIndexChanged.connect(lambda i: prefs.set_option('dot_size', ('Small', 'Normal', 'Large')[i]))
        self.color_header.addWidget(self.dot_size)
        self.color_header.addWidget(self.page.controls.led)
        layout.addLayout(self.color_header)
        # shape first (Surface / Dots), then color (Distance / Coverage / Plain): both are about the live model, and the
        # user hunted for the shape choice when it hid in the header
        self.style = Choices(self); self.style.addItems(('Surface', 'Dots'))
        self.style.setToolTip('How the live model is drawn: a shaded surface, or the dots it is made of.')
        self.style.setAccessibleName('3D view style')
        prefs = self.page.camera_prefs; self.style.setCurrentIndex(1 if prefs.show.get('view_style') == 'Dots' else 0)
        self.page.model.set_view_style(prefs.show.get('view_style', 'Surface'))
        self.style.currentTextChanged.connect(lambda name: (prefs.set_option('view_style', name), self.page.model.set_view_style(name), self.dot_size.setVisible(name == 'Dots')))
        self.dot_size.setVisible(prefs.show.get('view_style') == 'Dots')
        def follow_prefs():                                   # Show marks can set these too: keep the header in step
            names = ('Small', 'Normal', 'Large'); size = prefs.show.get('dot_size', 'Normal')
            if size in names and self.dot_size.currentIndex() != names.index(size):
                self.dot_size.blockSignals(True); self.dot_size.setCurrentIndex(names.index(size)); self.dot_size.blockSignals(False)
        prefs.changed.connect(follow_prefs)
        self.style.grid.setSpacing(self.fontMetrics().lineSpacing() // 2)
        layout.addWidget(self.style)
        layout.addWidget(widget)
        return group

    def sync_ir(self, value):
        self.ir.blockSignals(True); self.ir.setCurrentText(value); self.ir.blockSignals(False)

    def fit_width(self, width):
        # Flush nested grid hints before fixing the strip's height. A pending
        # compact-spacing change otherwise shrank it on the next camera click.
        for index in range(self.layout().count()):
            item=self.layout().itemAt(index)
            widget=item.widget();widget.ensurePolished()
            nested=widget.layout()
            if nested is not None:nested.invalidate();nested.activate()
            widget.updateGeometry();item.invalidate()
        self.layout().invalidate()
        layout=self.layout();margin=layout.contentsMargins()
        inner=max(1,width-margin.left()-margin.right())
        def tall(widget):return widget.heightForWidth(inner) if widget.hasHeightForWidth() else max(widget.sizeHint().height(),widget.minimumHeight())   # a fixed height outranks the hint
        border=self.height()-self.contentsRect().height()   # the panel's own outline takes its share too
        def stack(heights):return sum(heights)+max(0,len(heights)-1)*layout.spacing()+margin.top()+margin.bottom()+border
        guide=self.page.guide
        from app2.scan import first_scan
        guide_h=guide.tallest(inner,first_scan.every_step(),first_scan.TOTAL)   # room for the longest step at this width
        if guide.parentWidget() is self:guide.setFixedHeight(guide_h)   # Simple: shown or not, the rows below stay put
        def kept(widget):return not widget.isHidden() or widget.sizePolicy().retainSizeWhenHidden()
        shown=stack([tall(layout.itemAt(i).widget()) for i in range(layout.count()) if kept(layout.itemAt(i).widget())])
        # Both modes' rows set the height in both modes, so switching Simple and Advanced never moves the cameras
        c=self.page.controls;rows=[tall(c.gain_row),tall(c.range_row)];simple=[guide_h,tall(c.range_row)]
        if not self.compact:
            group=self.color_group.layout();group.invalidate();group.activate()
            rows.append(max(group.minimumSize().height(),max(tall(self.color_label),tall(c.led))+2*group.spacing()+tall(self.style)+tall(self.page.model.distance_mode)))   # header, shape row, color row
            simple.append(tall(c.led))
        # a short window has no room to spare under the cameras: there only the shown rows count
        self.setFixedHeight(max(shown,stack(rows),0 if self.compact else stack(simple)))

    def refresh_tips(self, page):
        c = page.controls
        for widget, key in ((page.go, 'Space'), (page.sides.new, 'N'), (c.led, 'L'), (c.auto_gain, 'G'), (c.auto_range, 'R')):
            text = widget.toolTip().split('\nShortcut:')[0]
            widget.setToolTip(text + '\nShortcut: ' + key)
        c.gain.setToolTip('Camera brightness. [ lower, ] higher, G turns Auto on or off.')
        for i, button in enumerate(page.model.distance_mode.buttons):
            button.setToolTip(('Color by how far the scanner was. Fills in after you pause.', 'Deep blue: well covered. Light blue: getting there. Purple: scan here.', 'One plain color, to check the shape.')[i] + '\nShortcut: V cycles 3D color.')
        for button in self.ir.buttons:
            button.setToolTip('Which depth camera picture to show.\nShortcut: I cycles Left, Right, Both.')
