# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model display choices over the Build and Edit views, one row along the bottom right of the view (2026-10-24):

  Look  [Shaded ⇄ Scan color]  [Flat facets ⇄ Points]   Overlay: None ▾  [Table turns | Scanner moves]   View ▾

Two look buttons, each a pair ("shaded and scan color needs to be one button that switches between the two, same with
flat and points"): the button shows the member in use; clicking it while it is lit flips to the other member, clicking
it while the other button is lit picks the member it shows. Overlay is one drop-down; the What is left sub-pair shows
only while that overlay is on.

The panel still works with one mode key at a time (an overlay replaces the look): picking an overlay emits its key,
None emits the last look picked."""
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QFrame,QToolButton,QCheckBox,QPushButton,QLabel,QMenu
from app2.qt.choices import Choices

LOOKS=('shaded','flat','points','scan_color','sides')
PAIRS=(('shaded','scan_color'),('flat','points','sides'))   # the two look buttons and the members each flips between
LEFT=('scan_guide','scan_guide_handheld')                 # the two What is left modes
LEFT_TEXT={'scan_guide':'Table turns','scan_guide_handheld':'Scanner moves'}
OVERLAY_ORDER=('what_is_left','quality','capture_distance')
NONE='__none__'


class PairButton(QPushButton):
    """One look button: shows the member in use, flips to the other on a click while lit."""
    def __init__(self, parent=None):
        super().__init__(parent);self.setCheckable(True);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus);self.members=[];self.member=None
        self.setStyleSheet('QPushButton {background:#20262e;color:#c6d4e4;border:1px solid #46515e;border-radius:5px;padding:0.18em 0.5em;} QPushButton:checked {background:#234f64;color:#ffffff;border-color:#67c5e8;}')

    def set_members(self, members):
        self.members=list(members)
        if self.member not in [k for k,_ in self.members]:self.show_member(self.members[0][0] if self.members else None)
        else:self.show_member(self.member)

    def show_member(self, key):
        self.member=key;text=dict(self.members).get(key,'')
        other=[t for k,t in self.members if k!=key]
        self.setText(text);self.setAccessibleName(text)
        self.setToolTip(('Click again: %s'%other[0]) if (other and self.isChecked()) else (text or ''))

    def setChecked(self, on):
        super().setChecked(on);self.show_member(self.member)

    def flip(self):
        ks=[k for k,_ in self.members]
        if len(ks)>1:self.show_member(ks[(ks.index(self.member)+1)%len(ks)])


class ViewModes(QWidget):
    def __init__(self, panel):
        super().__init__(panel.area)
        self.panel=panel;self._text='';self.options=();self.current=None;self.last_look='shaded';self._overlay_items=[]
        gap=max(2,self.fontMetrics().lineSpacing()//3)
        self.row=QHBoxLayout(self);self.row.setContentsMargins(0,0,0,0);self.row.setSpacing(gap)
        self.look_label=self._label('Look');self.row.addWidget(self.look_label)
        self.looks=[PairButton(self) for _ in PAIRS]
        for i,b in enumerate(self.looks):b.clicked.connect(lambda _=False,i=i:self._look_clicked(i));self.row.addWidget(b)
        self.row.addSpacing(gap*2)
        self.overlay_label=self._label('Overlay');self.row.addWidget(self.overlay_label)
        self.overlays=QToolButton();self.overlays.setFocusPolicy(Qt.FocusPolicy.StrongFocus);self.overlays.setToolTip('Color the model by what it needs: more scanning, frame support, or capture distance')
        self.overlay_menu=QMenu(self.overlays);self.overlays.clicked.connect(self._open_overlays);self.row.addWidget(self.overlays)
        self.left_kind=Choices(columns=2);self.row.addWidget(self.left_kind);self.left_kind.hide()
        for key in LEFT:self.left_kind.addItem(LEFT_TEXT[key],key)
        self.left_kind.activated.connect(lambda i:self._pick(self.left_kind.itemData(i)))
        self.left_kind.setToolTip('What is left for a part on the turntable, or for a scanner moved by hand')
        self.more=QToolButton(panel.area);self.more.setText('View ▾');self.more.setToolTip('Projection, marked pieces, and Home view settings');self.more.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # floats by Home view, bottom left
        self.popup=QFrame(self,Qt.WindowType.Popup);self.popup.setStyleSheet('QFrame {background:#20262e;color:#e7edf5;border:1px solid #46515e;}')
        form=QVBoxLayout(self.popup)
        self.projection=Choices();self.projection.addItem('Perspective','perspective');self.projection.addItem('Orthographic','orthographic');form.addWidget(self.projection)
        self.projection.activated.connect(lambda i:panel.set_projection_mode(self.projection.itemData(i)))
        self.marked=QCheckBox('Show marked pieces only');form.addWidget(self.marked)
        self.marked.toggled.connect(lambda on:panel.gl and panel.main.view.set_marked_only(on))
        self.home_set=QPushButton('Set this view as Home view');self.home_reset=QPushButton('Reset Home view')
        self.home_set.clicked.connect(panel.home_set_clicked);self.home_reset.clicked.connect(panel.home_reset_clicked)
        form.addWidget(self.home_set);form.addWidget(self.home_reset)
        self.more.setVisible(panel.gl)
        self.more.clicked.connect(self.open_options)
        self._set_overlay_text(NONE)

    def _label(self, text):
        lab=QLabel(text);lab.setProperty('role','hint');lab.setStyleSheet('color:#9aa5b4;');return lab

    # ---- overlay drop-down
    def overlay_keys(self):return [k for k,_ in self._overlay_items]
    def overlay_texts(self):return [t for _,t in self._overlay_items]

    def _set_overlay_text(self, key):
        text=dict(self._overlay_items).get(key,'None')
        self.overlays.setText('%s ▾'%text);self.overlays.setProperty('overlay',key)

    def _open_overlays(self):
        h=self.overlay_menu.sizeHint().height()
        self.overlay_menu.exec(self.overlays.mapToGlobal(QPoint(0,-h-4)))                   # upward: the button sits at the bottom of the view

    def pick_overlay(self, key):
        """As if the drop-down's entry key was chosen (tests, and the menu itself)."""
        self._pick(key)

    def _look_clicked(self, i):
        b=self.looks[i]
        if not b.member:return
        lit=self.current in [k for k,_ in b.members]               # as it was before the click (Qt toggles the button first)
        if lit and len(b.members)>1:b.flip()                        # lit: flip to the pair's other member
        for o in self.looks:o.setChecked(o is b)                   # until set_modes confirms
        self._pick(b.member)

    def _pick(self, key):
        """A button: the mode the panel should show (None -> the last look)."""
        if key==NONE:key=self.last_look if self.last_look in dict(self.options) else 'shaded'
        elif key=='what_is_left':
            picked=self.left_kind.itemData(max(0,self.left_kind.currentIndex()))
            key=picked if picked in dict(self.options) else next((k for k in LEFT if k in dict(self.options)),'scan_guide')
        self.panel.colour_picked.emit(key)

    def open_options(self):
        """The View popup opens upward from the button (it sits at the bottom of the view since 2026-10-24), and stays on screen."""
        self.popup.adjustSize();area=self.more.screen().availableGeometry()
        p=self.more.mapToGlobal(QPoint(0,-self.popup.height()-4))
        if p.y()<area.top():p=self.more.mapToGlobal(QPoint(0,self.more.height()+4))        # no room above: below after all
        p.setX(max(area.left(),min(p.x(),area.right()-self.popup.width()+1)));p.setY(max(area.top(),min(p.y(),area.bottom()-self.popup.height()+1)))
        self.popup.move(p);self.popup.show()

    def set_modes(self, modes, current):
        options=tuple(modes);keys=dict(options)
        if options!=self.options:
            self.options=options
            for b,pair in zip(self.looks,PAIRS):
                b.set_members([(k,keys[k]) for k in pair if k in keys]);b.setVisible(bool(b.members))
            over=[(k,t) for k,t in options if k not in LOOKS]
            self._overlay_items=[];self.overlay_menu.clear()
            if over:
                self._overlay_items.append((NONE,'None'))
                if any(k in LEFT for k,_ in over):self._overlay_items.append(('what_is_left','What is left'))
                for key,text in over:
                    if key not in LEFT:self._overlay_items.append((key,text))
                for key,text in self._overlay_items:
                    a=self.overlay_menu.addAction(text);a.setCheckable(True);a.setData(key)
                    a.triggered.connect(lambda _=False,k=key:self._pick(k))
            self.overlay_label.setVisible(bool(over));self.overlays.setVisible(bool(over))
        self.current=current
        if current in LOOKS:self.last_look=current
        self.left_kind.blockSignals(True)
        look=current if current in LOOKS else (self.last_look if self.last_look in keys else None)
        for b in self.looks:
            if look in [k for k,_ in b.members]:b.show_member(look)
            b.setChecked(current in LOOKS and look in [k for k,_ in b.members])
        over_key='what_is_left' if current in LEFT else (current if (current in keys and current not in LOOKS) else NONE)
        for a in self.overlay_menu.actions():a.setChecked(a.data()==over_key)
        self._set_overlay_text(over_key)
        if current in LEFT:self.left_kind.setCurrentIndex(self.left_kind.findData(current))
        self.left_kind.blockSignals(False)
        self.left_kind.setVisible(current in LEFT)
        self.row.invalidate();self.row.activate()
        self.updateGeometry()

    def fit_width(self, available):
        self.setMaximumWidth(max(1,available))

    def look_buttons(self):return [b for b in self.looks if b.members]

    def setText(self, text):
        self._text=text.replace('▾','').strip();self.setAccessibleName('Model display: '+self._text)

    def text(self):return self._text
