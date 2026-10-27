# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compact profile cards with an expandable advanced build recipe sheet."""
import os
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QSizePolicy, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QWidget, QStackedWidget, QScrollArea, QPushButton
from app2.qt.theme import ui_font
from app2.qt.widgets import button, StableLabel, quiet_scrollbar
from app2.qt import build_words as W
from app2.qt.tool_panel import big_button
from app2.builder import preset_store as store
from app2.qt.help_text import HelpButton

WIDTH = 760
HEIGHT = 720
HINTS = {
    'today': 'Keeps fine detail and true depth: raised text and small recesses stay. Gets noisier when many sides overlap.',
    'revo': 'Merges overlapping points and builds a new, very smooth surface. Fine text and sharp edges soften. Large file.',
    'revo-device': 'Like Detail, with more loose pieces removed and more smoothing. The largest file.',
    'plus': 'The closest match to the scanner\'s own model, and the smoothest result on scans with many sides. Takes longer.',
    'quick': 'Coarse detail and fewer frames for a quick check while scanning. Can leave holes.',
    'print': 'A closed, smooth model of about 250,000 triangles (about 5 MB), ready for the slicer. Fine text softens.',
    'auto': 'A balanced starting point. Chooses the mesh method to match the build method.',
    'custom': 'Choose your settings. Save a name to use them on another scan.',
}

# What each named preset does, in the Custom sheet's own fields: shown on its card, and Custom starts from it.
def preset_facts(name):
    from stages import params as PR
    from stages.methods import get_method
    from stages.graph import op_list
    recipe = PR.apply_preset({}, 'today' if name == 'auto' else name)
    method = PR.method_of(recipe, 'fused')
    fused = PR.resolve(recipe, 'fused', method)
    ops = op_list(recipe, 'fused.' + method, get_method('fused', method))
    cleanup = next((float(o['params']['rate']) for o in ops if o['tool'] == 'isolation' and o.get('on', True)), 0)
    smooth = next((int(o['params'].get('times', 1)) for o in ops if o['tool'] == 'smooth' and o.get('on', True)), 0)
    finish = any(o['tool'] == 'finish' and o.get('on', True) for o in recipe.get('ops', {}).get('meshed', []))
    return dict(method=method, detail=float(fused['point_distance']), cleanup=cleanup, smooth=smooth, finish=finish)


from stages import params as _PR
FACTS = {name: preset_facts(name) for name in [*_PR.PRESETS, 'auto']}
METHOD_NAMES = {'tsdf': 'Standard', 'points': 'Points', 'plus': 'Plus'}


def facts_line(f):
    text = '%s, %g mm, cleanup %g%%' % (METHOD_NAMES[f['method']], f['detail'], f['cleanup'])
    if f['smooth']: text += ', smooth x%d' % f['smooth']
    if f['finish']: text += ', print-ready'
    return text


def preset_choices():
    from stages import params as PR
    out=[(k,label,HINTS.get(k,'Build with these settings.')) for k,label,hint in W.ONETAP['presets'] if k in PR.PRESETS]
    return [('auto','Auto Yoink',HINTS['auto'])]+out+[('custom','Custom',HINTS['custom'])]


def tidy(text):
    return str(text).replace('(mm)',' mm').replace(' : ',': ').replace('  ',' ')


def plain(text):
    text=tidy(text)
    for before,after in [('TSDF','Standard'),('tsdf','Standard'),('Poisson','Watertight'),('As fused','As scanned'),('Isolation','Loose pieces'),('Overlap merge','Merge overlaps'),('Smooth','Smooth surface'),('none','No extra changes')]: text=text.replace(before,after)
    return text


class OneTapDialog(QFrame):
    preset_changed=Signal(str)
    ok=Signal(str)
    custom_changed=Signal(object)
    custom_ok=Signal(object)
    cancel=Signal()
    size_changed=Signal()

    def __init__(self,parent=None):
        super().__init__(parent); self.setObjectName('KeysSheet'); self.current=None; self._loading=False; self.settings_set=False
        self.choices=preset_choices(); self.saved={}; self.load_error=''
        try: self.saved=store.load_presets()
        except (ValueError,OSError) as exc: self.load_error='Saved presets could not be read.'
        self.choices += [('saved:'+name,name,'Your saved build settings.') for name in self.saved]
        v=QVBoxLayout(self); v.setContentsMargins(26,20,26,16); v.setSpacing(8)
        title=QLabel('YOLO Yoink'); title.setFont(ui_font(20,QFont.Weight.DemiBold)); title.setFixedHeight(34); title.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(title)
        tabs=QHBoxLayout(); tabs.setSpacing(6)
        self.simple_b=button('Simple','seg'); self.advanced_b=button('Advanced','seg')
        for tab in (self.simple_b,self.advanced_b):
            tab.setCheckable(True); tab.setFixedHeight(32); tabs.addWidget(tab)
        tabs.addStretch(1); v.addLayout(tabs)
        self.selector=QComboBox(); self.selector.setMaxVisibleItems(8); self.selector.setFixedHeight(36)
        for key,label,hint in self.choices: self.selector.addItem(label,key)
        self.btns={key:self.selector for key,label,hint in self.choices}
        self.selector.currentIndexChanged.connect(lambda i:self.pick(self.selector.itemData(i),True))
        self.preset_row_widget=QWidget(); preset_row=QHBoxLayout(self.preset_row_widget); preset_row.setContentsMargins(0,0,0,0); preset_row.addWidget(self.selector,1)
        self.preset_help=HelpButton('YOLO Yoink'); preset_row.addWidget(self.preset_help); v.addWidget(self.preset_row_widget)
        self.hint=StableLabel('',2); self.hint.setFixedHeight(38); v.addWidget(self.hint)
        # the picked recipe, big: the same clip and lettering built with it (a demo part, the same for everyone)
        self.preview=QWidget(); pv=QHBoxLayout(self.preview); pv.setContentsMargins(0,0,0,4); pv.setSpacing(12)
        self.preview_pic=QLabel(); self.preview_pic.setFixedSize(260,180); self.preview_pic.setStyleSheet('border:1px solid #414b58; border-radius:6px;')
        self.preview_text=QLabel(''); self.preview_text.setWordWrap(True); self.preview_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        pv.addWidget(self.preview_pic); pv.addWidget(self.preview_text,1); v.addWidget(self.preview)
        self.pages=QStackedWidget(); self.pages.setMinimumHeight(0); v.addWidget(self.pages,1)
        self.simple_page=QScrollArea(); self.simple_page.setWidgetResizable(True)
        self.simple_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.simple_page.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cards=QWidget(); self.card_grid=QGridLayout(cards); self.card_grid.setContentsMargins(0,4,0,4); self.card_grid.setSpacing(8)
        self.profile_cards={}
        subtitles={'auto':'Let the app choose (Classic)', 'custom':'Make it your own', **W.RECIPE['card_sub']}
        for key,label,hint in self.choices:
            facts=facts_line(preset_facts(key)) if key in FACTS else ('Start from any preset, then edit' if key=='custom' else (facts_line(dict(store.DEFAULTS, **self.saved[key[6:]])) if key.startswith('saved:') and key[6:] in self.saved else ''))
            fact=W.RECIPE['card_facts'].get('today' if key=='auto' else key,'')
            card=QPushButton(label+'\n'+subtitles.get(key,'Your saved settings')+('\n'+fact if fact else '')); card.setCheckable(True); card.setFixedHeight(96 if fact else 60)
            pic=os.path.join(os.path.dirname(__file__),'assets','presets','%s.png' % ('today' if key=='auto' else key))
            if os.path.exists(pic):           # the same clip and recess built with this recipe: what it keeps, what it smooths
                from PySide6.QtGui import QIcon
                from PySide6.QtCore import QSize
                card.setIcon(QIcon(pic)); card.setIconSize(QSize(100, 69))
            card.setMinimumWidth(0); card.setToolTip(hint+('\n'+facts if facts else ''))       # the numbers on hover (and in Advanced): the card says what it is for
            card.setStyleSheet('QPushButton {text-align:left;padding:6px 12px;border:1px solid #414b58;border-radius:7px;background:#282e36;} QPushButton:checked {background:#274a78;border-color:#639feb;} QPushButton:hover {border-color:#639feb;}')
            card.clicked.connect(lambda checked=False,k=key:self.pick(k,True))
            index=len(self.profile_cards);self.card_grid.addWidget(card,index//2,index%2);self.profile_cards[key]=card
        self.card_grid.setRowStretch((len(self.profile_cards)+1)//2,1)
        self.card_grid.setColumnStretch(0,1); self.card_grid.setColumnStretch(1,1)       # two equal columns, whatever the text
        for c in self.profile_cards.values(): c.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.simple_page.setWidget(cards); self.pages.addWidget(self.simple_page)
        self.advanced_page=QScrollArea(); self.advanced_page.setWidgetResizable(True)
        self.advanced_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.advanced_page.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        quiet_scrollbar(self.advanced_page.verticalScrollBar())
        advanced=QWidget(); av=QVBoxLayout(advanced); av.setContentsMargins(0,4,8,0); av.setSpacing(8)
        self.advanced_page.setWidget(advanced); self.pages.addWidget(self.advanced_page)
        self.simple_b.clicked.connect(lambda:self.set_mode(False)); self.advanced_b.clicked.connect(lambda:self.set_mode(True))
        self.controls=QFrame(); g=QGridLayout(self.controls); g.setContentsMargins(0,0,0,0); g.setSpacing(8)
        self.method=QComboBox()
        for label,key in [('Standard','tsdf'),('Points','points'),('Plus','plus')]: self.method.addItem(label,key)
        self.detail=QDoubleSpinBox(); self.detail.setRange(.1,2); self.detail.setSingleStep(.05); self.detail.setDecimals(2); self.detail.setSuffix(' mm'); self.detail.setValue(.4)
        self.cleanup=QDoubleSpinBox(); self.cleanup.setRange(0,100); self.cleanup.setSingleStep(.5); self.cleanup.setSuffix(' %'); self.cleanup.setValue(3)
        self.smooth=QSpinBox(); self.smooth.setRange(0,20)
        for row,(text,control) in enumerate([('Build method',self.method),('Detail size',self.detail),('Remove loose pieces',self.cleanup),('Smooth passes',self.smooth)]):
            control.setFixedHeight(30); g.addWidget(QLabel(text),row//2,(row%2)*2); g.addWidget(control,row//2,(row%2)*2+1)
        self.finish=QCheckBox('Finish for print after building'); self.finish.setFixedHeight(30); g.addWidget(self.finish,2,0,1,4)
        self.finish.setStyleSheet('QCheckBox {spacing:8px;} QCheckBox::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;background:#151a21;} QCheckBox::indicator:checked {background:#5b9df0;border:2px solid #dceaff;}')
        self.controls.setFixedHeight(106); av.addWidget(self.controls)
        helptext=QLabel('Custom settings: smaller detail takes longer. Zero cleanup keeps loose pieces.'); helptext.setFixedHeight(24)
        help_row=QHBoxLayout(); help_row.addWidget(helptext,1)
        self.custom_help=HelpButton('Custom settings', 'Cleanup removes pieces below this share of the largest piece. Smooth passes soften noise. Save preset keeps these settings for another scan.')
        help_row.addWidget(self.custom_help); av.addLayout(help_row)
        save=QHBoxLayout(); self.name=QLineEdit(); self.name.setPlaceholderText('Name these settings'); self.name.setMaxLength(60); self.name.setFixedHeight(32)
        self.save_b=button('Save preset','quiet'); self.save_b.setFixedSize(116,32); self.save_b.clicked.connect(self.save_current)
        save.addWidget(self.name,1); save.addWidget(self.save_b); av.addLayout(save)
        self.save_note=StableLabel('',1); self.save_note.setFixedHeight(24); av.addWidget(self.save_note)
        grid=QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(4); self.lines=[]; self.keys=[]
        for row,text in enumerate(['Build detail','Cleanup','Surface','Final changes']):
            label=QLabel(text); label.setFixedHeight(30); line=StableLabel('',2); line.setFixedHeight(30)
            grid.addWidget(label,row,0); grid.addWidget(line,row,1); self.lines.append(line); self.keys.append(label)
        grid.setColumnStretch(1,1); av.addLayout(grid); av.addStretch(1)
        self.foot=StableLabel('',2); self.foot.setFixedHeight(38); v.addWidget(self.foot)
        row=QHBoxLayout(); self.cancel_b=big_button('Cancel','quiet'); self.ok_b=big_button('Yoink it','build')
        self.cancel_b.setFixedHeight(44); self.ok_b.setFixedHeight(44); row.addWidget(self.cancel_b); row.addWidget(self.ok_b); v.addLayout(row)
        self.cancel_b.clicked.connect(self.cancel.emit); self.ok_b.clicked.connect(self.run)
        self.method.currentIndexChanged.connect(self.changed)
        for c in (self.detail,self.cleanup,self.smooth): c.valueChanged.connect(self.changed)
        self.finish.toggled.connect(self.changed)
        self.set_mode(False); self.fix_size(); self.hide()

    def set_mode(self, advanced):
        self.pages.setCurrentIndex(1 if advanced else 0)
        self.simple_b.setChecked(not advanced); self.advanced_b.setChecked(advanced)
        self.preset_row_widget.setVisible(advanced)
        self.hint.setVisible(advanced)
        self.fix_size(); self._refresh_hint()
        if self.parentWidget() is not None:
            parent=self.parentWidget(); self.move((parent.width()-self.width())//2,max(8,(parent.height()-self.height())//2))
        self.size_changed.emit()

    def _refresh_hint(self):
        hint=next((hint for k,label,hint in self.choices if k==self.current),'')
        if self.pages.currentIndex()==1 and self.current not in ('auto','custom') and not str(self.current).startswith('saved:'):
            hint+=' Choose Custom to edit settings.'
        self.hint.setText(hint)

    def fix_size(self):
        width,height=WIDTH,HEIGHT                          # Simple and Advanced share one size: no jump when switching
        parent=self.parentWidget()
        if parent is not None: width=min(width,max(320,parent.width()-24));height=min(height,max(360,parent.height()-24))
        compact = parent is not None and parent.window().height() <= 900
        self.preview_pic.setFixedSize(136, 84) if compact else self.preview_pic.setFixedSize(260, 180)
        self.preview_text.setMaximumHeight(84 if compact else 16777215)
        self.preview_text.setToolTip(HINTS.get('today' if self.current == 'auto' else self.current, ''))
        self.setFixedSize(width,height)
        if self.current: self._refresh_preview(self.current)

    def settings(self):
        return dict(method=self.method.currentData(),detail=self.detail.value(),cleanup=self.cleanup.value(),smooth=self.smooth.value(),finish=self.finish.isChecked())

    def payload(self): return {'preset':self.current,'fragment':store.recipe_fragment(self.settings())}

    def _refresh_preview(self,key):
        name='today' if key=='auto' else key
        pic=os.path.join(os.path.dirname(__file__),'assets','presets','%s.png' % name)
        if not key or not os.path.exists(pic): self.preview.hide(); return
        from PySide6.QtGui import QPixmap
        self.preview_pic.setPixmap(QPixmap(pic).scaled(self.preview_pic.size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation))
        label=next((l for k,l,_h in self.choices if k==key),key)
        self.preview_text.setText('<b>%s</b><br>%s<br><br><span style="color:#8b95a1">%s<br>Example: a clip and raised lettering on a test part, built with this recipe. Ultra Yoink builds your own scan every way to compare.</span>'
                                  % (label, HINTS.get(name,''), W.RECIPE['card_facts'].get(name,'')))
        if self.parentWidget() and self.parentWidget().window().height() <= 900:
            self.preview_text.setText('<b>%s</b><br>%s' % (label, HINTS.get(name, '')))
        self.preview.show()

    def pick(self,key,ask=False):
        self._loading=True; self.current=key
        for name,card in self.profile_cards.items(): card.setChecked(name==key)
        if key in self.profile_cards: QTimer.singleShot(0,lambda c=self.profile_cards[key]:self.simple_page.ensureWidgetVisible(c,0,8))   # the picked card in sight
        self.selector.blockSignals(True); self.selector.setCurrentIndex(self.selector.findData(key)); self.selector.blockSignals(False)
        editable=key=='custom' or bool(key and key.startswith('saved:'))
        if key and key.startswith('saved:'): settings=self.saved.get(key[6:],store.DEFAULTS)
        elif key in FACTS: settings=preset_facts(key)
        else: settings=None if self.settings_set else store.DEFAULTS      # Custom keeps what the last preset showed
        if settings is not None:
            self.settings_set=True
            self.method.setCurrentIndex(self.method.findData(settings['method'])); self.detail.setValue(settings['detail']); self.cleanup.setValue(settings['cleanup']); self.smooth.setValue(settings['smooth']); self.finish.setChecked(settings['finish'])
        self.controls.setEnabled(editable); self.name.setEnabled(editable); self.save_b.setEnabled(editable)
        if ask and key=='custom': self.set_mode(True)
        self._refresh_preview(key)
        self._refresh_hint()
        self.save_note.setText(self.load_error); self._loading=False
        self._pending()
        if ask:
            if key=='auto' or editable:self.custom_changed.emit(self.payload())
            else:self.preset_changed.emit(key)

    def _pending(self):
        for line in self.lines:line.setText('Checking settings...')
        self.foot.setText(''); self.ok_b.setEnabled(False)

    def changed(self,*args):
        if self._loading:return
        if self.current!='custom':
            self.current='custom'; self.selector.blockSignals(True); self.selector.setCurrentIndex(self.selector.findData('custom')); self.selector.blockSignals(False)
        self._pending(); self.custom_changed.emit(self.payload())

    def save_current(self):
        try:
            name=store.save_preset(self.name.text(),self.settings()); self.saved[name]=self.settings(); key='saved:'+name
            if key not in self.btns:
                self.choices.append((key,name,'Your saved build settings.')); self.selector.addItem(name,key); self.btns[key]=self.selector
            if key not in self.profile_cards:
                card=QPushButton(name+'\nYour saved settings\n'+facts_line(dict(store.DEFAULTS,**self.settings()))); card.setCheckable(True); card.setFixedHeight(76); card.setMinimumWidth(0)
                card.setStyleSheet(next(iter(self.profile_cards.values())).styleSheet()); card.setToolTip(name)
                card.clicked.connect(lambda checked=False,k=key:self.pick(k,True))
                index=len(self.profile_cards); self.card_grid.setRowStretch(index//2,0)
                self.card_grid.addWidget(card,index//2,index%2); self.profile_cards[key]=card
                self.card_grid.setRowStretch((len(self.profile_cards)+1)//2,1)
            self.save_note.setText('Saved '+name)
        except (ValueError,OSError) as exc:self.save_note.setText(str(exc))

    def run(self):
        if not self.current:return                         # nothing picked yet (the sheet opens on a preset)
        if self.current=='auto' or self.current=='custom' or self.current.startswith('saved:'):self.custom_ok.emit(self.payload())
        else:self.ok.emit(self.current)

    def show_summary(self,summary):
        if summary.get('preset')!=self.current:return
        for line,key in zip(self.lines,['fusion','ftools','mesh','mtools']):line.setText(plain(summary.get(key,'')))
        from app2.builder.stagehost import mins
        kept=int(summary.get('deletes') or 0)
        text='Keeps %d manual deletes. '%kept if kept else ''
        text += 'Already built: nothing to run.' if not summary.get('to_run') else 'About %s.'%mins(summary.get('est_s'))
        self.foot.setText(text); self.ok_b.setEnabled(True)

    def show_error(self,text):
        for line in self.lines:line.setText('')
        self.lines[0].setText(text); self.foot.setText(text); self.ok_b.setEnabled(False)
