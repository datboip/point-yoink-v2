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
    frames = int(fused.get('frames_per_side') or 1000)
    quality = int(PR.resolve(recipe, 'meshed', 'poisson')['quality'])
    return dict(method=method, detail=float(fused['point_distance']), cleanup=cleanup, smooth=smooth, finish=finish,
                frames=max(50, min(1000, frames)), quality=quality, strength=store.DEFAULTS['strength'])


from stages import params as _PR
FACTS = {name: preset_facts(name) for name in [*_PR.PRESETS, 'auto']}
METHOD_NAMES = {'tsdf': 'Standard', 'points': 'Points', 'plus': 'Plus'}


def facts_line(f):
    text = '%s, %g mm, cleanup %g%%' % (METHOD_NAMES[f['method']], f['detail'], f['cleanup'])
    if f.get('frames', 300) != 300: text += ', %d frames/side' % f['frames']
    if f['smooth']: text += ', smooth x%d' % f['smooth']
    if f['finish']: text += ', print-ready'
    return text


def preset_choices():
    from stages import params as PR
    out=[(k,label,HINTS.get(k,'Build with these settings.')) for k,label,hint in W.ONETAP['presets'] if k in PR.PRESETS]
    return [('auto','Auto Yoink',HINTS['auto'])]+out+[('custom','Custom',HINTS['custom'])]


CARD_STYLE = ('QPushButton {text-align:left;border:1px solid #414b58;border-radius:7px;background:#282e36;} QPushButton:checked {background:#274a78;border-color:#639feb;} '
              'QPushButton:hover {border-color:#639feb;} QPushButton QLabel {background:transparent;color:#e6ebf2;}')


def own_icon():
    """Stand-in picture for Custom and saved cards (no demo part exists for them): three sliders, so the cards line up with the rest."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
    pix = QPixmap(100, 69); pix.fill(QColor('#1d2229'))
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for i, knob in enumerate((.3, .7, .5)):
        y = 18 + i * 17; p.setPen(QPen(QColor('#414b58'), 3)); p.drawLine(14, y, 86, y)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor('#639feb')); p.drawEllipse(QRectF(14 + 72 * knob - 5, y - 5, 10, 10))
    p.end(); return pix


def recipe_card(text, pix):
    """A card button: its picture and a word-wrapped text beside it (a button's own text cannot wrap, so long lines were cut off)."""
    from PySide6.QtGui import QPixmap
    card = QPushButton(); card.setCheckable(True); card.setFixedHeight(96); card.setMinimumWidth(0)
    row = QHBoxLayout(card); row.setContentsMargins(12, 6, 12, 6); row.setSpacing(10)
    card.pic = QLabel(); card.pic.setFixedSize(100, 69); card.pic.setAlignment(Qt.AlignmentFlag.AlignCenter); card.pic.setPixmap(pix)
    card.text = QLabel(text); card.text.setWordWrap(True); card.text.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    card.text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred); card.text.setMinimumWidth(0)
    for w in (card.pic, card.text): w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); row.addWidget(w, 0 if w is card.pic else 1)
    return card


def fit_pix(path_or_pix, w=100, h=69):
    from PySide6.QtGui import QPixmap
    pix = path_or_pix if isinstance(path_or_pix, QPixmap) else QPixmap(path_or_pix)
    return pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


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
    ultra=Signal()
    size_changed=Signal()

    def __init__(self,parent=None):
        super().__init__(parent); self.setObjectName('KeysSheet'); self.current=None; self._loading=False; self.settings_set=False
        self.choices=preset_choices(); self.saved={}; self.load_error=''; self.pictures={}
        try: self.saved=store.load_presets()
        except (ValueError,OSError) as exc: self.load_error='Saved presets could not be read.'
        own=self.choices.pop()                                           # Custom ("make your own") goes last, after the presets and the saved ones
        self.choices += [('saved:'+name,name,'Your saved build settings.') for name in self.saved] + [own]
        v=QVBoxLayout(self); v.setContentsMargins(26,20,26,16); v.setSpacing(8)
        title=QLabel('YOLO Yoink'); title.setFont(ui_font(20,QFont.Weight.DemiBold)); title.setFixedHeight(34); title.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(title)
        tabs=QHBoxLayout(); tabs.setSpacing(6)
        self.simple_b=button('Simple','seg'); self.advanced_b=button('Advanced','seg'); self.compare_b=button('Compare','seg')
        self.compare_b.setToolTip('Your part as built with each recipe so far, side by side')
        for tab in (self.simple_b,self.advanced_b,self.compare_b):
            tab.setCheckable(True); tab.setFixedHeight(32); tabs.addWidget(tab)
        tabs.addStretch(1); v.addLayout(tabs)
        self.selector=QComboBox(); self.selector.setMaxVisibleItems(8); self.selector.setFixedHeight(36)
        for key,label,hint in self.choices:
            if key=='custom': self.selector.insertSeparator(self.selector.count())      # a line above Custom
            self.selector.addItem(label,key)
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
        subtitles={'auto':'Let the app choose (Classic)', 'custom':'Set your own numbers', **W.RECIPE['card_sub']}
        for key,label,hint in self.choices:
            facts=facts_line(preset_facts(key)) if key in FACTS else ('Start from any preset, then edit' if key=='custom' else (facts_line(dict(store.DEFAULTS, **self.saved[key[6:]])) if key.startswith('saved:') and key[6:] in self.saved else ''))
            fact=W.RECIPE['card_facts'].get('today' if key=='auto' else key,'') or facts
            pic=os.path.join(os.path.dirname(__file__),'assets','presets','%s.png' % ('today' if key=='auto' else key))
            card=recipe_card(label+'\n'+subtitles.get(key,'Your saved settings')+('\n'+fact if fact else ''),fit_pix(pic) if os.path.exists(pic) else own_icon())   # the demo part built with this recipe, or three sliders
            card.setToolTip(hint+('\n'+facts if facts else ''))       # the numbers on hover (and in Advanced): the card says what it is for
            card.setStyleSheet(CARD_STYLE)
            card.clicked.connect(lambda checked=False,k=key:self.pick(k,True))
            self.profile_cards[key]=card
        self.card_line=QFrame(); self.card_line.setFrameShape(QFrame.Shape.HLine); self.card_line.setStyleSheet('color:#414b58;')
        self.card_grid.setColumnStretch(0,1); self.card_grid.setColumnStretch(1,1)       # two equal columns, whatever the text
        for c in self.profile_cards.values(): c.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._lay_cards()
        self.simple_page.setWidget(cards); self.pages.addWidget(self.simple_page)
        self.advanced_page=QScrollArea(); self.advanced_page.setWidgetResizable(True)
        self.advanced_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.advanced_page.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        quiet_scrollbar(self.advanced_page.verticalScrollBar())
        advanced=QWidget(); av=QVBoxLayout(advanced); av.setContentsMargins(0,4,8,0); av.setSpacing(8)
        self.advanced_page.setWidget(advanced); self.pages.addWidget(self.advanced_page)
        # Compare: the user's own part, as each recipe built it, side by side (Ultra Yoink fills it with more)
        self.compare_page=QScrollArea(); self.compare_page.setWidgetResizable(True); self.compare_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        quiet_scrollbar(self.compare_page.verticalScrollBar())
        cmp=QWidget(); self.compare_grid=QGridLayout(cmp); self.compare_grid.setContentsMargins(0,4,8,4); self.compare_grid.setSpacing(10)
        self.compare_page.setWidget(cmp); self.pages.addWidget(self.compare_page)
        self.simple_b.clicked.connect(lambda:self.set_mode(False)); self.advanced_b.clicked.connect(lambda:self.set_mode(True)); self.compare_b.clicked.connect(lambda:self.set_mode('compare'))
        self.controls=QFrame(); g=QGridLayout(self.controls); g.setContentsMargins(0,0,0,0); g.setSpacing(8)
        self.method=QComboBox()
        for label,key in [('Standard','tsdf'),('Points','points'),('Plus','plus')]: self.method.addItem(label,key)
        self.detail=QDoubleSpinBox(); self.detail.setRange(.1,2); self.detail.setSingleStep(.05); self.detail.setDecimals(2); self.detail.setSuffix(' mm'); self.detail.setValue(.4)
        self.cleanup=QDoubleSpinBox(); self.cleanup.setRange(0,100); self.cleanup.setSingleStep(.5); self.cleanup.setSuffix(' %'); self.cleanup.setValue(3)
        self.smooth=QSpinBox(); self.smooth.setRange(0,20)
        self.frames=QSpinBox(); self.frames.setRange(50,1000); self.frames.setSingleStep(50); self.frames.setValue(300)
        self.frames.setToolTip('The most frames used from each side, spread evenly. Fewer builds faster; more can add detail but takes longer and can blur the surface.')
        self.quality=QSpinBox(); self.quality.setRange(1,7); self.quality.setValue(6)
        self.quality.setToolTip('How finely the Points method builds its surface, 1 (coarse) to 7 (finest, slowest). Standard and Plus builds do not use it.')
        self.strength=QSpinBox(); self.strength.setRange(5,100); self.strength.setSingleStep(5); self.strength.setSuffix(' %'); self.strength.setValue(30)
        self.strength.setToolTip('How hard each smooth pass works. Only matters when Smooth passes is above 0.')
        for row,(text,control) in enumerate([('Build method',self.method),('Detail size',self.detail),('Remove loose pieces',self.cleanup),('Smooth passes',self.smooth),
                                             ('Frames per side',self.frames),('Surface quality',self.quality),('Smooth strength',self.strength)]):
            control.setFixedHeight(30); g.addWidget(QLabel(text),row//2,(row%2)*2); g.addWidget(control,row//2,(row%2)*2+1)
        self.finish=QCheckBox('Finish for print after building (closed and smooth, about 250,000 triangles)'); self.finish.setFixedHeight(30); g.addWidget(self.finish,4,0,1,4)
        self.finish.setToolTip('After the build, makes one smooth, closed model of about 250,000 triangles, ready for the slicer: holes are closed, the surface is smoothed and the file is kept small (about 12.5 MB as STL). Leave it off to keep the model as scanned. You can still run Finish for print later in Build, and set its triangle count, smoothness and hole filling there.')
        self.finish.setStyleSheet('QCheckBox {spacing:8px;} QCheckBox::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;background:#151a21;} QCheckBox::indicator:checked {background:#5b9df0;border:2px solid #dceaff;}')
        self.controls.setFixedHeight(184); av.addWidget(self.controls)
        helptext=QLabel('Custom settings: smaller detail takes longer. Zero cleanup keeps loose pieces.'); helptext.setFixedHeight(24)
        help_row=QHBoxLayout(); help_row.addWidget(helptext,1)
        self.custom_help=HelpButton('Custom settings', 'Cleanup removes pieces below this share of the largest piece. Smooth passes soften noise, and Smooth strength sets how hard each pass works. Frames per side is how many frames each side contributes. Surface quality applies to the Points method only. Save preset keeps these settings for another scan.')
        help_row.addWidget(self.custom_help); av.addLayout(help_row)
        # naming and saving sit right under the preset picker (Advanced only), not in the middle of the numbers
        self.save_row_widget=QWidget(); sv=QVBoxLayout(self.save_row_widget); sv.setContentsMargins(0,0,0,0); sv.setSpacing(2)
        save=QHBoxLayout(); self.name=QLineEdit(); self.name.setPlaceholderText('Name these settings to save them'); self.name.setMaxLength(60); self.name.setFixedHeight(32)
        self.save_b=button('Save preset','quiet'); self.save_b.setFixedSize(116,32); self.save_b.clicked.connect(self.save_current)
        save.addWidget(self.name,1); save.addWidget(self.save_b); sv.addLayout(save)
        self.save_note=StableLabel('',1); self.save_note.setFixedHeight(22); sv.addWidget(self.save_note)
        v.insertWidget(v.indexOf(self.preset_row_widget)+1,self.save_row_widget)
        grid=QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(4); self.lines=[]; self.keys=[]
        for row,text in enumerate(['Build detail','Cleanup','Surface','Final changes']):
            label=QLabel(text); label.setFixedHeight(30); line=StableLabel('',2); line.setFixedHeight(30)
            grid.addWidget(label,row,0); grid.addWidget(line,row,1); self.lines.append(line); self.keys.append(label)
        grid.setColumnStretch(1,1); av.addLayout(grid); av.addStretch(1)
        self.foot=StableLabel('',2); self.foot.setFixedHeight(38); v.addWidget(self.foot)
        row=QHBoxLayout(); self.cancel_b=big_button('Cancel','quiet'); self.ok_b=big_button('Yoink it','build')
        self.ultra_b=big_button('Ultra Yoink','default'); self.ultra_b.setFixedHeight(44)
        self.ultra_b.setToolTip('Builds several recipes on this part, one after another, so you can flip between the results and keep the best. It shows the time and disk space first.')
        self.ultra_b.clicked.connect(self.ultra.emit)
        self.cancel_b.setFixedHeight(44); self.ok_b.setFixedHeight(44); row.addWidget(self.cancel_b); row.addWidget(self.ultra_b); row.addWidget(self.ok_b); v.addLayout(row)
        self.cancel_b.clicked.connect(self.cancel.emit); self.ok_b.clicked.connect(self.run)
        self.method.currentIndexChanged.connect(self.changed)
        for c in (self.detail,self.cleanup,self.smooth,self.frames,self.quality,self.strength): c.valueChanged.connect(self.changed)
        self.finish.toggled.connect(self.changed)
        self.set_mode(False); self.fix_size(); self.hide()

    def show_pictures(self, project):
        """Put the user's own part on every card whose recipe has built it in this project (the rest keep the demo clip)."""
        from PySide6.QtGui import QIcon, QPixmap
        from app2.recipe_pictures import existing, when, key_of
        self.pictures=existing(project) if project else {}
        for key,card in self.profile_cards.items():
            own=self.pictures.get(key_of(key))
            if own:
                pix=QPixmap(own[0]).scaled(100,69,Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation)
                card.pic.setPixmap(fit_pix(pix)); card.setToolTip(card.toolTip().split('\nYour part')[0]+'\nYour part, built %s' % when(own[1]))
        if self.current: self._refresh_preview(self.current)

    def _fill_compare(self):
        """One card per recipe this project has a picture for: the picture big, its name and date, and Use this recipe."""
        from PySide6.QtGui import QPixmap
        from app2.recipe_pictures import key_of, when
        while self.compare_grid.count():
            w=self.compare_grid.takeAt(0).widget()
            if w is not None: w.deleteLater()
        names={key_of(k):(k,label) for k,label,_h in self.choices}
        found=sorted(self.pictures.items(),key=lambda kv:-kv[1][1])          # newest first
        if not found:
            empty=QLabel('Nothing to compare yet.\n\nEach recipe you build on this part leaves a picture here. Ultra Yoink builds several in one go.')
            empty.setWordWrap(True); empty.setAlignment(Qt.AlignmentFlag.AlignCenter); empty.setStyleSheet('color:#8b95a1;font-size:14px;padding:30px')
            self.compare_grid.addWidget(empty,0,0); return
        for i,(pk,(path,t)) in enumerate(found):
            key,label=names.get(pk,('custom' if pk=='custom' else 'saved:'+pk[6:] if pk.startswith('saved-') else pk,pk.replace('-',' ').title()))
            card=QFrame(); card.setStyleSheet('QFrame {background:#20252c;border:1px solid #38434f;border-radius:8px;}'); cv=QVBoxLayout(card); cv.setContentsMargins(8,8,8,8); cv.setSpacing(4)
            pic=QLabel(); pic.setPixmap(QPixmap(path).scaled(300,200,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)); pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pic.setStyleSheet('border:0'); pic.setMinimumHeight(140)
            title=QLabel('<b>%s</b><br><span style="color:#8b95a1">built %s</span>' % (label,when(t))); title.setStyleSheet('border:0')
            use=button('Use this recipe','default'); use.clicked.connect(lambda checked=False,k=key:(self.set_mode(False),self.pick(k,True)))
            cv.addWidget(pic,1); cv.addWidget(title); cv.addWidget(use)
            self.compare_grid.addWidget(card,i//2,i%2)
        self.compare_grid.setColumnStretch(0,1); self.compare_grid.setColumnStretch(1,1)

    def _lay_cards(self):
        """Presets and saved ones in two columns, then a line, then Custom across the bottom."""
        for w in [*self.profile_cards.values(), self.card_line]: self.card_grid.removeWidget(w)
        for r in range(self.card_grid.rowCount()+1): self.card_grid.setRowStretch(r,0)
        rest=[k for k,_l,_h in self.choices if k!='custom' and k in self.profile_cards]
        for i,k in enumerate(rest): self.card_grid.addWidget(self.profile_cards[k],i//2,i%2)
        row=(len(rest)+1)//2
        self.card_grid.addWidget(self.card_line,row,0,1,2); self.card_grid.addWidget(self.profile_cards['custom'],row+1,0,1,2)
        self.card_grid.setRowStretch(row+2,1)

    def set_mode(self, advanced):
        """False: Simple, True: Advanced, 'compare': the Compare tab."""
        mode='compare' if advanced=='compare' else ('advanced' if advanced else 'simple'); advanced=mode=='advanced'
        self.pages.setCurrentIndex({'simple':0,'advanced':1,'compare':2}[mode])
        self.simple_b.setChecked(mode=='simple'); self.advanced_b.setChecked(advanced); self.compare_b.setChecked(mode=='compare')
        self.preset_row_widget.setVisible(advanced); self.save_row_widget.setVisible(advanced); self.ultra_b.setVisible(not advanced)
        self.hint.setVisible(advanced)
        if mode=='compare': self._fill_compare()
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
        return dict(method=self.method.currentData(),detail=self.detail.value(),cleanup=self.cleanup.value(),smooth=self.smooth.value(),finish=self.finish.isChecked(),
                    frames=self.frames.value(),quality=self.quality.value(),strength=float(self.strength.value()))

    def _gate(self):
        """Grey out what the current choices would ignore: quality is for Points only, strength needs a smooth pass."""
        self.quality.setEnabled(self.method.currentData()=='points'); self.strength.setEnabled(self.smooth.value()>0)

    def payload(self): return {'preset':self.current,'fragment':store.recipe_fragment(self.settings())}

    def _refresh_preview(self,key):
        if self.pages.currentIndex()==2: self.preview.hide(); return       # Compare shows the pictures themselves
        name='today' if key=='auto' else key
        pic=os.path.join(os.path.dirname(__file__),'assets','presets','%s.png' % name)
        from app2.recipe_pictures import key_of
        own=self.pictures.get(key_of(key))                 # the user's own part built with this recipe, when there is one
        if own: pic=own[0]
        if not key: self.preview.hide(); return
        if not os.path.exists(pic): self._placeholder(key); return
        from PySide6.QtGui import QPixmap
        self.preview_pic.setPixmap(QPixmap(pic).scaled(self.preview_pic.size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation))
        label=next((l for k,l,_h in self.choices if k==key),key)
        self.preview_text.setText('<b>%s</b><br>%s<br><br><span style="color:#8b95a1">%s<br>Example: a clip and raised lettering on a test part, built with this recipe. Ultra Yoink builds your own scan every way to compare.</span>'
                                  % (label, HINTS.get(name,''), W.RECIPE['card_facts'].get(name,'')))
        if own:
            from app2.recipe_pictures import when
            self.preview_text.setText('<b>%s</b><br>%s<br><br><span style="color:#8b95a1">%s<br>Your part, built with this recipe on %s.</span>'
                                      % (label, HINTS.get(name,''), W.RECIPE['card_facts'].get(name,''), when(own[1])))
        if self.parentWidget() and self.parentWidget().window().height() <= 900:
            self.preview_text.setText('<b>%s</b><br>%s' % (label, HINTS.get(name, '')))
        self.preview.show()

    def _placeholder(self,key):
        """Cards with no picture (Custom, saved settings, nothing built yet): a plain tile that says so."""
        from PySide6.QtGui import QPixmap, QPainter, QColor
        size=self.preview_pic.size(); pix=QPixmap(size); pix.fill(QColor('#1d2229'))
        p=QPainter(pix); p.setPen(QColor('#8b95a1')); p.drawText(pix.rect(),Qt.AlignmentFlag.AlignCenter,'No picture yet'); p.end()
        self.preview_pic.setPixmap(pix)
        label=next((l for k,l,_h in self.choices if k==key),key)
        hint=next((h for k,_l,h in self.choices if k==key),'')
        self.preview_text.setText('<b>%s</b><br>%s<br><br><span style="color:#8b95a1">Not built on your part yet. Yoink it, or run Ultra Yoink, and its picture shows up here.</span>' % (label,hint))
        if self.parentWidget() and self.parentWidget().window().height() <= 900:
            self.preview_text.setText('<b>%s</b><br>%s' % (label,hint))
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
            self.frames.setValue(settings['frames']); self.quality.setValue(settings['quality']); self.strength.setValue(int(settings['strength']))
        self.controls.setEnabled(editable); self.name.setEnabled(editable); self.save_b.setEnabled(editable); self._gate()
        self.name.clear()                                  # the box is for a NEW name: never the picked preset's own name again
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
        self._gate()
        if self.current!='custom':
            self.current='custom'; self.selector.blockSignals(True); self.selector.setCurrentIndex(self.selector.findData('custom')); self.selector.blockSignals(False)
        self._pending(); self.custom_changed.emit(self.payload())

    def save_current(self):
        try:
            name=store.save_preset(self.name.text(),self.settings()); self.saved[name]=self.settings(); key='saved:'+name
            if key not in self.btns:
                at=next(i for i,c in enumerate(self.choices) if c[0]=='custom')       # before Custom, which stays at the bottom
                self.choices.insert(at,(key,name,'Your saved build settings.')); self.selector.insertItem(self.selector.findData('custom')-1,name,key); self.btns[key]=self.selector
            if key not in self.profile_cards:
                card=recipe_card(name+'\nYour saved settings\n'+facts_line(dict(store.DEFAULTS,**self.settings())),own_icon()); card.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
                card.setStyleSheet(CARD_STYLE); card.setToolTip(name)
                card.clicked.connect(lambda checked=False,k=key:self.pick(k,True))
                self.profile_cards[key]=card; self._lay_cards()
            self.current=key; self.selector.blockSignals(True); self.selector.setCurrentIndex(self.selector.findData(key)); self.selector.blockSignals(False)
            for k,c in self.profile_cards.items(): c.setChecked(k==key)       # the saved settings are now the picked ones: one name, shown once
            self.name.clear(); self.save_note.setText('Saved '+name)
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
