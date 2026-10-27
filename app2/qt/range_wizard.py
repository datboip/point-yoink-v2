# SPDX-License-Identifier: AGPL-3.0-or-later
"""A separate, steady-size Range test with explicit review and Apply."""
from PySide6.QtCore import Signal, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QStackedWidget,QWidget,QProgressBar




class SweepRuler(QWidget):
    """The distances the sweep needs, near to far: each step fills green as it is measured, the scanner's distance
    is the marker."""
    def __init__(self):
        super().__init__();self.progress={};line=QFontMetrics(self.font()).lineSpacing();self.setMinimumHeight(line*4)
    def set_progress(self,progress):self.progress=progress or {};self.update()
    def paintEvent(self,event):
        from app2.qt.theme import qc
        g=self.progress;lo,hi=g.get('lo_mm'),g.get('hi_mm')
        if lo is None or hi is None or hi<=lo:return
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);fm=self.fontMetrics();line=fm.lineSpacing()
        pad=fm.horizontalAdvance('0000 mm')/2;w=self.width()-2*pad;top=line*1.2;h=line*1.2
        x=lambda mm:pad+w*(min(max(mm,lo),hi)-lo)/(hi-lo)
        p.setPen(Qt.PenStyle.NoPen);p.setBrush(qc('well'));p.drawRoundedRect(QRectF(pad,top,w,h),4,4)
        step=g.get('step_mm') or 5
        for mm,done in (g.get('bins') or {}).items():
            c=qc('kept' if done>=1 else 'warn');c.setAlpha(255 if done>=1 else 110);p.setBrush(c)
            p.drawRect(QRectF(x(mm-step/2),top,max(1,x(mm+step/2)-x(mm-step/2)),h))
        p.setPen(qc('ink2'))
        for mm in range(int(-(-lo//50)*50),int(hi)+1,50):
            p.drawLine(int(x(mm)),int(top+h),int(x(mm)),int(top+h+line/3))
            if mm%100==0:p.drawText(QRectF(x(mm)-pad,top+h+line/3,2*pad,line),Qt.AlignmentFlag.AlignHCenter,'%d mm'%mm)
        z=g.get('distance_mm')
        if z is not None:
            p.setPen(Qt.PenStyle.NoPen);p.setBrush(qc('accent'));p.drawRoundedRect(QRectF(x(z)-2,top-line/3,4,h+line*2/3),2,2)
            p.setPen(qc('ink'));p.drawText(QRectF(x(z)-pad,0,2*pad,line),Qt.AlignmentFlag.AlignHCenter,'%d mm'%z)


class RangeWizard(QDialog):
    start=Signal();stop=Signal();apply=Signal();defaults=Signal()
    def __init__(self,chart,parent=None):
        super().__init__(parent);self.setWindowTitle('Range test');self.active=False;self.result=None
        line=QFontMetrics(self.font()).lineSpacing();self.setFixedSize(line*34,line*25)
        v=QVBoxLayout(self);self.pages=QStackedWidget();v.addWidget(self.pages,1)
        def page(title):
            w=QWidget();lay=QVBoxLayout(w);heading=QLabel(title);heading.setStyleSheet('font-weight:bold');lay.addWidget(heading);self.pages.addWidget(w);return lay
        def text(lay,value):
            label=QLabel(value);label.setWordWrap(True);label.setMinimumWidth(1);lay.addWidget(label);return label
        a=page('1. Measure your distance bands')
        text(a,'This test finds the distances where your scanner sees best, so the distance bar is right for your scanner. Take the part off the bed. You will move the scanner slowly closer and farther and hold still at each spot. Your scan is not changed, and nothing is used until you press Apply.');a.addStretch()
        b=page('2. Check the setup');self.checks=text(b,'Waiting for the camera and a learned bed.');b.addStretch()
        c=page('3. Sweep the distance');text(c,'Point the scanner at the empty bed and move the scanner itself, slowly, closer and farther. The bed does not need to turn. Hold still at each spot until it turns green.')
        self.action=text(c,'Point the scanner at the empty bed');self.action.setStyleSheet('font-size:%dpx;font-weight:bold' % round(line*1.2))
        self.ruler=SweepRuler();c.addWidget(self.ruler);self.distance=text(c,'Distance: waiting for depth');self.span=QProgressBar();self.span.setRange(0,100);c.addWidget(self.span)
        text(c,'Green: measured. Orange: still measuring. The blue line is where the scanner is now.');c.addStretch()
        from app2.qt.steady import Steady
        self.hint_steady=Steady(self.action.setText,self)
        d=page('4. Review the result');d.addWidget(chart,1);self.bands=text(d,'No result yet');self.source=text(d,'Your current distance bar stays in use until you press Apply.')
        row=QHBoxLayout();self.back=QPushButton('Back');self.next=QPushButton('Next');self.begin=QPushButton('Begin sweep');self.finish=QPushButton('Stop and review');self.accept=QPushButton('Apply');self.keep=QPushButton('Keep defaults')
        for button in (self.back,self.next,self.begin,self.finish,self.accept,self.keep):row.addWidget(button)
        v.addLayout(row);self.back.clicked.connect(lambda:self.step(self.pages.currentIndex()-1));self.next.clicked.connect(lambda:self.step(self.pages.currentIndex()+1));self.begin.clicked.connect(self.start);self.finish.clicked.connect(self.stop);self.accept.clicked.connect(self.apply);self.keep.clicked.connect(self._defaults);self.step(0)
    def _defaults(self):self.defaults.emit();self.close()
    def step(self,index):
        self.pages.setCurrentIndex(max(0,min(3,index)));i=self.pages.currentIndex()
        for b,on in ((self.back,i in (1,3) and not self.active),(self.next,i==0),(self.begin,i==1),(self.finish,i==2),(self.accept,i==3),(self.keep,i==3)):b.setVisible(on)
    def set_state(self,state):
        from app2.scan.object_modes import label
        s=getattr(state,'range_test',None) or {};active=bool(s.get('active'));result=s.get('result')
        checks=s.get('setup') or {};bed=bool(checks.get('empty_mat'));mode=bool(checks.get('camera_match'));light=checks.get('light_state') or getattr(state,'led_status',None) or 'unknown'
        self.checks.setText('%s Bed learned and empty\n%s %s camera\nWhite light: %s\nKeep the bed empty during the test.' % ('✓' if bed else '○','✓' if mode else '○',label(getattr(state,'scan_mode',None) or 'near'),light))
        self.begin.setEnabled(bed and mode and not any(getattr(state,k,None) for k in ('recording','busy','building','learning')))
        if active:self.step(2)
        elif self.active:self.step(3)
        self.active=active;self.result=result;progress=s.get('progress') or {}
        z=progress.get('distance_mm');self.distance.setText('Distance: waiting for depth' if z is None else 'Distance: %.0f mm' % z)
        self.span.setValue(round(100*progress.get('fraction',0)));self.span.setFormat('%.0f of %.0f mm measured' % (progress.get('covered_mm',0),progress.get('target_mm',0)))
        self.ruler.set_progress(progress)
        # a hint stays up long enough to read; a newer one waits its turn
        hint=s.get('hint') or progress.get('action','Point the scanner at the empty bed')
        self.hint_steady.offer(hint,hint)
        ready=bool(result and result.get('complete') and result.get('camera_pair_confirmed'));self.accept.setEnabled(ready)
        if result:
            bands=result.get('distance_bands',[]);self.bands.setText(('Done. Press Apply to use it. ' if ready else 'Not finished. Cover more distances and hold still at each one. ')+ '; '.join('%s %g to %g mm' % ('Best' if b['state']=='strong depth' else 'OK',b['lo'],b['hi']) for b in bands))
        self.finish.setEnabled(active)
    def closeEvent(self,event):
        if self.active:self.stop.emit()
        super().closeEvent(event)
