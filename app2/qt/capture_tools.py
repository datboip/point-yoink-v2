# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small scan setup controls, sized by the current font and available space."""
import math
from PySide6.QtCore import Qt,Signal,QRectF,QPointF
from PySide6.QtGui import QPainter,QFontMetrics,QPen,QColor
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QDoubleSpinBox,QFormLayout
from app2.qt.theme import qc


class RangeChart(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.result=None;self.setMinimumHeight(QFontMetrics(self.font()).lineSpacing()*6)
    def set_result(self,result):
        self.result=result;self.setVisible(bool(result));self.setToolTip('Distance bins: fill, flatness noise, dots, dot fit and color overlay error. Best uses at least median fill and at most median flatness noise. Unmeasured gaps stay gray.');self.update()
    def plot_rect(self):
        fm=QFontMetrics(self.font());line=fm.lineSpacing()
        return QRectF(fm.horizontalAdvance('100%')+line,line*2,max(1,self.width()-fm.horizontalAdvance('100%')-line*2),max(1,self.height()-line*4))
    def paintEvent(self,event):
        if not self.result:return
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);fm=QFontMetrics(self.font());line=fm.lineSpacing()
        rows=self.result['bins'];lo,hi=self.result['observed_distance_mm'];box=self.plot_rect()
        p.fillRect(box,qc('well'));p.setPen(qc('ink'));p.drawText(QPointF(0,line),'Mat depth fill')
        for fraction in (0,.5,1):
            y=box.bottom()-box.height()*fraction
            p.setPen(qc('line'));p.drawLine(QPointF(box.left(),y),QPointF(box.right(),y))
            p.setPen(qc('ink2'));p.drawText(QRectF(0,y-line/2,box.left()-line/2,line),Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,'%.0f%%' % (100*fraction))
        from app2.scan.assist import RANGE_STEP_MM
        from app2.scan.range_guide import distance_levels,BAND_RGB
        span=max(hi-lo,RANGE_STEP_MM);width=box.width()*RANGE_STEP_MM/span
        for row in rows:
            x=box.left()+box.width()*(row['distance_mm']-lo)/span
            level=int(distance_levels([row['distance_mm']],self.result)[0]);color=BAND_RGB[('off','ok','best')[level]]
            h=box.height()*row['fill'];p.fillRect(QRectF(x-width/2,box.bottom()-h,max(1,width-line/4),h),QColor(*color))
        p.setPen(qc('ink2'));p.drawText(QRectF(0,self.height()-line*1.5,self.width(),line),Qt.AlignmentFlag.AlignCenter,'%g to %g mm, %d bins' % (lo,hi,len(rows)))
        p.end()


class CaptureTools(QWidget):
    overlay_changed=Signal(dict);overlay_auto=Signal();overlay_reset=Signal();range_test=Signal();range_reset=Signal();range_apply=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);v=QVBoxLayout(self);v.setContentsMargins(0,0,0,0);v.setSpacing(QFontMetrics(self.font()).lineSpacing()//3)
        row=QHBoxLayout();self.range_button=QPushButton('Range test');self.reset_range=QPushButton('Reset test');row.addWidget(self.range_button);row.addWidget(self.reset_range);v.addLayout(row)
        self.range_button.setToolTip('Empty learned dot mat only. Choose the scanner camera before PC mode. Move slowly from near to far, then stop and save. Both calibrations are checked. No frames are added to your scan.')
        self.range_button.clicked.connect(self.open_wizard);self.reset_range.clicked.connect(self.range_reset)
        # The test's own box with a fixed height: its hint, chart and result fill reserved room, so nothing above or
        # below moves when a test starts, runs or ends.
        from PySide6.QtWidgets import QFrame,QSizePolicy
        line=QFontMetrics(self.font()).lineSpacing()
        self.test_box=QFrame();self.test_box.setObjectName('RangeTestBox');bv=QVBoxLayout(self.test_box);bv.setContentsMargins(line//2,line//3,line//2,line//3);bv.setSpacing(line//4)
        self.hint=QLabel('');self.hint.setWordWrap(True);self.hint.setFixedHeight(2*line);bv.addWidget(self.hint)
        self.test_box.setToolTip('Range test: measures how well depth works at each distance for this scanner, so the distance bar can use your real bands. Use the empty learned dot mat only. Nothing is added to your scan.')
        res=QHBoxLayout();res.setSpacing(line//2);self.detail=QLabel('');self.detail.setFixedHeight(line);res.addWidget(self.detail,1)
        self.view_result=QPushButton('View result');self.view_result.setEnabled(False);res.addWidget(self.view_result);bv.addLayout(res)
        # The chart opens in its own small window, so the box keeps one size.
        from PySide6.QtWidgets import QDialog
        self.result_window=QDialog(self);self.result_window.setWindowTitle('Range test result');rw=QVBoxLayout(self.result_window)
        self.chart=RangeChart();self.chart.setMinimumSize(line*24,line*8);rw.addWidget(self.chart)
        from app2.qt.range_wizard import RangeWizard
        self.wizard=RangeWizard(self.chart,self);self.wizard.start.connect(self.range_test);self.wizard.stop.connect(self.range_test);self.wizard.apply.connect(self.range_apply);self.wizard.defaults.connect(self.range_reset)
        self.view_result.clicked.connect(lambda:(self.wizard.step(3),self.wizard.show()))
        for label in (self.hint,self.detail):label.setMinimumWidth(1)   # wrapped text never widens the panel
        m=bv.contentsMargins();self.test_box.setFixedHeight(m.top()+m.bottom()+2*line+bv.spacing()+max(line,self.view_result.sizeHint().height()));self.test_box.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed);v.addWidget(self.test_box)
        self.nudge=QWidget();form=QFormLayout(self.nudge);form.setContentsMargins(0,0,0,0);self.fields={}
        from app2.scan.fmt import W,H
        diagonal=math.hypot(W,H)
        for key,title,lo,hi,step,default in [('x','Horizontal (px)',-W/2,W/2,1,0),('y','Vertical (px)',-H/2,H/2,1,0),('angle','Rotate (deg)',-90,90,math.degrees(math.atan(1/diagonal)),0),('scale','Scale',.5,2,1/diagonal,1)]:
            field=QDoubleSpinBox();field.setRange(lo,hi);field.setDecimals(3);field.setSingleStep(step);field.setValue(default);field.valueChanged.connect(self._changed);self.fields[key]=field;form.addRow(title,field)
        row=QHBoxLayout();self.auto=QPushButton('Auto align');self.reset=QPushButton('Reset overlay');row.addWidget(self.auto);row.addWidget(self.reset);form.addRow(row)
        self.auto.clicked.connect(self.overlay_auto);self.reset.clicked.connect(self.overlay_reset)
        self.steps=QLabel('Steps: 1 px shift; rotation and scale move the image edge by about 1 px.');self.steps.setWordWrap(True);form.addRow(self.steps)
        self.fit=QLabel('Color overlay only');self.fit.setWordWrap(True);form.addRow(self.fit)
        self.setup_window=QDialog(self);self.setup_window.setWindowTitle('Camera setup');sv=QVBoxLayout(self.setup_window);sv.addWidget(self.nudge)
        self.test_box.hide()
    def open_wizard(self):
        self.wizard.step(0);self.wizard.show();self.wizard.raise_()
    def _changed(self):self.overlay_changed.emit({k:f.value() for k,f in self.fields.items()})
    def set_state(self,state,setup):
        overlay=getattr(state,'overlay',None) or {};manual=overlay.get('manual',{})
        for key,field in self.fields.items():
            field.blockSignals(True);field.setValue(manual.get(key,1 if key=='scale' else 0));field.blockSignals(False)
        fit=overlay.get('fit');self.fit.setText('Dot error %.2f to %.2f px (%d dots)' % (fit['before_px'],fit['after_px'],fit['matches']) if fit else 'Color overlay only')
        self.nudge.setEnabled(not state.recording);self.wizard.set_state(state)
        sweep=getattr(state,'range_test',None) or {};active=sweep.get('active',False);result=sweep.get('result');self.chart.set_result(result)
        self.range_button.setText('Range test');self.range_button.setEnabled(active or not (state.recording or state.busy or state.building or state.learning));self.reset_range.setEnabled(not active)
        self.hint.setText(sweep.get('hint','') if active else 'Empty dot mat on the bed, press Range test, then move slowly from close to far.')
        from scanner.bed import ACCEPT_HITS
        partial=bool((result is not None or sweep.get('attempted')) and len((result or {}).get('bins',[]))<ACCEPT_HITS)
        self.detail.setText('Not enough distances yet, keep moving' if partial else 'Result ready: View result for the chart' if result else '')
        self.view_result.setEnabled(bool(result))
        if result:self.chart.setToolTip('\n'.join('%g mm: %.0f%% fill, %.3f mm plane RMS, %g dots, dot RMS %s, overlay %s px; Small/Large votes %s' % (b['distance_mm'],100*b['fill'],b['plane_rms_mm'],b['dots'],b.get('dot_rms_mm'),b.get('overlay_error_px'),b['camera_matches']) for b in result['bins']))
