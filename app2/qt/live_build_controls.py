# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live build controls with a reserved status row and settings in a dialog."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QPushButton,QDialog,QFormLayout,QComboBox,QSpinBox,QDoubleSpinBox,QLabel


class LiveBuildControls(QWidget):
    changed=Signal(object,object,object,object);undo=Signal();next_pass=Signal();export=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);v=QVBoxLayout(self);v.setContentsMargins(0,0,0,0)
        from app2.qt.status_chip import LiveStatus
        self.note=LiveStatus();v.addWidget(self.note);row=QHBoxLayout();self.settings=QPushButton('Live settings');self.back=QPushButton('Undo recent');self.next=QPushButton('Next pass')
        self.settings.setToolTip('Detail and pass settings for Live build. Works when Capture is Live build and you are paused.')
        self.back.setToolTip('Take back the last few seconds of the live model, for example after a bump. Live build only.')
        from app2.qt.help_text import HelpButton
        self.help=HelpButton('Live build','Live build fuses the model while you scan and saves only views that add new surface or a better distance. Pick it under Capture. Live settings sets detail and passes; Undo recent removes the last couple of seconds. These buttons stay gray in other capture modes.')
        for w in (self.settings,self.back):row.addWidget(w)
        row.addWidget(self.help)
        v.addLayout(row);self.window=QDialog(self);self.window.setWindowTitle('Live build settings');form=QFormLayout(self.window)
        from app2.qt.choices import Choices
        self.detail=Choices();self.support=QSpinBox();self.support.setRange(1,65535);self.support.setValue(int(__import__('app2.scan.live_build',fromlist=['tracker_default']).tracker_default('COVER_GOOD')));self.share=QDoubleSpinBox();self.share.setRange(0,100);self.share.setValue(3);self.share.setSuffix('%');self.angle=Choices();self.angle.addItems(('Free','Low','Level','Flipped'))
        form.addRow('Detail',self.detail);form.addRow('Good views per region',self.support);form.addRow('Minimum added share',self.share);form.addRow('Angle pass',self.angle);form.addRow(self.next);self.save=QPushButton('Export live model');form.addRow(self.save);self.save.clicked.connect(self.export)
        self.angle.setToolTip('Low: scanner tilted about 25 degrees down. Level: scanner beside the part. Flipped: part flipped to see its bottom. Each pass is its own side.')
        self.warning=QLabel('Fine detail uses more memory. Coarse detail can lose thin walls. Build combined model makes the final model from the frames kept here.');self.warning.setWordWrap(True);self.warning.setMinimumWidth(1);form.addRow(self.warning)
        self.apply=QPushButton('Use for next side');form.addRow(self.apply);self.apply.clicked.connect(self._apply);self.settings.clicked.connect(self.window.show);self.back.clicked.connect(self.undo);self.next.clicked.connect(self.next_pass);self._mode=None
    def _apply(self):self.changed.emit(self.detail.currentData(),self.support.value(),self.share.value()/100,self.angle.currentText());self.window.close()
    def set_state(self,state):
        mode=getattr(state,'scan_mode',None) or 'near'
        if self._mode!=mode:
            self._mode=mode;self.detail.clear()
            from app2.scan.live_build import detail_levels
            for name,mm in detail_levels(mode).items():self.detail.addItem('%s (%g mm)' % (name,round(mm,3)),name)
        self.detail.setCurrentIndex(max(0,self.detail.findData(getattr(state,'live_detail',None) or 'Balanced')));self.angle.setCurrentText(getattr(state,'live_pass',None) or 'Free')
        live=getattr(state,'capture_mode',None)=='live';busy=bool(state.busy or state.building or state.learning)
        self.settings.setEnabled(live and not state.recording and not busy);self.back.setEnabled(live and bool(getattr(state,'live_undo',False)) and not busy);self.next.setEnabled(live and not state.recording and not busy);self.save.setEnabled(live and not state.recording and not busy and bool(state.picked_has_frames))
        data=getattr(state,'live_build',None) or {}
        text=('%d seen, %d kept. %.0f%% ready' % (data['seen'],data['kept'],data['done_share']*100)) if data else 'Live build: fuse live, keep useful views'
        self.note.set_status('ok' if data.get('pass_done') else 'warn' if live else 'idle',text,data.get('message','Low: aim about 25 degrees down. Level: aim at the side. Flipped: see the bottom.'))
