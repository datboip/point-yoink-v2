# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device captures, copied by an isolated cancellable worker, never by the GUI thread."""
import json, os, sys
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QProcess, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QLineEdit,QTreeWidget,QTreeWidgetItem,QProgressBar
from app2 import REPO

class DeviceCaptures(QFrame):
    connect_requested=Signal()
    dismissed=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);self.setObjectName('DeviceCaptures');self.setStyleSheet('#DeviceCaptures {background:#191d23;}')
        self.root=None;self.receipt=None;self.proc=None;self.mode=None;self.pending=[];self.buffer='';self.failed=False;self.closing=False
        self.mountpoint=str(Path.home()/'.cache/pointyoink/device-captures-mtp')
        box=QVBoxLayout(self);box.setContentsMargins(28,24,28,24);box.setSpacing(12)
        title=QLabel('Import device captures');title.setStyleSheet('font-size:20px;font-weight:600');box.addWidget(title)
        note=QLabel('Connect USB and choose File Transfer on the scanner. Screenshots and recordings are copied; originals stay on the device. Connecting saves and pauses the PC scan session.');note.setWordWrap(True);box.addWidget(note)
        row=QHBoxLayout();self.connect_b=QPushButton('Connect USB / Refresh');self.connect_b.clicked.connect(self.connect_requested.emit);row.addWidget(self.connect_b)
        self.all_b=QPushButton('Select all');self.all_b.clicked.connect(lambda:self.select_all(True));row.addWidget(self.all_b)
        self.none_b=QPushButton('Clear selection');self.none_b.clicked.connect(lambda:self.select_all(False));row.addWidget(self.none_b);row.addStretch();box.addLayout(row)
        self.files=QTreeWidget();self.files.setStyleSheet("QTreeWidget {background:#11161c;border:1px solid #38434f;border-radius:6px;} QTreeWidget::item {padding:7px;} QHeaderView::section {background:#282e36;color:#dce5ef;padding:7px;border:0;} QTreeView::indicator {width:16px;height:16px;border:1px solid #8290a6;border-radius:3px;background:#191f27;} QTreeView::indicator:checked {background:#5b9df0;border:2px solid #dceaff;}");self.files.setHeaderLabels(['Capture','Type','Size']);self.files.setRootIsDecorated(False);self.files.setColumnWidth(0,460);box.addWidget(self.files,1)
        dest=QHBoxLayout();dest.addWidget(QLabel('Save to'));self.destination=QLineEdit(str(Path.home()/'Pictures/PointYoink'));dest.addWidget(self.destination,1);box.addLayout(dest)
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);box.addWidget(self.progress)
        self.status=QLabel('Choose File Transfer on the scanner, then Connect USB.');self.status.setWordWrap(True);box.addWidget(self.status)
        bottom=QHBoxLayout();self.copy_b=QPushButton('Import selected');self.copy_b.clicked.connect(self.copy);bottom.addWidget(self.copy_b)
        self.folder_b=QPushButton('Open saved folder');self.folder_b.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.expanduser(self.destination.text())))));bottom.addWidget(self.folder_b)
        self.stop_b=QPushButton('Cancel transfer');self.stop_b.clicked.connect(self.cancel);bottom.addWidget(self.stop_b)
        self.close_b=QPushButton('Done');self.close_b.clicked.connect(self.dismiss);bottom.addWidget(self.close_b);box.addLayout(bottom)
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.timed_out)
        self._buttons(False);self.hide()
    def present(self):
        if hasattr(self.parentWidget(),"centralWidget"):self.parentWidget().centralWidget().setEnabled(False)
        self.setGeometry(self.parentWidget().rect().adjusted(20,40,-20,-20));self.show();self.raise_()
    def _buttons(self,busy):
        for w in (self.connect_b,self.files,self.destination,self.all_b,self.none_b):w.setEnabled(not busy)
        self.copy_b.setEnabled(not busy and bool(self.root) and self.files.topLevelItemCount()>0)
        self.stop_b.setEnabled(busy and self.mode!='unmount');self.close_b.setEnabled(not busy)
    def select_all(self,on):
        for i in range(self.files.topLevelItemCount()):self.files.topLevelItem(i).setCheckState(0,Qt.Checked if on else Qt.Unchecked)
    def connect_usb(self):
        if self.proc:return
        if self.root:self.scan_files()
        else:self.run('mount',['--mountpoint',self.mountpoint])
    def scan_files(self):
        self.files.clear();self.run('list',['--root',self.root])
    def copy(self):
        paths=[self.files.topLevelItem(i).data(0,Qt.UserRole) for i in range(self.files.topLevelItemCount()) if self.files.topLevelItem(i).checkState(0)==Qt.Checked]
        if not paths:self.status.setText('Select at least one screenshot or recording.');return
        dest=os.path.abspath(os.path.expanduser(self.destination.text().strip()))
        if not self.destination.text().strip():self.status.setText('Enter a folder to save the captures.');return
        self.run('copy',['--root',self.root,'--dest',dest,'--paths-json',json.dumps(paths)])
    def run(self,mode,args):
        self.mode=mode;self.buffer='';self.failed=False;self.pending=[]
        p=QProcess(self);self.proc=p;p.setWorkingDirectory(REPO);p.setProgram(sys.executable);p.setArguments(['-m','app2.transfer.captures',mode]+args)
        p.readyReadStandardOutput.connect(self.read);p.finished.connect(self.finished);p.errorOccurred.connect(self.process_error)
        self.status.setText({'mount':'Connecting to USB File Transfer...','list':'Reading screenshots and recordings...','copy':'Copying selected captures...','unmount':'Releasing USB...'}[mode])
        self.progress.setRange(0,0);self._buttons(True);self.timer.start(60000);p.start()
    def read(self):
        if not self.proc:return
        self.timer.start(60000);self.buffer+=bytes(self.proc.readAllStandardOutput()).decode('utf8','replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:self.worker_event(json.loads(line))
            except (ValueError,TypeError):continue
    def worker_event(self,event):
        kind=event.get('event',event.get('type',event.get('kind')))
        if kind=='mounted':self.root=event['root'];self.receipt=event
        elif kind in ('capture','item'):
            row=event.get('item',event);item=QTreeWidgetItem([row.get('relpath',row['name']),'Screenshot' if row['kind']=='image' else 'Recording',f"{row['size']/1048576:.1f} MB"])
            item.setData(0,Qt.UserRole,row['relpath']);item.setCheckState(0,Qt.Unchecked);self.files.addTopLevelItem(item)
        elif kind=='progress':
            self.progress.setRange(0,100);self.progress.setValue(int(100*event.get('bytes',0)/max(1,event.get('total',1))))
            self.status.setText('Copying '+event.get('relpath','capture'))
        elif kind=='saved':self.pending.append(event['path'])
        elif kind=='error':self.failed=True;self.status.setText(event.get('message',event.get('error','Transfer failed.')))
    def finished(self,code,*args):
        self.read();self.timer.stop();p=self.proc;self.proc=None;mode=self.mode
        stderr=bytes(p.readAllStandardError()).decode('utf8','replace') if p else ''
        if p:p.deleteLater()
        self.progress.setRange(0,100);self._buttons(False)
        if code!=0 or self.failed:
            if not self.failed:self.status.setText('Transfer stopped. '+(stderr.strip()[-300:] or 'Reconnect USB, choose File Transfer, and try again.'))
            self.closing=False;return
        self.progress.setValue(100)
        if mode=='mount':self.scan_files()
        elif mode=='list':self.status.setText(f'{self.files.topLevelItemCount()} captures found. Select the ones to import.')
        elif mode=='copy':self.status.setText(f'Saved {len(self.pending)} captures. Device originals were kept.')
        elif mode=='unmount':self.receipt=None;self.root=None;self._dismissed()
    def process_error(self,error):
        if error==QProcess.FailedToStart:
            self.timer.stop();self.proc.deleteLater();self.proc=None;self._buttons(False);self.status.setText('Could not start the transfer worker.')
    def cancel(self):
        if self.proc:self.failed=True;self.status.setText('Transfer canceled. Completed copies were kept.');self.proc.terminate();p=self.proc;QTimer.singleShot(1500,lambda:self.kill_pending(p))
    def kill_pending(self,p):
        if self.proc is p:p.kill()
    def timed_out(self):
        self.cancel();self.status.setText('The scanner stopped responding. Check USB and File Transfer, then try again.')
    def dismiss(self):
        if self.proc:return
        if self.receipt and self.receipt.get('owned'):
            self.closing=True;self.run('unmount',['--mountpoint',self.receipt['mountpoint'],'--token',self.receipt['token']])
        else:self._dismissed()
    def _dismissed(self):
        if hasattr(self.parentWidget(),"centralWidget"):self.parentWidget().centralWidget().setEnabled(True)
        self.hide();self.dismissed.emit()
