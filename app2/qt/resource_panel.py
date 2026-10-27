# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live resource details and limits for the next processing job."""
import math,time
from PySide6.QtCore import Qt,QTimer,QPointF,QPoint,QRect,QEvent
from PySide6.QtGui import QColor,QPainter,QPainterPath,QPen
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QSlider,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QWidget,QScrollArea
from scanner.stages import resources,runner
from app2.qt.theme import ui_font


def gib(value):return value/(1<<30)


class UsageGraph(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent);self.setMinimumHeight(140);self.series=[]
        self.setAccessibleName('Live resource graph')
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#161a1f'));p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width,height=max(1,self.width()-20),max(1,self.height()-24)
        for values,color in self.series:
            if len(values)<2:continue
            high=max(100.,max(values));path=QPainterPath()
            for i,value in enumerate(values):
                point=QPointF(10+i*width/max(1,len(values)-1),height+10-min(high,max(0,value))*height/high)
                if i:path.lineTo(point)
                else:path.moveTo(point)
            p.setPen(QPen(QColor(color),2));p.drawPath(path)


class ResourcePanel(QDialog):
    def __init__(self,tile,parent=None,sampler=None):
        super().__init__(parent);self.tile=tile;self.sampler=sampler
        self.setWindowTitle(tile+' details');self.resize(510,660)
        self.setObjectName('ResourceDetails_'+tile)
        self.setStyleSheet('QDialog {background:#20252c;color:#e6edf6;} QLabel {color:#e6edf6;}')
        outer=QVBoxLayout(self)
        self.body_scroll=QScrollArea(); self.body_scroll.setWidgetResizable(True); self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_scroll.setFrameShape(QScrollArea.Shape.NoFrame); outer.addWidget(self.body_scroll,1)
        body=QWidget(); layout=QVBoxLayout(body); self.body_scroll.setWidget(body)
        self.setMinimumSize(390,280)
        title=QLabel(tile+' use and limits');title.setFont(ui_font(18));layout.addWidget(title)
        self.current=QLabel();self.current.setWordWrap(True);layout.addWidget(self.current)
        self.accounting=QLabel();self.accounting.setWordWrap(True);layout.addWidget(self.accounting)
        self.graph=UsageGraph(self);layout.addWidget(self.graph)
        self.legend=QLabel('Blue: whole PC. Green: PointYoink. Last two minutes.');self.legend.setWordWrap(True);layout.addWidget(self.legend)
        self.average=QLabel();self.average.setWordWrap(True);layout.addWidget(self.average)
        self.mode=QComboBox();self.mode.addItem('Auto (recommended)','auto');self.mode.addItem('Manual','manual')
        self.mode.setCurrentIndex(0 if resources.limit_options()['mode']=='auto' else 1)
        self.controls=[];self.values={};self.value_labels={}
        if tile!='SCANNER':
            layout.addWidget(QLabel('Build limits'));layout.addWidget(self.mode)
            maximum=max(1,resources.memory_max_gb());cores=max(1,math.floor(resources.cpu_count()*.8))
            settings=resources.load();options=resources.limit_options()
            limits=[('memory','RAM budget (GiB)',maximum,round(settings['memory_gb'])),
                ('threads','CPU threads',cores,options.get('cpu_threads') or max(1,math.floor(resources.cpu_count()*settings['cpu_percent']/100))),
                ('gpu','GPU jobs at once',2,options['gpu_jobs'])]
            for key,name,high,value in limits:
                row=QHBoxLayout();label=QLabel();self.value_labels[key]=label;row.addWidget(label)
                slider=QSlider(Qt.Orientation.Horizontal);slider.setRange(1,high);slider.setValue(value);slider.setObjectName('resource_limit_'+key)
                self.values[key]=slider;slider.valueChanged.connect(lambda number,k=key,n=name:self.value_labels[k].setText(n+': '+str(number)))
                label.setText(name+': '+str(slider.value()));row.addWidget(slider,1);layout.addLayout(row);self.controls.append(slider)
            self.fusion_table=QTableWidget(0,3);self.fusion_table.setHorizontalHeaderLabels(['Fusion job','Planned VRAM','Last peak']);self.fusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.fusion_table.setMaximumHeight(150);self.fusion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.fusion_table.setVisible(tile=='GPU');layout.addWidget(self.fusion_table)
            self.gpu_note=QLabel();self.gpu_note.setWordWrap(True);layout.addWidget(self.gpu_note)
            self.save=QPushButton('Save limits for the next step');self.save.clicked.connect(self.save_limits);layout.addWidget(self.save)
            self.note=QLabel('Running jobs keep their current limits. A lower limit waits for them to finish.');self.note.setWordWrap(True);layout.addWidget(self.note)
            self.mode.currentIndexChanged.connect(self.update_mode);self.update_mode()
        else:
            self.motion_button=QPushButton('Open motion details');self.motion_button.clicked.connect(lambda:getattr(self.parentWidget(),'show_motion',lambda:None)());layout.addWidget(self.motion_button)
            self.mode.hide();self.note=QLabel('Link and motion details appear only while this app is connected.');self.note.setWordWrap(True);layout.addWidget(self.note)
        self.close_button=QPushButton('Close');self.close_button.clicked.connect(self.close);outer.addWidget(self.close_button)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(1000);self.refresh()
        self._fit_gpu_controls()

    def event(self,event):
        if event.type()==QEvent.Type.ShortcutOverride and event.key()==Qt.Key.Key_Escape:
            event.accept();return True
        return super().event(event)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape:
            self.close();event.accept();return
        super().keyPressEvent(event)

    def _fit_gpu_controls(self):
        if self.tile=='SCANNER':return
        from app2.builder.backend_notes import cuda_available
        usable=cuda_available() is not False
        self.values['gpu'].setVisible(usable);self.value_labels['gpu'].setVisible(usable)
        self.fusion_table.setVisible(usable and self.tile=='GPU')
        if not usable:self.gpu_note.setText('Builds run on the CPU.')

    def showEvent(self,event):
        super().showEvent(event); QTimer.singleShot(0,self.fit_window)

    def fit_window(self):
        parent=self.parentWidget()
        if parent is None:return
        available=QRect(parent.mapToGlobal(QPoint(0,0)),parent.size()).adjusted(12,12,-12,-12)
        if self.screen() is not None:available=available.intersected(self.screen().availableGeometry().adjusted(12,12,-12,-12))
        self.resize(min(510,available.width()),min(660,available.height()))
        self.move(available.center()-self.rect().center())

    def update_mode(self):
        automatic=self.mode.currentData()=='auto'
        recommendation=resources.recommended_limits() if automatic else None
        for slider in self.controls:slider.setEnabled(not automatic)
        if recommendation:
            for key,value in [('memory',recommendation['memory_gb']),('threads',recommendation['cpu_threads']),('gpu',recommendation['gpu_jobs'])]:self.values[key].setValue(round(value))
        self.save.setEnabled(not automatic or recommendation is not None)
        if automatic:
            from app2.builder.backend_notes import cuda_available
            self.note.setText('Auto checks available RAM, the desktop reserve, and CPU count before the next step.' if cuda_available() is False else 'Auto checks available RAM, the desktop reserve, CPU count, and free GPU memory before the next step.')
        else:self.note.setText('Saved limits start with the next step. Running jobs keep their current limits.')
        if not runner.scopes_available():self.note.setText(self.note.text()+' This PC has no systemd user session, so builds run without the RAM and CPU caps. Close other heavy apps while building.')

    def save_limits(self):
        try:
            resources.save_limits(self.values['memory'].value(),self.values['threads'].value(),self.values['gpu'].value(),self.mode.currentData())
            self.note.setText('Saved for the next build step. Running jobs keep their current limits.')
        except (OSError,ValueError) as error:self.note.setText('Limits were not saved: '+str(error))

    def refresh(self):
        self._fit_gpu_controls()
        if self.sampler is None:return
        sample=self.sampler.last or {};rows=list(getattr(self.sampler,'samples',[]))
        average=getattr(getattr(self.sampler,'build_usage',None),'average',{})
        if self.tile=='SCANNER':self.refresh_scanner();return
        key={'CPU':'cpu','GPU':'gpu','RAM':'ram_used'}[self.tile]
        if self.tile=='RAM':
            total=sample.get('ram_used',0);app=sample.get('app_ram',0)
            self.current.setText('RAM: %.1f of %.1f GiB used'%(gib(total),gib(sample.get('ram_total',0))))
            self.accounting.setText('PointYoink: %.2f GiB (%s). Rest of PC: %.2f GiB.'%(gib(app),sample.get('app_ram_kind','memory estimate'),gib(max(0,total-app))))
            values=lambda row:100*row.get('ram_used',0)/max(1,row.get('ram_total',1))
            own=lambda row:100*row.get('app_ram',0)/max(1,row.get('ram_total',1))
        elif self.tile=='CPU':
            total=sample.get('cpu',0);app=sample.get('app_cpu',0)
            self.current.setText('CPU: %.1f%% of all logical processors'%total)
            self.accounting.setText('PointYoink and its workers: %.1f%%. Rest of PC: %.1f%%.'%(app,max(0,total-app)))
            values=lambda row:row.get('cpu',0);own=lambda row:row.get('app_cpu',0)
        else:
            self.current.setText('GPU: %.1f%%. Video memory: %.1f of %.1f GiB.'%(sample.get('gpu',0),gib(sample.get('vram_used',0)),gib(sample.get('vram_total',0))))
            app=sample.get('app_vram')
            self.accounting.setText('App GPU use cannot be split reliably. '+('App video memory is unavailable.' if app is None else 'PointYoink video memory: %.2f GiB. Rest: %.2f GiB.'%(gib(app),gib(max(0,sample.get('vram_used',0)-app)))))
            values=lambda row:row.get('gpu',0);own=None
        if key not in sample:
            self.current.setText(self.tile+' information is unavailable.');self.accounting.setText('No current sample was reported.')
        self.graph.series=[([values(row) for row in rows],'#4a9dff')]
        if own:self.graph.series.append(([own(row) for row in rows],'#3fd16f'))
        self.graph.update()
        self.legend.setText('Blue: whole PC. '+('Green: PointYoink. ' if own else '')+'Last two minutes.')
        if average.get('samples',0) and key in average:
            if self.tile=='RAM':text='%.2f GiB used; PointYoink %.2f GiB'%(gib(average.get(key,0)),gib(average.get('app_ram',0)))
            elif self.tile=='CPU':text='whole PC %.1f%%; PointYoink %.1f%%; rest %.1f%%'%(average.get('cpu',0),average.get('app_cpu',0),max(0,average.get('cpu',0)-average.get('app_cpu',0)))
            else:text='%.1f%%; app video memory %s'%(average.get(key,0),'unavailable' if average.get('app_vram') is None else '%.2f GiB'%gib(average['app_vram']))
            self.average.setText('Last build average: '+text+' (%d samples over %.1f s).'%(average['samples'],average.get('seconds',0)))
        else:self.average.setText('Last build average: no build samples yet.')
        from app2.builder.backend_notes import cuda_available
        if cuda_available() is False:return
        from scanner.stages.parallel import GPU_JOB_BYTES,GPU_RESERVE
        free=max(0,sample.get('vram_total',0)-sample.get('vram_used',0)-GPU_RESERVE)
        slots=max(1,min(2,int(free//GPU_JOB_BYTES))) if sample.get('vram_total') else 1
        if self.mode.currentData()=='auto':self.values['gpu'].setValue(slots)
        self.gpu_note.setText('Standard GPU side budget: %.1f GiB per job plus %.1f GiB headroom. %d job%s %s now; Fused can need more.'%(gib(GPU_JOB_BYTES),gib(GPU_RESERVE),slots,'' if slots==1 else 's','fits' if slots==1 else 'fit'))
        if self.tile=='GPU':
            snapshot=getattr(getattr(self.parentWidget(),'build',None),'snap',None)
            details=fusion_memory_rows(snapshot)
            if details!=getattr(self,'_fusion_rows',None):
                self._fusion_rows=details;self.fusion_table.setRowCount(len(details))
                for row,values in enumerate(details):
                    for column,value in enumerate(values):self.fusion_table.setItem(row,column,QTableWidgetItem(value))

    def refresh_scanner(self):
        window=self.parentWidget();session=getattr(window,'session',None);source=getattr(session,'source',None)
        if source is None or not getattr(source,'running',False) or getattr(source,'kind','')=='replay':
            from app2.qt import link_status
            transfer = link_status.state.get('wifi') == 'receiving'
            self.current.setText('Scanner connected over Wi-Fi for file transfer' if transfer else 'Scanner not connected')
            self.accounting.setText('Transfer: %.1f MB/s. No live camera frame rate.'%(link_status.state.get('rate',0)/(1<<20)) if transfer else 'No live link, frame rate, or motion data.')
            self.motion_button.setEnabled(False);self.graph.hide();self.legend.hide();self.average.hide();return
        kind=getattr(source,'kind',type(source).__name__)
        link='Wi-Fi' if 'wifi' in str(kind).lower() else 'USB'
        self.current.setText('Scanner connected over '+link+'\n'+str(getattr(source,'label','')))
        self.motion_button.setEnabled(True)
        now=time.monotonic();previous=getattr(self,'_frame_sample',None);seq=getattr(source,'seq',0)
        fps=(seq-previous[1])/max(.01,now-previous[0]) if previous else None
        self._frame_sample=(now,seq)
        motion=getattr(getattr(window,'motion_panel',None),'client',None)
        status,sample,count,age=motion.snapshot() if motion is not None else ('Disconnected',None,0,None)
        line='Frame rate: measuring...' if fps is None else 'Frame rate: %.1f fps'%max(0,fps)
        if getattr(source,'depth_dev',None):line+='\nDepth stream: '+str(source.depth_dev)
        if sample is not None:line+='\nMotion: '+status+'; angles '+', '.join('%.1f'%v for v in sample.angles)+'; %d packets'%count
        else:line+='\nMotion stream: '+status
        self.accounting.setText(line);self.graph.show();self.legend.show();self.average.hide()
        self._rates=getattr(self,'_rates',[])+([max(0,fps)] if fps is not None else [])
        self._rates=self._rates[-120:];self.graph.series=[(self._rates,'#4a9dff')];self.graph.update();self.legend.setText('Live frames per second. Motion uses the existing connected stream; this panel does not connect it.')

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)


def fusion_memory_rows(snapshot):
    """Latest per-job plans and measured process peaks from recorded fusion node statistics."""
    if snapshot is None: return []
    sides = {side['id']: side.get('n') for side in getattr(snapshot, 'sides', []) or []}
    rows = []
    for group in getattr(snapshot, 'groups', []) or []:
        for node in group.get('nodes', []):
            identity = node.get('id', '')
            if identity != 'fused' and not identity.startswith('side.fuse@'): continue
            stats = node.get('stats') or {}
            planned, peak = stats.get('estimated_peak_bytes'), stats.get('peak_vram_bytes')
            if planned is None and peak is None: continue
            name = 'Fused' if identity == 'fused' else 'Side %s' % (sides.get(identity.split('@', 1)[1]) or node.get('n') or '')
            rows.append([name.strip(), 'Not recorded' if planned is None else '%.2f GiB' % gib(planned), 'Not recorded' if peak is None else '%.2f GiB' % gib(peak)])
    return rows
