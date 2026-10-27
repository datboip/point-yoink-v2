# SPDX-License-Identifier: AGPL-3.0-or-later
"""An in-window project picker with fixed cards and asynchronous model previews."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QEventLoop, QSaveFile, QIODevice
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QComboBox, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication, QStackedWidget
from app2.project_preview import project_rows, preview_bytes, PREVIEW_NAME, rename_project
from app2.qt.widgets import button, ElidedLabel


def save_scene_preview(project, view):
    """Save the current rendered scene as a standard PNG without changing model files."""
    if not project or not Path(project).is_dir() or not view.isVisible(): return False
    image = view.grabFramebuffer()
    if image.isNull(): return False
    image = image.scaled(560, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    file = QSaveFile(str(Path(project) / PREVIEW_NAME))
    if not file.open(QIODevice.OpenModeFlag.WriteOnly): return False
    if not image.save(file, 'PNG'): file.cancelWriting(); return False
    return file.commit()


class ProjectPicker(QFrame):
    finished = Signal()
    loaded = Signal(object)

    def __init__(self, parent, root):
        super().__init__(parent)
        self.setObjectName('ProjectPickerOverlay')
        self.setStyleSheet('#ProjectPickerOverlay {background:rgba(5,7,10,210);} #ProjectPickerSheet {background:#20252c;border:1px solid #3b424b;border-radius:10px;}')
        self.result = None; self.generation = 0; self.rows = []; self.labels = {}; self.preview_buttons = {}; self.editors = {}; self.batch = set(); self.batch_total = 0; self.batch_failed = 0; self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='project-preview'); self.futures = []
        self.loaded.connect(self.on_loaded)
        self.sheet = QFrame(self); self.sheet.setObjectName('ProjectPickerSheet')
        v = QVBoxLayout(self.sheet); v.setContentsMargins(20,20,20,20); v.setSpacing(12)
        title = QLabel('Open project'); title.setStyleSheet('font-size:20px;font-weight:600;'); title.setFixedHeight(30); v.addWidget(title)
        folder = QHBoxLayout(); self.folder = QLineEdit(str(root)); self.folder.setFixedHeight(36)
        refresh = button('Load folder', 'act'); refresh.setFixedSize(116,36); folder.addWidget(self.folder,1); folder.addWidget(refresh); v.addLayout(folder)
        refresh.clicked.connect(self.refresh); self.folder.returnPressed.connect(self.refresh)
        filters = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText('Search projects'); self.search.setFixedHeight(36)
        self.sort = QComboBox(); self.sort.addItems(['Newest first','Name']); self.sort.setFixedSize(150,36)
        self.generate = button('Generate previews', 'act'); self.generate.setFixedSize(162,36); self.generate.clicked.connect(self.generate_all)
        filters.addWidget(self.search,1); filters.addWidget(self.generate); filters.addWidget(self.sort); v.addLayout(filters)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards = QWidget(); self.grid = QGridLayout(self.cards); self.grid.setContentsMargins(0,0,8,0); self.grid.setSpacing(12)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.cards); v.addWidget(self.scroll,1)
        bottom = QHBoxLayout(); self.status = ElidedLabel(); self.status.setFixedHeight(36)
        self.cancel = button('Cancel','act'); self.cancel.setFixedSize(112,36); self.cancel.clicked.connect(self.reject)
        bottom.addWidget(self.status,1); bottom.addWidget(self.cancel); v.addLayout(bottom)
        self.search.textChanged.connect(self.render); self.sort.currentIndexChanged.connect(self.render)
        self.relayout(); self.refresh()

    def relayout(self):
        self.setGeometry(self.parentWidget().rect())
        self.sheet.setFixedSize(min(980,self.width()-24),min(680,self.height()-24))
        self.sheet.move((self.width()-self.sheet.width())//2,(self.height()-self.sheet.height())//2)

    def refresh(self):
        try:
            self.rows = project_rows(self.folder.text()); self.render()
        except OSError:
            self.rows = []; self.render(); self.status.setText('Could not read that folder.')

    def render(self, *_):
        self.generation += 1; generation = self.generation; self.labels = {}; self.preview_buttons = {}; self.editors = {}; self.batch = set()
        for future in self.futures: future.cancel()
        self.futures = []
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget(): item.widget().hide(); item.widget().deleteLater()
        query = self.search.text().strip().casefold()
        rows = [r for r in self.rows if query in (r['name']+' '+r['folder']).casefold()]
        if self.sort.currentIndex(): rows.sort(key=lambda r:r['name'].casefold())
        self.visible_rows = rows
        self.generate.setEnabled(bool(rows))
        for i, row in enumerate(rows):
            card = QFrame(); card.setFixedSize(296,260); card.setProperty('role','panel')
            layout = QVBoxLayout(card); layout.setContentsMargins(10,10,10,10); layout.setSpacing(4)
            pic = QLabel('Loading preview...'); pic.setFixedSize(276,136); pic.setAlignment(Qt.AlignmentFlag.AlignCenter); pic.setStyleSheet('background:#101318;border-radius:5px;font-size:12px;')
            layout.addWidget(pic)
            name_slot = QStackedWidget(); name_slot.setFixedHeight(32)
            label = ElidedLabel(); label.setText('Unnamed scan' if row['unnamed'] else row['name']); label.setToolTip(row['name'] + '\n' + row['folder'])
            name_slot.addWidget(label)
            edit_row = QWidget(); edit_layout = QHBoxLayout(edit_row); edit_layout.setContentsMargins(0,0,0,0); edit_layout.setSpacing(4)
            name = QLineEdit(); name.setMaxLength(100); name.setPlaceholderText('e.g. Chair mount'); name.setFixedHeight(30)
            save = button('Save','apply'); save.setFixedSize(58,30)
            edit_layout.addWidget(name,1); edit_layout.addWidget(save); name_slot.addWidget(edit_row); layout.addWidget(name_slot)
            detail = QLabel(datetime.fromtimestamp(row['date']).strftime('%b %d, %Y · %H:%M')+' · '+str(row['sides'])+(' side' if row['sides']==1 else ' sides')); detail.setFixedHeight(18); layout.addWidget(detail)
            actions = QHBoxLayout(); actions.setSpacing(4)
            open_button = button('Open','apply'); open_button.setFixedSize(76,30); open_button.clicked.connect(lambda _=False,path=row['path']:self.accept(path))
            rename = button('Name it' if row['unnamed'] else 'Rename','act'); rename.setFixedSize(82,30)
            preview = button('Preview','act','Generate a fresh single-view thumbnail'); preview.setFixedSize(110,30)
            for action in (open_button,rename,preview): actions.addWidget(action)
            layout.addLayout(actions); layout.addStretch(1)
            self.editors[row['path']] = (name_slot,name,label,rename,row)
            rename.clicked.connect(lambda _=False,path=row['path']:self.edit_name(path))
            save.clicked.connect(lambda _=False,path=row['path']:self.save_name(path)); name.returnPressed.connect(lambda path=row['path']:self.save_name(path))
            preview.clicked.connect(lambda _=False,r=row:self.generate_one(r))
            self.grid.addWidget(card,i//3,i%3); self.labels[row['path']] = pic; self.preview_buttons[row['path']] = preview
            self.request_preview(row)
        unnamed = sum(r['unnamed'] for r in rows)
        note = (' · Name your scans to find them later' if unnamed else '')
        self.status.setText(str(len(rows))+' projects'+note if rows else ('No matches.' if self.rows else 'No scan projects in this folder.'))

    def edit_name(self, path):
        slot,name,label,rename,row = self.editors[path]
        name.setText('' if row['unnamed'] else row['name']); slot.setCurrentIndex(1); name.setFocus(); name.selectAll()
        self.status.setText('Give this scan a short name, like Chair mount.')

    def save_name(self, path):
        slot,name,label,rename,row = self.editors[path]
        try: value = rename_project(path, name.text())
        except (OSError, ValueError): self.status.setText('Could not save the name. Enter a short name and try again.'); return
        row['name'] = value; row['unnamed'] = False; label.setText(value); label.setToolTip(value+'\n'+row['folder'])
        rename.setText('Rename'); slot.setCurrentIndex(0); self.status.setText('Saved name: '+value)

    def generate_one(self, row):
        if row['path'] in self.batch: return
        if not self.batch: self.batch_total = 0; self.batch_failed = 0
        self.batch.add(row['path']); self.batch_total += 1; self.generate.setEnabled(False); self.request_preview(row,True)
        self.status.setText('Generating preview...')

    def generate_all(self):
        if self.batch: return
        self.generate.setEnabled(False)
        self.batch = {r['path'] for r in self.visible_rows}; self.batch_total = len(self.batch); self.batch_failed = 0
        for row in self.visible_rows: self.request_preview(row,True)
        self.status.setText('Generating previews · 0%')

    def request_preview(self, row, force=False):
        if force: self.preview_buttons[row['path']].setEnabled(False); self.preview_buttons[row['path']].setText('Working...')
        future = self.pool.submit(preview_bytes,row,force); self.futures.append(future)
        future.add_done_callback(lambda f,g=self.generation,path=row['path'],force=force:self.deliver(f,g,path,force))


    def deliver(self, future, generation, path, force=False):
        try: data = future.result(); error = ''
        except Exception as e: data = None; error = str(e)
        try: self.loaded.emit((generation,path,data,error,force))
        except RuntimeError: pass

    def on_loaded(self, result):
        generation,path,data,error,force = result
        if generation != self.generation or path not in self.labels: return
        label = self.labels[path]; image = QImage.fromData(data) if data else QImage()
        if image.isNull():
            if not force: label.setText('Build a model for a preview')
        else:
            label.setPixmap(QPixmap.fromImage(image).scaled(label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        if force:
            button = self.preview_buttons[path]; button.setText('Preview'); button.setEnabled(True)
            self.batch.discard(path)
            self.generate.setEnabled(not self.batch)
            if image.isNull(): self.batch_failed += 1
            if self.batch:
                done = self.batch_total-len(self.batch); self.status.setText('Generating previews · %d%%' % (100*done//max(1,self.batch_total)))
            elif error: self.status.setText(error if isinstance(error,str) and len(error)<85 else 'Could not save this preview.')
            elif self.batch_failed: self.status.setText('Saved previews. %d projects need a built model or a smaller preview.' % self.batch_failed)
            else: self.status.setText('Saved previews')


    def accept(self, path):
        if not (Path(path)/'data').is_dir(): self.status.setText('That project is no longer available.'); return
        self.result = path; self.finished.emit()

    def reject(self):
        self.result = None; self.finished.emit()

    def run(self):
        loop = QEventLoop(self); self.finished.connect(loop.quit)
        app = QApplication.instance(); app.aboutToQuit.connect(loop.quit)
        central = self.parentWidget().centralWidget(); enabled = central.isEnabled(); central.setEnabled(False)
        self.show(); self.raise_(); self.search.setFocus()
        try: loop.exec()
        finally:
            self.generation += 1; self.pool.shutdown(wait=False,cancel_futures=True)
            self.hide(); central.setEnabled(enabled); app.aboutToQuit.disconnect(loop.quit)
        return self.result
