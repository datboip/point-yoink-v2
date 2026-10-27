"""Project previews and in-window picking use only temporary project/config paths."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_picker_cfg_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['XMODIFIERS']='@im=none'
import io
import json
import pytest
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
from app2.project_preview import project_rows, preview_bytes, PREVIEW_NAME
from app2.qt.project_picker import ProjectPicker, save_scene_preview


def project(root,name,stamp):
    path=root/name; (path/'data'/'side1').mkdir(parents=True)
    os.utime(path/'data'/'side1',(stamp,stamp))
    return path


def test_projects_sorted_without_preview_mtime_changing_order(tmp_path):
    older=project(tmp_path,'old',1000);newer=project(tmp_path,'new',2000)
    (tmp_path/'not-a-project').mkdir()
    (older/'.pointyoink-home.json').write_text(json.dumps({'name':'Friendly name'}))
    Image.new('RGB',(64,48),'red').save(older/PREVIEW_NAME)
    rows=project_rows(tmp_path)
    assert [r['folder'] for r in rows]==['new','old']
    assert rows[1]['name']=='Friendly name' and rows[1]['sides']==1
    assert project_rows(newer)[0]['path']==str(newer)


def test_existing_preview_is_readable_and_not_changed(tmp_path):
    p=project(tmp_path,'project',1000)
    Image.new('RGB',(1000,600),'red').save(p/PREVIEW_NAME)
    before=(p/PREVIEW_NAME).read_bytes()
    with Image.open(io.BytesIO(preview_bytes(project_rows(tmp_path)[0]))) as im:
        assert im.width<=560 and im.height<=320
    assert (p/PREVIEW_NAME).read_bytes()==before


def test_missing_image_is_rendered_and_cached_without_changing_mesh(tmp_path):
    import numpy as np
    from stages import plyio
    p=project(tmp_path,'mesh',1000);mesh=p/'model.ply'
    plyio.write_mesh(str(mesh),np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
    before=mesh.read_bytes();raw=preview_bytes(project_rows(tmp_path)[0])
    assert raw.startswith(b'\x89PNG') and (p/PREVIEW_NAME).read_bytes()==raw
    assert mesh.read_bytes()==before


def test_scene_capture_saves_standard_png(tmp_path):
    p=project(tmp_path,'capture',1000)
    class View:
        def isVisible(self):return True
        def grabFramebuffer(self):
            image=QImage(900,600,QImage.Format.Format_RGB32);image.fill(0xff334455);return image
    assert save_scene_preview(p,View())
    with Image.open(p/PREVIEW_NAME) as im:assert im.format=='PNG' and im.width<=560
    assert not save_scene_preview(tmp_path/'missing',View())


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_picker_stays_in_window_and_returns_selection(tmp_path,size):
    app=QApplication.instance() or QApplication([])
    from app2.qt import theme
    theme.install(app)
    for i in range(7):
        p=project(tmp_path,f'Project {i}',1000+i)
        Image.new('RGB',(120,80),'gray').save(p/'model.png')
    win=QMainWindow();win.setCentralWidget(QWidget());win.resize(*size);win.show()
    picker=ProjectPicker(win,tmp_path);picker.show();app.processEvents()
    assert not picker.isWindow() and win.rect().contains(picker.sheet.geometry())
    assert picker.grid.itemAtPosition(0,2).widget().geometry().right() < picker.scroll.viewport().width()
    card_sizes=[picker.grid.itemAt(i).widget().size() for i in range(picker.grid.count())]
    picker.search.setText('Project 2');app.processEvents()
    assert picker.grid.count()==1 and picker.grid.itemAt(0).widget().size()==card_sizes[0]
    chosen=str(tmp_path/'Project 2')
    QTimer.singleShot(20,lambda:picker.accept(chosen))
    assert picker.run()==chosen and win.centralWidget().isEnabled()
    picker.deleteLater();win.close();win.deleteLater();app.processEvents()


def test_shell_open_uses_picker_and_escape_keeps_workspace(tmp_path):
    app=QApplication.instance() or QApplication([])
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from app2.qt.shell import MainWindow
    project(tmp_path,'Project A',1000)
    def offline(*args,**kwargs):raise RuntimeError('No scanner in test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path));win.show_step(1);win.show();win.activateWindow();app.processEvents()
    original=win.stack.currentIndex();observed=[]
    def escape():
        observed.append(win.project_picker is not None and win.project_picker.isVisible())
        QTest.keyClick(win,Qt.Key.Key_Escape)
    QTimer.singleShot(40,escape)
    QTimer.singleShot(1500,lambda:win.project_picker.reject() if win.project_picker is not None else None)
    assert win._pick_folder() is None
    assert observed==[True] and win.project_picker is None
    assert win.stack.currentIndex()==original and win.centralWidget().isEnabled()
    win.build.shutdown();win.close();win.deleteLater();app.processEvents()


def test_workspace_project_card_closes_current_session_before_open(tmp_path):
    from types import SimpleNamespace
    from app2.qt.shell import MainWindow
    from app2.scan.ui_model import ScanState
    app=QApplication.instance() or QApplication([])
    selected=project(tmp_path,'Project A',1000)
    def offline(*args,**kwargs):raise RuntimeError('No scanner in test')
    win=MainWindow(offline,gl=False,projects_dir=str(tmp_path))
    closed=[]
    win.session=SimpleNamespace(close=lambda **kwargs:closed.append(kwargs))
    win.state=ScanState(project=str(tmp_path/'Other'),source_kind='replay',source_label='Replay')
    win.show();win.show_workspace();app.processEvents()
    try:
        card=win.workspace.findChild(QPushButton,'WorkspaceProjectOpen')
        assert card is not None
        card.click()
        assert closed==[{'then':('open',str(selected))}]
        assert win.stack.currentWidget() is win.build_slot          # it has a side: it lands on Build
        win._closing=False;win.state=ScanState(project=str(selected),source_kind='replay',source_label='Replay')
        win.show_workspace();card=win.workspace.findChild(QPushButton,'WorkspaceProjectOpen');card.click()
        assert len(closed)==1 and win.stack.currentWidget() is win.build_slot   # already open: the session stays
    finally:
        win._really=True;win.build.shutdown();win.close();win.deleteLater();app.processEvents()


def test_regenerate_replaces_report_or_old_preview(tmp_path):
    import numpy as np
    from stages import plyio
    p=project(tmp_path,'Project12345',1000)
    mesh=p/'model.ply';plyio.write_mesh(str(mesh),np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
    Image.new('RGB',(200,200),'red').save(p/'model.png')
    Image.new('RGB',(200,200),'blue').save(p/PREVIEW_NAME)
    before=mesh.read_bytes();old=(p/PREVIEW_NAME).read_bytes()
    row=project_rows(tmp_path)[0];raw=preview_bytes(row,force=True)
    assert raw != old and (p/PREVIEW_NAME).read_bytes()==raw
    assert mesh.read_bytes()==before
    (p/PREVIEW_NAME).unlink()
    assert preview_bytes(project_rows(tmp_path)[0]) == raw


def test_generate_without_model_keeps_existing_picture(tmp_path):
    p=project(tmp_path,'Project12345',1000)
    Image.new('RGB',(100,100),'blue').save(p/PREVIEW_NAME)
    old=(p/PREVIEW_NAME).read_bytes()
    with pytest.raises(ValueError,match='Build a model'):
        preview_bytes(project_rows(tmp_path)[0],force=True)
    assert (p/PREVIEW_NAME).read_bytes()==old


def test_friendly_name_preserves_folder_and_metadata(tmp_path):
    from app2.project_preview import rename_project,project_name,META_NAME
    p=project(tmp_path,'ProjectLive12345',1000)
    meta={'versions':{'mesh':{'reviewed':True}},'current':{'combined':'abc'}}
    (p/META_NAME).write_text(json.dumps(meta))
    assert project_rows(tmp_path)[0]['unnamed']
    assert rename_project(p,'  Chair mount  ') == 'Chair mount'
    actual=json.loads((p/META_NAME).read_text());assert actual==dict(meta,name='Chair mount')
    assert project_name(p)=='Chair mount' and (p/'data'/'side1').is_dir()
    row=project_rows(tmp_path)[0];assert not row['unnamed'] and row['folder']=='ProjectLive12345'
    with pytest.raises(ValueError):rename_project(p,' ')
    assert project_name(p)=='Chair mount'


def test_corrupt_metadata_is_not_replaced_by_rename(tmp_path):
    from app2.project_preview import rename_project,META_NAME
    p=project(tmp_path,'Project12345',1000);f=p/META_NAME;f.write_text('{broken')
    with pytest.raises(ValueError):rename_project(p,'Mount')
    assert f.read_text()=='{broken'


def test_card_rename_keeps_fixed_card_and_filters_by_name(tmp_path):
    app=QApplication.instance() or QApplication([])
    p=project(tmp_path,'Project12345',1000)
    win=QMainWindow();win.setCentralWidget(QWidget());win.resize(1280,800);win.show()
    picker=ProjectPicker(win,tmp_path);picker.show();app.processEvents()
    before=picker.grid.itemAt(0).widget().geometry();path=str(p)
    picker.edit_name(path);picker.editors[path][1].setText('Seat mount');picker.save_name(path);app.processEvents()
    assert picker.grid.itemAt(0).widget().geometry()==before
    picker.search.setText('seat');assert picker.grid.count()==1
    assert picker.editors[path][2].text()=='Seat mount'
    picker.pool.shutdown(wait=True,cancel_futures=True);picker.hide();win.close();win.deleteLater();app.processEvents()


def test_header_name_saves_on_enter_and_is_not_reset_while_typing(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from types import SimpleNamespace
    from app2.qt.shell import MainWindow
    from app2.project_preview import project_name
    app=QApplication.instance() or QApplication([])
    p=project(tmp_path,'Project12345',1000)
    def offline(*args,**kwargs):raise RuntimeError('No scanner in test')
    from app2.scan.ui_model import ScanState
    win=MainWindow(offline,gl=False);win.state=ScanState(project=str(p));win.show()
    win.show_step(0)
    win.header.set_project('Sep 28, 2026',p.name,'')
    win.activateWindow();win.header.folder.setFocus();app.processEvents()
    QTest.keyClicks(win.header.folder,'Pump bracket')
    win.header.set_project('Sep 28, 2026',p.name,'')
    assert win.header.folder.text()=='Pump bracket' and not project_name(p)
    QTest.keyClick(win.header.folder,Qt.Key.Key_Return)
    assert project_name(p)=='Pump bracket'
    win.build.shutdown();win.close();win.deleteLater();app.processEvents()
