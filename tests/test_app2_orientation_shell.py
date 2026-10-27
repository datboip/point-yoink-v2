"""Project orientation lifecycle through the real shell, without showing GL windows."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-orientation-shell-'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['XMODIFIERS']='@im=none'
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt.shell import MainWindow
from app2.qt.navcube import preset
from app2.scan.ui_model import ScanState
from app2.view_orientation import load_orientation, save_orientation


@pytest.fixture
def shell(tmp_path):
    app=QApplication.instance() or QApplication([])
    def offline(*args,**kwargs): raise RuntimeError('No scanner in orientation test')
    window=MainWindow(offline,gl=True,projects_dir=str(tmp_path))
    assert not window.isVisible()
    yield window
    window._really=True; window.build.shutdown(); window.close(); window.deleteLater(); app.processEvents()


def state(project):
    return ScanState(project=str(project),source_kind='replay',source_label='Replay',light_remaining=None)


def test_cube_signal_saves_friendly_metadata_and_syncs_all_views(shell,tmp_path):
    project=tmp_path/'mount'; project.mkdir()
    metadata={'name':'Chair mount','versions':[{'id':'kept','reviewed':True}]}
    path=project/'.pointyoink-home.json'; path.write_text(json.dumps(metadata))
    model=project/'model.ply'; model.write_bytes(b'Geometry must not change')
    shell.on_state(state(project))
    assert len(shell._view_cubes)==3
    shell.show_step(1)
    assert shell._menu_states()['front'] and shell._menu_states()['top']
    source=shell.build.vp.main.view.nav_cube
    source.view.cam.orbit(31,-19)
    source.set_reference('top')
    saved=json.loads(path.read_text())
    assert saved['name']=='Chair mount' and saved['versions']==metadata['versions']
    assert np.allclose(load_orientation(project),source.basis)
    for cube in shell._view_cubes: assert np.allclose(cube.basis,source.basis)
    assert model.read_bytes()==b'Geometry must not change'


def test_project_changes_reload_saved_basis_in_all_three_cubes(shell,tmp_path):
    first=tmp_path/'first'; second=tmp_path/'second'; first.mkdir(); second.mkdir()
    first_basis=preset('front').T @ preset('top')
    second_basis=preset('front').T @ preset('left')
    save_orientation(first,first_basis); save_orientation(second,second_basis)
    for project,expected in ((first,first_basis),(second,second_basis),(first,first_basis)):
        shell.on_state(state(project))
        assert shell._orientation_project==str(project)
        for cube in shell._view_cubes: assert np.allclose(cube.basis,expected)


def test_bad_metadata_is_not_overwritten_by_cube_signal(shell,tmp_path):
    project=tmp_path/'broken'; project.mkdir()
    path=project/'.pointyoink-home.json'; path.write_text('broken metadata')
    shell.on_state(state(project))
    cube=shell.build.vp.main.view.nav_cube
    cube.view.cam.orbit(32,18); cube.set_reference('front')
    assert path.read_text()=='broken metadata'
    for current in shell._view_cubes: assert np.allclose(current.basis,np.eye(3))
