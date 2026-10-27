"""Viewer labels can change without rotating project geometry or camera position."""
import os,json,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import app2,numpy as np,pytest
from PySide6.QtWidgets import QApplication,QWidget
from app2.gl_core import Camera
from app2.qt.navcube import NavCube,preset
from app2.view_orientation import load_orientation,save_orientation,validate_basis


@pytest.mark.parametrize('face',['front','top'])
def test_reference_matches_current_camera_and_named_snap(face):
    app=QApplication.instance() or QApplication([])
    view=QWidget();view.cam=Camera();view.cam.frame(np.zeros(3),np.ones(3)*20)
    view.cam.orbit(48,-32);cube=NavCube(view);before=view.cam.view.copy();pivot=view.cam.pivot.copy();signals=[]
    cube.basis_changed.connect(signals.append);cube.set_reference(face)
    assert len(signals)==1 and np.allclose(view.cam.view,before)
    assert np.allclose(cube.rot(),preset(face))
    view.cam.orbit(12,35);cube.go_face(face)
    assert np.allclose(view.cam.view[:3,:3],before[:3,:3])
    assert np.allclose(view.cam.pivot,pivot)
    other=QWidget();other.cam=Camera();othercube=NavCube(other);othercube.set_basis(cube.basis)
    othercube.go_face('top');cube.go_face('top')
    assert np.allclose(other.cam.view[:3,:3],view.cam.view[:3,:3])
    view.deleteLater();other.deleteLater();app.processEvents()


def test_orientation_persists_and_preserves_project_metadata(tmp_path):
    metadata={'name':'Mount','versions':[{'id':'a','reviewed':True}]}
    path=tmp_path/'.pointyoink-home.json';path.write_text(json.dumps(metadata))
    mesh=tmp_path/'model.ply';mesh.write_bytes(b'unchanged geometry')
    basis=preset('front').T @ preset('top');save_orientation(tmp_path,basis)
    saved=json.loads(path.read_text())
    assert saved['name']==metadata['name'] and saved['versions']==metadata['versions']
    assert np.allclose(load_orientation(tmp_path),basis)
    assert mesh.read_bytes()==b'unchanged geometry'
    path.write_text('broken metadata')
    with pytest.raises(ValueError):save_orientation(tmp_path,basis)
    assert path.read_text()=='broken metadata' and np.allclose(load_orientation(tmp_path),np.eye(3))


def test_invalid_basis_is_refused():
    for value in (np.zeros((3,3)),np.eye(4),np.diag([-1,1,1]),[[float('nan')]*3]*3):
        with pytest.raises(ValueError):validate_basis(value)


def test_build_cube_is_not_lifted_by_the_notice_bar():
    """The notice and the rebuild bar float at the top of the view now: the cube stays put whether or not they show."""
    from app2.qt.build_view import BuildViewPanel
    app=QApplication.instance() or QApplication([])
    panel=BuildViewPanel(True);panel.resize(1000,800);panel.area.resize(1000,700)
    panel.area.layout().activate();panel._place()
    base=panel.main.view.nav_cube.bottom_inset      # the view strip along the bottom lifts it a fixed amount; the notice adds nothing
    panel.notice.show_text('Saved model');panel._place()
    assert panel.main.view.nav_cube.bottom_inset==base
    panel.notice.show_text(None);panel._place()
    assert panel.main.view.nav_cube.bottom_inset==base
    panel.deleteLater();app.processEvents()
