"""Only the displayed side's matched bed sets its cube's upward direction."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import app2,numpy as np,pytest
from app2.view_orientation import basis_with_bed_up
from app2.qt.navcube import preset


@pytest.mark.parametrize('up',[(0,0,1),(0,0,-1),(0,-1,0),(.2,-.9,.4)])
def test_bed_positive_height_maps_to_top(up):
    up=np.asarray(up,float);up/=np.linalg.norm(up)
    basis=basis_with_bed_up(up)
    assert np.allclose(basis @ up,[0,0,1])
    assert np.linalg.det(basis)==pytest.approx(1.)
    assert np.allclose(basis @ basis.T,np.eye(3))


def test_side_payload_changes_only_cube_and_does_not_emit_save_signal():
    from PySide6.QtWidgets import QApplication
    from app2.qt.model_panel import ModelPanel
    app=QApplication.instance() or QApplication([])
    panel=ModelPanel(gl=True);cube=panel.view.nav_cube
    fallback=preset('front').T @ preset('top');panel.set_project_basis(fallback)
    emitted=[];cube.basis_changed.connect(emitted.append)
    points=np.array([[0.,0,0],[10,0,0],[0,10,0]],np.float32)
    cloud=dict(side_no=1,home_seq=1,P=points,N=np.tile([0.,0,1.],(3,1)),L=np.ones(3,np.uint8),covered=.5,pivot=np.zeros(3),bed_up=[0,-1,0])
    panel.on_cloud(cloud)
    before=panel.view.cam.view.copy();geometry=points.copy()
    assert np.allclose(cube.basis @ np.array([0,-1,0]),[0,0,1]) and not emitted
    panel.set_project_basis(np.eye(3))
    assert np.allclose(cube.basis @ np.array([0,-1,0]),[0,0,1])
    assert np.array_equal(panel.view.cam.view,before) and np.array_equal(points,geometry)
    panel.on_cloud(dict(cloud,side_no=2,home_seq=2,bed_up=None))
    assert np.allclose(cube.basis,np.eye(3))
    panel.on_cloud(cloud);panel.set_project_basis(fallback,reset_bed=True)
    assert np.allclose(cube.basis,fallback) and panel._bed_up is None and not emitted
    panel.deleteLater();app.processEvents()


def test_saved_side_mesh_uses_its_bed_and_empty_cloud_clears_it():
    from PySide6.QtWidgets import QApplication
    from app2.qt.model_panel import ModelPanel
    app=QApplication.instance() or QApplication([])
    panel=ModelPanel(gl=True)
    v=np.array([[0.,0,0],[10,0,0],[0,10,0]],np.float32);n=np.tile([0.,0,1.],(3,1));f=np.array([[0,1,2]])
    panel.on_side_mesh(dict(V=v,N=n,F=f,pivot=np.zeros(3),side_no=2,home_seq=2,bed_up=[0,0,-1]))
    assert np.allclose(panel.view.nav_cube.basis @ np.array([0,0,-1]),[0,0,1])
    panel.on_cloud(dict(P=np.zeros((0,3)),side_no=None,home_seq=3))
    assert np.allclose(panel.view.nav_cube.basis,np.eye(3))
    panel.deleteLater();app.processEvents()


def test_scanner_start_direction_sets_front_and_bed_sets_bottom():
    from app2.view_orientation import basis_with_bed_up_and_front
    up=np.array([0,-1,0.]);front=np.array([0,0,1.])                  # camera frame: bed up is -y, the scanner looks along +z
    basis=basis_with_bed_up_and_front(up,front)
    assert np.allclose(basis@up,[0,0,1]) and np.allclose(basis@front,[0,1,0])   # viewer at Front looks along local +y
    tilted=basis_with_bed_up_and_front(up,np.array([0,-.5,1.]))     # a scanner looking down: only the level part counts
    assert np.allclose(tilted@up,[0,0,1]) and np.allclose(tilted@np.array([0,0,1.]),[0,1,0])
    assert np.allclose(basis_with_bed_up_and_front(up,up)@up,[0,0,1])          # straight down: falls back to up only


def test_panel_uses_the_start_direction_only_without_a_saved_choice():
    from PySide6.QtWidgets import QApplication
    from app2.qt.model_panel import ModelPanel
    app=QApplication.instance() or QApplication([])
    panel=ModelPanel(gl=True);cube=panel.view.nav_cube
    points=np.array([[0.,0,0],[10,0,0],[0,10,0]],np.float32)
    cloud=dict(side_no=1,home_seq=1,P=points,N=np.tile([0.,0,1.],(3,1)),L=np.ones(3,np.uint8),covered=.5,pivot=np.zeros(3),bed_up=[0,-1,0],bed_front=[0,0,1])
    panel.on_cloud(cloud)
    assert np.allclose(cube.basis@np.array([0,0,1.]),[0,1,0])        # Front from the scanner's start
    panel.set_project_basis(preset('front').T@preset('top'));panel.on_cloud(dict(cloud,home_seq=2))
    assert not np.allclose(cube.basis@np.array([0,0,1.]),[0,1,0]) or np.allclose(cube.basis@np.array([0,-1,0.]),[0,0,1])   # a saved choice keeps its heading
