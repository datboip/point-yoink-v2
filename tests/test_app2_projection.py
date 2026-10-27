"""Projection switching preserves the camera and selection still respects occlusion."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import app2,numpy as np,pytest
from app2.gl_core import Camera
from app2.builder.select_core import Elements


def camera():
    cam=Camera();cam.frame(np.array([-10.,-10.,-10.]),np.array([10.,10.,10.]));return cam


def ndc(cam,point):
    p=cam.projection(1.) @ cam.view @ np.append(point,1.)
    return p[:3]/p[3]


def test_toggle_keeps_angle_pivot_and_pivot_plane_scale():
    cam=camera();cam.orbit(21,13);cam.pan(10,7,400)
    before=cam.view.copy();pivot=cam.pivot.copy()
    right=np.linalg.inv(cam.view)[:3,0]
    points=[cam.pivot,cam.pivot+right*5]
    projected=[ndc(cam,p)[:2] for p in points]
    cam.set_projection_mode('orthographic')
    assert np.array_equal(cam.view,before) and np.array_equal(cam.pivot,pivot)
    assert np.allclose(projected,[ndc(cam,p)[:2] for p in points])
    cam.set_projection_mode('perspective');assert np.array_equal(cam.view,before)
    with pytest.raises(ValueError):cam.set_projection_mode('bad')


@pytest.mark.parametrize('mode',['perspective','orthographic'])
def test_project_unproject_and_zoom(mode):
    cam=camera();cam.set_projection_mode(mode);point=np.array([3.,2.,-4.])
    projected=ndc(cam,point)
    recovered=cam.unproject((projected[0]+1)*200-.5,(1-projected[1])*200-.5,(projected[2]+1)/2,400,400)
    assert np.allclose(recovered,point)
    span=abs(ndc(cam,[5,0,0])[0]-ndc(cam,[0,0,0])[0])
    cam.zoom(1)
    assert abs(ndc(cam,[5,0,0])[0]-ndc(cam,[0,0,0])[0])>span


def test_orthographic_parallel_edges_keep_equal_sizes_at_depth():
    cam=camera();cam.set_projection_mode('orthographic')
    assert ndc(cam,[5,0,0])[0]==pytest.approx(ndc(cam,[5,0,-8])[0])
    cam.set_projection_mode('perspective')
    assert ndc(cam,[5,0,0])[0]>ndc(cam,[5,0,-8])[0]


@pytest.mark.parametrize('mode',['perspective','orthographic'])
def test_visible_selection_rejects_hidden_and_behind_camera(mode):
    cam=camera();cam.set_projection_mode(mode);projection=cam.projection(1.)
    points=np.array([[0.,0,0],[0,0,-8],[0,0,1000.]])
    elements=Elements(points,np.tile([0.,0,1],(3,1)))
    depth=np.full((100,100),(ndc(cam,[0,0,0])[2]+1)/2,np.float32)
    got=elements.select_screen(np.ones((100,100),bool),cam.view,projection,cam.near,cam.far,depth)
    assert got.tolist()==[True,False,False]
    assert cam.select(points,np.ones((100,100),bool),depth,.1).tolist()==[0]


def test_display_menu_changes_both_compare_projections():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import BuildViewPanel
    app=QApplication.instance() or QApplication([])
    panel=BuildViewPanel(True);panel.set_colours([('shaded','Shaded'),('flat','Flat facets')],'shaded')
    action=next(a for a in panel.colour_menu.actions() if a.data()=='projection:orthographic')
    action.trigger()
    assert panel.main.view.cam.projection_mode=='orthographic'
    assert panel.before.view.cam.projection_mode=='orthographic'
    panel.set_colours([('shaded','Shaded'),('flat','Flat facets')],'flat')
    assert action.isChecked()
    panel.deleteLater();app.processEvents()
