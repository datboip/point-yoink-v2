import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
import numpy as np,trimesh
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import QPointF
from app2.qt.edit_view import EditView
import pytest

def test_mesh_points_preserves_surface_depth_and_selection():
    if os.environ.get("QT_QPA_PLATFORM")=="offscreen" or not os.environ.get("DISPLAY"): pytest.skip("Requires private Xvfb GL")
    app=QApplication.instance() or QApplication([]);v=EditView();v.resize(500,500)
    m=trimesh.creation.icosphere(subdivisions=2,radius=10)
    v.show_items([dict(key='m',kind='mesh',V=np.float32(m.vertices),N=np.float32(m.vertex_normals),F=np.uint32(m.faces))],edit='m');v.show();QTest.qWait(300)
    assert v.isValid() and not v.error
    first=v.grabFramebuffer();a=v.world_at(QPointF(250,250));selection=v.selection.copy()
    v.set_mesh_points(True);QTest.qWait(100);second=v.grabFramebuffer();b=v.world_at(QPointF(250,250))
    assert first!=second
    assert np.allclose(a,b,atol=0.001),(a,b)
    assert np.array_equal(selection,v.selection)
    v.set_mesh_points(False);QTest.qWait(50);assert first==v.grabFramebuffer()
    v.close();print('Mesh/Points rendering changes pixels, preserves surface picking and selection, and restores original image.')
