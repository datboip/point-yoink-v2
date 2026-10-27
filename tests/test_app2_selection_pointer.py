"""Real GL selection gestures on a temporary synthetic mesh, run on private Xvfb."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_pointer_'))/'config.json')
import numpy as np
import pytest
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app2.qt.edit_view import EditView


def test_tool_pointer_changes_and_brush_hover():
    app=QApplication.instance() or QApplication([])
    view=EditView()
    for tool in ('rect','lasso','brush','magic'):
        view.set_tool(tool);assert view.cursor().shape()==Qt.CursorShape.CrossCursor
    view.set_tool('orbit');assert view.cursor().shape()==Qt.CursorShape.OpenHandCursor
    assert view.hasMouseTracking()
    view.close()


def test_real_gl_rect_lasso_brush_magic_and_depth_reach():
    if os.environ.get('QT_QPA_PLATFORM')=='offscreen' or not os.environ.get('DISPLAY'):pytest.skip('Run with private Xvfb and xcb for real GL selection.')
    app=QApplication.instance() or QApplication([])
    view=EditView();view.resize(400,400)
    vertices=[];faces=[]
    for z in (1.,-1.):
        base=len(vertices)
        vertices.extend((x,y,z) for y in np.linspace(-10,10,11) for x in np.linspace(-10,10,11))
        for y in range(10):
            for x in range(10):
                a=base+y*11+x;faces.extend([(a,a+1,a+11),(a+1,a+12,a+11)])
    vertices=np.array(vertices,np.float32);faces=np.array(faces,np.int32)
    normals=np.tile([0.,0.,1.],(len(vertices),1)).astype(np.float32)
    view.show_items([dict(key='mesh',kind='mesh',V=vertices,N=normals,F=faces)],edit='mesh')
    view.show();QTest.qWait(500);assert view.isValid() and not view.error
    def stroke(tool,points):
        view.set_tool(tool);view.selection_mode='replace'
        QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=QPoint(*points[0]))
        for point in points[1:]:QTest.mouseMove(view,QPoint(*point),10)
        QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=QPoint(*points[-1]));QTest.qWait(50)
        return int(view.selection.sum())
    view.through=False;front=stroke('rect',[(10,10),(390,390)])
    view.through=True;both=stroke('rect',[(10,10),(390,390)])
    assert front==200 and both==400
    subset=stroke('lasso',[(10,10),(200,10),(200,390),(10,390)])
    assert 0<subset<both
    view.brush_px=35;painted=stroke('brush',[(190,190),(210,210)])
    assert 0<painted<both
    view.through=False;view.set_tool('magic');view.selection_mode='replace'
    QTest.mouseClick(view,Qt.MouseButton.LeftButton,pos=QPoint(200,200));QTest.qWait(50)
    assert int(view.selection.sum())==200
    camera=view.cam.view.copy()
    QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=QPoint(200,200));QTest.mouseMove(view,QPoint(215,215));QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=QPoint(215,215))
    assert np.array_equal(view.cam.view,camera)
    for point_display in (False,True):
        view.set_mesh_points(point_display);view.clear_selection();view.set_tool('magic');QTest.qWait(50)
        QTest.mouseMove(view,QPoint(190,195),10);QTest.qWait(240)
        assert view.hover_mask is not None and int(view.hover_mask.sum())==200
        assert not view.selection.any() and len(view.mask_points())==0
        if not point_display:
            amber=view.grabFramebuffer().pixelColor(200,200)
            QTest.mouseClick(view,Qt.MouseButton.LeftButton,pos=QPoint(190,195));QTest.qWait(40)
            red=view.grabFramebuffer().pixelColor(200,200)
            assert amber.green()>red.green()+20
            view.clear_selection()
        view.set_tool('brush');QTest.mouseMove(view,QPoint(205,205),10);QTest.qWait(240)
        assert view.hover_mask is not None and 0<int(view.hover_mask.sum())<200
        assert not view.selection.any() and len(view.mask_points())==0
    view.set_tool('orbit');assert view.hover_mask is None
    view.close()
