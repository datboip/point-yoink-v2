"""Selection geometry and shared editor ownership, with isolated configuration."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_edit_config_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from app2.builder.mesh_selection import magic_region
from app2.qt.edit_view import EditView
from app2.qt.edit_page import EditPage


def mesh():
    return np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1],[4,0,0],[5,0,0],[4,1,0]],float), np.array([[0,1,2],[2,1,3],[1,4,3],[5,6,7]])


def test_magic_stops_at_sharp_edge_and_separate_piece():
    v,f=mesh()
    assert magic_region(v,f,0,30).tolist() == [True,True,False,False]
    assert magic_region(v,f,0,30,np.array([1,0,1,1],bool)).tolist() == [True,False,False,False]


def test_topology_reuses_original_editor_graph_and_does_not_mutate_geometry():
    app=QApplication.instance() or QApplication([])
    view=EditView(); v,f=mesh(); original=f.copy()
    view.show_items([dict(key='model',kind='mesh',V=v,F=f,N=None)],edit='model')
    view.selection[0]=True
    assert view.topology_selection('grow')
    assert view.selection.tolist()==[True,True,False,False]
    assert view.topology_selection('connected')
    assert view.selection.tolist()==[True,True,True,False]
    assert np.array_equal(f,original)
    view.show_items([dict(key='new',kind='mesh',V=v,F=f[:1],N=None)],edit='new')
    assert not view.selection.any() and view._face_graph_cache is None
    view.close()


def test_selection_modes_and_clear_preview():
    app=QApplication.instance() or QApplication([])
    view=EditView(); view.selection=np.array([1,0,0],bool)
    view.selection_mode='replace'; view._apply_selection(np.array([0,1,0],bool))
    assert view.selection.tolist()==[False,True,False]
    view.selection_mode='add'; view._apply_selection(np.array([0,0,1],bool))
    assert view.selection.tolist()==[False,True,True]
    view._subtract=True; view._apply_selection(np.array([0,1,0],bool))
    assert view.selection.tolist()==[False,False,True]
    view.close()


def test_edit_page_reuses_and_returns_one_workspace():
    app=QApplication.instance() or QApplication([])
    class Build(QWidget):
        gl=False
        def __init__(self):
            super().__init__(); self.left=QWidget(self); QVBoxLayout(self.left); self.stages=QWidget(); self.left.layout().addWidget(self.stages); self.calls=[]
        def show_stage(self,*args,**kwargs): self.calls.append((args,kwargs))
        def set_tool(self,tool): self.calls.append(tool)
    build=Build(); page=EditPage(); page.attach(build)
    assert page.box.itemAt(0).widget() is build
    assert build.stages.isHidden()
    assert build.calls[0][0]==('meshed',)
    assert page.detach() is build
    assert not build.stages.isHidden()
    assert page.build is None and page.box.count()==0
    build.close(); page.close()


def test_edit_keeps_selected_side_instead_of_jumping_to_finished_mesh():
    app=QApplication.instance() or QApplication([])
    class Build(QWidget):
        gl=False; stage='side'
        def __init__(self):
            super().__init__(); self.left=QWidget(self); QVBoxLayout(self.left)
            self.stages=QWidget(); self.left.layout().addWidget(self.stages); self.calls=[]
        def show_stage(self,*args,**kwargs): self.calls.append(args)
        def set_tool(self,_tool): pass
    build=Build(); page=EditPage(); page.attach(build)
    assert not build.calls and page.title.text()=='Side surface'
    page.detach(); build.close(); page.close()


def test_brush_mask_uses_exact_radius_and_connects_stroke():
    from PySide6.QtCore import QPointF
    app=QApplication.instance() or QApplication([])
    view=EditView(); view.resize(160,120); view.tool='brush'; view.brush_px=10
    view._stroke=[QPointF(30,30),QPointF(70,30)]
    mask=view._mask()
    assert mask[30,30] and mask[30,50] and mask[30,70]
    assert not mask[30,10] and not mask[10,30] and not mask[30,90]
    view.close()


def test_edit_hides_the_whole_rail_box_and_the_build_rail_has_no_more_tools_row():
    import inspect
    from app2.qt import edit_page, build_words as W
    src = inspect.getsource(edit_page)
    assert "build.rail_scroll.hide()" in src and "build.rail_scroll.show()" in src          # its empty box no longer takes half the column
    assert "More tools" not in [r["row"] for r in W.STAGES]                                  # lives in Edit now
    assert "_more_menu" not in src and "Hole Filling" in src and "Remove slivers" in src          # the extras are buttons, not a dropdown


def test_clone_lives_in_edit_not_in_the_build_rail():
    import inspect
    from app2.qt import edit_page, build_words as W
    assert [r for r in W.STAGES if "clone" in r["id"]] == []                      # no rail row
    assert "Clone" not in [t["name"] for p in ("fused", "meshed") for t in W.PANELS[p]["tabs"]]
    assert W.MORE_TOOLS["clone"]["name"] == "Clone"                                # opened from Edit like Cut shape and Add fin
    assert '"Clone"' in inspect.getsource(edit_page) and "points_ok" in inspect.getsource(edit_page)


def test_edit_names_the_level_and_says_what_carries_down():
    from app2.qt.edit_page import EditPage
    c = EditPage.CARRY
    assert "Fused and Meshed" in c["side"] and "Meshed" in c["fused"] and "Last step" in c["meshed"]


def test_mesh_options_are_accessible_and_restore_the_build_workspace():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QHBoxLayout
    from app2.qt.build_view import BuildViewPanel
    app=QApplication.instance() or QApplication([])
    class Build(QWidget):
        gl=False;stage='meshed'
        def __init__(self):
            super().__init__();layout=QHBoxLayout(self)
            self.left=QWidget();self.left.setFixedWidth(248);QVBoxLayout(self.left)
            self.stages=QWidget();self.left.layout().addWidget(self.stages)
            self.vp=BuildViewPanel(False);self.right=QWidget();self.right.setFixedWidth(400)
            layout.addWidget(self.left);layout.addWidget(self.vp,1);layout.addWidget(self.right)
        def show_stage(self,*args,**kwargs):pass
        def set_tool(self,tool):pass
    page=EditPage();build=Build();page.resize(1280,800);page.attach(build);page.show();app.processEvents()
    try:
        button=page.options_button
        assert button.objectName()=='EditMeshOptions' and button.accessibleName()=='Show edit options'
        assert button.isVisible() and button.isEnabled() and not build.right.isVisible()
        assert page.edge_button.isHidden()
        button.setFocus();QTest.keyClick(button,Qt.Key.Key_Space);app.processEvents()
        assert build.right.isVisible() and button.accessibleName()=='Hide edit options'
        QTest.keyClick(button,Qt.Key.Key_Space);app.processEvents()
        assert not build.right.isVisible() and button.text()=='Options'
        assert page.detach() is build and not build.right.isHidden() and not build.vp.edit_workspace
        assert page.edge_button.isHidden()
    finally:page.detach();build.close();page.close()


def test_selection_view_includes_the_items_orientation():
    """The brush picked faces a turn away from the circle on a model drawn with a saved orientation (2026-10-22)."""
    app=QApplication.instance() or QApplication([])
    view=EditView(); v,f=mesh()
    view.show_items([dict(key='model',kind='mesh',V=v,F=f,N=None)],edit='model')
    assert np.allclose(view._edit_view(), view.cam.view)
    M=np.eye(4); M[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]; M[:3,3]=[5,0,0]
    view.set_item_model('model',M)
    assert np.allclose(view._edit_view(), np.asarray(view.cam.view,float)@M)
    view.close()


def test_grow_and_shrink_by_distance_take_several_rings():
    app=QApplication.instance() or QApplication([])
    n=24; v=np.array([[x,y,0] for y in range(n+1) for x in range(n+1)],float); f=[]
    for y in range(n):
        for x in range(n):
            a=y*(n+1)+x; b=a+1; c=a+n+1; d=c+1; f+=[[a,b,c],[b,d,c]]
    f=np.array(f); mid=(n//2*n+n//2)*2
    view=EditView(); view.show_items([dict(key='model',kind='mesh',V=v,F=f,N=None)],edit='model')
    view.selection[mid]=True
    assert view.topology_selection('grow',mm=0); one=int(view.selection.sum())
    view.selection[:]=False; view.selection[mid]=True
    assert view.topology_selection('grow',mm=4); four=int(view.selection.sum())
    assert four > one + 10                                    # about 4 mm of rings, not one ring
    assert view.topology_selection('shrink',mm=4) and int(view.selection.sum()) < four


def test_find_loose_selects_floaters_but_not_the_main_piece():
    app=QApplication.instance() or QApplication([])
    n=12; v=[[x,y,0] for y in range(n+1) for x in range(n+1)]; f=[]
    for y in range(n):
        for x in range(n):
            a=y*(n+1)+x; f+=[[a,a+1,a+n+1],[a+1,a+n+2,a+n+1]]
    base=len(v); v+=[[50,50,5],[51,50,5],[50,51,5],[60,60,5],[61,60,5],[60,61,5]]; f+=[[base,base+1,base+2],[base+3,base+4,base+5]]
    v=np.array(v,float); f=np.array(f)
    view=EditView(); view.show_items([dict(key='model',kind='mesh',V=v,F=f,N=None)],edit='model')
    assert view.select_loose(5) == (2, 2)
    assert view.selection[-2:].all() and not view.selection[:-2].any()
