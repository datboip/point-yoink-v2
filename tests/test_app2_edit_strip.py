"""Selection strip and side pager stay in their own modes without moving scenes."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_edit_strip_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt.build_view import BuildViewPanel
from app2.qt.stage_list import Pager


@pytest.mark.parametrize('width',[492,900])
def test_selection_state_keeps_scene_region_within_each_workspace(width):
    app=QApplication.instance() or QApplication([])
    view=BuildViewPanel(False);view.resize(width,616);view.show();app.processEvents()
    assert not view.rect_t.isVisible()
    scene=view.area.geometry()
    for enabled in [True,False,True,False]:
        view.set_edit_workspace(enabled);app.processEvents()
        assert view.area.geometry()==scene
        view.set_tools(True);view.set_selection(7,'polygons');app.processEvents()
        assert not view.slot.isVisible()
        assert view.area.geometry()==scene
        assert not view.rect_t.isVisible() and not view.sel_delete.isVisible()
    view.close()


@pytest.mark.parametrize('width',[200,248])
def test_combined_pager_has_no_source_controls(width):
    app=QApplication.instance() or QApplication([])
    pager=Pager();pager.resize(width,72);pager.show();app.processEvents()
    pager.set_page(0,0,False);app.processEvents()
    assert not pager.isVisible()
    for paged in [True,False,True,False]:
        pager.set_page(12,22,paged);app.processEvents()
        assert pager.isVisible()==paged
        assert not pager.all.isVisible()
        assert pager.prev.isVisible()==paged and pager.next.isVisible()==paged
        assert pager.lab.isVisible()==paged and pager.height()==72
        if paged:
            assert pager.text.fontMetrics().horizontalAdvance(pager.text.text())<=pager.text.width()
            assert pager.prev.geometry().right()<pager.text.geometry().left()
    pager.close()


@pytest.mark.parametrize('size',[(1280,800),(1688,1044)])
def test_edit_right_settings_collapse_and_build_return(size):
    from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout
    from app2.qt.edit_page import EditPage
    app=QApplication.instance() or QApplication([])
    class Build(QWidget):
        gl=False
        def __init__(self):
            super().__init__();layout=QHBoxLayout(self)
            self.left=QWidget();self.left.setFixedWidth(248);QVBoxLayout(self.left)
            self.stages=QWidget();self.left.layout().addWidget(self.stages)
            self.vp=BuildViewPanel(False);self.right=QWidget();self.right.setFixedWidth(400)
            layout.addWidget(self.left);layout.addWidget(self.vp,1);layout.addWidget(self.right)
        def show_stage(self,*a,**kw):pass
        def set_tool(self,t):pass
    page=EditPage();page.resize(*size);build=Build();page.attach(build);page.show();app.processEvents()
    assert not build.right.isVisible() and page.edge_button.isHidden()
    assert page.options_button.isVisible() and page.options_button.text()=='Options'
    assert page.options_button.accessibleName()=='Show edit options'
    large=build.vp.width()
    page.options_button.click();app.processEvents()
    assert build.right.isVisible() and page.edge_button.isHidden() and build.vp.width()<large
    assert page.options_button.text()=='Hide' and page.options_button.accessibleName()=='Hide edit options'
    page.options_button.click();app.processEvents()
    assert not build.right.isVisible() and page.edge_button.isHidden() and build.vp.width()==large
    assert page.options_button.text()=='Options'
    assert page.detach() is build and not build.right.isHidden()
    assert page.edge_button.isHidden() and not build.vp.edit_workspace
    page.attach(build);app.processEvents();assert not build.right.isVisible()
    assert page.options_button.isVisible() and page.options_button.isEnabled()
    page.options_button.click();app.processEvents();assert build.right.isVisible()
    page.detach();build.close();page.close()


def test_hidden_build_selection_cannot_arm_or_delete():
    from types import SimpleNamespace
    from app2.qt.build_step import BuildStepPage
    calls=[]
    view=SimpleNamespace(tools_on=True,set_tool=lambda tool:calls.append(tool))
    page=SimpleNamespace(gl=True,stage='meshed',fin=SimpleNamespace(clear=lambda:None),slice_tool=SimpleNamespace(clear=lambda:None),vp=SimpleNamespace(edit_workspace=False,view=view,set_tool=lambda tool:None))
    BuildStepPage.set_tool(page,'rect');BuildStepPage.set_tool(page,'lasso')
    assert calls==['orbit','orbit']
    # Returning before engine/state access also prevents a stale selection delete.
    BuildStepPage.delete_selection(page)


def test_edit_help_stays_enabled_without_mesh_and_undo_mirrors_host():
    from types import SimpleNamespace
    from PySide6.QtWidgets import QPushButton
    from app2.qt.edit_page import EditPage
    app=QApplication.instance() or QApplication([])
    page=EditPage();page.resize(800,700);page.palette.setParent(page);page.palette.setGeometry(0,0,248,650);page.palette.show();page.show()
    undo=QPushButton();redo=QPushButton();undo.setEnabled(False);redo.setEnabled(False);calls=[]
    page.build=SimpleNamespace(gl=False,running=lambda:False,undo_b=undo,redo_b=redo,undo=calls.append)
    page.sync();app.processEvents()
    assert page.mode_help.isEnabled() and not page.mode.isEnabled()
    assert not page.undo_button.isEnabled() and 'Build the result' in page.reason.text()
    page.mode_help.click();app.processEvents();assert page._help_card.isVisible()
    undo.setEnabled(True);page.sync();page.undo_button.click();assert calls==[-1]
    assert not page.redo_button.isEnabled()
    page.build=None;page.close();undo.close();redo.close()


def test_edit_delete_action_and_count_need_a_selection():
    import numpy as np
    from types import SimpleNamespace
    from app2.qt.edit_page import EditPage
    app=QApplication.instance() or QApplication([])
    page=EditPage();calls=[]
    view=SimpleNamespace(tools_on=True,tool='lasso',selection=np.array([True,False,True]))
    page.build=SimpleNamespace(gl=True,stage='meshed',running=lambda:False,vp=SimpleNamespace(view=view),delete_selection=lambda:calls.append('delete'),clear_selection=lambda:calls.append('clear'))
    page.sync();assert page.reason.text()=='2 faces selected' and page.delete_button.isEnabled()
    assert not calls
    page.delete_button.click();page.clear_button.click();assert calls==['delete','clear']
    view.selection[:]=False;page.sync();assert not page.delete_button.isEnabled()
    assert any('thin bridge' in button.toolTip() for button in page.tools)
    page.build=None;page.close()


def test_compact_palette_icons_and_segment_bindings():
    from types import SimpleNamespace
    from PySide6.QtWidgets import QPushButton
    from app2.qt.edit_page import EditPage
    app=QApplication.instance() or QApplication([])
    page=EditPage();through=QPushButton();through.setCheckable(True)
    view=SimpleNamespace(selection_mode='replace')
    page.build=SimpleNamespace(gl=True,vp=SimpleNamespace(view=view,through=through))
    for name in ['lasso','rect','brush','magic']:assert not page.tool_buttons[name].icon().isNull()
    page.mode_buttons['Subtract'].click();assert page.mode.currentText()=='Subtract' and view.selection_mode=='subtract'
    page.through_button.click();assert through.isChecked()
    page.visible_button.click();assert not through.isChecked()
    page.build=None;page.close();through.close()
