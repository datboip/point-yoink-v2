"""Real Qt pick/apply controls route mesh edits into an isolated stage recipe."""
import json
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_edit_flow_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import app2
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication
from app2.qt.bridge import SessionBridge
from app2.qt.build_step import BuildStepPage
from stages.engine import Engine
from stages.project import Project


@pytest.fixture
def workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = BuildStepPage(SessionBridge(), gl=True)
    page.show_stage('meshed')
    mesh = trimesh.creation.box(extents=[20,20,4])
    item = dict(key='part', kind='mesh', V=np.asarray(mesh.vertices,dtype=np.float32),
                F=np.asarray(mesh.faces,dtype=np.uint32), N=np.asarray(mesh.vertex_normals,dtype=np.float32))
    page.vp.view.show_items([item], edit='part')
    page.vp.view.tools_on = True
    (tmp_path/'project'/'data').mkdir(parents=True)
    engine = Engine(Project.open(str(tmp_path/'project'), create=True))
    calls = []
    def call(name, *args, **kwargs):
        calls.append((name,args))
        if name == 'add_op': engine.add_op(*args)
        elif name == 'set_op': engine.set_op(*args)
    page.call = call
    yield page, engine, calls
    QThreadPool.globalInstance().waitForDone(5000)
    page.shutdown(); page.deleteLater(); app.processEvents()


def pick(page, point):
    page.vp.view.surface_picked.emit(point)
    QThreadPool.globalInstance().waitForDone(5000)
    QApplication.processEvents()


@pytest.mark.parametrize('tab,attribute,key', [('Add fin','fin','fin-preview'),('Cut gap','slice_tool','slice-preview')])
def test_two_picks_preview_apply_and_recipe_undo(workspace, tab, attribute, key):
    page, engine, calls = workspace
    page.panel['meshed'].show_tab(tab); page.vp.view.tools_on = True
    page.on_extra(tab,0)
    tool = getattr(page, attribute)
    pick(page, [-6.,0,2]); pick(page, [6.,0,2])
    if tab == 'Add fin':
        assert key not in page.vp.view.items
        assert not page.panel['meshed'].body(tab).main.isEnabled()
        pick(page, [0., 5., 2.])
    count = 3 if tab == 'Add fin' else 2
    assert len(tool.anchors) == count and key in page.vp.view.items
    assert page.vp.view.items[key]['colour'].tolist() == pytest.approx([1.,.2,.15])
    page.on_button(tab)
    assert not tool.anchors and key not in page.vp.view.items
    operation = engine.chain_ops('meshed')[-1][2]
    expected = 'fin' if tab == 'Add fin' else 'slice_gap'
    assert operation['tool'] == expected
    assert len(json.loads(operation['params']['anchors'])) == count
    engine.undo(); assert not engine.chain_ops('meshed')
    engine.redo(); assert engine.chain_ops('meshed')[-1][2]['params'] == operation['params']
    restored = Engine(Project.open(engine.p.root, create=False))
    assert restored.chain_ops('meshed')[-1][2]['params'] == operation['params']


def test_tab_stage_and_other_mesh_invalidate_picks(workspace):
    page, _, _ = workspace
    page.panel['meshed'].show_tab('Add fin'); page.vp.view.tools_on = True
    page.on_extra('Add fin',0); page.fin.pick([-6.,0,2])
    page.panel['meshed'].show_tab('Cut gap')
    QThreadPool.globalInstance().waitForDone(5000); QApplication.processEvents()
    assert not page.fin.anchors and page.fin.source is None
    page.vp.view.tools_on = True; page.on_extra('Cut gap',0)
    page.show_stage('fused')
    assert page.slice_tool.source is None and not page.slice_tool.anchors


@pytest.mark.parametrize("edit_mode,expected_tool", [(True,"lasso"),(False,"orbit")])
def test_selection_tool_change_cancels_pending_fin(workspace,edit_mode,expected_tool):
    page, _, _ = workspace
    page.vp.set_edit_workspace(edit_mode)
    page.panel['meshed'].show_tab('Add fin'); page.vp.view.tools_on = True
    page.on_extra('Add fin',0); page.fin.pick([-6.,0,2])
    page.set_tool('lasso')
    QThreadPool.globalInstance().waitForDone(5000); QApplication.processEvents()
    assert page.vp.view.tool == expected_tool
    assert not page.fin.anchors and page.fin.source is None


def test_yolo_preserves_manual_fin_and_gap(workspace):
    from stages.params import apply_preset
    page, engine, _ = workspace
    anchors = json.dumps([[[-6,0,2],[0,0,1]],[[6,0,2],[0,0,1]]])
    engine.add_op('meshed', {'tool':'fin','params':{'anchors':anchors}})
    engine.add_op('meshed', {'tool':'slice_gap','params':{'anchors':anchors}})
    manual = [operation for _, _, operation in engine.chain_ops('meshed')]
    for name in ('revo', 'revo-device', 'plus'):
        recipe = apply_preset(engine.recipe(), name)
        assert [o for o in recipe['ops']['meshed'] if o['tool'] in ('fin','slice_gap')] == manual


@pytest.mark.parametrize('tab,attribute,key', [('Add fin','fin','fin-preview'),('Cut gap','slice_tool','slice-preview')])
def test_invalid_preview_removes_previous_red_geometry(workspace, tab, attribute, key):
    page, _, _ = workspace
    page.panel['meshed'].show_tab(tab); page.vp.view.tools_on = True
    page.on_extra(tab,0)
    pick(page, [-6.,0,2]); pick(page, [6.,0,2])
    tool = getattr(page, attribute)
    if tab == 'Add fin': pick(page, [0., 5., 2.])
    assert key in page.vp.view.items
    tool.anchors[1] = tool.anchors[0]
    tool.preview()
    assert key not in page.vp.view.items


def test_light_countdown_emits_without_waiting_for_frame_queue(monkeypatch):
    from app2.scan.session import ScanSession
    from app2.scan.light_timer import LightTimer
    session = ScanSession.__new__(ScanSession)
    session.light_timer = LightTimer(); session.light_timer.start(0)
    session.led = True; session._light_seen = None
    now = [300.]
    monkeypatch.setattr('app2.scan.session.time.monotonic', lambda: now[0])
    emitted, commands = [], []
    session._emit_state = lambda: emitted.append(session.light_timer.remaining(now[0]))
    session._log = lambda text: None
    def led_off(on):
        commands.append(on); session.led = on; session.light_timer.stop()
    session._set_led = led_off
    session._tick_light(); assert emitted == [120]
    now[0] = 301; session._tick_light(); assert emitted[-1] == 119
    session._tick_light(); assert len(emitted) == 2
    now[0] = 420; session._tick_light()
    assert commands == [False] and emitted[-1] is None


@pytest.mark.parametrize('name,attribute,preview', [('Add fin','fin','fin-preview'),('Cut gap','slice_tool','slice-preview')])
def test_edit_entry_button_arms_and_previews_without_second_pick_button(workspace,name,attribute,preview,monkeypatch):
    from app2.qt.edit_page import EditPage
    from PySide6.QtWidgets import QPushButton
    page, engine, _ = workspace
    editor=EditPage();editor.build=page
    page.vp.set_edit_workspace(True)
    # This fixture has a real mesh and recipe but no scanner-backed snapshot.
    monkeypatch.setattr(page,'_refresh_tools',lambda:None)
    page.vp.view.tools_on=True
    button=next(b for b in editor.palette.findChildren(QPushButton) if b.text()==name)
    button.click()
    assert page.panel['meshed'].current==name
    assert getattr(page,attribute).source is not None and page.vp.view.tool=='fin'
    pick(page,[-6.,0,2]);pick(page,[6.,0,2])
    if name == 'Add fin': pick(page,[0.,5.,2.])
    assert preview in page.vp.view.items
    page.on_button(name)
    assert engine.chain_ops('meshed')[-1][2]['tool']==('fin' if name=='Add fin' else 'slice_gap')
    editor.build=None;editor.deleteLater();QApplication.processEvents()


def test_cut_shape_one_pick_preview_apply_and_undo(workspace):
    page,engine,_=workspace
    page.panel['meshed'].show_tab('Cut shape');page.vp.view.tools_on=True
    page.on_extra('Cut shape',0);pick(page,[0.,0.,2.])
    assert len(page.drill.anchors)==1 and 'drill-preview' in page.vp.view.items
    assert page.vp.view.tool=='fin'                      # stays in pick mode: another click moves the cut (2026-10-22)
    first=page.drill.anchors[0];pick(page,[1.,0.,2.])
    assert len(page.drill.anchors)==1 and page.drill.anchors[0]!=first
    page.on_button('Cut shape')
    operation=engine.chain_ops('meshed')[-1][2]
    assert operation['tool']=='drill'
    assert len(json.loads(operation['params']['anchors']))==1
    assert 'drill-preview' not in page.vp.view.items
    engine.undo();assert not engine.chain_ops('meshed')
    engine.redo();assert engine.chain_ops('meshed')[-1][2]==operation
