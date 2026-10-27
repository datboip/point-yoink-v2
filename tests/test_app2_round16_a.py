"""Private editing and project orientation regressions."""
import json
from types import SimpleNamespace as NS
import numpy as np
from app2.qt import build_step as B


def test_undo_runs_and_publishes_after_uncached_change():
    runs=[]
    p=NS(start_run=lambda r:runs.append(r) or True, _refresh_rebuild=lambda:None)
    B.BuildStepPage._after_undo(p, {'to_run':1})
    assert len(runs)==1 and runs[0].publish and runs[0].to is None


def test_applied_edit_rebuilds_descendants_and_published_surface():
    p=NS(published='old.ply')
    run=B.BuildStepPage._edit_run(p)
    assert run.publish and run.to is None and p.published is None


def test_fin_third_and_fourth_spots_change_plane():
    from stages.tools.fin import rib_geometry
    def geometry(spots):
        anchors=[[[-6,0,2],[0,0,1]],[[6,0,2],[0,0,1]]]+[[p,[0,0,1]] for p in spots]
        return rib_geometry(dict(thickness=2,reach=8,shape='triangle',anchors=json.dumps(anchors)))[0]
    a=geometry([[0,4,2]]);b=geometry([[0,-4,2]]);c=geometry([[0,4,2],[0,4,6]])
    assert a[:,1].max()>4 and b[:,1].min()<-4
    assert c[:,2].max()>a[:,2].max()+1


def test_fin_needs_three_picks_shared_cuts_keep_their_counts():
    from app2.qt.fin_tool import FinTool
    from app2.qt.slice_tool import SliceTool
    from app2.qt.drill_tool import DrillTool
    assert (FinTool.minimum_picks,SliceTool.minimum_picks,DrillTool.minimum_picks)==(3,2,1)
    assert FinTool.maximum_picks is None


def test_orientation_prompt_once_per_project(tmp_path):
    from app2.view_orientation import orientation_prompt_pending,dismiss_orientation_prompt
    assert orientation_prompt_pending(tmp_path)
    dismiss_orientation_prompt(tmp_path)
    assert not orientation_prompt_pending(tmp_path)
    assert orientation_prompt_pending(tmp_path/'other')


def test_saved_export_orientation_survives_restart(tmp_path):
    p=NS(host_root=str(tmp_path),_orient_path=lambda:str(tmp_path/'orient.json'))
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    B.BuildStepPage._orient_save(p,R)
    assert np.array_equal(B.BuildStepPage._orient_load(p),R)
    p._orient_load=lambda:B.BuildStepPage._orient_load(p)
    assert B.BuildStepPage._orient_kw(p)=={'orient':R.tolist()}
    from app2.view_orientation import orientation_prompt_pending
    assert not orientation_prompt_pending(tmp_path)


def test_native_fin_engine_history_restart_publish_and_undo(tmp_path):
    import hashlib
    from test_stages_plus import _PlusRun, arg
    from test_app2_fin_validation import l_part
    from stages import plyio
    from stages.engine import Engine
    from stages.project import Project
    class NativeFinFixture(_PlusRun):
        def fake_cmd(self,cmd):
            import os
            if os.path.basename(cmd[1])=='fuse.py':
                mesh=l_part();plyio.write_mesh(arg(cmd,'--out'),mesh.vertices,mesh.faces)
                return 'STAGE done {"faces": 20}\n'
            return super().fake_cmd(cmd)
    case=NativeFinFixture();case.setUp()
    try:
        case.e.clear_ops('fused.plus');case.run_()
        model=__import__('pathlib').Path(case.tmp)/'model.ply'
        import time
        initial=model.read_bytes();initial_faces=len(plyio.read_mesh(str(model))[1]);entries,cursor=case.e.hist.entries()
        started=time.monotonic()
        anchors=[[[12,0,2],[0,0,1]],[[2,0,12],[1,0,0]],[[20,0,2],[0,0,1]]]
        case.e.add_op('meshed',{'tool':'fin','params':{'anchors':json.dumps(anchors),'thickness':2,'reach':8}})
        entries2,cursor2=case.e.hist.entries()
        assert cursor2==cursor+1
        case.run_();changed=model.read_bytes();assert changed!=initial
        apply_seconds=time.monotonic()-started;changed_faces=len(plyio.read_mesh(str(model))[1])
        restored=Engine(Project.open(case.tmp),out=case.out)
        assert restored.chain_ops('meshed')[-1][2]['tool']=='fin'
        case.run_();assert model.read_bytes()==changed
        case.e.undo();case.run_();assert model.read_bytes()==initial
        case.e.redo();case.run_();assert model.read_bytes()==changed
        print('NATIVE_FIN_MEASUREMENT '+json.dumps(dict(initial_faces=initial_faces,fin_faces=changed_faces,apply_seconds=round(apply_seconds,4),undo_exact=True,redo_exact=True,history_steps=1)))
    finally:case.tearDown()


import pytest
from test_app2_mesh_edit_flow import workspace,pick

@pytest.mark.parametrize('tab,toolname,points',[
    ('Add fin','fin',[[-6.,0,2],[6.,0,2],[0.,5,2]]),
    ('Cut gap','slice_gap',[[-6.,0,2],[6.,0,2]]),
    ('Cut shape','drill',[[0.,0,2]]),
    ('Smooth','smooth',[]),('Simplify','simplify',[]),('Hole Filling','fill_holes',[])])
def test_each_apply_stores_history_reopens_and_requests_publish(workspace,tab,toolname,points):
    from stages.engine import Engine
    from stages.project import Project
    page,engine,_=workspace
    runs=[];page.start_run=runs.append
    original=page.call
    def call(name,*args,**kwargs):
        original(name,*args,**kwargs)
        if kwargs.get('then'):kwargs['then'](None)
    page.call=call
    page.panel['meshed'].show_tab(tab);page.vp.view.tools_on=True
    before=engine.hist.entries()[1]
    if points:
        page.on_extra(tab,0)
        for point in points:pick(page,point)
    page.on_button(tab)
    op=engine.chain_ops('meshed')[-1][2]
    assert op['tool']==toolname
    assert engine.hist.entries()[1]==before+1
    assert runs[-1].publish and runs[-1].to is None
    restored=Engine(Project.open(engine.p.root))
    assert restored.chain_ops('meshed')[-1][2]==op
    restored.undo();assert not restored.chain_ops('meshed')
    restored.redo();assert restored.chain_ops('meshed')[-1][2]==op


def test_prompt_accept_opens_orient_and_dismiss_is_project_local(workspace):
    from PySide6.QtWidgets import QApplication
    from app2.view_orientation import orientation_prompt_pending
    page,engine,_=workspace
    page.host_root=engine.p.root;page._orientation_prompt_ready=True
    opened=[];page._pick_stage=opened.append
    page._maybe_prompt_orientation()
    prompt=page._orientation_prompt
    assert prompt and prompt.text()=='Is the model sitting the right way up?'
    next(b for b in prompt.buttons() if b.text()=='Fix it').click()
    QApplication.processEvents()
    assert opened==['orient'] and not orientation_prompt_pending(engine.p.root)
    page._orientation_prompt_ready=True;page._maybe_prompt_orientation()
    assert page._orientation_prompt is None


def test_delete_selection_uses_publish_route():
    calls=[];runs=[];cleared=[]
    view=NS(selection=np.array([True]),mask_points=lambda:np.array([[1,2,3]]),clear_selection=lambda:cleared.append(True),hide_selected=lambda:cleared.append('hidden'))
    def call(name,*args,**kwargs):
        calls.append((name,args));kwargs['then']({'got':1})
    page=NS(gl=True,vp=NS(edit_workspace=True,view=view),running=lambda:False,sel_path='mesh.ply',
        shown={'path':'mesh.ply'},_view_path=lambda:'mesh.ply',stage='meshed',_side=lambda:None,
        _run_to=lambda spec:'meshed',call=call,set_tool=lambda x:None,start_run=lambda r:runs.append(r) or True,
        _edit_run=lambda:B.BuildStepPage._edit_run(page),notice=lambda *a,**k:None,
        pending_changed=lambda:None,flush_pending_edits=lambda:B.BuildStepPage.flush_pending_edits(page))
    B.BuildStepPage.delete_selection(page)
    # the faces are hidden at once and the one rebuild waits (pending_edits) until it is flushed
    assert calls[0][0]=='h_mask_delete' and cleared==['hidden'] and page.pending_edits==1 and not runs
    assert B.BuildStepPage.flush_pending_edits(page) and page.pending_edits==0
    assert runs[0].publish and runs[0].to is None


def test_fin_picks_markers_and_preview_follow_saved_model_rotation(workspace):
    page,_,_=workspace
    M=np.eye(4);M[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];M[:3,3]=[4,5,6]
    page.vp.view.items['part']['model']=M
    page.panel['meshed'].show_tab('Add fin');page.vp.view.tools_on=True;page.on_extra('Add fin',0)
    local=np.array([[-6.,0,2],[6.,0,2],[0.,5,2]])
    for point in local:pick(page,M[:3,:3]@point+M[:3,3])
    assert np.allclose([a[0] for a in page.fin.anchors],local)
    assert np.allclose([m[0] for m in page.vp.view.markers],local@M[:3,:3].T+M[:3,3])
    assert np.allclose(page.vp.view.items['fin-preview']['model'],M)


def test_changing_projects_cancels_prompt_without_marking_it_seen(workspace):
    from app2.view_orientation import orientation_prompt_pending
    page,engine,_=workspace
    page.host_root=engine.p.root;page._orientation_prompt_ready=True
    page._maybe_prompt_orientation();assert page._orientation_prompt is not None
    page.attach(None)
    assert not page._orientation_prompt_ready and page._orientation_prompt is None
    assert orientation_prompt_pending(engine.p.root)


def test_scan_page_successful_build_enables_prompt_without_owned_run():
    shown=[]
    page=NS(prog=NS(done=True,model='model.ply'),run_mine=None,_stopped=False,
        _drop_host=lambda:None,_ensure_host=lambda **kw:None,_refresh_all=lambda:None,
        show_built=lambda:shown.append('model'),_quality={'old-model':object()})
    B.BuildStepPage._run_ended(page)
    assert page._orientation_prompt_ready
    assert shown == ['model'] and page._quality == {}


@pytest.mark.parametrize('fmt',['stl','3mf'])
def test_saved_project_orientation_reaches_export_vertices(tmp_path,fmt):
    import trimesh
    from stages.plyio import write_mesh
    from app2.builder.stagehost import StageHost
    mesh=trimesh.creation.box(extents=[3,5,7]);src=tmp_path/'mesh.ply';dst=tmp_path/('out.'+fmt)
    write_mesh(str(src),mesh.vertices,mesh.faces)
    page=NS(host_root=str(tmp_path),_orient_path=lambda:str(tmp_path/'orient.json'))
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    B.BuildStepPage._orient_save(page,R);page._orient_load=lambda:B.BuildStepPage._orient_load(page)
    orient=B.BuildStepPage._orient_kw(page)['orient']
    StageHost.h_export(NS(),str(src),str(dst),fmt,orient=orient)
    loaded=trimesh.load(str(dst),force='mesh',process=False)
    assert np.allclose(loaded.extents,[7,5,3],atol=1e-5)
    assert abs(loaded.bounds[0,2])<1e-5
