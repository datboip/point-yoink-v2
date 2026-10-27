"""Integration between finished edits, orientation, and the scan guide."""
from types import SimpleNamespace as NS
import numpy as np
from app2.qt.build_step import BuildStepPage


def test_what_is_left_uses_finished_surface_not_report_point_cloud():
    snap=NS(final=lambda stage:'meshed#fin',out=lambda node,*kind:
            '/finished-with-fin.ply' if node=='meshed#fin' else '/old-coverage.npz')
    p=NS(snap=snap,stage='report',_side=lambda:None)
    assert BuildStepPage._view_path(p)=='/finished-with-fin.ply'


def test_orientation_is_applied_before_guide_bounds(monkeypatch):
    from app2.qt import scan_guide
    matrix=np.eye(4);matrix[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    observed=[]; view=NS(items={},set_flat=lambda _:None,set_mesh_points=lambda _:None)
    view.show_items=lambda items,**kw:setattr(view,'items',{it['key']:it for it in items})
    monkeypatch.setattr(scan_guide,'show',lambda v,g,**kw:observed.append(v.items['main'].get('model',np.eye(4)).copy()))
    noop=lambda *a,**kw:None
    controls=[]
    viewport=NS(view=view,guide_controls=NS(setVisible=lambda value:controls.append(value)),_place=noop)
    p=NS(view_id=7,fin=NS(clear=noop),slice_tool=NS(clear=noop),vp=viewport,
         _should_frame=lambda _:False,_scan_guides={('/mesh.ply','scan_guide'):{'poses':[]}},
         colour_mode={'meshed':'scan_guide'},stage='meshed',_ghost_legend=noop,
         _orient_view=lambda:view.items['main'].update(model=matrix),_spots_view=noop,
         _maybe_prompt_orientation=noop,_nudge_view=noop,_refresh_colours=noop,
         _platter_preview=lambda:False,pending_view=None,_refresh_info=noop,snap=None,
         _refresh_chips=noop,_refresh_tools=noop,state=None)
    BuildStepPage.on_view(p,{'id':7,'req':{'path':'/mesh.ply','stage':'meshed'},
         'items':[{'key':'main','kind':'mesh','V':np.zeros((3,3))}],'edit':'main'})
    assert len(observed)==1 and np.array_equal(observed[0],matrix)
    assert controls == [True]


def test_lost_tracking_clears_pose_before_target_change():
    from app2.qt.model_panel import ModelPanel
    noop=lambda *a,**kw:None; reused=[]
    p=NS(view=NS(follow_scanner=noop),_home_from=noop,_guide_update=reused.append,
         _guide_pose=np.eye(4),scan_guide={'poses':[{}]},overlay=NS(update=noop),pivot=None)
    ModelPanel.on_pose(p,{'T_mm':np.eye(4),'ok':False})
    assert p._guide_pose is None and not p.overlay.guidance['ready']
    ModelPanel._guide_changed(p)
    assert reused==[]


def test_pick_tools_tolerate_report_and_lineup_panels():
    from app2.qt.fin_tool import FinTool
    for stage in ('report','lineup'):
        p=NS(page=NS(stage=stage,panel={stage:NS()}),tab='Add fin')
        assert FinTool.body(p) is None


def test_finished_orientation_also_applies_to_what_is_left():
    applied=[];noop=lambda *a,**k:None
    matrix=np.eye(4);matrix[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    control=NS(setText=noop,setEnabled=noop)
    view=NS(set_item_model=lambda key,M:applied.append(M),tool='orbit')
    p=NS(gl=True,stage='report',orient_box=object(),vp=NS(view=view),
         _orient_item=lambda:('main',{}),_orient_matrix=lambda key:matrix,
         _orient_spots=[],_orient_on=lambda:False,_fill_spot_list=noop,
         orient_count=control,orient_list=object(),_orient_remove=noop,
         orient_undo=control,orient_clear=control,orient_bottom=control,
         orient_state=control,_orient_load=lambda:matrix[:3,:3],_orient_before=noop)
    BuildStepPage._orient_view(p)
    assert len(applied)==1 and np.array_equal(applied[0],matrix)


def test_project_switch_does_not_carry_another_projects_guide_mode(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from app2.qt.bridge import SessionBridge
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=QApplication.instance() or QApplication([])
    page=BuildStepPage(SessionBridge(),gl=False)
    try:
        page.colour_mode={'meshed':'scan_guide_handheld','fused':'shaded'}
        page._scan_guides={('old-mesh','scan_guide_handheld'):{'poses':[]}}
        page.attach(NS())
        assert page.colour_mode.get('meshed') not in ('scan_guide','scan_guide_handheld')
        assert page.colour_mode['fused']=='shaded' and page._scan_guides=={}
    finally:page.shutdown();page.close()
