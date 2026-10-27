import json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def test_side_and_points_do_not_offer_global_pose_guide(app,tmp_path,monkeypatch):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    from stages import color
    p=BuildStepPage(SessionBridge(),gl=False);p.gl=True
    monkeypatch.setattr(p,'_display_path',lambda:str(tmp_path/'mesh.ply'))
    monkeypatch.setattr(color,'has_colors',lambda *_:False)
    choices=[];p.vp.set_colours=lambda modes,current:choices.append([m[0] for m in modes])
    p.stage='side';p.shown={'kind':'mesh'};p._refresh_colours()
    assert 'scan_guide' not in choices[-1] and 'quality' in choices[-1]
    p.stage='fused';p.shown={'kind':'points'};p._refresh_colours();assert 'scan_guide' not in choices[-1]
    p.shown={'kind':'mesh'};p._refresh_colours();assert 'scan_guide' in choices[-1]
    p.gl=False;p.shutdown();p.close()


def lineups(tmp_path):
    root=tmp_path/'stages/nodes/lineup.adjust';old=root/'old/sides.json';new=root/'new/sides.json'
    old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
    A=np.eye(4);B=A.copy();B[0,3]=100
    old.write_text(json.dumps(dict(sides=[dict(id='side',T=A.tolist())])))
    new.write_text(json.dumps(dict(sides=[dict(id='side',T=B.tolist())])))
    os.utime(old,(1,1));os.utime(new,(100,100))
    return old,new,A


def test_selected_graph_artifact_wins_over_newer_cache(tmp_path):
    from app2.builder.scan_guide import selected_lineup
    old,new,A=lineups(tmp_path)
    rows,path=selected_lineup(tmp_path,lineup_file=old)
    assert path==str(old) and np.allclose(rows[0]['T'],A)


def test_published_frames_select_matching_old_artifact(tmp_path):
    from app2.builder.scan_guide import selected_lineup
    old,new,A=lineups(tmp_path)
    rows,path=selected_lineup(tmp_path,frames={'side':A})
    assert path==str(old) and np.allclose(rows[0]['T'],A)
    assert selected_lineup(tmp_path,frames={'unknown':A})==([],None)


def test_stagehost_supplies_selected_lineup(tmp_path,monkeypatch):
    from app2.builder.stagehost import StageHost
    from app2.builder import scan_guide
    old,new,A=lineups(tmp_path);calls=[]
    monkeypatch.setattr(scan_guide,'guide',lambda project,src,mode='turntable',**kw:calls.append(kw) or {})
    host=StageHost.__new__(StageHost);host.p=SimpleNamespace(root=str(tmp_path))
    host.e=SimpleNamespace(_begin=lambda:None,output=lambda node,kind=None:str(old) if kind=='sides' else None)
    host.h_scan_guide('mesh.ply')
    assert calls==[{'lineup_file':str(old)}]


def test_guidance_stays_above_six_line_touchup_legend(app):
    from app2.qt.scan_guide import guidance_rect,touch_legend_height
    from app2.qt.model_panel import ModelOverlay
    o=ModelOverlay();o.resize(600,500);o.touch={}
    rect=guidance_rect(o);legend_top=o.height()-touch_legend_height(o)
    assert rect.bottom()<legend_top and rect.left()==14
    o.touch=None;assert guidance_rect(o).bottom()<o.height()
    o.close()


def test_pose_text_names_height_distance_and_handheld_coordinates():
    from app2.builder.scan_guide import placement_text
    T=np.eye(4);T[:3,3]=[0,80,200]
    r=dict(bed=[[0,1,0],-10],center=[0,0,0],mode='handheld',poses=[dict(T=T.tolist(),text='Tilt down 20 degrees')])
    placement_text(r);p=r['poses'][0]
    assert p['height_above_bed_mm']==70 and p['distance_from_center_mm']==pytest.approx(np.sqrt(46400))
    assert '70 mm above the bed' in p['text'] and 'part center' in p['text'] and 'Model position' in p['text']


def test_suggestion_rank_three_does_not_use_done_green():
    from scanner.scan_guide import RANK_COLORS
    assert RANK_COLORS[2]!='#7de38d'
    rgb=[int(RANK_COLORS[2][i:i+2],16) for i in (1,3,5)]
    assert rgb[2]>rgb[1] and rgb[2]>rgb[0]

def test_published_model_never_uses_current_recipe_lineup(tmp_path,monkeypatch):
    from app2.builder.stagehost import StageHost
    from app2.builder import scan_guide
    old,new,A=lineups(tmp_path);calls=[]
    monkeypatch.setattr(scan_guide,'guide',lambda project,src,mode='turntable',**kw:calls.append(kw) or {})
    host=StageHost.__new__(StageHost);host.p=SimpleNamespace(root=str(tmp_path))
    host.e=SimpleNamespace(_begin=lambda:None,output=lambda node,kind=None:str(new) if kind=='sides' else None)
    host.h_scan_guide(str(tmp_path/'model.ply'))
    assert calls==[{}], 'Published model must resolve its saved mapping, not the changed recipe'
