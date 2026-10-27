"""Round 13 UI and existing recipe bindings, offscreen and without a device."""
import os,tempfile,json,types,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['POINTYOINK_NO_DEVICE']='1'
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='round13-tests-'))/'config.json')
import app2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt import build_step as B,build_words as W
from app2.qt.bridge import SessionBridge
from app2.builder.stagehost import StageHost,Snapshot,apply_event
from stages.engine import Engine
from stages.project import Project
from stages import params as PR

@pytest.fixture
def app():return QApplication.instance() or QApplication([])

@pytest.fixture
def host(tmp_path):
    for side in ('a','b'):
        folder=tmp_path/'data'/side/'cache';folder.mkdir(parents=True);(folder/'pointyoink_track.json').write_text('{}')
    h=StageHost.__new__(StageHost);h.e=Engine(Project.open(str(tmp_path),create=True));h.e.preset('revo-device');return h


def test_fill_preview_invalidates_when_parameters_change(app):
    p=B.BuildStepPage(SessionBridge(),False);p.stage='meshed'
    assert p.fix_ui['meshed']['state'].wordWrap()
    p._fix['meshed']['spot']='[[0,0,0],[0,0,1]]';p._display_path=lambda:'/model.ply'
    params={'action':'fill','spot':p._fix['meshed']['spot'],'diameter':20}
    p._fix_params=lambda stage:copy.deepcopy(params)
    signature=p._fix_signature('meshed');p._fix['meshed']['preview']=signature
    calls=[];p.notice=calls.append;p.call=lambda *a,**k:calls.append('apply')
    params['diameter']=30;p._fix_apply('meshed')
    assert calls==['Press Preview patch first, then apply it.']
    p.shutdown();p.close()


def test_guessed_faces_are_purple_uint8_after_load(tmp_path):
    from stages.plyio import write_mesh
    from app2.builder.view_data import ViewLoader
    V=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32);F=np.array([[0,1,2],[0,2,3]])
    path=tmp_path/'mesh.ply';write_mesh(str(path),V,F)
    np.save(tmp_path/'guessed.npy',np.packbits([False,True]))
    loader=ViewLoader(lambda *a:None)
    try:
        result=loader._load(dict(items=[dict(key='main',path=str(path))],mode='shaded',edit=None))
        rgb=result['items'][0]['face_rgb'];assert rgb.dtype==np.uint8 and list(rgb[1])==[184,102,242] and not np.array_equal(rgb[0],rgb[1])
    finally:loader.close()


def test_filled_area_stat_uses_faces_and_area():
    assert W.operation_stats('local_fix',dict(added_faces=123,filled_mm2=45.25),'mesh')=='Filled 123 faces over 45.2 mm2'


def test_preview_matches_apply_and_does_not_change_source(tmp_path):
    from test_fill_gap import sheet,P
    from stages.plyio import write_mesh,read_mesh
    from stages.tools.local_fix import preview_job,apply
    V,F=sheet();path=tmp_path/'input.ply';write_mesh(str(path),V,F);before=path.read_bytes()
    preview=preview_job({'in':str(path),'out_dir':str(tmp_path),'params':P})
    new,_,st=apply('mesh',(V,F,None),P);patch=read_mesh(preview['path']);mask=st['_guessed']
    assert path.read_bytes()==before and len(patch[1])==mask.sum()
    assert np.allclose(np.sort(patch[0][patch[1]].reshape(-1,3),axis=0),np.sort(new[0][new[1][mask]].reshape(-1,3),axis=0),atol=1e-5)
    assert st['filled_mm2']>0 and np.array_equal(new[1][:len(F)],F)


def test_whole_fill_stitches_boundary_and_refuses_broken_patch():
    from test_fill_gap import sheet
    from stages.tools import guess_fill as G
    V,F=sheet();new,mask,st=G.apply((V,F,None),{'min_mm':2,'max_mm':30})
    assert st['filled']==1 and st['filled_mm2']>0 and mask.sum()==st['guessed_faces']
    assert len(G.boundary_loops(new[1]))==1
    assert not G.stitches_loop(np.array([[0,1,2]]),np.array([0,1,2,3]))
    assert G.boundary_loops(np.zeros((0,3),int))==[]


def test_preview_callback_rejects_old_result_and_keeps_model(app):
    from PySide6.QtWidgets import QLabel
    overlays=[];model=np.eye(4);model[0,3]=10
    q=types.SimpleNamespace(stage='meshed',view_id=8,_fix_signature=lambda st:('meshed','mesh','params'),_fix_on=lambda:True,
        _orient_item=lambda:('main',{'model':model}),_fix={'meshed':{'preview_stats':{'added_faces':4,'filled_mm2':12.5}}},
        vp=types.SimpleNamespace(view=types.SimpleNamespace(set_fill_preview=overlays.append)),_fix_view=lambda:None,
        fix_ui={'meshed':{'state':QLabel()}},notice=lambda text:None)
    req={'stage':'meshed','signature':('meshed','mesh','params')}
    B.BuildStepPage._on_fill_preview(q,dict(id=7,req=req,items=[{'key':'patch'}]));assert not overlays
    B.BuildStepPage._on_fill_preview(q,dict(id=8,req=req,items=[{'key':'patch'}]))
    assert np.array_equal(overlays[0]['model'],model) and q._fix['meshed']['preview']==req['signature']
    assert 'Filled 4 faces over 12.5 mm2' in q.fix_ui['meshed']['state'].text()


def test_preview_survives_context_restore_without_becoming_editable(app):
    from app2.qt.edit_view import EditView
    v=EditView();patch={'key':'patch'};v.set_fill_preview(patch);v._todo={};v._restore_scene()
    assert v._todo['fill_preview'] is patch and 'patch' not in v.items and v.edit_key is None
    v.close()




def test_host_preview_file_survives_job_cleanup(host,tmp_path):
    from test_fill_gap import sheet,P
    from stages.plyio import write_mesh
    V,F=sheet();path=tmp_path/'sheet.ply';write_mesh(str(path),V,F)
    before=host.e.recipe();result=host.h_fill_preview(str(path),P)
    assert Path(result['path']).is_file() and result['stats']['added_faces']>0
    assert host.e.recipe()==before
