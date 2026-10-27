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


def test_raw_has_no_grid_button_and_uses_placed_raw_result(app):
    assert len(W.PANELS['raw']['tabs'][0]['extra'])==2
    requests=[]
    q=types.SimpleNamespace(stage='raw',view_id=1,_side=lambda:{'id':'a'},_solo_wanted=lambda:False,snap=types.SimpleNamespace(sides=[dict(id='a',active=True),dict(id='b',active=True)],final=lambda k:k,out=lambda k:'/b.ply'),loader=types.SimpleNamespace(request=requests.append))
    q.call=lambda name,then=None,fail=None:then({'T':{'a':np.eye(4),'b':np.eye(4)}})
    B.BuildStepPage._request_side_in_place(q,'/raw-a.ply','shaded')
    result=requests[0];assert result['stage']=='raw' and result['edit'] is None and result['items'][0]['path']=='/raw-a.ply'
    assert result['items'][1]['colour'] is not None
    for name in ('_request_all_sides','_all_sides_layout','_all_legend','_all_planes'):assert not hasattr(B.BuildStepPage,name)


