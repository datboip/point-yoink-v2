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


def test_retry_event_keeps_result_as_built_warning():
    p=types.SimpleNamespace(nodes={},text='',current=None)
    lines=apply_event('node',dict(id='fused',state='retry',retry=True,warn=['Built the classic way: plus ran out of GPU memory']),p)
    assert p.nodes['fused']['state']=='warn' and p.nodes['fused']['retry']
    assert 'Built the classic way' in lines[0]


def test_retry_row_and_view_are_built(app,tmp_path):
    p=B.BuildStepPage(SessionBridge(),False);p.stage='fused';file=tmp_path/'mesh.ply';file.write_text('fixture')
    node=dict(id='fused',status='retry',retry=True,method='plus',params={},stats={'mode':'classic (fallback)'},warn=['Built the classic way: plus ran out of GPU memory'])
    p.snap=Snapshot(groups=[dict(name='fused',nodes=[node])],sides=[],files={'fused':{'mesh':str(file)}},finals={'fused':'fused'},fused_kind='mesh',fused_method='plus')
    assert p._row_state('fused')[0]=='Built, will retry'
    assert all('not built' not in text.lower() for text,_ in p.view_label())
    assert p.view_label()[0][0]=='Built the classic way: plus ran out of GPU memory'
    from app2.qt.theme import T
    assert all(color is None or color in T for _,color in p.view_label())
    from app2.qt.widgets import fit_parts
    assert fit_parts(p.view_label(),300)[0][0].startswith('Built the classic way')
    p.snap=None;p.shutdown();p.close()


