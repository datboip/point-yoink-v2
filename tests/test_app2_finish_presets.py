"""Recipe wiring checks with temporary projects and config."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp()) / 'config.json')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import app2
import pytest
from app2.builder.stagehost import StageHost
from app2.builder.preset_store import recipe_fragment, DEFAULTS

@pytest.fixture
def host(tmp_path):
    from stages.engine import Engine
    from stages.project import Project
    (tmp_path / 'data').mkdir()
    h = StageHost.__new__(StageHost)
    h.e = Engine(Project.open(str(tmp_path), create=True))
    return h

def test_finish_updates_last_and_undo_restores_settings(host):
    host.h_finish({'smooth': 5})
    first = host.e.chain_ops('meshed')[-1][2]['id']
    host.h_finish({'smooth': 8})
    ops = host.e.chain_ops('meshed')
    assert len(ops) == 1 and ops[-1][2]['id'] == first
    assert ops[-1][2]['params']['smooth'] == 8
    host.e.undo()
    assert host.e.chain_ops('meshed')[-1][2]['params']['smooth'] == 5
    host.e.redo()
    assert host.e.chain_ops('meshed')[-1][2]['params']['smooth'] == 8

def test_existing_finish_history_kept(host):
    for value in [5, 5, 8, 5, 5]:
        host.e.add_op('meshed', {'tool': 'finish', 'params': {'smooth': value}})
    ids = [o['id'] for _, _, o in host.e.chain_ops('meshed')]
    host.h_finish({'smooth': 6})
    assert [o['id'] for _, _, o in host.e.chain_ops('meshed')] == ids
    assert host.e.chain_ops('meshed')[-1][2]['params']['smooth'] == 6

def test_custom_recipe_preview_apply_undo(host):
    before = host.e.recipe()
    fragment = recipe_fragment(dict(DEFAULTS, detail=.25, smooth=2))
    host.h_plan = lambda **kw: {'to_run': [], 'est_s': 0}
    summary = host.h_custom_onetap(fragment, 'custom')
    assert summary['preset'] == 'custom' and '0.25' in summary['fusion']
    assert host.e.recipe() == before
    host.h_apply_custom(fragment, 'custom')
    assert host.e.recipe()['params']['fused']['point_distance'] == .25
    host.e.undo()
    assert host.e.recipe() == before

def test_old_preset_preview_cannot_replace_new_one():
    from app2.qt.build_step import BuildStepPage
    class Page:
        _preview_onetap = BuildStepPage._preview_onetap
        def __init__(self):
            self.host = object(); self.calls = []; self.results = []
            self.onetap_dlg = type('Dialog', (), {'show_summary': self.results.append})()
        def call(self, name, *args, **kw): self.calls.append(kw)
        def _center_onetap(self): pass
    p = Page()
    p._preview_onetap('h_custom_onetap', ())
    p._preview_onetap('h_custom_onetap', ())
    p.calls[1]['then']({'new': True})
    p.calls[0]['then']({'old': True})
    assert p.results == [{'new': True}]
