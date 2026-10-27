"""Atomic applied-edit checkboxes and read-only plans use stable stage cache keys."""
import os,tempfile,copy
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-toggle-'))/'config.json')
import app2
import pytest
from types import SimpleNamespace
from app2.builder.stagehost import StageHost,MUTATING
from stages.engine import Engine
from stages.project import Project


@pytest.fixture
def host(tmp_path):
    (tmp_path/'data').mkdir()
    host=StageHost.__new__(StageHost)
    host.e=Engine(Project.open(str(tmp_path),create=True))
    for tool in ('smooth','slivers','finish'):
        host.e.add_op('meshed',{'tool':tool,'params':{}})
    return host


def test_batch_is_one_undo_entry_and_retains_other_edits(host):
    before=host.e.recipe();chain=host.e.chain_ops('meshed')
    changes={chain[0][2]['id']:False,chain[1][2]['id']:False}
    assert 'h_toggle_many' in MUTATING and 'h_toggle_plan' not in MUTATING
    assert host.h_toggle_many('meshed',changes)=={'changed':2,'states':changes}
    after=host.e.recipe()
    assert [o.get('on',True) for _,_,o in host.e.chain_ops('meshed')]==[False,False,True]
    assert host.e.chain_ops('meshed')[-1][2]==chain[-1][2]
    host.e.undo();assert host.e.recipe()==before
    host.e.redo();assert host.e.recipe()==after
    # A no-op Apply adds no Undo entry.
    host.h_toggle_many('meshed',changes);host.e.undo();assert host.e.recipe()==before


@pytest.mark.parametrize('bad',[{'missing':False},{'missing':True},{'op-0001':'false'},{0:False}])
def test_invalid_batch_changes_nothing(host,bad):
    before=host.e.recipe();first=host.e.chain_ops('meshed')[0][2]['id']
    changes={first:False};changes.update(bad)
    with pytest.raises(ValueError):host.h_toggle_many('meshed',changes)
    assert host.e.recipe()==before


def test_default_fused_ids_resolve_before_materialization(host):
    host.e.set({'fused.method':'points'})
    chain=host.e.chain_ops('fused')
    assert len(chain)>1 and all(o['id'].startswith('default-') for _,_,o in chain)
    before=host.e.recipe()
    changes={operation['id']:False for _,_,operation in chain}
    assert host.h_toggle_many('fused',changes)['changed']==sum(bool(o['on']) for _,_,o in chain)
    assert all(not o['on'] for _,_,o in host.e.chain_ops('fused'))
    host.e.undo();assert host.e.recipe()==before


def test_plan_uses_hypothetical_recipe_without_saving(host,monkeypatch):
    before=host.e.recipe();first=host.e.chain_ops('meshed')[0][2]['id'];seen=[]
    def plan(to=None,recipe=None):
        seen.append((to,copy.deepcopy(recipe)))
        return {'to_run':1,'est_s':2,'ids':['meshed'],'stale':[],'waiting':[],'cached':3}
    monkeypatch.setattr(host,'h_plan',plan)
    result=host.h_toggle_plan('meshed',{first:False},to='meshed')
    assert result['changed']==1 and result['to_run']==1
    assert seen[0][0]=='meshed' and seen[0][1]['ops']['meshed'][0]['on'] is False
    assert host.e.recipe()==before
    with pytest.raises(ValueError):host.h_toggle_plan('meshed',{'missing':False})
    assert len(seen)==1 and host.e.recipe()==before


def test_changed_upstream_state_needs_new_keys_but_repeat_can_reuse(host,monkeypatch):
    # Use the production graph and content-key calculation without invoking scanner work.
    monkeypatch.setattr(host.e,'code_digest',lambda step:'fixed-test-code')
    def keys():
        graph=host.e.graph();outputs={};result={}
        for step in graph.steps:
            if step.alias:
                digest=outputs[step.inputs[step.alias]].digest
            else:
                digest=host.e.step_key(step,outputs)
            outputs[step.id]=SimpleNamespace(digest=digest)
            if step.kind=='op' and step.group=='meshed':result[step.op['id']]=digest
        return result
    initial=keys();chain=host.e.chain_ops('meshed');first=chain[0][2]['id'];last=chain[-1][2]['id']
    cache=set(initial.values())
    host.h_toggle_many('meshed',{first:False});changed=keys()
    assert changed[last]!=initial[last] and changed[last] not in cache
    cache.update(changed.values())
    host.h_toggle_many('meshed',{first:True});again=keys()
    assert again==initial and all(key in cache for key in again.values())
    host.h_toggle_many('meshed',{first:False});assert keys()==changed


def test_actual_plan_reports_counts_without_committing(host):
    before=host.e.recipe();first=host.e.chain_ops('meshed')[0][2]['id']
    result=host.h_toggle_plan('meshed',{first:False},to='meshed')
    assert isinstance(result['to_run'],int) and result['to_run']>=0
    assert isinstance(result['est_s'],int) and isinstance(result['cached'],int)
    assert result['changed']==1 and host.e.recipe()==before
