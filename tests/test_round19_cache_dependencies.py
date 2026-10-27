"""Cache identities track executed code and resolved command behavior."""
import dataclasses, os, sys
from pathlib import Path
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
from types import SimpleNamespace
import pytest
from stages import params as PR
from stages import methods
from stages.engine import Engine

@pytest.fixture
def keyed(tmp_path,monkeypatch):
    stage=tmp_path/'stages';shutil.copytree(methods.HERE,stage,ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(methods,'HERE',str(stage))
    def key(node,name,params=None):
        m=methods.get_method(node,name)
        values=PR.resolve({},node,name,None,{}) if params is None else params
        st=SimpleNamespace(method=m,params=values,produces=m.produces,kind=node,id=node,inputs={},leaf={})
        e=Engine.__new__(Engine);e._codes={};e._fcode={};e._libs={};e._lib=lambda _: 'fixture-version'
        return e.step_key(st,{})
    return stage,key

def test_bed_helper_change_does_not_invalidate_fusion_or_keep(keyed):
    stage,key=keyed;nodes=[('side.fuse','tsdf'),('fused','tsdf'),('side.keep','keep_part'),('side.platter','auto')]
    before=[key(*n) for n in nodes]
    path=stage/'methods/side_platter.py'
    if not path.exists():path=stage/'methods/side.py'
    source=path.read_text();assert 'return supported(jload(path))' in source
    path.write_text(source.replace('return supported(jload(path))','return False'))
    after=[key(*n) for n in nodes]
    assert after[:3]==before[:3], 'An unrelated bed helper must not rebuild fusion or cleanup'
    assert after[3]!=before[3], 'The step executing the changed bed helper must rebuild'

def test_fusion_backend_change_invalidates_both_fusions_only(keyed):
    stage,key=keyed;nodes=[('side.fuse','tsdf'),('fused','tsdf'),('side.platter','auto'),('side.keep','keep_part')]
    before=[key(*n) for n in nodes];path=stage/'methods/fusion_events.py'
    if not path.exists():path=stage/'methods/side.py'
    source=path.read_text();assert 'self.device=None' in source
    path.write_text(source.replace('self.device=None','self.device="CPU:0"',1))
    after=[key(*n) for n in nodes]
    assert after[0]!=before[0] and after[1]!=before[1]
    assert after[2:]==before[2:]

def test_saved_frame_worker_wrapper_change_invalidates_platter(keyed):
    stage,key=keyed;before=key('side.platter','auto');path=stage/'methods/side.py'
    source=path.read_text()
    # The compatibility worker entry remains explicitly keyed, including its imports.
    if 'return observe(job)' in source:source=source.replace('return observe(job)','return dict(observe(job), worker_revision=2)')
    else:source=source.replace('return _observe_no_base(job)','return dict(_observe_no_base(job), worker_revision=2)')
    path.write_text(source);assert key('side.platter','auto')!=before

def test_fixed_resolved_value_ignores_unused_schema_default(keyed):
    _,key=keyed;params=PR.resolve({},'side.keep','keep_part',None,{})
    params['min_share']=.25;before=key('side.keep','keep_part',params)
    schema=PR.COMMON['side.keep'];old=schema[0]
    try:
        schema[0]=dataclasses.replace(old,default=.08)
        assert key('side.keep','keep_part',params)==before
    finally:schema[0]=old

def test_emitted_flag_changes_still_invalidate(keyed):
    _,key=keyed;params=PR.resolve({},'side.keep','keep_part',None,{})
    before=key('side.keep','keep_part',params);schema=PR.COMMON['side.keep'];old=schema[0]
    try:
        schema[0]=dataclasses.replace(old,script_default=.05)
        assert key('side.keep','keep_part',params)!=before
        schema[0]=dataclasses.replace(old,legacy_flag='--changed-min-share');params['min_share']=.25
        a=key('side.keep','keep_part',params);schema[0]=old
        assert key('side.keep','keep_part',params)!=a
    finally:schema[0]=old

def test_transitive_observer_body_change_invalidates_platter_only(keyed,tmp_path,monkeypatch):
    _,key=keyed;original=Path(methods.script('no_base.py')).read_text()
    override=tmp_path/'repo/scanner';override.mkdir(parents=True);worker=override/'no_base.py';worker.write_text(original)
    monkeypatch.setenv('POINTYOINK_STAGES_REPO',str(override.parent))
    nodes=[('side.fuse','tsdf'),('side.platter','auto'),('side.keep','keep_part')];before=[key(*n) for n in nodes]
    assert 'if len(paths)<8:' in original
    worker.write_text(original.replace('if len(paths)<8:','if len(paths)<9:'))
    after=[key(*n) for n in nodes]
    assert after[0]==before[0] and after[1]!=before[1] and after[2]==before[2]

def test_changed_resolved_default_invalidates_step(keyed):
    _,key=keyed;before=key('side.keep','keep_part');schema=PR.COMMON['side.keep'];old=schema[0]
    try:
        schema[0]=dataclasses.replace(old,default=.08)
        assert key('side.keep','keep_part')!=before
    finally:schema[0]=old

def test_manually_emitted_side_flags_ignore_unused_schema_mapping(keyed):
    _,key=keyed;params=PR.resolve({},'side.fuse','tsdf',None,{})
    before=key('side.fuse','tsdf',params);schema=PR.COMMON['side.fuse'];old=schema[0]
    try:
        # execute emits --voxel manually and skips this schema entry.
        schema[0]=dataclasses.replace(old,default=.6,legacy_flag='--unused-voxel-schema')
        assert key('side.fuse','tsdf',params)==before
    finally:schema[0]=old

def test_fused_flags_ignore_common_schema_not_used_by_worker(keyed):
    _,key=keyed;params=PR.resolve({},'fused','tsdf',None,{})
    before=key('fused','tsdf',params);schema=PR.COMMON['fused'];index=next(i for i,p in enumerate(schema) if p.name=='point_distance');old=schema[index]
    try:
        schema[index]=dataclasses.replace(old,default=.8,legacy_flag='--unused-common-schema')
        assert key('fused','tsdf',params)==before
    finally:schema[index]=old

def test_inherited_executor_behavior_change_invalidates_only_its_users(keyed,monkeypatch):
    _,key=keyed
    class InheritedMethod(methods.Method):
        node='fixture';name='inherited';version=1
        def run(self,ctx):return {'outputs':{'mesh':'before.ply'}}
    def inherited_key():
        st=SimpleNamespace(method=InheritedMethod(),params={},produces='mesh',kind='fixture',id='fixture',inputs={},leaf={})
        e=Engine.__new__(Engine);e._codes={};e._fcode={};e._libs={};e._lib=lambda _:'fixture-version'
        return e.step_key(st,{})
    before=inherited_key();independent=key('side.keep','keep_part')
    assert InheritedMethod().execute(None)['outputs']['mesh']=='before.ply'
    def changed_execute(self,ctx):
        return {'outputs':{'mesh':'changed-executor.ply'}}
    monkeypatch.setattr(methods.Method,'execute',changed_execute)
    assert InheritedMethod().execute(None)['outputs']['mesh']=='changed-executor.ply'
    assert inherited_key()!=before, 'The selected inherited executor is code this method runs'
    assert key('side.keep','keep_part')==independent, 'An overriding executor does not use Method.execute'


def test_inherited_argv_helper_behavior_change_invalidates(keyed,monkeypatch):
    _,_=keyed
    class InheritedMethod(methods.Method):
        node='fixture';name='inherited';version=1
        def run(self,ctx):return {'outputs':{'mesh':'before.ply'}}
    def digest():
        e=Engine.__new__(Engine);e._codes={};e._fcode={};e._libs={};e._lib=lambda _:'fixture-version'
        return e.code_digest(SimpleNamespace(method=InheritedMethod(),params={},produces='mesh'))
    ctx=SimpleNamespace(call=lambda *_args,**_kwargs:'')
    before=digest();assert InheritedMethod().execute(ctx)['outputs']=={'mesh':'before.ply'}
    def changed_argv(self,ctx):return [['changed-worker']]
    monkeypatch.setattr(methods.Method,'argv',changed_argv)
    assert InheritedMethod().execute(ctx)['outputs']=={}
    assert digest()!=before
    argv_key=digest()
    monkeypatch.setattr(methods.Method,"outputs",{"mesh":"changed-output-map.ply"})
    assert InheritedMethod().execute(ctx)["outputs"]=={"mesh":"changed-output-map.ply"}
    assert digest()!=argv_key
