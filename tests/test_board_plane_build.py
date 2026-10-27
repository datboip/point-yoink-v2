import copy
import json
import types
import pytest
from scanner.board_plane import generic, supported
from scanner.stages.methods.side import PlatterAuto


def plane():
    return dict(n=[0,0,1], d=0., source='saved-ringed-board-depth', provenance=dict(version=1,
        fit=dict(frame_fits=30, stable_frame_share=1., inlier_share=.98, rms_mm=.4, dot_median_height_mm=.5),
        heldout=dict(frames=30, stable_frame_share=1., supported_frame_share=1.)))


def test_live_plane_and_generic_auto_are_distinguished():
    assert generic({}) and generic(dict(n=[0,0,1],d=0,points=100,tilt_deg=.2))
    assert not generic(dict(n=[0,0,1],d=0,frames=30))
    assert not generic(plane())


def test_no_saved_ir_skips_board_fitter(tmp_path):
    (tmp_path/'cache').mkdir()
    calls=[]
    ctx=types.SimpleNamespace(out_dir=str(tmp_path/'out'),step=types.SimpleNamespace(side='s'),
        project=types.SimpleNamespace(folder=lambda _:str(tmp_path)),call=lambda *a,**kw:calls.append(a))
    assert PlatterAuto().board_plane(ctx) is None and calls==[]


@pytest.mark.parametrize('section,field,value',[('fit','rms_mm',1.),('fit','frame_fits',9),
    ('fit','inlier_share',.5),('heldout','frames',9),('heldout','stable_frame_share',.5),
    ('heldout','supported_frame_share',.5)])
def test_support_gate_rejects_insufficient_evidence(section,field,value):
    p=plane(); assert supported(p)
    p['provenance'][section][field]=value; assert not supported(p)
    assert not supported(dict(n=[0,0,1],d=0,source='saved-ringed-board-depth'))


@pytest.mark.parametrize('trusted',[False,True])
def test_large_cut_only_bypasses_guard_with_measured_board_support(tmp_path,trusted,monkeypatch):
    p=plane()
    if not trusted: p['provenance']['heldout']['supported_frame_share']=.2
    path=tmp_path/'plane.json'; path.write_text(json.dumps(p))
    method=PlatterAuto(); monkeypatch.setattr(method,'plane_arg',lambda ctx:str(path))
    warnings=[]
    ctx=types.SimpleNamespace(out_dir=str(tmp_path),step=types.SimpleNamespace(side='s',n=11),
        project=types.SimpleNamespace(folder=lambda _:str(tmp_path)), params=dict(margin=1.,max_tilt=8.),
        inputs={'raw':types.SimpleNamespace(files={'mesh':'raw.ply'})},
        call=lambda *a,**k:'STAGE wrong_plane {"removed_share":0.67}\n',
        legacy=lambda *a,**k:None,warn=warnings.append)
    result=method.execute(ctx)
    assert (result.get('alias_of')=='raw') is (not trusted)
    assert bool(warnings) is (not trusted)
    if trusted: assert result['stats']['source']=='board dots'
