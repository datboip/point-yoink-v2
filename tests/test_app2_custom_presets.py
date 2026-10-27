"""Saved preferences and fixed-size recipe sheet use isolated paths."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_presets_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from app2.builder import preset_store as store
from app2.qt.onetap import OneTapDialog


def test_saved_preferences_roundtrip_without_scan_data(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    settings=dict(method='plus',detail=.25,cleanup=1.5,smooth=2,finish=True)
    store.save_preset('Thin bracket',settings)
    assert store.load_presets()=={'Thin bracket':settings}
    fragment=store.recipe_fragment(settings)
    assert store.validate_fragment(fragment)==fragment
    assert all(op['tool'] not in ('delete','fin') for ops in fragment['ops'].values() for op in ops)
    assert not (tmp_path/'config.json').exists()


def test_reject_invalid_and_source_specific_settings(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    for invalid in [dict(detail=0),dict(smooth=1.1),dict(cleanup=float('nan')),dict(method='unknown'),dict(scans='secret')]:
        with pytest.raises(ValueError):store.save_preset('Bad',invalid)
    fragment=store.recipe_fragment(store.DEFAULTS)
    fragment['ops']['meshed'].append(dict(tool='delete',params={'mask':'scan.ply'}))
    with pytest.raises(ValueError):store.validate_fragment(fragment)
    assert not store.store_path().exists()


def test_custom_signals_saved_choice_and_fixed_regions(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=QApplication.instance() or QApplication([])
    dialog=OneTapDialog(); dialog.show(); app.processEvents()
    size=dialog.size(); changed=[]; run=[]
    dialog.custom_changed.connect(changed.append); dialog.custom_ok.connect(run.append)
    for key in dialog.btns:
        dialog.pick(key); app.processEvents(); assert dialog.size()==size
    dialog.pick('custom',True); app.processEvents(); size=dialog.size()   # Custom opens the larger settings page, fixed from here
    dialog.detail.setValue(.25)
    assert changed[-1]['fragment']['set']['fused.point_distance']==.25
    dialog.name.setText('Small parts'); dialog.save_current(); assert 'saved:Small parts' in dialog.btns
    dialog.pick('saved:Small parts',True)
    dialog.show_summary(dict(preset=dialog.current,fusion='Standard 0.25 mm',ftools='none',mesh='As fused',mtools='none',to_run=1,est_s=30))
    dialog.ok_b.click(); assert run[-1]['fragment']==changed[-1]['fragment']
    assert dialog.ok_b.text()=='Yoink it' and dialog.size()==size
    assert dialog.ok_b.geometry().bottom() <= dialog.height()-8
    dialog.close()


def test_corrupt_presets_not_overwritten(tmp_path,monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    store.store_path().write_text('broken')
    with pytest.raises(ValueError): store.save_preset('Good',store.DEFAULTS)
    assert store.store_path().read_text()=='broken'


def test_cleanup_runs_on_fused_and_old_saves_still_load():
    """Loose pieces go on Fused (Fused showed floaters with them in Meshed); a preset saved the old way still loads."""
    from app2.builder.preset_store import recipe_fragment, validate_fragment
    f = recipe_fragment(dict(method='tsdf', detail=0.4, cleanup=3.0, smooth=2, finish=False))
    assert [o['tool'] for o in f['ops']['fused.tsdf']] == ['isolation'] and [o['tool'] for o in f['ops']['meshed']] == ['smooth']
    old = {'set': f['set'], 'reset_nodes': ['fused', 'meshed'],
           'ops': {'fused.tsdf': [], 'meshed': [dict(tool='isolation', params={'rate': 3.0}), dict(tool='smooth', params={'strength': 30.0, 'times': 2})]}}
    assert validate_fragment(old) == f
