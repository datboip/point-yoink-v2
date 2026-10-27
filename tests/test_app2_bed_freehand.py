"""Freehand (no bed): the last entry of the bed list; picking it drops the bed and the cut, the project remembers it, the
card says so, and the session turns Remove base off (a freehand mode: the table can be moved or
the camera moved around it)."""
import os, sys, json
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scanner'))


def test_pick_freehand_clears_the_bed_and_is_remembered(tmp_path, monkeypatch):
    import bed as bedlib
    from app2.scan.bed_adapter import ScannerBed
    from app2.scan.bedhook import FREEHAND
    monkeypatch.setattr(bedlib, 'config_dir', lambda: str(tmp_path / 'cfg')); (tmp_path / 'cfg').mkdir()
    a = ScannerBed.__new__(ScannerBed); a.learner = None; a.closing = False; a.project = str(tmp_path); a.bed = object(); a.plane = None; a.prior = None
    (tmp_path / 'bed.json').write_text('{}')
    assert a.profiles()[-1] == FREEHAND
    msg = a.pick_profile(FREEHAND, [])
    assert 'Freehand' in msg and a.bed is None and a.plane is None and a.freehand and a.profile == FREEHAND
    assert not (tmp_path / 'bed.json').exists() and (tmp_path / 'bed_prev.json').exists()
    assert bedlib.read_project_profile(str(tmp_path))['kind'] == 'none' and bedlib.picked_profile() == FREEHAND
    assert a.base_text(NS(bed=None), True) == 'base: none (freehand)'
    b = ScannerBed.__new__(ScannerBed); b.bed = b.plane = b.pending = None; b.profile = None
    b._choose(str(tmp_path)); assert b.freehand and b.profile == FREEHAND


def test_bed_card_and_session_pick():
    from app2.scan.ui_model import ScanState, bed_card
    from app2.scan.bedhook import FREEHAND
    from app2.scan.session import ScanSession
    st = ScanState(bed_available=True, bed=None, bed_profile=FREEHAND, n_sides=0)
    assert bed_card(st)['sub'].startswith('Freehand')
    s = ScanSession.__new__(ScanSession); log = []; off = []
    s.recording = s.busy = s.building = False; s.bed = NS(learning=False, pick_profile=lambda n, sides: 'ok'); s.project = NS(sides=[])
    s._log = log.append; s.remove_base = True; s._set_remove_base = lambda on: off.append(on); s._reset_mode_detection = lambda: None
    s.picked = None; s._emit_sides = lambda: None; s._emit_state = lambda: None
    s._pick_bed(FREEHAND); assert off == [False] and any('Remove base' in l for l in log)
