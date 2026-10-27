"""Disposable scanner folders exercise the shared USB/Wi-Fi project copy path."""
from pathlib import Path

import pytest

from app2.transfer.projects import import_projects, list_projects


def scanner(tmp_path):
    root = tmp_path / 'device'; side = root / 'Projects' / 'ProjectLive123' / 'data' / 'side1'
    side.mkdir(parents=True)
    (side / 'property.rvproj').write_text('{}')
    (side / 'cache').mkdir(); (side / 'cache' / 'frame_000_0000.dph').write_bytes(b'raw-data')
    (side.parent.parent / 'model.ply').write_bytes(b'ply')
    return root


def test_inventory_and_full_import_keep_both(tmp_path):
    root = scanner(tmp_path); library = tmp_path / 'library'; events = []
    rows = list(list_projects(root, library))
    assert [(r['name'], r['sides'], r['mesh'], r['exists']) for r in rows] == [('ProjectLive123', 1, True, False)]
    note = lambda kind, **data: events.append((kind, data))
    for _ in range(2): import_projects(root, library, ['ProjectLive123'], note)
    assert (library / 'ProjectLive123' / 'data' / 'side1' / 'cache' / 'frame_000_0000.dph').read_bytes() == b'raw-data'
    assert [kind for kind, _ in events if kind in ('saved', 'exists')] == ['saved', 'exists']   # a re-import skips by default
    assert not (library / 'ProjectLive123 (1)').exists()
    assert list(list_projects(root, library))[0]['exists']
    import_projects(root, library, ['ProjectLive123'], note, existing='copy')
    assert (library / 'ProjectLive123 (1)' / 'model.ply').read_bytes() == b'ply'
    assert len([kind for kind, _ in events if kind == 'saved']) == 2
    assert not list(library.glob('.pointyoink-import-*'))


def test_symlink_and_interrupted_copy_never_publish(tmp_path):
    root = scanner(tmp_path); library = tmp_path / 'library'
    link = root / 'Projects' / 'ProjectLive123' / 'data' / 'side1' / 'escape'
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match='link'): import_projects(root, library, ['ProjectLive123'])
    link.unlink()
    def interrupt(kind, **_data):
        if kind == 'progress': raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): import_projects(root, library, ['ProjectLive123'], interrupt, chunk_size=2)
    assert not (library / 'ProjectLive123').exists()
    assert not list(library.glob('.pointyoink-import-*'))


def test_destination_cannot_be_on_scanner(tmp_path):
    root = scanner(tmp_path)
    with pytest.raises(ValueError, match='outside scanner'):
        import_projects(root, root / 'copied', ['ProjectLive123'])


def test_sizes_are_cached_until_the_project_changes(tmp_path, monkeypatch):
    import json
    from app2.transfer.projects import project_sizes
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'cfg' / 'config.json'))
    root = scanner(tmp_path); proj = root / 'Projects' / 'ProjectLive123'
    (proj / 'ProjectLive123.revo').write_text(json.dumps({'edit_time': '1', 'model_mesh_count': 1}))
    runs = []
    for _ in range(2): got = []; project_sizes(root, lambda kind, **d: got.append(d)); runs.append([d for d in got if d['done']][0])
    assert not runs[0].get('cached') and runs[1].get('cached') and runs[0]['full'] == runs[1]['full']
    (proj / 'ProjectLive123.revo').write_text(json.dumps({'edit_time': '2', 'model_mesh_count': 1}))   # edited on the scanner
    got = []; project_sizes(root, lambda kind, **d: got.append(d))
    assert not [d for d in got if d['done']][0].get('cached')


def test_wifi_received_projects_move_straight_into_the_library(tmp_path, monkeypatch):
    import os, time
    from PySide6.QtWidgets import QApplication
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'cfg' / 'config.json'))
    app = QApplication.instance() or QApplication([])
    from app2.qt.device_projects import DeviceProjects
    lib = tmp_path / 'lib'; stage = lib / '.wifi-incoming' / 'abc'; (stage / 'ProjectX' / 'data' / 'side1').mkdir(parents=True)
    d = DeviceProjects(str(lib)); d._t0 = time.monotonic(); d._place_received(str(stage))
    assert (lib / 'ProjectX' / 'data' / 'side1').is_dir() and not stage.exists()     # moved, holding folder gone
    assert d.just_imported and 'Received 1 project' in d.status.text()


def test_screenshots_and_recordings_tabs_list_and_pick(tmp_path, monkeypatch):
    from datetime import datetime
    from PySide6.QtWidgets import QApplication
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'cfg' / 'config.json'))
    app = QApplication.instance() or QApplication([])
    from app2.qt.device_projects import DeviceProjects, capture_time, capture_text
    assert capture_time('09102026222015.mp4') == datetime(2026, 9, 10, 22, 20, 15) and capture_time('notadate.png') is None
    assert capture_text('09102026222015.mp4') == 'Sep 10, 2026, 10:20:15 PM'
    d = DeviceProjects(str(tmp_path / 'lib')); d.root = str(tmp_path / 'device')
    rows = [dict(relpath='Screenshots/09102026030754.png', name='09102026030754.png', size=5, kind='image', modified=1),
            dict(relpath='ScreenRecordings/09102026222015.mp4', name='09102026222015.mp4', size=9, kind='video', modified=2)]
    d._caps_ready(rows); d.tabs.setCurrentIndex(1)
    assert d.caps.count() == 1 and d.caps_picked() == []          # only the screenshot on its tab
    d.select_all(True); assert d.caps_picked() == ['Screenshots/09102026030754.png'] and 'Import 1 screenshot' in d.copy_b.text()
    d.tabs.setCurrentIndex(2); assert d.caps.count() == 1 and d.caps_picked() == []
    d.tabs.setCurrentIndex(0); assert d.caps.isHidden() and not d.files.isHidden()
    import threading                    # the thumbnail reader may still be inside OpenCV; exiting under it aborts the process
    for t in threading.enumerate():
        if t.name.startswith('import-'): t.join(10)


def test_capture_folders_are_kept_and_copies_go_flat(tmp_path, monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'cfg' / 'config.json'))
    from app2 import library_dir
    from app2.transfer.captures import copy_captures
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    assert library_dir.capture_dir('screenshots') == str(tmp_path / 'home/PointYoink/Screenshots') and library_dir.capture_dir('recordings') == str(tmp_path / 'home/PointYoink/Recordings')
    library_dir.set_default(str(tmp_path / 'proj')); library_dir.set_capture_dir('screenshots', str(tmp_path / 'shots'))
    assert library_dir.default_dir() == str(tmp_path / 'proj') and library_dir.capture_dir('screenshots') == str(tmp_path / 'shots')   # one setting never wipes another
    root = tmp_path / 'dev'; (root / 'Screenshots').mkdir(parents=True); (root / 'Screenshots' / '09102026030754.png').write_bytes(b'png')
    copy_captures(root, tmp_path / 'shots', ['Screenshots/09102026030754.png'], flat=True)
    assert (tmp_path / 'shots' / '09102026030754.png').read_bytes() == b'png'


def test_everything_saves_inside_one_pointyoink_folder(tmp_path, monkeypatch):
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'cfg' / 'config.json')); monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    from app2 import library_dir
    home = tmp_path / 'home'
    assert {k: library_dir.folder(k) for k in library_dir.FOLDERS} == {k: str(home / 'PointYoink' / n) for k, (_key, n, _old) in library_dir.FOLDERS.items()}
    assert library_dir.export_path('part_print.stl') == str(home / 'PointYoink/Exports/part_print.stl') and not (home / 'PointYoink').exists()
    library_dir.ensure_parent(library_dir.export_path('part_print.stl')); assert (home / 'PointYoink/Exports').is_dir()
    (home / 'revopoint-scans-models').mkdir()                       # an older install keeps its library until the user moves it
    assert library_dir.default_dir() == str(home / 'revopoint-scans-models')
    (home / 'PointYoink/Scans').mkdir()
    assert library_dir.default_dir() == str(home / 'PointYoink/Scans')
    library_dir.set_home(str(tmp_path / 'elsewhere')); library_dir.set_folder('exports', str(tmp_path / 'mine'))
    assert library_dir.folder('recordings') == str(tmp_path / 'elsewhere/Recordings') and library_dir.folder('exports') == str(tmp_path / 'mine')
