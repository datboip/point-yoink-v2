# SPDX-License-Identifier: AGPL-3.0-or-later
from app2 import update_check as u


def test_parse_and_compare():
    assert u.parse('v2.0.4') == (2, 0, 4) and u.parse('2.0.10') == (2, 0, 10) and u.parse('2.0.4-pre') is None
    assert u.compare('2.0.3', 'v2.0.4') == 'newer'
    assert u.compare('2.0.9', 'v2.0.10') == 'newer'
    assert u.compare('2.0.4', 'v2.0.4') == 'current' and u.compare('2.1.0', 'v2.0.9') == 'current'
    assert u.compare('2.0.3', '') == 'unknown' and u.compare('2.0.3', 'nightly') == 'unknown'


def test_latest_survives_a_dead_network():
    assert u.latest(lambda: 'v2.0.4') == 'v2.0.4'
    def boom(): raise OSError('offline')
    assert u.latest(boom) == ''


def test_once_a_day_and_once_per_release():
    from app2 import update_check as u
    assert u.due(dict(update_notice=True, update_checked=0), 100000)
    assert not u.due(dict(update_notice=True, update_checked=100000), 100000 + 3600)
    assert not u.due(dict(update_notice=False, update_checked=0), 10 ** 9)
    assert u.worth_telling('2.0.5', 'v2.0.6', '') and not u.worth_telling('2.0.5', 'v2.0.6', 'v2.0.6')
    assert not u.worth_telling('2.0.6', 'v2.0.6', '') and not u.worth_telling('2.0.5', '', '')


def test_notice_at_start_says_it_once(tmp_path, monkeypatch):
    import os, time
    monkeypatch.setenv('POINTYOINK_CONFIG', str(tmp_path / 'config.json'))
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from app2 import general_prefs as G
    from app2.qt.shell import MainWindow
    win = MainWindow(lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')), gl=False, projects_dir=str(tmp_path))
    said = []; win.say = lambda text, *a, **k: said.append(text)
    assert win.auto_update_check(latest=lambda: 'v99.0.0', now=10 ** 9)
    for _ in range(50):
        app.processEvents(); time.sleep(0.02)
        if said: break
    assert said and '99.0.0 is available' in said[0] and G.get('update_seen') == 'v99.0.0'
    assert not win.auto_update_check(latest=lambda: 'v99.0.0', now=10 ** 9 + 60)          # not again the same day
    win.close()
