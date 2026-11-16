# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import page without a scanner: a click on a project row ticks it, a second click clears it; the folder follows Preferences."""
import os, tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp()) / 'config.json')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def _page(lib):
    QApplication.instance() or QApplication([])
    from app2.qt.device_projects import DeviceProjects
    return DeviceProjects(lib)


def test_row_click_ticks_and_clears(tmp_path):
    d = _page(str(tmp_path / 'lib')); d.mode = 'list'
    d.worker_event(dict(event='project', name='ProjectLive11012026101010', sides=2, mesh=True))
    item = d.files.topLevelItem(0)
    d.files.itemClicked.emit(item, 0); assert d.picked(item) == 'models'
    d.files.itemClicked.emit(item, 0); assert d.picked(item) is None
    item.setCheckState(5, Qt.Checked); assert d.picked(item) == 'full'
    d.files.itemClicked.emit(item, 3); assert d.picked(item) == 'full'          # the tick cells keep their own clicks


def test_wifi_rows_ignore_clicks(tmp_path):
    d = _page(str(tmp_path / 'lib')); d.mode = 'wifi'
    d.worker_event(dict(event='project', name='ProjectLive11012026101011', sides=1, mesh=False))
    item = d.files.topLevelItem(0); d.files.itemClicked.emit(item, 0)
    assert item.data(3, Qt.CheckStateRole) is None


def test_folder_follows_preferences_only_from_the_default(tmp_path):
    from app2 import library_dir
    library_dir.set_default(str(tmp_path / 'first'))
    d = _page(None); assert d.library.endswith('first')
    library_dir.set_default(str(tmp_path / 'second')); d.show(); assert d.library.endswith('second')
    d.hide(); other = _page(str(tmp_path / 'given'))
    library_dir.set_default(str(tmp_path / 'third')); other.show(); assert other.library.endswith('given')
    d.close(); other.close()
