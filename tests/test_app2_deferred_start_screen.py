"""With no session at launch (the app waits for Scan, Open or New), the startup screen must not stay up over the workspace."""
import os
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XMODIFIERS'] = '@im=none'
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_deferred_')) / 'config.json')

from PySide6.QtWidgets import QApplication
from app2.qt import theme
from app2.qt.shell import MainWindow


class Session:
    def __init__(self, *a, **k): pass
    def start(self): pass
    def close(self, then=None): pass


def test_deferred_launch_hides_the_startup_screen_and_starts_no_session():
    app = QApplication.instance() or QApplication([]); theme.install(app)
    made = []
    win = MainWindow(lambda bridge, **over: made.append(over) or Session(), gl=False, defer_new_project=True)
    win.show(); app.processEvents()
    assert made == [] and not win.startup.isVisible()
    win.start_session(project="/nowhere", last=False); app.processEvents()
    assert len(made) == 1 and not win.startup.isVisible()           # the first project opened must not flash the splash either
    win.deleteLater(); app.processEvents()
