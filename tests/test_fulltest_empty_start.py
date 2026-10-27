"""Opening the workspace should not silently create an empty scan project."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XMODIFIERS", "@im=none")
os.environ.setdefault("POINTYOINK_NO_DEVICE", "1")
os.environ.setdefault("POINTYOINK_CONFIG", "/tmp/pointyoink-fulltest-empty-start-test-config.json")

from PySide6.QtWidgets import QApplication

from app2.qt import theme
from app2.qt.shell import MainWindow


def test_workspace_defers_new_project_until_scan_is_chosen(tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    calls = []

    def make_session(_bridge, **kwargs):
        calls.append(kwargs)
        (tmp_path / "ProjectLiveNew" / "data").mkdir(parents=True)
        raise RuntimeError("offline test session")

    window = MainWindow(make_session, gl=False, projects_dir=str(tmp_path), defer_new_project=True)
    window.show(); app.processEvents()
    try:
        assert not calls
        assert list(tmp_path.iterdir()) == []
        window.workspace.ribbon_tools["Scan"].click(); app.processEvents()
        assert calls == [dict(project=None, last=False)]
        assert (tmp_path / "ProjectLiveNew" / "data").is_dir()
    finally:
        window._really = True
        window.build.shutdown()
        window.close(); window.deleteLater(); app.processEvents()
