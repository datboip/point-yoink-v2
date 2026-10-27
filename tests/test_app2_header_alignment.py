"""Workspace changes keep shared header and status positions steady."""
import os
import tempfile
from pathlib import Path
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-header-")) / "config.json")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt import theme
from app2.qt.shell import MainWindow

@pytest.mark.parametrize("size", [(1100, 800), (1280, 800), (1688, 1044)])
def test_workspace_header_and_status_alignment(size, tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    def offline(*args, **kwargs):
        raise RuntimeError("No device in header test")
    win = MainWindow(offline, gl=False, projects_dir=str(tmp_path))
    win.resize(*size); win.show(); app.processEvents()
    win.header.set_project("28 Sep 2026, 05:21", "ProjectLive09282026052140", "Test project")
    try:
        positions = []
        for step in (0, 1, 2, 1, 0):
            win.show_step(step); app.processEvents()
            positions.append(tuple((w.x(), w.y(), w.width(), w.height()) for w in
                             (win.header.step, win.header.project, win.header.folder, win.header.source)))
            assert win.header.project.geometry().right() < win.header.folder.geometry().left()
            assert win.scan.status.isVisible() == (step in (0, 2))       # Build has no scanner status line
            assert win.scan.status.parentWidget() is not win.scan
            if step != 1: assert win.stack.geometry().bottom() < win.scan.status.geometry().top()
        assert all(p == positions[0] for p in positions), repr(positions)
        win.show_step(2)
        assert "Select faces" in win.scan.status.hint.text()
        win.show_step(0)
        assert win.scan.status._context_hint is None
    finally:
        win._really = True; win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()
