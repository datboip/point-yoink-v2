"""Regression checks for Build inside the shared Workspace shell.

All windows run offscreen, with an isolated config and no scanner session.
"""
import os
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XMODIFIERS'] = '@im=none'
os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp(prefix='pointyoink_layout_')) / 'config.json')

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication
from app2.qt import theme
from app2.qt.shell import MainWindow
from app2.qt.onetap import OneTapDialog


@pytest.fixture(params=[(1280, 800), (1688, 1044)])
def window(request):
    app = QApplication.instance() or QApplication([])
    theme.install(app)
    def offline(*args, **kwargs):
        raise RuntimeError('Isolated test: no scanner')
    win = MainWindow(offline, gl=False)
    win.resize(*request.param); win.show_step(1); win.build.show_stage('meshed'); win.show()
    app.processEvents()
    yield win
    win.build.shutdown(); win.close(); win.deleteLater(); app.processEvents()


def rect(widget, root):
    pos = widget.mapTo(root, QPoint(0, 0))
    return (pos.x(), pos.y(), widget.width(), widget.height())


def test_shared_navigation_and_large_scene(window):
    page = window.build
    left = rect(page.left, window)
    view = rect(page.vp, window); right = rect(page.right, window)
    assert not window.rail.isVisible()
    assert window.workspace.ribbon_scroll.isVisible()
    assert left[0] + left[2] < view[0]
    assert view[0] + view[2] < right[0]
    assert right[0] + right[2] < window.width()
    assert view[3] > window.height() * .55


def test_finish_stays_below_each_tool(window):
    page = window.build; panel = page.panel['meshed']; finish = panel.blocks['Finish for print']
    scene = rect(page.vp, window)
    for name in ('Mesh', 'Isolation', 'Smooth', 'Remove slivers'):
        panel.show_tab(name); QApplication.processEvents()
        active = panel.body()
        assert rect(finish, panel)[1] >= rect(active, panel)[1] + active.height() + 12
        assert rect(page.vp, window) == scene


def test_compare_uses_the_same_scene(window):
    view = window.build.vp; before = rect(view.area, window)
    for enabled in (True, False, True):
        view.set_comparing(enabled); QApplication.processEvents()
        assert rect(view.area, window) == before
        if enabled:
            assert view.before.isVisible()
            assert abs(view.before.width() - view.main.width()) <= 1


def test_preset_sheet_keeps_its_size(window):
    dialog = OneTapDialog(window); dialog.show(); QApplication.processEvents()
    size = dialog.size()
    labels = dict((key, label) for key, label, hint in dialog.choices)
    assert [labels[k] for k in ('revo', 'revo-device', 'plus')] == ['Detail Yoink', 'Deep Clean', 'Sharp Yoink']
    assert dialog.ok_b.text() == 'Yoink it'
    for key in dialog.btns:
        dialog.pick(key); QApplication.processEvents()
        assert dialog.size() == size
    assert dialog.width() <= window.width() and dialog.height() <= window.height()
    dialog.hide(); dialog.deleteLater()


def test_finish_does_not_move_when_mesh_state_and_text_change(window):
    page=window.build;panel=page.panel['meshed'];panel.show_tab('Mesh')
    finish=panel.blocks['Finish for print'];mesh=panel.body('Mesh')
    panel.show_if({'mesh'});page._poisson_in_sight(panel,False);QApplication.processEvents()
    before=rect(finish,panel);width=page.scroll.viewport().width()
    for on in (True,False,True):
        panel.show_if({'mesh','poisson'} if on else {'mesh'})
        page._poisson_in_sight(panel,on)
        finish.controls[0].set_hint('Fewer triangles make a smaller file. The model has 250000 now.')
        finish.set_note('Saved a-long-project-name_print.stl')
        QApplication.processEvents()
        assert rect(finish,panel)==before
        assert page.scroll.viewport().width()==width


def test_empty_message_shows_and_hides():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import BuildViewPanel
    app = QApplication.instance() or QApplication([])
    vp = BuildViewPanel(False); vp.resize(900, 700); vp.show(); app.processEvents()
    vp.set_empty("Nothing to show"); app.processEvents()
    assert vp.empty.isVisible() and vp.empty.text() == "Nothing to show"
    vp.set_empty(None); assert not vp.empty.isVisible()
