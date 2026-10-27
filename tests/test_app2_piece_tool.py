"""The Piece tool (2026-10-25): one click selects the whole connected piece under it; a click on nothing says so."""
import numpy as np
from types import SimpleNamespace as NS
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication


def _view(seed_ids):
    from app2.qt.edit_view import EditView
    QApplication.instance() or QApplication([])
    v = EditView.__new__(EditView)
    F = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])                 # faces 0 and 1 share an edge; face 2 is its own piece
    v.items = {'m': dict(kind='mesh', F=F)}; v.edit_key = 'm'; v.selection = np.zeros(3, bool); v.selection_mode = 'replace'; v._subtract = False
    v._face_graph_cache = None; v.would_go = None; v.hover_mask = None; v.tools_on = True; v.n = 3; v.gone = None; v.gone_stack = []
    v._px = lambda: (20, 20); v.devicePixelRatioF = lambda: 1.0
    v.select_mask = lambda mask, through: np.isin(np.arange(3), seed_ids)
    v._push_flags = lambda: None; v.clear_hover = lambda: None
    return v


def test_click_takes_the_connected_piece():
    v = _view([0]); m = v._piece_region_at(QPointF(5, 5))
    assert m.tolist() == [True, True, False]
    v._piece_pick(QPointF(5, 5)); assert v.selection.tolist() == [True, True, False]
    v.selection_mode = 'add'; v.select_mask = lambda mask, through: np.isin(np.arange(3), [2])
    v._piece_pick(QPointF(5, 5)); assert v.selection.tolist() == [True, True, True]


def test_click_on_nothing_says_so():
    v = _view([]); heard = []
    from app2.qt.edit_view import EditView
    class Sig:
        def emit(self, x): heard.append(x)
    v.selected_nothing = Sig()
    v._piece_pick(QPointF(5, 5)); assert heard == ['piece'] and not v.selection.any()


def test_deleted_faces_hide_until_the_rebuild_and_undo_shows_the_last_group():
    v = _view([0]); pushed = []; v._push_flags = lambda: pushed.append(1)
    v.selection[:] = [True, True, False]
    assert v.hide_selected() == 2 and v.hidden_count() == 2 and not v.selection.any()
    v.selection[:] = [False, False, True]; v.hide_selected(); assert v.hidden_count() == 3
    v.unhide_last(); assert v.gone.tolist() == [True, True, False]
    v.unhide_last(); assert v.gone is None and v.hidden_count() == 0


def test_a_hidden_face_cannot_be_picked_again():
    v = _view([0]); v._push_flags = lambda: None; v._subtract = False
    v.selection[:] = [True, False, False]; v.hide_selected()
    v._apply_selection(np.array([True, True, False])); assert v.selection.tolist() == [False, True, False]


def test_delete_in_edit_hides_now_and_rebuilds_once_later():
    from app2.qt.build_step import BuildStepPage
    calls = []; notices = []
    view = NS(selection=np.array([True, False, True]), mask_points=lambda: np.zeros((2, 3), np.float32), hide_selected=lambda: calls.append('hide') or 2,
              clear_selection=lambda: calls.append('clear'))
    page = NS(gl=True, stage='meshed', running=lambda: False, vp=NS(edit_workspace=True, view=view), sel_path='p', _view_path=lambda: 'p', shown={'path': 'p'},
              _side=lambda: {'id': 's'}, _run_to=lambda t: t, notice=lambda text, level=None, secs=None: notices.append(text),
              call=lambda name, *a, then=None, **k: calls.append(name) or then({'got': 2}), start_run=lambda run: calls.append('run') or True,
              _edit_run=lambda: 'edit', set_tool=lambda t: calls.append('tool:' + t), pending_changed=lambda: BuildStepPage.pending_changed(page))
    BuildStepPage.delete_selection(page)
    assert calls == ['h_mask_delete', 'hide'] and page.pending_edits == 1 and 'wait for one rebuild' in notices[-1]
    BuildStepPage.delete_selection(page); assert page.pending_edits == 2
    assert BuildStepPage.flush_pending_edits(page) and calls[-1] == 'run' and page.pending_edits == 0
    assert not BuildStepPage.flush_pending_edits(page)                      # nothing waiting: no run


def test_shift_adds_even_in_replace_mode():
    v = _view([0]); v._piece_pick(QPointF(5, 5)); assert v.selection.tolist() == [True, True, False]
    v.select_mask = lambda mask, through: np.isin(np.arange(3), [2]); v._force_add = True
    v._piece_pick(QPointF(5, 5)); assert v.selection.tolist() == [True, True, True]      # Replace mode, Shift held: kept the first piece
