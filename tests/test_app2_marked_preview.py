"""Marked-only is a temporary display filter and always restores the whole mesh."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import app2,numpy as np
from PySide6.QtWidgets import QApplication
from app2.qt.build_view import BuildViewPanel


def test_marked_filter_menu_enable_refresh_and_reset():
    app=QApplication.instance() or QApplication([])
    panel=BuildViewPanel(True);panel.set_colours([('shaded','Shaded')],'shaded');view=panel.main.view
    item=dict(key='mesh',kind='mesh',V=np.array([[0.,0,0],[1,0,0],[0,1,0]]),F=np.array([[0,1,2]]),N=np.tile([0.,0,1],(3,1)))
    view.show_items([item],edit='mesh');panel._refresh_marked_action()
    action=panel._marked_action;assert not action.isEnabled()
    before=view.cam.view.copy();geometry=item['V'].copy()
    view.show_would_go(np.array([True]));panel._refresh_marked_action();assert action.isEnabled()
    action.trigger();assert view.marked_only and panel.colour_btn.text().startswith('Marked only')
    panel.set_colours([('shaded','Shaded')],'shaded');assert action.isChecked()
    assert np.array_equal(view.cam.view,before) and np.array_equal(item['V'],geometry)
    view.clear_would_go();assert not view.marked_only and not action.isEnabled()
    assert panel.colour_btn.text().startswith('Shaded')
    view.show_would_go(np.array([True]));view.set_marked_only(True)
    view.show_items([item],edit='mesh',frame=False)
    assert not view.marked_only and view.would_go is None
    panel.deleteLater();app.processEvents()
