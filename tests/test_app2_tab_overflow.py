"""Tool tabs scroll in one 34 px row without widening the tool panel."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink_tabs_'))/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea
from PySide6.QtGui import QFontMetrics,QFont
from app2.qt.tool_panel import TabBar,EXTRA_QSS


@pytest.mark.parametrize('width',[330,400])
def test_all_tabs_reachable_with_arrows_without_widening_panel(width):
    app=QApplication.instance() or QApplication([])
    window=QWidget(); window.resize(width,300); window.setStyleSheet(EXTRA_QSS)
    outer=QVBoxLayout(window); outer.setContentsMargins(0,0,0,0)
    scroll=QScrollArea(); scroll.setWidgetResizable(True); outer.addWidget(scroll)
    panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(14,10,14,10)
    basic=['Mesh','Isolation','Simplify','Smooth']; extras=['Remove slivers','Fill holes','Add fin','Cut gap','Overlap Detection']
    tabs=TabBar(basic); layout.addWidget(tabs); layout.addStretch()
    tabs.set_more([(name,True) for name in extras]); scroll.setWidget(panel)
    window.show(); app.processEvents()
    assert tabs.height()==34 and panel.width()<=scroll.viewport().width()
    assert not hasattr(tabs,'menu')
    assert tabs.scroll.horizontalScrollBarPolicy()==Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    for name in basic+extras:
        tabs.show_tab(name); app.processEvents()
        active=tabs.btns[name]; rect=QRect(active.mapTo(tabs.scroll.viewport(),QPoint()),active.size())
        assert tabs.scroll.viewport().rect().contains(rect)
        assert tabs.height()==34 and panel.width()<=scroll.viewport().width()
        font=QFont(active.font());font.setBold(True)
        assert QFontMetrics(font).horizontalAdvance(name)+20<=active.width()
    tabs.show_tab('Mesh'); app.processEvents(); reached=set()
    for _ in range(50):
        for name,button in tabs.btns.items():
            rect=QRect(button.mapTo(tabs.scroll.viewport(),QPoint()),button.size())
            if tabs.scroll.viewport().rect().intersects(rect): reached.add(name)
        if not tabs.right.isEnabled(): break
        tabs.right.click(); app.processEvents()
    assert reached==set(basic+extras)
    assert not tabs.right.isEnabled() and tabs.left.isEnabled()
    for _ in range(50):
        if not tabs.left.isEnabled(): break
        tabs.left.click(); app.processEvents()
    assert tabs.scroll.horizontalScrollBar().value()==0
    for arrow in (tabs.left,tabs.right): assert tabs.rect().contains(arrow.geometry())
    window.close()


def test_disabled_and_removed_extras_do_not_leave_buttons():
    app=QApplication.instance() or QApplication([])
    tabs=TabBar(['Mesh','Smooth']);tabs.resize(302,34);tabs.show()
    tabs.set_more([('Remove slivers',True),('Fill holes',False)]); app.processEvents()
    assert not tabs.btns['Fill holes'].isEnabled()
    tabs.set_tab_enabled('Smooth',False); assert not tabs.btns['Smooth'].isEnabled()
    selected=[]; tabs.picked.connect(selected.append)
    tabs.btns['Remove slivers'].click(); assert selected==['Remove slivers']
    tabs.set_more([('Fill holes',True)]); assert 'Remove slivers' not in tabs.btns
    tabs.drop_extras(); assert set(tabs.btns)=={'Mesh','Smooth'}
    assert tabs.height()==34; tabs.close()
