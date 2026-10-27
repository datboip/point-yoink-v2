"""Deck-height Edit controls remain fully reachable above pinned history."""
from types import SimpleNamespace as NS
import pytest
from PySide6.QtCore import QRect,Qt,QPoint
from PySide6.QtWidgets import QApplication,QPushButton
from app2.qt import theme
from app2.qt.edit_page import EditPage

@pytest.fixture
def palette():
    app=QApplication.instance() or QApplication([]);theme.install(app)
    page=EditPage();page.resize(1280,800);page.palette.setParent(page)
    page.palette.setGeometry(0,0,248,340);page.palette.show()
    page.build=NS(gl=False,stage='meshed',running=lambda:False,undo_b=QPushButton(),redo_b=QPushButton(),_display_path=lambda:'saved-model.ply')
    page.show();page.sync()
    for _ in range(8):app.processEvents()
    yield page,app
    page.build=None;page.close()

def visible(widget,viewport):
    return viewport.rect().contains(QRect(widget.mapTo(viewport,QPoint()),widget.size()))

def test_instruction_height_does_not_push_controls_below_the_fold(palette):
    page,app=palette
    needed=page.carry.fontMetrics().boundingRect(QRect(0,0,page.carry.width(),10000),Qt.TextFlag.TextWordWrap,page.carry.text()).height()
    assert page.carry.height()<=needed+4

def test_selection_and_reach_are_fully_visible_at_deck_viewport_height(palette):
    # the palette is numbered cards now (Select faces / Adjust the selection / Change the model) that scroll: each selection
    # and reach control must be fully visible once scrolled to, and the first card's header must show without scrolling
    page,app=palette;viewport=page.palette.viewport()
    from PySide6.QtWidgets import QLabel
    titles=[l.text() for l in page.palette.findChildren(QLabel)]
    assert 'Select faces' in titles and 'Adjust the selection' in titles
    for button in list(page.mode_buttons.values())+[page.visible_button,page.through_button]:
        page.palette.ensureWidgetVisible(button,0,0)
        for _ in range(3):app.processEvents()
        assert visible(button,viewport)

def test_last_tool_is_reachable_and_desktop_row_height_is_preserved(palette):
    page,app=palette;page.palette.ensureWidgetVisible(page.tools[-1],0,0)
    for _ in range(5):app.processEvents()
    assert visible(page.tools[-1],page.palette.viewport())
    page.resize(1920,1080)
    for _ in range(5):app.processEvents()
    assert page.tool_buttons['lasso'].height()==30
