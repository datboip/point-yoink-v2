"""Projects list: each card has a small trash button at its left edge, which asks first (Delete 1.2 GB? Delete / Keep), deletes the folder off a thread
and refreshes the list; the open project's Delete is off."""
import os, time
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton
from app2.qt import theme


def _home(tmp_path, names):
    from app2.qt.workspace_home import WorkspaceHome
    app = QApplication.instance() or QApplication([]); theme.install(app)
    for n in names: (tmp_path / n / 'data' / 'side1').mkdir(parents=True); (tmp_path / n / 'data' / 'side1' / 'x.bin').write_bytes(b'0' * 10)
    home = WorkspaceHome(str(tmp_path)); home.resize(1024, 740); home.refresh_projects(); home.show(); app.processEvents()
    home.projects_dropdown.click(); app.processEvents(); assert home.projects_popover.isVisible()
    return app, home


def _cards(home):
    return [c for c in home.projects_popover.findChildren(type(home.projects_popover), 'WorkspaceProjectCard') if not c.isHidden()]   # a refresh hides the old cards before Qt deletes them


def _buttons(card, name):
    from PySide6.QtWidgets import QAbstractButton
    return [b for b in card.findChildren(QAbstractButton) if b.objectName() == name]


def test_delete_asks_then_removes_the_folder_and_refreshes(tmp_path):
    app, home = _home(tmp_path, ['ProjectLive111', 'ProjectLive222'])
    gone = []; home.project_deleted.connect(gone.append)
    try:
        cards = _cards(home); assert len(cards) == 2
        card = cards[0]; ask = _buttons(card, 'WorkspaceProjectTrash')[0]; yes = _buttons(card, 'WorkspaceProjectDeleteYes')[0]
        assert ask.isEnabled() and not yes.isVisible()
        ask.click(); app.processEvents()
        assert not ask.isEnabled() and yes.isVisible()
        keep = [b for b in card.findChildren(QPushButton) if b.text() == 'Keep'][0]
        keep.click(); app.processEvents()
        assert ask.isEnabled() and not yes.isVisible()                   # Keep: nothing happened
        ask.click(); app.processEvents(); yes.click()
        for _ in range(200):
            app.processEvents(); time.sleep(0.01)
            if gone: break
        assert len(gone) == 1 and not os.path.exists(gone[0])
        app.processEvents()
        assert len(_cards(home)) == 1 and home.projects_dropdown.text() == 'Your projects (1)  ▾'
    finally: home.close()


def test_open_project_cannot_be_deleted_from_the_list(tmp_path):
    app, home = _home(tmp_path, ['ProjectLive111'])
    try:
        home.set_active_project(str(tmp_path / 'ProjectLive111')); home.refresh_projects(); app.processEvents()
        ask = _buttons(_cards(home)[0], 'WorkspaceProjectTrash')[0]
        assert not ask.isEnabled() and 'open' in ask.toolTip()
    finally: home.close()
