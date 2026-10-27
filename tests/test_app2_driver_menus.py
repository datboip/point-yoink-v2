"""Menu actions are discovered by name and modal requests remain answerable."""
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow
from app2.diagnostics.driver import Driver


@pytest.fixture
def menus(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    menu = window.menuBar().addMenu("&File")
    action = menu.addAction("Preferences...\tCtrl+P")
    window.show()
    app.processEvents()
    driver = Driver(window, str(tmp_path / "menus.sock"))
    yield app, window, menu, action, driver
    driver.server.close()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_menu_and_action_are_found_by_names(menus):
    app, window, menu, action, driver = menus
    calls = []
    action.triggered.connect(lambda: calls.append("opened"))
    menu.insertAction(action, menu.addAction("Unrelated"))
    rows = driver.cmd_menus({})["menus"]
    row = next(row for row in rows if row["menu"] == "File")
    assert any(a["text"] == "Preferences..." for a in row["actions"])
    result = driver.cmd_menu({"menu": "File", "action": "Preferences..."})
    assert result == {"queued": "Preferences..."} and calls == []
    app.processEvents()
    assert calls == ["opened"]


def test_menu_refresh_can_disable_an_action(menus):
    app, window, menu, action, driver = menus
    menu.aboutToShow.connect(lambda: action.setEnabled(False))
    with pytest.raises(ValueError, match="disabled"):
        driver.cmd_menu({"menu": "File", "action": "Preferences..."})
    assert not driver.cmd_menus({})["menus"][0]["actions"][0]["enabled"]


def test_hidden_and_ambiguous_actions_are_refused(menus):
    app, window, menu, action, driver = menus
    action.setVisible(False)
    with pytest.raises(ValueError, match="one action"):
        driver.cmd_menu({"menu": "File", "action": "Preferences..."})
    action.setVisible(True)
    menu.addAction("Preferences...")
    with pytest.raises(ValueError, match="one action"):
        driver.cmd_menu({"menu": "File", "action": "Preferences..."})


def test_modal_dialog_is_queued_after_response(menus):
    app, window, menu, action, driver = menus
    dialog = QDialog(window)
    observed = []
    def show_dialog():
        observed.append("opened")
        QTimer.singleShot(0, dialog.accept)
        dialog.exec()
    action.triggered.connect(show_dialog)
    result = driver.handle({"cmd": "menu", "menu": "File", "action": "Preferences..."})
    assert result["ok"] and observed == []
    app.processEvents()
    assert observed == ["opened"]


def test_main_menu_cannot_bypass_an_open_modal_dialog(menus):
    app, window, menu, action, driver = menus
    dialog = QDialog(window)
    dialog.setModal(True)
    dialog.show()
    app.processEvents()
    try:
        with pytest.raises(ValueError, match="active dialog"):
            driver.cmd_menu({"menu": "File", "action": "Preferences..."})
    finally:
        dialog.accept()


def test_queued_action_rechecks_enabled_state(menus):
    app, window, menu, action, driver = menus
    calls = []
    action.triggered.connect(lambda: calls.append(True))
    driver.cmd_menu({"menu": "File", "action": "Preferences..."})
    action.setEnabled(False)
    app.processEvents()
    assert calls == []
