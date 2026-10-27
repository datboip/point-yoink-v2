"""Minimize a build: the card leaves the view and a compact strip in the top bar carries it; click the strip to bring the card back."""
from PySide6.QtWidgets import QApplication
from app2.qt.build_view import ProgressCard
from app2.qt.workspace_home import MiniProgress


def test_minimize_docks_and_restores_and_end_clears():
    app = QApplication.instance() or QApplication([])
    card = ProgressCard(); mini = MiniProgress(); got = []
    card.status.connect(mini.set_status); card.docked_changed.connect(mini.setVisible); mini.restore.connect(card.undock)
    card.show_state("YOLO Yoink", "Line-up: placing sides", "base auto", 0.25, 12, 0, overall="Step 3 of 9, about 4 min left"); app.processEvents()
    assert card.isVisible() and not mini.isVisible() and mini.bar.value() == 250
    card._toggle_size(); app.processEvents()
    assert card.docked and not card.isVisible() and mini.isVisible()
    card.show_state("YOLO Yoink", "Fusion", "", 0.5, 20, 0, overall="Step 5 of 9"); app.processEvents()
    assert not card.isVisible() and mini.bar.value() == 500 and "Step 5" in mini._full          # the strip keeps updating, the card stays away
    mini.restore.emit(); app.processEvents()
    assert card.isVisible() and not mini.isVisible() and not card.docked
    card._toggle_size(); card.hide(); app.processEvents()
    assert not mini.isVisible() and not card.docked                                              # the build ended


def test_strip_grows_to_fit_the_whole_line():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([]); mini = MiniProgress(); mini.show()
    mini.set_status("YOLO Yoink  ·  Step 19 of 29, about 7 min left", 0.6); app.processEvents()
    assert "7 min left" in mini.text.text() and not mini.text.text().endswith("…")
    w1 = mini.width()
    mini.set_status("YOLO Yoink  ·  Step 19 of 29, about 7 min left and then some more words to be long", 0.6); app.processEvents()
    assert mini.width() >= w1 and mini.width() <= 600


def test_every_build_card_can_be_minimized_not_just_yolo():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([]); card = ProgressCard(); mini = MiniProgress()
    card.status.connect(mini.set_status); card.docked_changed.connect(mini.setVisible)
    card.show_state("Working", "Starting", "", 0.1, 4, 0, overall="Step 1 of 6, about 5 min left"); app.processEvents()
    assert card.size_b.isVisibleTo(card)
    card.size_b.click(); app.processEvents()
    assert mini.isVisible() and not card.isVisible() and "Working" in mini._full


def test_pause_button_toggles_signals_and_resets_when_the_build_ends():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([]); card = ProgressCard(); got = []; lines = []
    card.pause.connect(got.append); card.status.connect(lambda t, f: lines.append(t))
    card.show_state("Working", "Starting", "", 0.4, 4, 0, overall="Step 2 of 6"); app.processEvents()
    assert card.pause_b.text() == "Pause"
    card.pause_b.click(); assert got == [True] and card.pause_b.text() == "Resume" and card.pct.text() == "Paused" and lines[-1].startswith("Paused")
    card.show_state("Working", "Fusion", "", 0.5, 9, 0, overall="Step 3 of 6"); assert card.pct.text() == "Paused" and lines[-1].startswith("Paused")   # an update keeps it
    card.pause_b.click(); assert got == [True, False] and card.pause_b.text() == "Pause"
    card.pause_b.click(); card.hide(); assert not card.paused and card.pause_b.text() == "Pause"                                              # the build ended


def test_log_rows_show_warning_text_and_failures():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import ProgressCard
    QApplication.instance() or QApplication([])
    c = ProgressCard()
    n = {"a": {"id": "side.platter@1", "n": 6, "state": "stale"}}
    c.show_state("Building model", "x", "", .1, 1, 0, nodes=n)
    n["a"] = {"id": "side.platter@1", "n": 6, "state": "warn", "secs": 9, "warn": ["plate top rejected; used Auto"]}
    c.show_state("Building model", "Side 6 platter", "", .2, 5, 0, nodes=n)
    n["a"] = {"id": "fused", "state": "failed", "error": "RuntimeError: out of memory"}
    c.show_state("Building model", "Fused", "", .3, 8, 0, nodes=n)
    text = c.log.toPlainText()
    assert "plate top rejected; used Auto" in text and "out of memory" in text and "FAILED" in text


def test_two_cards_share_one_docked_state():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import ProgressCard
    QApplication.instance() or QApplication([])
    a = ProgressCard(); b = ProgressCard(); got = []
    for c in (a, b): c.show_state("Building model", "x", "", .2, 3, 0)
    a.docked_changed.connect(lambda on: (b.set_docked(on), got.append(on)))
    a._toggle_size(); assert a.docked and b.docked and not b.isVisible() and got == [True]
    b.show_state("Building model", "y", "", .3, 4, 0); assert not b.isVisible()          # a docked card stays in the strip while it updates
    a.undock(); assert not a.docked
    b.set_docked(False); assert not b.docked


def test_a_click_on_the_strip_restores():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from app2.qt.workspace_home import MiniProgress
    QApplication.instance() or QApplication([])
    m = MiniProgress(); m.set_status("Working  ·  Step 3 of 27", 0.1); m.show(); got = []
    m.restore.connect(lambda: got.append(1))
    QTest.mouseClick(m.text, Qt.MouseButton.LeftButton)              # on the words
    QTest.mouseClick(m, Qt.MouseButton.LeftButton, pos=m.rect().bottomLeft() + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(20, -3))   # on the bar
    assert len(got) == 2


def test_a_failed_build_keeps_its_card_until_close():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import ProgressCard
    QApplication.instance() or QApplication([])
    c = ProgressCard(); c.show_state("Building model", "Fused", "", .4, 30, 0)
    c.show_failed("fusing stopped at frame 1721 of 1800: CUDA out of memory", 31)
    assert c.holding and "out of memory" in c.log.toPlainText() and c.stop_b.text() == "Close" and c.pct.text() == "Failed"
    c.hide(); assert c.holding                                            # the page's idle tick does not take it away
    c.stop_b.click(); assert not c.holding and not c.isVisible() and c.stop_b.text() == "Stop"
    c.show_state("Building model", "Side 1 fuse", "", .1, 2, 0); c.show_failed("x", 3)
    c.show_state("Building model", "Side 1 fuse", "", .1, 2, 0); assert not c.holding and "x" not in c.log.toPlainText()   # a new build clears it


def test_log_keeps_the_scroll_position_while_it_updates():
    from PySide6.QtWidgets import QApplication
    from app2.qt.build_view import ProgressCard
    QApplication.instance() or QApplication([])
    c = ProgressCard(); c.resize(640, 600); c.show()
    nodes = {}
    for i in range(40):
        nodes["s%d" % i] = {"id": "side.fuse@%d" % i, "n": i, "state": "stale", "secs": 1}; c.show_state("Building model", "x", "", .1, i, 0, nodes=dict(nodes))
        nodes["s%d" % i]["state"] = "done"; c.show_state("Building model", "x", "", .1, i, 0, nodes=dict(nodes))
    QApplication.processEvents()
    bar = c.log.verticalScrollBar(); assert bar.maximum() > 0 and bar.value() == bar.maximum()        # following the newest row
    bar.setValue(10); c.show_state("Building model", "y", "", .2, 41, 0, nodes=dict(nodes))       # the user scrolled up: a new live line
    assert bar.value() == 10
    bar.setValue(bar.maximum()); c.show_state("Building model", "z", "", .3, 42, 0, nodes=dict(nodes))
    assert bar.value() == bar.maximum()                                                          # at the end again: it follows


def test_strip_gives_way_to_the_buttons_in_a_narrow_bar(tmp_path):
    """The minimized build strip never rides over Advanced and Guide: with little room it shrinks and elides (2026-10-22)."""
    from PySide6.QtWidgets import QApplication
    from app2.qt.workspace_home import WorkspaceHome
    app = QApplication.instance() or QApplication([]); home = WorkspaceHome(str(tmp_path)); home.resize(1688, 1044); home.show()           # the app's default size: the meters at the right take 520 px
    home.mini.show(); home.mini.set_status("Building model  ·  8 of 24 steps done, 2 running, about 45 s left", 0.4)
    app.processEvents(); app.processEvents()
    assert home.mini.geometry().right() <= home.level_button.geometry().left()
    assert home.mini.text.text().startswith("8 of 24")                 # squeezed: the progress part, not "Building …"
    home.close()
