"""A settings tab's Apply is quiet (not the solid blue) until a setting changes; tool tabs and Detect stay solid."""
from PySide6.QtWidgets import QApplication
from app2.qt import build_words as W
from app2.qt.stage_panels import TabBody

def test_apply_is_quiet_until_a_change():
    app = QApplication.instance() or QApplication([])
    tabs = [(st, t) for st, p in W.PANELS.items() for t in p.get("tabs", []) if t.get("button")]
    bodies = [TabBody(st, t) for st, t in tabs]
    settings = [b for b in bodies if b.settings_only()]
    assert settings and any(not b.settings_only() for b in bodies)
    b = settings[0]
    for c in b.controls: c.set_engine(c.value())
    app.processEvents()
    assert b.main.property("quiet") is True
    c = next(c for c in b.controls if b.shown(c))
    c.engine_value = None; c._refresh_dot()                       # differs from what the engine has: something to apply
    assert b.main.property("quiet") is False
    for o in bodies:
        if not o.settings_only() and o.main is not None:
            o.set_mode("idle"); assert not o.main.property("quiet")
