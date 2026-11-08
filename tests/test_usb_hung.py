from app2.scan import usb_mode


def test_wifi_alive_but_no_usb_says_turn_scanner_off_and_on(monkeypatch):
    monkeypatch.setattr(usb_mode, "usb_mode", lambda root="": None)
    monkeypatch.setattr(usb_mode, "wifi_alive", False)
    assert usb_mode.pill_text() == "Scanner not connected"
    monkeypatch.setattr(usb_mode, "wifi_alive", True)
    assert usb_mode.pill_text() == "Scanner not on USB"
    assert "off and on" in usb_mode.help_text()


def test_wifi_alive_does_not_hide_a_real_usb_mode(monkeypatch):
    monkeypatch.setattr(usb_mode, "usb_mode", lambda root="": "transfer")
    monkeypatch.setattr(usb_mode, "wifi_alive", True)
    assert usb_mode.pill_text() == "Scanner in File Transfer mode"


def test_a_full_build_clears_the_waiting_deletes():
    import types
    from app2.qt.build_step import BuildStepPage
    from app2.builder.stagehost import StagesRun
    seen = []
    state = types.SimpleNamespace(recording=False, building=False)
    session = types.SimpleNamespace(build=lambda b: seen.append(b))
    page = types.SimpleNamespace(_shut=False, session=session, state=state, run_mine=None, pending_edits=2,
                                 _cancel_measurements=lambda: None, pending_changed=lambda: seen.append("changed"),
                                 _run_watchdog=types.SimpleNamespace(start=lambda ms: None), _refresh_all=lambda: None)
    assert BuildStepPage._launch_run(page, StagesRun()) and page.pending_edits == 0 and "changed" in seen
    page.pending_edits = 1; page.run_mine = None
    BuildStepPage._launch_run(page, StagesRun(to="lineup.adjust")); assert page.pending_edits == 1


def test_a_delete_starts_the_rebuild_timer_and_it_waits_for_a_quiet_moment():
    import types
    from app2.qt.build_step import BuildStepPage
    started = []; flushed = []
    timer = types.SimpleNamespace(start=lambda: started.append(1))
    sel = types.SimpleNamespace(any=lambda: True)
    page = types.SimpleNamespace(_shut=False, pending_edits=1, gl=True, vp=types.SimpleNamespace(view=types.SimpleNamespace(selection=sel)),
                                 running=lambda: False, busy_call=None, _auto_rebuild=timer, flush_pending_edits=lambda: flushed.append(1))
    BuildStepPage._auto_rebuild_now(page)
    assert started and not flushed                        # red faces are selected: wait longer
    sel.any = lambda: False; started.clear()
    BuildStepPage._auto_rebuild_now(page)
    assert flushed and not started
