"""Cut base: two big choices (Auto / Manual), a Keep the base switch and the small Where to look chips, all
over the one recipe value the chips used to set; and the status line that says what the build found."""
import os, tempfile, types
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="cut-choice-test-")) / "config.json")
import pytest
from PySide6.QtWidgets import QApplication
from app2.qt import build_step as B, build_words as W
from app2.qt.stage_panels import StagePanel, make_control
from app2.qt.tool_panel import CutChoiceRow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _row(app):
    return make_control(dict(kind="cutchoice", label="Cut base", path="side[x].platter.method", default="auto"))


def test_the_side_panel_cut_base_tab_has_the_new_control_behind_the_old_path(app):
    p = StagePanel("side"); body = p.bodies["Platter"]
    ctl = next((c for c in body.controls if str(c.c.get("path") or "").endswith("platter.method")), None)
    assert isinstance(ctl, CutChoiceRow) and ctl.value() == "auto"
    assert not any(isinstance(c, CutChoiceRow) for c in body.controls if c is not ctl)


def test_the_one_value_drives_the_two_buttons_the_switch_and_the_chips(app):
    r = _row(app); got = []; r.moved.connect(got.append)
    assert r.b_me.isChecked() and not r.b_hand.isChecked() and not r.sw.on and not r.where_box.isHidden()
    r.b_hand.click()
    assert got[-1] == "manual" and r.value() == "manual" and r.b_hand.isChecked() and not r.b_me.isChecked() and r.where_box.isHidden()
    r.b_me.click()
    assert got[-1] == "auto" and r.value() == "auto"
    r.btns["plate"].click()
    assert got[-1] == "plate" and r.value() == "plate" and r.b_me.isChecked() and r.btns["plate"].isChecked()
    r.sw.toggled.emit(True)                                  # Keep the base
    assert got[-1] == "none" and r.value() == "none" and not r.b_me.isEnabled() and not r.b_hand.isEnabled()
    r.sw.toggled.emit(False)                                 # back to the way it looked before (Plate), not to Auto
    assert got[-1] == "plate" and r.value() == "plate" and r.b_me.isEnabled()


def test_the_engine_value_sets_the_picture_and_the_dot_follows_a_change(app):
    r = _row(app)
    r.set_engine("plate")
    assert r.value() == "plate" and r.b_me.isChecked() and r.btns["plate"].isChecked() and not r.changed()
    r.pick("manual", emit=True)
    assert r.changed()
    r.set_engine("none")
    assert r.value() == "none" and r.sw.on and not r.changed()


class _Probe:
    _cut_status = B.BuildStepPage._cut_status
    def __init__(self, preview=False, share=0.0): self._pv = preview; self.platter_share = share
    def _platter_preview(self): return self._pv


def test_the_status_line_says_what_the_build_found():
    s = _Probe()._cut_status
    assert s(None, "auto") == (W.CUT["unbuilt"], None)
    assert s({"stats": {"source": "board dots"}}, "plate") == (W.CUT["found"]["board dots"], None)
    assert s({"stats": {"source": "live plane"}}, "auto") == (W.CUT["found"]["live plane"], None)
    assert s({"stats": {"source": "uncut (plane crossed the part)"}}, "plate") == (W.CUT["crossed"], "warn")
    assert s({"stats": {"source": "failed"}}, "auto") == (W.CUT["failed"], "warn")
    assert s({"stats": {"source": "board dots"}}, "none") == (W.CUT["kept"], None)
    assert s({"stats": None, "last_stats": {"source": "board dots"}}, "plate") == (W.CUT["found"]["board dots"], None)   # waiting to run again: the last build
    assert s({"stats": {"source": "board dots"}}, "manual") == (W.CUT["hand_wait"], None)                 # Manual: the spots, not the last build
    assert s({"stats": {"source": "board dots"}}, "manual", 3) == (W.CUT["hand_ready"], None)
    assert s({"stats": {"source": "manual"}}, "manual") == (W.CUT["found"]["manual"], None)              # cut by the spots already: say so, not "not marked yet"


def test_the_status_line_adds_the_pending_cuts_share_and_warns_above_40_percent():
    t, lvl = _Probe(preview=True, share=0.12)._cut_status({"stats": {"source": "board dots"}}, "plate")
    assert t == W.CUT["found"]["board dots"] + W.CUT["share"].format(pct=12) and lvl is None
    t, lvl = _Probe(preview=True, share=0.58)._cut_status({"stats": {"source": "plate"}}, "plate")
    assert t.endswith(W.CUT["share_big"].format(pct=58)) and lvl == "warn"
    t, _ = _Probe(preview=False, share=0.58)._cut_status({"stats": {"source": "plate"}}, "plate")
    assert "%" not in t                                      # no cut pending: the share is not shown


def test_the_spot_counter_counts_up_to_three():
    assert W.SPOTS["count_need"].format(n=1, s="") == "1 of 3 spots picked"
    assert W.SPOTS["count"].format(n=4, s="s") == "4 spots picked"


def test_auto_and_keep_cut_at_once_manual_waits_for_the_spots(app):
    r = _row(app); r.set_engine("manual"); ran = []
    class P:
        _cut_auto_apply = B.BuildStepPage._cut_auto_apply
        stage = "side"; busy_call = None; _tools_on = True
        panel = {"side": types.SimpleNamespace(current="Platter", bodies={"Platter": None})}
        def _ctl(self, body, suffix): return r
        def running(self): return False
        def on_button(self, tab): ran.append(tab)
    p = P()
    r.pick("manual", emit=True); p._cut_auto_apply("manual"); assert ran == []
    r.pick("auto", emit=True); p._cut_auto_apply("auto"); assert ran == ["Platter"]
    r.set_engine("auto"); p._cut_auto_apply("auto"); assert ran == ["Platter"]        # nothing changed: no run
    r.pick("none", emit=True); P.busy_call = object(); later = []
    import app2.qt.build_step as BS
    real = BS.QTimer.singleShot; BS.QTimer.singleShot = lambda ms, fn: later.append(fn)
    try:
        p._cut_auto_apply("none"); assert ran == ["Platter"] and len(later) == 1          # busy: tried again shortly, not dropped
        P.busy_call = None; later.pop()(); assert ran == ["Platter", "Platter"]
    finally: BS.QTimer.singleShot = real


def test_engine_warnings_read_plainly():
    w = W.plain_warning
    assert w("Side 1: the base plane found runs through the part (it would remove 53% of the side), so this side was left uncut.").startswith("Side 1: Auto could not find the base")
    assert w("plate top rejected (turntable reference is not a broad flat layer); used Auto") == "No plate found under the part, so the turntable was used instead."
    assert w("Side 8: its base plane was ignored for placing. 3 of its 3 good fits run through it, so it is probably not the real base.").startswith("Side 8: its base looked wrong")
    assert w("something new") == "something new"                                  # no plain version: the engine's own words
