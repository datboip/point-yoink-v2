"""Cuts must stay tied to their project, side, current result, and processing state."""
from types import SimpleNamespace

import pytest
from app2.builder.stagehost import Snapshot
from app2.qt import build_step as B


@pytest.fixture(autouse=True)
def private_config(monkeypatch, tmp_path):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))


def snapshot(status="waiting", files=None, extra=(), chains=None):
    nodes = [{"id": "meshed", "status": status}] + list(extra)
    return Snapshot(groups=[{"name": "meshed", "nodes": nodes}], finals={}, files=files or {}, chains=chains or {}, recipe={})


class ExportProbe:
    _export_src = B.BuildStepPage._export_src
    _export_needs_rebuild = B.BuildStepPage._export_needs_rebuild
    _export_as = B.BuildStepPage._export_as
    export_stl = B.BuildStepPage.export_stl
    open_export = B.BuildStepPage.open_export

    def __init__(self, root, snap):
        self.host_root = str(root); self.state = None; self.snap = snap
        self.notices = []; self.panel = {"meshed": SimpleNamespace(body=lambda name: None)}
        self.calls = []

    def tell(self, text): self.notices.append(text)
    def close_export(self): pass
    def call(self, *args, **kwargs): self.calls.append(args)
    def ask_save(self, *args): raise AssertionError("An obsolete model must never reach the Save dialog")


@pytest.mark.parametrize("status", ["stale", "waiting", "failed"])
def test_export_blocks_old_published_mesh_after_upstream_cut(tmp_path, status):
    old = tmp_path / "model.ply"; old.write_bytes(b"old model with base")
    cut = tmp_path / "fused-cut.ply"; cut.write_bytes(b"current fused model without base")
    snap = snapshot(status, files={"fused": {"mesh": str(cut)}})
    probe = ExportProbe(tmp_path, snap)
    assert probe._export_needs_rebuild()
    assert probe._export_src() is None
    probe.export_stl(); probe.open_export(); probe._export_as("ply", False)
    assert probe.calls == []
    assert probe.notices == [B.EXPORT_REBUILD] * 3
    assert old.read_bytes() == b"old model with base"


def test_unbuilt_cut_blocks_legacy_fallback(tmp_path):
    (tmp_path / "model.ply").write_bytes(b"old")
    snap = snapshot(chains={"fused": [("fused", 0, {"tool": "cut_plane", "on": True})]})
    assert ExportProbe(tmp_path, snap)._export_src() is None
    snap = snapshot(extra=[{"id": "side.platter@a", "method": "manual", "status": "waiting"}])
    assert ExportProbe(tmp_path, snap)._export_src() is None


def test_stale_upstream_with_waiting_meshed_blocks_legacy_export(tmp_path):
    (tmp_path / "model.ply").write_bytes(b"old")
    snap = snapshot(extra=[{"id": "side.platter@a", "method": "plate", "status": "stale"}])
    assert ExportProbe(tmp_path, snap)._export_src() is None


def test_current_mesh_wins_and_untouched_legacy_model_stays_exportable(tmp_path):
    old = tmp_path / "model.ply"; old.write_bytes(b"old")
    current = tmp_path / "cut.ply"; current.write_bytes(b"cut")
    assert ExportProbe(tmp_path, snapshot("fresh", {"meshed": {"mesh": str(current)}}))._export_src() == str(current)
    assert ExportProbe(tmp_path, snapshot())._export_src() == str(old)
    assert ExportProbe(tmp_path, None)._export_src() == str(old)


def test_missing_current_cut_mesh_never_falls_back_to_uncut_published_file(tmp_path):
    (tmp_path / "model.ply").write_bytes(b"old")
    snap = snapshot("fresh", {"meshed": {"mesh": str(tmp_path / "missing-cut.ply")}})
    assert ExportProbe(tmp_path, snap)._export_src() is None


class BaseProbe:
    _base_state = B.BuildStepPage._base_state

    def __init__(self, nodes, planes):
        self.nodes = nodes
        self.snap = SimpleNamespace(out=lambda node, kind: "plane.json" if node in planes else None)

    def _nodes(self, stage): return self.nodes


def node(sid, status, method="plate", source="plate top"):
    return {"id": "side.platter@" + sid, "method": method, "status": status, "stats": {"source": source}}


def test_configured_but_unbuilt_cut_is_not_reported_applied():
    probe = BaseProbe([node("a", "waiting"), node("b", "stale")], {"side.platter@b"})
    text, state, _, _ = probe._base_state()
    assert text == "Cut applied on 0 of 2" and state == "Needs build"


def test_soft_failed_cut_does_not_count_even_when_engine_warn_is_done():
    probe = BaseProbe([node("a", "warn", source="failed"), node("b", "fresh")], {"side.platter@a", "side.platter@b"})
    text, state, _, _ = probe._base_state()
    assert text == "Cut applied on 1 of 2" and state == "Check 1 side"
    probe = BaseProbe([dict(node("a", "warn", source="uncut (plane crossed the part)"), n=1), dict(node("b", "warn", source="failed"), n=4)], {"side.platter@a", "side.platter@b"})
    _, state, _, tip = probe._base_state()
    assert state == "Check 2 sides" and tip.startswith("Sides 1, 4:")          # which sides, and Manual: something a first-timer can act on
    probe = BaseProbe([node("a", "warn", source="live plane"), node("b", "fresh")], {"side.platter@a", "side.platter@b"})
    _, state, _, _ = probe._base_state()
    assert state == ""                                                            # plate rejected, turntable used: it cut, nothing to check


def test_cut_without_a_plane_is_not_reported_applied():
    assert BaseProbe([node("a", "fresh")], set())._base_state()[:2] == ("Cut applied on 0 of 1", "Needs build")


def test_current_successful_cut_and_explicit_off_are_distinguished():
    probe = BaseProbe([node("a", "fresh"), node("b", "off", method="none")], {"side.platter@a"})
    assert probe._base_state()[:2] == ("Cut applied on 1 of 2", "Some sides")
    probe = BaseProbe([node("a", "fresh")], {"side.platter@a"})
    assert probe._base_state()[:2] == ("Cut applied on 1 of 1", "")


class ManualProbe:
    _spots_on = B.BuildStepPage._spots_on
    _spot_add = B.BuildStepPage._spot_add
    _spot_set = B.BuildStepPage._spot_set
    _cut_remove = B.BuildStepPage._cut_remove
    _spot_height_all = B.BuildStepPage._spot_height_all
    def _cut_local(self, world): return [float(x) for x in world]        # no line-up: the view shows the side as scanned

    def __init__(self, all_sides=False):
        self.gl = True; self.stage = "side"; self.all_sides = all_sides
        self._spots = {}; self._tools_on = True; self.busy_call = None; self.building = False
        self.panel = {"side": SimpleNamespace(current="Platter", bodies={"Platter": object()})}
        self.notices = []; self.view_updates = 0
        self.vp = SimpleNamespace(view=SimpleNamespace(tools_on=True))

    def _side(self): return {"id": "a"}
    def _ctl(self, *args): return SimpleNamespace(value=lambda: "manual")
    def _all_mode(self): return self.all_sides
    def _spots_view(self): self.view_updates += 1
    def _spots_plane(self, **kwargs): raise AssertionError("Blocked cut must not fit a plane")
    def _cut_plane(self, *args): raise AssertionError("Blocked cut must not fit a plane")
    def _cut_on(self): return True
    def running(self): return self.building
    def notice(self, text): self.notices.append(text)


def test_all_sides_manual_world_pick_is_refused_until_side_selected():
    probe = ManualProbe(all_sides=True)
    assert not probe._spots_on()
    probe._spot_add([100, 0, 50]); probe._spot_set()
    assert probe._spots == {} and probe.view_updates == 0
    assert probe.notices == [B.MANUAL_SIDE]
    probe.all_sides = False
    assert probe._spots_on()
    probe._spot_add([1, 2, 3])
    assert probe._spots == {"a": [[1.0, 2.0, 3.0]]}


@pytest.mark.parametrize("busy", ["build", "host", "disabled"])
def test_custom_cut_actions_do_nothing_while_busy(busy):
    probe = ManualProbe()
    if busy == "build": probe.building = True
    elif busy == "host": probe.busy_call = {"name": "set"}
    else: probe._tools_on = False
    probe._spot_set(); probe._spot_height_all()
    probe.stage = "fused"
    probe._cut_remove("fused")
    assert not probe.notices


def test_readonly_or_inactive_fused_view_cannot_apply_old_spots():
    probe = ManualProbe(); probe.stage = "fused"
    probe.vp.view.tools_on = False
    probe._cut_remove("fused")
    probe.vp.view.tools_on = True
    probe._cut_remove("meshed")


def test_attach_clears_all_cut_and_orientation_picks():
    class Probe:
        attach = B.BuildStepPage.attach
        _reset_orientation_prompt = B.BuildStepPage._reset_orientation_prompt
        _cancel_measurements = B.BuildStepPage._cancel_measurements
        _cancel_memory_choice = B.BuildStepPage._cancel_memory_choice
        def _clear_compare_change(self): pass
        def close_onetap(self): pass
        def close_export(self): pass
        def _update_log_button(self): pass
        def _drop_host(self): pass
        def _refresh_all(self): pass
    probe = Probe()
    probe.log = probe.result = SimpleNamespace(clear=lambda: None)
    probe._spots = {"a": [[1, 2, 3]]}; probe._spots_built = {"a"}
    probe._cspots = {"fused": [[1, 2, 3]], "meshed": [[4, 5, 6]]}
    probe._orient_spots = [[1, 2, 3]]
    probe._orientation_prompt_ready = True
    rejected=[];probe._memory_pending=object()
    probe._memory_dialog=SimpleNamespace(reject=lambda:rejected.append(True))
    probe.attach(object())
    assert probe._spots == {} and probe._spots_built == set()
    assert probe._cspots == {"fused": [], "meshed": []} and probe._orient_spots == []
    assert not probe._orientation_prompt_ready and probe._orientation_prompt is None
    assert rejected==[True] and probe._memory_pending is None and probe._memory_dialog is None
