"""The quality view: levels from frame counts, a color for every vertex from its nearest sample, and the loader taking the colors."""
import numpy as np
from scanner import frame_quality as Q


def test_levels_follow_frame_counts():
    lv = Q.levels(np.array([0, 5, 6, 19, 20, 400]))
    assert list(lv) == [0, 0, 1, 1, 2, 2]


def test_levels_scale_with_frames_sampled_per_side():
    assert list(Q.levels(np.array([2, 3, 9, 10]), per_side=75)) == [0, 1, 1, 2]


def test_paint_uses_nearest_sample():
    sample = np.array([[0, 0, 0], [10, 0, 0]], float); frames = np.array([1, 100])
    c = Q.paint(np.array([[1, 0, 0], [9, 0, 0]], float), sample, frames)
    assert (c[0] == Q.PALETTE[0]).all() and (c[1] == Q.PALETTE[2]).all()


def test_loader_colors_a_mesh_in_quality_mode(tmp_path, monkeypatch):
    from app2.builder import view_data as VD
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32); F = np.array([[0, 1, 2]], np.uint32)
    monkeypatch.setattr(VD, "load_file", lambda p: dict(kind="mesh", V=V, N=V, F=F, vrgb=None))
    ld = VD.ViewLoader.__new__(VD.ViewLoader); ld.cache = __import__("collections").OrderedDict(); ld.loads = []; ld.cache_n = 4
    ld.elem_cache = ld.cache.__class__()
    rgb = np.tile(Q.PALETTE[0], (3, 1)); p = tmp_path / "m.ply"; p.write_text("x")
    out = ld._load(dict(items=[dict(key="main", path=str(p), rgb=rgb)], mode="quality", check_mem=False))
    assert (out["items"][0]["face_rgb"] == Q.PALETTE[0]).all()


def test_color_menu_quality_measures_once_then_asks_for_the_view(tmp_path):
    from app2.qt import build_step as B
    rgb = tmp_path / "q.npy"; np.save(rgb, np.tile(Q.PALETTE[0], (3, 1)))

    class State:
        def __init__(self): self.text = ""; self.visible = False
        def setText(self, text): self.text = text
        def show(self): self.visible = True
        def setVisible(self, visible): self.visible = visible

    class Host:
        def __init__(self): self.calls = []
        def call(self, name, *args, **kwargs):
            self.calls.append((name, args)); return len(self.calls)

    class Probe:
        _show_quality = B.BuildStepPage._show_quality
        call = B.BuildStepPage.call
        on_result = B.BuildStepPage.on_result
        def __init__(self):
            self.gl = True; self.stage = "fused"; self.colour_mode = {"fused": "quality"}
            self.notes = []; self.views = 0; self._shut = False
            self.host = Host(); self.tickets = {}; self._one_shot_jobs = {}
            self.job_state = State(); self.job_progress = State()
        def _refresh_progress(self): pass
        def _display_path(self): return "/p/mesh.ply"
        def notice(self, t, level="warn", secs=8): self.notes.append(t)
        def request_view(self, *a, **k): self.views += 1
        def _refresh_colours(self): pass
        def _refresh_tools(self): pass

    p = Probe(); p._show_quality()
    assert p.host.calls == [("h_quality", ("/p/mesh.ply",))]
    assert p.views == 0 and p.notes == []
    assert p.job_state.visible and p.job_progress.visible
    assert p.job_state.text == "Measuring scan quality. Finding spots that need more frames."
    assert len(p._one_shot_jobs) == 1 and len(p.tickets) == 1
    p.on_result({"ticket": 1, "name": "h_quality", "ok": True,
                 "value": {"rgb": str(rgb), "share": [0.1, 0.3, 0.6]}})
    assert p.views == 1 and len(p.notes) == 1
    assert "10%" in p.notes[0] and "30%" in p.notes[0] and "60%" in p.notes[0]
    assert not p.job_progress.visible and not p._one_shot_jobs and not p.tickets
    assert p.job_state.text.startswith("Done: Measuring scan quality.")
    p._show_quality()
    assert len(p.host.calls) == 1 and p.views == 2  # a second pick reuses the colors


def test_a_failed_measure_goes_back_to_shaded(tmp_path):
    from app2.qt import build_step as B

    class Probe:
        _show_quality = B.BuildStepPage._show_quality
        def __init__(self): self.gl = True; self.stage = "meshed"; self.colour_mode = {"meshed": "quality"}; self.notes = []
        def _display_path(self): return "/p/mesh.ply"
        def call(self, name, *a, then=None, fail=None, **k): fail("Line up the sides first")
        def notice(self, t, level="warn", secs=8): self.notes.append(t)
        def request_view(self, *a, **k): pass
        def _refresh_colours(self): pass
    p = Probe(); p._show_quality()
    assert p.colour_mode["meshed"] == "shaded" and "Line up the sides first" in p.notes[-1]
