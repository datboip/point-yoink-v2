from types import SimpleNamespace
import pytest

from scanner.stages.engine import Engine


def test_side_estimate_uses_frame_count_and_measured_rate():
    engine = Engine.__new__(Engine)
    engine._side_frames = {"short": 58, "long": 2700}
    history = {"secs": {"side.fuse": [30.0]},
               "size_secs": {"side.fuse": [[300, 30.0]]}}
    short = SimpleNamespace(id="side.fuse@short", kind="side.fuse", side="short", inputs={}, params={})
    long = SimpleNamespace(id="side.fuse@long", kind="side.fuse", side="long", inputs={}, params={})
    assert engine._weight(short, history) == pytest.approx(5.8)
    assert engine._weight(long, history) == pytest.approx(270.0)


def test_pair_estimate_scales_with_number_of_pairs():
    engine = Engine.__new__(Engine)
    engine._side_frames = {str(i): 100 for i in range(12)}
    step = SimpleNamespace(id="lineup.pairs", kind="lineup.pairs", side=None,
                           inputs={"mesh@%d" % i: "side.keep@%d" % i for i in range(12)}, params={})
    assert engine._size(step) == 66
    assert engine._weight(step, {"size_secs": {"lineup.pairs": [[3, 3.0]]}}) == pytest.approx(66.0)
    assert engine._weight(step, {"secs": {"lineup.pairs": [3]}}) == 3        # a time with no size is used as it was


def test_cache_hits_are_not_timings(tmp_path, monkeypatch):
    from scanner.stages import store
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    store.note_global_secs("side.fuse", 0.0, 300); store.note_global_secs("side.fuse", 0.01, 300)
    assert store.global_secs() == {}
    store.note_global_secs("side.fuse", 30.0, 300)
    assert store.global_size_secs() == {"side.fuse": [[300.0, 30.0]]}
    engine = Engine.__new__(Engine); engine._side_frames = {"one": 600}
    step = SimpleNamespace(id="side.fuse@one", kind="side.fuse", side="one", inputs={}, params={})
    assert engine._weight(step, {"size_secs": {"side.fuse": [[300, 0.0]]}}) == pytest.approx(60.0)   # old 0 s rows skipped


def test_previous_stage_time_scales_after_frames_are_added():
    engine = Engine.__new__(Engine)
    engine._side_frames = {"one": 600}
    step = SimpleNamespace(id="side.fuse@one", kind="side.fuse", side="one", inputs={}, params={})
    history = {"nodes": {"side.fuse@one": {"secs": 20.0, "size": 300}}}
    assert engine._weight(step, history) == pytest.approx(40.0)
