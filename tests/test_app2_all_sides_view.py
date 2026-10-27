"""Side and Raw use placed ghosts, without an all-sides grid."""
import types
import numpy as np
from app2.qt import build_step as B

def test_side_view_puts_the_picked_side_first_and_ghosts_the_placed_others():
    reqs = []; calls = {}
    class Q:
        _request_side_in_place = B.BuildStepPage._request_side_in_place
        stage = "side"; view_id = 4
        snap = types.SimpleNamespace(sides=[{"id": "a", "active": True}, {"id": "b", "active": True}, {"id": "c", "active": False}, {"id": "d", "active": True}],
                                     final=lambda k: k, out=lambda k: {"side.keep@b": "/b.ply", "side.keep@c": "/c.ply", "side.keep@d": "/d.ply"}.get(k))
        loader = types.SimpleNamespace(request=reqs.append)
        def _side(s): return {"id": "a"}
        def _solo_wanted(s): return False
        def call(s, name, then=None, fail=None): calls["name"] = name; then({"T": {"a": np.eye(4), "b": np.eye(4), "c": np.eye(4)}})
    q = Q(); q._request_side_in_place("/a-uncut.ply", "shaded")
    assert calls["name"] == "h_lineup" and len(reqs) == 1
    r = reqs[0]; keys = [it["key"] for it in r["items"]]
    assert keys == ["a", "b"] and r["edit"] == "a" and r["path"] == "/a-uncut.ply" and r["check_mem"]     # c left out, d has no place yet
    assert r["items"][0]["path"] == "/a-uncut.ply" and "colour" not in r["items"][0] and r["items"][1]["colour"] is not None
    assert q._ghosts == ["b"]
    reqs.clear(); q._side = lambda: {"id": "d"}; q._request_side_in_place("/d.ply", "shaded")
    assert [it["key"] for it in reqs[0]["items"]] == ["d"] and q._ghosts == []                              # unplaced: alone


def test_side_view_without_a_lineup_shows_the_side_alone():
    reqs = []
    class Q:
        _request_side_in_place = B.BuildStepPage._request_side_in_place
        stage = "side"; view_id = 1; snap = types.SimpleNamespace(sides=[{"id": "a", "active": True}, {"id": "b", "active": True}], final=lambda k: k, out=lambda k: "/x.ply")
        loader = types.SimpleNamespace(request=reqs.append)
        def _side(s): return {"id": "a"}
        def _solo_wanted(s): return False
        def call(s, name, then=None, fail=None): fail(RuntimeError("no line-up"))
    q = Q(); q._request_side_in_place("/a.ply", "shaded")
    assert [it["key"] for it in reqs[0]["items"]] == ["a"] and "model" not in reqs[0]["items"][0] and q._ghosts == [] and not reqs[0]["check_mem"]


def test_ghost_legend_marks_each_ghosted_side_at_its_placed_middle():
    class V:
        def __init__(s): s.markers = None; s.items = {}
        def set_markers(s, m): s.markers = m
    class Q:
        _ghost_legend = B.BuildStepPage._ghost_legend
        stage = "side"; _ghosts = ["b"]; snap = types.SimpleNamespace(sides=[{"id": "a", "n": 1}, {"id": "b", "n": 2}])
        def __init__(s): s.vp = types.SimpleNamespace(view=V())
    q = Q(); M = np.eye(4); M[:3, 3] = [10, 0, 0]
    q.vp.view.items = {"a": dict(kind="points", P=np.zeros((5, 3))), "b": dict(kind="points", P=np.array([[0, 0, 0], [1, 1, 1], [5, 0, 0.0]]), model=M)}
    q._ghost_legend()
    assert len(q.vp.view.markers) == 1 and q.vp.view.markers[0][0] == [15.0, 0.0, 0.0] and q.vp.view.markers[0][2] == 2    # the ghost's farthest point from the picked side
    q.stage = "fused"; q.vp.view.markers = None; q._ghost_legend(); assert q.vp.view.markers == []      # its dots go with the stage
    q.vp.view.markers = None; q._ghost_legend(); assert q.vp.view.markers is None                       # then other stages' marks are left alone



def test_the_cut_base_tab_shows_the_side_alone_so_its_base_can_be_checked_or_clicked():
    reqs = []
    class Q:
        _request_side_in_place = B.BuildStepPage._request_side_in_place
        stage = "side"; view_id = 1
        snap = types.SimpleNamespace(sides=[{"id": "a", "active": True}, {"id": "b", "active": True}], final=lambda k: k, out=lambda k: "/b.ply")
        loader = types.SimpleNamespace(request=reqs.append)
        def _side(s): return {"id": "a"}
        def _solo_wanted(s): return True
        def call(s, name, then=None, fail=None): then({"T": {"a": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "b": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}})
    q = Q(); q._request_side_in_place("/a.ply", "shaded")
    assert [it["key"] for it in reqs[-1]["items"]] == ["a"] and q._ghosts == [] and q._view_solo


def test_ghost_dots_keep_apart_when_two_sides_saw_the_same_spot():
    import numpy as np
    line = np.stack([np.linspace(100, 200, 400), np.zeros(400), np.zeros(400)], 1)      # both ghosts: the same far strip
    marks = []
    view = types.SimpleNamespace(items={"a": dict(kind="points", P=np.zeros((50, 3)), model=None),
                                        "b": dict(kind="points", P=line, model=None), "c": dict(kind="points", P=line.copy(), model=None)},
                                 set_markers=marks.extend)
    q = types.SimpleNamespace(stage="side", _ghosts=["b", "c"], vp=types.SimpleNamespace(view=view),
                              snap=types.SimpleNamespace(sides=[dict(id="a", n=1), dict(id="b", n=2), dict(id="c", n=3)]))
    B.BuildStepPage._ghost_legend(q)
    (pb, _, _), (pc, _, _) = marks
    assert np.linalg.norm(np.subtract(pb, pc)) > 30                     # they used to land on the same point


def test_ghost_dots_go_when_the_view_leaves_the_side_stages():
    cleared = []
    view = types.SimpleNamespace(items={}, set_markers=cleared.append)
    q = types.SimpleNamespace(stage="meshed", _ghost_marks=True, vp=types.SimpleNamespace(view=view))
    B.BuildStepPage._ghost_legend(q)
    assert cleared == [[]] and q._ghost_marks is False                  # they stayed on the finished model after YOLO
    B.BuildStepPage._ghost_legend(q)
    assert cleared == [[]]                                               # not ours any more: another tool's marks are left alone
