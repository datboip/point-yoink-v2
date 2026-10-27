"""Manual base cuts survive fusion-method changes, presets, and saved recipe history."""
import copy
import json

import numpy as np
import pytest

from test_stages_core import make_project
from stages import params as PR
from stages.engine import Engine
from stages.project import Project
from stages.recipe import normalise
from stages.tools import plane_cut


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config" / "config.json"))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    make_project(str(project_dir), n=2)
    return Engine(Project.open(str(project_dir)))


def cut(d=10., **extra):
    return {"tool": "cut_plane", "params": {"nx": 0., "ny": 0., "nz": 1., "d": d, "margin": 0., **extra}}


def cuts_in_chain(engine, token):
    return [(lk, op) for lk, _, op in engine.chain_ops(token) if op.get("tool") == "cut_plane"]


def assert_cut_geometry(engine, token, expected, heights=(0., 5., 15., 25.)):
    data = {"P": np.column_stack((np.zeros(len(heights)), np.zeros(len(heights)), heights))}
    for _, op in cuts_in_chain(engine, token):
        if op.get("on", True):
            data, _, _ = plane_cut.apply("points", data, op["params"])
    assert data["P"][:, 2].tolist() == expected


def test_fused_cut_stays_active_when_switching_fusion_methods(engine):
    oid = engine.add_op("fused", cut())
    for method in ("points", "plus", "tsdf"):
        engine.set({"fused.method": method})
        active = cuts_in_chain(engine, "fused")
        assert [(lk, op["id"]) for lk, op in active] == [("fused", oid)]
        assert_cut_geometry(engine, "fused", [15., 25.])
        g = engine.graph()
        assert g.by_id[g.final("fused")].op["id"] == oid
        assert g.by_id["meshed"].inputs["fused"] == g.final("fused")
        assert g.by_id["report"].inputs["mesh"] == g.final("meshed")


@pytest.mark.parametrize("preset", tuple(PR.PRESETS))
def test_each_yolo_preset_retains_fused_and_meshed_cuts(engine, preset):
    fused_id = engine.add_op("fused", cut())
    meshed_id = engine.add_op("meshed", cut(20.))
    engine.preset(preset)
    assert [op["id"] for _, op in cuts_in_chain(engine, "fused")] == [fused_id]
    assert [op["id"] for _, op in cuts_in_chain(engine, "meshed")] == [meshed_id]
    assert_cut_geometry(engine, "fused", [15., 25.])
    assert_cut_geometry(engine, "meshed", [25.])
    if preset == "print":
        tools = [op["tool"] for _, _, op in engine.chain_ops("meshed")]
        assert tools.index("finish") < tools.index("cut_plane")


def test_removing_one_cut_retains_other_after_reopen_undo_redo(engine):
    first = engine.add_op("fused", cut(3.))
    second = engine.add_op("fused", cut(10.))
    engine.drop_op("fused", first)
    assert [op["id"] for _, op in cuts_in_chain(engine, "fused")] == [second]
    reopened = Engine(Project.open(engine.p.root))
    assert [op["id"] for _, op in cuts_in_chain(reopened, "fused")] == [second]
    reopened.undo()
    assert [op["id"] for _, op in cuts_in_chain(reopened, "fused")] == [first, second]
    reopened.redo()
    assert [op["id"] for _, op in cuts_in_chain(reopened, "fused")] == [second]
    assert_cut_geometry(reopened, "fused", [15., 25.])


def test_legacy_active_cut_migrates_without_changing_outline_or_metadata():
    old = dict(cut(poly=plane_cut.outline([[0, 0, 10], [3, 0, 10], [0, 3, 10]], [0, 0, 1]), pad=.5),
               id="op-0007", on=False, anchor={"order": ["side-a"], "T": {"side-a": np.eye(4).tolist()}},
               recorded={"input_digest": "kept-for-history"})
    recipe = {"ops": {"fused.tsdf": [{"id": "op-0006", "tool": "smooth", "on": True, "params": {}}, old]}, "next_op": 8}
    before = copy.deepcopy(recipe)
    updated = normalise(recipe)
    assert updated["ops"]["fused"] == [old]
    assert [o["tool"] for o in updated["ops"]["fused.tsdf"]] == ["smooth"]
    assert normalise(updated) == updated
    assert recipe == before


def test_legacy_parked_cut_is_not_activated_until_its_method_is_selected(engine):
    old = dict(cut(20.), id="op-0010", on=True)
    recipe = engine.recipe()
    recipe["ops"] = {"fused.points": [old]}
    recipe["next_op"] = 11
    engine.hist.record(recipe, "Existing parked cut")
    assert cuts_in_chain(engine, "fused") == []
    engine.preset("today")
    assert cuts_in_chain(engine, "fused") == []
    assert engine.recipe()["ops"]["fused.points"] == [old]
    engine.set({"fused.method": "points"})
    assert cuts_in_chain(engine, "fused") == [("fused", old)]


def test_presets_preserve_manual_side_plane(engine, tmp_path):
    sid = engine.p.sides()[0].id
    params = {"n": [0, 0, 1], "d": 10., "margin": .5, "keep_above": True}
    path = tmp_path / "manual-plane.json"
    path.write_text(json.dumps(params))
    engine.platter(sid, str(path))
    for preset in PR.PRESETS:
        engine.preset(preset)
        graph = engine.graph()
        assert graph.by_id["side.platter@" + sid].method.name == "manual"
        assert graph.by_id["side.platter@" + sid].params["plane"] == params


@pytest.mark.parametrize("kind", ("launch", "killed", "crash", "unknown", "exit"))
def test_failed_manual_side_cut_stops_instead_of_restoring_raw_base(tmp_path, kind):
    from types import SimpleNamespace
    from stages.methods.side import PlatterManual
    from stages.runner import StepError

    def failed_call(*args, **kwargs):
        raise StepError("cut failed", "worker failed", rc=1, kind=kind)

    ctx = SimpleNamespace(project=SimpleNamespace(folder=lambda _side: str(tmp_path)),
                          step=SimpleNamespace(side="side-a", n=1), out_dir=str(tmp_path),
                          inputs={"raw": SimpleNamespace(files={"mesh": str(tmp_path / "raw.ply")})},
                          params={"plane": {"n": [0, 0, 1], "d": 10., "margin": 1.}, "margin": 1.},
                          call=failed_call, legacy=lambda *args, **kwargs: None)
    with pytest.raises(StepError, match="manual base cut failed") as caught:
        PlatterManual().execute(ctx)
    assert caught.value.kind == kind
    assert caught.value.tail == "worker failed"


def regenerated_mesh(tmp_path):
    """Tiny downstream result with faces regenerated inside and outside bounded cut regions."""
    from stages import plyio
    vertices = []
    for x, z in ((0., 0.), (0., 15.), (0., 25.), (50., 0.)):
        vertices.extend([[x, 0., z], [x + 1., 0., z], [x, 1., z]])
    vertices = np.asarray(vertices)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    path = tmp_path / "regenerated.ply"
    plyio.write_mesh(str(path), vertices, faces)
    return path


def replay_result(engine, source, folder):
    from types import SimpleNamespace
    from stages import plyio
    from stages.methods.retained_cuts import job
    graph = engine.graph()
    step = graph.by_id[graph.final("meshed")]
    folder.mkdir(exist_ok=True)
    calls = []

    def run_job(fn, payload, mem_gb):
        calls.append((fn, mem_gb))
        return job(dict(payload, out_dir=str(folder)))

    ctx = SimpleNamespace(inputs={"parent": SimpleNamespace(files={"mesh": str(source)})},
                          params=step.params, job=run_job)
    result = step.method.execute(ctx)
    assert calls == [("methods.retained_cuts:job", 3)]
    path = folder / result["outputs"]["mesh"]
    vertices, faces, _ = plyio.read_mesh(str(path))
    return graph, path, vertices[faces].mean(axis=1), result


def test_final_mesh_replay_preserves_outline_reaches_publish_and_stl(engine, tmp_path):
    import struct
    from types import SimpleNamespace
    from app2.builder.export_model import write_model

    poly = plane_cut.outline([[-2, -2, 10], [3, -2, 10], [3, 3, 10], [-2, 3, 10]], [0, 0, 1])
    fused_id = engine.add_op("fused", cut(poly=poly, pad=0.))
    meshed_id = engine.add_op("meshed", cut(20., poly=poly, pad=0.))
    # A later smoothing/finish step could recreate these faces. Replay runs after all Meshed tools.
    finish_id = engine.add_op("meshed", {"tool": "smooth", "params": {}})
    source = regenerated_mesh(tmp_path)
    graph, output, centers, result = replay_result(engine, source, tmp_path / "result")
    final = graph.by_id[graph.final("meshed")]
    assert final.id == "meshed.retained_cuts"
    assert final.inputs["parent"] == "meshed#" + finish_id
    assert [c["source"] for c in final.params["cuts"]] == ["fused#" + fused_id, "meshed#" + meshed_id]
    assert centers[:, 2].tolist() == [25., 0.]
    assert centers[-1, 0] > 50.  # outside the spots' outline is still retained
    assert result["stats"]["faces_in"] == 4
    assert result["stats"]["faces"] == 2
    assert [c["removed"] for c in result["stats"]["retained_cuts"]] == [1, 1]
    assert graph.by_id["report"].inputs["mesh"] == final.id

    outs = {final.id: SimpleNamespace(files={"mesh": str(output)}, bdigest="clipped", step=final.id, meta={"key": "cut-key"}),
            "report": SimpleNamespace(files={}, bdigest="report")}
    published = engine._publish(graph, outs, SimpleNamespace(emit=lambda *a, **k: None))
    assert open(published, "rb").read() == output.read_bytes()
    assert json.loads((tmp_path / "project/stages/published.json").read_text())["node"] == final.id
    stl = tmp_path / "export.stl"
    exported = write_model(published, str(stl), "stl")
    assert exported["faces"] == 2
    raw = stl.read_bytes()
    assert struct.unpack_from("<I", raw, 80)[0] == 2
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    exported_centers = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].mean(axis=1)
    for constraint in final.params["cuts"]:
        assert not plane_cut.below(exported_centers, constraint["params"]).any()


def test_replay_updates_when_cuts_are_toggled_removed_and_undone(engine, tmp_path):
    first = engine.add_op("fused", cut())
    second = engine.add_op("meshed", cut(20.))
    source = regenerated_mesh(tmp_path)
    _, _, centers, _ = replay_result(engine, source, tmp_path / "both")
    assert centers[:, 2].tolist() == [25.]
    engine.toggle_op("meshed", second)
    graph, _, centers, _ = replay_result(engine, source, tmp_path / "one")
    assert centers[:, 2].tolist() == [15., 25.]
    assert [c["source"] for c in graph.by_id[graph.final("meshed")].params["cuts"]] == ["fused#" + first]
    engine.drop_op("fused", first)
    assert "meshed.retained_cuts" not in engine.graph().by_id
    engine.undo()
    _, _, centers, _ = replay_result(engine, source, tmp_path / "undo")
    assert centers[:, 2].tolist() == [15., 25.]
    reopened = Engine(Project.open(engine.p.root))
    _, _, centers, _ = replay_result(reopened, source, tmp_path / "reopened")
    assert centers[:, 2].tolist() == [15., 25.]


def test_replay_key_depends_on_cut_shape_and_upstream_mesh(engine):
    from types import SimpleNamespace
    oid = engine.add_op("fused", cut())
    graph = engine.graph()
    step = graph.by_id[graph.final("meshed")]
    parent = step.inputs["parent"]
    key = engine.step_key(step, {parent: SimpleNamespace(digest="mesh-a")})
    assert engine.step_key(step, {parent: SimpleNamespace(digest="mesh-b")}) != key
    engine.set_op("fused", oid, {"d": 12.})
    changed = engine.graph().by_id["meshed.retained_cuts"]
    assert engine.step_key(changed, {changed.inputs["parent"]: SimpleNamespace(digest="mesh-a")}) != key


def test_replay_is_not_an_extra_editable_cut(engine):
    oid = engine.add_op("fused", cut())
    assert [op["id"] for _, op in cuts_in_chain(engine, "fused")] == [oid]
    assert cuts_in_chain(engine, "meshed") == []
    graph = engine.graph()
    assert graph.final("meshed") == "meshed.retained_cuts"
    assert graph.lists["meshed"] == []
    assert engine.targets(graph, "meshed") == [graph.final("meshed")]


@pytest.mark.parametrize("poly", ("not-json", "[]", [[0, 0], [1, 0]], [[0, 0], [1, 1], [2, 2]],
                                   [[0, 0], [1, 1], [float("nan"), 0]],
                                   [[0, 0], [1, 1], [0, float("inf")]], [[0, 0, 0]] * 3))
def test_invalid_bounded_outline_never_falls_back_to_unbounded_cut(poly):
    with pytest.raises(ValueError, match="Cut base outline"):
        plane_cut.below(np.array([[1000., 1000., 0.]]), cut(poly=poly)["params"])


def test_explicit_unbounded_plane_keeps_existing_behavior():
    points = np.array([[1000., 1000., 0.], [0., 0., 25.]])
    assert plane_cut.below(points, cut()["params"]).tolist() == [True, False]
    assert plane_cut.below(points, cut(poly="")["params"]).tolist() == [True, False]


def test_final_replay_is_idempotent(engine, tmp_path):
    engine.add_op("fused", cut())
    source = regenerated_mesh(tmp_path)
    _, first, centers, _ = replay_result(engine, source, tmp_path / "first")
    _, second, centers_again, stats = replay_result(engine, first, tmp_path / "second")
    np.testing.assert_array_equal(centers_again, centers)
    assert first.read_bytes() == second.read_bytes()
    assert stats["stats"]["removed"] == 0


def test_cli_show_includes_final_retained_cut_sources(engine, capsys):
    from stages.cli import print_show
    oid = engine.add_op("fused", cut())
    print_show(engine, "meshed", as_json=True)
    shown = json.loads(capsys.readouterr().out)
    assert shown[-1]["id"] == engine.graph().final("meshed")
    assert shown[-1]["params"]["cuts"][0]["source"] == "fused#" + oid
