"""A preset is one fixed outcome: it starts from the side results, never on top of another preset's tools or settings,
and applying it twice (or A, then B, then A) gives the same recipe each time."""
from scanner.stages import params as PR


def recipe():
    return {"schema": 1, "preset": "auto", "params": {}, "ops": {}, "sides": {}, "next_op": 1}


def strip(r):
    return {"params": r.get("params"), "ops": {k: [dict(o, id=None) for o in v] for k, v in (r.get("ops") or {}).items()}}


def test_a_preset_does_not_stack_on_another():
    r = PR.apply_preset(recipe(), "revo-device")
    assert r["ops"]["fused.points"][0]["tool"] == "isolation" and r["ops"]["fused.points"][0]["params"]["rate"] == 15.0
    r2 = PR.apply_preset(r, "today")                                  # the points tools go, tsdf is set fresh
    assert "fused.points" not in r2["ops"] and r2["params"].get("fused", {}).get("method", "tsdf") == "tsdf"   # tsdf is the default, pruned from the recipe
    assert strip(r2) == strip(PR.apply_preset(recipe(), "today"))


def test_the_same_preset_is_the_same_recipe_every_time():
    a = PR.apply_preset(recipe(), "plus"); b = PR.apply_preset(a, "revo"); c = PR.apply_preset(b, "plus")
    assert strip(a) == strip(c) and strip(a) == strip(PR.apply_preset(a, "plus"))


def test_user_edits_survive_a_preset():
    r = recipe(); r["ops"] = {"fused": [{"id": "op-0001", "tool": "cut_plane", "params": {"nx": 0, "ny": 0, "nz": 1, "d": 0, "margin": 1}}],
                              "meshed": [{"id": "op-0002", "tool": "delete", "params": {}}]}
    r2 = PR.apply_preset(r, "revo-device")
    assert r2["ops"]["fused"][0]["tool"] == "cut_plane" and r2["ops"]["meshed"][0]["tool"] == "delete"
