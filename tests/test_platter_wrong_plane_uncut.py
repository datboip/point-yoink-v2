"""A base plane that runs through the part is not applied: the side goes on uncut with a warning (a cut that removes most of a side is a loss
for good; a plate left in is removed later)."""
import json, types
from scanner.stages.methods.side import PlatterAuto


def test_a_plane_through_the_part_leaves_the_side_uncut(tmp_path):
    calls = []
    class Ctx:
        step = types.SimpleNamespace(side="s1", n=11); params = {"margin": 1.0, "max_tilt": 8.0}; out_dir = str(tmp_path); fuse_cap = 4
        inputs = {"raw": types.SimpleNamespace(files={"mesh": "/x/raw.ply"})}; project = types.SimpleNamespace(folder=lambda sid: "/x/side")
        warns = []
        def call(self, args, **k):
            calls.append(args)
            return ("STAGE wrong_plane " + json.dumps({"removed_share": 0.639}) + "\n"
                    "platter: 120 support points, 0.2 deg off the spin axis; kept 689758 triangles in 3 pieces, part height 53.7 mm\n")
        def legacy(self, *a, **k): pass
        def warn(self, t): self.warns.append(t)
    ctx = Ctx(); r = PlatterAuto().execute(ctx)
    assert r.get("alias_of") == "raw" and not r.get("outputs")
    assert ctx.warns and "64%" in ctx.warns[0] and "left uncut" in ctx.warns[0] and ctx.warns[0].startswith("Side 11")


def test_a_sane_cut_goes_through(tmp_path):
    class Ctx:
        step = types.SimpleNamespace(side="s1", n=3); params = {"margin": 1.0, "max_tilt": 8.0}; out_dir = str(tmp_path); fuse_cap = 4
        inputs = {"raw": types.SimpleNamespace(files={"mesh": "/x/raw.ply"})}; project = types.SimpleNamespace(folder=lambda sid: "/x/side")
        warns = []
        def call(self, args, **k): return "platter: 120 support points, 0.2 deg off the spin axis; kept 900000 triangles in 2 pieces, part height 80.0 mm\n"
        def legacy(self, *a, **k): pass
        def warn(self, t): self.warns.append(t)
    r = PlatterAuto().execute(Ctx())
    assert r["outputs"]["mesh"] == "cut.ply" and r["stats"]["faces_kept"] == 900000
