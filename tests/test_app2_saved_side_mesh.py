"""A saved side picked on the Scan page shows the staged build's own result for it (stages/nodes/side.keep@<folder>,
else side.fuse@<folder>), not just its thumbnail, now that the old build/side<k>.ply is gone."""
import os, time
from types import SimpleNamespace


def _node(root, step, folder, key, name, age=0):
    d = os.path.join(root, "stages", "nodes", step + "@" + folder, key); os.makedirs(d)
    open(os.path.join(d, "meta.json"), "w").write("{}"); p = os.path.join(d, name); open(p, "wb").write(b"ply")
    t = time.time() - age; os.utime(p, (t, t)); return p


def test_side_mesh_falls_back_to_the_newest_staged_result(tmp_path):
    from app2.scan.project import Project
    root = str(tmp_path); folder = "10222026203503"
    side = SimpleNamespace(name=folder, cache=os.path.join(root, "data", folder, "cache"))
    os.makedirs(side.cache); track = os.path.join(side.cache, "pointyoink_track.json"); open(track, "w").write("{}")
    t = time.time() - 1000; os.utime(track, (t, t))
    pr = Project.__new__(Project); pr.path = root
    assert pr.side_mesh(side) is None                                     # nothing built yet
    raw = _node(root, "side.fuse", folder, "aaaa", "raw.ply", age=500)
    assert pr.side_mesh(side) == raw                                      # fused side when there is no keep
    old = _node(root, "side.keep", folder, "bbbb", "side.ply", age=400)
    new = _node(root, "side.keep", folder, "cccc", "side.ply", age=100)
    assert pr.side_mesh(side) == new                                      # the newest keep wins
    os.utime(track, None)                                                 # the side was scanned again since: stale
    assert pr.side_mesh(side) is None
