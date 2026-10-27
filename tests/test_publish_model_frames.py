"""A published model comes with where each side sits in it, so Touch up can track a new side against the built model."""
import json, os, sys, tempfile, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scanner"))
import numpy as np
import model_frames
from stages.engine import Engine


def test_publish_writes_frames_json_that_touch_up_reads():
    root = tempfile.mkdtemp(); sides = os.path.join(root, "sides.json")
    T = np.eye(4); T[:3, 3] = [10, 20, 30]
    json.dump({"sides": [{"id": "a", "n": 1, "folder": "data/09282026222119", "T": T.tolist()}, {"id": "b", "n": 2, "folder": "data/09292026005744/", "T": np.eye(4).tolist()}]}, open(sides, "w"))
    e = Engine.__new__(Engine); e.p = types.SimpleNamespace(root=root, guard_write=lambda p: None)
    e._model_frames({"lineup.adjust": types.SimpleNamespace(files={"sides": sides})})
    got = model_frames.read(root)
    assert sorted(got) == ["09282026222119", "09292026005744"] and np.allclose(got["09282026222119"][:3, 3], [10, 20, 30])


def test_missing_line_up_output_is_not_an_error():
    e = Engine.__new__(Engine); e.p = types.SimpleNamespace(root=tempfile.mkdtemp(), guard_write=lambda p: None)
    e._model_frames({"lineup.adjust": types.SimpleNamespace(files={"sides": "/no/such/file.json"})})
    assert model_frames.read(e.p.root) == {}
