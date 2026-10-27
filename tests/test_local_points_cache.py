import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
from stages.local_points_cache import Writer, batches, collect, local_paths


class LocalPointCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.stages = self.root / "stages"
        self.side = self.root / "data/a"
        (self.side / "cache").mkdir(parents=True); (self.side / "param").mkdir()
        for name in ("frame_000_0000.dph", "frame_000_0000.inf", "pointyoink_register_pose.pose"):
            (self.side / "cache" / name).write_bytes(b"input")
        (self.side / "param/Pl.bin").write_bytes(b"calibration")
        self.sides = {"sides": [dict(id="a", folder="data/a", T=np.eye(4).tolist())]}
        self.params = dict(step=2, frames_per_side=300, edge_px=6., z0=260., point_distance=.4)

    def keys(self, sides=None, params=None):
        return local_paths(sides or self.sides, str(self.root), str(self.stages), params or self.params, {})

    def data(self):
        P = np.arange(30, dtype=np.float64).reshape(10, 3) / 7
        return P, P / 13, np.arange(10, dtype=np.float64) / 9, np.eye(4)

    def test_placement_and_world_grid_change_reuse_local_but_cut_and_depth_invalidate(self):
        before = self.keys(); moved = copy.deepcopy(self.sides); moved["sides"][0]["T"][0][3] = .2
        self.assertEqual(before, self.keys(moved))
        self.assertEqual(before, self.keys(params=dict(self.params, point_distance=.8)))
        moved["sides"][0]["plane"] = dict(n=[0, 0, 1], d=3, margin=1)
        self.assertNotEqual(before, self.keys(moved))
        (self.side / "cache/frame_000_0000.dph").write_bytes(b"new depth")
        self.assertNotEqual(before, self.keys())

    def test_lossless_batches_and_corrupt_manifest_rejection(self):
        folder, key = self.keys()["a"]; writer = Writer(folder, key); expected = self.data()
        writer.append(*expected); writer.finish()
        got = list(batches(folder, key)); self.assertEqual(len(got), 1)
        for a, b in zip(expected, got[0]): np.testing.assert_array_equal(a, b)
        p = Path(folder) / "meta.json"; meta = json.loads(p.read_text()); meta["batches"][0]["points"] += 1
        p.write_text(json.dumps(meta))
        with self.assertRaises(ValueError): list(batches(folder, key))

    def test_uncompressed_fallback_and_interrupted_writer(self):
        folder, key = self.keys()["a"]
        with patch.dict(sys.modules, {"lz4": None, "lz4.frame": None}):
            writer = Writer(folder, key); self.assertEqual(writer.codec, "raw")
            writer.append(*self.data()); writer.finish()
            got = list(batches(folder, key))
        np.testing.assert_array_equal(got[0][0], self.data()[0])
        second = Writer(str(Path(folder).parent / "interrupted"), "interrupted")
        partial = second.temp; second.append(*self.data()); second.abort()
        self.assertFalse(partial.exists()); self.assertFalse(second.path.exists())

    def test_gc_protects_latest_local_and_evicts_old_local_and_world_table(self):
        parent = self.stages / "cache/side-local-points/a"
        for key, age in (("old", 100), ("latest", 200)):
            writer = Writer(str(parent / key), key); writer.append(*self.data()); writer.finish()
            os.utime(parent / key / "meta.json", (age, age))
        world = self.stages / "cache/side-fusion/a/old.npz"; world.parent.mkdir(parents=True); world.write_bytes(b"cached table")
        result = collect(str(self.stages), 1, 100)
        self.assertTrue((parent / "latest/meta.json").exists())
        self.assertFalse((parent / "old").exists()); self.assertFalse(world.exists())
        self.assertGreater(result["bytes"], 1)
