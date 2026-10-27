import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
from stages.side_fusion_cache import table_paths, load_table, save_table


class SideFusionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.stages = self.root / "stages"
        for sid in ("a", "b"):
            for folder in ("cache", "param"): (self.root / sid / folder).mkdir(parents=True)
            for name in ("frame_000_0000.dph", "frame_000_0000.inf", "pointyoink_register_pose.pose"):
                (self.root / sid / "cache" / name).write_bytes(b"original")
            (self.root / sid / "param/Pl.bin").write_bytes(b"calibration")
        self.sj = {"sides": [{"id": sid, "folder": sid, "T": np.eye(4).tolist()} for sid in ("a", "b")]}
        self.params = dict(point_distance=0.4, step=2, frames_per_side=300, edge_px=6.0, z0=260.0, workers=2, min_sides=1)

    def keys(self, sj=None, params=None, deletes=None):
        return table_paths(sj or self.sj, str(self.root), str(self.stages), params or self.params, deletes or {})

    def test_pose_cut_depth_calibration_and_settings_invalidate_only_the_changed_side(self):
        before = self.keys()
        moved = np.eye(4); moved[0, 3] = 0.5
        for field, value in (("T", moved.tolist()),
                             ("plane", {"n": [0, 0, 1], "d": 4, "margin": 1})):
            sj = copy.deepcopy(self.sj); sj["sides"][0][field] = value
            after = self.keys(sj); self.assertNotEqual(before["a"], after["a"]); self.assertEqual(before["b"], after["b"])
        for rel in ("cache/frame_000_0000.dph", "cache/pointyoink_register_pose.pose", "param/Pl.bin"):
            p = self.root / "a" / rel; old = p.read_bytes(); p.write_bytes(old + b"changed")
            after = self.keys(); self.assertNotEqual(before["a"], after["a"]); self.assertEqual(before["b"], after["b"])
            p.write_bytes(old)
        for name in ("step", "frames_per_side", "edge_px", "z0", "point_distance"):
            params = dict(self.params); params[name] += 1
            self.assertNotEqual(before["a"], self.keys(params=params)["a"])

    def test_side_order_worker_count_and_global_thresholds_do_not_invalidate(self):
        before = self.keys(); sj = copy.deepcopy(self.sj); sj["sides"].reverse()
        p = dict(self.params, workers=6, min_sides=2)
        self.assertEqual(before, self.keys(sj, p))

    def test_region_content_and_missing_frame_metadata_invalidate(self):
        (self.stages / "regions").mkdir(parents=True)
        mask = self.stages / "regions/mask.npz"; mask.write_bytes(b"mask")
        deletes = {"a": [{"shape": "mask", "region": "regions/mask.npz"}]}
        before = self.keys(deletes=deletes); mask.write_bytes(b"new mask")
        self.assertNotEqual(before["a"], self.keys(deletes=deletes)["a"])
        before = self.keys(); (self.root / "a/cache/frame_000_0000.inf").unlink()
        self.assertNotEqual(before["a"], self.keys()["a"])

    def test_roundtrip_and_corrupt_table_miss(self):
        path, key = self.keys()["a"]; K = np.array([1, 3], np.int64); C = np.arange(18, dtype=np.float64).reshape(2, 9)
        save_table(path, key, 7, K, C); n, gotK, gotC = load_table(path, key)
        self.assertEqual(n, 7); np.testing.assert_array_equal(gotK, K); np.testing.assert_array_equal(gotC, C)
        self.assertIsNone(load_table(path, "wrong")); Path(path).write_bytes(b"interrupted")
        self.assertIsNone(load_table(path, key))
        save_table(path, key, 7, K, C)
        self.assertIsNotNone(load_table(path, key)); self.assertEqual(len(list(Path(path).parent.glob(".table-*"))), 0)
