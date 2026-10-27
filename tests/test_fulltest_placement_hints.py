"""Synthetic pose checks for the Line-up How scanned filter."""

import numpy as np

from scanner.placement_hints import hint_ok


def test_same_flipped_and_sideways_match_distinct_rotations():
    same = np.eye(4)
    flipped = np.diag([1, -1, -1, 1])
    sideways = np.eye(4)
    sideways[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    up = [0, 0, 1]
    for pose, expected in ((same, "same"), (flipped, "flipped"), (sideways, "sideways")):
        for how in ("same", "flipped", "sideways"):
            assert hint_ok(how, pose, up, up) is (how == expected)


def test_missing_plane_normals_do_not_reject_a_fit():
    assert hint_ok("flipped", np.eye(4), None, [0, 0, 1])
    assert hint_ok("flipped", np.eye(4), [0, 0, 1], None)
