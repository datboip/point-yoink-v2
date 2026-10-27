# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small orientation calculations shared by Build controls and tests."""

import numpy as np


def bottom_rotation(spots, model_points):
    """Return a proper rotation that points the selected bottom face upward.

    The larger share of model samples determines which side of the selected
    plane contains the part. Return None when the spots do not span a plane.
    """
    pts = np.asarray(spots, float)
    c = pts.mean(axis=0)
    _u, singular, vt = np.linalg.svd(pts - c)
    if singular[1] < 1e-3 * max(singular[0], 1e-9) or singular[1] < 2.0:
        return None
    normal = vt[2] / np.linalg.norm(vt[2])
    sample = np.asarray(model_points, float)
    if ((sample - c) @ normal > 0).sum() < ((sample - c) @ normal < 0).sum():
        normal = -normal
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, up)
    sine = float(np.linalg.norm(axis))
    cosine = float(normal @ up)
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= sine
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def front_rotation(view_direction, current_rotation):
    """Turn about world up so the camera's viewing direction points toward +Y."""
    forward = np.asarray(view_direction, float)
    theta = np.pi / 2 - np.arctan2(forward[1], forward[0])
    c, s = np.cos(theta), np.sin(theta)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return turn @ np.asarray(current_rotation, float)
