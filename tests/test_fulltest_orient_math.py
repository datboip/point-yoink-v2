"""Geometry checks for Build's bottom and front orientation controls."""

import numpy as np

from app2.builder.orient_math import bottom_rotation, front_rotation


def test_bottom_uses_side_containing_most_model_points():
    # The selected face is x=0, with the body on the positive-x side.
    spots = np.array([[0, -10, -10], [0, 10, -10], [0, 10, 10], [0, -10, 10]])
    model = np.array([[20, 0, 0], [22, 3, 4], [18, -2, -4], [-1, 0, 0]])
    rotation = bottom_rotation(spots, model)
    assert rotation is not None
    assert np.allclose(rotation @ [1, 0, 0], [0, 0, 1], atol=1e-8)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1)


def test_bottom_rejects_collinear_spots():
    spots = [[0, 0, 0], [4, 0, 0], [8, 0, 0]]
    assert bottom_rotation(spots, [[0, 0, 1]]) is None


def test_front_turns_about_world_up_and_preserves_existing_bottom():
    bottom = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    oriented = front_rotation([1, 0, 0], bottom)
    assert np.allclose(oriented @ [1, 0, 0], [0, 0, 1], atol=1e-8)
    assert np.allclose(oriented @ [0, 1, 0], [-1, 0, 0], atol=1e-8)
    assert np.allclose(oriented.T @ oriented, np.eye(3), atol=1e-8)
