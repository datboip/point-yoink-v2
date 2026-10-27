# SPDX-License-Identifier: AGPL-3.0-or-later
"""A measured bed outline limits automatic base removal."""
import numpy as np


OUTLINE_FIELDS = ('centre', 'radius_mm', 'outline', 'x_axis', 'outside_below_mm')


def removed(points, plane):
    """Mask points removed by a plane, preserving overhangs outside its bed.

    A manual plane without a measured outline keeps its full-plane meaning.
    Zero outside depth means that nothing beside a known outline is cut.
    """
    points = np.asarray(points)
    n = np.asarray(plane['n'], float)
    margin = float(plane.get('margin', 1.0))
    height = points @ n - float(plane['d'])
    above = bool(plane.get('keep_above', True))
    bad = height < margin if above else height > -margin
    if not above or plane.get('centre') is None or plane.get('radius_mm') is None:
        return bad
    center = np.asarray(plane['centre'], float)
    radius = float(plane['radius_mm'])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        return bad
    outline = plane.get('outline')
    if outline and plane.get('x_axis') is not None:
        from scanner.bed import Outline
        shape = Outline.from_json(outline, radius).placed(np.asarray(plane['x_axis'], float))
        distance = shape.sdf(points, n, center).reshape(height.shape)
    else:
        delta = points - center
        distance = np.sqrt(np.maximum((delta * delta).sum(-1) - (delta @ n) ** 2, 0)) - radius
    over = distance <= margin
    threshold = float(plane.get('outside_below_mm', 0))
    return bad & (over | ((height < -threshold) if threshold > 0 else False))
