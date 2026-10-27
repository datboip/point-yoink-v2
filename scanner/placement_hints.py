# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure orientation filter for Line-up's How scanned hints."""

import numpy as np


def hint_ok(how, candidate, side_up=None, base_up=None):
    """Whether a candidate pose matches the side's direction against the base."""
    if how is None or side_up is None or base_up is None:
        return True
    up_base = np.asarray(base_up, float)
    up_side = np.asarray(candidate, float)[:3, :3] @ np.asarray(side_up, float)
    cosine = float(up_side @ up_base / (np.linalg.norm(up_side) * np.linalg.norm(up_base) + 1e-12))
    return cosine > .55 if how == "same" else cosine < -.55 if how == "flipped" else abs(cosine) < .6
