# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable side-method imports and saved worker entry points.

Implementation code is separated so bed edits do not invalidate fusion or cleanup.
"""
from .stage_events import _stage_json
from .fusion_events import _FusionBackend
from .side_fuse import SideFuseTsdf
from .side_keep import SideKeep, KEPT_RE
from .side_platter import (_Platter, PlatterAuto, PlatterPlate, PlatterSearch,
                           PlatterManual, PlatterNone, PLATTER_RE)


def observe_no_base(job):
    """Compatibility entry for saved stage-job requests."""
    from .side_platter import observe_no_base as _observe_no_base
    return _observe_no_base(job)
