"""The live tracker fuses only pixels seen reasonably head-on (65 degrees): edge-on depth is tracked against but not added to the model."""
import numpy as np
import track


def _plane(tilt_deg, W=80, H=60, fx=400.0):
    cx, cy = W / 2, H / 2; grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)); u, v = grid
    # a plane through (0, 0, 300 mm) tilted by tilt about the vertical axis, seen by a pinhole camera: z = z0 / (1 - tan(t) * (u - cx) / fx)
    t = np.tan(np.radians(tilt_deg)); z_mm = 300.0 / np.maximum(1.0 - t * (u - cx) / fx, 0.2)
    d = (z_mm * 10).astype(np.uint16); ca = []
    out = track.clean_frame(d, (fx, fx, cx, cy), grid, grazing=80.0, cosang_out=ca)
    return out, ca[0]


def test_a_face_seen_at_72_degrees_is_tracked_but_not_fused():
    out, cos = _plane(72.0); kept = out > 0
    assert kept.mean() > 0.5                                                        # the 80 degree tracking cut keeps it
    ang = np.degrees(np.arccos(np.clip(cos[kept], 0, 1)))
    assert 65 < np.median(ang) < 80
    strict = kept & (cos >= np.cos(np.radians(track.Tracker.FUSE_GRAZING)))
    assert strict.sum() < 0.1 * kept.sum()                                           # the 65 degree fuse cut drops it


def test_a_face_seen_head_on_is_fused_whole():
    out, cos = _plane(10.0); kept = out > 0
    strict = kept & (cos >= np.cos(np.radians(track.Tracker.FUSE_GRAZING)))
    assert strict.sum() > 0.95 * kept.sum()
