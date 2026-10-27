"""Auto white balance makes a warm-tinted gray part neutral, leaves a truly colored part mostly colored, and painting
uses at most PAINT_FRAMES frames per side, evenly spread."""
import numpy as np
from scanner.color_frames import white_gains, spread, PAINT_FRAMES


def test_warm_gray_comes_out_neutral():
    rng = np.random.default_rng(0); lum = rng.uniform(0.1, 0.8, 5000)[:, None]
    warm = np.clip(lum * np.array([1.3, 1.0, 0.75]), 0, 1)        # gray seen under warm light
    fixed = warm * white_gains(warm)
    med = np.median(fixed, axis=0)
    assert np.ptp(med) / med.mean() < 0.02


def test_strong_color_is_only_partly_pulled():
    red = np.tile([0.8, 0.1, 0.1], (3000, 1)) * np.random.default_rng(1).uniform(0.5, 1, (3000, 1))
    g = white_gains(red)
    assert g.min() >= 0.6 and g.max() <= 1.6                     # bounded: red stays clearly red
    assert np.median(red * g, axis=0)[0] > 2 * np.median(red * g, axis=0)[1]


def test_paint_uses_spread_frames_per_side():
    jobs = [("/p/data/a/cache/frame_000_%04d.dph" % i,) for i in range(431)] + [("/p/data/b/cache/frame_000_%04d.dph" % i,) for i in range(50)]
    kept = spread(jobs)
    a = [j for j in kept if "/a/" in j[0]]; b = [j for j in kept if "/b/" in j[0]]
    assert len(a) == PAINT_FRAMES and len(b) == 50 and a[0] == jobs[0] and a[-1] == jobs[430]
