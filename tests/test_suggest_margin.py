import numpy as np

from scanner.suggest_margin import suggest_margin_from_heights


def test_near_plane_layer_returns_a_small_non_applied_number():
    heights = np.concatenate((np.full(1000, 0.0), np.full(500, 8.0),
                              np.linspace(12.0, 80.0, 2500)))
    result = suggest_margin_from_heights(heights)
    assert result["suggest_margin_mm"] == 0.5
    assert result["confidence"] == "layer edge"
    assert np.mean(heights <= result["suggest_margin_mm"]) <= 0.40


def test_distant_competing_flat_layer_is_not_named_the_plate():
    heights = np.concatenate((np.full(1000, 0.0), np.full(800, 63.0),
                              np.linspace(80.0, 150.0, 1000)))
    result = suggest_margin_from_heights(heights)
    assert result["suggest_margin_mm"] == 0.0
    assert result["confidence"] == "none"
    assert "competing" in result["reason"]


def test_sparse_side_does_not_receive_an_unsupported_margin():
    heights = np.linspace(-30.0, 70.0, 10000)
    result = suggest_margin_from_heights(heights)
    assert result["suggest_margin_mm"] == 0.0
    assert result["confidence"] == "none"


def test_guard_caps_suggestion_below_forty_percent_quantile():
    heights = np.concatenate((np.full(3500, 0.0), np.linspace(0.1, 1.0, 500),
                              np.full(6000, 20.0)))
    result = suggest_margin_from_heights(heights)
    assert result["suggest_margin_mm"] <= result["max_guarded_margin_mm"]
    assert np.mean(heights <= result["suggest_margin_mm"]) <= 0.40
