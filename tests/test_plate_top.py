import numpy as np

from scanner.plate_top import choose_plate_top


def _grid(width, height, z):
    x, y = np.meshgrid(np.linspace(-width / 2, width / 2, 55),
                       np.linspace(-height / 2, height / 2, 55))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _normals(points):
    return np.tile([0, 0, 1.0], (len(points), 1))


def test_broad_plate_top_is_chosen_above_turntable():
    base = _grid(300, 300, 0)
    plate = _grid(280, 280, 12)
    part = np.vstack([_grid(80, 80, z) for z in (22, 25, 28, 31, 34)])
    points = np.vstack((base, plate, part))
    top, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 1)
    assert top == 10.0
    assert report["cut_height_mm"] == 11.0
    assert report["height_mm"] == 12.5
    assert report["removed_share"] <= 0.40


def test_small_flat_part_bottom_is_rejected():
    points = np.vstack((_grid(300, 300, 0), _grid(60, 60, 12), _grid(60, 60, 30)))
    top, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 1)
    assert top is None
    assert "no distinct" in report["reason"]


def test_cut_below_plate_does_not_count_plate_itself_as_removed():
    plate = _grid(280, 280, 12)
    points = np.vstack((_grid(300, 300, 0), plate, plate, _grid(80, 80, 30)))
    top, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 1)
    assert top is not None
    assert report["removed_share"] <= 0.40


def test_actual_cut_over_forty_percent_is_rejected():
    base = _grid(300, 300, 0)
    points = np.vstack((base, base, base, _grid(280, 280, 12), _grid(80, 80, 30)))
    top, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 1)
    assert top is None
    assert "40%" in report["reason"]


def test_riser_plate_surface_is_found_and_actual_cut_stays_below_top():
    base = _grid(350, 400, 0)
    plate = np.vstack((_grid(340, 380, 63.2), _grid(340, 380, 64.2)))
    part = np.vstack([_grid(150, 140, z) for z in (70, 75, 80, 85, 90)])
    points = np.vstack((base, plate, part))
    plane, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 2.8)
    assert report["height_mm"] == 63.5
    assert report["cut_height_mm"] == 63.0
    assert plane + 2.8 == 63.0
    assert report["removed_share"] <= 0.40


def test_multiple_similar_broad_layers_are_ambiguous():
    points = np.vstack((_grid(350, 350, 0), _grid(300, 300, 35), _grid(300, 300, 70)))
    plane, report = choose_plate_top(points, _normals(points), [0, 0, 1], 0, 1)
    assert plane is None
    assert "multiple competing" in report["reason"]
