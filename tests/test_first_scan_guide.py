from types import SimpleNamespace
from app2.scan.first_scan import step


def state(**kw):
    base = dict(n_sides=0, picked_has_frames=False, building=False, model_exists=False, learning=False,
                bed_available=True, bed=None, recording=False)
    base.update(kw); return SimpleNamespace(**base)


def test_steps_follow_a_first_scan_one_instruction_at_a_time():
    walk = [state(), state(learning=True), state(bed=1), state(bed=1, n_sides=1, recording=True),
            state(bed=1, n_sides=1, picked_has_frames=True), state(bed=1, n_sides=2),
            state(bed=1, n_sides=2, recording=True), state(bed=1, n_sides=2, picked_has_frames=True),
            state(bed=1, n_sides=2, building=True), state(bed=1, n_sides=2, model_exists=True)]
    got = [(n, target) for n, _, _, target in map(step, walk)]
    assert got == [(1, 'learn'), (1, None), (2, 'go'), (3, 'go'), (4, 'new'), (5, 'go'), (5, 'go'), (5, 'build'), (6, None), (6, None)]
    assert step(None) is None
