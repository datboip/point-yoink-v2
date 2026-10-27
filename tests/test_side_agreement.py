"""Which side does not line up with the others."""
import numpy as np
from scanner import side_agreement as A


def cloud(seed, shift=(0, 0, 0), n=3000):
    r = np.random.default_rng(seed); P = r.uniform(-50, 50, (n, 3)); P[:, 2] = 0.0          # a flat 100 mm patch
    return P + np.asarray(shift, float)


def test_the_odd_side_is_found_and_the_others_are_not():
    pts = {k: cloud(k) for k in range(1, 6)}
    pts[6] = cloud(6, shift=(0, 0, 40))                                   # lies 40 mm off the rest
    res = A.agreement_of_points(pts)
    assert res[6]["agree"] < 0.1 and res[6]["alone"] > 0.9
    assert all(res[k]["agree"] > 0.9 for k in range(1, 6))
    assert A.flag_bad(res, base_n=1) == [6]


def test_base_is_never_flagged_and_few_sides_are_never_judged():
    pts = {1: cloud(1, shift=(0, 0, 40)), 2: cloud(2), 3: cloud(3), 4: cloud(4), 5: cloud(5)}
    assert 1 not in A.flag_bad(A.agreement_of_points(pts), base_n=1)
    assert A.flag_bad(A.agreement_of_points({1: cloud(1), 2: cloud(2, shift=(0, 0, 40)), 3: cloud(3)}), base_n=1) == []


def test_at_most_a_quarter_of_the_sides_are_flagged():
    pts = {k: cloud(k) for k in range(1, 5)}
    for k in range(5, 9): pts[k] = cloud(k, shift=(0, 0, 40 + 10 * k))   # four odd sides in eight: only the worst two
    res = A.agreement_of_points(pts)
    assert len(A.flag_bad(res, base_n=1)) <= 2


def test_a_noisy_side_that_sits_on_the_others_is_not_flagged():
    from scanner.side_agreement import flag_bad
    good = {n: {"alone": 0.0, "agree": 0.95, "points": 1000} for n in range(1, 8)}
    res = dict(good); res.update({8: {"alone": 0.0, "agree": 0.68, "points": 1000},        # noisy, but nowhere alone: in
                                  9: {"alone": 0.193, "agree": 0.78, "points": 1000},      # a fifth of it lies where no side is: out
                                  10: {"alone": 0.03, "agree": 0.70, "points": 1000}})     # some alone and low agree: out
    assert flag_bad(res, base_n=1) == [10, 9]
