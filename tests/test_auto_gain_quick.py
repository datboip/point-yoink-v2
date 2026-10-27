"""Quick Auto gain: moves on when a stretch adds nothing, stays while it adds, wraps, and finishes when no gain adds anything."""
from app2.scan.assist import AutoGain


def run(gains_added, quick=True):
    """Feed one 45 degree stretch per entry; each entry is the covered count after that stretch."""
    ag = AutoGain(quick=quick); key = object(); counts = iter(gains_added); first = [True]; g = 4; out = []
    deg = 0.0
    def fn(): return next(counts)
    ag.tick(key, deg, g, fn)
    for _ in range(len(gains_added) - 1):
        deg += ag.window; new, _ = ag.tick(key, deg, g, fn)
        if new is not None: g = new
        out.append(g)
    return out, ag


def test_stays_while_adding_then_moves_on():
    out, ag = run([1000, 1500, 2000, 2010, 2500, 2505])      # +50%, +33%, flat (move on), +24% (stay), flat (move on)
    assert out[:2] == [4, 4] and out[2] != 4 and out[3] == out[2] and out[4] != out[3]


def test_done_when_nothing_adds_at_any_gain():
    out, ag = run([1000] + [1001] * 12)
    assert ag.turn["done"] and "done" in ag.msg


def test_classic_takes_a_gain_per_full_turn():
    ag = AutoGain(quick=False); key = object(); c = iter([100, 200, 300]); g = 4
    ag.tick(key, 0.0, g, lambda: next(c)); assert ag.tick(key, 90.0, g, lambda: next(c)) == (None, None)
    new, _ = ag.tick(key, 360.0, g, lambda: next(c)); assert new is not None and new != g


def test_dial_turn_number_counts_full_turns():
    ag = AutoGain(); key = object(); c = iter([100] * 40)
    ag.tick(key, 0.0, 4, lambda: next(c))
    for d in range(45, 730, 45): ag.tick(key, float(d), 4, lambda: next(c))
    assert ag.turn["n"] == 2


def test_brightness_steps_one_gain_from_the_parts_ir_and_holds_two_seconds():
    ag = AutoGain(); ag.bounds = (1, 9)
    assert ag.brightness(0.0, 0.05, 4, 10.0) == 3 and 'blown out' in ag.msg     # clipped: down one
    assert ag.brightness(0.0, 0.05, 3, 11.0) is None                            # held for two seconds
    assert ag.brightness(0.2, 0.0, 3, 12.5) == 4 and 'dark' in ag.msg           # dark, no clipping: up one
    assert ag.brightness(0.2, 0.01, 4, 15.0) is None                            # dark but some clipping: stay
    assert ag.brightness(0.01, 0.0, 9, 18.0) is None and ag.brightness(0.5, 0.0, 9, 21.0) is None   # fine; at the top bound
    assert ag.brightness(0.0, 0.5, 1, 24.0) is None                             # at the bottom bound


def test_surface_rule_changes_the_brightness_thresholds():
    ag = AutoGain(); ag.bounds = (1, 3); ag.clip_max = 0.01; ag.dark_max = 0.2          # Sprayed: strict on glare, patient with dark
    assert ag.brightness(0.0, 0.015, 2, 10.0) == 1                                        # 1.5% blown out is already too much
    assert ag.brightness(0.15, 0.0, 1, 13.0) is None                                      # 15% dark: not yet
    assert ag.brightness(0.25, 0.0, 1, 16.0) == 2 and ag.brightness(0.25, 0.0, 3, 19.0) is None   # up, but never past the bound


def test_a_step_up_that_did_not_help_is_not_repeated():
    ag = AutoGain(); ag.bounds = (1, 9)
    assert ag.brightness(0.20, 0.0, 2, 10.0) == 3                          # dark: up
    assert ag.brightness(0.46, 0.0, 3, 13.0) is None and 'did not help' in ag.msg   # still as dark: hold, no hunting
    assert ag.brightness(0.10, 0.0, 3, 16.0) is None                       # better and under the limit: nothing to do
    assert ag.brightness(0.25, 0.0, 3, 19.0) == 4                          # the dark came back on its own: try again
    assert ag.brightness(0.0, 0.05, 4, 22.0) == 3                          # glare always steps down
