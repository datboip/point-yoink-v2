# SPDX-License-Identifier: AGPL-3.0-or-later
"""The step-by-step guide for a first scan: one plain instruction at a time, worked out from what the page shows.
Each step names the button to press (target), or None while the user only has to wait or move."""

TOTAL = 6


def step(s):
    """(number, title, text, target) for state s, or None when there is nothing to guide (no scanner, a replay)."""
    if s is None: return None
    n_sides = int(getattr(s, 'n_sides', 0) or 0); frames = bool(getattr(s, 'picked_has_frames', False))
    if getattr(s, 'building', False):
        return 6, 'Building your model', 'This takes a few minutes. You can watch it on the right.', None
    if getattr(s, 'model_exists', False) and not getattr(s, 'recording', False):
        return 6, 'Done', 'Your model is built. Open Edit to clean it up, or Export to save it for printing.', None
    if getattr(s, 'learning', False):
        return 1, 'Learning the bed', 'Let the empty bed turn once. It stops by itself when it is done.', None
    if getattr(s, 'bed_available', False) and getattr(s, 'bed', None) is None and n_sides == 0:
        return 1, 'Learn the bed', 'Take everything off the bed. Press Learn bed and let the bed turn once.', 'learn'
    if getattr(s, 'recording', False):
        return (3 if n_sides <= 1 else 5, 'Scanning side %d' % max(1, n_sides),
                'Turn the bed slowly, one full turn. Watch the model fill in. Press Pause when it looks complete.', 'go')
    if n_sides <= 1 and not frames:
        return (2, 'Place the part', 'Put the part in the middle of the bed. Hold the scanner so the distance bar '
                'says Best, then press Start.', 'go')
    if n_sides <= 1:
        return 4, 'Flip the part', 'Turn the part over so the scanner can see the bottom. Then press + New side.', 'new'
    if not frames:
        return 5, 'Scan side %d' % n_sides, 'Press Start, then turn the bed one full turn.', 'go'
    return (5, 'Build or scan more', 'Scan another side if parts are still missing. Otherwise press Build combined '
            'model to make the finished model.', 'build')


def every_step():
    """One example of each step, side numbers at two digits, so the card can keep room for the longest one."""
    from types import SimpleNamespace as S
    base = dict(n_sides=0, picked_has_frames=False, building=False, model_exists=False, learning=False,
                bed_available=True, bed=None, recording=False)
    cases = [dict(building=True), dict(model_exists=True), dict(learning=True), {}, dict(recording=True, n_sides=10),
             dict(bed='set', n_sides=1), dict(bed='set', n_sides=1, picked_has_frames=True),
             dict(bed='set', n_sides=10), dict(bed='set', n_sides=10, picked_has_frames=True)]
    return [step(S(**dict(base, **case))) for case in cases]
