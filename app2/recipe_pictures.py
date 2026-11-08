# SPDX-License-Identifier: AGPL-3.0-or-later
"""A picture of the user's own part for each recipe that has built it, kept in the project folder (never exported).
The YOLO Yoink window and Ultra Yoink show them, so the difference between recipes is seen on the real part."""
import io
import os
import time
from pathlib import Path

DIR = '.pointyoink-recipes'
SIZE = (560, 320)


def key_of(preset):
    """The file a recipe's picture lives in: Auto is Classic, a saved preset keeps its own (saved-<name>), anything else is 'custom'."""
    import re
    preset = 'today' if preset in (None, '', 'auto') else str(preset)
    if preset.startswith('saved:'): return 'saved-' + (re.sub(r'[^A-Za-z0-9]+', '-', preset[6:]).strip('-').lower() or 'preset')
    return preset if re.fullmatch(r'[A-Za-z0-9_-]+', preset) else 'custom'


def path_of(project, key):
    return Path(project) / DIR / (key_of(key) + '.png')


def existing(project):
    """{recipe key: (png path, saved time)} for the pictures this project has."""
    out = {}
    try:
        for f in (Path(project) / DIR).glob('*.png'):
            out[f.stem] = (str(f), f.stat().st_mtime)
    except OSError:
        pass
    return out


def render_png(model_path):
    """PNG bytes of the model drawn like the project list picture, or None (too big, not a mesh, unreadable)."""
    import numpy as np
    import shade
    from app2.builder.view_data import estimate_bytes, load_file
    if not model_path or not os.path.isfile(model_path) or estimate_bytes(model_path) > 2048 * 1024 ** 2: return None   # it runs on a worker thread; a fine-detail build is a few hundred MB to load
    model = load_file(model_path)
    if model['kind'] != 'mesh' or not len(model['V']) or not len(model['F']): return None
    vertices = model['V'].astype(np.float64)
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    vertices *= .85 / max(float(np.abs(vertices).max()), 1e-9)
    faces = model['F'][::max(1, (len(model['F']) + 39999) // 40000)]
    if model.get('vrgb') is not None and int(np.asarray(model['vrgb']).max()) > 0:
        from app2.builder.color_preview import render
        image = render(vertices, faces, model['vrgb'], size=SIZE)
    else:
        image = shade.render(vertices, faces, size=SIZE, grid=False, gizmo=False)
    data = io.BytesIO(); image.save(data, format='PNG')
    return data.getvalue()


def save(project, preset, model_path):
    """Draw the model and keep it as this recipe's picture. True when saved."""
    try:
        raw = render_png(model_path)
        if raw is None: return False
        dest = path_of(project, preset); dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(dest)
        return True
    except Exception:
        return False


def when(saved):
    return time.strftime('%d %b', time.localtime(saved)).lstrip('0')
