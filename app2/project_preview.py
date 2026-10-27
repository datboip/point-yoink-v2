# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project cards and small PNG previews, separate from scan/model data."""
import json
from datetime import datetime
import os
import re
import tempfile
from pathlib import Path

PREVIEW_NAME = '.pointyoink-preview.png'


META_NAME = '.pointyoink-home.json'


def project_name(project):
    try:
        meta = json.loads((Path(project) / META_NAME).read_text())
        return str(meta.get('name') or '') if isinstance(meta, dict) else ''
    except (OSError, ValueError):
        return ''


def needs_name(name):
    return not name.strip() or bool(re.fullmatch(r'Project(?:Live)?[0-9]+', name, re.IGNORECASE))


def rename_project(project, name):
    """Store a friendly name, keeping the folder, scans, and other metadata intact."""
    name = name.strip()
    if not name: raise ValueError('Enter a project name.')
    if len(name) > 100: raise ValueError('Use 100 characters or fewer.')
    dest = Path(project) / META_NAME
    meta = json.loads(dest.read_text()) if dest.exists() else {}
    if not isinstance(meta, dict): raise ValueError('Project details could not be read.')
    meta['name'] = name
    _atomic_write(dest, (json.dumps(meta, indent=2) + '\n').encode())
    return name


def _atomic_write(dest, raw):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix='.pointyoink-', delete=False) as f:
            temporary = f.name; f.write(raw)
        os.replace(temporary, dest); temporary = None
    finally:
        if temporary:
            try: os.unlink(temporary)
            except OSError: pass


def scan_time(folder_name):
    """The time a scanner project was scanned, from its folder name (ProjectMMDDYYYYHHMMSS or ProjectLive...), or None."""
    m = re.fullmatch(r'Project(?:Live)?(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})', folder_name)
    if not m: return None
    mo, d, y, h, mi, se = map(int, m.groups())
    try: return datetime(y, mo, d, h, mi, se).timestamp()
    except ValueError: return None


BACKUP_WORDS = {
    "none": ("only copy", "Scanned with this app, so these frames exist only on this PC. There is no backup."),
    "scanner": ("from scanner", "This project came from the scanner, which still has its own copy unless you deleted it there."),
    "unknown": ("", ""),
}


def backup_kind(path):
    """Where else the scan data lives: "scanner" (a project the MIRACO made: Project<14 digits>, or a models-only import),
    "none" (scanned live with this app: ProjectLive..., the only copy), else "unknown"."""
    p = Path(path); name = p.name
    if name.startswith("ProjectLive"): return "none"
    if re.fullmatch(r"Project\d{14}", name) or (p / ".pointyoink-models-only").is_file(): return "scanner"
    return "unknown"


def project_rows(folder):
    root = Path(folder).expanduser()
    projects = [root] if (root / 'data').is_dir() else list(root.iterdir())
    rows = []
    for p in projects:
        if not p.is_dir() or not (p / 'data').is_dir():
            continue
        try:
            meta = json.loads((p / '.pointyoink-home.json').read_text())
            if not isinstance(meta, dict): meta = {}
        except (OSError, ValueError):
            meta = {}
        models = [f for f in p.glob('*.ply') if f.is_file()]
        models.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        model = p / 'model.ply' if (p / 'model.ply').is_file() else next(iter(models), None)
        sides = [f for f in (p / 'data').iterdir() if f.is_dir()]
        stamp = scan_time(p.name) or max([f.stat().st_mtime for f in sides + models] or [(p / 'data').stat().st_mtime])   # a scanner project: when it was scanned
        pictures = [p / PREVIEW_NAME, p / 'model.png']
        pictures.extend(f for f in sorted(p.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
                        if not any(w in f.name for w in ('needs', 'compare', 'plane')))      # a coverage map or a comparison is not the model
        pictures.extend(sorted((p / 'data').glob('*/preview.png')))        # the scanner's own side pictures
        name = str(meta.get('name') or p.name); t = scan_time(p.name)
        shown = ('Scan · ' + datetime.fromtimestamp(t).strftime('%b %d, %I:%M %p').replace(' 0', ' ')) if needs_name(name) and t else name
        rows.append(dict(backup=backup_kind(p), path=str(p.resolve()), name=shown, unnamed=needs_name(name), folder=p.name, models_only=(p / '.pointyoink-models-only').is_file(),
                         date=stamp, sides=len(sides), model=str(model) if model else None,
                         pictures=[str(f) for f in pictures if f.is_file()]))
    return sorted(rows, key=lambda row: row['date'], reverse=True)


def preview_bytes(row, force=False):
    """Read an existing image or render a bounded mesh preview on a worker thread."""
    import io
    from PIL import Image
    cache = Path(row['path']) / PREVIEW_NAME
    stale = False
    if cache.is_file() and row['model']:
        try: stale = cache.stat().st_mtime_ns < Path(row['model']).stat().st_mtime_ns
        except OSError: stale = True
    pictures = [] if force or stale else ([str(cache)] if cache.is_file() else []) + row['pictures']
    if row['model']:
        pictures = [p for p in pictures if Path(p).name == PREVIEW_NAME]
    for path in pictures:
        try:
            with Image.open(path) as source:
                source.thumbnail((560, 320)); image = source.convert('RGB')
                data = io.BytesIO(); image.save(data, format='PNG'); return data.getvalue()
        except (OSError, ValueError):
            continue
    if not row['model']:
        if force: raise ValueError('Build a model to generate its preview.')
        return None
    from app2.builder.view_data import estimate_bytes, load_file
    if estimate_bytes(row['model']) > 256 * 1024 ** 2:
        if force: raise ValueError('Open the model in Build to save its preview.')
        return None
    import numpy as np
    import shade
    model = load_file(row['model'])
    if model['kind'] != 'mesh' or not len(model['V']) or not len(model['F']):
        return None
    vertices = model['V'].astype(np.float64)
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    vertices *= .85 / max(float(np.abs(vertices).max()), 1e-9)
    faces = model['F'][::max(1, (len(model['F']) + 39999) // 40000)]
    if model.get("vrgb") is not None:
        from app2.builder.color_preview import render
        image = render(vertices,faces,model["vrgb"])
    else:
        image = shade.render(vertices, faces, size=(560, 320), grid=False, gizmo=False)
    data = io.BytesIO(); image.save(data, format='PNG'); raw = data.getvalue()
    dest = Path(row['path']) / PREVIEW_NAME
    if force or stale or not dest.exists():
        try: _atomic_write(dest, raw)
        except OSError:
            if force: raise
    return raw
