# SPDX-License-Identifier: AGPL-3.0-or-later
"""The folders PointYoink saves into, all inside one home folder (~/PointYoink unless the user moved it): Scans for the
project library, Exports for exported models, Screenshots and Recordings for files copied off the scanner. Each one can be
moved on its own in File > Preferences; --root still wins for the library. The choices are kept in library.json next to
the app's config.json (POINTYOINK_CONFIG). An install that already saves into an older default keeps using it until the
user picks a new folder, so no library goes missing after an update."""
import json, os
from pathlib import Path

HOME = "~/PointYoink"
FOLDERS = {"projects": ("projects_dir", "Scans", "~/revopoint-scans-models"),
           "exports": ("exports_dir", "Exports", None),
           "screenshots": ("screenshots_dir", "Screenshots", "~/Pictures/PointYoink"),
           "recordings": ("recordings_dir", "Recordings", "~/Videos/PointYoink")}


def _store():
    config = Path(os.environ.get("POINTYOINK_CONFIG") or Path.home() / ".config/pointyoink/config.json")
    return config.parent / "library.json"


def _read():
    try:
        d = json.loads(_store().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError): return {}


def _write(key, path):
    d = _read(); d[key] = os.path.abspath(os.path.expanduser(path))
    p = _store(); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(p)


def _full(path): return os.path.abspath(os.path.expanduser(path))


def home(): return _full(_read().get("home_dir") or HOME)
def set_home(path): _write("home_dir", path)


def folder(kind):
    """Where this kind of file goes: the folder the user picked, else <home>/<Name>, else an older default still in use."""
    key, name, old = FOLDERS[kind]; d = _read()
    if d.get(key): return _full(d[key])
    new = os.path.join(home(), name)
    if old and not os.path.isdir(new) and os.path.isdir(_full(old)): return _full(old)
    return new


def set_folder(kind, path): _write(FOLDERS[kind][0], path)


def export_path(name):
    """The suggested save path for an export: the Exports folder plus the file name."""
    return os.path.join(folder("exports"), name)


def ensure_parent(path):
    """Make the folder a save dialog opens in, so a new Exports folder is there the first time."""
    try: os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError: pass


def default_dir(): return folder("projects")
def set_default(path): set_folder("projects", path)
def capture_dir(kind): return folder(kind)
def set_capture_dir(kind, path): set_folder(kind, path)
