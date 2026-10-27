# SPDX-License-Identifier: AGPL-3.0-or-later
"""General preferences (File > Preferences, General tab), kept under "general" in the app's config.json next to the scan
settings. Each has a default here, so a missing file means the shipped behavior.

  default_look        "shaded" | "scan_color"   what the Build view shows when a project opens (2026-10-24: shaded)
  light_minutes       int                       the fill light asks "still using it?" after this many minutes
  timelapse_watermark bool                      the PointYoink mark on exported timelapses, on by default
  open_last           bool                      start on the newest project instead of the Projects page"""
import json, os
from pathlib import Path

DEFAULTS = dict(default_look="shaded", light_minutes=5, timelapse_watermark=True, open_last=False)


def _path():
    return Path(os.environ.get("POINTYOINK_CONFIG") or Path.home() / ".config/pointyoink/config.json")


def load():
    try:
        d = json.loads(_path().read_text()).get("general", {})
        d = d if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError): d = {}
    out = dict(DEFAULTS)
    for k, v in d.items():
        if k in DEFAULTS and isinstance(v, type(DEFAULTS[k])): out[k] = v
    if out["default_look"] not in ("shaded", "scan_color"): out["default_look"] = "shaded"
    out["light_minutes"] = max(1, min(60, int(out["light_minutes"])))
    return out


def get(key): return load()[key]


def save(**values):
    p = _path()
    try: data = json.loads(p.read_text())
    except (OSError, ValueError): data = {}
    if not isinstance(data, dict): data = {}
    cur = data.get("general", {}); cur = cur if isinstance(cur, dict) else {}
    cur.update({k: v for k, v in values.items() if k in DEFAULTS}); data["general"] = cur
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".general.tmp"); tmp.write_text(json.dumps(data, indent=2)); tmp.replace(p)
