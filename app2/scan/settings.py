# SPDX-License-Identifier: AGPL-3.0-or-later
"""Last live scan controls, kept in the app config."""
import json
import os
from pathlib import Path


def path():
    return Path(os.environ.get("POINTYOINK_CONFIG") or Path.home()/".config/pointyoink/config.json")


def load():
    try:
        data=json.loads(path().read_text()).get("scan", {})
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError,TypeError): return {}


def save(**values):
    p=path()
    try: data=json.loads(p.read_text())
    except (OSError,ValueError): data={}
    if not isinstance(data,dict): data={}
    current=data.get("scan", {})
    if not isinstance(current,dict): current={}
    current.update(values);data["scan"]=current
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".scan.tmp");tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
