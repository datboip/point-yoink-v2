# SPDX-License-Identifier: AGPL-3.0-or-later
"""A project's scanner guide mode, shared by Build and Touch up."""
import json
import os
from pathlib import Path

MODES=('turntable','handheld')


def selected_mode(project):
    if not project:return 'turntable'
    try:
        mode=json.loads((Path(project)/'build'/'scan-guide-mode.json').read_text()).get('mode')
        return mode if mode in MODES else 'turntable'
    except (OSError,ValueError,TypeError,AttributeError):return 'turntable'


def select_mode(project,mode):
    if mode not in MODES:raise ValueError('Choose table turns or scanner moves')
    if not project:return
    path=Path(project)/'build'/'scan-guide-mode.json';path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(mode=mode)))
    os.replace(tmp,path)
