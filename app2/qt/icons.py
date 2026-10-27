# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qt access to the shared PointYoink icon assets."""
from pathlib import Path
from PySide6.QtGui import QIcon, QPixmap
from app2 import REPO


def icon(name, state='default'):
    root = Path(REPO) / 'assets' / 'icons' / 'png'
    result = QIcon()
    for size in (32, 36, 40, 48):
        for variant, mode in ((state, QIcon.Mode.Normal), ('muted', QIcon.Mode.Disabled)):
            path = root / variant / str(size) / (name + '.png')
            if path.is_file(): result.addPixmap(QPixmap(str(path)), mode)
    return result
