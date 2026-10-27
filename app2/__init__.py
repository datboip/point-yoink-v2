# SPDX-License-Identifier: AGPL-3.0-or-later
"""PointYoink 2.0: the six-step app (Scan, Build, Review, Clean, Measure, Export) in Qt.
app2.scan holds everything the Scan step does without a window (project, sides, frames, tracking, saving);
app2.builder runs a build and follows its progress; app2.qt is only the window on top of both.
(The build package is app2/builder, not app2/build: the repo's .gitignore drops every folder called build/.)"""
import os, sys
from app2.version import VERSION as __version__

# the checkout this package sits in (a clone or a git worktree anywhere); POINTYOINK_REPO overrides it
_CHECKOUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.expanduser(os.environ.get("POINTYOINK_REPO") or _CHECKOUT))
SCANNER = os.path.join(REPO, "scanner")
for _p in (SCANNER, REPO):                 # track.py, fuse.py, register.py live in the repo root; table_plane.py in scanner/
    if _p not in sys.path: sys.path.insert(0, _p)
