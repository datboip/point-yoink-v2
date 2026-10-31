# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app version. Change it only when cutting a release, never on launch.
The window title and About also carry the code's own mark (commit, and "+" when files changed since), so a running build can always be told apart."""
import os, subprocess

VERSION = "2.0.2"


def _build():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:                                                # a packaged copy has no git: the build script leaves the mark here
        with open(os.path.join(root, "app2", "BUILD")) as f: return f.read().strip()
    except OSError: pass
    try:
        run = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=3).stdout.strip()
        head = run("rev-parse", "--short", "HEAD")
        if not head: return ""
        return head + ("+" if run("status", "--porcelain", "--untracked-files=no") else "")
    except Exception: return ""


BUILD = _build()
APP_TITLE = f"PointYoink {VERSION}" + (f" ({BUILD})" if BUILD else "")
