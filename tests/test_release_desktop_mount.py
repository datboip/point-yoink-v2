# SPDX-License-Identifier: AGPL-3.0-or-later
"""The desktop's own MTP mount of the scanner is released before USB import (only the Revopoint one)."""
from types import SimpleNamespace
from app2.transfer import captures


def test_only_the_scanner_mount_is_released(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/gio')
    calls = []
    listing = ("Drive(0): Samsung SSD\n"
               "Mount(0): MIRACO -> mtp://Revopoint_MIRACO_A1B2/\n"
               "Mount(1): Phone -> mtp://SAMSUNG_Android_XYZ/\n")
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=listing if cmd[:3] == ['gio', 'mount', '-l'] else '', returncode=0)
    assert captures.release_desktop_mount(run) == ['mtp://Revopoint_MIRACO_A1B2/']
    assert ['gio', 'mount', '-u', 'mtp://SAMSUNG_Android_XYZ/'] not in calls


def test_no_gio_does_nothing(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: None)
    assert captures.release_desktop_mount(lambda *a, **k: 1 / 0) == []
