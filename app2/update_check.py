# SPDX-License-Identifier: AGPL-3.0-or-later
"""Help > Check for updates: asks GitHub for the newest release, only when the user clicks. Nothing is sent but the request."""
import json
import re
import urllib.request

API = 'https://api.github.com/repos/datboip/point-yoink-v2/releases/latest'
PAGE = 'https://github.com/datboip/point-yoink-v2/releases/latest'


def parse(text):
    """'v2.0.4' or '2.0.4' as (2, 0, 4); None when it is not a plain release number."""
    m = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', str(text).strip())
    return tuple(int(x) for x in m.groups()) if m else None


def latest(fetch=None):
    """The newest release tag, or '' when GitHub cannot be reached or answers oddly."""
    try:
        if fetch is not None: return str(fetch())
        req = urllib.request.Request(API, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'PointYoink'})
        with urllib.request.urlopen(req, timeout=8) as r:
            return str(json.load(r).get('tag_name', ''))
    except Exception:
        return ''


def compare(current, newest):
    """'newer' when a newer release exists, 'current' when this is the newest, 'unknown' when it cannot be told."""
    a, b = parse(current), parse(newest)
    if a is None or b is None: return 'unknown'
    return 'newer' if b > a else 'current'
