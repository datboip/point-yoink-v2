# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anonymize a piece of text before it is shared. Pure Python, no Qt.

`Scrubber(context).scrub(text)` replaces what identifies a person or their files with stable aliases (the same input gets the same
alias inside one report) and keeps what a developer needs: versions, GPU, stage names, timings, numbers, error text.
It is a best effort and says so in the report window: the user reads the copy before sharing it. Not a promise.

context: home (str), user, host, roots (library folders), projects (project folder paths), extra (more literals to hide).
"""
import ipaddress
import os
import re
import socket
import getpass

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:.]*(?:%[A-Za-z0-9_.-]+)?(?![0-9A-Za-z:])")
MAC = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")
MAC_CISCO = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}(?![0-9A-Fa-f])")
OTHER_HOME = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[^/\\\s:'\"]+")
PROJECT_NAME = re.compile(r"Project(?:Live)?(?:Merged-)?[0-9][0-9-]{5,}")
SIDE_ID = re.compile(r"(?<!\d)\d{14}(?!\d)")
WIFI_CODE = re.compile(r"(?i)(\b(?:Wi[- ]?Fi|WLAN|share|pair(?:ing)?|scanner)\b(?:\s+\w+){0,2}\s+code\s*[:=#]?\s*)(\d{4,8})\b")
WIFI_CODE_KEY = re.compile(r"(?i)(\b(?:wifi|share|pairing)[_-]?code\b[\"']?\s*[:=]\s*[\"']?)(\d{4,8})\b")
SERIAL = re.compile(r"(?i)(\b(?:serial(?:\s+number)?|iSerial|sn|s/n)\b\s*[:=#]?\s*)([A-Za-z0-9._-]{6,})")
SSID = re.compile(r"(?i)(\bssid\b[\"']?\s*[:=]\s*[\"']?)([^\"'\n,}\]]+)")
LABELED_TOKEN = re.compile(r"(?i)(\b(?:token|api[_-]?key|access[_-]?key|secret|password|passphrase)\b[\"']?\s*[:=]\s*[\"']?|\bpairing\s+key\s+|\bAuthorization\s*:\s*Bearer\s+|\bBearer\s+)([A-Za-z0-9._~+/-]+(?:\r?\n[ \t]*[A-Za-z0-9._~+/-]{8,})?)")
URL = re.compile(r"(?i)\b(?:https?|ftp|sftp|ssh|file)://[^\s'\"<>)]*")
SCP_URL = re.compile(r"\b(?:git|ssh)@[A-Za-z0-9.-]+:[^\s'\"<>]+")
SAFE_HOSTS = ("github.com", "www.github.com", "pypi.org", "python.org", "www.python.org", "open3d.org", "www.open3d.org", "doc.qt.io", "www.qt.io")
HOST = re.compile(r"(?<![A-Za-z0-9._-])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9._-])")
HOST_CONTEXT = re.compile(r"(?i)(\b(?:host(?:name)?|connected\s+to|peer)\s*[:=]?\s*)([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)")
HOST_SUFFIXES = {"com", "net", "org", "io", "edu", "gov", "local", "lan", "internal", "example", "dev", "app"}
TOKEN = re.compile(r"(?<![A-Za-z0-9_-])(?:[0-9a-fA-F]{32,}|[A-Za-z0-9_-]{40,})(?![A-Za-z0-9_-])")
KEEP_IPS = {"127.0.0.1", "0.0.0.0"}
EMPTY_VALUES = {"none", "missing", "unknown", "unset", "unavailable", "not found"}
PATH_START = r"(?:/home/|/Users/|/mnt/[A-Za-z]/Users/|/media/|/run/media/|[A-Za-z]:\\Users\\|\\\\[^\\\s]+\\|~/)"
PATH_QUOTED = re.compile(r"(?P<quote>[\"'])" + PATH_START + r"[^\"'\n]{1,500}(?P=quote)", re.I)
PATH_WITH_EXT = re.compile(PATH_START + r"(?:[^\"'<>,;}\]\n]|\r?\n[ \t]+){1,500}?\.(?:py|ply|stl|obj|json|dph|inf|irl|png|jpg|jpeg|log|txt|gz|zip|pose|bin|db|sqlite|yaml|yml|toml|md|sh|exe|dll|so)\b", re.I)
PATH_TOKEN = re.compile(PATH_START + r"[^\s\"'<>,;}\]]+", re.I)


def _word(name):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", re.I)


class Scrubber:
    def __init__(self, context=None):
        c = dict(context or {})
        self.home = c.get("home") or os.path.expanduser("~")
        self.user = c.get("user") or _safe(getpass.getuser)
        self.host = c.get("host") or _safe(socket.gethostname)
        self.roots = [r for r in c.get("roots", []) if r]
        self.projects = [p for p in c.get("projects", []) if p]
        self.extra = [e for e in c.get("extra", []) if e]
        self.alias = {}                                            # literal -> alias, stable inside this report

    def _next(self, kind, key):
        a = self.alias.get((kind, key))
        if a is None:
            n = sum(1 for k in self.alias if k[0] == kind)
            a = self.alias[(kind, key)] = ("Project-%s" % _letters(n)) if kind == "project" else ("side-%d" % (n + 1))
        return a

    def scrub(self, text):
        t = str(text)
        # 1. things the app knows: longest first so a project path goes before the folder around it
        for p in sorted(set(self.projects), key=len, reverse=True):
            alias = self._next("project", os.path.basename(p.rstrip("/\\")))
            t = re.sub(re.escape(p), lambda _m: alias, t, flags=re.I)
            t = re.sub(re.escape(os.path.basename(p.rstrip("/\\"))), lambda _m: alias, t, flags=re.I)
        for r in sorted(set(self.roots), key=len, reverse=True):
            t = re.sub(re.escape(r), "<library>", t, flags=re.I)
        for lit in self.extra:
            t = re.sub(re.escape(lit), "<hidden>", t, flags=re.I)
        # 2. patterns
        t = PROJECT_NAME.sub(lambda m: self._next("project", m.group(0)), t)
        t = SIDE_ID.sub(lambda m: self._next("side", m.group(0)), t)
        t = URL.sub(_url_alias, t)
        t = SCP_URL.sub("<url>", t)
        t = LABELED_TOKEN.sub(lambda m: m.group(0) if m.group(2).lower() in EMPTY_VALUES
                              and not m.group(1).endswith(("'", '"')) else m.group(1) + "<token>", t)
        t = SSID.sub(lambda m: m.group(0) if any(m.group(2).strip().lower() == value
                     or m.group(2).strip().lower().startswith(value + " ") for value in EMPTY_VALUES)
                     and not m.group(1).endswith(("'", '"')) else m.group(1) + "<ssid>", t)
        t = WIFI_CODE.sub(lambda m: m.group(1) + "<code>", t)
        t = WIFI_CODE_KEY.sub(lambda m: m.group(1) + "<code>", t)
        t = SERIAL.sub(lambda m: m.group(1) + "<serial>", t)
        t = PATH_QUOTED.sub(lambda m: m.group("quote") + _path_alias(m) + m.group("quote"), t)
        t = PATH_WITH_EXT.sub(_path_alias, t)
        t = PATH_TOKEN.sub(_path_alias, t)
        t = OTHER_HOME.sub(lambda m: _home_alias(m.group(0)), t)
        if self.home and len(self.home) > 1:
            t = re.sub(re.escape(self.home), "~", t, flags=re.I)
        t = EMAIL.sub(lambda m: m.group(0) if m.group(0).endswith(".service") else "<email>", t)
        t = MAC.sub("<mac>", t)
        t = MAC_CISCO.sub("<mac>", t)
        t = IPV6.sub(lambda m: "<ip>" if _ipv6(m.group(0).split("%", 1)[0]) else m.group(0), t)
        t = IPV4.sub(lambda m: m.group(0) if (m.group(0) in KEEP_IPS or not _ipv4(m.group(0)) or _looks_like_version(m)) else "<ip>", t)
        t = HOST_CONTEXT.sub(lambda m: m.group(1) + (m.group(2) if m.group(2).lower() in SAFE_HOSTS
                             or re.fullmatch(r"[\d.]+", m.group(2)) else "<computer>"), t)
        t = HOST.sub(lambda m: "<computer>" if m.group(0).lower().rsplit(".", 1)[-1] in HOST_SUFFIXES
                     and m.group(0).lower() not in SAFE_HOSTS else m.group(0), t)
        t = TOKEN.sub("<token>", t)
        # 3. who and where: after the paths, so /home/<user> is already gone and the bare name is what is left
        for name, alias in ((self.user, "<user>"), (self.host, "<computer>")):
            if name and len(name) >= 3: t = _word(name).sub(alias, t)
        return t


def _home_alias(path):
    return "/home/<user>" if path.startswith("/home/") else ("/Users/<user>" if path.startswith("/Users/") else "C:\\Users\\<user>")


def _path_alias(m):
    """Keep only scanner-generated path markers; personal filenames are not safe."""
    path = m.group(0)
    frame = re.search(r"\bframe_\d{3}_\d{4}\.(?:dph|inf|irl)\b", path, re.I)
    stage = re.search(r"\b(?:lineup\.(?:fix|pairs|place|refine)|side\.(?:fuse|platter|keep))\b", path, re.I)
    return "<path>" + ("/" + frame.group(0) if frame else "") + ("/" + stage.group(0) if stage else "")


def _url_alias(m):
    raw = m.group(0)
    body = raw.split("://", 1)[1]
    host = body.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].lower()
    path = "/" + body.split("/", 1)[1].split("?", 1)[0].split("#", 1)[0] if "/" in body else ""
    if host in ("github.com", "www.github.com") and re.fullmatch(r"/datboip/point-yoink-v2/issues(?:/\d+)?", path):
        return "https://" + host + path
    if host in SAFE_HOSTS and not path:
        return "https://" + host
    return "<url>"


def _looks_like_version(m):
    """"firmware 1.0.0.129" is four numbers with dots like an address, but it is a version."""
    before = m.string[max(0, m.start() - 24):m.start()]
    return bool(re.search(r"(?i)\b(?:firmware|version|build|fw|v)\s*[:=]?\s*$", before))


def _safe(fn):
    try: return fn()
    except Exception: return ""


def _letters(n):
    s = ""
    n += 1
    while n: n, r = divmod(n - 1, 26); s = chr(65 + r) + s
    return s


def _ipv4(s):
    try: ipaddress.IPv4Address(s); return True
    except ValueError: return False


def _ipv6(s):
    try: ipaddress.IPv6Address(s); return True
    except ValueError: return False
