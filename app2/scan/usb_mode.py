# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which mode the MIRACO is plugged in with, from the USB ids in sysfs (no device access, nothing sent): the scanner
changes its USB identity with its mode, so "plugged in but in File Transfer" can be told from "not plugged in"."""
import os

VENDOR = "2207"                                     # Rockchip, the MIRACO's board
MODES = {"0017": "transfer", "0005": "pc", "110c": "pc"}   # File Transfer (MTP) / PC mode (cameras)
MODELS = {"0017": "miraco", "0005": "miraco", "110c": "range"}   # which scanner each identity belongs to
TEXT = {
    "transfer": ("Scanner in File Transfer mode",
                 "The scanner is plugged in with File Transfer on. Choose PC mode on the scanner to scan, or use Import to copy its projects."),
    "pc": ("Scanner starting", "The scanner is in PC mode. Waiting for its cameras..."),
    None: ("Scanner not connected",
           "Scanner not connected. Connect the USB cable and choose PC mode on the scanner to scan. Saved sides can still be built, edited and exported."),
}

RANGE_LATER = ("RANGE: coming in 2.0.7",
               "A Revopoint RANGE is plugged in. This version scans with the MIRACO only, so nothing is sent to the RANGE. RANGE support comes in PointYoink 2.0.7.")

wifi_alive = False                                  # set by the motion panel: the scanner is sending motion over Wi-Fi right now
HUNG = ("Scanner not on USB",
        "The scanner is on Wi-Fi but its USB side stopped answering, so a new cable will not help. Turn the scanner off and on, then choose PC mode.")


def usb_mode(root="/sys/bus/usb/devices"):
    """'transfer', 'pc', 'other' (a Rockchip device in another mode) or None (no scanner on USB)."""
    if os.environ.get("POINTYOINK_NO_DEVICE") == "1": return None     # tests and no-device runs never look at hardware
    try: names = os.listdir(root)
    except OSError: return None
    found = None
    for n in names:
        try:
            with open(os.path.join(root, n, "idVendor")) as f: v = f.read().strip()
            if v != VENDOR: continue
            with open(os.path.join(root, n, "idProduct")) as f: p = f.read().strip()
        except OSError: continue
        found = MODES.get(p, found or "other")
        if found in ("transfer", "pc"): return found
    return found


def scanner_model(root="/sys/bus/usb/devices"):
    """'miraco', 'range' or None: the scanner on USB (a MIRACO first when both are plugged in)."""
    if os.environ.get("POINTYOINK_NO_DEVICE") == "1": return None
    try: names = os.listdir(root)
    except OSError: return None
    found = None
    for n in names:
        try:
            with open(os.path.join(root, n, "idVendor")) as f: v = f.read().strip()
            if v != VENDOR: continue
            with open(os.path.join(root, n, "idProduct")) as f: p = f.read().strip()
        except OSError: continue
        m = MODELS.get(p)
        if m == "miraco": return m
        found = m or found
    return found


def camera_model(node, root="/sys/class/video4linux"):
    """'miraco', 'range' or None for the scanner a camera node (/dev/videoN) belongs to. None: no such camera, or not a
    scanner we know. Reads sysfs only. With POINTYOINK_NO_DEVICE=1 there is no scanner (everything is refused anyway)."""
    if os.environ.get("POINTYOINK_NO_DEVICE") == "1": return None
    try:
        usb = os.path.dirname(os.path.realpath(os.path.join(root, os.path.basename(node), "device")))
        with open(os.path.join(usb, "idVendor")) as f: v = f.read().strip()
        with open(os.path.join(usb, "idProduct")) as f: p = f.read().strip()
    except OSError: return None
    return MODELS.get(p) if v == VENDOR else None


def _text(i):
    m = usb_mode()
    if m == "pc" and scanner_model() == "range": return RANGE_LATER[i]
    return HUNG[i] if m is None and wifi_alive else TEXT.get(m, TEXT[None])[i]


def pill_text(): return _text(0)
def help_text(): return _text(1)
