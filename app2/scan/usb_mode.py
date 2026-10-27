# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which mode the MIRACO is plugged in with, from the USB ids in sysfs (no device access, nothing sent): the scanner
changes its USB identity with its mode, so "plugged in but in File Transfer" can be told from "not plugged in"."""
import os

VENDOR = "2207"                                     # Rockchip, the MIRACO's board
MODES = {"0017": "transfer", "0005": "pc", "110c": "pc"}   # File Transfer (MTP) / PC mode (cameras)
TEXT = {
    "transfer": ("Scanner in File Transfer mode",
                 "The scanner is plugged in with File Transfer on. Choose PC mode on the scanner to scan, or use Import to copy its projects."),
    "pc": ("Scanner starting", "The scanner is in PC mode. Waiting for its cameras..."),
    None: ("Scanner not connected",
           "Scanner not connected. Connect the USB cable and choose PC mode on the scanner to scan. Saved sides can still be built, edited and exported."),
}


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


def pill_text(): return TEXT.get(usb_mode(), TEXT[None])[0]
def help_text(): return TEXT.get(usb_mode(), TEXT[None])[1]
