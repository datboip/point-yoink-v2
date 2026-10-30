# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scanner's white fill light. It is not part of plain v4l2: the scanner takes it through a USB extension control
(UVCIOC_CTRL_QUERY, unit 4, selector 7), found by testing the scanner. Only the fill light register is ever written;
the IR projectors are left alone, and nothing else is sent."""
import os

IOC_XU = 0xC0107521                # UVCIOC_CTRL_QUERY
LED_COMMAND = "echo s 0xb02 %d >/dev/rk_preisp"      # 0xb02 = white fill LED


def led_command(on):
    return LED_COMMAND % (1 if on else 0)


def _send(command, node="/dev/video0"):
    import ctypes, fcntl
    class Query(ctypes.Structure):
        _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8), ("query", ctypes.c_uint8),
                    ("size", ctypes.c_uint16), ("data", ctypes.POINTER(ctypes.c_uint8))]
    data = (command.encode() + b"\x00" * 60)[:60]; array = (ctypes.c_uint8 * 60)(*data)
    fd = os.open(node, os.O_RDWR)
    try: fcntl.ioctl(fd, IOC_XU, Query(4, 7, 0x01, 60, array))
    finally: os.close(fd)


def set_fill_led(on):
    _send(led_command(on))
