# SPDX-License-Identifier: AGPL-3.0-or-later
from app2.scan import scanner_led


def test_only_the_fill_light_register_is_written(monkeypatch):
    sent = []
    monkeypatch.setattr(scanner_led, "_send", lambda command, node="/dev/video0": sent.append(command))
    scanner_led.set_fill_led(True); scanner_led.set_fill_led(False); scanner_led.set_fill_led(5)
    assert sent == ["echo s 0xb02 1 >/dev/rk_preisp", "echo s 0xb02 0 >/dev/rk_preisp", "echo s 0xb02 1 >/dev/rk_preisp"]
    assert all("0xb02" in s for s in sent)


def test_the_scanner_loader_finds_it_without_a_dev_folder():
    from app2.scan import sources
    assert sources.set_fill_led is scanner_led.set_fill_led
