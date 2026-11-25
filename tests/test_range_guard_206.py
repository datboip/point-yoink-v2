# SPDX-License-Identifier: AGPL-3.0-or-later
"""2.0.6 scans with the MIRACO only: a RANGE (2207:110c) on USB is named, and nothing is sent to it."""
import os, pytest


def fake_usb(tmp_path, product):
    d = tmp_path / "usb" / "1-5"; d.mkdir(parents=True)
    (d / "idVendor").write_text("2207\n"); (d / "idProduct").write_text(product + "\n")
    v = tmp_path / "v4l" / "video0"; v.mkdir(parents=True); (v / "device").symlink_to(d / "1-5:1.0", target_is_directory=True)
    (d / "1-5:1.0").mkdir()
    return str(tmp_path / "usb"), str(tmp_path / "v4l")


@pytest.mark.parametrize("product,model", [("110c", "range"), ("0005", "miraco"), ("0017", "miraco")])
def test_models(tmp_path, monkeypatch, product, model):
    monkeypatch.delenv("POINTYOINK_NO_DEVICE", raising=False)
    from app2.scan import usb_mode
    usb, v4l = fake_usb(tmp_path, product)
    assert usb_mode.scanner_model(usb) == model
    assert usb_mode.camera_model("/dev/video0", v4l) == model


def test_range_is_refused_and_named(monkeypatch):
    monkeypatch.delenv("POINTYOINK_NO_DEVICE", raising=False)
    from app2.scan import sources, usb_mode
    monkeypatch.setattr(usb_mode, "camera_model", lambda node, root=None: "range")
    monkeypatch.setattr(sources.os.path, "exists", lambda p: True)
    sent = []; monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: sent.append(a))
    with pytest.raises(RuntimeError): sources.set_ir_gain(4)
    with pytest.raises(RuntimeError): sources.fill_led(True)
    src = sources.LiveSource()
    assert not src.present() and "2.0.7" in src.gone_text()
    with pytest.raises(RuntimeError): src.restart()
    assert sent == []
    monkeypatch.setattr(usb_mode, "usb_mode", lambda root=None: "pc"); monkeypatch.setattr(usb_mode, "scanner_model", lambda root=None: "range")
    assert "2.0.7" in usb_mode.pill_text()
