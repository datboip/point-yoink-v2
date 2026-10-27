"""The developer driver: off by default, and when on it can find, click, type, read text and take small pictures."""
import json
import os
import socket
import tempfile
import threading
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XMODIFIERS"] = "@im=none"
os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink_driver_")) / "config.json")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app2.diagnostics import driver as D


@pytest.fixture
def rig(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(D, "SHOTS", tempfile.mkdtemp(prefix="pointyoink_shots_"))
    win = QWidget(); lay = QVBoxLayout(win); clicks = []
    btn = QPushButton("Do the thing"); btn.clicked.connect(lambda: clicks.append(1)); box = QLineEdit(); lay.addWidget(QLabel("Name here")); lay.addWidget(box); lay.addWidget(btn)
    win.show(); app.processEvents()
    path = os.path.join(tempfile.mkdtemp(prefix="pyk_"), "d.sock"); drv = D.Driver(win, path)

    def rpc(**req):
        out = {}
        def go():
            s = socket.socket(socket.AF_UNIX); s.settimeout(20); s.connect(path); s.sendall((json.dumps(req) + "\n").encode()); buf = b""
            while not buf.endswith(b"\n"): buf += s.recv(65536)
            out.update(json.loads(buf))
        t = threading.Thread(target=go); t.start()
        while t.is_alive(): app.processEvents(); t.join(0.01)
        return out
    yield rpc, clicks, box, win
    win.close(); drv.server.close(); app.processEvents()


def test_it_does_nothing_unless_the_environment_names_a_socket(monkeypatch):
    app = QApplication.instance() or QApplication([]); monkeypatch.delenv("POINTYOINK_DRIVER", raising=False)
    assert D.maybe_start(QWidget()) is None


def test_tree_texts_click_and_type(rig):
    rpc, clicks, box, win = rig
    tree = rpc(cmd="tree")
    assert tree["ok"] and any(c["text"] == "Do the thing" and c["kind"] == "button" for c in tree["controls"])
    edit = next(c for c in tree["controls"] if c["kind"] == "edit"); assert edit["near"] == "Name here"
    assert "Name here" in rpc(cmd="texts")["texts"]
    assert rpc(cmd="click", text="the thing")["ok"] and clicks == [1]
    assert rpc(cmd="type", id=edit["id"], value="hello", enter=False)["ok"] and box.text() == "hello"


def test_errors_are_answered_and_a_picture_is_small_and_saved(rig):
    rpc, clicks, box, win = rig
    assert rpc(cmd="nope")["ok"] is False and "known:" in rpc(cmd="nope")["error"]
    bad = rpc(cmd="click", text="not there"); assert bad["ok"] is False and "no control" in bad["error"]
    shot = rpc(cmd="screenshot", max_width=200)
    assert shot["ok"] and os.path.exists(shot["path"]) and shot["size"][0] <= 200 and shot["bytes"] < 20000


def test_explicit_screenshot_path_does_not_touch_default_folder(rig, tmp_path, monkeypatch):
    rpc, _clicks, _box, _win = rig
    unused = tmp_path / "unused-default"
    monkeypatch.setattr(D, "SHOTS", str(unused))
    path = tmp_path / "isolated" / "checkpoint.jpg"
    result = rpc(cmd="screenshot", path=str(path), max_width=200)
    assert result["ok"] and path.is_file()
    assert not unused.exists()
