# SPDX-License-Identifier: AGPL-3.0-or-later
"""Developer driver: lets a test tool use the running app. OFF unless POINTYOINK_DRIVER=<socket path> is set.

A local Unix socket (owner only) takes one JSON object per line and answers with one JSON line. It can only do what a user can do with the
mouse and keyboard on this window, and read what is on screen. No network, never started by default, not meant for release builds.

Commands (all JSON, "cmd" first):
  tree [contains]                  the visible controls: id, kind, text, enabled, checked, rect, near (the label next to it)
  texts [contains]                 every visible text on screen (labels, buttons, tab names, chips in the 3D panes), top to bottom
  click text=... [nth] [exact] | id=...      click a button, tab, rail row, check box or combo by its label
  click_at pane=main|base|moving|before|<id> x=0..1 y=0..1     click inside a (3D) view at a fraction of its size
  type text=...|id=... value=... [enter]     click a box, select all, type, optionally press Enter
  choose id=...|text=... item=...            pick an entry of a combo box
  menus                            menu and action names with their enabled states
  menu menu="File" action="Preferences..."  activate an enabled menu action by name
  key keys="Ctrl+N"                          a key press
  idle ms=300                                let the app work for a while (events keep running)
  screenshot [id|pane|popup] [path] [max_width]    a small JPEG of the window or one widget, saved under ~/.cache/pointyoink/driver-shots
  state                            the Build page's stage, tab and line-up pose of every side as the engine has it
  models [pane]                    where each item sits in a 3D view (its model matrix: move in mm, turn in degrees)
  model_points [pane] [count]      click positions (x, y fractions) that land ON the shown model, spread over it, with the model point under each:
                                   aim clicks from these instead of a screenshot (a click in empty space adds nothing)
  view_stats [pane]                the view's camera and the model's size: elements, bounds, the share of the pane the model covers
"""
import json
import os
import time
import weakref

from PySide6.QtCore import QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLabel, QLineEdit, QPlainTextEdit, QTabBar, QWidget)

SHOTS = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "pointyoink", "driver-shots")


def maybe_start(win):
    """Start the driver on the socket named by POINTYOINK_DRIVER, else do nothing (the normal case)."""
    path = os.environ.get("POINTYOINK_DRIVER")
    return Driver(win, path) if path else None


class Driver(QObject):
    def __init__(self, win, path):
        from PySide6.QtNetwork import QLocalServer          # only when the driver is asked for: the app itself never needs Qt's network module
        super().__init__(win); self.win = win; self.path = path; self.ids = {}; self.next_id = 1; self.buffers = {}
        QLocalServer.removeServer(path)
        self.server = QLocalServer(self); self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not self.server.listen(path): raise RuntimeError("driver: cannot listen on %s: %s" % (path, self.server.errorString()))
        self.server.newConnection.connect(self._accept)
        print("driver: listening on %s (developer tool)" % path, flush=True)

    # ------------------------------------------------------------------ socket
    def _accept(self):
        sock = self.server.nextPendingConnection(); self.buffers[sock] = b""
        sock.readyRead.connect(lambda s=sock: self._read(s)); sock.disconnected.connect(lambda s=sock: self.buffers.pop(s, None))

    def _read(self, sock):
        if sock not in self.buffers: return                      # the client hung up before we read it
        self.buffers[sock] += bytes(sock.readAll())
        while sock in self.buffers and b"\n" in self.buffers[sock]:
            line, self.buffers[sock] = self.buffers[sock].split(b"\n", 1)
            if not line.strip(): continue
            try: resp = self.handle(json.loads(line))
            except Exception as e: resp = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            sock.write((json.dumps(resp, default=str) + "\n").encode()); sock.flush()

    def handle(self, req):
        fn = getattr(self, "cmd_" + str(req.get("cmd", "")), None)
        if fn is None: return {"ok": False, "error": "unknown cmd; known: " + ", ".join(n[4:] for n in dir(self) if n.startswith("cmd_"))}
        out = fn(req); out = out if isinstance(out, dict) else {"result": out}
        out.setdefault("ok", True); return out

    # ------------------------------------------------------------------ finding things
    def _windows(self): return [w for w in QApplication.topLevelWidgets() if w.isVisible()]

    def _visible(self):
        return [w for w in QApplication.allWidgets() if w.isVisible() and QWidget.window(w).isVisible() and w.width() > 0 and w.height() > 0]

    def _id(self, w):
        for k, ref in self.ids.items():
            if ref() is w: return k
        k = "w%d" % self.next_id; self.next_id += 1; self.ids[k] = weakref.ref(w); return k

    def _near(self, w):
        """The text of the closest label above or to the left of w in the same parent."""
        best, bd = "", 1e9
        for l in (w.parentWidget().findChildren(QLabel) if w.parentWidget() else []):
            if l is w or not l.text() or not l.isVisible(): continue
            a, b = l.mapTo(QWidget.window(w), QPoint(0, 0)), w.mapTo(QWidget.window(w), QPoint(0, 0))
            if a.y() <= b.y() + 4 and a.x() <= b.x() + 4:
                d = abs(b.y() - a.y()) * 3 + abs(b.x() - a.x())
                if d < bd: best, bd = l.text()[:60], d
        return best

    def _controls(self):
        out = []
        for w in self._visible():
            origin = w.mapTo(QWidget.window(w), QPoint(0, 0)); rect = [origin.x(), origin.y(), w.width(), w.height()]
            if isinstance(w, QAbstractButton) and (w.text() or w.accessibleName()): out.append((w, "check" if w.isCheckable() else "button", w.text() or w.accessibleName(), w.isChecked() if w.isCheckable() else None, rect, None))
            elif isinstance(w, QTabBar):
                for i in range(w.count()):
                    r = w.tabRect(i); o = w.mapTo(QWidget.window(w), r.topLeft()); out.append((w, "tab", w.tabText(i), w.currentIndex() == i, [o.x(), o.y(), r.width(), r.height()], i))
            elif isinstance(w, QComboBox): out.append((w, "combo", w.currentText(), None, rect, None))
            elif isinstance(w, QAbstractSpinBox): out.append((w, "spin", w.text(), None, rect, None))
            elif isinstance(w, QLineEdit): out.append((w, "edit", w.text() or w.placeholderText(), None, rect, None))
            elif hasattr(w, "sid") and hasattr(w, "name") and isinstance(getattr(w, "name"), QLabel): out.append((w, "row", w.name.text(), getattr(w, "current", None), rect, None))
        return out

    def _find(self, req):
        if req.get("name"):
            matches = [control for control in self._controls() if control[0].objectName() == req["name"]]
            if len(matches) != 1: raise ValueError("expected one control named %r (%d found)" % (req["name"], len(matches)))
            return matches[0][0], matches[0][5]
        if req.get("id"):
            ref = self.ids.get(req["id"]); w = ref() if ref else None
            if w is None: raise ValueError("no widget with id %s (call tree again)" % req["id"])
            return w, None
        text = str(req.get("text", "")); exact = bool(req.get("exact")); nth = int(req.get("nth", 0))
        hits = [c for c in self._controls() if (c[2] == text if exact else text.lower() in c[2].lower())]
        if len(hits) <= nth: raise ValueError("no control with text %r (%d found); use tree" % (text, len(hits)))
        hits.sort(key=lambda c: (c[4][1], c[4][0])); c = hits[nth]; return c[0], c[5]

    # ------------------------------------------------------------------ commands
    @staticmethod
    def _menu_text(action):
        return action.text().split("\t", 1)[0].replace("&", "").strip()

    def _menus(self):
        bar = getattr(self.win, "menuBar", lambda: None)()
        return [a for a in bar.actions() if a.isVisible() and a.menu()] if bar else []

    def cmd_menus(self, req):
        result = []
        for action in self._menus():
            menu = action.menu()
            menu.aboutToShow.emit()
            result.append({"menu": self._menu_text(action), "enabled": action.isEnabled(),
                           "actions": [{"text": self._menu_text(a), "enabled": a.isEnabled(),
                                        "submenu": a.menu() is not None}
                                       for a in menu.actions() if a.isVisible() and not a.isSeparator()]})
        return {"menus": result}

    def cmd_menu(self, req):
        if QApplication.activeModalWidget() is not None:
            raise ValueError("close the active dialog before using the main menu")
        menus = [a for a in self._menus() if self._menu_text(a) == str(req.get("menu", ""))]
        if len(menus) != 1:
            raise ValueError("expected one menu with this name; call menus")
        menu = menus[0].menu()
        menu.aboutToShow.emit()
        actions = [a for a in menu.actions() if a.isVisible() and not a.isSeparator()
                   and self._menu_text(a) == str(req.get("action", ""))]
        if len(actions) != 1:
            raise ValueError("expected one action with this name; call menus")
        action = actions[0]
        if not menus[0].isEnabled() or not action.isEnabled() or action.menu() is not None:
            raise ValueError("menu action is disabled or opens a submenu")
        # A modal dialog must not block this socket response. Its nested Qt loop
        # remains available to subsequent driver requests.
        def trigger_if_available():
            if QApplication.activeModalWidget() is None and menus[0].isEnabled() and action.isEnabled() and action.isVisible():
                action.trigger()
        QTimer.singleShot(0, trigger_if_available)
        return {"queued": self._menu_text(action)}

    def cmd_tree(self, req):
        needle = str(req.get("contains", "")).lower()
        rows = [dict(id=self._id(w), name=w.objectName(), kind=k, text=t[:80], checked=chk, enabled=w.isEnabled(), rect=r, near=self._near(w) if k in ("spin", "edit", "combo") else None)
                for w, k, t, chk, r, _i in sorted(self._controls(), key=lambda c: (c[4][1], c[4][0])) if needle in w.objectName().lower() or needle in t.lower() or needle in (self._near(w) if k in ("spin", "edit", "combo") else "").lower()]
        return {"controls": rows, "count": len(rows)}

    def cmd_texts(self, req):
        needle = str(req.get("contains", "")).lower(); seen, rows = set(), []
        for w in self._visible():
            if isinstance(w, QLabel) and w.text().strip():
                o = w.mapTo(QWidget.window(w), QPoint(0, 0)); rows.append((o.y(), o.x(), w.text().strip().replace("\n", " ")[:200]))
            elif hasattr(w, "parts") and hasattr(w, "note"):                    # the chips painted over a 3D view
                o = w.mapTo(QWidget.window(w), QPoint(0, 0)); rows.append((o.y(), o.x(), " | ".join([str(p[0]) for p in w.parts] + ([str(w.note)] if w.note else []) + ([str(w.center)] if getattr(w, "center", None) else []))[:200]))
            elif isinstance(w, QPlainTextEdit) and w.toPlainText().strip():
                o = w.mapTo(QWidget.window(w), QPoint(0, 0)); rows.append((o.y(), o.x(), "[text box] " + w.toPlainText().strip().splitlines()[-1][:160]))
        out = []
        for y, x, t in sorted(rows):
            if t and t not in seen and needle in t.lower(): seen.add(t); out.append(t)
        return {"texts": out}

    def _click(self, w, pos=None):
        pos = pos or w.rect().center(); QTest.mouseClick(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos); QApplication.processEvents()

    def cmd_click(self, req):
        w, tab = self._find(req)
        clicked = "tab" if isinstance(w, QTabBar) else getattr(w, "text", lambda: type(w).__name__)()
        if tab is not None: self._click(w, w.tabRect(tab).center())
        elif isinstance(w, QComboBox): w.showPopup(); w.hidePopup()
        else: self._click(w)
        QTest.qWait(int(req.get("wait", 120))); return {"clicked": clicked}

    def _pane(self, name):
        build = getattr(self.win, "build", None); vp = getattr(build, "vp", None)
        panes = {"main": getattr(vp, "main", None), "base": getattr(vp, "pa", None), "moving": getattr(vp, "pb", None), "before": getattr(vp, "before", None),
                 "strip": getattr(getattr(self.win, "workspace", None), "mini", None)}
        p = panes.get(name)
        if p is not None: return p.view if getattr(p, "view", None) is not None else p
        w, _t = self._find({"id": name}); return w

    @staticmethod
    def _shown_points(w):
        """The shown model's points: the editable elements, or (a pane that is not editable, like Points' base and moving) the vertices
        or points of its items placed by their model matrices. None when nothing is shown."""
        import numpy as np
        E = getattr(w, "elements", None)
        if E is not None and len(E[0]): return np.asarray(E[0], float)
        parts = []
        for it in (getattr(w, "items", None) or {}).values():
            A = it.get("V") if it.get("V") is not None else it.get("P")
            if A is None or not len(A): continue
            A = np.asarray(A, float); M = it.get("model")
            parts.append(A if M is None else A @ np.asarray(M, float)[:3, :3].T + np.asarray(M, float)[:3, 3])
        return np.vstack(parts) if parts else None

    def cmd_model_points(self, req):
        import numpy as np
        w = self._pane(str(req.get("pane", "main"))); n = int(req.get("count", 12))
        X = self._shown_points(w)
        if X is None: return {"points": [], "note": "nothing shown"}
        rng = np.random.default_rng(1)
        pw, ph = w._px(); k = w.devicePixelRatioF(); out = []
        for i in rng.permutation(len(X))[:4000]:
            # project through the same maths as the pick, then ask the pick what is there
            pj = np.asarray(w.cam.projection(pw / ph), float); V = np.asarray(w.cam.view, float)
            eye = V[:3, :3] @ X[i] + V[:3, 3]; cw = float(pj[3, :3] @ eye + pj[3, 3])
            if cw <= 1e-6: continue
            clip = pj[:3, :3] @ eye + pj[:3, 3]; sx = (clip[0] / cw * 0.5 + 0.5) * pw; sy = (0.5 - clip[1] / cw * 0.5) * ph
            if not (0 <= sx < pw and 0 <= sy < ph): continue
            from PySide6.QtCore import QPointF
            hit = w.pick_front(QPointF(sx / k, sy / k), r=1)
            if hit is not None:
                out.append({"x": round(sx / pw, 4), "y": round(sy / ph, 4), "world": [round(float(c), 2) for c in hit]})
                if len(out) >= n: break
        return {"points": out}

    def cmd_view_stats(self, req):
        import numpy as np
        w = self._pane(str(req.get("pane", "main"))); X = self._shown_points(w)
        if X is None: return {"elements": 0}
        return {"elements": int(len(X)), "min": [round(float(c), 1) for c in X.min(0)], "max": [round(float(c), 1) for c in X.max(0)],
                "tool": getattr(w, "tool", None), "size": [w.width(), w.height()]}

    def cmd_click_at(self, req):
        w = self._pane(str(req.get("pane", "main"))); x, y = float(req["x"]), float(req["y"])
        self._click(w, QPoint(int(w.width() * x), int(w.height() * y))); QTest.qWait(int(req.get("wait", 200)))
        return {"clicked_at": [int(w.width() * x), int(w.height() * y)], "size": [w.width(), w.height()]}

    def cmd_type(self, req):
        w, _t = self._find(req); self._click(w); QTest.keyClick(w, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier); QTest.keyClicks(w, str(req["value"]))
        if req.get("enter", True): QTest.keyClick(w, Qt.Key.Key_Return)
        QTest.qWait(150); return {"typed": str(req["value"])}

    def cmd_choose(self, req):
        w, _t = self._find(req)
        if not isinstance(w, QComboBox): raise ValueError("not a combo box")
        item = str(req["item"]); i = w.findText(item)
        if i < 0: i = next((k for k in range(w.count()) if item.lower() in w.itemText(k).lower()), -1)
        if i < 0: raise ValueError("no entry %r; entries: %s" % (item, [w.itemText(k) for k in range(w.count())]))
        w.setCurrentIndex(i); QTest.qWait(150); return {"chosen": w.itemText(i)}

    def cmd_key(self, req):
        seq = QKeySequence(str(req["keys"])); target = QApplication.activePopupWidget() or QApplication.activeModalWidget() or self.win
        k = seq[0]; QTest.keyClick(target, Qt.Key(k.key()), k.keyboardModifiers()); QTest.qWait(120); return {"pressed": str(req["keys"])}

    def cmd_idle(self, req):
        # a real event loop that sleeps until the timer: QTest.qWait polls and held the GIL 80% of the time, which slowed
        # every scan thread while a test waited (profiled)
        from PySide6.QtCore import QEventLoop
        loop = QEventLoop(); QTimer.singleShot(int(req.get("ms", 300)), loop.quit); loop.exec()
        return {"idle": int(req.get("ms", 300))}

    def cmd_colour(self, req):
        """Pick a Color mode of the Build view as the menu would (popup menus cannot be driven): colour mode=quality."""
        b = self.win.build; b.pick_colour(str(req["mode"]))
        return {"mode": b.colour_mode.get(b.stage), "gl": bool(b.gl), "display_path": b._display_path(), "host": b.host is not None, "tickets": len(b.tickets), "notice": getattr(getattr(b.vp, "notice", None), "text", None) and b.vp.notice.text.text()}

    def cmd_scanner(self, req):
        """Read-only: is the scanner streaming, and does it deliver a color picture (size, age) next to depth and IR."""
        import time
        ses = getattr(self.win, "session", None); src = getattr(ses, "source", None)
        if src is None: return {"session": ses is not None, "source": None}
        out = {"kind": type(src).__name__, "frame_age_s": round(time.monotonic() - src.last_t, 2), "frames": getattr(src, "seq", None), "recording": bool(getattr(ses, "recording", False))}
        c = src.colour(); out["colour_bytes"] = len(c) if c else 0
        if c and req.get("save"): open(str(req["save"]), "wb").write(c); out["saved"] = str(req["save"])
        if req.get("watch"):                                   # how often a new color picture arrives, in `watch` seconds
            seen = set(); t0 = time.monotonic()
            while time.monotonic() - t0 < float(req["watch"]):
                cc = src.colour(); seen.add(hash(cc) if cc else 0); import PySide6.QtTest as QT; QT.QTest.qWait(50)
            out["distinct_colour_frames"] = len(seen)
        if c:
            import numpy as np, cv2
            im = cv2.imdecode(np.frombuffer(c, np.uint8), cv2.IMREAD_COLOR)
            out["colour_size"] = None if im is None else [int(im.shape[1]), int(im.shape[0])]
            if im is not None: out["colour_mean_bgr"] = [round(float(x), 1) for x in im.reshape(-1, 3).mean(0)]
        out["gone"] = src.gone_text() if hasattr(src, "gone_text") else ""
        return out

    def cmd_state(self, req):
        import numpy as np
        b = getattr(self.win, "build", None)
        if b is None: return {"build": None}
        lu = (getattr(b, "lineup", None) or {}).get("T") or {}
        poses = {k[-6:]: [round(float(x), 1) for x in np.asarray(v, float)[:3, 3]] for k, v in lu.items()}
        stage = getattr(b, "stage", None); panel = b.panel.get(stage) if hasattr(b, "panel") else None
        return {"stage": stage, "tab": getattr(panel, "current", None), "side_k": getattr(b, "side_k", None), "lineup_moves_mm": poses,
                "points_mode": bool(getattr(b, "_pts", None)), "all_mode": bool(b._all_mode()) if hasattr(b, "_all_mode") else None,
                "view_shown": {k: v for k, v in (getattr(b, "shown", {}) or {}).items() if k in ("stage", "path", "kind", "n")}}

    def cmd_models(self, req):
        import numpy as np
        w = self._pane(str(req.get("pane", "main"))); items = getattr(w, "items", None)
        if items is None: raise ValueError("that pane has no items")
        out = {}
        for k, it in items.items():
            M = it.get("model"); M = np.eye(4) if M is None else np.asarray(M, float)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1))))
            out[str(k)[-6:]] = {"move_mm": [round(float(x), 1) for x in M[:3, 3]], "turn_deg": round(ang, 1), "visible": it.get("visible", True)}
        return {"items": out, "only": sorted(str(x)[-6:] for x in (getattr(w, "only", None) or []))}

    def cmd_screenshot(self, req):
        target = req.get("pane") or req.get("id")
        w = (QApplication.activePopupWidget() or self.win) if target == "popup" else (self._pane(str(target)) if target else self.win)
        img = w.grabFramebuffer() if hasattr(w, "grabFramebuffer") else w.grab().toImage()
        mw = int(req.get("max_width", 900))
        if img.width() > mw: img = img.scaledToWidth(mw, Qt.TransformationMode.SmoothTransformation)
        path = req.get("path") or os.path.join(SHOTS, "shot-%s.jpg" % time.strftime("%H%M%S"))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        img.save(path, "PNG" if path.lower().endswith(".png") else "JPG", -1 if path.lower().endswith(".png") else 72); return {"path": path, "size": [img.width(), img.height()], "bytes": os.path.getsize(path)}
