# SPDX-License-Identifier: AGPL-3.0-or-later
"""File > Clean up temporary files: what the app keeps besides your projects, and ways to clear it. The Wi-Fi holding
folder lives inside the project library on purpose (same drive, so a finished transfer is an instant move); anything a
transfer left there can be moved into the library or deleted. Saved scanner info (sizes, names) can be cleared too."""
import os, shutil
from pathlib import Path
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
from app2 import disk_usage
from app2.qt.theme import T


def _config_dir():
    return Path(os.environ.get("POINTYOINK_CONFIG") or Path.home() / ".config/pointyoink/config.json").parent


class CleanupDialog(QDialog):
    def __init__(self, library, parent=None):
        super().__init__(parent); self.setWindowTitle("Clean up temporary files"); self.setMinimumWidth(560)
        self.setStyleSheet("QDialog {background:%s;} QLabel {color:%s; background:transparent;} QLabel[role=\"hint\"] {color:%s;}" % (T["window"], T["ink"], T["ink2"]))
        self.library = Path(library); self.hold = self.library / ".wifi-incoming"
        self.v = QVBoxLayout(self); self.v.setSpacing(12); self.v.setContentsMargins(20, 18, 20, 16)
        self.body = QVBoxLayout(); self.v.addLayout(self.body)
        close = QPushButton("Close"); close.clicked.connect(self.accept); row = QHBoxLayout(); row.addStretch(1); row.addWidget(close); self.v.addLayout(row)
        self.fill()

    def _leftovers(self):
        """Projects a Wi-Fi transfer left in the holding folder: [(project folder, bytes)]."""
        out = []
        if self.hold.is_dir():
            for session in self.hold.iterdir():
                if not session.is_dir(): continue
                for proj in session.iterdir():
                    if proj.is_dir(): out.append((proj, disk_usage.folder_bytes(str(proj))))
        return out

    def _section(self, title, text, buttons):
        box = QFrame(); box.setObjectName("CleanBox"); box.setStyleSheet("QFrame#CleanBox {border:1px solid #3a414b; border-radius:8px;}")
        v = QVBoxLayout(box); v.setContentsMargins(14, 10, 14, 12); v.setSpacing(6)
        t = QLabel(title); t.setStyleSheet("font-weight:600;"); v.addWidget(t)
        d = QLabel(text); d.setWordWrap(True); d.setProperty("role", "hint"); v.addWidget(d)
        if buttons:
            h = QHBoxLayout(); h.setSpacing(8)
            for label, fn, enabled in buttons:
                b = QPushButton(label); b.setEnabled(enabled); b.clicked.connect(fn); h.addWidget(b)
            h.addStretch(1); v.addLayout(h)
        self.body.addWidget(box)

    def fill(self):
        while self.body.count():
            w = self.body.takeAt(0).widget()
            if w: w.deleteLater()
        left = self._leftovers(); total = sum(n for _, n in left)
        names = ", ".join(p.name for p, _ in left[:4]) + (" and %d more" % (len(left) - 4) if len(left) > 4 else "")
        self._section("Wi-Fi holding folder",
                      ("%d project%s left by a transfer (%s): %s." % (len(left), "s" if len(left) != 1 else "", disk_usage.nice(total), names)) if left
                      else "Empty. Wi-Fi transfers land here first, inside your project library, and move into it when they finish.",
                      [("Move into the library", self.move_left, bool(left)), ("Delete leftovers", self.delete_left, bool(left)),
                       ("Open folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.hold if self.hold.is_dir() else self.library))), True)])
        cfg = _config_dir(); files = [cfg / "device-sizes.json", cfg / "device-names.json"]
        size = sum(f.stat().st_size for f in files if f.is_file())
        self._section("Saved scanner info",
                      "Project sizes read off the scanner (so the Import list fills instantly) and the names you gave scanner projects: %s." % disk_usage.nice(size),
                      [("Clear saved sizes", self.clear_sizes, (cfg / "device-sizes.json").is_file())])

    def move_left(self):
        moved = 0
        for proj, _ in self._leftovers():
            target = self.library / proj.name; n = 1
            while os.path.lexists(target): target = self.library / f"{proj.name} ({n})"; n += 1
            try: os.rename(proj, target); moved += 1
            except OSError: shutil.move(str(proj), str(target)); moved += 1
        self._drop_empty(); self.fill()
        if self.parent() and hasattr(self.parent(), "workspace"): self.parent().workspace.refresh_projects()
        QMessageBox.information(self, "Moved", "Moved %d project%s into your library." % (moved, "s" if moved != 1 else ""))

    def delete_left(self):
        left = self._leftovers()
        if QMessageBox.question(self, "Delete leftovers", "Delete %d project%s from the Wi-Fi holding folder (%s)? The scanner still has its own copies."
                                % (len(left), "s" if len(left) != 1 else "", disk_usage.nice(sum(n for _, n in left)))) != QMessageBox.Yes: return
        for proj, _ in left: shutil.rmtree(proj, ignore_errors=True)
        self._drop_empty(); self.fill()

    def _drop_empty(self):
        if self.hold.is_dir():
            for session in self.hold.iterdir():
                if session.is_dir() and not any(session.iterdir()): session.rmdir()

    def clear_sizes(self):
        try: (_config_dir() / "device-sizes.json").unlink()
        except OSError: pass
        self.fill()
