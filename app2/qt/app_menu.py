# SPDX-License-Identifier: AGPL-3.0-or-later
"""About card stays inside the existing window; main menus use the Qt menu bar."""
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QWidget, QVBoxLayout, QLabel, QPushButton
from app2.qt.startup import ICON
from app2.version import APP_TITLE

REPO_URL='https://github.com/datboip/point-yoink-v2'


class AboutCard(QWidget):
    dismissed = Signal()

    def __init__(self,parent):
        super().__init__(parent)
        self.setStyleSheet('#AboutPanel {background:#20252c;border:1px solid #3b424b;border-radius:8px;}')
        self.panel=QFrame(self); self.panel.setObjectName('AboutPanel'); self.panel.setFixedSize(420,360)
        layout=QVBoxLayout(self.panel); layout.setContentsMargins(20,20,20,20); layout.setSpacing(12)
        self.logo=QLabel(); self.logo.setFixedSize(96,96)
        self.logo.setPixmap(QPixmap(str(ICON)).scaled(96,96,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(self.logo,0,Qt.AlignmentFlag.AlignHCenter)
        self.version=QLabel(APP_TITLE); self.version.setFixedHeight(32); self.version.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(self.version)
        tagline=QLabel('YOINK. CLEAN. KEEP.'); tagline.setFixedHeight(24); tagline.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(tagline)
        details=QLabel('Scan, build, clean up and export 3D models. AGPL-3.0.'); details.setFixedHeight(24); details.setToolTip('GNU Affero General Public License v3: the LICENSE file in the repository has the terms.'); details.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(details)
        self.link=QLabel('<a href="%s" style="color:#67c5e8;">GitHub: %s</a>' % (REPO_URL, REPO_URL.replace('https://',''))); self.link.setFixedHeight(24)
        self.link.setAlignment(Qt.AlignmentFlag.AlignCenter); self.link.setOpenExternalLinks(True); self.link.setToolTip('The code, releases and issue tracker'); layout.addWidget(self.link)
        layout.addStretch()
        self.done=QPushButton('Close'); self.done.setFixedHeight(32); self.done.clicked.connect(self.dismiss); layout.addWidget(self.done)
        parent.installEventFilter(self); self.hide()

    def relayout(self):
        """Most of the window, so the guide is read, not scrolled through a letterbox."""
        r=self.parentWidget().rect(); self.setGeometry(r)
        self.panel.setFixedSize(max(640,min(1000,int(r.width()*0.7))),max(520,min(860,int(r.height()*0.85))))
        self.panel.move((self.width()-self.panel.width())//2,(self.height()-self.panel.height())//2)

    def relayout(self):
        self.setGeometry(self.parentWidget().rect())
        self.panel.move((self.width()-self.panel.width())//2,(self.height()-self.panel.height())//2)

    def eventFilter(self,obj,event):
        if obj is self.parentWidget() and event.type()==QEvent.Type.Resize: self.relayout()
        return super().eventFilter(obj,event)

    def showEvent(self,event):
        self.relayout(); super().showEvent(event)

    def mousePressEvent(self,event):
        if not self.panel.geometry().contains(event.position().toPoint()): self.dismiss()
        else: super().mousePressEvent(event)

    def dismiss(self):
        self.hide(); self.dismissed.emit()


class GuideCard(AboutCard):
    """Local, scrollable instructions; no automatic browser or network access."""
    def __init__(self,parent):
        from PySide6.QtWidgets import QTextBrowser
        QWidget.__init__(self,parent)
        self.setStyleSheet('#GuidePanel {background:#20252c;border:1px solid #3b424b;border-radius:8px;}')
        self.panel=QFrame(self); self.panel.setObjectName('GuidePanel'); self.panel.setFixedSize(640,520)    # relayout grows it with the window
        layout=QVBoxLayout(self.panel); layout.setContentsMargins(20,20,20,20); layout.setSpacing(12)
        title=QLabel('Getting started'); title.setFixedHeight(26); layout.addWidget(title)
        self.text=QTextBrowser(); self.text.setOpenExternalLinks(False); self.text.setOpenLinks(False)
        self.text.setStyleSheet('QTextBrowser {background:#15191f;color:#e6edf6;border:0;padding:12px;font-size:14px;}')
        self.text.document().setDefaultStyleSheet('h1 {font-size:22px;margin-bottom:6px;} h2 {font-size:18px;color:#9fd8ee;margin-top:22px;margin-bottom:6px;} p, li {line-height:140%;margin-top:4px;margin-bottom:8px;}')
        path=ICON.parent/'app2'/'GETTING_STARTED.md'
        try: content=path.read_text()
        except OSError: content='The guide could not be read. Check app2/GETTING_STARTED.md.'
        self.text.setMarkdown(content); layout.addWidget(self.text,1)
        self.done=QPushButton('Close'); self.done.setFixedHeight(32); self.done.clicked.connect(self.dismiss); layout.addWidget(self.done)
        parent.installEventFilter(self); self.hide()

    def relayout(self):
        """Most of the window, so the guide is read, not scrolled through a letterbox."""
        r=self.parentWidget().rect(); self.setGeometry(r)
        self.panel.setFixedSize(max(640,min(1000,int(r.width()*0.7))),max(520,min(860,int(r.height()*0.85))))
        self.panel.move((self.width()-self.panel.width())//2,(self.height()-self.panel.height())//2)
