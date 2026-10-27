# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short, click-to-read help in a child card inside the main window."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton,QFrame,QVBoxLayout,QLabel

HELP = {
    'Loose piece size': 'Removes floating bits: connected pieces smaller than this share of the biggest. Raise it if junk is left; lower it if real parts vanish. It can mark inner cavity surfaces too, so check the red before Delete.',
    'Perspective': 'Shows depth like your eye does: near parts look bigger. Only changes the view, not the model or the export.',
    'Orthographic': 'A flat view with no depth: parallel edges stay parallel, handy for lining up a side view. Only changes the view, not the model or the export.',
    'Orbit': 'Drag to turn the view around the part. Scroll to zoom. Only changes the view, not the model.',
    'Quality': 'How much fine detail the Watertight surface keeps. Raise it if small details look melted; lower it if building is slow or runs out of memory.',
    'Close small holes': 'Patches small gaps next to the scanned surface. Turn it off if it closes openings that belong in the part.',
    'Target triangles': 'How many triangles the print file keeps. Lower it for a smaller file; raise it if curves look flat or small details are lost.',
    'Smoothness': 'Softens bumpy scan noise. Raise it if the surface looks rough; lower it if sharp edges get rounded. 0 leaves the surface alone.',
    'Fill holes up to': 'Closes holes up to about this size (the mm shown is an estimate). Raise it if holes are left in the print; 0 leaves every hole open.',
    'Detail level': 'How closely the closed surface follows the scan (also called Poisson depth). Start at 9. Raise it if detail is missing; lower it if you run out of memory.',
    'Keep surface within (mm)': 'Cuts away new surface that strays farther than this from the scan. Lower it if blobs appear where nothing was scanned; raise it if holes reopen.',
    'Thickness (mm)': 'How thick the new fin is. Raise it for a stronger fin; keep it above the thinnest wall your printer can make.',
    'Reach (mm)': 'Add fin: how far the fin sticks out from its two spots. Smooth area: how wide a bump gets flattened. Raise it for a longer fin or to smooth wider bumps.',
    'Gap width (mm)': 'How wide the slot is. The red box shows what gets cut. Raise it if the gap is too narrow to separate the pieces.',
    'Cut depth (mm)': 'How far the slot goes into the part. Raise it if the join is not cut through; too deep can split the part in two.',
    'Shape': 'The shape of the tool. Add fin: Triangular is a straight rib, Rounded a curved one. Cut, clone and smooth tools: Cylinder covers a round area, Box a square one, Sphere a ball.',
    'Surface': 'How the final model is made. Auto picks for you. As scanned keeps the surface from Fused. Watertight wraps it in one closed skin, best for printing. Joined points links the points as they are, so holes stay open. Pick Watertight if the print has holes.',
    'Blend method': 'How the depth frames are combined into one shape. Standard is fast and fine for most parts. Points keeps sharper edges but is slower. Plus gives a sharper surface and drops surface hidden inside. Try Points or Plus if edges look soft.',
    'Point spacing (mm)': 'How close together the model points are. Smaller keeps more detail but takes longer and uses more memory. Lower it for small, detailed parts; raise it for big parts or a quick look.',
    'Merge distance (mm)': 'Where two passes left a doubled layer closer than this, one layer is kept. Raise it if walls look doubled; lower it if thin walls merge. Only with the Points blend method.',
    'Capture': 'How the scanner records.\n\nContinuous: records until you press Pause. Use this most of the time.\n\nSingle shot: grabs about 10 frames, then stops. Press again for 5 more, up to 25. Good for one tricky spot.\n\nBurst: grabs 25 frames, then stops.\n\nSmart frames: like Continuous, but skips frames that add nothing new. Saves disk space.\n\nLive build: builds the model while you scan and keeps only frames that help. Undo recent takes back the last few seconds. Uses more memory.',
    'Range': 'Only depth between these two distances is kept, so the background is left out. Auto keeps it wrapped around the part. Dragging a handle turns Auto off.',
    'IR gain': 'How bright the depth camera sees. Raise it if a dark part shows holes. Lower it if a shiny part glares. Auto picks it for you while you scan.',
    'Add fin': 'Adds a thin rib between two spots to make a weak area stronger. Pick the spots, check the red preview, then press Apply. The model must be closed first (Watertight).',
    'Cut gap': 'Cuts a slot through an unwanted join, like two parts that grew together. Pick both ends, check the red slot, then press Apply. The cut faces are closed, so it still prints.',
    'Grow': 'Adds one ring of touching faces around the selection. Press it again to grow more.',
    'Shrink': 'Takes one ring of faces off the edge of the selection.',
    'Magic': 'Click the surface to select nearby faces that point about the same way. Lower Magic angle to stop sooner at bends and edges.',
    'Brush': 'Paint over faces to select them. Lower Brush radius to work on small areas.',
    'Rectangle': 'Drag a box over what to select (R). Hold Ctrl to take faces away. Turn on Through to also select the back.',
    'Lasso': 'Draw a loop around what to select (L). Hold Ctrl to take faces away.',
    'Connected piece': 'Selects every face joined to the selection. A thin bridge can join junk to the main part, so check the red before Delete.',
    'Fill LED': 'The scanner\'s white light. Use it on dark parts or in a dim room. After 5 minutes it asks if you still need it: Yes keeps it on 2 more minutes, and no answer in 30 seconds turns it off.',
    'YOLO Yoink': 'Builds the whole model in one go. Simple: pick a preset and press Yoink it. Advanced: pick Custom to change and save settings. Both show a summary before building.',
    'Motion': 'Reads the scanner\'s motion sensor over WiFi. Optional: scanning works without it. The pictures still come over USB.',
}


def help_for(control):
    label=control.get('label','Setting')
    text=HELP.get(label) or control.get('hint_plain') or control.get('hint')
    if text and '{' not in text: return text
    return 'Choose %s, then press Apply to use the change.' % label.lower()


class HelpButton(QPushButton):
    def __init__(self,title,text=None,parent=None):
        super().__init__('?',parent); self.title=title; self.help_text=text or HELP.get(title,'')
        self.setFixedSize(20,20); self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName('Help: '+title);self.setToolTip('Help: '+title)
        self.setStyleSheet('QPushButton {border:1px solid #59616d;border-radius:10px;padding:0;color:#b9c4d2;background:transparent;}')
        self.clicked.connect(self.open_help)

    def open_help(self):
        window=self.window(); card=getattr(window,'_help_card',None)
        if card is None:
            card=QFrame(window);card.setObjectName('HelpCard');card.setFixedSize(340,154)
            card.setStyleSheet('#HelpCard {background:#252b33;border:1px solid #59616d;border-radius:8px;}')
            layout=QVBoxLayout(card);layout.setContentsMargins(12,10,12,10)
            card.title=QLabel();card.title.setFixedHeight(22);layout.addWidget(card.title)
            card.text=QLabel();card.text.setWordWrap(True);card.text.setFixedHeight(72);layout.addWidget(card.text)
            close=QPushButton('Close');close.setFixedSize(76,26);close.clicked.connect(card.hide);layout.addWidget(close,0,Qt.AlignmentFlag.AlignRight)
            window._help_card=card
        if card.isVisible() and card.title.text()==self.title and card.text.text()==self.help_text: card.hide(); return      # its own ? again closes it
        card.title.setText(self.title);card.text.setText(self.help_text)
        h=max(72,card.text.heightForWidth(316));card.text.setFixedHeight(h);card.setFixedHeight(h+82)      # the card grows to fit its text
        point=self.mapTo(window,self.rect().bottomLeft())
        card.move(max(8,min(point.x(),window.width()-card.width()-8)),max(8,min(point.y()+4,window.height()-card.height()-8)))
        card.show();card.raise_()
