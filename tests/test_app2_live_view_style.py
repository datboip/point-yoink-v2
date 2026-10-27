"""Surface or Dots for the live model; Surface shows nothing, not dots, while the first surface is on its way."""
import os,tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp())/'config.json')
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np


def _cloud(**extra):
    P=np.array([[0.,0,0],[10,0,0],[0,10,0]],np.float32);N=np.tile([0.,0,1.],(3,1)).astype(np.float32)
    return dict(side_no=1,home_seq=1,P=P,N=N,L=np.ones(3,np.uint8),B=np.full(3,255,np.uint8),covered=.5,pivot=np.zeros(3,np.float32),bed_up=None,**extra)


def test_surface_hides_dots_until_the_mesh_and_dots_shows_them():
    from PySide6.QtWidgets import QApplication
    from app2.qt.model_panel import ModelPanel
    app=QApplication.instance() or QApplication([])
    panel=ModelPanel(gl=True);calls=[]
    panel.view.set_points=lambda *a,**k:calls.append(('points',a[0] is not None))
    panel.view.set_mesh=lambda *a,**k:calls.append(('mesh',a[0] is not None))
    panel.set_view_style('Surface');panel.on_cloud(_cloud(surface_soon=True))
    assert ('points',True) not in calls and ('mesh',True) not in calls          # the first frames: nothing flashes
    mesh=_cloud(V=np.zeros((3,3),np.float32),VN=np.tile([0.,0,1.],(3,1)).astype(np.float32),F=np.array([[0,1,2]],np.int32),VL=np.ones(3,np.uint8),VB=np.full(3,255,np.uint8))
    calls.clear();panel.on_cloud(mesh);assert ('mesh',True) in calls
    calls.clear();panel.set_view_style('Dots');assert ('points',True) in calls and ('mesh',True) not in calls   # the last cloud, redrawn as dots
    calls.clear();panel.on_cloud(_cloud(surface_soon=True));assert ('points',True) in calls                     # Dots never waits
