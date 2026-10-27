import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication,QWidget

@pytest.fixture
def app():return QApplication.instance() or QApplication([])

def test_new_surface_fades_without_relighting_old_geometry():
    from app2.scan.timelapse import SurfaceAge,GROWTH_PALETTE
    p=np.array([[0.,0,0],[10,0,0]])
    a=SurfaceAge(p,1,8)
    assert a.levels(p[:1],0)[0]==len(GROWTH_PALETTE)-1
    levels=a.levels(p,8)
    assert levels[0]==0 and levels[1]==len(GROWTH_PALETTE)-1
    assert a.levels(p,16).tolist()==[0,0]

def test_default_timelapse_is_one_turn_for_all_lengths(app,tmp_path):
    from app2.qt.timelapse_sheet import TimelapseSheet
    host=QWidget();host.resize(1280,800);s=TimelapseSheet(host)
    try:
        s.present(str(tmp_path),1,formats=("mp4","gif"))
        assert float(s.turns.text())==1
        s.pick('length',30);assert float(s.turns.text())==1
    finally:s.close();host.close()
