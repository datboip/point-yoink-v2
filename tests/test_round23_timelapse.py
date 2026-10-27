"""Camera motion during growth, fixed framing, and true plane margins."""
import numpy as np


def test_orbit_has_constant_aim_and_distance_at_every_angle():
    from app2.scan.timelapse import ShotCamera
    points=np.array([[x,y,z] for x in (-110,110) for y in (-55,55) for z in (0,55)],float)
    cam=ShotCamera(points,[0,0,1],[1,0,0],16/9)
    distances=[];centers=[]
    for angle in np.linspace(0,360,97):
        distance,center=cam._fit(angle);distances.append(distance);centers.append(center)
        view=cam.view(angle);proj=cam.proj(view);eye=points@view[:3,:3].T+view[:3,3]
        assert np.max(np.abs(proj[0,0]*eye[:,0]/-eye[:,2]))<=1
        assert np.max(np.abs(proj[1,1]*eye[:,1]/-eye[:,2]))<=1
        x=proj[0,0]*eye[:,0]/-eye[:,2]-proj[0,2];y=proj[1,1]*eye[:,1]/-eye[:,2]-proj[1,2]
        assert abs(x.max()+x.min())<1e-12 and abs(y.max()+y.min())<1e-12
    assert np.ptp(distances)==0 and np.allclose(centers,centers[0])


def test_side_cut_honors_its_saved_margin():
    from app2.scan.timelapse import SidePlay
    side=SidePlay.__new__(SidePlay);side.plane=(np.array([0,0,1]),0.,True);side.margin=4.
    points=np.array([[0,0,2],[0,0,5]],np.float32);normals=np.ones_like(points);levels=np.array([2,2],np.uint8)
    kept=side.cut(points,normals,levels)
    assert len(kept[0])==1 and kept[0][0,2]==5


def test_plain_cpu_shading_has_no_coverage_palette():
    from app2.scan.timelapse import CPURenderer,ShotCamera,BACKGROUND
    points=np.array([[x,y,0] for x in range(-15,16,3) for y in range(-15,16,3)],np.float32)
    normals=np.tile([0,0,1],(len(points),1)).astype(np.float32)
    cam=ShotCamera(points,[0,0,1],[1,0,0],16/9);view=cam.view(0)
    image=CPURenderer().render(points,normals,np.arange(len(points),dtype=np.uint8)%3,view,cam.proj(view),320,180)
    foreground=np.any(image!=BACKGROUND,axis=2)
    assert foreground.any()
    assert np.all(image[foreground,0]==image[foreground,1]) and np.all(image[foreground,1]==image[foreground,2])


def test_sheet_defaults_to_one_turn_until_the_user_edits(tmp_path):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QWidget
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from app2.qt.timelapse_sheet import TimelapseSheet
    app=QApplication.instance() or QApplication([]);host=QWidget();host.resize(1280,800);host.show()
    sheet=TimelapseSheet(host)
    try:
        sheet.present(str(tmp_path),1,formats=('mp4','gif'))
        sheet.pick('length',15);assert float(sheet.turns.text())==1
        sheet.turns.selectAll();QTest.keyClicks(sheet.turns,'1.25')
        sheet.pick('length',30);assert float(sheet.turns.text())==1.25
        args=sheet.args();assert args[args.index('--turns')+1]=='1.25'
        app.processEvents();assert host.rect().contains(sheet.geometry())
        assert sheet.rect().contains(sheet.ok_b.geometry())
        sheet.turns.setText('-1');sheet.run()
        assert not sheet.busy and 'nonnegative' in sheet.status.text()
        sheet.turns.setText('nan');sheet.run();assert not sheet.busy
    finally:host.close();host.deleteLater();app.processEvents()
