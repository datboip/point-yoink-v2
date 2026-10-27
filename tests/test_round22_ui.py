import os, time, json
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['POINTYOINK_NO_DEVICE'] = '1'
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_scan_distance_choice_is_separate_from_frame_policy(app):
    from app2.qt.widgets import Controls
    c = Controls()
    try:
        assert [c.scan_mode.itemData(i) for i in range(c.scan_mode.count())] == ['near', 'far']
        assert c.capture_mode.findData('smart') >= 0
    finally: c.close()


def test_scan_distance_dispatch_and_saved_state_do_not_change_frame_policy(app,tmp_path):
    from app2.qt.scan_page import ScanPage
    from app2.qt.bridge import SessionBridge
    from test_app2_workflow_fixes import state
    p=ScanPage(SessionBridge(),False);calls=[]
    p.session=SimpleNamespace(set_scan_mode=calls.append)
    try:
        original=state(str(tmp_path),capture_mode='smart');s=SimpleNamespace(**{k:getattr(original,k) for k in original.__slots__})
        s.scan_mode='far';s.picked_has_frames=False
        p.on_state(s)
        assert p.controls.scan_mode.currentData()=='far' and p.controls.capture_mode.currentData()=='smart'
        p.controls.scan_mode.setCurrentIndex(0)
        assert calls==['near']
        s.picked_has_frames=True;p.on_state(s)
        assert not p.controls.scan_mode.isEnabled()
    finally:p.close()


def test_far_range_uses_backend_limits_without_clipping(app,tmp_path):
    from app2.qt.widgets import Controls
    from app2.scan import ui_model
    from test_app2_workflow_fixes import state
    s=state(str(tmp_path));buttons=ui_model.buttons(s);s=SimpleNamespace(**{k:getattr(s,k) for k in s.__slots__})
    c=Controls()
    try:
        defaults=(c.range.lo,c.range.hi)
        s.range_limits=(0,1000);s.near=300;s.far=950;c.set_state(s,buttons)
        assert (c.range.lo,c.range.hi)==s.range_limits and c.range.values==[300,950]
        s.range_limits=None;c.set_state(s,buttons)
        assert (c.range.lo,c.range.hi)==defaults
    finally:c.close()


def test_side_card_keeps_saved_distance_mode_visible(app):
    from app2.qt.widgets import SideCard
    c = SideCard()
    try:
        c.set_info(dict(no=1, name='side', frames=12, lost=0, picked=True, scan_mode='far'), None)
        assert c.scan_mode.text() == 'Large part' and not c.scan_mode.isHidden()
        c.set_info(dict(no=1, name='side', frames=12, lost=0, picked=True), None)
        assert c.scan_mode.text() == 'Object not saved'
    finally: c.close()


def test_wifi_tree_stays_browsable_while_receiving(app, tmp_path):
    from app2.qt.device_projects import DeviceProjects
    p = DeviceProjects(str(tmp_path)); p.mode = 'wifi'; p.proc = SimpleNamespace()
    try:
        p._buttons()
        assert p.files.isEnabled()
        assert not p.copy_b.isEnabled()
    finally: p.proc = None; p.close()


def test_wifi_row_reads_incoming_six_sides_and_received_bytes(app, tmp_path):
    from app2.qt.device_projects import DeviceProjects
    lib = tmp_path/'library'; stage = lib/'.wifi-incoming'/'share'
    project = stage/'Projects'/'ProjectSaved'
    for i in range(6): (project/'data'/str(i)).mkdir(parents=True)
    f = project/'data'/'0'/'preview.png'; f.write_bytes(b'png')
    p = DeviceProjects(str(lib)); p.mode = 'wifi'; p._t0 = time.monotonic(); p.root = str(tmp_path/'usb')
    try:
        p.worker_event(dict(event='received_file', stage=str(stage), name='ProjectSaved', scan='0',
                            relpath='Projects/ProjectSaved/data/0/preview.png', file_bytes=3))
        it = p.files.topLevelItem(0); row = it.data(0, Qt.UserRole+1)
        assert row['path'] == str(project)
        assert it.text(2) == '6'
        assert 'received' in it.text(3) and 'received' in it.text(5)
        assert row['models'] == row['full'] == 3
        assert it.data(3,Qt.ItemDataRole.CheckStateRole) is None
        assert p.files.isColumnHidden(4) and p.files.isColumnHidden(6)
        assert p.files.columnWidth(3)>=p.files.fontMetrics().horizontalAdvance(it.text(3))
        p.worker_event(dict(event='received_file', stage=str(stage), name='ProjectSaved', scan='0',
                            relpath='Projects/ProjectSaved/data/0/preview.png', file_bytes=3))
        assert it.data(0, Qt.UserRole+1)['full'] == 3
    finally: p.close()


def test_repeat_wifi_share_updates_results_without_duplicate_or_pc_edits(app, tmp_path, monkeypatch):
    from app2.qt import device_projects as M
    monkeypatch.setattr(M, 'adopt_project', lambda p: dict(issues=[]))
    lib = tmp_path/'library'; p = M.DeviceProjects(str(lib)); p._t0 = time.monotonic()
    name = 'ProjectSaved'
    def incoming(tag):
        stage = lib/'.wifi-incoming'/tag; src = stage/name
        (src/'data'/'side').mkdir(parents=True)
        (src/(name+'.revo')).write_text(json.dumps(dict(guid="same-fixture-project",nodes=[dict(type=2,guid="side")],model_mesh_count=0)))
        return stage, src
    try:
        stage, src = incoming('first'); (src/'data'/'side'/'preview.png').write_bytes(b'early')
        p._place_received(str(stage)); target = lib/name
        (target/'stages').mkdir(); (target/'stages'/'recipe.json').write_text('my edit')
        (target/'model.ply').write_bytes(b'PC model')
        stage, src = incoming('second')
        (src/'data'/'side'/'fuse_mesh.ply').write_bytes(b'scanner result')
        (src/'data'/'side'/'preview.png').write_bytes(b'finished')
        (src/'model.ply').write_bytes(b'scanner model')
        p._place_received(str(stage))
        assert not (lib/(name+' (1)')).exists()
        assert (target/'data'/'side'/'fuse_mesh.ply').read_bytes() == b'scanner result'
        assert (target/'data'/'side'/'preview.png').read_bytes() == b'finished'
        assert (target/'stages'/'recipe.json').read_text() == 'my edit'
        assert (target/'model.ply').read_bytes() == b'PC model'
    finally: p.close()


def test_color_scan_border_does_not_follow_frame_depth(monkeypatch):
    from app2.scan import preview as P
    K = np.array([[600., 0, 400], [0, 600, 300], [0, 0, 1]])
    monkeypatch.setattr(P, 'RGB_CAL', (K, np.eye(3), np.array([70.,0,0])))
    d = np.zeros((600,800), np.uint16); img = np.zeros((600,800,3), np.uint8)
    a = P.depth_guides(img.copy(), d, (600.,600.,400.,300.), 160, (60,255,60))
    b = P.depth_guides(img.copy(), d, (600.,600.,400.,300.), 430, (60,255,60))
    assert np.array_equal(a,b)


def test_composer_reuses_calibrated_overlay_across_depth_frames(monkeypatch):
    import threading
    from app2.scan import preview as P
    K = np.array([[600.,0,400],[0,600,300],[0,0,1]])
    monkeypatch.setattr(P,'RGB_CAL',(K,np.eye(3),np.array([70.,0,0])))
    c=P.Composer((600.,600.,400.,300.),lambda *a:None);c.stop();c.thread.join(2)
    c.area=None;c.scan_geometry=None;c.zref=None;c.jpeg=None;c.rgb=np.zeros((600,800,3),np.uint8)
    c.set_scan_geometry(300,600,'far'); area=c.area
    a=c.compose(np.full((600,800),3100,np.uint16),np.zeros((600,800),np.uint8),False)
    b=c.compose(np.full((600,800),3100,np.uint16),np.zeros((600,800),np.uint8),False)
    c.set_scan_geometry(300,600,'far')
    assert c.area is area
    # Qt consumes stable vector payloads; the raster carries only depth dots.
    assert np.array_equal(a.colour_guides['box'],b.colour_guides['box'])
    assert np.array_equal(a.colour_guides['aim'],b.colour_guides['aim'])
    assert np.array_equal(a.colour,b.colour)
    c.set_scan_geometry(100,250,'near');assert c.area is not area


def test_live_box_hugs_real_depth_coverage_at_part_distance(monkeypatch):
    import threading
    from app2.scan import preview as P
    K = np.array([[600.,0,400],[0,600,300],[0,0,1]])
    monkeypatch.setattr(P,'RGB_CAL',(K,np.eye(3),np.array([70.,0,0])))
    c=P.Composer((600.,600.,400.,300.),lambda *a:None);c.stop();c.thread.join(2)
    c.area=None;c.scan_geometry=None;c.zref=None;c.jpeg=None;c.rgb=np.zeros((600,800,3),np.uint8);c.cover=[];c.live=None
    c.set_scan_geometry(255,1000,'far')
    d=np.zeros((600,800),np.uint16); d[100:500,200:600]=4300          # depth only in the stereo overlap
    c.compose(d,np.zeros((600,800),np.uint8),False)
    live=c.guide_area(); assert live is not None and live.get('live')
    x=live['outline'][:,0]; env=c.area['outline'][:,0]
    assert x.max()-x.min() < 0.7*(env.max()-env.min())               # narrower than the full-frame envelope
    # drawn at the part's 430 mm: the edge pixel u=200 lands at (200-400)*430/600 + 70 mm, projected
    assert x.min() == pytest.approx(400+600*((200-400)*430/600+70)/430, abs=6)   # edge trim ignores the thinnest 1%
        # the depth picture's middle pixel at 430 mm, the same spot the IR view's + marks
    assert live['center'][0] == pytest.approx(400+600*((399.5-400)*430/600+70)/430, abs=1)


def test_wifi_empty_side_folder_count_and_malformed_index(app,tmp_path):
    from app2.qt.device_projects import DeviceProjects
    lib=tmp_path/'library';stage=lib/'.wifi-incoming'/'share';project=stage/'ProjectSaved'
    (project/'data').mkdir(parents=True);(project/'ProjectSaved.revo').write_text('{partial')
    p=DeviceProjects(str(lib));p.mode='wifi';p._t0=time.monotonic()
    try:
        p.worker_event(dict(event='received_file',stage=str(stage),name='ProjectSaved',scan=None,
                            relpath='ProjectSaved/ProjectSaved.revo',file_bytes=8))
        it=p.files.topLevelItem(0)
        assert it.text(2)=='0' and 'received' in it.text(5)
    finally:p.close()


def test_scan_area_is_calibrated_range_envelope_not_arbitrary_pixels(monkeypatch):
    from app2.scan import preview as P
    K = np.array([[600.,0,400],[0,600,300],[0,0,1]])
    monkeypatch.setattr(P,'RGB_CAL',(K,np.eye(3),np.array([70.,0,0])))
    area = P.scan_area((600.,600.,400.,300.), 300, 600)
    assert area['near'][0,0] == pytest.approx(140)
    assert area['far'][0,0] == pytest.approx(70)
    assert area['center'][1] == pytest.approx(300)
    assert P.scan_area((600.,600.,400.,300.), 600, 300) is None


def test_projected_depth_area_honors_rotation_and_native_resolution(monkeypatch):
    from app2.scan import preview as P
    angle=np.deg2rad(8);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    K=np.array([[550.,0,380],[0,560,290],[0,0,1]]);T=np.array([35.,12.,4.])
    monkeypatch.setattr(P,'RGB_CAL',(K,R,T))
    a=P.scan_area((600.,600.,400.,300.),300,600)
    b=P.scan_area((300.,300.,200.,150.),300,600,depth_size=(400,300))
    point=np.array([-200.,-150.,300.]);rgb_point=R@point+T;q=K@rgb_point
    assert np.allclose(a['near'][0],q[:2]/q[2])
    assert np.allclose(a['near'][0],b['near'][0])
    assert np.allclose(a['center'],b['center'])


def test_calibrated_crosshair_uses_image_rect_inside_letterboxed_tile(app,monkeypatch):
    from app2.scan import preview as P
    from app2.qt.widgets import ImageTile
    from PySide6.QtCore import QRectF
    K=np.array([[600.,0,400],[0,600,300],[0,0,1]])
    monkeypatch.setattr(P,'RGB_CAL',(K,np.eye(3),np.array([70.,0,0])))
    area=P.scan_area((600.,600.,400.,300.),300,600)
    tile=ImageTile();tile.resize(1200,600);tile.set_image(np.zeros((600,800,3),np.uint8))
    try:
        rect=tile.image_rect(QRectF(tile.rect()))
        x=rect.left()+area['center'][0]*rect.width()/800
        assert rect.left()==pytest.approx(200)
        assert x>tile.width()/2 and x<rect.right()
    finally:tile.close()


def test_incomplete_repeat_share_keeps_original_and_retry_files(tmp_path):
    from app2.transfer.incoming import install_received
    src=tmp_path/'incoming'/'ProjectSaved';target=tmp_path/'library'/'ProjectSaved'
    (src/'data'/'side').mkdir(parents=True);(target/'data'/'side').mkdir(parents=True)
    (src/'data'/'side'/'fuse_mesh.ply.part').write_bytes(b'half')
    (target/'data'/'side'/'preview.png').write_bytes(b'old')
    with pytest.raises(ValueError,match='incomplete'):install_received(src,target)
    assert (target/'data'/'side'/'preview.png').read_bytes()==b'old' and src.exists()


def test_malformed_repeat_index_does_not_overwrite_a_valid_copy(tmp_path):
    from app2.transfer.incoming import install_received
    src=tmp_path/'incoming'/'ProjectSaved';target=tmp_path/'library'/'ProjectSaved'
    (src/'data'/'side').mkdir(parents=True);(target/'data'/'side').mkdir(parents=True)
    (src/'ProjectSaved.revo').write_text('{partial')
    (src/'data'/'side'/'preview.png').write_bytes(b'new')
    (target/'data'/'side'/'preview.png').write_bytes(b'old')
    with pytest.raises(ValueError,match='index is incomplete'):install_received(src,target)
    assert (target/'data'/'side'/'preview.png').read_bytes()==b'old' and src.exists()
