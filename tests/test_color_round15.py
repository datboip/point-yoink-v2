"""Calibration, surface interpolation, and real color TSDF smoke tests."""
import json,struct,sys
from pathlib import Path
import numpy as np
from scanner import color_frames
from scanner.stages import color,plyio
import fuse


def test_full_rgb_size_and_half_depth_scaling_project_same_surface():
    K=np.array([[600.,0,400],[0,600,300],[0,0,1]]);R=np.eye(3);T=np.array([20.,0,0])
    uv,z=color_frames.project([[0.,0,200.]],(K,R,T),(2000,1500))
    np.testing.assert_allclose(uv,[[1150,750]])
    # A depth pixel representing the same ray scales its coordinate and K together.
    for factor in [1.,.5]:
        fx,cx,u=1000*factor,400*factor,500*factor
        X=(u-cx)*200/fx
        point,_=color_frames.project([[X,0,200]],(K,R,T),(2000,1500))
        np.testing.assert_allclose(point,[[1300,750]])


def test_color_transfer_interpolates_triangle_surface():
    V=np.array([[0,0,0],[1,0,0],[0,1,0]],float);F=np.array([[0,1,2]])
    C=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
    actual=color.transfer(V,F,C,np.array([[.25,.25,.1],[0,0,0]]))
    np.testing.assert_array_equal(actual,[[128,64,64],[255,0,0]])


def test_mesh_ordering_keeps_colors_attached(tmp_path):
    import open3d as o3d
    V=np.array([[1,0,0],[0,1,0],[0,0,0]],float);F=np.array([[0,1,2]])
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    mesh.vertex_colors=o3d.utility.Vector3dVector(np.eye(3));result=fuse._canonical_mesh(mesh)
    by_point={tuple(v):tuple(c) for v,c in zip(np.asarray(result.vertices),np.asarray(result.vertex_colors))}
    assert by_point[(1.,0.,0.)]==(1.,0.,0.) and by_point[(0.,0.,0.)]==(0.,0.,1.)


def test_cpu_tsdf_integrates_color_and_off_stays_gray(tmp_path,monkeypatch):
    from PIL import Image
    import open3d as o3d
    side=tmp_path/'side';cache=side/'cache';param=side/'param';cache.mkdir(parents=True);param.mkdir()
    kd=np.array([[800.,0,400],[0,800,400],[0,0,1]],np.float32)
    kr=np.array([[800.,0,400],[0,600,300],[0,0,1]],np.float32)
    for name,K in [('Pl.bin',kd),('Prgb.bin',kr)]: (param/name).write_bytes(struct.pack('<10f',0,*K.ravel()))
    (param/'LC_RT.bin').write_bytes(struct.pack('<12f',*np.eye(3).ravel(),0,0,0))
    for index in range(3):
        base=cache/f'frame_000_{index:04d}'
        base.with_suffix('.dph').write_bytes(np.full((4,4),1000,np.uint16).tobytes())
        base.with_suffix('.inf').write_bytes(b'\0'*16+struct.pack('<16d',*np.eye(4).ravel()))
        Image.new('RGB',(4,4),(220,30,10)).save(base.with_suffix('.img'),format='JPEG')
    results=[]
    for mode in ['on','off']:
        output=tmp_path/(mode+'.ply')
        monkeypatch.setattr(sys,'argv',['fuse.py','--frames',str(cache),'--calib',str(param/'Pl.bin'),'--out',str(output),'--width','4','--height','4','--voxel','2','--filter','0','--poses','none','--color',mode])
        monkeypatch.setattr(fuse,'apply_mem_cap',lambda:None)
        assert fuse.main()==0
        mesh=o3d.io.read_triangle_mesh(str(output));assert len(mesh.triangles)
        results.append(mesh)
    assert results[0].has_vertex_colors() and not results[1].has_vertex_colors()
    np.testing.assert_array_equal(np.asarray(results[0].vertices),np.asarray(results[1].vertices))
    assert np.median(np.asarray(results[0].vertex_colors)[:,0])>.7


def test_exposure_balance_is_bounded_and_improves_dark_second_frame(tmp_path,monkeypatch):
    import open3d as o3d
    x,y=np.meshgrid(np.linspace(-40,40,21),np.linspace(-40,40,21))
    V=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-100.)))
    K=np.array([[800.,0,400.],[0,600.,300.],[0,0,1.]])
    monkeypatch.setattr(color_frames,'calibration',lambda _: (K,np.eye(3),np.zeros(3)))
    monkeypatch.setattr(color_frames,'read_rgb',lambda f:np.full((32,32,3),200 if str(f).endswith('0000.dph') else 120,np.uint8))
    jobs=[]
    for k in range(2):
        path=tmp_path/('frame_000_%04d.dph'%k);np.full((32,32),1000,np.uint16).tofile(path)
        jobs.append((str(path),{fuse.frame_key(str(path)):np.eye(4)},None,(32.,32.,16.,16.),None,(32,32),True,None))
    medians=[]
    for balance in [False,True]:
        mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(V);mesh.vertex_normals=o3d.utility.Vector3dVector(np.tile([0,0,-1],(len(V),1)))
        stats=color_frames.paint(mesh,jobs,balance=balance)
        assert stats['coverage']==1 and max(stats['gain_max'])<=1.25
        medians.append(np.median(np.asarray(mesh.vertex_colors)))
    assert medians[1]>medians[0]


def test_save_color_unticks_and_stays_unticked():
    """The state carries save_color (it had no slot, so every update ticked the box again)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from app2.scan.ui_model import ScanState
    from app2.qt.widgets import Controls
    app = QApplication.instance() or QApplication([])
    assert ScanState(save_color=False).save_color is False
    c = Controls(); seen = []; c.save_color_toggled.connect(seen.append)
    c.save_color.setChecked(False); app.processEvents()
    assert seen == [False]
