"""Color export, point transfer, live pairing, and UI regressions without devices."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['POINTYOINK_NO_DEVICE']='1'
import zipfile,xml.etree.ElementTree as ET
import numpy as np
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scanner"))   # "from stages import ..." below
from stages import plyio,color
from app2.builder import export_model


def colored(path):
    V=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32);F=np.array([[0,1,2]])
    C=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
    plyio.write_mesh(str(path),V,F,colors=C);return V,F,C


def test_3mf_has_exact_triangle_corner_rgb_and_prusa_palette(tmp_path):
    src=tmp_path/'src.ply';V,F,C=colored(src);out=tmp_path/'out.3mf'
    export_model.write_model(str(src),str(out),'3mf')
    with zipfile.ZipFile(out) as z:
        root=ET.fromstring(z.read('3D/3dmodel.model'))
        ns={'m':'http://schemas.microsoft.com/3dmanufacturing/material/2015/02','c':'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
        colors=[x.attrib['color'] for x in root.findall('.//m:color',ns)]
        tri=root.find('.//c:triangle',ns)
        assert [colors[int(tri.attrib['p'+str(k)])] for k in [1,2,3]]==['#FF0000','#00FF00','#0000FF']
        assert tri.attrib['{http://schemas.slic3r.org/3mf/2017/06}mmu_segmentation']=='4'
        assert '#555555' in z.read('Metadata/Slic3r_PE.config').decode()


def test_oriented_export_keeps_colors(tmp_path):
    src=tmp_path/'src.ply';_,_,C=colored(src)
    target=export_model.orient_ply(str(src),np.array([[0,-1,0],[1,0,0],[0,0,1]]))
    try:np.testing.assert_array_equal(color.colors(target),C)
    finally:os.unlink(target)


def test_point_to_mesh_and_point_tools_color_transfer(tmp_path):
    src=str(tmp_path/'src.ply');V,_,C=colored(src);dest=str(tmp_path/'points.npz')
    plyio.save_points(dest,V,np.tile([0,0,1],(3,1)),np.ones(3),np.ones(3),np.zeros(3,np.uint64),[],.4)
    color.carry_job({'source':src,'target':dest});assert color.has_colors(dest)
    np.testing.assert_array_equal(color.colors(dest),C)
    mesh=str(tmp_path/'mesh.ply');colored(mesh)
    color.carry_job({'source':dest,'target':mesh});np.testing.assert_array_equal(color.colors(mesh),C)


def test_live_source_nearest_host_timestamp_and_stale_rejection():
    from app2.scan.sources import FrameSource
    s=FrameSource();s.color_arrived(b'first',10.,'f1');s.color_arrived(b'next',10.06,'f2')
    packet=s.color_sample(10.05)
    assert packet['jpeg']==b'next' and packet['frame_id']=='f2'
    assert abs(packet['t']-10.05)<.011
    assert s.color_sample(10.161) is None


def test_writer_is_atomic_and_flushes(tmp_path):
    from app2.scan.color_writer import ColorWriter
    w=ColorWriter();path=str(tmp_path/'frame_000_0001.img')
    assert w.submit(path,b'jpeg');w.flush();w.close()
    assert (tmp_path/'frame_000_0001.img').read_bytes()==b'jpeg'
    assert w.written==1 and w.bytes==4 and not list(tmp_path.glob('*.tmp*'))


def test_export_sheet_stl_color_notice_and_fused_visible_control():
    from PySide6.QtWidgets import QApplication,QWidget
    from app2.qt.export_sheet import ExportDialog
    from app2.qt.build_words import PANELS
    app=QApplication.instance() or QApplication([]);parent=QWidget();d=ExportDialog(parent)
    d.set_has_color(True);d.pick('stl');assert 'does not save scan color' in d.hint.text()
    controls=PANELS['fused']['tabs'][0]['controls']
    assert any(c.get('path')=='fused.color' and not c.get('advanced',False) for c in controls)


def test_colored_geometry_uses_color_digest_in_downstream_cache(tmp_path):
    from types import SimpleNamespace as NS
    from stages.engine import Engine
    path=tmp_path/'mesh.ply';colored(path)
    engine=Engine.__new__(Engine);step=NS(id='fused',produces='mesh')
    meta={'_dir':str(tmp_path),'outputs':{'mesh':{'file':'mesh.ply'}},'digest':'colored-bytes','geo_digest':'same-geometry'}
    assert engine._out_meta(step,meta,{}).digest=='colored-bytes'
    V,F,_=colored(path);plyio.write_mesh(str(path),V,F)
    assert engine._out_meta(step,meta,{}).digest=='same-geometry'


def test_color_home_thumbnail_contains_surface_rgb():
    from app2.builder.color_preview import render
    import trimesh
    m=trimesh.creation.box();C=np.tile([255,0,0],(len(m.vertices),1))
    image=np.asarray(render(m.vertices,m.faces,C))
    assert np.any((image[:,:,0]>80)&(image[:,:,1]==0)&(image[:,:,2]==0))


def test_build_color_menu_offers_scan_color_but_defaults_to_shaded(tmp_path):
    from types import SimpleNamespace as NS
    from app2.qt.build_step import BuildStepPage
    src=tmp_path/'mesh.ply';colored(src);seen=[]
    page=NS(shown={'kind':'mesh'},stage='fused',gl=True,colour_mode={},_display_path=lambda:str(src),vp=NS(set_colours=lambda choices,current:seen.append((choices,current))))
    BuildStepPage._refresh_colours(page)
    assert seen[-1][1]=='shaded' and ('scan_color','Scan color') in seen[-1][0]      # 2026-10-24: shaded by default, color one click away
    page.colour_mode['fused']='scan_color';BuildStepPage._refresh_colours(page)
    assert seen[-1][1]=='scan_color'


def test_worker_backpressure_drops_instead_of_waiting_for_disk(tmp_path,monkeypatch):
    import threading,time
    from app2.scan import color_writer
    entered=threading.Event();release=threading.Event();real=color_writer.os.replace
    def slow(a,b):entered.set();release.wait(3);real(a,b)
    monkeypatch.setattr(color_writer.os,'replace',slow)
    w=color_writer.ColorWriter(limit=1)
    try:
        assert w.submit(str(tmp_path/'first.img'),b'first');assert entered.wait(2)
        assert w.submit(str(tmp_path/'second.img'),b'second')
        start=time.perf_counter();assert not w.submit(str(tmp_path/'third.img'),b'third')
        assert time.perf_counter()-start<.1 and w.dropped==1
    finally:release.set();w.flush();w.close()
    assert w.written==2 and not w.thread.is_alive()


def test_capture_save_color_setting_defaults_on_and_emits():
    from PySide6.QtWidgets import QApplication
    from app2.qt.widgets import Controls
    app=QApplication.instance() or QApplication([]);widget=Controls();seen=[]
    widget.save_color_toggled.connect(seen.append)
    assert widget.save_color.isChecked() and '770 MB' in widget.save_color.toolTip()
    widget.save_color.setChecked(False);assert seen==[False]


def test_smoothing_retains_exact_colors_when_vertex_indices_stay(tmp_path):
    src=str(tmp_path/'source.ply');V,F,C=colored(src);dest=str(tmp_path/'smooth.ply')
    plyio.write_mesh(dest,V+np.array([.1,0,.1]),F)
    color.carry_job({'source':src,'target':dest});np.testing.assert_array_equal(color.colors(dest),C)


def test_new_colored_model_refreshes_an_old_home_preview(tmp_path):
    from app2.project_preview import preview_bytes,PREVIEW_NAME
    from PIL import Image
    import io,time
    cache=tmp_path/PREVIEW_NAME;Image.new('RGB',(560,320),(0,255,0)).save(cache)
    os.utime(cache,ns=(1,1));src=tmp_path/'model.ply';colored(src)
    image=np.asarray(Image.open(io.BytesIO(preview_bytes({'path':str(tmp_path),'model':str(src),'pictures':[]}))))
    assert not np.all(image==[0,255,0])
    assert cache.stat().st_mtime_ns>=src.stat().st_mtime_ns
