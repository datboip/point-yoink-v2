"""Actual private shell Undo routes restore a native fin's published model and view."""
import os,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from PySide6.QtCore import Qt,QThreadPool
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app2.qt.shell import MainWindow
from app2.qt import theme
from app2.diagnostics.driver import Driver
from stages import plyio
from test_stages_plus import _PlusRun,arg
from test_app2_fin_validation import l_part


class NativeFixture(_PlusRun):
    def fake_cmd(self,cmd):
        if os.path.basename(cmd[1])=='fuse.py':
            mesh=l_part();plyio.write_mesh(arg(cmd,'--out'),mesh.vertices,mesh.faces)
            return 'STAGE done {"faces": 20}\n'
        return super().fake_cmd(cmd)


@pytest.mark.parametrize('route',['shortcut','menu','button'])
def test_native_fin_undo_from_actual_shell_routes(tmp_path,monkeypatch,route):
    monkeypatch.delenv('POINTYOINK_DRIVER',raising=False)
    monkeypatch.setenv('POINTYOINK_NO_DEVICE','1')
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    app=QApplication.instance() or QApplication([]);theme.install(app)
    def no_session(*args,**kwargs):raise AssertionError('No scanner session may start')
    win=MainWindow(no_session,gl=True,projects_dir=str(tmp_path),defer_new_project=True)
    case=NativeFixture();case.setUp();case.e.clear_ops('fused.plus');case.run_()
    base_graph,base_info,_,_=case.e.plan()
    base_report=case.e.store.dir(base_graph.by_id['report'].store,base_info['report']['key'])
    page=win.build;model=Path(case.tmp)/'model.ply';initial=model.read_bytes()
    page.clock.stop();page.poll.stop();page.debounce.stop()
    page.stage='meshed';page.host_root=case.tmp;page.host=NS(busy=False,close=lambda *a:None,status=lambda:None)
    page._tools_on=True;page._can=(True,False);page._refresh_tools=lambda:None
    page._refresh_all=lambda:None;page._refresh_rebuild=lambda:None
    page.vp.set_edit_workspace(True)
    runs=[]
    def show_model():
        V,F,_=plyio.read_mesh(str(model))
        import trimesh
        m=trimesh.Trimesh(V,F,process=False)
        page.vp.view.show_items([dict(key='part',kind='mesh',V=V.astype('float32'),F=F.astype('uint32'),N=np.asarray(m.vertex_normals,dtype='float32'),colour=np.array([.7,.7,.7],dtype='float32'))],edit='part',frame=True)
        page.vp.view.tools_on=True;page.undo_b.setEnabled(True)
        page.shown={'path':str(model),'kind':'mesh','n':len(F)}
        page.vp.main.set_chips(page.view_label(),'Polygons: %d'%len(F),None)
    def run(backend):
        result=case.e.run(to=backend.to,publish=backend.publish,quiet=True)
        assert result.ok,result.error
        runs.append(dict(faces=len(plyio.read_mesh(str(model))[1]),publish=backend.publish))
        show_model()
    page.start_run=run
    def call(name,*args,**kwargs):
        if name=='h_plan':result={'to_run':len(case.e.plan()[2])}
        else:result=getattr(case.e,name)(*args)
        if kwargs.get('then'):kwargs['then'](result)
    page.call=call
    win.stack.setCurrentWidget(win.build_slot);win.startup.hide();win.resize(1400,950);win.show();win.activateWindow();page.setFocus();QTest.qWait(180)
    show_model();page.vp.view.cam.orbit(0,220);page.vp.view.update()
    private=Driver(win,str(tmp_path/'undo-private.sock'))
    out=Path(os.environ.get('POINTYOINK_UNDO_ARTIFACTS',str(tmp_path/'screens')));out.mkdir(parents=True,exist_ok=True)
    try:
        page.panel['meshed'].show_tab('Add fin');page.vp.view.tools_on=True;page.on_extra('Add fin',0)
        points=[[12.,0,2],[2.,0,12]]
        if getattr(page.fin,'minimum_picks',2)>=3:points.append([20.,0,2])
        for point in points:
            page.vp.view.surface_picked.emit(point);QThreadPool.globalInstance().waitForDone(5000);app.processEvents()
        page.on_button('Add fin');QTest.qWait(150)
        if not runs[-1]['publish']:
            # Seed the earlier application as a published model to isolate Undo.
            case.run_();show_model()
        import shutil
        shutil.rmtree(base_report)  # Only this fixture's old Report cache.
        page.vp.view.cam.orbit(0,220);page.vp.view.update();QTest.qWait(100)
        assert len(page.vp.view.items['part']['F'])==60 and model.read_bytes()!=initial
        if route=='button':
            win.edit.attach(page);win.stack.setCurrentWidget(win.edit);win.edit.sync();QTest.qWait(100)
        win.grab().save(str(out/(route+'-before-undo.png')))
        page.setFocus();win.activateWindow();QTest.qWait(100)
        if route=='shortcut':QTest.keyClick(win,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
        elif route=='menu':private.cmd_menu({'menu':'Edit','action':'Undo'})
        else:win.edit.undo_button.click()
        QTest.qWait(180)
        if len(page.vp.view.items['part']['F'])==20:page.vp.view.cam.orbit(0,220)
        page.vp.view.update();QTest.qWait(100)
        win.grab().save(str(out/(route+'-after-undo.png')))
        assert model.read_bytes()==initial
        assert len(page.vp.view.items['part']['F'])==20
        assert runs[-1]=={'faces':20,'publish':True}
        print('UNDO_UI_ROUTE',route,'60 -> 20 faces, published bytes restored')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        private.server.close();page.host=None;win.close();win.deleteLater();app.processEvents();case.tearDown()
