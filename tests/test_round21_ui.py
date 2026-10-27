"""Derived RAM checks, run-local admission, and resource/Edit affordances."""
import io,os,threading,time
from types import SimpleNamespace as NS
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
from PySide6.QtTest import QTest


@pytest.fixture
def app():
    from app2.qt import theme
    app=QApplication.instance() or QApplication([]);theme.install(app);return app


@pytest.mark.parametrize('total,available,low',[(8,4,True),(14,10,True),(64,40,False)])
def test_planned_ram_uses_uncapped_requests_and_existing_reserve(monkeypatch,total,available,low):
    from app2.builder.build_memory import planned_memory
    from scanner.stages import resources as R
    monkeypatch.setattr(R,'memory_info',lambda:(total*R.GIB,available*R.GIB))
    monkeypatch.setattr(R,'load',lambda:dict(memory_gb=16,cpu_percent=80))
    monkeypatch.setattr(R,'limit_options',lambda:dict(mode='manual',cpu_threads=16,gpu_jobs=2))
    monkeypatch.setattr(R,'cpu_count',lambda:20)
    steps=[NS(id='side.platter@'+str(i),alias=None,kind='side.platter',inputs={},params={},method=NS(mem_gb=8,gpu=False)) for i in range(2)]
    engine=NS(mem_gb=None,plan=lambda **kwargs:(None,{},steps,0))
    result=planned_memory(engine)
    assert result['low'] is low and result['required']==8*R.GIB and result['peak']==16*R.GIB
    assert result['reserve']==R.desktop_reserve_bytes(total*R.GIB)
    assert result['usable']==max(0,available*R.GIB-result['reserve'])


def test_cached_plan_needs_no_warning_even_on_small_machine(monkeypatch):
    from app2.builder.build_memory import planned_memory
    from scanner.stages import resources as R
    monkeypatch.setattr(R,'memory_info',lambda:(8*R.GIB,3*R.GIB))
    engine=NS(mem_gb=None,plan=lambda **kwargs:(None,{},[],0))
    result=planned_memory(engine)
    assert not result['low'] and result['required']==0


@pytest.mark.parametrize('accept',[True,False])
def test_memory_choice_is_nonblocking_and_run_local(app,accept):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    from app2.builder.stagehost import StagesRun
    page=BuildStepPage(SessionBridge(),gl=False);page.session=NS();page.host=NS(busy=False,close=lambda *a:None)
    page.host_root='';page._ensure_host=lambda **kwargs:None
    launches=[];callbacks=[]
    page.call=lambda name,*args,**kwargs:callbacks.append(kwargs['then'])
    backend=StagesRun();started=time.monotonic()
    assert page.check_build(backend,launches.append)
    assert time.monotonic()-started<.1 and not launches
    callbacks[0](dict(low=True,unknown=False,available=4<<30,required=8<<30,peak=8<<30,reserve=2<<30))
    dialog=page._memory_dialog;dialog.show();app.processEvents()
    try:
        assert dialog.isVisible() and not launches
        (dialog.continue_button if accept else dialog.cancel_button).click()
        assert bool(launches) is accept
        if accept:
            assert launches[0].env==dict(POINTYOINK_BUILD_WORKERS='1',POINTYOINK_BUILD_GPU_SLOTS='1')
            assert not hasattr(backend,'env')
    finally:page.shutdown();page.close()


def test_build_does_not_launch_before_async_low_ram_choice(app):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    from app2.builder.stagehost import StagesRun
    launches=[];callbacks=[];page=BuildStepPage(SessionBridge(),gl=False)
    page.session=NS(build=launches.append);page.state=NS(recording=False,building=False,any_frames=True)
    page.host=NS(busy=False,close=lambda *args:None);page.host_root=''
    page._ensure_host=lambda **kwargs:None;page._refresh_all=lambda:None
    page.call=lambda name,*args,**kwargs:callbacks.append(kwargs['then'])
    try:
        assert page.start_run(StagesRun())
        assert not launches and len(callbacks)==1
        callbacks[0](dict(low=True,unknown=False,available=4<<30,required=8<<30,peak=8<<30,reserve=2<<30))
        page._memory_dialog.cancel_button.click()
        assert not launches and page.run_mine is None
    finally:page.shutdown();page.close()


def test_actual_engine_admission_consumes_the_fewer_jobs_override(tmp_path,monkeypatch):
    from test_stages_parallel import engine
    from test_stages_core import install_fakes,remove_fakes,FakeMixin
    from stages import resources as R
    install_fakes();original=FakeMixin.execute;active=[0];peak=[0];lock=threading.Lock()
    monkeypatch.setattr(R,'budget',lambda memory_gb=None:R.Budget(32*R.GIB,32,16,1600))
    monkeypatch.setattr('stages.parallel.gpu_slots',lambda requested=2:(min(2,requested),None))
    def execute(self,context):
        if context.step.kind!='side.fuse':return original(self,context)
        with lock:active[0]+=1;peak[0]=max(peak[0],active[0])
        try:time.sleep(.04);return original(self,context)
        finally:
            with lock:active[0]-=1
    monkeypatch.setattr(FakeMixin,'execute',execute)
    try:
        monkeypatch.setenv('POINTYOINK_BUILD_WORKERS','1');monkeypatch.setenv('POINTYOINK_BUILD_GPU_SLOTS','1')
        assert engine(tmp_path/'serial').run(to='lineup',publish=False).ok
        assert peak[0]==1
        peak[0]=0;monkeypatch.setenv('POINTYOINK_BUILD_WORKERS','24');monkeypatch.setenv('POINTYOINK_BUILD_GPU_SLOTS','2')
        assert engine(tmp_path/'parallel').run(to='lineup',publish=False).ok
        assert peak[0]==2
    finally:remove_fakes()


def test_build_runner_delivers_only_this_runs_environment_to_actual_child(tmp_path):
    import json,sys
    from app2.builder.runner import BuildRunner
    from app2.builder.stagehost import StagesRun
    child=tmp_path/'child.py';result=tmp_path/'environment.json'
    child.write_text("import json,os,sys\njson.dump({name:os.environ.get(name) for name in ('POINTYOINK_BUILD_WORKERS','POINTYOINK_BUILD_GPU_SLOTS')},open(sys.argv[1],'w'))\nprint('STAGE done {\\\"model\\\":null}')\n")
    backend=StagesRun(publish=False);backend.env=dict(POINTYOINK_BUILD_WORKERS='1',POINTYOINK_BUILD_GPU_SLOTS='1')
    backend.argv=lambda project:[sys.executable,str(child),str(result)]
    before={key:os.environ.get(key) for key in backend.env};finished=[]
    runner=BuildRunner(str(tmp_path),backend,lambda *args:None,finished.append)
    runner.start();runner.thread.join(10)
    assert not runner.thread.is_alive() and runner.prog.returncode==0
    assert json.loads(result.read_text())==backend.env
    assert {key:os.environ.get(key) for key in backend.env}==before


@pytest.mark.parametrize('tile',['RAM','CPU','GPU','SCANNER'])
def test_escape_closes_actual_resource_window_and_stops_timer(app,monkeypatch,tile):
    from app2.qt.shell import MainWindow
    from app2.qt.resource_panel import ResourcePanel
    monkeypatch.setattr('app2.builder.backend_notes.cuda_available',lambda:False)
    window=MainWindow(lambda *a,**kw:None,gl=False,defer_new_project=True);window.show()
    panel=ResourcePanel(tile,window);panel.show();panel.activateWindow();QTest.qWait(30)
    try:
        QTest.keyClick(panel.close_button,Qt.Key.Key_Escape);app.processEvents()
        assert not panel.isVisible() and not panel.timer.isActive()
    finally:panel.close();window.close()


@pytest.mark.parametrize('capability',[False,None,True])
def test_gpu_build_controls_follow_known_cuda_capability(app,monkeypatch,capability):
    from app2.qt.resource_panel import ResourcePanel
    monkeypatch.setattr('app2.builder.backend_notes.cuda_available',lambda:capability)
    sample=dict(cpu=20,app_cpu=5,ram_used=4<<30,ram_total=14<<30,gpu=10,vram_total=4<<30,vram_used=1<<30)
    panel=ResourcePanel('RAM',sampler=NS(last=sample,samples=[sample],build_usage=NS(average={})))
    panel.show();app.processEvents()
    try:
        assert panel.values['gpu'].isHidden() is (capability is False)
        assert panel.value_labels['gpu'].isHidden() is (capability is False)
        if capability is False:assert panel.gpu_note.text()=='Builds run on the CPU.'
        else:assert 'Standard GPU side budget' in panel.gpu_note.text()
    finally:panel.close()


def test_edit_blank_edge_handle_replaced_by_left_options_toggle(app):
    from app2.qt.edit_page import EditPage
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    window=QWidget();layout=QVBoxLayout(window);edit=EditPage();layout.addWidget(edit)
    build=BuildStepPage(SessionBridge(),gl=False);window.resize(1280,800);window.show();edit.attach(build);app.processEvents()
    try:
        assert edit.edge_button.isHidden() and not build.right.isVisible()
        assert edit.options_button.isVisible() and edit.options_button.text()=='Options'
        for stage,title in [('side','Side surface'),('fused','Fused surface'),('meshed','Finished model')]:
            build.stage=stage;edit.sync();app.processEvents()
            assert edit.title.text()==title
            assert edit.title.fontMetrics().horizontalAdvance(title)<=edit.title.width()
        edit.options_button.click();app.processEvents();assert build.right.isVisible()
        edit.options_button.click();assert not build.right.isVisible()
        assert build.scroll.verticalScrollBarPolicy()==Qt.ScrollBarPolicy.ScrollBarAsNeeded
    finally:edit.detach();build.shutdown();build.close();window.close()


@pytest.mark.parametrize('available,required,peak',[(4,8,16),(10,8,8)])
def test_low_ram_dialog_reserves_full_wrapped_message_height(app,available,required,peak):
    from app2.qt.build_memory import BuildMemoryDialog
    dialog=BuildMemoryDialog(dict(available=available<<30,required=required<<30,reserve=int(3.5*(1<<30)),peak=peak<<30))
    dialog.open();app.processEvents()
    try:
        assert dialog.message.height()>=dialog.message.heightForWidth(dialog.message.width())
        assert dialog.message.geometry().bottom()<dialog.continue_button.mapTo(dialog,dialog.continue_button.rect().topLeft()).y()
    finally:dialog.close()


def test_build_run_check_cannot_refresh_after_page_shutdown(app):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    from app2.builder.stagehost import StagesRun
    page=BuildStepPage(SessionBridge(),gl=False);events=[]
    page.state=NS(recording=False,building=False,any_frames=True)
    page.session=NS(build=lambda backend:None);page.notice=lambda *a:events.append('notice');page._refresh_all=lambda:events.append('refresh')
    backend=StagesRun()
    try:
        if hasattr(page,'_launch_run'):page._launch_run(backend)
        else:page.start_run(backend)
        page.shutdown();events.clear()
        QTest.qWait(4100);assert events==[]
        page._run_check(backend);assert events==[]
        assert page._run_watchdog.parent() is page and not page._run_watchdog.isActive()
    finally:page.shutdown();page.close()


def test_project_attach_cancels_owned_build_run_check(app):
    from app2.qt.build_step import BuildStepPage
    from app2.qt.bridge import SessionBridge
    page=BuildStepPage(SessionBridge(),gl=False)
    try:
        page._run_watchdog_backend=object();page._run_watchdog.start(30)
        page.attach(None);assert not page._run_watchdog.isActive() and page._run_watchdog_backend is None
    finally:page.shutdown();page.close()
