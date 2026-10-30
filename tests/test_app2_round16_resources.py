"""Resource tiles, accounting, build averages, and next-job controls use private data."""
import os,json
from types import SimpleNamespace as NS
from collections import deque
import pytest
from PySide6.QtCore import QPoint,Qt
from PySide6.QtWidgets import QApplication,QMainWindow
from PySide6.QtTest import QTest
from app2.qt import usage_meter as U
from scanner.stages import resources as R

@pytest.fixture(autouse=True)
def private(tmp_path,monkeypatch):
    monkeypatch.delenv('POINTYOINK_DRIVER',raising=False)
    monkeypatch.setenv('POINTYOINK_CONFIG',str(tmp_path/'config.json'))
    monkeypatch.setenv('POINTYOINK_NO_DEVICE','1')
    monkeypatch.setattr(R,'memory_info',lambda:(64*R.GIB,40*R.GIB))
    monkeypatch.setattr(R,'cpu_count',lambda:16)

@pytest.fixture
def app():return QApplication.instance() or QApplication([])

@pytest.mark.parametrize('index,tile',list(enumerate(['SCANNER','CPU','GPU','RAM'])))
def test_each_tile_opens_panel(app,monkeypatch,index,tile):
    samples=NS(last={},hist={key:deque() for key in ['cpu','gpu','ram']})
    monkeypatch.setattr(U,'_shared',samples)
    hits=[];monkeypatch.setattr(U.UsageMeter,'open_panel',lambda self,key:hits.append(key),raising=False)
    widget=U.UsageMeter();widget.show();app.processEvents()
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,pos=QPoint(index*(widget.TILE_W+8)+10,10))
    assert hits==[tile];widget.close()

def test_gpu_accounting_includes_owned_graphics_and_compute_only():
    xml='<nvidia_smi_log><gpu><processes>'+''.join('<process_info><pid>%d</pid><type>%s</type><used_memory>%d MiB</used_memory></process_info>'%row for row in [(100,'G',30),(101,'C',50),(900,'C',200)])+'</processes></gpu></nvidia_smi_log>'
    assert U.gpu_process_memory(xml,{100,101})==80*(1<<20)
    assert U.gpu_process_memory('<nvidia_smi_log><gpu/></nvidia_smi_log>',{100}) is None

def test_last_build_average_persists_and_excludes_idle_samples():
    usage=U.BuildUsage();usage.add({'cpu':99})
    usage.set_building(True);usage.add({'cpu':40,'app_cpu':10});usage.add({'cpu':60,'app_cpu':20});usage.set_building(False)
    usage.add({'cpu':99})
    assert usage.average['samples']==2 and usage.average['cpu']==50 and usage.average['app_cpu']==15
    assert U.BuildUsage().average==usage.average

def test_manual_limits_change_next_budget_not_existing_budget():
    R.save_limits(6,2,1,'manual');old=R.budget()
    R.save_limits(8,4,2,'manual');new=R.budget()
    assert (old.memory_gb,old.cpu_threads)==(6,2)
    assert (new.memory_gb,new.cpu_threads,R.gpu_job_limit())==(8,4,2)

def test_auto_rechecks_hardware_and_desktop_reserve(monkeypatch):
    R.save_limits(6,2,2,'auto')
    assert R.budget().cpu_threads==12 and R.budget().memory_gb==8
    monkeypatch.setattr(R,'memory_info',lambda:(64*R.GIB,18*R.GIB))
    assert R.budget().memory_gb==2

def test_pending_jobs_use_changed_limits(monkeypatch):
    from scanner.stages.parallel import Jobs
    R.save_limits(6,2,1,'manual');jobs=Jobs(NS(memory_gb=None,cancelled=False),R.budget(),2)
    R.save_limits(8,4,2,'manual')
    with jobs.lease(None,False,None) as budget:assert (budget.memory_gb,budget.cpu_threads)==(8,4)
    R.save_limits(4,1,1,'manual')
    with jobs.lease(None,False,None) as budget:assert (budget.memory_gb,budget.cpu_threads)==(4,1)

def test_resource_panel_shows_accounting_and_last_average(app):
    from app2.qt.resource_panel import ResourcePanel
    sample=dict(cpu=60,app_cpu=20,ram_used=10*R.GIB,ram_total=64*R.GIB,app_ram=2*R.GIB,vram_total=10*R.GIB,vram_used=3*R.GIB,gpu=70,app_vram=R.GIB)
    sm=NS(last=sample,samples=[sample,sample],build_usage=NS(average=dict(sample,samples=2,seconds=4)))
    panel=ResourcePanel('CPU',sampler=sm)
    assert '20.0%' in panel.accounting.text() and '40.0%' in panel.accounting.text()
    assert 'Last build average' in panel.average.text() and len(panel.graph.series)==2
    panel.mode.setCurrentIndex(0);assert not panel.values['threads'].isEnabled()
    panel.mode.setCurrentIndex(1);panel.values['threads'].setValue(3);panel.save.click()
    assert R.budget().cpu_threads==3 and 'next build step' in panel.note.text()
    panel.close()

def test_scanner_details_never_connect_and_clear_when_disconnected(app):
    from app2.qt.resource_panel import ResourcePanel
    calls=[];client=NS(snapshot=lambda:('Receiving',NS(angles=(1,2,3)),4,.1),connect=lambda *args:calls.append(args))
    win=QMainWindow();win.session=NS(source=NS(running=True,kind='wifi',seq=20));win.motion_panel=NS(client=client)
    panel=ResourcePanel('SCANNER',win,sampler=NS(last={},samples=[],build_usage=NS(average={})))
    assert 'Wi-Fi' in panel.current.text() and 'Motion: Receiving' in panel.accounting.text() and calls==[]
    win.session=None;panel.refresh();assert panel.current.text()=='Scanner not connected' and 'Motion: Receiving' not in panel.accounting.text()
    panel.close();win.close()


def test_lower_limit_waits_for_running_job_and_keeps_its_budget():
    import threading
    from scanner.stages.parallel import Jobs
    R.save_limits(8,4,2,'manual');jobs=Jobs(NS(memory_gb=None,cancelled=False),R.budget(),2)
    entered=threading.Event();release=threading.Event();second=threading.Event();budgets=[]
    def first_job():
        with jobs.lease(None,False,None) as b:
            budgets.append(b);entered.set();release.wait(2)
    def next_job():
        with jobs.lease(None,False,None) as b:budgets.append(b);second.set()
    a=threading.Thread(target=first_job);a.start();assert entered.wait(1)
    R.save_limits(4,1,1,'manual');b=threading.Thread(target=next_job);b.start()
    assert not second.wait(.08)
    assert (budgets[0].memory_gb,budgets[0].cpu_threads)==(8,4)
    release.set();a.join(2);b.join(2)
    assert second.is_set() and (budgets[1].memory_gb,budgets[1].cpu_threads)==(4,1)


def test_gpu_job_setting_blocks_new_job_until_old_job_finishes(monkeypatch):
    import threading
    from scanner.stages.parallel import Jobs
    monkeypatch.setattr('scanner.stages.parallel.subprocess.check_output',lambda *a,**kw:'12800')
    R.save_limits(12,8,2,'manual');jobs=Jobs(NS(memory_gb=None,cancelled=False),R.budget(),2)
    entered=threading.Event();release=threading.Event();next_entered=threading.Event()
    def first_job():
        with jobs.lease(2,True,'side.fuse'):entered.set();release.wait(2)
    def next_job():
        with jobs.lease(2,True,'side.fuse'):next_entered.set()
    a=threading.Thread(target=first_job);a.start();assert entered.wait(1)
    R.save_limits(12,8,1,'manual');b=threading.Thread(target=next_job);b.start()
    assert not next_entered.wait(.08)
    release.set();a.join(2);b.join(2)
    assert next_entered.is_set() and jobs.peak_jobs==1


def test_scanner_panel_reports_receiving_wifi_files_without_live_camera(app,monkeypatch):
    from app2.qt.resource_panel import ResourcePanel
    from app2.qt import link_status
    monkeypatch.setitem(link_status.state,'wifi','receiving');monkeypatch.setitem(link_status.state,'rate',2<<20)
    win=QMainWindow();win.session=None
    panel=ResourcePanel('SCANNER',win,sampler=NS(last={},samples=[],build_usage=NS(average={})))
    assert 'Wi-Fi for file transfer' in panel.current.text() and '2.0 MB/s' in panel.accounting.text()
    panel.close();win.close()


def test_shown_resource_panel_closes_normally(app):
    from app2.qt.resource_panel import ResourcePanel
    panel=ResourcePanel('CPU',sampler=NS(last={},samples=[],build_usage=NS(average={})))
    panel.show();app.processEvents();assert panel.isVisible()
    panel.close_button.click();app.processEvents();assert not panel.isVisible()


def test_gpu_panel_lists_each_recorded_fusion_plan_and_process_peak(app):
    from app2.qt.resource_panel import ResourcePanel,fusion_memory_rows
    snapshot=NS(sides=[{'id':'scan-a','n':3}],groups=[{'nodes':[
        {'id':'side.fuse@scan-a','stats':{'estimated_peak_bytes':R.GIB,'peak_vram_bytes':R.GIB//2}},
        {'id':'fused','stats':{'estimated_peak_bytes':2*R.GIB,'peak_vram_bytes':R.GIB}},
        {'id':'report','stats':{}}]}])
    assert fusion_memory_rows(snapshot)==[['Side 3','1.00 GiB','0.50 GiB'],['Fused','2.00 GiB','1.00 GiB']]
    win=QMainWindow();win.build=NS(snap=snapshot)
    panel=ResourcePanel('GPU',win,sampler=NS(last={'gpu':50},samples=[],build_usage=NS(average={})))
    assert panel.fusion_table.rowCount()==2 and panel.fusion_table.item(1,2).text()=='1.00 GiB'
    panel.close();win.close()


def test_missing_gpu_data_is_unavailable_and_not_zero(app):
    from app2.qt.resource_panel import ResourcePanel
    panel=ResourcePanel('GPU',sampler=NS(last={},samples=[],build_usage=NS(average={'samples':2,'cpu':50})))
    assert 'unavailable' in panel.current.text() and '0.0%' not in panel.current.text()
    assert 'no build samples' in panel.average.text()
    panel.close()
