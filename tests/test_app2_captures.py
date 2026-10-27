"""Device capture helpers use disposable directories and simulated commands only."""
import os
import tempfile
from pathlib import Path
os.environ['POINTYOINK_CONFIG']=str(Path(tempfile.mkdtemp(prefix='pointyoink-captures-'))/'config.json')
import json
import subprocess
import sys
import pytest
from app2.transfer import captures as C


def make(root,path,data=b'picture'):
    full=root/path; full.parent.mkdir(parents=True,exist_ok=True); full.write_bytes(data); return full


def test_listing_images_and_nested_videos_skips_projects_and_links(tmp_path):
    make(tmp_path,'Screenshots/a.png'); make(tmp_path,'Recordings/old/b.mp4')
    make(tmp_path,'Projects/p/recording.mp4'); make(tmp_path,'Other/a.png')
    (tmp_path/'linked.mp4').symlink_to(tmp_path/'Recordings/old/b.mp4')
    rows=C.list_captures(tmp_path)
    assert {row['relpath'] for row in rows}=={'Screenshots/a.png','Recordings/old/b.mp4'}
    assert {row['kind'] for row in rows}=={'image','video'}
    with pytest.raises(ValueError,match='Too many'): C.list_captures(tmp_path,max_entries=1)


def test_copy_preserves_folders_metadata_and_never_overwrites(tmp_path):
    root=tmp_path/'device'; dest=tmp_path/'copies'
    first=make(root,'Screenshots/a.png',b'first'); os.utime(first,(123456,123456))
    make(root,'Other/a.mp4',b'movie'); events=[]
    paths=['Screenshots/a.png','Other/a.mp4']
    C.copy_captures(root,dest,paths,lambda event,**payload:events.append((event,payload)),chunk_size=2)
    C.copy_captures(root,dest,['Screenshots/a.png'])
    assert (dest/'Screenshots/a.png').read_bytes()==b'first'
    assert (dest/'Screenshots/a (1).png').read_bytes()==b'first'
    assert (dest/'Other/a.mp4').read_bytes()==b'movie'
    assert int((dest/'Screenshots/a.png').stat().st_mtime)==123456
    assert sum(event=='saved' for event,payload in events)==2
    assert not list(dest.rglob('*.part'))


@pytest.mark.parametrize('path',['../outside.mp4','/tmp/a.mp4','Screenshots/../../a.png','a\\b.mp4'])
def test_bad_relative_paths_are_refused(tmp_path,path):
    root=tmp_path/'device';root.mkdir()
    with pytest.raises(ValueError): C.copy_captures(root,tmp_path/'dest',[path])


def test_cancel_cleans_partial_file(tmp_path):
    root=tmp_path/'device'; make(root,'Screenshots/a.png',b'a'*100)
    def cancel(*args,**kwargs): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): C.copy_captures(root,tmp_path/'dest',['Screenshots/a.png'],cancel,chunk_size=10)
    assert not list((tmp_path/'dest').rglob('*.part'))
    assert not (tmp_path/'dest/Screenshots/a.png').exists()


def test_symlink_escape_refused(tmp_path):
    root=tmp_path/'device';root.mkdir(); source=make(tmp_path,'outside.mp4')
    (root/'escape.mp4').symlink_to(source)
    with pytest.raises(ValueError): C.copy_captures(root,tmp_path/'dest',['escape.mp4'])


def test_targeted_mtp_detection_fake_sysfs(tmp_path):
    d=tmp_path/'1-2'; d.mkdir()
    (d/'idVendor').write_text('2207');(d/'busnum').write_text('1');(d/'devnum').write_text('7')
    interface=d/'1-2:1.0';interface.mkdir();(interface/'bInterfaceClass').write_text('06')
    assert C.mtp_device(tmp_path)==(1,7)
    (interface/'bInterfaceClass').write_text('0e')
    with pytest.raises(ValueError,match='File Transfer'): C.mtp_device(tmp_path)


def test_existing_mount_reused_without_subprocess(tmp_path,monkeypatch):
    monkeypatch.delenv('POINTYOINK_NO_DEVICE',raising=False)
    monkeypatch.setattr(C,'mounted',lambda path:True)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**k:pytest.fail('Must not mount again'))
    events=[]; C.mount_device(tmp_path,lambda event,**info:events.append((event,info)))
    assert events[0][1]['owned'] is False and events[0][1]['token'] is None


def test_unmount_wrong_token_never_runs_command(tmp_path,monkeypatch):
    monkeypatch.delenv('POINTYOINK_NO_DEVICE',raising=False)
    C._receipt(tmp_path).write_text(json.dumps(dict(mountpoint=str(tmp_path),token='right',uid=os.getuid())))
    monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:pytest.fail('Must not unmount'))
    with pytest.raises(ValueError,match='not opened'): C.unmount_device(tmp_path,'wrong')


def test_cli_json_lines_without_device_access(tmp_path):
    root=tmp_path/'device';make(root,'Screenshots/a.png')
    env=dict(os.environ,POINTYOINK_NO_DEVICE='1')
    result=subprocess.run([sys.executable,'-m','app2.transfer.captures','list','--root',str(root)],capture_output=True,text=True,env=env,check=True)
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert [event['event'] for event in events]==['capture','done']
    blocked=subprocess.run([sys.executable,'-m','app2.transfer.captures','mount','--mountpoint',str(tmp_path/'mnt')],capture_output=True,text=True,env=env)
    assert blocked.returncode==1 and json.loads(blocked.stdout)['event']=='error'


def test_mount_command_targets_one_bus_device_and_receipt(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.delenv('POINTYOINK_NO_DEVICE',raising=False)
    checks=iter([False,True]); monkeypatch.setattr(C,'mounted',lambda path:next(checks))
    monkeypatch.setattr(C,'mtp_device',lambda:(4,9)); commands=[]
    def start(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(communicate=lambda timeout:('',''),returncode=0,poll=lambda:0)
    monkeypatch.setattr(C.subprocess,'Popen',start)
    monkeypatch.setattr(C,'mount_source',lambda path:'pointyoink-captures-'+json.loads(C._receipt(Path(path)).read_text())['token'])
    events=[]; path=tmp_path/'mount'; C.mount_device(path,lambda event,**info:events.append((event,info)))
    assert commands[0][:3]==['jmtpfs',str(path),'-device=4,9']
    assert commands[0][3]=='-o' and commands[0][4].startswith('fsname=pointyoink-captures-')
    receipt=json.loads(C._receipt(path).read_text())
    assert events[0][1]['owned'] and events[0][1]['token']==receipt['token']


def test_copy_to_device_itself_refused(tmp_path):
    make(tmp_path,'Screenshots/a.png')
    with pytest.raises(ValueError,match='PC folder'): C.copy_captures(tmp_path,tmp_path/'copies',['Screenshots/a.png'])


@pytest.mark.parametrize('ours',[True,False])
def test_mount_cancel_only_cleans_its_unique_mount(tmp_path,monkeypatch,ours):
    from types import SimpleNamespace
    monkeypatch.delenv('POINTYOINK_NO_DEVICE',raising=False)
    path=tmp_path/'mount'; started=[]; unmounts=[]
    monkeypatch.setattr(C,'mtp_device',lambda:(4,9))
    monkeypatch.setattr(C,'mounted',lambda path:bool(started))
    def source(path):
        if not started: return None
        return started[0][-1].removeprefix('fsname=') if ours else 'unrelated-device'
    monkeypatch.setattr(C,'mount_source',source)
    def interrupt(timeout): raise KeyboardInterrupt()
    def start(command,**kwargs):
        started.append(command)
        return SimpleNamespace(communicate=interrupt,poll=lambda:None,terminate=lambda:None,wait=lambda timeout:0)
    monkeypatch.setattr(C.subprocess,'Popen',start)
    monkeypatch.setattr(C.subprocess,'run',lambda command,**kwargs:(unmounts.append(command) or SimpleNamespace(returncode=0)))
    with pytest.raises(KeyboardInterrupt): C.mount_device(path,lambda *a,**k:None)
    assert unmounts==([['fusermount','-u',str(path)]] if ours else [])
    assert C._receipt(path).exists() is not ours


def test_cancelled_mount_receipt_allows_explicit_recovery(tmp_path,monkeypatch):
    monkeypatch.delenv('POINTYOINK_NO_DEVICE',raising=False)
    path=tmp_path/'mount'; token='previous-worker-token'
    C._receipt(path).write_text(json.dumps(dict(mountpoint=str(path),token=token,uid=os.getuid())))
    monkeypatch.setattr(C,'mounted',lambda path:True)
    monkeypatch.setattr(C,'mount_source',lambda path:'pointyoink-captures-'+token)
    monkeypatch.setattr(C.subprocess,'Popen',lambda *a,**k:pytest.fail('Existing mount must be reused'))
    events=[]; C.mount_device(path,lambda event,**payload:events.append(payload))
    assert events[0]['owned'] and events[0]['token']==token
