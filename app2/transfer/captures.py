# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device capture listing/copy worker. Every CLI result is one flushed JSON line."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import subprocess
import sys
import tempfile

IMAGES = {'.png', '.jpg', '.jpeg'}
VIDEOS = {'.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v'}


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def list_captures(root, max_depth=4, max_entries=10000):
    root = Path(root).resolve(strict=True); rows = []; seen = 0
    def walk(folder, depth):
        nonlocal seen
        import itertools
        with os.scandir(folder) as listing:
            entries = list(itertools.islice(listing, max_entries-seen+1))
        if len(entries)+seen > max_entries: raise ValueError('Too many files. Choose a smaller capture folder.')
        entries.sort(key=lambda entry: entry.name.lower())
        for entry in entries:
            seen += 1
            if seen > max_entries: raise ValueError('Too many files. Choose a smaller capture folder.')
            if entry.is_symlink(): continue
            path = Path(entry.path); rel = path.relative_to(root)
            if entry.is_dir(follow_symlinks=False):
                if entry.name == 'Projects': continue
                if depth < max_depth: walk(path, depth+1)
                continue
            if not entry.is_file(follow_symlinks=False): continue
            suffix = path.suffix.lower()
            kind = 'video' if suffix in VIDEOS else 'image' if suffix in IMAGES and 'Screenshots' in rel.parts[:-1] else None
            if kind:
                stat = entry.stat(follow_symlinks=False)
                rows.append(dict(relpath=rel.as_posix(), name=path.name, size=stat.st_size, kind=kind, modified=stat.st_mtime))
    walk(root, 0)
    return sorted(rows, key=lambda row: (-row['modified'], row['relpath']))


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Use a file path inside the capture folder.')
    return Path(*path.parts)


def _inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root): raise ValueError('A capture path leaves its folder.')
    return resolved


def copy_captures(root, dest, paths, notify=lambda *_a, **_k: None, chunk_size=1024*1024, flat=False):
    root = Path(root).resolve(strict=True); dest = Path(dest).resolve()
    if dest.is_relative_to(root): raise ValueError("Choose a PC folder outside the scanner storage.")
    dest.mkdir(parents=True, exist_ok=True)
    if len(paths) > 10000: raise ValueError('Select fewer files at once.')
    for value in dict.fromkeys(paths):
        rel = _relative(value); source = _inside(root, root/rel)
        if source.suffix.lower() not in IMAGES | VIDEOS or not source.is_file(): raise ValueError('That capture file is unavailable.')
        parent = _inside(dest, dest if flat else dest/rel.parent); parent.mkdir(parents=True, exist_ok=True)
        parent = _inside(dest, parent)
        total = source.stat().st_size; copied = 0
        fd, temporary = tempfile.mkstemp(prefix='.'+rel.name+'.', suffix='.part', dir=parent)
        try:
            with os.fdopen(fd, 'wb') as target, source.open('rb') as original:
                while True:
                    chunk = original.read(chunk_size)
                    if not chunk: break
                    target.write(chunk); copied += len(chunk)
                    notify('progress', relpath=rel.as_posix(), bytes=copied, total=total)
                target.flush(); os.fsync(target.fileno())
            if copied != total: raise ValueError('The capture changed during copying. Copy it again.')
            # Linking atomically publishes a complete file without replacing any existing copy.
            candidate = parent/rel.name; number = 1
            while True:
                try: os.link(temporary, candidate); break
                except FileExistsError:
                    candidate = parent/('%s (%d)%s' % (rel.stem, number, rel.suffix)); number += 1
            os.utime(candidate, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns))
            notify('saved', relpath=rel.as_posix(), path=str(candidate), bytes=copied)
        finally:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def mounted(path, mountinfo='/proc/self/mountinfo'):
    wanted = os.path.abspath(path)
    for line in Path(mountinfo).read_text().splitlines():
        fields = line.split()
        if len(fields) > 4:
            decoded = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1],8)), fields[4])
            if os.path.abspath(decoded) == wanted: return True
    return False


def mtp_device(sysroot='/sys/bus/usb/devices'):
    devices = []
    for vendor in Path(sysroot).glob('*/idVendor'):
        if vendor.read_text().strip().lower() != '2207': continue
        folder = vendor.parent
        if not any(p.read_text().strip() == '06' for p in folder.glob('*/bInterfaceClass')): continue
        devices.append((int((folder/'busnum').read_text()), int((folder/'devnum').read_text())))
    if not devices: raise ValueError('Choose File Transfer on the scanner, then connect again.')
    if len(devices) != 1: raise ValueError('More than one scanner is in File Transfer. Connect one at a time.')
    return devices[0]


def _device_allowed():
    if os.environ.get('POINTYOINK_NO_DEVICE') == '1': raise ValueError('Device access is disabled in this run.')


def _receipt(path):
    return path.parent/('.'+path.name+'.pointyoink-owner.json')


def mount_source(path, mountinfo='/proc/self/mountinfo'):
    """Read the kernel's mount source without touching a possibly stalled mount."""
    wanted = os.path.abspath(path)
    decode = lambda text: re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1],8)), text)
    for line in Path(mountinfo).read_text().splitlines():
        fields = line.split()
        if len(fields) > 4 and os.path.abspath(decode(fields[4])) == wanted:
            tail = fields[fields.index('-')+1:]
            return decode(tail[1]) if len(tail) > 1 else None
    return None


def _cleanup_owned_mount(path, token):
    marker = 'pointyoink-captures-' + token
    if mount_source(path) == marker:
        result = subprocess.run(['fusermount','-u',str(path)],capture_output=True,text=True,timeout=10)
        if result.returncode: return False
    elif mounted(path):
        return False  # A different mount appeared: never touch it.
    receipt = _receipt(path)
    try:
        if json.loads(receipt.read_text()).get('token') == token: receipt.unlink()
    except (OSError, ValueError): pass
    return True


def mount_device(path, notify=emit):
    _device_allowed(); path = Path(path).absolute()
    if mounted(path):
        try: receipt = json.loads(_receipt(path).read_text())
        except (OSError, ValueError): receipt = {}
        token = receipt.get('token')
        owned = (bool(token) and receipt.get('uid') == os.getuid()
                 and receipt.get('mountpoint') == str(path)
                 and mount_source(path) == 'pointyoink-captures-'+token)
        notify('mounted', mountpoint=str(path), root=str(path/'Internal shared storage'), owned=owned, token=token if owned else None); return
    bus, device = mtp_device(); path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()): raise ValueError('The mount folder must be empty.')
    token = secrets.token_hex(16); marker = 'pointyoink-captures-'+token
    receipt = dict(mountpoint=str(path), token=token, uid=os.getuid(), bus=bus, device=device)
    # Save identity before spawning so a killed worker can recover this exact mount.
    receipt_path = _receipt(path)
    fd = os.open(receipt_path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd,'w') as handle: json.dump(receipt,handle)
    proc = None
    try:
        proc = subprocess.Popen(['jmtpfs', str(path), '-device=%d,%d' % (bus,device), '-o', 'fsname='+marker], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _out, error = proc.communicate(timeout=35)
        if proc.returncode or mount_source(path) != marker:
            raise ValueError('USB could not open. Choose File Transfer and close file-manager windows using the scanner, then retry.')
        notify('mounted', mountpoint=str(path), root=str(path/'Internal shared storage'), owned=True, token=token)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        try: _cleanup_owned_mount(path,token)
        except (OSError, subprocess.TimeoutExpired): pass  # Receipt permits recovery on the next explicit connect.
        raise


def unmount_device(path, token, notify=emit):
    _device_allowed(); path = Path(path).absolute(); receipt_path = _receipt(path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('mountpoint') != str(path) or receipt.get('uid') != os.getuid() or not secrets.compare_digest(receipt.get('token',''), token or ''):
        raise ValueError('This mount was not opened by this transfer session.')
    if mounted(path):
        if mount_source(path) != 'pointyoink-captures-'+token:
            raise ValueError('A different device now uses this mount folder. It was not disconnected.')
        result = subprocess.run(['fusermount','-u',str(path)],capture_output=True,text=True,timeout=10)
        if result.returncode: raise ValueError('The scanner folder is still in use. Close it and retry disconnecting.')
    receipt_path.unlink(); notify('unmounted', mountpoint=str(path))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command',choices=['list','copy','mount','unmount'])
    parser.add_argument('--root'); parser.add_argument('--dest'); parser.add_argument('--paths-json',default='[]')
    parser.add_argument('--mountpoint'); parser.add_argument('--token'); parser.add_argument('--flat', action='store_true'); args = parser.parse_args()
    def cancelled(*_args): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,cancelled)
    try:
        if args.command == 'list':
            for row in list_captures(args.root): emit('capture', **row)
        elif args.command == 'copy':
            paths = json.loads(args.paths_json)
            if not isinstance(paths,list) or not all(isinstance(path,str) for path in paths): raise ValueError('Select capture files to copy.')
            copy_captures(args.root,args.dest,paths,emit,flat=args.flat)
        elif args.command == 'mount': mount_device(args.mountpoint)
        elif args.command == 'unmount': unmount_device(args.mountpoint,args.token)
        emit('done'); return 0
    except KeyboardInterrupt:
        emit('cancelled'); return 2
    except Exception as exc:
        emit('error', message=str(exc)); return 1


if __name__ == '__main__': sys.exit(main())
