# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import complete scanner projects from USB storage or a Wi-Fi receive folder.

The same inventory and copy path serves both transports. Device files are read
only; a project becomes visible in the PC library after every file is verified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import threading

from app2.transfer.captures import emit, mount_device, unmount_device
from scanner.adopt_device_scan import adopt_project


def _safe_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name or ':' in name:
        raise ValueError('Invalid project name.')
    return name


MODELS_ONLY_MARK = '.pointyoink-models-only'      # a project imported without its raw frames: view and export, no rebuild


def _files(project, models_only=False):
    """Walk without following links, and reject files that could escape the project. models_only: leave out the raw
    frame folders (data/<side>/cache) without walking them: over USB each of those thousands of files costs a second."""
    found = []
    for folder, dirs, files in os.walk(project, followlinks=False):
        if models_only: dirs[:] = [d for d in dirs if d != 'cache']
        dirs.sort(); files.sort()
        for name in dirs + files:
            path = Path(folder) / name
            if path.is_symlink(): raise ValueError('Project contains a link: ' + str(path.relative_to(project)))
        for name in files:
            path = Path(folder) / name
            if not path.is_file(): raise ValueError('Project contains an unsupported file.')
            found.append((path.relative_to(project), path.stat().st_size))
            if len(found) > 200000: raise ValueError('Project contains too many files.')
    return found


def _model_counts(project):
    """(meshes, point clouds) the scanner finished for this project, from its <name>.revo index (a small JSON file, read
    in one go; the meshes themselves sit at data/<side>/fuse_mesh.ply, too deep to look for cheaply over USB)."""
    try:
        with open(project / (project.name + '.revo'), encoding='utf8') as f: index = json.load(f)
        return int(index.get('model_mesh_count') or 0), int(index.get('model_pointcloud_count') or 0)
    except (OSError, ValueError, TypeError, AttributeError): pass
    n = 0                                     # no index (a PC-made project): .ply files at the top and in each side folder
    for d in [project] + [p for p in (project / 'data').iterdir() if p.is_dir() and not p.is_symlink()]:
        try: n += sum(1 for f in d.iterdir() if f.suffix.lower() == '.ply' and f.is_file())
        except OSError: continue
    return n, 0


def list_projects(root, dest=None):
    """dest: the PC library; each row then says whether a project of that name is already in it (exists). Yields one
    row per project as it is read, and never walks every file: over USB File Transfer each file lookup is slow and a
    full walk of every project outlasted the app's no-progress timeout. Size and file count come with the import, which
    checks every file anyway (size None here)."""
    root = Path(root).expanduser().resolve(strict=True)
    library = Path(dest).expanduser().resolve() if dest else None
    source = root / 'Projects' if (root / 'Projects').is_dir() else root
    if source.is_symlink(): raise ValueError('The Projects folder is a link.')
    for project in sorted(source.iterdir(), key=lambda p: p.name.lower()):
        if project.is_symlink() or not project.is_dir() or not (project / 'data').is_dir(): continue
        sides = [p for p in (project / 'data').iterdir() if p.is_dir() and not p.is_symlink()]; counts = _model_counts(project)
        yield dict(name=project.name, sides=len(sides), files=None, size=None, mesh=sum(counts) > 0, meshes=counts[0],
                   modified=project.stat().st_mtime, exists=bool(library and os.path.lexists(library / project.name)))


def _cache_path():
    """Sizes already read off the scanner, next to the app's config (POINTYOINK_CONFIG), so tests never touch the real one."""
    config = Path(os.environ.get('POINTYOINK_CONFIG') or Path.home() / '.config/pointyoink/config.json')
    return config.parent / 'device-sizes.json'


def _load_cache():
    try: return json.loads(_cache_path().read_text())
    except (OSError, ValueError): return {}


def _save_cache(cache):
    p = _cache_path(); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache)); tmp.replace(p)


def _stamp(project):
    """What changes when the project changes on the scanner: the .revo index's edit time and the side folders."""
    try:
        with open(project / (project.name + '.revo'), encoding='utf8') as f: edit = json.load(f).get('edit_time')
    except (OSError, ValueError, AttributeError): edit = None
    try: sides = sorted(d.name for d in (project / 'data').iterdir() if d.is_dir())
    except OSError: sides = []
    return '%s|%s' % (edit, ','.join(sides))


BAD_FILE_BYTES = 2 << 30          # no single scan file is this big; the MTP index can report garbage sizes


def project_sizes(root, notify=emit, every=400):
    """After the list: each project's size as models only (everything but data/<side>/cache, the raw frames) and in full,
    plus its file count. Over USB File Transfer each file costs about a second to copy, so the count gives the time.
    Reports every `every` files (size events with done=False) so a big project never looks stuck."""
    root = Path(root).expanduser().resolve(strict=True)
    source = root / 'Projects' if (root / 'Projects').is_dir() else root
    projects = [p for p in source.iterdir() if not p.is_symlink() and p.is_dir() and (p / 'data').is_dir()]
    sides = {p: sum(1 for d in (p / 'data').iterdir() if d.is_dir()) for p in projects}
    cache = _load_cache(); todo = []
    for project in sorted(projects, key=lambda p: (sides[p], p.name.lower())):     # fewest sides first: most rows fill fast
        stamp = _stamp(project); hit = cache.get(project.name)
        if hit and hit.get('stamp') == stamp:                    # unchanged since last read: shown at once, no walk over USB
            notify('size', name=project.name, models=hit['models'], full=hit['full'], files=hit['files'], mfiles=hit.get('mfiles'), bad=hit.get('bad', 0),
                   done=True, cached=True, state='seen', first_seen=hit.get('first_seen'))
        else: todo.append((project, stamp, 'changed' if hit else 'new', (hit or {}).get('first_seen') or time.time()))
    for project, stamp, state, first_seen in todo:
        models = full = files = bad = mfiles = 0
        for folder, dirs, names in os.walk(project, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
            raw = 'cache' in Path(folder).relative_to(project).parts
            for name in names:
                try: n = (Path(folder) / name).stat(follow_symlinks=False).st_size
                except OSError: continue
                if n > BAD_FILE_BYTES: bad += 1; continue        # the scanner's index sometimes holds a garbage size (seen: one frame at 129 TB)
                full += n; files += 1
                if not raw: models += n; mfiles += 1
                if files % every == 0: notify('size', name=project.name, models=models, full=full, files=files, mfiles=mfiles, bad=bad, done=False, state=state, first_seen=first_seen)
        notify('size', name=project.name, models=models, full=full, files=files, mfiles=mfiles, bad=bad, done=True, state=state, first_seen=first_seen)
        cache[project.name] = dict(stamp=stamp, models=models, full=full, files=files, mfiles=mfiles, bad=bad, first_seen=first_seen); _save_cache(cache)


def import_projects(root, dest, names, notify=emit, chunk_size=1024*1024, existing='skip', what=None):
    """existing: what to do when the library already has a project of that name. 'skip' leaves it and says so
    ('exists' event); 'copy' imports another copy as "Name (1)". what: {name: 'models' | 'full'} (default full); models
    copies the finished meshes, previews and index without the raw frames and marks the project models-only."""
    if existing not in ('skip', 'copy'): raise ValueError('existing must be skip or copy')
    root = Path(root).expanduser().resolve(strict=True)
    source = root / 'Projects' if (root / 'Projects').is_dir() else root
    dest = Path(dest).expanduser().resolve()
    if dest == root or dest.is_relative_to(root): raise ValueError('Choose a PC project library outside scanner storage.')
    if not names or len(names) > 100 or len(names) != len(set(names)): raise ValueError('Select 1 to 100 distinct projects.')
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        _safe_name(name)
        project = source / name
        if project.is_symlink() or not project.is_dir() or not (project / 'data').is_dir():
            raise ValueError('Project is unavailable: ' + name)
        if existing == 'skip' and os.path.lexists(dest / name):
            notify('exists', name=name, path=str(dest / name)); continue
        models_only = (what or {}).get(name) == 'models'
        files = _files(project, models_only)
        bad = [rel for rel, size in files if size > BAD_FILE_BYTES]        # garbage sizes in the scanner's index: skip, say so
        if bad: notify('badfiles', name=name, files=[str(r) for r in bad][:20], count=len(bad)); files = [f for f in files if f[1] <= BAD_FILE_BYTES]
        total = sum(size for _, size in files); copied = 0
        staging = Path(tempfile.mkdtemp(prefix='.pointyoink-import-', dir=dest))
        started = time.monotonic()
        try:
            for rel, size in files:
                origin = project / rel; target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with origin.open('rb') as input_file, target.open('wb') as output_file:
                    count = 0
                    while True:
                        block = input_file.read(chunk_size)
                        if not block: break
                        output_file.write(block); count += len(block); copied += len(block)
                        notify('progress', name=name, bytes=copied, total=total,
                               rate=round(copied/max(time.monotonic()-started, .01)))
                    output_file.flush(); os.fsync(output_file.fileno())
                if count != size or origin.stat().st_size != size:
                    raise ValueError('A scanner file changed during transfer: ' + str(rel))
                shutil.copystat(origin, target, follow_symlinks=False)
            if not (staging / 'data').is_dir(): raise ValueError('Project has no scan data.')
            if models_only: (staging / MODELS_ONLY_MARK).write_text('Imported without raw frames: finished models only.\n')
            prepared = adopt_project(staging) if not models_only else None
            candidate = dest / name; index = 1
            while os.path.lexists(candidate):
                candidate = dest / f'{name} ({index})'; index += 1
            staging.rename(candidate)
            notify('saved', name=name, path=str(candidate), bytes=copied, prepared=prepared)
        finally:
            if staging.exists(): shutil.rmtree(staging)


def receive_wifi(dest, notify=emit):
    from wifi import Receiver
    dest = Path(dest).expanduser().resolve(); dest.mkdir(parents=True, exist_ok=True)
    finished=False
    class ProjectReceiver(Receiver):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._transfer=threading.local()
        @property
        def current_file(self): return getattr(self._transfer,'path','')
        def _file(self,h,body):
            rel=str(h.get('path','')).replace('\\','/').strip('/')
            valid=rel and '..' not in rel.split('/') and not os.path.isabs(rel)
            if valid: self._transfer.path=rel
            super()._file(h,body)
            if not valid or not self._on or not (Path(self.stage)/rel).is_file(): return
            from app2.transfer.incoming import file_identity
            name,scan=file_identity(rel)
            if not name: return
            notify('received_file',name=name,scan=scan,relpath=rel,stage=self.stage,
                   file_bytes=(Path(self.stage)/rel).stat().st_size)
    def event(kind,info):
        nonlocal finished
        if kind == 'done': finished = True
        if kind=='progress': info=dict(info,relpath=receiver.current_file)
        notify(kind,**info,stage=receiver.stage)
    receiver=ProjectReceiver(str(dest),on_event=event)
    receiver.start()
    notify('listening', code=receiver.code, ip=receiver.ip, stage=receiver.stage)
    try:
        while not finished: time.sleep(.2)
    finally:
        receiver.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['mount', 'unmount', 'list', 'sizes', 'import', 'wifi'])
    parser.add_argument('--root'); parser.add_argument('--dest'); parser.add_argument('--mountpoint'); parser.add_argument('--token')
    parser.add_argument('--names-json', default='[]')
    parser.add_argument('--existing', choices=['skip', 'copy'], default='skip')
    parser.add_argument('--what-json', default='{}')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.command == 'mount': mount_device(args.mountpoint)
        elif args.command == 'unmount': unmount_device(args.mountpoint, args.token)
        elif args.command == 'list':
            for row in list_projects(args.root, args.dest): emit('project', **row)
        elif args.command == 'sizes': project_sizes(args.root)
        elif args.command == 'import':
            names = json.loads(args.names_json)
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names): raise ValueError('Select projects to import.')
            import_projects(args.root, args.dest, names, existing=args.existing, what=json.loads(args.what_json))
        elif args.command == 'wifi': receive_wifi(args.dest)
        emit('complete'); return 0
    except KeyboardInterrupt:
        emit('cancelled'); return 2
    except Exception as exc:
        emit('error', message=str(exc)); return 1


if __name__ == '__main__': sys.exit(main())
