# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inventory and installation for saved Wi-Fi share folders, without device access."""
import os
import json
from pathlib import Path
import shutil
import tempfile


def project_path(stage, name):
    if not name or Path(name).name != name or name in ('.', '..'): raise ValueError('Invalid project name.')
    stage = Path(stage)
    return (stage / 'Projects' if (stage / 'Projects').is_dir() else stage) / name


def file_identity(relpath):
    parts = Path(str(relpath).replace('\\', '/')).parts
    if parts and parts[0] == 'Projects': parts = parts[1:]
    if not parts or any(p in ('.', '..', '/') for p in parts): return None, None
    scan = parts[2] if len(parts) > 2 and parts[1] == 'data' else None
    return parts[0], scan


def side_names(project):
    try: return sorted(p.name for p in (Path(project) / 'data').iterdir() if p.is_dir() and not p.is_symlink())
    except OSError: return []


def project_identity(folder):
    """Read stable scanner identities, including indexes kept after a folder rename."""
    if Path(folder).is_symlink(): return None
    indexes = list(Path(folder).glob('*.revo'))
    if len(indexes) != 1: return None
    try: index = json.loads(indexes[0].read_text())
    except (OSError, ValueError): return None
    if not isinstance(index, dict) or not isinstance(index.get('guid'), str) or not index['guid'].strip(): return None
    sides = set(side_names(folder))
    nodes=index.get('nodes', [])
    if not isinstance(nodes,list): return None
    indexed = {str(n.get('guid')) for n in nodes
               if isinstance(n, dict) and n.get('type') == 2 and n.get('guid')}
    if indexed and not indexed.issubset(sides): return None
    for name in sides:
        path = Path(folder) / 'data' / name / 'property.rvproj'
        if path.exists():
            try: prop = json.loads(path.read_text())
            except (OSError, ValueError): return None
            if not isinstance(prop,dict): return None
            if prop.get('guid') and str(prop['guid']) != name: return None
    return index['guid'], sides


def same_project(source, target):
    a, b = project_identity(source), project_identity(target)
    return bool(a and b and a[0] == b[0] and a[1] & b[1])


def received_target(source, target):
    """Only a proven re-share updates a copy. Other collisions get a new folder."""
    target = Path(target)
    if not target.exists() or same_project(source, target): return target
    number = 1
    while True:
        candidate = target.with_name('%s (%d)' % (target.name, number))
        if not candidate.exists() or same_project(source,candidate): return candidate
        number += 1


def install_received(source, target):
    """Fill scanner files in a matching copy; keep local recipes and built outputs."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink(): raise ValueError('Project contains a link.')
    from app2.transfer.projects import _files
    files = _files(source)
    if any(str(rel).endswith('.part') for rel, _ in files): raise ValueError('The incoming project is incomplete.')
    indexes=list(source.glob('*.revo'))
    if len(indexes)>1: raise ValueError('The incoming project has multiple indexes.')
    index=indexes[0] if indexes else None
    if index is not None:
        try: metadata = json.loads(index.read_text())
        except (OSError, ValueError): raise ValueError('The incoming project index is incomplete.')
        if not isinstance(metadata, dict): raise ValueError('The incoming project index is incomplete.')
    if not target.exists():
        shutil.move(str(source), str(target)); return False
    if not same_project(source, target): raise ValueError('This is a different scan project. Use a separate copy.')
    # Scanner sides and scanner index are their own namespace. PC stages, labels,
    # and published model files are not replaced by a later scanner share.
    for rel, size in files:
        if rel.parts[0] != 'data' and (index is None or rel.name != index.name): continue
        origin, dst = source / rel, target / rel
        if index is not None and rel.name==index.name and rel.parts[0]!='data':
            dst=next(target.glob('*.revo'))
        if any(p.is_symlink() for p in [dst, *dst.parents] if p != target.parent):
            raise ValueError('Project contains a link.')
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.received-', dir=dst.parent)
        try:
            with os.fdopen(fd, 'wb') as out, origin.open('rb') as inp:
                shutil.copyfileobj(inp, out); out.flush(); os.fsync(out.fileno())
            if Path(tmp).stat().st_size != size: raise ValueError('A received file changed during installation.')
            shutil.copystat(origin, tmp); os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    shutil.rmtree(source)
    return True
