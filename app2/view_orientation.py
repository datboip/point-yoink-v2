# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project-local view labels. This basis never changes mesh or exported coordinates."""
import json
from pathlib import Path
import numpy as np
from app2.project_preview import META_NAME, _atomic_write


def validate_basis(value):
    try: basis = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc: raise ValueError('Invalid view orientation.') from exc
    if (basis.shape != (3, 3) or not np.isfinite(basis).all() or
            not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(basis), 1., atol=1e-6)):
        raise ValueError('Invalid view orientation.')
    return basis.copy()


def load_orientation(project):
    try:
        metadata = json.loads((Path(project) / META_NAME).read_text())
        return validate_basis(metadata['view_orientation']['basis'])
    except (OSError, ValueError, KeyError, TypeError):
        return np.eye(3)


def save_orientation(project, basis):
    basis = validate_basis(basis)
    path = Path(project) / META_NAME
    metadata = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(metadata, dict): raise ValueError('Project details must be an object.')
    metadata['view_orientation'] = {'version': 1, 'basis': basis.tolist()}
    _atomic_write(path, (json.dumps(metadata, indent=2) + '\n').encode('utf-8'))


def basis_with_bed_up(up, basis=None):
    """Use the matched bed's upward normal as local Z, retaining the view's heading."""
    previous = np.eye(3) if basis is None else validate_basis(basis)
    z = np.asarray(up, dtype=float)
    if z.shape != (3,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-9:
        raise ValueError('Invalid bed direction.')
    z = z / np.linalg.norm(z)
    y = previous[1] - np.dot(previous[1], z) * z
    if np.linalg.norm(y) > .05:
        y /= np.linalg.norm(y); x = np.cross(y, z)
    else:
        x = previous[0] - np.dot(previous[0], z) * z
        x /= np.linalg.norm(x); y = np.cross(z, x)
    return validate_basis(np.stack((x, y, z)))


def basis_with_bed_up_and_front(up, front):
    """Local Z from the bed's upward normal and local +Y along `front`, the direction the scanner looked at the start
    of the side (so the side the scanner started on is the cube's Front, and Bottom is the bed). The user can still
    set any face from the cube's menu afterwards; that saved choice outranks this."""
    z = np.asarray(up, dtype=float); y = np.asarray(front, dtype=float)
    if z.shape != (3,) or y.shape != (3,) or not np.isfinite(z).all() or not np.isfinite(y).all() or np.linalg.norm(z) < 1e-9:
        raise ValueError('Invalid bed direction.')
    z = z / np.linalg.norm(z); y = y - np.dot(y, z) * z
    if np.linalg.norm(y) < .05: return basis_with_bed_up(z)
    y /= np.linalg.norm(y); x = np.cross(y, z)
    return validate_basis(np.stack((x, y, z)))


def orientation_prompt_pending(project):
    try:
        metadata = json.loads((Path(project) / META_NAME).read_text())
    except (OSError, ValueError): metadata = {}
    if not isinstance(metadata, dict): metadata = {}
    return not bool(metadata.get('orientation_prompt_seen', False)) and not (Path(project) / 'orient.json').exists()


def dismiss_orientation_prompt(project):
    path = Path(project) / META_NAME
    try: metadata = json.loads(path.read_text())
    except (OSError, ValueError): metadata = {}
    if not isinstance(metadata, dict): metadata = {}
    metadata['orientation_prompt_seen'] = True
    _atomic_write(path, (json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
