# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit distance calibrations from scanner exports, with saved provenance.

A distance preset cannot identify physical optics. Far needs an explicitly
selected export; neither focal length nor an undocumented index selects it.
"""
from pathlib import Path
import hashlib
import json
import shutil
import struct
import numpy as np


class CalibrationUnavailable(ValueError):
    pass


def load(folder, mode, size=(800, 600)):
    if mode not in ('near', 'far'):
        raise CalibrationUnavailable('Choose Small part or Large part calibration.')
    folder = Path(folder)
    try:
        raw = (folder / 'Pl.bin').read_bytes()
        cw, ch = struct.unpack_from('<HH', raw)
        matrix = np.asarray(struct.unpack_from('<9f', raw, 4)).reshape(3, 3)
    except (OSError, struct.error):
        raise CalibrationUnavailable('Large part camera not set up: scan anything with Large part on the scanner, then import it (Settings, Camera setup).' if mode == 'far'
                                     else 'Camera calibration is unavailable.') from None
    if len(raw) != 40 or min(cw, ch) <= 0 or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise CalibrationUnavailable('The calibration file is invalid. Choose a scanner export.')
    width, height = size
    if min(width, height) <= 0:
        raise CalibrationUnavailable('The depth frame size is unknown.')
    intrinsic = (matrix[0, 0] * width / cw, matrix[1, 1] * height / ch,
                 matrix[0, 2] * width / cw, matrix[1, 2] * height / ch)
    try:
        prop = json.loads((folder.parent / 'property.rvproj').read_text())
    except (OSError, ValueError):
        prop = {}
    source = dict(kind='selected scanner export', declared_mode=mode,
                  source_side=prop.get('guid', folder.parent.name),
                  calibration_sha256=hashlib.sha256(raw).hexdigest(),
                  calibration_size=[cw, ch], depth_size=list(size),
                  hardware_mode_verified=False)
    profile=json.loads(Path(__file__).with_name('camera_profiles.json').read_text())['miraco']
    source['working_distance_mm']=profile['working_distance_mm'][mode]
    source['working_distance_source']=profile['source']
    serial = prop.get('scan_param', {}).get('scanner_sn')
    if serial:
        source['scanner_identity_sha256'] = hashlib.sha256(str(serial).encode()).hexdigest()
    saved = folder / 'calibration-source.json'
    if saved.exists():
        try:
            provenance = json.loads(saved.read_text())
        except (OSError, ValueError):
            raise CalibrationUnavailable('Calibration source information is invalid.') from None
        if provenance.get('declared_mode') != mode or provenance.get('calibration_sha256') != source['calibration_sha256']:
            raise CalibrationUnavailable('Calibration source does not match this mode or file.')
        source.update(provenance)
        source['depth_size'] = list(size)
    source['working_distance_mm']=profile['working_distance_mm'][mode]
    source['working_distance_source']=profile['source']
    source['distance_support_kind']='saved-frame observations, not optical limits'
    return dict(folder=str(folder), intrinsics=tuple(float(x) for x in intrinsic), source=source)


def copy_parameters(calibration, target):
    """Copy the complete selected camera bundle, not just its depth intrinsics."""
    origin, target = Path(calibration['folder']), Path(target)
    if origin.resolve() == target.resolve():
        return
    files = list(origin.iterdir())
    if any(file.is_symlink() for file in files):
        raise CalibrationUnavailable('Calibration contains a link.')
    snapshot={file.name:file.read_bytes() for file in files if file.is_file() and file.suffix=='.bin'}
    if hashlib.sha256(snapshot.get('Pl.bin',b'')).hexdigest()!=calibration['source']['calibration_sha256']:
        raise CalibrationUnavailable('The calibration changed while it was being copied.')
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob('*.bin'):
        old.unlink()
    for name,content in snapshot.items():
        (target/name).write_bytes(content)
    (target / 'calibration-source.json').write_text(json.dumps(calibration['source'], indent=2) + '\n')


def store(folder, config_folder, mode, size=(800, 600)):
    selected = load(folder, mode, size)
    target = Path(config_folder) / 'calibrations' / mode / 'param'
    copy_parameters(selected, target)
    return load(target, mode, size)


def store_far(folder, config_folder, size=(800, 600)):
    return store(folder, config_folder, 'far', size)
