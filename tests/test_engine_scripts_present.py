import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_every_script_the_engine_runs_is_in_the_repo():
    from scanner.stages.methods import ROOT_SCRIPTS, script
    missing = [n for n in ROOT_SCRIPTS if not os.path.exists(script(n))]
    assert not missing, "the engine runs these but they are not in the repo: %s" % missing
    named = set()
    for folder in ('app2', 'scanner'):
        for path in (ROOT / folder).rglob('*.py'):
            named.update(re.findall(r'script\(["\']([\w.]+\.py)["\']\)', path.read_text(errors='ignore')))
    assert not [n for n in sorted(named) if not os.path.exists(script(n))]


def test_simplify_runs_end_to_end(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    pytest.importorskip('fast_simplification')
    sphere = trimesh.creation.icosphere(subdivisions=4)
    src, out = tmp_path / 'in.ply', tmp_path / 'out.ply'
    sphere.export(src)
    result = subprocess.run([sys.executable, str(ROOT / 'process.py'), str(src), str(out), '--clean', '--isolation-rate', '0',
                             '--smooth-times', '0', '--simplify-pct', '25'], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-600:]
    small = trimesh.load(out)
    assert 0.1 * len(sphere.faces) < len(small.faces) < 0.5 * len(sphere.faces)
