# SPDX-License-Identifier: AGPL-3.0-or-later
"""The points (Revo-style) path uses each side's depth limit, like the side fusion: a Far side keeps its far frames."""
import os, tempfile
from pathlib import Path
os.environ.setdefault('POINTYOINK_CONFIG', str(Path(tempfile.mkdtemp()) / 'config.json'))


def _side(tmp_path, profile):
    from app2.scan.scan_mode import write
    folder = tmp_path / 'data' / 'side'; (folder / 'cache').mkdir(parents=True); (folder / 'param').mkdir()
    (folder / 'cache' / 'pointyoink_track.json').write_text('{}')
    write(folder, profile)
    from stages.project import Project
    from stages.engine import Engine
    return Engine(Project.open(tmp_path))


def test_far_side_depth_reaches_the_fused_points_step(tmp_path):
    eng = _side(tmp_path, dict(mode='far', voxel_mm=.75, far=1000))
    assert eng.graph().by_id['fused'].leaf.get('max_depth') == {'side': 1000}


def test_near_side_keeps_the_fused_leaf_unchanged(tmp_path):
    eng = _side(tmp_path, dict(mode='near'))
    assert 'max_depth' not in eng.graph().by_id['fused'].leaf      # existing projects keep their fused fingerprint


def test_depth_limit_is_part_of_the_side_caches(tmp_path):
    from stages.side_fusion_cache import table_paths
    from stages.local_points_cache import local_paths
    sj = {'sides': [{'id': 'a', 'folder': 'a', 'T': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}
    (tmp_path / 'a' / 'cache').mkdir(parents=True); (tmp_path / 'a' / 'param').mkdir()
    (tmp_path / 'a' / 'cache' / 'pointyoink_register_pose.pose').write_bytes(b'0'); (tmp_path / 'a' / 'param' / 'Pl.bin').write_bytes(b'0'); stages = tmp_path / 'stages'; (stages / 'cache').mkdir(parents=True)
    p = dict(point_distance=.4, step=2, frames_per_side=300, edge_px=2, z0=0)
    for fn in (table_paths, local_paths):
        plain = fn(sj, str(tmp_path), str(stages), p, {})['a'][1]
        assert fn(sj, str(tmp_path), str(stages), p, {}, {})['a'][1] == plain
        assert fn(sj, str(tmp_path), str(stages), p, {}, {'a': 1000.0})['a'][1] != plain
