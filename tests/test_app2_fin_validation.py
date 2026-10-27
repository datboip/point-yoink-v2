"""Both fin ends attach to real surfaces and invalid geometry never reaches union."""
import json
import os
import tempfile
from pathlib import Path

os.environ['POINTYOINK_CONFIG'] = str(Path(tempfile.mkdtemp()) / 'config.json')
import app2
import numpy as np
import pytest
import trimesh
from stages.tools.fin import join_fin, nearest_surface, rib_geometry


def params(a=(-6, 0, 2), b=(6, 0, 2), normals=((0, 0, 1), (0, 0, 1)), **extra):
    return dict(anchors=json.dumps([[list(a), list(normals[0])], [list(b), list(normals[1])]]),
                thickness=2., reach=8., shape='triangle', **extra)


def l_part():
    profile = np.array([[0, 0], [20, 0], [20, 2], [2, 2], [2, 20], [0, 20]])
    v = np.array([[x, y, z] for y in (-5, 5) for x, z in profile])
    cap = [(0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5)]
    f = cap + [tuple(i + 6 for i in reversed(face)) for face in cap]
    for i in range(6):
        j = (i + 1) % 6
        f.extend([(i, i + 6, j), (j, i + 6, j + 6)])
    mesh = trimesh.Trimesh(v, f); mesh.fix_normals()
    assert mesh.is_volume
    return mesh


@pytest.mark.parametrize('shape', ['triangle', 'rounded'])
def test_fin_joins_both_faces_of_thin_l_part(shape):
    body = l_part()
    p = params((12, 0, 2), (2, 0, 12), ((0, 0, 1), (1, 0, 0)))
    p['shape'] = shape
    v, f = join_fin(body.vertices, body.faces, p)
    out = trimesh.Trimesh(v, f)
    assert out.is_volume and len(out.split()) == 1
    assert out.volume > body.volume


@pytest.mark.parametrize('point', [(25, 0, 2), (6, 0, 1), (6, 0, 3)])
def test_rejects_second_anchor_away_from_surface(point):
    body = trimesh.creation.box(extents=[20, 20, 4])
    with pytest.raises(ValueError, match='Both attachment spots'):
        join_fin(body.vertices, body.faces, params(b=point))


@pytest.mark.parametrize('key,value', [('thickness', 0), ('thickness', float('nan')),
                                     ('reach', -1), ('shape', 'unknown'), ('anchors', 'bad json')])
def test_invalid_parameters(key, value):
    p = params(); p[key] = value
    with pytest.raises(ValueError): rib_geometry(p)


def test_normals_are_normalized_and_opposite_normals_rejected():
    v, f = rib_geometry(params(normals=((0, 0, 10), (0, 0, 5))))
    expected, _ = rib_geometry(params())
    assert np.allclose(v, expected)
    with pytest.raises(ValueError, match='opposite'):
        rib_geometry(params(normals=((0, 0, 1), (0, 0, -1))))
    with pytest.raises(ValueError, match='surface directions'):
        rib_geometry(params(normals=((0, 0, 0), (0, 0, 1))))


def test_nearest_surface_includes_large_triangles_with_distant_centers():
    vertices = [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
    faces = [[0, 1, 2], [3, 4, 5]]
    closest, distance, index = nearest_surface(vertices, faces, [[.1, .1, .01]])
    assert index.tolist() == [0]
    assert distance[0] == pytest.approx(.01)
    assert np.allclose(closest, [[.1, .1, 0]])


def test_async_surface_pick_and_stale_result_are_guarded():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from types import SimpleNamespace
    from PySide6.QtCore import QObject, Signal, QThreadPool
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.fin_tool import FinTool
    app = QApplication.instance() or QApplication([])
    class View(QObject):
        surface_picked = Signal(object)
        arrow_move = Signal(int, float, bool)
        gizmo = None
        def set_tool(self, value): self.tool = value
        def set_gizmo(self, centre, radius=None, rings=True, arrows=True): self.gizmo = dict(c=centre, r=radius)
    page = QWidget(); page.gl = True; page.stage = 'meshed'
    view = View(); view.tool = 'fin'; view.edit_key = 'body'
    source = dict(kind='mesh', V=np.array([[0.,0,0], [100,0,0], [0,100,0]]), F=np.array([[0,1,2]]))
    view.items = {'body': source}
    messages = []
    page.panel = {'meshed': SimpleNamespace(bodies={'Add fin': SimpleNamespace(set_msg=messages.append)})}
    page.vp = SimpleNamespace(view=view)
    tool = FinTool(page); tool.source = source
    tool.pick([.1, .1, .01])
    assert tool.pending and not tool.anchors
    QThreadPool.globalInstance().waitForDone(5000); app.processEvents()
    assert len(tool.anchors) == 1 and np.allclose(tool.anchors[0][0], [.1,.1,0])
    assert not tool.pending and view.tool == 'fin'
    tool.pick([.2, .2, .01]); tool.clear()
    QThreadPool.globalInstance().waitForDone(5000); app.processEvents()
    assert tool.anchors == [] and tool.source is None
    page.deleteLater(); app.processEvents()


def test_legacy_directions_along_end_line_have_a_specific_hint():
    with pytest.raises(ValueError, match='attachment directions'):
        rib_geometry(params(normals=((1, 0, 0), (1, 0, 0))))


def test_new_plane_points_on_end_line_request_another_plane_spot():
    p = params()
    anchors = json.loads(p['anchors'])
    anchors.append([[0, 0, 2], [0, 0, 1]])
    p['anchors'] = json.dumps(anchors)
    with pytest.raises(ValueError, match='plane spots lie on the end line'):
        rib_geometry(p)
