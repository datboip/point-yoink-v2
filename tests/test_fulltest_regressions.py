"""Focused checks for export orientation, project cards, point fits, and Build status."""

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XMODIFIERS", "@im=none")
os.environ.setdefault("POINTYOINK_NO_DEVICE", "1")
os.environ.setdefault("POINTYOINK_CONFIG", "/tmp/pointyoink-fulltest-regressions-config.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))

import numpy as np
import trimesh

from app2.builder.export_model import orient_ply
from app2.project_preview import project_rows
from app2.qt.lineup_panel import PairPicks
from app2.qt.shell import MainWindow
from stages import plyio


def test_orient_export_rotates_about_center_and_sets_bottom_to_zero(tmp_path):
    mesh = trimesh.creation.box(extents=(2, 4, 6))
    vertices = np.asarray(mesh.vertices) + (20, -5, 8)
    source = tmp_path / "model.ply"
    plyio.write_mesh(str(source), vertices, mesh.faces)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)

    output = Path(orient_ply(str(source), rotation))
    try:
        turned, faces, _normals = plyio.read_mesh(str(output))
    finally:
        output.unlink()
    assert len(faces) == len(mesh.faces)
    assert np.allclose(np.mean(turned, axis=0)[:2], 0, atol=1e-5)
    assert np.isclose(turned[:, 2].min(), 0)
    assert np.allclose(np.ptp(turned, axis=0), (6, 4, 2), atol=1e-5)


def test_point_pairs_reject_a_mirrored_fit():
    moving = np.array([[0, 0, 0], [20, 0, 0], [0, 30, 0], [0, 0, 40]], float)
    base = moving @ np.diag([-1, 1, 1]) + (10, -5, 2)
    picks = PairPicks()
    for b, m in zip(base, moving):
        picks.pairs.append([list(b), list(m)])

    transform = picks.matrix()
    assert transform is not None
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)
    assert max(picks.residuals()) > 1


def test_project_card_ignores_derived_pictures(tmp_path):
    project = tmp_path / "ProjectLive09282026052140"
    (project / "data" / "side1").mkdir(parents=True)
    for name in ("actual.png", "model_needs.png", "model_compare.png", "model_plane.png"):
        (project / name).write_bytes(b"test")

    pictures = [Path(path).name for path in project_rows(tmp_path)[0]["pictures"]]
    assert "actual.png" in pictures
    assert not {"model_needs.png", "model_compare.png", "model_plane.png"} & set(pictures)


def test_header_pill_counts_stages_and_only_shows_in_build():
    class Field:
        def __init__(self, value=""):
            self.value = value
            self.tooltip = None
            self.enabled = None
            self.visible = None

        def text(self): return self.value
        def setText(self, value): self.value = value
        def setToolTip(self, value): self.tooltip = value
        def setEnabled(self, value): self.enabled = value
        def isEnabled(self): return True
        def setVisible(self, value): self.visible = value

    class Stub: pass

    message = "Side changed. Fused, Meshed and Coverage will rebuild (about 4 min)."
    window = Stub(); window.build_slot = object()
    window.stack = Stub(); window.stack.currentWidget = lambda: window.build_slot
    window.build = Stub(); window.build.vp = Stub(); window.build.vp.rebuild = Stub()
    window.build.vp.rebuild.text = Field(message)
    window.build.vp.rebuild.btn = Field()
    window.build.vp.rebuild.isHidden = lambda: False
    class HeaderStub:
        def __init__(self):
            self.pill_text = Field(); self.pill_btn = Field(); self.pill = Field()

        def set_rebuild_text(self, label, tooltip):
            self.pill_text.setText(label)
            self.pill_text.setToolTip(tooltip)
            self.pill.setToolTip(tooltip)

    window.header = HeaderStub()

    MainWindow._sync_rebuild_pill(window)
    assert window.header.pill_text.value == "3 to rebuild \u00b7 about 4 min"
    assert window.header.pill_text.tooltip == message
    assert window.header.pill_btn.enabled is True
    assert window.header.pill.visible is True

    window.stack.currentWidget = lambda: object()
    MainWindow._sync_rebuild_pill(window)
    assert window.header.pill.visible is False
