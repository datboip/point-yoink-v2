"""Export formats: STL, OBJ, PLY and 3MF each write a file trimesh reads back (colors kept in PLY and OBJ), the ZIP holds
the model and the project's side folders, the Export sheet remembers the last format, and the model's size reads
"W x D x H mm". Offscreen, no scanner, an isolated config; every file in temp folders."""
import os
import tempfile
import zipfile
from pathlib import Path

os.environ["POINTYOINK_CONFIG"] = str(Path(tempfile.mkdtemp(prefix="pointyoink-export-")) / "config.json")
os.environ["POINTYOINK_NO_DEVICE"] = "1"
os.environ["XMODIFIERS"] = "@im=none"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import trimesh

from app2.builder import export_model as X
from app2.builder.stagehost import StageHost


def box_ply(path, size=(10.0, 20.0, 30.0), colors=True):
    """A closed box (12 triangles), its corner at (5, -3, 2), as the Build step's binary mesh PLY."""
    from stages import plyio
    m = trimesh.creation.box(size); V = m.vertices + np.array(size) / 2 + np.array([5.0, -3.0, 2.0])
    C = np.tile(np.array([200, 40, 10], np.uint8), (len(V), 1)) if colors else None
    plyio.write_mesh(str(path), V, m.faces, colors=C); return str(path)


def project(root, sides=2):
    p = Path(root) / "ProjectLiveZ"
    for i in range(sides):
        cache = p / "data" / ("side%d" % i) / "cache"; cache.mkdir(parents=True)
        (cache / "pointyoink_track.json").write_text("{}"); (cache / "frame_000_0000.png").write_bytes(b"\x89PNG%d" % i)
    (p / "stages").mkdir(parents=True, exist_ok=True)
    return str(p)


@pytest.mark.parametrize("fmt", [k for k, _l in X.formats()])
def test_each_format_writes_a_file_trimesh_reads_back(tmp_path, fmt):
    src = box_ply(tmp_path / "model.ply"); out = str(tmp_path / ("out." + fmt))
    r = StageHost.h_export(None, src, out, fmt)               # the host's call, as the Build step makes it
    assert r["faces"] == 12 and r["bytes"] == os.path.getsize(out) > 0 and r["format"] == fmt
    assert r["dims"] == "10.0 x 20.0 x 30.0 mm"
    m = trimesh.load(out, force="mesh")
    assert len(m.faces) == 12
    assert np.allclose(m.bounds, [[5, -3, 2], [15, 17, 32]], atol=1e-3)
    if fmt in ("ply", "obj"):
        assert m.visual.kind == "vertex" and tuple(m.visual.vertex_colors[0][:3]) == (200, 40, 10)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_stl_and_3mf_are_offered_and_3mf_is_left_out_when_trimesh_cannot_write_it(monkeypatch):
    keys = [k for k, _l in X.formats()]
    assert keys[0] == "stl" and {"obj", "ply"} <= set(keys)
    monkeypatch.setitem(X._OK, "3mf", False)
    assert "3mf" not in [k for k, _l in X.formats()]
    with pytest.raises(ValueError): X.write_model("nothing.ply", "x.3mf", "3mf")


def test_no_triangles_is_an_error(tmp_path):
    from stages import plyio
    src = str(tmp_path / "pts.ply"); plyio.write_mesh(src, np.zeros((3, 3)), np.zeros((0, 3), np.int64))
    with pytest.raises(ValueError, match="no triangles"): X.write_model(src, str(tmp_path / "a.obj"), "obj")


def test_zip_holds_the_model_and_the_side_folders(tmp_path):
    root = project(tmp_path); src = box_ply(Path(root) / "stages" / "meshed.ply")
    out = str(tmp_path / "ProjectLiveZ_scan.zip")
    r = StageHost.h_export(None, src, out, "ply", root)
    assert r["sides"] == 2 and r["faces"] == 12 and r["bytes"] == os.path.getsize(out)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert "ProjectLiveZ_scan.ply" in names
        for i in range(2):
            assert "data/side%d/cache/pointyoink_track.json" % i in names
            assert z.read("data/side%d/cache/frame_000_0000.png" % i) == b"\x89PNG%d" % i
        assert not any(n.startswith("stages/") for n in names)            # the model only, not the build's files
        z.extract("ProjectLiveZ_scan.ply", tmp_path / "x")
    assert len(trimesh.load(tmp_path / "x" / "ProjectLiveZ_scan.ply", force="mesh").faces) == 12


def test_dims_of_a_known_box():
    assert X.dims_text(X.dims([[1, -2, 3], [13.34, 48, 10.06], [5, 5, 5]])) == "12.3 x 50.0 x 7.1 mm"
    assert X.dims_text((10, 20, 30)) == "10.0 x 20.0 x 30.0 mm"


def test_model_dims_reads_the_ply_on_the_host(tmp_path):
    src = box_ply(tmp_path / "m.ply", size=(42.0, 8.5, 101.3), colors=False)
    r = StageHost.h_model_dims(None, src)
    assert r["text"] == "42.0 x 8.5 x 101.3 mm"
    assert abs(r["w"] - 42) < 1e-4 and abs(r["d"] - 8.5) < 1e-4 and abs(r["h"] - 101.3) < 1e-3


def test_sheet_remembers_the_last_format_in_the_isolated_config(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt import theme
    from app2.qt.export_sheet import ExportDialog
    monkeypatch.setenv("POINTYOINK_CONFIG", str(tmp_path / "config.json"))
    a = QApplication.instance() or QApplication([]); theme.install(a)
    host = QWidget(); host.resize(900, 700)
    try:
        assert X.last_format() == "stl"                                     # nothing saved: STL
        d = ExportDialog(host); got = []; d.ok.connect(lambda f, z: got.append((f, z)))
        d.present(["stl", "obj", "ply"], X.last_format())
        assert d.choice() == ("stl", False) and d.fmt_b["3mf"].isHidden() and d.size.text() == "Size: measuring..."
        d.fmt_b["obj"].click(); d.zip.setChecked(True); d.ok_b.click()
        assert got == [("obj", True)]
        X.save_format(got[0][0])                                           # what the Build step does on Export
        assert (tmp_path / "export.json").exists() and X.last_format() == "obj"
        d.present(["stl", "obj", "ply"], X.last_format())
        assert d.choice() == ("obj", False)                                 # the format stays; the ZIP is asked each time
        d.set_dims("10.0 x 20.0 x 30.0 mm"); assert d.size.text() == "Size: 10.0 x 20.0 x 30.0 mm"
        d.present(["stl", "ply"], "obj"); assert d.choice()[0] == "stl"      # a format not offered falls back
        d.present(["stl"], "stl", can_zip=False); assert not d.zip.isEnabled()
        X.save_format("nope"); assert X.last_format() == "obj"                # an unknown format is not kept
    finally:
        host.deleteLater(); a.processEvents()
