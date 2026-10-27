"""The Export sheet says how big the file will be."""
import os, tempfile
import numpy as np
from app2.builder import export_model as X


def _ply(v, f, color=False):
    p = os.path.join(tempfile.mkdtemp(), "m.ply")
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n%selement face %d\nproperty list uchar int vertex_indices\nend_header\n" % (
        v, "property uchar red\nproperty uchar green\nproperty uchar blue\n" if color else "", f)
    open(p, "wb").write(head.encode() + b"\0" * 16)
    return p


def test_counts_from_the_header_only():
    assert X.ply_counts(_ply(1000, 2000)) == (1000, 2000, False)
    assert X.ply_counts(_ply(10, 20, True)) == (10, 20, True)
    assert X.ply_counts("/no/such/file.ply") == (0, 0, False)


def test_estimates_and_words():
    e = X.size_estimates(1_000_000, 2_000_000)
    assert e["stl"] == 84 + 50 * 2_000_000 and e["stl"] > 99e6
    assert e["obj"] > e["ply"] > 0 and e["3mf"] < e["obj"]
    assert X.human_bytes(53_400_000) == "53 MB" and X.human_bytes(1_250_000_000) == "1.2 GB" and X.human_bytes(800) == "800 bytes"


def test_scan_folders_total():
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "a")); open(os.path.join(d, "a", "x"), "wb").write(b"1" * 1500)
    open(os.path.join(d, "y"), "wb").write(b"1" * 500)
    assert X.folders_bytes([d]) == 2000


def test_sheet_shows_file_size_and_zip_total():
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.export_sheet import ExportDialog
    app = QApplication.instance() or QApplication([]); host = QWidget(); host.resize(900, 700); host.show()
    d = ExportDialog(host); d.present(["stl", "obj", "ply", "3mf"], "stl", can_zip=True); app.processEvents()
    d.set_files(X.size_estimates(1_000_000, 2_000_000), 19_000_000_000); app.processEvents()
    assert d.filesize.text() == "File: about 100 MB"
    d.pick("obj"); assert d.filesize.text().startswith("File: about") and "100 MB" not in d.filesize.text()
    d.zip.setChecked(True); app.processEvents()
    assert d.filesize.text().startswith("ZIP: about 19") and "scan data" in d.filesize.text()


def test_open_edges_counts_the_border_of_an_open_mesh():
    """A closed tetrahedron has no open edges; one triangle has three; an unreadable file gives None."""
    import struct
    def ply(V, F):
        p = os.path.join(tempfile.mkdtemp(), "m.ply")
        head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n" % (len(V), len(F))
        body = b"".join(struct.pack("<3f", *v) for v in V) + b"".join(struct.pack("<B3i", 3, *f) for f in F)
        open(p, "wb").write(head.encode() + body); return p
    V = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert X.open_edges(ply(V, [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)])) == 0
    assert X.open_edges(ply(V, [(0, 1, 2)])) == 3
    assert X.open_edges(ply(V, [(0, 1, 2), (0, 3, 1)])) == 4
    assert X.open_edges("/no/such/file.ply") is None


def test_export_sheet_says_closed_or_not():
    from PySide6.QtWidgets import QApplication, QWidget
    from app2.qt.export_sheet import ExportDialog
    app = QApplication.instance() or QApplication([]); host = QWidget(); host.resize(900, 700); host.show()
    d = ExportDialog(host); d.present(["stl"], "stl"); app.processEvents()
    assert d.closed.text().startswith("Closed surface: checking") and not d.watertight_b.isVisible()
    d.set_closed(449); app.processEvents()
    assert "449 open edges" in d.closed.text() and d.watertight_b.isVisible()
    d.set_closed(449, can_fix=False); assert not d.watertight_b.isVisible()
    d.set_closed(0); assert d.closed.text().startswith("Closed surface: yes") and not d.watertight_b.isVisible()
    fired = []; d.watertight.connect(lambda: fired.append(1)); d.set_closed(5); d.watertight_b.click(); assert fired
    host.close()
