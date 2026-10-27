"""Fused (points) keys on the cell merge module, not on the cleanup tools; the overlap and smooth tools key on the backend (CUDA or CPU)."""
from scanner.stages import methods as M
from scanner.stages.tools import point_tools as PT


def test_fused_points_does_not_depend_on_the_cleanup_code():
    from scanner.stages.methods.fuse_points import FusedPoints
    files = [n for n, _p in FusedPoints().code({})[0]]
    assert "stages/tools/merge_cells.py" in files and "stages/tools/point_tools.py" not in files


def test_cleanup_tools_carry_the_backend_in_their_key(monkeypatch):
    monkeypatch.setenv("POINTYOINK_GPU", "0")
    assert "backend=cpu" in PT.Overlap().code({})[1] and "backend=cpu" in PT.Smooth().code({})[1]
    assert not any(l.startswith("backend=") for l in PT.Isolation().code({})[1])
    monkeypatch.delenv("POINTYOINK_GPU")
    from scanner.stages.tools import gpu
    want = "backend=cuda" if gpu.available() else "backend=cpu"
    assert want in PT.Overlap().code({})[1]
