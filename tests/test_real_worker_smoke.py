"""Saved Run1 frames through the real engine, subprocess workers, and project host."""
import importlib.util
import json
from pathlib import Path
import queue
import time
import zipfile

import pytest

pytestmark = [pytest.mark.real_worker, pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("open3d", "scipy", "cv2", "trimesh", "PIL")),
    reason="Real-worker smoke needs Open3D, SciPy, OpenCV, trimesh, and Pillow.")]


def test_saved_frames_build_edit_undo_redo_publish_and_reopen(tmp_path, monkeypatch):
    from app2.builder.stagehost import StageHost
    from stages import plyio
    from stages.keys import file_digest_raw, output_digest

    # Fused TSDF requests CUDA by default; hide devices from its real child process.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path / "saved-run1"
    root.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "real_worker_run1.zip"
    with zipfile.ZipFile(fixture) as archive:
        archive.extractall(root)
    messages = queue.Queue()
    host = None
    times = {}
    snapshot = None

    def emit(name, value):
        messages.put((name, value))

    def wait_result(ticket, deadline=120):
        nonlocal snapshot
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            name, value = messages.get(timeout=max(0.01, end-time.monotonic()))
            if name == "stages_status":
                snapshot = value
                assert not value.error, value.error
            if name == "stages_result" and value["ticket"] == ticket:
                assert value["ok"], value["error"]
                return value["value"]
        pytest.fail("The private project host did not finish its call.")

    def call(name, *args, **kwargs):
        started = time.monotonic()
        result = wait_result(host.call(name, *args, **kwargs))
        times.setdefault(name, []).append(round(time.monotonic()-started, 3))
        if name == "run":
            assert result["ok"], result.get("error") or result.get("failed")
        return result

    def get_snapshot():
        nonlocal snapshot
        while True:
            try: messages.get_nowait()
            except queue.Empty: break
        host.status()
        end = time.monotonic()+30
        while time.monotonic() < end:
            name, value = messages.get(timeout=max(0.01, end-time.monotonic()))
            if name == "stages_status":
                assert not value.error, value.error
                snapshot = value
                return value
        pytest.fail("The private project host did not send status.")

    def published_bytes():
        data = (root / "model.ply").read_bytes()
        publication = json.loads((root / "stages" / "published.json").read_text())
        assert publication["digest"] == output_digest({"mesh": file_digest_raw(root / "model.ply")})
        vertices, faces, _ = plyio.read_mesh(str(root / "model.ply"))
        assert len(vertices) and len(faces)
        return data, vertices, faces

    try:
        host = StageHost(str(root), emit, mem_gb=4)
        assert host.opened.wait(30) and not host.open_error
        call("set", {"side.fuse.voxel": 2.0, "side.fuse.gpu": False,
                     "side.platter.method": "auto",
                     "fused.method": "tsdf",
                     "fused.color": "off", "fused.point_distance": 2.0,
                     "fused.frames_per_side": 8, "meshed.method": "as_fused"})
        call("clear_ops", "fused.tsdf")
        call("run", publish=True)
        initial, vertices, faces = published_bytes()
        before = get_snapshot()
        baseline_cursor = before.cursor
        fusion_nodes = [n for g in before.groups for n in g["nodes"]
                        if n["id"] == "fused" or n["id"].startswith("side.fuse@")]
        assert len(fusion_nodes) == 2
        assert all(n["stats"]["backend"] == "CPU:0" for n in fusion_nodes)
        assert not before.chains.get("meshed")
        # The platter stage must dispatch the actual no-base worker successfully.
        jobs = [(p, json.loads(p.read_text())) for p in (root / "stages" / "nodes").rglob("job_*.json")]
        observers = [(p, job) for p, job in jobs if job.get("fn", "").endswith(":observe_no_base")]
        assert observers, "The saved-frame no-base observer did not run."
        for path, job in observers:
            # Store moves the worker scratch directory after the result is written.
            result = path.with_name(Path(job["result"]).name)
            evidence = json.loads(result.read_text())
            assert isinstance(evidence["no_base"], bool) and evidence["frames"] == 8
        platter = next(n for g in before.groups for n in g["nodes"] if n["id"].startswith("side.platter@"))
        assert not platter.get("retry"), platter
        assert "ModuleNotFoundError" not in "\n".join(p.read_text() for p in (root / "stages" / "nodes").rglob("log.txt"))
        centres = vertices[faces].mean(axis=1)
        selected = centres[:max(1, min(8, len(centres)//10))]
        change = call("h_mask_delete", "meshed", selected.tolist(), len(selected))
        assert change["got"] == change["want"] == len(selected) and change["got"] > 0
        call("run", publish=True)
        edited, _, edited_faces = published_bytes()
        assert edited != initial and len(edited_faces) < len(faces)
        applied = get_snapshot()
        ops = applied.chains["meshed"]
        assert len(ops) == 1 and ops[0][2]["tool"] == "delete"
        assert applied.cursor == baseline_cursor+1
        call("undo"); call("run", publish=True)
        assert published_bytes()[0] == initial
        undone = get_snapshot()
        assert undone.cursor == baseline_cursor and not undone.chains.get("meshed")
        call("redo"); call("run", publish=True)
        assert published_bytes()[0] == edited
        redone = get_snapshot()
        assert redone.cursor == baseline_cursor+1 and redone.chains["meshed"] == ops
        history = redone.entries
        host.close(timeout=30)
        assert not host.thread.is_alive()
        host = StageHost(str(root), emit, mem_gb=4)
        assert host.opened.wait(30) and not host.open_error
        reopened = get_snapshot()
        assert reopened.cursor == redone.cursor and reopened.entries == history
        assert reopened.chains["meshed"] == ops
        assert published_bytes()[0] == edited
        assert Path(reopened.out(reopened.final("meshed"), "mesh")).read_bytes() == edited
        print("REAL_WORKER_SMOKE "+json.dumps(dict(times=times, fixture_bytes=fixture.stat().st_size,
              frames=8, initial_faces=len(faces), edited_faces=len(edited_faces),
              undo_exact=True, redo_exact=True, reopen_exact=True, history_cursor=reopened.cursor)))
    finally:
        if host is not None:
            host.close(timeout=30)
