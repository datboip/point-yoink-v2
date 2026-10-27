"""Bounded classic CUDA frame preparation keeps input order and releases workers."""
import threading

import pytest

import fuse


@pytest.mark.parametrize("jobs", [[], [7], list(range(30))])
def test_results_keep_input_order(jobs):
    assert list(fuse._ordered_frame_results(jobs, lambda job: job * 3)) == [job * 3 for job in jobs]


def test_slow_first_frame_does_not_change_order():
    second_started = threading.Event()

    def load(job):
        if job == 0:
            assert second_started.wait(2), "second preparation worker did not start"
        elif job == 1:
            second_started.set()
        return job

    assert list(fuse._ordered_frame_results(range(8), load)) == list(range(8))


def test_submission_stays_bounded_until_consumer_advances():
    submitted = []

    def jobs():
        for job in range(100):
            submitted.append(job)
            yield job

    results = fuse._ordered_frame_results(jobs(), lambda job: job)
    try:
        assert next(results) == 0
        assert submitted == [0, 1, 2, 3]
        assert next(results) == 1
        assert submitted == [0, 1, 2, 3, 4]
    finally:
        results.close()
    assert submitted == [0, 1, 2, 3, 4]


def test_close_joins_owned_workers():
    threads = set()
    lock = threading.Lock()

    def load(job):
        with lock:
            threads.add(threading.current_thread())
        return job

    results = fuse._ordered_frame_results(range(100), load)
    assert next(results) == 0
    results.close()
    assert threads
    assert len(threads) <= 2
    assert all(not thread.is_alive() for thread in threads)


def test_failed_load_joins_owned_workers_and_propagates_error():
    threads = set()
    lock = threading.Lock()

    def load(job):
        with lock:
            threads.add(threading.current_thread())
        if job == 1:
            raise OSError("depth read failed")
        return job

    results = fuse._ordered_frame_results(range(100), load)
    with pytest.raises(OSError, match="depth read failed"):
        list(results)
    assert threads
    assert all(not thread.is_alive() for thread in threads)


def test_single_worker_prepares_only_when_requested_on_caller_thread():
    caller = threading.current_thread()
    prepared = []

    def load(job):
        assert threading.current_thread() is caller
        prepared.append(job)
        return job

    results = fuse._ordered_frame_results(range(5), load, workers=1)
    assert prepared == []
    assert next(results) == 0
    assert prepared == [0]
    results.close()
    assert prepared == [0]


def test_skipped_frame_keeps_its_position():
    assert list(fuse._ordered_frame_results(range(5), lambda job: None if job == 2 else job)) == [0, 1, None, 3, 4]


@pytest.fixture
def clear_thread_environment(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fuse.os, "sched_getaffinity", lambda pid: set(range(24)))


def test_preparation_uses_four_workers_on_a_big_pc_and_two_on_a_small_one(clear_thread_environment, monkeypatch):
    assert fuse._classic_prepare_workers() == 4                       # 24 threads: 4 (no gain past 4, measured)
    monkeypatch.setattr(fuse.os, "sched_getaffinity", lambda pid: set(range(8)))
    assert fuse._classic_prepare_workers() == 2


@pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"])
def test_explicit_one_thread_budget_disables_prefetch(clear_thread_environment, monkeypatch, name):
    monkeypatch.setenv(name, "1")
    assert fuse._classic_prepare_workers() == 1


def test_single_cpu_affinity_disables_prefetch(clear_thread_environment, monkeypatch):
    monkeypatch.setattr(fuse.os, "sched_getaffinity", lambda pid: {0})
    assert fuse._classic_prepare_workers() == 1


def test_nested_openmp_limit_uses_first_level(clear_thread_environment, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1,4")
    assert fuse._classic_prepare_workers() == 1


def test_malformed_thread_limit_does_not_fail_loading(clear_thread_environment, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "invalid")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "0")
    assert fuse._classic_prepare_workers() == 4


def test_affinity_unavailable_uses_cpu_count(clear_thread_environment, monkeypatch):
    def unavailable(pid):
        raise OSError("unavailable")

    monkeypatch.setattr(fuse.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(fuse.os, "cpu_count", lambda: 1)
    assert fuse._classic_prepare_workers() == 1


def _capture_integration(monkeypatch, tmp_path, specs, gpu, frame_filter):
    """Keep real CPU preparation; replace only Open3D integration and mesh I/O."""
    import sys
    from types import SimpleNamespace
    from pathlib import Path
    import numpy as np

    captured = []

    class Tensor:
        def __init__(self, values, dtype=None):
            self.array = np.array(values, copy=True)

        @property
        def shape(self):return self.array.shape
        def numpy(self):return self.array.copy()
        def __getitem__(self,key):return Tensor(self.array[key])

    class Image:
        def __init__(self, values):
            self.array = np.array(values.array if isinstance(values, Tensor) else values, copy=True)

        @property
        def shape(self):return self.array.shape
        def numpy(self):return self.array.copy()
        def __getitem__(self,key):return Tensor(self.array[key])
        def to(self, device):return self

    class Intrinsic:
        def __init__(self, width, height, fx, fy, cx, cy):
            self.intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    class Mesh:
        vertices = [0, 1, 2]
        triangles = [0]

        def to_legacy(self):
            return self

        def compute_vertex_normals(self):
            pass

        def scale(self, factor, center):
            pass

    def capture(depth, intrinsic, extrinsic):
        captured.append((depth.array.copy(), intrinsic.copy(), extrinsic.copy()))

    class Grid:
        def __init__(self,**kwargs):
            self.device=kwargs.get('device');self.capacity=kwargs.get('block_count',1)
        def hashmap(self):return SimpleNamespace(capacity=lambda:self.capacity)
        def compute_unique_block_coordinates(self,*args):
            if self.device=='CUDA:0':BLOCK_CALLS['claim'].append(args[-1])
            return Tensor([[0,0,0]])

        def integrate(self, coords, depth, intrinsic, extrinsic, *args):
            BLOCK_CALLS["write"].append(args[-1])
            capture(depth, intrinsic.array, extrinsic.array)

        def extract_triangle_mesh(self, **kwargs):
            return Mesh()

    class Volume:
        def __init__(self, **kwargs):
            pass

        def integrate(self, depth, intrinsic, extrinsic):
            capture(depth, intrinsic.intrinsic_matrix, extrinsic)

        def extract_triangle_mesh(self):
            return Mesh()

    def write_mesh(path, mesh, **kwargs):
        Path(path).write_bytes(b"synthetic mesh output " * 8)
        return True

    fake = SimpleNamespace(
        core=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:None,release_cache=lambda:None), Device=str,
                             Tensor=Tensor, float32="float32", float64="float64"),
        camera=SimpleNamespace(PinholeCameraIntrinsic=Intrinsic),
        t=SimpleNamespace(geometry=SimpleNamespace(VoxelBlockGrid=Grid, Image=Image)),
        geometry=SimpleNamespace(Image=Image, RGBDImage=SimpleNamespace(
            create_from_color_and_depth=lambda color, depth, **kwargs: depth)),
        pipelines=SimpleNamespace(integration=SimpleNamespace(
            ScalableTSDFVolume=Volume, TSDFVolumeColorType=SimpleNamespace(NoColor=0))),
        io=SimpleNamespace(write_triangle_mesh=write_mesh),
    )
    monkeypatch.setitem(sys.modules,"open3d",fake)
    from scanner import fusion_memory
    monkeypatch.setattr(fusion_memory,'free_bytes',lambda device:8*1024**3)
    monkeypatch.setattr(fusion_memory.Peak,'start',lambda self:self)
    # This fixture captures frame inputs; native mesh ordering has its own tests.
    monkeypatch.setattr(fuse, "_canonical_mesh", lambda mesh: mesh)
    monkeypatch.setattr(fuse, "apply_mem_cap", lambda: None)
    monkeypatch.setattr(fuse, "_classic_prepare_workers", lambda: 2)
    argv = ["fuse.py", "--out", str(tmp_path / "capture.ply"), "--filter", str(frame_filter)]
    if gpu:
        argv.append("--gpu")
    for spec in specs:
        argv.extend(("--set", spec))
    monkeypatch.setattr(sys, "argv", argv)
    assert fuse.main() == 0
    return captured


BLOCK_CALLS = {"claim": [], "write": []}


def _synthetic_frame_sets(tmp_path):
    import json
    import struct
    import numpy as np

    sets = []
    for side, (width, height) in enumerate(((8, 6), (6, 4), (8, 6))):
        folder = tmp_path / str(side)
        cache = folder / "cache"
        cache.mkdir(parents=True)
        (folder / "property.rvproj").write_text(json.dumps({
            "scan_param": {"depth_width": width, "depth_height": height}}))
        calib = folder / "Pl.bin"
        calib.write_bytes(struct.pack("<10f", 0, 400, 0, 400, 0, 400, 300, 0, 0, 1))
        poses = []
        for frame in range(5):
            depth = np.full((height, width), 1000 + frame * 10 + side * 20, np.uint16)
            depth[0, 0] = 6500  # max-depth rejection, followed by normal filtering
            if side == 0 and frame == 3:
                depth = depth.ravel()[:-1]  # malformed dimensions are skipped
            name = cache / ("frame_0_%04d" % frame)
            name.with_suffix(".dph").write_bytes(depth.tobytes())
            pose = np.eye(4)
            pose[:3, 3] = [frame * 2 + side, side * 3, 0]
            if not (side == 0 and frame == 4):
                name.with_suffix(".inf").write_bytes(b"\0" * 16 + struct.pack("<16d", *pose.ravel()))
            if frame != 1:
                registered = pose.copy()
                registered[0, 3] += 1
                poses.append(struct.pack("<ii16d", 0, frame, *registered.ravel()))
        if side < 2:
            name = "pointyoink_register_pose.pose" if side == 0 else "global_register_pose.pose"
            (cache / name).write_bytes(struct.pack("<i", len(poses)) + b"".join(poses))
        transform = np.eye(4)
        transform[:3, 3] = [side * 13, -side * 7, side * 2]
        transform_path = folder / "transform.json"
        transform_path.write_text(json.dumps({"matrix": transform.tolist()}))
        plane = folder / "plane.json"
        plane.write_text(json.dumps({"n": [1, 0, 0], "d": -10, "margin": 0, "keep_above": True}))
        deletes = folder / "deletes.json"
        deletes.write_text(json.dumps({"stages": str(folder), "ops": [{
            "on": True, "frame": "side", "shape": "box",
            "box": {"min": [-1000, -1000, -1000], "max": [1000, 0, 0]}}]}))
        sets.append((str(cache), str(calib), str(transform_path), str(plane), str(deletes)))
    return sets


@pytest.mark.parametrize("frame_filter", [0, 3])
def test_gpu_preparation_exactly_matches_serial_cpu_with_mixed_sizes_and_cuts(monkeypatch, tmp_path, frame_filter):
    import numpy as np

    sets = _synthetic_frame_sets(tmp_path)
    plain = [",".join(spec[:3]) for spec in sets]
    plane_only = [",".join(spec[:4]) for spec in sets]
    delete_only = [",".join((*spec[:3], "", spec[4])) for spec in sets]
    cut = [",".join(spec) for spec in sets]
    original = _capture_integration(monkeypatch, tmp_path, plain, False, frame_filter)
    plane = _capture_integration(monkeypatch, tmp_path, plane_only, False, frame_filter)
    deleted = _capture_integration(monkeypatch, tmp_path, delete_only, False, frame_filter)
    serial = _capture_integration(monkeypatch, tmp_path, cut, False, frame_filter)
    parallel = _capture_integration(monkeypatch, tmp_path, cut, True, frame_filter)
    # Side 0 skips one omitted own pose, one malformed frame, and one missing .inf.
    assert len(serial) == len(parallel) == 12
    assert [item[0].shape for item in serial] == [(6, 8)] * 2 + [(4, 6)] * 5 + [(6, 8)] * 5
    assert any(np.count_nonzero(b[0]) < np.count_nonzero(a[0]) for a, b in zip(original, plane))
    assert any(np.count_nonzero(b[0]) < np.count_nonzero(a[0]) for a, b in zip(original, deleted))
    assert sum(np.count_nonzero(item[0]) for item in serial) > 0
    for expected, actual in zip(serial, parallel):
        for serial_array, parallel_array in zip(expected, actual):
            np.testing.assert_array_equal(parallel_array, serial_array)


def test_gpu_frames_claim_blocks_wider_than_the_band_they_write(monkeypatch, tmp_path):
    """Claimed at the written band, voxels near a block border missed frames: a step on every border (rows of triangles)."""
    BLOCK_CALLS["claim"].clear(); BLOCK_CALLS["write"].clear()
    sets = _synthetic_frame_sets(tmp_path)
    _capture_integration(monkeypatch, tmp_path, [",".join(spec[:3]) for spec in sets], True, 0)
    assert BLOCK_CALLS["write"] and len(BLOCK_CALLS["claim"]) == len(BLOCK_CALLS["write"])
    assert all(c >= 3 * w for c, w in zip(BLOCK_CALLS["claim"], BLOCK_CALLS["write"]))
