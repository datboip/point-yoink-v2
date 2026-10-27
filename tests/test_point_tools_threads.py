"""Running the cleanup's chunks side by side gives exactly the same points as one at a time, and never asks for more threads than memory allows."""
import os
import numpy as np
from scanner.stages.tools import point_tools as PT


def cloud(n=24000, seed=3):
    r = np.random.default_rng(seed); P = np.column_stack([r.uniform(-20, 20, n), r.uniform(-20, 20, n), r.normal(0, .15, n)])
    N = np.tile([0.0, 0.0, 1.0], (n, 1)) + r.normal(0, .02, (n, 3)); N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, N, np.ones(n)


def test_threaded_equals_serial_exactly():
    P, N, W = cloud()
    a = PT.project_lr(P, N, W, radius=1.5, chunk=5000, threads=1, gpu=False)
    b = PT.project_lr(P, N, W, radius=1.5, chunk=5000, threads=4, gpu=False)
    for x, y in zip(a, b): assert np.array_equal(x, y)


def test_smooth_and_overlap_tools_unchanged_by_threads(monkeypatch):
    monkeypatch.setenv("POINTYOINK_GPU", "0")
    P, N, W = cloud(12000); d = {"P": P.astype(np.float32), "N": N.astype(np.float32), "W": W, "NF": np.ones(len(P)), "S": np.ones(len(P), np.uint64)}
    from scanner.stages import params as PR
    p = PR.resolve_tool("smooth", "points", {})
    monkeypatch.setenv("POINTYOINK_THREADS", "1"); one = PT.apply("smooth", d, p)[0]["P"]
    monkeypatch.setenv("POINTYOINK_THREADS", "4"); four = PT.apply("smooth", d, p)[0]["P"]
    assert np.array_equal(one, four)


def test_thread_budget_respects_the_request_cap_and_env(monkeypatch):
    monkeypatch.delenv("POINTYOINK_THREADS", raising=False); monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert PT.thread_budget(3) == 3 and PT.thread_budget(100) <= 8 and PT.thread_budget(0) >= 1
    monkeypatch.setenv("OMP_NUM_THREADS", "5"); assert PT.thread_budget() <= 5
    monkeypatch.setenv("POINTYOINK_THREADS", "2"); assert PT.thread_budget() <= 2


def test_gpu_overlap_merge_matches_cpu():
    import pytest
    from scanner.stages.tools import gpu
    if not gpu.available(): pytest.skip("no CUDA build of Open3D")
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 20, (20000, 3)); P[:, 2] = np.sin(P[:, 0] / 3) + rng.normal(0, 0.05, len(P))
    N = np.tile([0, 0, 1.0], (len(P), 1)); W = rng.uniform(0.5, 2, len(P))
    a = PT.project_lr(P, N, W, radius=1.5, gpu=False); b = PT.project_lr(P, N, W, radius=1.5)
    assert np.abs(a[0] - b[0]).max() < 0.02 and np.abs(a[1] - b[1]).max() < 0.01 and (a[2] == b[2]).mean() > 0.999
