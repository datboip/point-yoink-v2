import numpy as np
import pytest

from scanner.stages.tools import point_tools as pt


def cloud():
    x, y = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
    P = np.column_stack((x.ravel(), y.ravel(), 0.01 * np.sin(x.ravel())))
    N = np.tile([0., 0., 1.], (len(P), 1))
    return P, N, np.ones(len(P))


def test_forced_cpu_schedule_matches_individual_passes(monkeypatch):
    monkeypatch.setenv("POINTYOINK_GPU", "0")
    P, N, W = cloud(); passes = [dict(k=16, sig_n=sn) for sn in (.3, .2)]
    expectedP, expectedN = P, N; meds = []
    for p in passes:
        expectedP, expectedN, expectednn, expectedspread = pt.project_lr(expectedP, expectedN, W, gpu=False, **p)
        meds.append(round(float(np.median(expectedspread)), 4))
    got = pt.project_passes(P, N, W, passes, diagnostics=True)
    for actual, expected in zip(got[:4], (expectedP, expectedN, expectednn, expectedspread)):
        np.testing.assert_array_equal(actual, expected)
    assert got[4] == meds
    assert got[5]["gpu_resident"] is False
    assert got[5]["gpu_fallback"] is False


def test_cuda_reuses_index_when_positions_do_not_change(monkeypatch):
    from scanner.stages.tools import gpu, gpu_resident
    if not gpu.available(): pytest.skip("CUDA unavailable")
    monkeypatch.delenv("POINTYOINK_GPU", raising=False)
    P, N, W = cloud(); P[:, 2] = 0
    got = gpu_resident.project_many(P, N, W, [dict(k=16), dict(k=16)])
    np.testing.assert_array_equal(got[0], P)
    assert got[5]["gpu_index_builds"] == 1


def test_gpu_failure_restarts_cpu_from_input(monkeypatch):
    from scanner.stages.tools import gpu, gpu_resident
    monkeypatch.delenv("POINTYOINK_GPU", raising=False)
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu_resident, "project_many", lambda *args: (_ for _ in ()).throw(RuntimeError("allocation failed")))
    P, N, W = cloud(); original = P.copy(); passes = [dict(k=16)]
    expected = pt.project_lr(P, N, W, gpu=False, **passes[0])
    got = pt.project_passes(P, N, W, passes)
    np.testing.assert_array_equal(P, original)
    for actual, want in zip(got[:4], expected): np.testing.assert_array_equal(actual, want)
    assert got[5]["gpu_resident"] is False
    assert got[5]["gpu_fallback"] is True


@pytest.mark.parametrize("fallback", [False, True])
def test_cpu_fallback_is_retried_under_cuda_key(monkeypatch, fallback):
    from scanner.stages.tools import common, gpu
    monkeypatch.delenv("POINTYOINK_GPU", raising=False)
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(common, "run_tool_job", lambda ctx, tool: dict(outputs={}, stats={"gpu_fallback": fallback, "gpu_resident": not fallback}))
    result = pt.Overlap().execute(object())
    assert bool(result.get("transient")) == fallback


@pytest.mark.parametrize("expected,actual", [(True, False), (False, True)])
def test_child_backend_change_does_not_keep_output_under_wrong_key(monkeypatch, expected, actual):
    from scanner.stages.tools import common, gpu
    monkeypatch.delenv("POINTYOINK_GPU", raising=False)
    monkeypatch.setattr(gpu, "available", lambda: expected)
    monkeypatch.setattr(common, "run_tool_job", lambda ctx, tool: dict(outputs={}, stats={"gpu_resident": actual, "gpu_fallback": False}))
    assert pt.Overlap().execute(object())["transient"] is True
