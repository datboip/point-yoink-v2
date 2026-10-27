"""Fusion reports actual execution devices without inferring CUDA from a request."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scanner"))

from stages import params as PR
from stages.methods.side import SideFuseTsdf
from stages.methods.fuse_tsdf import FusedTsdf
from stages.methods.fuse_plus import FusedPlus
from stages.runner import StepError, Cancelled


WARNING = "GPU was requested, but fusion ran on the CPU. Surface coverage can differ. Review the result."


def events(*devices):
    return ["STAGE device " + json.dumps({"device": device}) for device in devices] + [
        'STAGE integrate {"done": 3, "total": 3}',
        'STAGE done {"faces": 12, "verts": 8}',
    ]


class Context:
    def __init__(self, tmp_path, method, responses, gpu=True, drop_enclosed=False):
        self.params = PR.resolve({}, method.node, method.name)
        if method.node == "side.fuse":
            self.params["gpu"] = gpu
        if method.name == "plus":
            self.params.update(drop_enclosed=drop_enclosed, fallback=True)
        self.out_dir = str(tmp_path / "out")
        Path(self.out_dir).mkdir()
        self.project = SimpleNamespace(root=str(tmp_path), stages=str(tmp_path / "stages"),
                                       folder=lambda side: str(tmp_path / side))
        self.step = SimpleNamespace(side="s1", n=1, leaf={})
        self.n_sides = 1
        self.fuse_cap = 3
        sides = tmp_path / "sides.json"
        sides.write_text(json.dumps({"sides": [{"id": "s1", "frames": "s1/cache", "calib": "s1/param/Pl.bin",
                                                "T": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}))
        self.inputs = {"lineup": SimpleNamespace(files={"sides": str(sides)})}
        self.responses = list(responses)
        self.warns = []
        self.calls = []
        self.progress_events = []
        self.legacy_events = []

    def warn(self, text):
        self.warns.append(text)

    def progress(self, *args):
        self.progress_events.append(args)

    def legacy(self, *args, **kwargs):
        self.legacy_events.append((args, kwargs))

    def call(self, args, on_line=None, **kwargs):
        self.calls.append((list(args), kwargs))
        lines, error = self.responses.pop(0)
        for line in lines:
            if on_line:
                on_line(line)
        if error:
            raise error
        if "--out" in args:
            Path(args[args.index("--out") + 1]).write_bytes(b"synthetic mesh\n")
        elif Path(args[0]).name == "drop_enclosed.py":
            Path(args[2]).write_bytes(b"synthetic cleanup mesh\n")
        return "\n".join(lines)


@pytest.mark.parametrize("method_type", [SideFuseTsdf, FusedTsdf, FusedPlus])
@pytest.mark.parametrize("devices,expected,warning", [
    (["CUDA:0"], "CUDA:0", False),
    (["CPU:0"], "CPU:0", True),
    (["CUDA:0", "CPU:0"], "CPU:0", True),
    (["CPU:0", "CUDA:0"], "CUDA:0", False),
    ([], "unknown", False),
])
def test_actual_last_fusion_device_is_recorded(tmp_path, method_type, devices, expected, warning):
    method = method_type()
    ctx = Context(tmp_path, method, [(events(*devices), None)])
    result = method.execute(ctx)
    assert result["stats"]["device"] == expected
    assert result["stats"]["backend"] == expected
    assert result["stats"]["gpu_requested"] is True
    assert (WARNING in ctx.warns) is warning
    assert len(ctx.warns) == int(warning)
    assert result["stats"]["faces"] == 12
    assert len(ctx.progress_events) == 1
    assert "--gpu" in ctx.calls[0][0]


def test_intentional_cpu_side_is_recorded_without_fallback_warning(tmp_path):
    method = SideFuseTsdf()
    ctx = Context(tmp_path, method, [(events("CPU:0"), None)], gpu=False)
    result = method.execute(ctx)
    assert result["stats"]["device"] == "CPU:0"
    assert result["stats"]["backend"] == "CPU:0"
    assert result["stats"]["gpu_requested"] is False
    assert not ctx.warns
    assert "--gpu" not in ctx.calls[0][0]
    assert ctx.calls[0][1]["gpu"] is False


@pytest.mark.parametrize("last_event", [
    'STAGE device {"device": "GPU"}',
    'STAGE device {"device": null}',
    'STAGE device {}',
    'STAGE device []',
    'STAGE device not-json',
])
def test_invalid_last_device_cannot_leave_a_false_cuda_claim(tmp_path, last_event):
    method = SideFuseTsdf()
    ctx = Context(tmp_path, method, [(events("CUDA:0") + [last_event], None)])
    result = method.execute(ctx)
    assert result["stats"]["device"] == "unknown"
    assert result["stats"]["backend"] == "unknown"
    assert not ctx.warns


@pytest.mark.parametrize("fallback_devices,expected,warning", [
    (["CPU:0"], "CPU:0", True),
    (["CUDA:0"], "CUDA:0", False),
    ([], "unknown", False),
])
def test_plus_classic_retry_reports_only_the_successful_attempt(tmp_path, fallback_devices, expected, warning):
    method = FusedPlus()
    failure = StepError("plus failed", tail="CUDA allocation failed", rc=4, kind="killed")
    ctx = Context(tmp_path, method, [(events("CUDA:0"), failure), (events(*fallback_devices), None)])
    result = method.execute(ctx)
    assert result["stats"]["device"] == expected
    assert result["stats"]["backend"] == expected
    assert result["stats"]["mode"] == "classic (fallback)"
    assert result["transient"] is True
    assert result["fail"] == {"kind": failure.kind, "rc": failure.rc, "sig": failure.sig}
    assert result["warn"][0].startswith("Built the classic way: plus failed")
    assert (WARNING in ctx.warns) is warning
    assert len(ctx.calls) == 2
    assert "--mode" in ctx.calls[0][0]
    assert "--mode" not in ctx.calls[1][0]


def test_failed_plus_cpu_attempt_does_not_warn_if_classic_uses_cuda(tmp_path):
    method = FusedPlus()
    ctx = Context(tmp_path, method, [(events("CPU:0"), StepError("failed", rc=1)), (events("CUDA:0"), None)])
    result = method.execute(ctx)
    assert result["stats"]["device"] == "CUDA:0"
    assert not ctx.warns


def test_plus_cleanup_device_does_not_replace_fusion_device(tmp_path):
    method = FusedPlus()
    ctx = Context(tmp_path, method, [
        (events("CUDA:0"), None),
        (['STAGE device {"device": "CPU:0"}', 'STAGE drop_enclosed {"triangles_out": 9}'], None),
    ], drop_enclosed=True)
    result = method.execute(ctx)
    assert result["stats"]["device"] == "CUDA:0"
    assert result["stats"]["faces"] == 9
    assert not ctx.warns


def test_cancelled_plus_does_not_fall_back(tmp_path):
    method = FusedPlus()
    ctx = Context(tmp_path, method, [(events("CPU:0"), Cancelled())])
    with pytest.raises(Cancelled):
        method.execute(ctx)
    assert len(ctx.calls) == 1
    assert not ctx.warns


def test_disabled_plus_fallback_still_raises(tmp_path):
    method = FusedPlus()
    failure = StepError("failed", rc=1)
    ctx = Context(tmp_path, method, [(events("CPU:0"), failure)])
    ctx.params["fallback"] = False
    with pytest.raises(StepError) as caught:
        method.execute(ctx)
    assert caught.value is failure
    assert len(ctx.calls) == 1
    assert not ctx.warns


@pytest.mark.parametrize("method_type", [SideFuseTsdf, FusedTsdf, FusedPlus])
def test_shared_parser_is_in_method_cache_dependencies(method_type):
    method = method_type()
    assert "methods/side.py" in method.modules
    files, _ = method.code({})
    assert "stages/methods/side.py" in dict(files)


def test_reporting_modules_come_from_this_source_copy():
    import inspect
    for method_type in (SideFuseTsdf, FusedTsdf, FusedPlus):
        module_path = Path(inspect.getfile(method_type)).resolve()
        assert module_path.is_relative_to((ROOT / "scanner/stages/methods").resolve())
