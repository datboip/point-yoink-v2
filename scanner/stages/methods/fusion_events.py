# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fusion backend and memory event reporting."""
import re
from .stage_events import _stage_json

class _FusionBackend:
    """Track the last reported fusion device, including an in-process CPU retry."""
    def __init__(self):
        self.device=None;self.memory={}

    def on_line(self,line):
        memory=_stage_json(line,"fusion_memory")
        if memory:self.memory.update(memory)
        plan=_stage_json(line,"fusion_memory_plan")
        if plan:self.memory.update(plan)
        if not line.startswith("STAGE device "):
            return
        self.device = None
        event = _stage_json(line, "device")
        value = event.get("device") if isinstance(event, dict) else None
        if isinstance(value, str) and re.fullmatch(r"(?:CPU|CUDA):[0-9]+", value.strip()):
            self.device = value.strip()

    def stats(self, ctx, gpu_requested):
        if gpu_requested and self.device and self.device.startswith("CPU:"):
            ctx.warn("GPU was requested, but fusion ran on the CPU. Surface coverage can differ. Review the result.")
        return dict(self.memory,backend=self.device or "unknown",device=self.device or "unknown",gpu_requested=bool(gpu_requested))


