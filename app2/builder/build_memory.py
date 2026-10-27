# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare planned worker requests with the existing desktop RAM reserve."""
import os
from scanner.stages import resources


def planned_memory(engine, to=None, force=(), from_=None, with_params=None):
    graph, _info, steps, _estimate = engine.plan(to=to, force=force, from_=from_, with_params=with_params)
    requests = []
    for step in steps:
        if step.alias or step.method is None: continue
        # Declared requests are checked before admission clamps them to the
        # available budget. VRAM reservations are not physical RAM requests.
        request = getattr(step.method, 'mem_gb', None)
        if request is None: request = resources.load()['memory_gb']
        requests.append(dict(node=step.id, bytes=int(request * resources.GIB)))
    info = resources.memory_info()
    largest = max((request['bytes'] for request in requests), default=0)
    peak = largest
    try:
        from scanner.stages.parallel import Admission
        budget = resources.budget(engine.mem_gb)
        wanted = resources.Budget(int(budget.requested_gb*resources.GIB),budget.requested_gb,budget.cpu_threads,budget.cpu_quota)
        admission = Admission(wanted,resources.gpu_job_limit(),max(1,int(os.environ.get('POINTYOINK_BUILD_WORKERS',resources.cpu_count()))))
        pending = list(steps); ids={step.id for step in steps}; done=set()
        amounts={request['node']:request['bytes'] for request in requests}
        while pending:
            batch=[]
            for step in pending:
                if any(parent in ids and parent not in done for parent in step.inputs.values()):continue
                if batch and not admission.accepts(step):continue
                admission.take(step);batch.append(step)
            if not batch:break
            peak=max(peak,sum(amounts.get(step.id,0) for step in batch))
            for step in batch:admission.release(step);pending.remove(step);done.add(step.id)
    except ValueError:pass
    if info is None:
        return dict(low=bool(requests), unknown=True, required=largest, peak=peak, steps=requests)
    total, available = info
    reserve = resources.desktop_reserve_bytes(total)
    usable = max(0, available - reserve)
    return dict(low=bool(peak > usable), unknown=False, total=total, available=available,
                reserve=reserve, usable=usable, required=largest, peak=peak, steps=requests)
