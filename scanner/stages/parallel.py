# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ready-step and child-job admission within one processing budget."""
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from . import resources
from .runner import Cancelled

GIB = resources.GIB
GPU_JOB_BYTES = 4 * GIB  # one Standard side measured 3922 MiB, including its CUDA context
GPU_RESERVE = 512 * 1024 ** 2


def gpu_slots(requested=2):
    requested = min(requested, resources.gpu_job_limit())
    try:
        free = int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'],text=True,timeout=5).strip()) * 1024 ** 2
    except (OSError,ValueError,subprocess.SubprocessError):return 1, None
    # The production ceiling is two. Three is measured by the admission probe,
    # and is refused before allocation when its grids cannot fit.
    fits = max(1, (free - GPU_RESERVE) // GPU_JOB_BYTES)
    return max(1,min(2,requested,int(fits))), free


def step_cost(st, budget):
    gpu = st.kind in ('side.fuse','fused') and getattr(st.method,'gpu',False) and st.params.get('gpu',True)
    mem = 4 if st.kind=='side.fuse' else budget.memory_gb if st.kind=='fused' else getattr(st.method,'mem_gb',3)
    return gpu, min(budget.memory_bytes,int((mem or budget.memory_gb)*GIB)), min(budget.cpu_threads,4 if gpu else 8) if st.kind.startswith('side.') else budget.cpu_threads


class Admission:
    """Caps for ready nodes, separate from the hard caps assigned to children."""
    def __init__(self,budget,gpu_limit=2,workers=None):
        self.budget=budget;self.gpu_limit=gpu_limit
        self.workers=workers or max(1,budget.cpu_threads)
        self.running={}
    def refresh(self, budget, gpu_limit):
        """New steps use new limits; already running reservations remain unchanged."""
        self.budget = budget
        self.gpu_limit = gpu_limit

    def accepts(self,st):
        gpu,mem,cpu=step_cost(st,self.budget)
        return (len(self.running)<self.workers and
                sum(x[1] for x in self.running.values())+mem<=self.budget.memory_bytes and
                sum(x[2] for x in self.running.values())+cpu<=self.budget.cpu_threads and
                (not gpu or sum(x[0] for x in self.running.values())<self.gpu_limit))
    def take(self,st):self.running[st.id]=step_cost(st,self.budget)
    def release(self,st):self.running.pop(st.id,None)


class Jobs:
    """All child MemoryMax and CPUQuota slices sum to this build's budget."""
    def __init__(self,runner,budget,gpu_limit):
        self.runner=runner;self.budget=budget;self.gpu_limit=gpu_limit
        self.cv=threading.Condition();self.used_mem=0;self.used_cpu=0;self.gpus=0
        self.peak_mem=0;self.peak_cpu=0;self.peak_jobs=0;self.active=0
    @contextmanager
    def lease(self,mem_gb,gpu,kind):
        default = 4 if kind=='side.fuse' else self.budget.memory_gb
        memory=min(self.budget.memory_bytes,int((mem_gb or default)*GIB))
        threads=min(self.budget.cpu_threads,4 if gpu else 8) if kind and kind.startswith('side.') else self.budget.cpu_threads
        with self.cv:
            while True:
                if self.runner.cancelled:raise Cancelled()
                # Refresh the physical headroom before admitting a new child.
                fresh=resources.budget(self.runner.memory_gb)
                # A job never asks for more than is free now (the serial build did the same); asking for the whole
                # budget while another program holds some of it would wait forever.
                self.budget = fresh
                default = 4 if kind=='side.fuse' else fresh.memory_gb
                memory = min(fresh.memory_bytes, int((mem_gb or default)*GIB))
                threads = min(fresh.cpu_threads,4 if gpu else 8) if kind and kind.startswith('side.') else fresh.cpu_threads
                if (self.used_mem+memory<=fresh.memory_bytes and
                    self.used_cpu+threads<=fresh.cpu_threads and
                    (not gpu or self.gpus<min(self.gpu_limit,resources.gpu_job_limit()))):
                    if gpu and self.gpus:
                        try:
                            free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'],text=True,timeout=5).strip())*1024**2
                        except (OSError,ValueError,subprocess.SubprocessError):free=0
                        if free<GPU_JOB_BYTES+GPU_RESERVE:self.cv.wait(.2);continue
                    break
                self.cv.wait(.2)
            self.used_mem+=memory;self.used_cpu+=threads;self.gpus+=bool(gpu);self.active+=1
            self.peak_mem=max(self.peak_mem,self.used_mem);self.peak_cpu=max(self.peak_cpu,self.used_cpu);self.peak_jobs=max(self.peak_jobs,self.active)
        try:
            yield resources.Budget(memory,self.budget.requested_gb,threads,min(self.budget.cpu_quota,threads*100))
        finally:
            with self.cv:
                self.used_mem-=memory;self.used_cpu-=threads;self.gpus-=bool(gpu);self.active-=1;self.cv.notify_all()
