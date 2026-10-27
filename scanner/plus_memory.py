# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan Plus grid capacity before allocating device memory."""
import math
import os
import subprocess

MIB = 1024 ** 2
GIB = 1024 ** 3
BLOCK_CLAIM = 3.0


def free_vram(device):
    if not str(device).startswith('CUDA:'): return None
    index = str(device).split(':')[-1]
    try:
        value = subprocess.check_output(['nvidia-smi', '--id=' + index, '--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'], text=True, timeout=5)
        free = int(value.strip().splitlines()[0]) * MIB
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeError('Plus cannot read free GPU memory. Use classic or check nvidia-smi.')
    cap = os.environ.get('POINTYOINK_PLUS_VRAM_GB')
    if cap is not None: free = min(free, int(float(cap) * GIB))
    return free


def peak_bytes(capacity, resolution, frame_blocks, old_capacity=0):
    # Five attributes: 3 float32 + 2 uint8. Include key/hash overhead,
    # old storage during a resize, frame tensors, grid readback, and reserve.
    working = max(frame_blocks * resolution ** 3 * 160, capacity * resolution ** 3 * 32)
    return int((capacity + old_capacity) * (resolution ** 3 * 14 + 128)
               + working + 768 * MIB)


def choose_spacing(requested, estimates, free):
    budget = int(free * .80)
    for estimate in estimates:
        if estimate['peak_bytes'] <= budget:
            return dict(estimate, requested_voxel=requested, free_bytes=free, budget_bytes=budget)
    raise RuntimeError('Plus GPU memory budget is too small even at the coarsest tested spacing; use classic or fewer frames.')


def sample_blocks(a, jobs, load, voxel, resolution):
    import numpy as np
    import open3d as o3d
    import open3d.core as core
    grid = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf',), attr_dtypes=(core.float32,), attr_channels=(1,),
              voxel_size=voxel, block_resolution=resolution, block_count=1, device=core.Device('CPU:0'))
    keys = set(); largest = 0; samples = 0
    # Spread samples across the complete job list, including both ends of each side.
    selected = set(range(0, len(jobs), 10)) | {len(jobs) - 1}
    for i in range(len(jobs)):
        if i == 0 or os.path.dirname(jobs[i][0]) != os.path.dirname(jobs[i-1][0]): selected.add(i)
        if i == len(jobs)-1 or os.path.dirname(jobs[i][0]) != os.path.dirname(jobs[i+1][0]): selected.add(i)
    for i in sorted(selected):
        if i < 0: continue
        frame = load(jobs[i])
        if frame is None: continue
        _, depth, pose, cal = frame; fx, fy, cx, cy = cal
        K = core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], core.float64)
        coords = grid.compute_unique_block_coordinates(o3d.t.geometry.Image(core.Tensor(depth)), K,
                  core.Tensor(np.linalg.inv(pose), core.float64), 1 / a.depth_scale, a.max_depth, a.plus_trunc / voxel * BLOCK_CLAIM)
        rows = coords.numpy(); largest = max(largest, len(rows)); samples += 1
        keys.update(map(tuple, rows.tolist()))
    capacity = max(64, math.ceil(len(keys) * 2.0) + largest)
    if a.plus_block_count > 0: capacity = max(capacity, int(a.plus_block_count))
    return dict(voxel=voxel, resolution=resolution, sample_frames=samples, sampled_blocks=len(keys),
                capacity=capacity, frame_blocks=largest, peak_bytes=peak_bytes(capacity, resolution, largest))


def plan(a, jobs, load, device, emit):
    free = free_vram(device)
    if free is None: return None
    
    if .8 * free <= 768 * MIB:
        raise RuntimeError('Plus GPU memory budget is too small for its working buffers.')
    requested = float(a.voxel); candidates = sorted(set([requested] + [x for x in (.35, .4, .45, .5, .6, .8, 1.) if x > requested]))
    estimates = []
    for voxel in candidates:
        resolution = int(a.plus_block_res) if a.plus_block_res else max(4, round(a.plus_block_mm / voxel))
        estimate = sample_blocks(a, jobs, load, voxel, resolution); estimates.append(estimate)
        emit('plus_estimate', **estimate, free_bytes=free)
        if estimate['peak_bytes'] <= .8 * free: break
    result = choose_spacing(requested, estimates, free)
    result['attempts'] = estimates
    emit('plus_memory', **result)
    if result['voxel'] != requested:
        emit('warn', msg='plus: %g mm; %g mm would need %.2f GiB, GPU budget %.2f GiB' %
             (result['voxel'], requested, estimates[0]['peak_bytes'] / GIB, result['budget_bytes'] / GIB))
    return result
