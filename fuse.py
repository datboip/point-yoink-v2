#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# PC-side fusion for PointYoink: raw MIRACO depth frames -> fused mesh, on your PC.
# Memory-capped subprocess. Integrates frames one at a time into a TSDF voxel grid
# (bounded memory even for hundreds of frames), then extracts a triangle mesh.
#   python3 fuse.py --frames <cache dir> --calib <param/Pl.bin> --out mesh.ply [--voxel 0.3] [--gpu]
# Formats (verified against the scanner's own fuse.ply to 0.13 mm):
#   .dph  = WxH uint16 depth, depth_mm = raw * 0.1
#   .inf  = 16-byte header + 4x4 float64 camera->world pose (+ a second 4x4 we don't need)
#   Pl.bin = 10 float32: [_, fx, _, cx, _, fy, cy, _, _, 1]
#   camera frame = OpenCV frame with Y and Z flipped: p_scanner = diag(1,-1,-1) . p_opencv
import sys, os, argparse, resource, json, struct, glob

MEM_CAP_GB = float(os.environ.get("POINTYOINK_MEM_CAP_GB", "12"))

def apply_mem_cap():
    # CPU path only: CUDA reserves huge virtual address ranges, so RLIMIT_AS would
    # make cudaMalloc fail with a bogus "out of memory". VRAM is the GPU's own bound.
    try:
        cap = int(MEM_CAP_GB * 1024**3); resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception: pass

def emit(stage, **kw): print("STAGE %s %s" % (stage, json.dumps(kw)), flush=True)

def _classic_prepare_workers():
    """Frame-prep workers within the configured thread budget: 2, or 4 on a PC with 16+ threads (one side at 956 frames:
    17.7 s with 2, 13.4 s with 4, no gain past 4: the GPU feed takes over)."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    limits = [4 if available >= 16 else 2, max(1, available)]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            value = int(os.environ.get(name, "").split(",", 1)[0])
        except ValueError:
            continue
        if value > 0:
            limits.append(value)
    return min(limits)


def _ordered_frame_results(jobs, load, workers=2, pending=4):
    """Prepare a bounded number of frames while yielding in integration order.

    Closing the iterator or a failed load cancels queued work and joins only our
    workers. Running loads finish before the caller can discard their inputs.
    """
    if workers <= 1:
        for job in jobs:
            yield load(job)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    iterator = iter(jobs)
    queue = deque()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="depth-prep")
    try:
        for _ in range(max(1, pending)):
            try:
                job = next(iterator)
            except StopIteration:
                break
            queue.append(pool.submit(load, job))
        while queue:
            result = queue.popleft().result()
            yield result
            try:
                job = next(iterator)
            except StopIteration:
                continue
            queue.append(pool.submit(load, job))
    finally:
        for future in queue:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def parse_set_spec(spec):
    """Read one aligned side's frame set, including optional cut and delete regions."""
    import numpy as np
    parts = spec.split(",", 4)
    T = None
    if len(parts) > 2 and parts[2]:
        T = np.array(json.load(open(parts[2]))["matrix"] if parts[2].endswith(".json") else json.loads(parts[2]),
                     dtype=np.float64).reshape(4, 4)
    plane = json.load(open(parts[3])) if len(parts) > 3 and parts[3] else None
    deletes = json.load(open(parts[4])) if len(parts) > 4 and parts[4] else None
    return parts[0], parts[1], T, plane, deletes

def read_calib(path):
    v = struct.unpack("<10f", open(path, "rb").read())
    return v[1], v[5], v[3], v[6]            # fx, fy, cx, cy

def read_rgb_calib(param_dir):
    """The color camera of this scanner, from its own calibration folder: K (3x3, Prgb.bin at 800x600) and the depth to
    color camera transform R (3x3), T (mm) (LC_RT.bin). None when either file is missing or short."""
    import numpy as np
    try:
        k = struct.unpack("<10f", open(os.path.join(param_dir, "Prgb.bin"), "rb").read(40))
        rt = struct.unpack("<12f", open(os.path.join(param_dir, "LC_RT.bin"), "rb").read(48))
    except (OSError, struct.error): return None
    return np.array(k[1:10]).reshape(3, 3), np.array(rt[:9]).reshape(3, 3), np.array(rt[9:12])

def read_pose(path):
    import numpy as np
    return np.array(struct.unpack_from("<16d", open(path, "rb").read(), 16)).reshape(4, 4)

def read_global_poses(path):
    """The scanner's globally registered poses (cache/global_register_pose.pose): int32 count, then per
    frame an int32 pass number, int32 frame number and a 4x4 float64 camera->world pose (mm). These are
    the poses the device itself fuses with; after a Resume Scan the second pass sits tens of mm away from
    its own per-frame odometry, so building from the .inf poses alone leaves ghost surfaces."""
    import numpy as np
    b = open(path, "rb").read(); n = struct.unpack_from("<i", b, 0)[0]; out = {}
    for i in range(n):
        off = 4 + i * 136
        if off + 136 > len(b): break
        seq, frame = struct.unpack_from("<ii", b, off)
        out[(seq, frame)] = np.array(struct.unpack_from("<16d", b, off + 8)).reshape(4, 4)
    return out

def frame_key(dph):
    import re
    m = re.search(r"frame_(\d+)_(\d+)\.dph$", dph)
    return (int(m.group(1)), int(m.group(2))) if m else None

# ------------------------------------------------------------------------------------------------ --mode plus
# The tsdf_plus method, on the same frames, clean-up, platter cut,
# poses and per-set frame cap as the classic fuse above. Differences from classic:
#   - its own integration kernel on an Open3D VoxelBlockGrid, in mm: every voxel of the blocks a frame touches is
#     projected into the frame (nearest pixel, sdf = depth - z) and keeps order-independent SUMS: sb/wb (tsdf sum and
#     count of band updates, |sdf| <= trunc), wf (count of free-space updates, sdf > trunc), ns (number of distinct
#     sets with a band update). 16^3 blocks at 0.3 mm (4.8 mm) set how far free-space votes reach; that keeps a thin lip.
#   - extraction on the CPU: tsdf = (sb + wf) / (wb + wf), weight = wb + wf (plain TSDF averaging), then a weighted
#     Gaussian blur of the tsdf (sigma 0.8 voxel, weights capped at their 90th percentile) ONLY on voxels that 3 or more
#     sets saw: it merges misregistered copies of a wall (doubled walls -68 % on the headrest) and leaves 1-2-side
#     details (thin lip) alone. Marching cubes where all 8 corners weigh > 2.
#   - no keep_part here (build_project runs keep_part.py after, as for classic).
# Tuned on one part (the 15-side headrest). --plus-block-mm keeps the free-space reach when --voxel changes.
BLOCK_CLAIM = 3.0          # Standard GPU fusion: blocks claimed per frame cover 3x the truncation band (see the integrate loop)


def _plus_smooth_field(tsdf, weight, keys, res, sigma, chunk=1000, cap_pct=90.0):
    """Weighted Gaussian blur of the tsdf over observed voxels (sigma in voxels), block by block: each block is padded
    with its neighbours' voxels (looked up through the sorted block keys), blurred and cropped. Blocks are [z, y, x].
    Every block is independent of the chunk size (chunk only bounds RAM: 1000 blocks ~ 0.5 GB of temporaries)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    pad = int(np.ceil(2 * sigma)); P = res + 2 * pad; n = len(keys)
    k0 = keys.min(0) - 1; span = keys.max(0) - k0 + 2
    lin = lambda k: ((k[..., 0] - k0[0]) * span[1] + (k[..., 1] - k0[1])) * span[2] + (k[..., 2] - k0[2])
    kl = lin(keys.astype(np.int64)); order = np.argsort(kl); kls = kl[order]
    w = np.where(weight > 0, np.minimum(weight, np.percentile(weight[weight > 0], cap_pct)), 0).astype(np.float32)
    tw = (w * tsdf).reshape(n, -1); w = w.reshape(n, -1)
    r = np.arange(-pad, res + pad)
    gz, gy, gx = np.meshgrid(r, r, r, indexing="ij")
    off = np.stack([gx, gy, gz], -1).reshape(-1, 3)                     # voxel offsets (x, y, z) from the block origin
    nb = np.floor_divide(off, res); loc = off - nb * res
    locidx = (loc[:, 2] * res + loc[:, 1]) * res + loc[:, 0]
    out = tsdf.reshape(n, -1).copy()
    starts = list(range(0, n, chunk)); done = [0]
    import threading
    lock = threading.Lock()

    def work(c0):
        kb = keys[c0:c0 + chunk].astype(np.int64)
        nk = kb[:, None, :] + nb[None]
        q = lin(nk); del nk
        pos = np.searchsorted(kls, q).clip(0, n - 1); found = kls[pos] == q; del q
        bi = np.where(found, order[pos], 0); del pos
        W = np.where(found, w[bi, locidx[None]], 0).reshape(-1, P, P, P)
        TW = np.where(found, tw[bi, locidx[None]], 0).reshape(-1, P, P, P); del bi, found
        den = gaussian_filter(W, (0, sigma, sigma, sigma), truncate=2.0, mode="constant")[:, pad:-pad, pad:-pad, pad:-pad]
        num = gaussian_filter(TW, (0, sigma, sigma, sigma), truncate=2.0, mode="constant")[:, pad:-pad, pad:-pad, pad:-pad]
        del W, TW
        blk = out[c0:c0 + chunk]; den = den.reshape(len(kb), -1); num = num.reshape(len(kb), -1)
        ok = (den > 1e-6) & (w[c0:c0 + chunk] > 0)
        blk[ok] = num[ok] / den[ok]                                     # each chunk writes only its own rows
        with lock:
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == len(starts): emit("phase", what="smooth", done=done[0], total=len(starts), text="Smoothing the surface, part %d of %d" % (done[0], len(starts)))
    threads = max(1, min(4, int(os.environ.get("OMP_NUM_THREADS") or 4)))       # 1000 blocks are about 0.5 GB of temporaries each
    if threads > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=threads) as ex: list(ex.map(work, starts))
    else:
        for c0 in starts: work(c0)
    return out.reshape(tsdf.shape).astype(np.float32)


def _canonical_mesh(mesh):
    """Stable vertex and face order without changing the surface or winding."""
    import numpy as np,open3d as o3d
    V=np.asarray(mesh.vertices);F=np.asarray(mesh.triangles)
    order=np.lexsort((V[:,2],V[:,1],V[:,0]))
    sorted_v=V[order];same=np.all(sorted_v[1:]==sorted_v[:-1],axis=1)
    if same.any():
        # Coincident vertices can belong to different sheets. Order these ties
        # by their incident triangle geometry, keeping each vertex separate.
        group=np.cumsum(np.r_[True,~same])-1;classes=np.empty(len(order),np.int64);classes[order]=group
        tied=np.zeros(len(order),bool);tied[:-1]|=same;tied[1:]|=same
        duplicate=np.zeros(len(order),bool);duplicate[order[tied]]=True
        selected=duplicate[F].any(axis=1);geometry=classes[F[selected]];first=geometry.argmin(1)
        geometry=np.stack([geometry[np.arange(len(geometry)),(first+k)%3] for k in range(3)],1)
        signatures={int(v):[] for v in order[tied]}
        for face,geo in zip(F[selected],geometry):
            for v in face:
                if duplicate[v]:signatures[int(v)].append(tuple(geo))
        starts=np.r_[0,np.flatnonzero(~same)+1];ends=np.r_[starts[1:],len(order)]
        for start,end in zip(starts,ends):
            if end-start>1:order[start:end]=sorted(order[start:end],key=lambda v:tuple(sorted(signatures[int(v)])))
    inverse=np.empty(len(order),np.int64);inverse[order]=np.arange(len(order))
    F=inverse[F];first=F.argmin(1)
    F=np.stack([F[np.arange(len(F)),(first+k)%3] for k in range(3)],1)
    F=F[np.lexsort((F[:,2],F[:,1],F[:,0]))]
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[order]),o3d.utility.Vector3iVector(F.astype(np.int32)))
    if mesh.has_vertex_colors():result.vertex_colors=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[order])
    result.compute_vertex_normals();return result


def _plus_integrate(a, jobs, load, dev, res):
    """GPU (or CPU) integration of every job into sum accumulators; returns them downloaded to numpy (float32 sb, wb,
    wf and uint8 ns per voxel of every active block, plus the int32 block keys)."""
    import threading, time
    import numpy as np, open3d as o3d, open3d.core as o3c
    from multiprocessing.pool import ThreadPool
    from scanner.plus_memory import BLOCK_CLAIM as PLUS_BLOCK_CLAIM
    vox = a.voxel; T = a.plus_trunc
    names = ("sb", "wb", "wf", "ns", "last")
    count = int(a.plus_block_count)
    if count <= 0:     # auto: ~6 blocks of 4.8 mm per frame (4,003 frames -> 16,872 blocks on the headrest), at least 4000
        count = max(4000, int(6 * len(jobs) * (4.8 / (res * vox)) ** 2))
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=names, attr_dtypes=(o3c.float32,) * 3 + (o3c.uint8,) * 2,
                                        attr_channels=((1),) * len(names), voxel_size=vox, block_resolution=res,
                                        block_count=count, device=dev)
    hm = vbg.hashmap()
    # the set a frame belongs to (its frames folder), as a 1..255 byte for the distinct-set count
    set_of = {}
    for j in jobs: set_of.setdefault(os.path.dirname(os.path.abspath(j[0])), len(set_of))
    tag = lambda dph: (set_of[os.path.dirname(os.path.abspath(dph))] % 255) + 1
    one = lambda v: o3c.Tensor(np.array([v], np.uint8), device=dev)
    t0 = time.time(); n = 0; cap0 = hm.capacity()
    # frames are read and cleaned in threads (numpy releases the GIL); runs of one frame size go one after another so
    # the clean-up's shared pixel grid never changes size under a running thread; at most 4 x workers frames in flight
    runs = []
    for j in jobs:
        if runs and runs[-1][0][5] == j[5]: runs[-1].append(j)
        else: runs.append([j])
    workers = max(1, int(a.plus_workers)); sem = threading.Semaphore(4 * workers); stop = threading.Event()

    def integrate_frame(r):
        dph, d, Tcm, cal = r
        fx, fy, cx, cy = cal; Hh, Ww = d.shape
        K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
        ext = np.linalg.inv(Tcm)
        dt = o3c.Tensor(d).to(dev); dimg = o3d.t.geometry.Image(dt)
        coords = vbg.compute_unique_block_coordinates(dimg, K, o3c.Tensor(ext, o3c.float64), 1.0 / a.depth_scale,
                                                      a.max_depth, T / vox * PLUS_BLOCK_CLAIM)
        if getattr(a, "_plus_memory", None):
            from scanner.plus_memory import peak_bytes, free_vram
            # Open3D reserves for every incoming key, including existing keys.
            _,present=hm.find(coords)
            new_coords=coords[present.logical_not()]
            needed=hm.size()+int(new_coords.shape[0])
            old = hm.capacity(); predicted = old if needed <= old else max(2 * old, needed)
            required = peak_bytes(predicted, res, coords.shape[0], old if predicted != old else 0)
            budget = a._plus_memory["budget_bytes"]
            if predicted != old:
                budget = min(budget, int(free_vram(dev) * .8) + old * res ** 3 * 14)
            if required > budget:
                raise RuntimeError("Plus GPU memory budget reached before grid growth; use coarser spacing or fewer frames.")
        _,present=hm.find(coords)
        new_coords=coords[present.logical_not()]
        if new_coords.shape[0]:hm.activate(new_coords)
        buf,_=hm.find(coords)
        X, idx = vbg.voxel_coordinates_and_flattened_indices(buf)
        R = o3c.Tensor(np.ascontiguousarray(ext[:3, :3].T).astype(np.float32), device=dev)
        t = o3c.Tensor(ext[:3, 3].astype(np.float32).reshape(1, 3), device=dev)
        Xc = X.matmul(R) + t
        z = Xc[:, 2].contiguous(); zc = z.clip(1.0, 1e9)
        u = (Xc[:, 0] * fx / zc + (cx + 0.5)).floor(); v = (Xc[:, 1] * fy / zc + (cy + 0.5)).floor()
        inb = u.ge(0).logical_and(u.le(Ww - 1)).logical_and(v.ge(0)).logical_and(v.le(Hh - 1)).logical_and(z.gt(1.0))
        lin = v.clip(0, Hh - 1).to(o3c.int64) * Ww + u.clip(0, Ww - 1).to(o3c.int64)
        dd = (dt.reshape((-1,)).to(o3c.float32) * a.depth_scale)[lin]; sdf = dd - z
        valid = inb.logical_and(dd.gt(0)).logical_and(sdf.ge(-T))
        band = valid.logical_and(sdf.le(T)); free = valid.logical_and(sdf.gt(T))
        sel = band.logical_or(free).nonzero()[0]
        if sel.shape[0] == 0: return False
        ii = idx[sel]; s = sdf[sel]
        bf = band[sel].to(o3c.float32); fr = free[sel].to(o3c.float32)
        tsdf = (s / T).clip(-1.0, 1.0)
        A = {k: vbg.attribute(k).reshape((-1,)) for k in names}       # after activate: a rehash moves them
        A["sb"][ii] = A["sb"][ii] + bf * tsdf
        A["wb"][ii] = A["wb"][ii] + bf
        A["wf"][ii] = A["wf"][ii] + fr
        ib = ii[band[sel].nonzero()[0]]; sno = tag(dph)
        ibn = ib[A["last"][ib].ne(one(sno)).nonzero()[0]]
        A["ns"][ibn] = A["ns"][ibn] + one(1)
        A["last"][ib] = o3c.Tensor.full((ib.shape[0],), sno, o3c.uint8, dev)
        return True

    def feed(run):
        for j in run:
            while not sem.acquire(timeout=0.5):             # abortable: an error in the loop below must not leave this thread
                if stop.is_set(): return                   # waiting forever (the pool's exit joins it: that was a hang, not an error)
            if stop.is_set(): return
            yield j
    with ThreadPool(workers) as pool:
        try:
            for run in runs:
                for r in pool.imap(load, feed(run), chunksize=1):
                    try:
                        if r is None: continue
                        if not integrate_frame(r): continue
                        n += 1
                        if n % 10 == 0 or n == len(jobs): emit("integrate", done=n, total=len(jobs))
                    finally:
                        sem.release()
                        if str(dev).startswith("CUDA:") and r is not None:
                            # Open3D caches differently sized frame buffers. Release
                            # them after all frame tensor references leave scope.
                            o3c.cuda.synchronize()
                            o3c.cuda.release_cache()
        except BaseException as e:
            stop.set()
            for _ in range(4 * workers): sem.release()
            where = "frame %d of %d" % (n + 1, len(jobs))
            why = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            if "out of memory" in why.lower() or "cuda" in why.lower():
                why += ". The fused grid did not fit on the GPU: use a bigger point spacing (0.4 mm) or fewer frames per side."
            print("ERROR fusing stopped at %s: %s" % (where, why), flush=True)
            emit("error", msg="fusing stopped at %s: %s" % (where, why))
            raise
    emit("integrate", done=len(jobs), total=len(jobs))
    emit("phase", what="grid", text="Reading the fused grid back from the GPU")
    nb = hm.size(); cap = hm.capacity()
    grid_mb = cap * res ** 3 * 14 / 1048576.0
    emit("plus_grid", frames=n, blocks=nb, capacity=cap, capacity_start=cap0, grid_mb=round(grid_mb), device=str(dev),
         secs=round(time.time() - t0, 1))
    if cap > cap0: emit("warn", msg="plus: block hash map grew from %d to %d blocks" % (cap0, cap))
    ab = hm.active_buf_indices().to(o3c.int64)
    acc = dict(keys=hm.key_tensor()[ab].cpu().numpy().astype(np.int32), voxel=np.float32(vox), res=np.int32(res), trunc=np.float32(T))
    for k in ("sb", "wb", "wf"):
        acc[k] = vbg.attribute(k)[ab].cpu().numpy().reshape(nb, res, res, res).astype(np.float32)
    acc["ns"] = vbg.attribute("ns")[ab].cpu().numpy().reshape(nb, res, res, res).astype(np.uint8)
    del vbg, hm
    return acc


def _plus_extract(a, acc):
    """CPU: combine the sums, side-gated field smoothing, marching cubes (legacy mesh in mm)."""
    import numpy as np, open3d as o3d, open3d.core as o3c
    res = int(acc["res"]); vox = float(acc["voxel"]); keys = acc["keys"]
    num = acc.pop("sb") + acc["wf"]; den = acc.pop("wb") + acc.pop("wf")
    tsdf = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0).astype(np.float32); del num
    weight = den.astype(np.float32); del den
    ns = acc.pop("ns")
    if a.plus_field_smooth > 0:
        sm = _plus_smooth_field(tsdf, weight, keys, res, a.plus_field_smooth)
        if a.plus_smooth_min_sides > 1:           # leave voxels seen by few sets (thin lip, one-side details) unblurred
            sm = np.where(ns >= a.plus_smooth_min_sides, sm, tsdf)
        tsdf = sm; del sm
    del ns
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
                                        attr_channels=((1), (1)), voxel_size=vox, block_resolution=res,
                                        block_count=int(len(keys) * 1.3) + 16, device=o3c.Device("CPU:0"))
    hm = vbg.hashmap(); kt = o3c.Tensor(keys)
    hm.activate(kt); buf, _ = hm.find(kt); bi = buf.to(o3c.int64)
    vbg.attribute("tsdf")[bi] = o3c.Tensor(tsdf.reshape(len(keys), res, res, res, 1)); del tsdf
    vbg.attribute("weight")[bi] = o3c.Tensor(weight.reshape(len(keys), res, res, res, 1)); del weight
    emit("phase", what="mesh", text="Pulling the surface out of the grid (marching cubes)")
    return vbg.extract_triangle_mesh(weight_threshold=a.plus_min_weight).to_legacy()


def fuse_plus(a, jobs, load, dev):
    """--mode plus: integrate (or read --plus-from-acc), extract, write a.out atomically. load(job) -> (dph, cleaned and
    platter-cut uint16 depth, 4x4 OpenCV camera -> output frame in mm, (fx, fy, cx, cy)) or None."""
    import time, resource
    import numpy as np, open3d as o3d
    t0 = time.time()
    requested_voxel = a.voxel
    a._plus_memory = None
    if not a.plus_from_acc:
        from scanner.plus_memory import plan
        a._plus_memory = plan(a, jobs, load, dev, emit)
        if a._plus_memory:
            a.voxel = a._plus_memory["voxel"]
            a.plus_field_smooth *= requested_voxel / a.voxel
            a.plus_block_count = a._plus_memory["capacity"]
    res = int(a.plus_block_res) if a.plus_block_res else max(4, int(round(a.plus_block_mm / a.voxel)))
    emit("plus", voxel=a.voxel, trunc_mm=a.plus_trunc, block_res=res, block_mm=round(res * a.voxel, 3),
         min_weight=a.plus_min_weight, field_smooth=a.plus_field_smooth, smooth_min_sides=a.plus_smooth_min_sides)
    if a.plus_from_acc:
        z = np.load(a.plus_from_acc); acc = {k: z[k] for k in z.files}; del z
        if int(acc["res"]) != res or abs(float(acc["voxel"]) - a.voxel) > 1e-6:
            emit("warn", msg="plus: accumulators are %.3g mm / %d^3 blocks; using those" % (float(acc["voxel"]), int(acc["res"])))
    else:
        acc = _plus_integrate(a, jobs, load, dev, res)
        if a.plus_acc:
            np.savez(a.plus_acc, **acc); emit("plus_acc", path=a.plus_acc, mb=round(os.path.getsize(a.plus_acc) / 1048576))
    emit("extract"); emit("phase", what="extract", text="Combining the sides' sums")
    t1 = time.time()
    mesh = _plus_extract(a, acc); del acc
    emit("phase", what="order", text="Sorting the mesh")
    # canonical vertex and triangle order: the hash maps hand blocks out in a different order every run, and the same
    # surface must give the same bytes (the stages engine's early cutoff compares them)
    V = np.asarray(mesh.vertices); Tr = np.asarray(mesh.triangles)
    o = np.lexsort((V[:, 2], V[:, 1], V[:, 0])); inv = np.empty(len(o), np.int64); inv[o] = np.arange(len(o))
    Tr = inv[Tr]; r = Tr.argmin(1)
    Tr = np.stack([Tr[np.arange(len(Tr)), (r + k) % 3] for k in range(3)], 1)       # smallest index first, same winding
    Tr = Tr[np.lexsort((Tr[:, 2], Tr[:, 1], Tr[:, 0]))]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[o]), o3d.utility.Vector3iVector(Tr.astype(np.int32)))
    del V, Tr, o, inv, r
    mesh.compute_vertex_normals()
    _tmp = a.out + ".tmp.%d.ply" % os.getpid()
    ok = o3d.io.write_triangle_mesh(_tmp, mesh, write_ascii=False)
    if not ok or not os.path.exists(_tmp) or os.path.getsize(_tmp) < 64:
        try: os.remove(_tmp)
        except Exception: pass
        print("ERROR could not write %s" % a.out, flush=True); return 3
    os.replace(_tmp, a.out)
    emit("done", verts=len(mesh.vertices), faces=len(mesh.triangles), mb=round(os.path.getsize(a.out) / 1048576, 1),
         mode="plus", voxel=a.voxel, requested_voxel=requested_voxel, extract_secs=round(time.time() - t1, 1), secs=round(time.time() - t0, 1),
         peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2))
    return 0


def _plus_requested(argv):
    for i, x in enumerate(argv):
        if x == "--mode" and i + 1 < len(argv) and argv[i + 1] == "plus": return True
        if x == "--mode=plus": return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames"); ap.add_argument("--calib")
    ap.add_argument("--set", action="append", default=[], help="frames_dir,calib,transform.json[,plane.json[,deletes.json]] (repeat for aligned sides; transform is scan -> base in mm; the optional plane and side deletes filter depth before fusion)")
    ap.add_argument("--plane", default=None, help="base cut for a single --frames run: plane.json from Remove base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.3, help="voxel size in mm (detail; smaller = finer, more RAM)")
    ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--depth-scale", type=float, default=0.1, help="mm per raw unit")
    ap.add_argument("--max-depth", type=float, default=600.0, help="mm; ignore farther pixels")
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--white-balance", choices=("on","off"), default="on", help="with color: neutralize the camera's overall tint")
    ap.add_argument("--color", choices=("auto","on","off"), default="auto", help="saved scan color: auto requires color frames and calibration on every active side")
    ap.add_argument("--poses", default="auto", help="global pose table: auto (use <frames>/global_register_pose.pose if present), none, or a path")
    # defaults measured against the scanner's One-tap Edit, matching its Mesh panel:
    # frame filter 3 mm / 80 deg removes the edge and grazing pixels that made double skins; a 3-voxel band keeps thin sheets
    # from cancelling out; 3 views per voxel drops one-frame noise. Our surface then sits within 0.3 mm of the scanner's (95%).
    ap.add_argument("--min-weight", type=float, default=3.0, help="GPU: drop voxels seen fewer than N times (raise to cut noise)")
    ap.add_argument("--trunc", type=float, default=3.0, help="truncation band in voxels (wider merges nearby double surfaces, narrower keeps thin sheets)")
    ap.add_argument("--filter", type=float, default=3.0, help="per-frame clean-up: drop depth pixels at jumps bigger than this many mm and at grazing view angles (0 = off; the scanner does this)")
    ap.add_argument("--set-max-frames", type=int, default=0, help="with several --set: use at most this many frames of each, evenly spread (0 = all). A long set otherwise outvotes the rest: a side of 790 frames, seeing a thin lip edge-on, erased it where 250-frame sides had seen it")
    ap.add_argument("--grazing", type=float, default=80.0, help="with --filter: drop pixels whose surface tilts more than this many degrees from the view ray")
    # --mode plus (see fuse_plus above); every --plus-* flag is ignored by the classic mode
    ap.add_argument("--mode", default="classic", choices=("classic", "plus"), help="classic: today's TSDF (default); plus: the bake-off's tsdf_plus (sum accumulators, side-gated field smoothing)")
    ap.add_argument("--plus-trunc", type=float, default=1.2, help="plus: truncation band in mm (narrower doubled walls on the headrest)")
    ap.add_argument("--plus-block-mm", type=float, default=4.8, help="plus: block edge in mm (block_res = round(this / voxel)); sets how far free-space votes reach: 2.4 mm lost the thin lip")
    ap.add_argument("--plus-block-res", type=int, default=0, help="plus: voxels per block edge (0 = from --plus-block-mm)")
    ap.add_argument("--plus-block-count", type=int, default=0, help="plus: starting capacity of the block hash map (0 = auto from the frame count; it grows by itself when full). VRAM = count x block_res^3 x 14 B: 16,872 blocks of 16^3 (0.92 GB) on the 15-side headrest")
    ap.add_argument("--plus-min-weight", type=float, default=2.0, help="plus: marching cubes needs weight > this at all 8 corners")
    ap.add_argument("--plus-field-smooth", type=float, default=0.8, help="plus: weighted Gaussian blur of the tsdf, sigma in voxels (0 = off)")
    ap.add_argument("--plus-smooth-min-sides", type=int, default=3, help="plus: blur only voxels with band updates from at least this many sets")
    ap.add_argument("--plus-workers", type=int, default=6, help="plus: frame loading threads")
    ap.add_argument("--plus-acc", default=None, help="plus: also save the accumulators here (.npz) for re-extraction")
    ap.add_argument("--plus-from-acc", default=None, help="plus: skip integration, extract from saved accumulators")
    a = ap.parse_args()
    import numpy as np, open3d as o3d
    # decide the real backend first: "--gpu" on a machine without CUDA runs on the CPU and needs the cap
    if not (a.gpu and o3d.core.cuda.is_available()): apply_mem_cap()
    # one or more frame sets: frames, calibration, transform, plane, side delete regions
    sets = []
    if a.set:
        for spec in a.set:
            sets.append(parse_set_spec(spec))
    elif a.frames and a.calib: sets.append((a.frames, a.calib, None, json.load(open(a.plane)) if a.plane else None, None))
    else: emit("error", msg="give --frames and --calib, or --set"); return 2
    fx, fy, cx, cy = read_calib(sets[0][1])
    emit("calib", fx=round(fx, 2), fy=round(fy, 2), cx=round(cx, 2), cy=round(cy, 2))
    jobs = []      # (dph, gp dict, T mm, calib tuple, plane, (W, H), own poses, side deletes)
    def frame_size(frames, dphs):
        """The scan's depth frame size: the scanner's property.rvproj says (Standard accuracy saves 400x300, High 800x600),
        else the file size tells (2 bytes per pixel)."""
        try:
            sp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(frames)), "property.rvproj"))).get("scan_param", {})
            w, h = int(sp.get("depth_width") or 0), int(sp.get("depth_height") or 0)
            if w and h: return w, h
        except Exception: pass
        px = os.path.getsize(dphs[0]) // 2
        for w, h in ((800, 600), (400, 300), (640, 480), (1280, 800), (320, 240)):
            if w * h == px: return w, h
        return a.width, a.height
    for frames, calib, T, P, deletes in sets:
        dphs = sorted(glob.glob(os.path.join(frames, "*.dph")))[::max(1, a.every)]
        if not dphs: emit("error", msg="no .dph frames in " + frames); return 2
        if a.set_max_frames and len(dphs) > a.set_max_frames:
            dphs = [dphs[k] for k in np.unique(np.linspace(0, len(dphs) - 1, a.set_max_frames).round().astype(int))]
        Wf, Hf = frame_size(frames, dphs)
        gp = {}
        if a.poses == "auto":
            pose_path = os.path.join(frames, "global_register_pose.pose")          # the scanner's own registration
            if not os.path.exists(pose_path): pose_path = os.path.join(frames, "pointyoink_register_pose.pose")   # ours (register.py)
        else: pose_path = a.poses
        own = False                                      # our own table (register.py / track.py): a frame missing from it was dropped on purpose
        if a.poses != "none" and os.path.exists(pose_path):
            try: gp = read_global_poses(pose_path); own = os.path.basename(pose_path) == "pointyoink_register_pose.pose"
            except Exception as e: emit("warn", msg="global poses unreadable: %r" % (e,)); gp = {}
        cal = read_calib(calib)
        if Wf != 800:                                    # Pl.bin is for the 800x600 frames; scale it for the other sizes
            k = Wf / 800.0; cal = (cal[0] * k, cal[1] * k, cal[2] * k, cal[3] * k)
        emit("frames", count=len(dphs), global_poses=len(gp), set=os.path.basename(os.path.dirname(frames)), size="%dx%d" % (Wf, Hf))
        jobs += [(d, gp, T, cal, P, (Wf, Hf), own, deletes) for d in dphs]
    dphs = [j[0] for j in jobs]
    from scanner.color_frames import enabled as color_enabled
    use_color=color_enabled(sets,a.color)
    fuse_color=False      # the volume fuses gray; paint() colors the extracted mesh from the photos (it replaced the fused color anyway)

    W, H = a.width, a.height
    # depth in meters = raw * depth_scale(mm) / 1000  ->  Open3D divides by depth_scale
    o3d_depth_scale = 1000.0 / a.depth_scale
    depth_trunc_m = a.max_depth / 1000.0
    voxel_m = a.voxel / 1000.0
    F = np.diag([1.0, -1.0, -1.0, 1.0])       # OpenCV -> scanner camera frame

    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    dev = o3d.core.Device("CUDA:0") if (a.gpu and o3d.core.cuda.is_available()) else o3d.core.Device("CPU:0")
    emit("device",device=str(dev))
    if str(dev).startswith("CUDA:"):
        from scanner.fusion_memory import Peak
        Peak(dev,emit).start()

    use_t = a.gpu and o3d.core.cuda.is_available()
    if a.mode == "plus": pass                  # fuse_plus makes its own grid
    elif use_t:
        vbg=None
        K = o3d.core.Tensor(intr.intrinsic_matrix, o3d.core.float64)
    else:
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_m, sdf_trunc=voxel_m * a.trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8 if fuse_color else o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    n = 0; cur_cal = None; grid = None; cur_size = None
    cos_min = float(np.cos(np.radians(a.grazing)))
    def clean_frame(d, cal, pixel_grid=None):
        """Drop the pixels the scanner would not trust: depth jumps (edges, flying pixels) and surfaces seen edge-on."""
        nonlocal grid
        fx, fy, cx, cy = cal
        z = d.astype(np.float32) * a.depth_scale; valid = z > 0
        # depth jumps to any 4-neighbour
        jump = np.zeros_like(valid)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            zs = np.roll(z, (dy, dx), axis=(0, 1)); vs = np.roll(valid, (dy, dx), axis=(0, 1))
            jump |= vs & (np.abs(zs - z) > a.filter)
        edge = jump | ~valid
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, -1, 0) | np.roll(edge, 1, 1) | np.roll(edge, -1, 1)   # one-pixel skirt
        # surface normal from the depth gradient, compared with the view ray
        if pixel_grid is None:
            if grid is None or grid[0].shape != d.shape:
                u, v = np.meshgrid(np.arange(d.shape[1], dtype=np.float32), np.arange(d.shape[0], dtype=np.float32)); grid = (u, v)
            pixel_grid = grid
        u, v = pixel_grid; X = (u - cx) * z / fx; Y = (v - cy) * z / fy
        dXu = np.roll(X, -1, 1) - np.roll(X, 1, 1); dYu = np.roll(Y, -1, 1) - np.roll(Y, 1, 1); dZu = np.roll(z, -1, 1) - np.roll(z, 1, 1)
        dXv = np.roll(X, -1, 0) - np.roll(X, 1, 0); dYv = np.roll(Y, -1, 0) - np.roll(Y, 1, 0); dZv = np.roll(z, -1, 0) - np.roll(z, 1, 0)
        nx = dYu * dZv - dZu * dYv; ny = dZu * dXv - dXu * dZv; nz = dXu * dYv - dYu * dXv
        nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        rl = np.sqrt(X * X + Y * Y + z * z) + 1e-9
        cosang = np.abs((nx * X + ny * Y + nz * z) / (nl * rl))
        bad = edge | (valid & (cosang < cos_min))
        d[bad] = 0
        return d
    def cull_below_plane(d, pose_mm, cal, P, pixel_grid=None):
        """Zero the depth pixels that sit on the removed side of this scan's base plane (plane in the scan's own mm frame)."""
        nonlocal grid
        fx, fy, cx, cy = cal
        if pixel_grid is None:
            if grid is None or grid[0].shape != d.shape:
                u, v = np.meshgrid(np.arange(d.shape[1], dtype=np.float32), np.arange(d.shape[0], dtype=np.float32)); grid = (u, v)
            pixel_grid = grid
        u, v = pixel_grid; z = d.astype(np.float32) * a.depth_scale          # mm, OpenCV camera frame
        X = (u - cx) * z / fx; Y = (v - cy) * z / fy
        M = pose_mm @ F                                                # camera(OpenCV) -> scan world, mm
        h = (M[0, 0] * X + M[0, 1] * Y + M[0, 2] * z + M[0, 3]) * P["n"][0] + (M[1, 0] * X + M[1, 1] * Y + M[1, 2] * z + M[1, 3]) * P["n"][1] \
            + (M[2, 0] * X + M[2, 1] * Y + M[2, 2] * z + M[2, 3]) * P["n"][2]
        margin = float(P.get("margin", 1.0))
        from scanner.base_cut import removed
        points=np.stack([M[0,0]*X+M[0,1]*Y+M[0,2]*z+M[0,3],M[1,0]*X+M[1,1]*Y+M[1,2]*z+M[1,3],M[2,0]*X+M[2,1]*Y+M[2,2]*z+M[2,3]],axis=-1)
        bad = removed(points, P)
        d[bad & (d > 0)] = 0
        return d
    def cull_side_deletes(d, pose_mm, cal, deletes):
        if not deletes: return d
        from scanner.stages.tools.edits import mask_side_depth
        return mask_side_depth(d, pose_mm, cal, deletes["ops"], deletes["stages"], a.depth_scale)
    if a.mode == "plus":
        def load_plus(job):
            """The classic loop's frame preparation below, for one job: same reads, clean-up, pose choice and platter
            cut. -> (dph, depth, camera(OpenCV) -> output 4x4 in mm, cal) or None."""
            dph, gp, T, cal, P, size, own, deletes = job
            if not os.path.exists(dph[:-4] + ".inf"): return None
            Wj, Hj = size
            raw = np.frombuffer(open(dph, "rb").read(), dtype=np.uint16)
            if raw.size != Wj * Hj: emit("warn", msg="frame %s has %d pixels, expected %dx%d; skipped" % (os.path.basename(dph), raw.size, Wj, Hj)); return None
            d = raw.reshape(Hj, Wj).copy()
            d[d * a.depth_scale > a.max_depth] = 0
            if a.filter > 0: d = clean_frame(d, cal)
            pose = gp.get(frame_key(dph))
            if pose is None and own and gp: return None
            if pose is None: pose = read_pose(dph[:-4] + ".inf")
            else: pose = pose.copy()
            if P is not None: d = cull_below_plane(d, pose, cal, P)
            d = cull_side_deletes(d, pose, cal, deletes)
            if T is not None: pose = T @ pose
            return dph, d, pose @ F, cal
        result=fuse_plus(a, jobs, load_plus, dev)
        if result==0 and use_color:
            from scanner.color_frames import paint
            mesh=o3d.io.read_triangle_mesh(a.out);paint(mesh,jobs,a.depth_scale,tolerance_mm=2*a.voxel+1,emit=emit,white=a.white_balance=='on')
            temp=a.out+".color.tmp.ply";o3d.io.write_triangle_mesh(temp,mesh);os.replace(temp,a.out)
        return result
    prepared = None
    if use_t:
        import threading
        prep_local = threading.local()
        def prepare_gpu(job):
            # CPU preparation only. The main thread owns all CUDA operations.
            dph, gp, T, cal, P, size, own, deletes = job
            inf = dph[:-4] + ".inf"
            if not os.path.exists(inf): return None
            Wj, Hj = size
            raw = np.frombuffer(open(dph, "rb").read(), dtype=np.uint16)
            if raw.size != Wj * Hj:
                emit("warn", msg="frame %s has %d pixels, expected %dx%d; skipped" % (os.path.basename(dph), raw.size, Wj, Hj))
                return None
            d = raw.reshape(Hj, Wj).copy()
            d[d * a.depth_scale > a.max_depth] = 0
            pixel_grid = None
            if a.filter > 0 or P is not None:
                pixel_grid = getattr(prep_local, "grid", None)
                if pixel_grid is None or pixel_grid[0].shape != d.shape:
                    pixel_grid = np.meshgrid(np.arange(Wj, dtype=np.float32), np.arange(Hj, dtype=np.float32))
                    prep_local.grid = pixel_grid
            if a.filter > 0: d = clean_frame(d, cal, pixel_grid)
            pose = gp.get(frame_key(dph))
            if pose is None and own and gp: return None
            if pose is None: pose = read_pose(inf)
            else: pose = pose.copy()
            if P is not None: d = cull_below_plane(d, pose, cal, P, pixel_grid)
            d = cull_side_deletes(d, pose, cal, deletes)
            if T is not None: pose = T @ pose
            pose[:3, 3] /= 1000.0
            extr = np.linalg.inv(pose @ F)
            return d, extr
        from scanner.fusion_memory import plan
        a._fusion_color=fuse_color
        gpu_plan=plan(a,jobs,prepare_gpu,dev,emit)
        vbg=o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf","weight","color") if fuse_color else ("tsdf","weight"),
             attr_dtypes=(o3d.core.float32,)*(3 if fuse_color else 2),attr_channels=(1,1,3) if fuse_color else (1,1),
             voxel_size=voxel_m,block_resolution=16,block_count=gpu_plan["capacity"],device=dev)
        gpu_plan["capacity"]=vbg.hashmap().capacity()
        workers = _classic_prepare_workers()
        prepared = _ordered_frame_results(jobs, prepare_gpu, workers=workers, pending=max(4, 2 * workers))
    try:
        for i, (dph, gp, T, cal, P, size, own, deletes) in enumerate(jobs):
            frame = next(prepared) if prepared is not None else None
            inf = dph[:-4] + ".inf"
            if not os.path.exists(inf): continue
            if cal != cur_cal or size != cur_size:    # each scan can carry its own calibration and frame size
                cur_cal = cal; cur_size = size; fx, fy, cx, cy = cal; W, H = size
                if use_t: K = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3d.core.float64)
                else: intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            if use_t:
                if frame is None: continue
                d, extr = frame
            else:
                raw = np.frombuffer(open(dph, "rb").read(), dtype=np.uint16)
                if raw.size != W * H: emit("warn", msg="frame %s has %d pixels, expected %dx%d; skipped" % (os.path.basename(dph), raw.size, W, H)); continue
                d = raw.reshape(H, W).copy()
                d[d * a.depth_scale > a.max_depth] = 0
                if a.filter > 0: d = clean_frame(d, cal)
                pose = gp.get(frame_key(dph))            # camera->world (scanner frame), translation in mm
                if pose is None and own and gp: continue  # track.py lost this frame: the scanner's pose for it is the slip we are avoiding
                if pose is None: pose = read_pose(inf)
                else: pose = pose.copy()
                if P is not None: d = cull_below_plane(d, pose, cal, P)      # drop the table before it ever reaches the volume
                d = cull_side_deletes(d, pose, cal, deletes)
                if T is not None: pose = T @ pose         # this scan's frame -> the base scan's frame (mm)
                pose[:3, 3] /= 1000.0                     # -> meters, to match the depth units
                extr = np.linalg.inv(pose @ F)            # world->camera (OpenCV frame)
            if use_t:
                dimg = o3d.t.geometry.Image(o3d.core.Tensor(d)).to(dev)
                ext = o3d.core.Tensor(extr, o3d.core.float64)
                # claim the blocks of a band 3x the one written: claimed at 1x, voxels near a block's border missed some frames and
                # left a step on every border (rows of little triangles on smooth faces, a grid with smaller blocks); same speed
                coords = vbg.compute_unique_block_coordinates(dimg, K, ext, o3d_depth_scale, depth_trunc_m, a.trunc * BLOCK_CLAIM)
                if fuse_color:
                    from scanner.color_frames import aligned
                    cimg=o3d.t.geometry.Image(o3d.core.Tensor(aligned(dph,d,cal,a.depth_scale))).to(dev)
                for start in range(0,int(coords.shape[0]),gpu_plan["chunk_blocks"]):
                    chunk=coords[start:start+gpu_plan["chunk_blocks"]]
                    if fuse_color:vbg.integrate(chunk,dimg,cimg,K,ext,o3d_depth_scale,depth_trunc_m,a.trunc)
                    else:vbg.integrate(chunk,dimg,K,ext,o3d_depth_scale,depth_trunc_m,a.trunc)
                if vbg.hashmap().capacity()>gpu_plan["capacity"]:
                    raise RuntimeError("CUDA fusion grid grew beyond the complete input plan; stopped before further allocation.")
                chunk=None;coords=None;dimg=None;ext=None
                if fuse_color:cimg=None
                o3d.core.cuda.synchronize();o3d.core.cuda.release_cache()
            else:
                dimg = o3d.geometry.Image(d)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(__import__("scanner.color_frames",fromlist=["aligned"]).aligned(dph,d,cal,a.depth_scale) if fuse_color else np.zeros((H, W, 3), np.uint8)), dimg,
                    depth_scale=o3d_depth_scale, depth_trunc=depth_trunc_m, convert_rgb_to_intensity=False)
                vol.integrate(rgbd, intr, extr)
            n += 1
            if n % 10 == 0 or n == len(dphs):
                emit("integrate", done=n, total=len(dphs))

    finally:
        if prepared is not None:
            prepared.close()

    emit("extract")
    if use_t:
        mesh = vbg.extract_triangle_mesh(weight_threshold=a.min_weight).to_legacy()
    else:
        mesh = vol.extract_triangle_mesh()
    mesh=_canonical_mesh(mesh)
    # back to mm for the rest of the pipeline
    mesh.scale(1000.0, center=(0, 0, 0))
    if use_color:
        from scanner.color_frames import paint
        color_stats=paint(mesh,jobs,a.depth_scale,tolerance_mm=2*a.voxel+1,emit=emit,white=a.white_balance=='on')
        emit("color_done",**color_stats)
    _tmp = a.out + ".tmp.%d.ply" % os.getpid()   # write to a temp then atomically replace: a crash/OOM mid-write must not truncate the previous good model (a rebuild has no .versions backup)
    ok = o3d.io.write_triangle_mesh(_tmp, mesh, write_ascii=False)
    if not ok or not os.path.exists(_tmp) or os.path.getsize(_tmp) < 64:   # a failed writer must never replace the previous good file
        try: os.remove(_tmp)
        except Exception: pass
        print("ERROR could not write %s" % a.out, flush=True); return 3
    os.replace(_tmp, a.out)
    emit("done", verts=len(mesh.vertices), faces=len(mesh.triangles),
         mb=round(os.path.getsize(a.out) / 1048576, 1))
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except MemoryError:
        emit("error", msg="out of memory - try a larger --voxel or --every 2"); sys.exit(3)
    except Exception as e:
        msg = str(e)
        if _plus_requested(sys.argv):
            # plus never re-runs itself on the CPU: the caller (build_project) falls back to the classic fuse instead
            oom = "out of memory" in msg.lower() or "cuda" in msg.lower()
            emit("error", msg=("plus: GPU out of memory: " if oom else "plus: ") + msg[:300], oom=oom); sys.exit(4 if oom else 1)
        # A GPU that runs out of VRAM (big combines fuse hundreds of frames at once) shouldn't just fail:
        # re-exec on CPU. A fresh process means no leftover CUDA context and the CPU memory cap applies
        # cleanly; the parent keeps reading this same stdout, so the fallback needs nothing from it (just slower).
        if "--gpu" in sys.argv and ("out of memory" in msg.lower() or "cuda" in msg.lower()):
            from scanner.fusion_memory import finish_all
            finish_all()
            emit("warn", msg="GPU memory budget could not fit the complete grid after planning and bounded batches; falling back to CPU (slower)")
            os.execv(sys.executable, [sys.executable] + [a for a in sys.argv if a != "--gpu"])
        emit("error", msg=msg); sys.exit(1)
