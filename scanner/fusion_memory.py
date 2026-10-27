# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bound fusion grids before allocation and report this process's sampled GPU peak."""
import atexit,os,subprocess,threading,time
MIB=1024**2;GIB=1024**3
_MONITORS=[]

def free_bytes(device):
    value=subprocess.check_output(['nvidia-smi','--id='+str(device).split(':')[-1], '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5)
    return int(value.strip().splitlines()[0])*MIB

def allocation_plan(keys,largest,free,color=False,resolution=16):
    # Chunked integrate reserves for its incoming keys, even if those keys already exist.
    chunk=min(256,max(1,int(largest)))
    capacity=max(64,len(keys)+chunk+16)
    attributes=20 if color else 8
    resident=capacity*(resolution**3*attributes+128)
    workspace=max(chunk*resolution**3*160,capacity*resolution**3*32)
    peak=resident+workspace+768*MIB;budget=int(free*.8)
    if peak>budget:raise RuntimeError('CUDA fusion memory budget: %.2f GiB needed, %.2f GiB available; the complete grid cannot fit.'%(peak/GIB,budget/GIB))
    return dict(capacity=capacity,active_blocks=len(keys),largest_frame_blocks=int(largest),chunk_blocks=chunk,estimated_peak_bytes=peak,budget_bytes=budget)

def plan(a,jobs,prepare,device,emit):
    import numpy as np,open3d as o3d
    core=o3d.core;cpu=core.Device('CPU:0');resolution=16
    probe=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf',),attr_dtypes=(core.float32,),attr_channels=(1,),voxel_size=a.voxel/1000,block_resolution=resolution,block_count=1,device=cpu)
    keys=set();largest=0;frames=0;initial_free=free_bytes(device)
    for i,job in enumerate(jobs):
        frame=prepare(job)
        if frame is None:continue
        depth,extr=frame;fx,fy,cx,cy=job[3]
        K=core.Tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],core.float64)
        coords=probe.compute_unique_block_coordinates(o3d.t.geometry.Image(core.Tensor(depth)),K,core.Tensor(extr,core.float64),1000/a.depth_scale,a.max_depth/1000,a.trunc*3.)
        rows=coords.numpy();largest=max(largest,len(rows));keys.update(map(tuple,rows.tolist()));frames+=1
        if i%50==0:allocation_plan(keys,largest,initial_free,bool(getattr(a,"_fusion_color",False)))
        if i%50==0 or i==len(jobs)-1:emit('phase',what='gpu_plan',text='Checking GPU grid: frame %d of %d'%(i+1,len(jobs)))
    result=allocation_plan(keys,largest,free_bytes(device),bool(getattr(a,'_fusion_color',False)))
    result['planned_frames']=frames;emit('fusion_memory_plan',**result)
    return result

class Peak:
    def __init__(self,device,emit):
        self.device=str(device);self.emit=emit;self.peak=0;self.samples=0;self.stop=threading.Event();self.closed=False
        self.thread=threading.Thread(target=self._run,name='fusion-vram',daemon=True)
    def sample(self):
        try:
            text=subprocess.check_output(['nvidia-smi','--id='+self.device.split(':')[-1],'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=2)
            usage=[int(row.split(',')[1])*MIB for row in text.splitlines() if row.split(',')[0].strip()==str(os.getpid())]
            self.peak=max(self.peak,max(usage,default=0));self.samples+=1
        except (OSError,ValueError,subprocess.SubprocessError):pass
    def _run(self):
        while not self.stop.is_set():self.sample();self.stop.wait(.2)
    def start(self):
        self.sample();self.thread.start();_MONITORS.append(self);atexit.register(self.finish);return self
    def finish(self):
        if self.closed:return
        self.closed=True;self.stop.set();self.thread.join(3);self.sample()
        self.emit('fusion_memory',device=self.device,peak_vram_bytes=self.peak,peak_vram_mib=round(self.peak/MIB,1),samples=self.samples,sample_interval_s=.2,pid=os.getpid())

def finish_all():
    for monitor in _MONITORS:monitor.finish()
