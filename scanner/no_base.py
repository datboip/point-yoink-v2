# SPDX-License-Identifier: AGPL-3.0-or-later
"""Identify tracked close-up passes that have enough views but no visible bed."""
import glob,os
import numpy as np

def observe(job):
    from fuse import read_global_poses,read_calib,frame_key
    from scanner.table_plane import SidePlane
    from scanner.base_geometry import OBSERVED_FRAME_BUDGET
    import struct
    folder=job['folder'];w,h=job['size']
    pose_path=os.path.join(folder,'cache','pointyoink_register_pose.pose')
    if not os.path.isfile(pose_path):pose_path=os.path.join(folder,'cache','global_register_pose.pose')
    try:
        cal=read_calib(os.path.join(folder,'param','Pl.bin'))
        cal=tuple(x*w/800 for x in cal)
        poses=read_global_poses(pose_path)
    except (OSError,ValueError,struct.error):return dict(no_base=False,reason='saved view information is missing',frames=0)
    if not all(np.isfinite(cal)) or min(cal[:2])<=0:return dict(no_base=False,reason='calibration is unknown',frames=0)
    paths=[p for p in sorted(glob.glob(os.path.join(folder,'cache','*.dph'))) if frame_key(p) in poses]
    if len(paths)<8:return dict(no_base=False,reason='not enough observed frames',frames=len(paths))
    indices=sorted(set(np.linspace(0,len(paths)-1,min(OBSERVED_FRAME_BUDGET,len(paths))).round().astype(int)))
    plane=SidePlane();F=np.diag([1.,-1.,-1.,1.])
    for i in indices:
        depth=np.fromfile(paths[i],dtype=np.uint16)
        if len(depth)!=w*h:continue
        plane.add(depth.reshape(h,w),cal,poses[frame_key(paths[i])]@F)
    if sum(len(points) for points in plane.pts)<5000:return dict(no_base=False,reason='not enough depth to decide',frames=len(plane.pts))
    if plane.spin_axis() is None:return dict(no_base=False,reason='view motion is not enough to decide',frames=len(plane.pts))
    absent=plane.plane() is None
    return dict(no_base=absent,reason='No base in view: nothing to cut' if absent else 'base observed',frames=len(plane.pts))
