"""Native Open3D volume persistence and exact recent Undo, without hardware."""
from pathlib import Path
from types import SimpleNamespace as NS
import struct
import numpy as np


def test_exact_native_live_cache_and_unsaved_repeat_undo(tmp_path):
    import open3d as o3d
    from track import Tracker
    from app2.scan.live_build import QualityField,RecentFusion
    from app2.scan.live_cache import write,read,restore_tracker,restore_field
    from app2.scan.tracking import grid_snapshot
    from app2.scan import fmt
    cal=(100.,100.,40.,30.);d=np.full((60,80),3000,np.uint16)
    trk=Tracker(cal,80,60,o3d.core.Device('CPU:0'),voxel_mm=1.)
    (tmp_path/'param').mkdir();(tmp_path/'cache').mkdir();(tmp_path/'param/Pl.bin').write_bytes(struct.pack('<HH9f',80,60,100,0,40,0,100,30,0,0,1))
    name=fmt.frame_name(0,0);(tmp_path/'cache'/(name+'.dph')).write_bytes(d.tobytes());side=NS(folder=str(tmp_path),cache=str(tmp_path/'cache'),entries=[(0,0,np.eye(4))],scan_profile=dict(live_build=dict(voxel_mm=1.)))
    field=QualityField(1.,dict(observed_distance_mm=[100,500]),support=1)
    trk.seed(d,np.eye(4));field.observe([[0,0,300]],[300]);assert write(side,trk,field)
    data=read(side);assert data is not None and restore_field(data).state()==field.state()
    expected=grid_snapshot(trk);undo=RecentFusion(3,6);undo.before(trk,0,side.entries,[])
    # This repeat is deliberately not stored in the side's frame list.
    trk.seed(d,np.eye(4));assert not field.observe([[0,0,300]],[300])['keep'];undo.restore(trk)
    def same(a,b):
        ka,va=a;kb,vb=b;oa=np.lexsort(ka.numpy().T);ob=np.lexsort(kb.numpy().T)
        return np.array_equal(ka.numpy()[oa],kb.numpy()[ob]) and all(np.array_equal(x.numpy()[oa],y.numpy()[ob]) for x,y in zip(va,vb))
    assert same(expected,grid_snapshot(trk))
    trk.seed(d,np.eye(4));restore_tracker(trk,data);assert same(expected,grid_snapshot(trk))
    (tmp_path/'cache'/(name+'.dph')).write_bytes(np.full_like(d,3010).tobytes());assert read(side) is None
    (tmp_path/'live-volume.npz').write_bytes(b'broken cache');assert read(side) is None
