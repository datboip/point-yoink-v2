import numpy as np
from track import registration_stride,REGISTRATION_MIN_POINTS


def test_sparse_far_uses_more_real_samples_without_lowering_minimum():
    z=np.ones((30,30),np.uint16)
    assert np.count_nonzero(z[::2,::2])<REGISTRATION_MIN_POINTS<=np.count_nonzero(z)
    assert registration_stride(z,2,True)==1
    assert registration_stride(z,2,False)==2
    z[:]=0;z[:5]=1
    assert registration_stride(z,2,True)==2


def test_dense_frame_preserves_normal_sampling():
    assert registration_stride(np.ones((100,100),np.uint16),2,True)==2


def test_camera_drop_keeps_measured_search_prior_without_confirming_camera():
    from scanner.bed import SideBed
    old=object();recent=object()
    s=object.__new__(SideBed);s.prior=old
    s.passes={0:dict(camera=None,reacquire_prior=recent)}
    assert s.camera_bed(0) is None
    assert s.prior_for(0) is recent
