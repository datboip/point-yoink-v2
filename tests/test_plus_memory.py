import pytest
from scanner.plus_memory import peak_bytes, choose_spacing, free_vram, GIB


def test_memory_includes_buffers_and_resize_overlap():
    small=peak_bytes(4000,16,500)
    assert small>4000*16**3*14
    assert peak_bytes(8000,16,500,4000)>peak_bytes(8000,16,500)


def test_choose_keeps_spacing_if_it_fits_and_logs_requested_if_coarsened():
    estimates=[dict(voxel=.3,peak_bytes=11*GIB),dict(voxel=.35,peak_bytes=7*GIB)]
    chosen=choose_spacing(.3,estimates,10*GIB)
    assert chosen['voxel']==.35 and chosen['requested_voxel']==.3 and chosen['budget_bytes']==8*GIB
    assert choose_spacing(.3,estimates,20*GIB)['voxel']==.3


def test_no_candidate_fits_is_an_explicit_failure():
    with pytest.raises(RuntimeError,match='too small'):choose_spacing(.3,[dict(voxel=.4,peak_bytes=4*GIB)],GIB)


def test_vram_cap_only_limits_real_free_memory(monkeypatch):
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'10240\n')
    monkeypatch.setenv('POINTYOINK_PLUS_VRAM_GB','2')
    assert free_vram('CUDA:0')==2*GIB
    assert free_vram('CPU:0') is None
