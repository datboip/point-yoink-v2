from app2.scan.range_test import result
from app2.scan.range_guide import distance_levels
import numpy as np


def sample(z,fill=.9,noise=.1,mode='near'):
    return dict(accepted=True,distance_mm=z,fill=fill,plane_rms_mm=noise,dots=12,dot_rms_mm=.3,overlay_error_px=2,camera_match=mode)


def test_sweep_has_evidence_metrics_bins_date_and_camera_pair_proof():
    rows=[sample(z,.9 if 205<=z<=400 else .5,.1 if 205<=z<=400 else .4) for z in range(200,451,5) for _ in range(3)]
    r=result(rows,'near','sha');assert r['complete'] and r['camera_pair_confirmed'] and len(r['bins'])==51
    short=result(rows[:12],'near','sha');assert not short['complete']   # 20 mm of sweep must not replace the bands
    assert r['date'] and r['evidence']==rows and r['calibration_sha256']=='sha'
    assert set(r['bins'][0])>={'fill','plane_rms_mm','dots','dot_rms_mm','overlay_error_px','camera_matches'}
    levels=distance_levels(np.array([200,210]),r);assert levels[1]==2 and levels[0]==1


def test_static_mat_or_wrong_mode_does_not_claim_a_sweep_or_camera_change():
    r=result([sample(350) for _ in range(10)],'far','sha');assert not r['complete'] and not r['camera_pair_confirmed']
    assert result([dict(accepted=False)],'near','sha') is None


def test_missing_distance_bins_stay_unmeasured_instead_of_extending_best():
    r=result([sample(z) for z in [200,205,240,245]],'near','sha');bands=r['distance_bands']
    assert not any(b['lo']<=220<=b['hi'] for b in bands)
