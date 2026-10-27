import numpy as np
from app2.scan.stream_health import StreamHealth


def test_three_coincident_gaps_no_longer_claim_periodicity():
    h=StreamHealth(64);t=0
    for i in range(25):
        t+=1. if i in (5,12,18) else 1/3
        assert not h.add(t,50000)


def test_regular_dropouts_need_four_events_and_hold_after_recovery():
    h=StreamHealth(64)
    for i in range(31):warning=h.add(i/3,20 if i%10==0 else 50000)
    assert warning and h.evidence['events']==4 and h.evidence['kind']=='near-empty depth'
    assert h.add(31/3,50000)
    assert not h.add(1000,50000)


def test_uneven_pc_delivery_is_not_a_periodic_dropout():
    rng=np.random.default_rng(25);h=StreamHealth(64);t=0
    for dt in rng.lognormal(np.log(1/3),.5,1000):
        t+=dt;assert not h.add(t,50000)
