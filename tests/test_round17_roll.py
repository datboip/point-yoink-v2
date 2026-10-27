import numpy as np
from scanner.scan_guide import guidance,look_at

def test_stand_roll_is_not_ready_and_has_plain_correction():
    T=look_at([0,100,200],[0,0,0],[0,1,0]);C=T.copy();a=np.pi/2
    C[:3,:3]=T[:3,:3]@np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    r=guidance(C,T)
    assert not r['ready'] and 'Roll' in r['text'] and '90 degrees' in r['text']

def test_full_turn_yaw_stays_ignored_with_correct_roll():
    T=look_at([0,100,200],[0,0,0],[0,1,0]);C=T.copy()
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    C[:3,:3]=R@T[:3,:3];C[:3,3]=R@T[:3,3]
    assert guidance(C,T,turntable=True)['ready']
