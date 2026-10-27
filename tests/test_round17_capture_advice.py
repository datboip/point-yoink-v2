import numpy as np
from scanner import scan_guide as g


class Sampled:
    def __init__(self,masks,angles,groups=None):
        self.masks=np.asarray(masks,bool);self.angles=np.asarray(angles)
        self.axis=np.array([0,1,0]);self.center=np.zeros(3)
        self.N=np.tile([0,0,1],(self.masks.shape[1],1))
        self.groups=np.arange(self.masks.shape[1]) if groups is None else np.asarray(groups)
    def reachable(self,*_):return self.masks.any(0)
    def visibility(self,*_):return self.angles,self.masks
    def phase_pose(self,T,angle):return g.Reachability.phase_pose(self,T,angle)


def setup(monkeypatch,checker,weights):
    T=g.look_at(np.array([0,20,200]),[0,0,0],[0,1,0])
    monkeypatch.setattr(g,'candidates',lambda *_:[dict(T=T,tilt=6,distance=201,height=20,mode='turntable')])
    return g.plan(checker,weights),T


def test_one_compact_patch_is_single_not_full_turn(monkeypatch):
    c=Sampled([[True]*4]*4,[0,90,180,270],groups=[0]*4)
    r,T=setup(monkeypatch,c,[25]*4)
    p=r['poses'][0]
    assert p['capture']=='Single' and p['hold_still']
    assert p['share']==1 and 'hold still' in p['text']


def test_all_around_band_requires_full_turn(monkeypatch):
    c=Sampled(np.eye(6,dtype=bool),np.arange(0,360,60))
    r,T=setup(monkeypatch,c,[100]*6)
    p=r['poses'][0]
    assert p['capture']=='Continuous' and not p['hold_still']
    assert p['share']==1 and len(p['indices'])==6


def test_two_small_patches_score_separate_realizable_phases(monkeypatch):
    c=Sampled([[False,False],[True,False],[False,True]],[0,90,180],groups=[0,1])
    r,T=setup(monkeypatch,c,[100,100])
    assert len(r['poses'])==2
    assert [p['share'] for p in r['poses']]==[.5,.5]
    assert [p['table_angle_deg'] for p in r['poses']]==[90,180]
    assert [p['indices'] for p in r['poses']]==[[0],[1]]
    assert all(p['hold_still'] and p['capture']=='Single' for p in r['poses'])
    assert np.allclose(r['poses'][0]['T'],c.phase_pose(T,90))


def test_patch_hidden_initially_uses_visible_static_pose(monkeypatch):
    V=np.array([[995,-5,100],[1005,-5,100],[1005,5,100],[995,5,100]])
    c=g.Reachability(V,[[0,1,2],[0,2,3]],[[0,0,100]],[[0,0,1]],g.Lens(100,100,100,100,200,200,10,500),axis=[0,1,0],center=[0,0,100])
    monkeypatch.setattr(g,'candidates',lambda *_:[dict(T=np.eye(4),tilt=0,distance=100,height=0,mode='turntable')])
    assert not c.reachable(np.eye(4),False)[0]
    r=g.plan(c,[100]);p=r['poses'][0]
    assert p['capture']=='Single' and p['table_angle_deg']>0
    assert c.reachable(np.asarray(p['T']),False).tolist()==[True]
    assert p['indices']==[0]


def test_live_static_target_rejects_wrong_table_phase():
    C=g.look_at(np.array([0,20,200]),[0,0,0],[0,1,0])
    c=Sampled([[True]],[0]);T=c.phase_pose(C,90)
    assert g.guidance(C,T,hold_still=False)['ready']
    advice=g.guidance(C,T,hold_still=True)
    assert not advice['ready'] and 'Turn the table' in advice['text']
    assert g.guidance(T,T,hold_still=True)['ready']


def test_mostly_repeated_static_view_uses_smart_frames(monkeypatch):
    c=Sampled([[True]*4],[0])
    T=g.look_at(np.array([0,20,200]),[0,0,0],[0,1,0])
    candidates=[dict(T=T,tilt=6,distance=201,height=20,mode='turntable'),dict(T=T.copy(),tilt=10,distance=201,height=20,mode='turntable')]
    monkeypatch.setattr(g,'candidates',lambda *_:candidates)
    results=[(np.array([0]),np.array([[True,True,False,False]])),(np.array([0,180]),np.array([[True,True,False,False],[True,False,True,True]]))]
    c.visibility=lambda pose,*_:results[0] if pose is T else results[1]
    r=g.plan(c,[500,500,20,20])
    assert [p['capture'] for p in r['poses']]==['Burst','Smart frames']
    p=r['poses'][1]
    assert p['hold_still'] and p['table_angle_deg']==180 and p['indices']==[2,3]
    assert p['share']==40/1040
