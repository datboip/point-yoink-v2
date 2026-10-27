# SPDX-License-Identifier: AGPL-3.0-or-later
"""Color dots lifted onto a measured bed plane, without inventing depth."""
import cv2
import numpy as np
from scanner import bed

# A quarter turn makes the center fit observable from distinct orientations.
COLOR_LEARN_TURN_DEG=90.
OUTLINE_SAMPLES=64
RESUME_PLANES=3  # bed, part, and background may all be visible after moving back


def observations(rgb, calibration, plane):
    if rgb is None or calibration is None or plane is None:return None
    K,R,T=calibration;n,off=plane[:2]
    dots=bed.detect_dots(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    if not len(dots):return None
    rays=np.column_stack([dots[:,:2],np.ones(len(dots))])@np.linalg.inv(K).T@R
    origin=-np.asarray(T)@R
    den=rays@np.asarray(n)
    with np.errstate(divide='ignore',invalid='ignore'):
        distance=-(origin@np.asarray(n)+off)/den
        points=origin+distance[:,None]*rays
    valid=np.isfinite(points).all(1)&(distance>0)&(points[:,2]>0)
    return dots[valid],points[valid]


def solve_pose(rgb,calibration,plane,model):
    if model is None or not bed.has_dots(model):return None
    sample=observations(rgb,calibration,plane)
    if sample is None:return None
    dots,points=sample;n,off=plane[:2]
    if len(plane)>2 and plane[2] is not None:
        points=points[np.linalg.norm(points-np.asarray(plane[2]),axis=1)<=model.radius_mm+(model.dot_diam_mm or 0)/2]
    if len(points)<bed.HIT_MIN:return None
    x,y=bed.basis(n);p=np.column_stack([points@x,points@y])
    reg=bed.register_2d(p,model.map2d)
    if not bed.is_hit(reg):return None
    B,center,axis=bed.bed_frame(n,off,reg.R,reg.t)
    from app2.scan.rim import basis
    x,y=basis(n)
    span=min(np.ptp(p[reg.mask],axis=0))
    return dict(angle_deg=float(np.degrees(np.arctan2(axis@y,axis@x)))%360,B_cam=B,inliers=int(reg.inliers),runner_up=int(reg.runner_up),rms_mm=float(reg.rms),span_mm=float(span),source='Color bed dots')


def depth_pose(rgb,calibration,depth,cal,model):
    """Try measured planes; a visible floor must not disqualify the bed.

    Color must match the physical dot map on a plane supported by depth.
    Incompatible color placements on different planes remain uncertain.
    """
    v,u=np.nonzero(depth[::2,::2]);v*=2;u*=2
    if len(u)<bed.SURFACE_MIN:return None
    points=bed.backproject(u.astype(float),v.astype(float),depth[v,u]*.1,cal)
    points=points[np.linspace(0,len(points)-1,min(len(points),20000),dtype=int)]
    candidates=[]
    for _ in range(RESUME_PLANES):
        plane=bed.ransac_plane(points,bed.SURFACE_BAND,bed.RANSAC_RING[1],bed.SURFACE_MIN)
        if plane is None:break
        n,off,inside=plane
        hit=solve_pose(rgb,calibration,(n,off),model)
        if hit is not None:candidates.append(hit)
        points=points[~inside]
        if len(points)<bed.SURFACE_MIN:break
    if not candidates:return None
    from app2.scan.bed_resume import difference
    best=max(candidates,key=lambda c:(c['inliers'],-c['rms_mm']))
    if any(difference(best['B_cam'],c['B_cam'])[0]>bed.AGREE_CENTRE or difference(best['B_cam'],c['B_cam'])[1]>bed.AGREE_DEG for c in candidates):return None
    return best


class ColorBedLearner(bed.BedLearner):
    """The IR learner's registration and validation, with extra color sights.

    IR depth still establishes the plane. Color candidates close to an IR dot
    are omitted, so two cameras never count as two independent sightings.
    """
    def __init__(self,*args,radius_mm=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.color_frames={};self.color_stats=[];self.radius_mm=radius_mm
        self.full_views=[];self.fast_check=None;self.completion='full turn'

    def ready(self):
        if super().ready():return True
        if self.turned()<COLOR_LEARN_TURN_DEG or len(self.full_views)<bed.ACCEPT_HITS:return False
        if np.degrees(np.ptp([v[0] for v in self.full_views]))<COLOR_LEARN_TURN_DEG:return False
        C,_,_=self._kept()
        if len(C)<bed.LEARN_MIN_DOTS:return False
        # Each kept dot must occur in three separated color-supported views.
        counts=np.zeros(len(C),int)
        for _,points,_ in self.full_views:
            counts+=(np.linalg.norm(C[:,None]-points[None],axis=2).min(1)<bed.INLIER_TOL)
        if np.any(counts<bed.ACCEPT_HITS):return False
        centers=np.array([v[2] for v in self.full_views])
        if np.max(np.linalg.norm(centers-np.median(centers,axis=0),axis=1))>bed.AGREE_CENTRE:return False
        self._check_map()
        check=self.map_check or {}
        if check.get('seen',0)<bed.LEARN_SEEN_MIN or check.get('halves',0)<bed.LEARN_HALVES_MIN:return False
        self.fast_check=dict(views=len(self.full_views),minimum_dot_views=int(counts.min()),turn_deg=self.turned(),**check)
        self.completion='full color views'
        return True

    def _full_color_view(self,calibration,shape):
        if self.radius_mm is None or self.c_est is None:return False
        n,off=self.plane;x,y=self.e
        angles=np.linspace(0,2*np.pi,OUTLINE_SAMPLES,endpoint=False)
        points=(self.c_est[0]+self.radius_mm*np.cos(angles))[:,None]*x+(self.c_est[1]+self.radius_mm*np.sin(angles))[:,None]*y-off*n
        K,R,T=calibration;q=(points@R.T+T)@K.T
        if np.any(q[:,2]<=0):return False
        xy=q[:,:2]/q[:,2:];h,w=shape[:2]
        return bool(np.all((xy>=bed.DOT_BORDER)&(xy<np.array([w,h])-bed.DOT_BORDER)))

    def add_color(self,d,ir,color,calibration,**kwargs):
        self.color_frames[id(d)]=(color,calibration)
        return self.add(d,ir,**kwargs)

    def _register(self,dl,P3,d,ir,fr=None):
        packet=self.color_frames.pop(id(d),None)
        extra=0;color_aligned=False;rgb=None
        if packet is not None and self.plane is not None:
            from app2.scan.preview import decoded
            jpeg,calibration=packet
            rgb=decoded(jpeg) if jpeg is not None else None
            sample=observations(rgb,calibration,self.plane) if rgb is not None else None
            if sample is not None:
                color_dots,points=sample
                lifted=bed.lift(dl,self.cal,*self.plane)
                good=np.isfinite(lifted).all(1)
                # Shared dots measure the inter-camera bias before color adds
                # anything to the map. An uncertain fit keeps IR-only learning.
                n,off=self.plane;x,y=bed.basis(n)
                color2=np.column_stack([points@x,points@y])
                ir2=np.column_stack([lifted[good]@x,lifted[good]@y])
                reg=bed.register_2d(color2,ir2) if min(len(color2),len(ir2))>=bed.HIT_MIN else None
                if not bed.is_hit(reg):points=points[:0];color_dots=color_dots[:0]
                else:
                    color_aligned=True
                    aligned=color2@reg.R.T+reg.t
                    if self.radius_mm is not None and self.c_est is not None:
                        inside=np.linalg.norm(aligned-self.c_est,axis=1)<=self.radius_mm+bed.INLIER_TOL
                        aligned=aligned[inside];color_dots=color_dots[inside]
                    points=aligned[:,:1]*x+aligned[:,1:]*y-off*np.asarray(n)
                if good.any() and len(points):
                    from scipy.spatial import cKDTree
                    distance=cKDTree(lifted[good]).query(points)[0]
                    keep=distance>bed.INLIER_TOL
                    points=points[keep];color_dots=color_dots[keep]
                if len(points):
                    fx,fy,cx,cy=self.cal
                    radius=color_dots[:,2]*fx/np.sqrt(calibration[0][0,0]*calibration[0][1,1])
                    pixels=np.zeros((len(points),dl.shape[1]),float)
                    pixels[:,:3]=np.column_stack([fx*points[:,0]/points[:,2]+cx,fy*points[:,1]/points[:,2]+cy,radius])
                    if dl.shape[1]>=6:pixels[:,3:6]=np.column_stack([np.pi*radius**2,2*radius,2*radius])
                    dl=np.vstack([dl,pixels]);P3=np.vstack([P3,points]);extra=len(points)
                    if fr is not None:fr=np.r_[fr,np.zeros(extra,fr.dtype)]
        result=super()._register(dl,P3,d,ir,fr)
        full=result is not None and color_aligned and self._full_color_view(calibration,rgb.shape)
        if full and (not self.full_views or abs(self.angs[-1]-self.full_views[-1][0])>=np.radians(bed.HINT_APART_DEG)):
            _,points,R,t=self.obs[-1]
            self.full_views.append((self.angs[-1],points@R.T+t,self.c_est.copy()))
        self.color_stats.append(dict(frame=self._fno,extra_dots=extra,full_view=full))
        return result
