# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conservative color-only foreground connected to measured part depth."""
from collections import deque
import cv2
import numpy as np
from track import REGISTRATION_MIN_POINTS


class ColorSurface:
    def __init__(self):self.history=deque();self.last=None
    def update(self,rgb,depth,part,cal,now,scene=None):
        from app2.scan import preview
        if preview.RGB_CAL is None or part is None or not np.any(part):self.history.clear();self.last=None;return None
        h,w=rgb.shape[:2];seed=np.zeros((h,w),np.uint8);seen=np.zeros((h,w),np.uint8);bed=np.zeros((h,w),np.uint8)
        v,u=np.nonzero(depth>0);x,y=preview.depth_px_to_rgb(u,v,depth[v,u].astype(float)*.1,cal);finite=np.isfinite(x)&np.isfinite(y);a=np.rint(x[finite]).astype(int);b=np.rint(y[finite]).astype(int);u=u[finite];v=v[finite];inside=(a>=0)&(a<w)&(b>=0)&(b<h);a,b,u,v=a[inside],b[inside],u[inside],v[inside]
        seen[b,a]=1;foreground=part[v,u];seed[b[foreground],a[foreground]]=1;bed[b[~foreground],a[~foreground]]=1
        if seed.sum()<REGISTRATION_MIN_POINTS or bed.sum()<REGISTRATION_MIN_POINTS:return None
        measured_background=bed.copy()
        pitch=max(1,round(max(w/depth.shape[1],h/depth.shape[0])));kernel=np.ones((2*pitch+1,2*pitch+1),np.uint8)
        if scene is not None:
            # Depth often sees only white mat dots, not the dark mat between
            # them. Train the background color on the independently measured
            # bed footprint too, excluding the observed part silhouette.
            plane=scene.get('bed_plane') or {};center=plane.get('center_mm');radius=plane.get('radius_mm')
            if center is None or radius is None:return None
            from app2.scan.rim import basis
            from scanner.bed import HINT_APART_DEG
            x,y=basis(plane['normal']);a=np.radians(np.arange(0,360,HINT_APART_DEG));P=np.asarray(center)+radius*(np.cos(a)[:,None]*x+np.sin(a)[:,None]*y)
            if np.any(P[:,2]<=0):return None
            u0=P[:,0]*cal[0]/P[:,2]+cal[2];v0=P[:,1]*cal[1]/P[:,2]+cal[3];px,py=preview.depth_px_to_rgb(u0,v0,P[:,2],cal)
            if not np.isfinite(px).all() or not np.isfinite(py).all():return None
            area=np.zeros((h,w),np.uint8);cv2.fillPoly(area,[np.rint(np.column_stack([px,py])).astype(np.int32)],1)
            contours,_=cv2.findContours(cv2.dilate(seed,kernel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(area,contours,-1,0,cv2.FILLED)
            if area.sum()<REGISTRATION_MIN_POINTS:return None
            bed=area
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        fg=lab[seed.astype(bool)];bg=lab[bed.astype(bool)];a=np.median(fg,axis=0);b=np.median(bg,axis=0)
        fs=np.maximum(1,1.482602218505602*np.median(np.abs(fg-a),axis=0));bs=np.maximum(1,1.482602218505602*np.median(np.abs(bg-b),axis=0))
        # Require independent color contrast. When colors overlap, report an
        # unknown silhouette instead of filling background from a depth hull.
        if np.linalg.norm((a-b)/np.maximum(fs,bs))<3:return None
        df=np.sum(((lab-a)/fs)**2,axis=2);db=np.sum(((lab-b)/bs)**2,axis=2);candidate=(df<db)&(df<=3**2*lab.shape[2])
        # A measured background return is stronger evidence than color
        # similarity. Close only the projection sampling gaps, and keep these
        # pixels out of the connected foreground. Otherwise a gray turntable
        # can join a gray part and its missing depth gets mislabeled as an arm.
        background=cv2.dilate(measured_background,kernel).astype(bool)
        foreground=cv2.dilate(seed,kernel).astype(bool)
        candidate[background&~foreground]=False
        count,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);ids=np.unique(labels[seed.astype(bool)]);ids=ids[ids!=0];mask=np.isin(labels,ids)
        from app2.scan.preview import Composer
        self.history.append((now,mask))
        while self.history and now-self.history[0][0]>Composer.COVER_S:self.history.popleft()
        from scanner.bed import ACCEPT_HITS
        if len(self.history)<ACCEPT_HITS:return None
        stable=mask&(np.sum([m for _,m in self.history],axis=0)>len(self.history)/2)
        # Allow projection sampling gaps of one depth pixel, not a large
        # dilation that would hide the missing arm.
        seen=cv2.dilate(seen,kernel)
        missing=stable&~seen.astype(bool);v,u=np.nonzero(missing)
        if len(u)>1.5*int(seed.sum()):self.last=None;return None   # more 'missing part' than measured part: the room matched the part's color, not a part
        self.last=dict(pixels=int(len(u)),mask=missing,part_mask=stable)
        return self.last
