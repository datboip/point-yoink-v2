# SPDX-License-Identifier: AGPL-3.0-or-later
"""Distance-keyed stereo overlap edge. A dark object never defines the camera view."""
from collections import deque
import numpy as np
import cv2
from track import REGISTRATION_MIN_POINTS
from app2.scan.assist import RANGE_STEP_MM


class DepthFootprint:
    def __init__(self,window):
        self.window=float(window);self.reset();self.shape=None
    def reset(self):self.samples={};self.span=None
    def update(self,depth,scene,now):
        if self.shape!=depth.shape:self.reset()
        self.shape=depth.shape;valid=depth>0
        if scene.get('source')=='part' and scene.get('mask') is not None:valid &= ~scene['mask']
        distance=scene.get('bed_distance_mm') or scene.get('depth_mm')
        if distance is None and valid.any():distance=float(np.median(depth[valid]))*.1
        if distance is None:return None
        key=round(distance/RANGE_STEP_MM);self.span=scene.get('range_mm')
        h,w=depth.shape
        plane=scene.get('bed_plane');cal=scene.get('cal')
        if plane and cal:
            yy,xx=np.indices(depth.shape);z=depth.astype(float)*.1;n=np.asarray(plane['normal']);height=z*((xx-cal[2])*n[0]/cal[0]+(yy-cal[3])*n[1]/cal[1]+n[2])+plane['offset_mm']
            valid &= np.abs(height)<=3*max(scene.get('noise_mm',0),.1/np.sqrt(12))
        rows=valid.any(axis=1)
        if np.count_nonzero(valid)<REGISTRATION_MIN_POINTS:return None
        left=np.where(rows,valid.argmax(axis=1),np.nan);right=np.where(rows,w-1-valid[:,::-1].argmax(axis=1),np.nan)
        bucket=self.samples.setdefault(key,deque());bucket.append((float(now),left,right))
        # Keep only observations from a similar measured distance. The window
        # bounds both memory and the lifetime of a scene-dependent observation.
        for k in list(self.samples):
            q=self.samples[k]
            while q and now-q[0][0]>self.window:q.popleft()
            if not q:del self.samples[k]
        from scanner.bed import ACCEPT_HITS
        if len(bucket)<ACCEPT_HITS:return None
        a=np.stack([x[1] for x in bucket]);b=np.stack([x[2] for x in bucket]);support=np.isfinite(a).sum(0)
        yy=np.flatnonzero(support>=ACCEPT_HITS)
        if len(yy)<np.sqrt(REGISTRATION_MIN_POINTS):return None
        # Outward percentiles reject isolated farthest pixels without drawing
        # across interior holes. Only one side is the stereo overlap boundary.
        left=nan_percentile(a[:,yy],5);right=nan_percentile(b[:,yy],95)
        use_left=np.median(left)>np.median(w-1-right)
        x=left if use_left else right
        if cal:
            from app2.scan import preview
            if preview.PARAM_DIR:
                from app2.scan.stereo_view import Stereo
                stereo=Stereo.load(preview.PARAM_DIR,cal)
                if stereo is not None:
                    row_z=np.full(len(yy),distance)
                    if plane:
                        n=np.asarray(plane['normal']);den=(yy-cal[3])*n[1]/cal[1]+n[2];row_z=np.divide(-plane['offset_mm'],den,out=row_z,where=np.abs(den)>np.finfo(float).eps)
                    shift=stereo.right(np.column_stack([np.zeros(len(yy)),yy]),row_z)[:,0]
                    use_left=np.median(shift)<0
                    x=left if use_left else right
                    expected=np.maximum(0,-shift) if use_left else np.minimum(w-1,w-1-shift)
                    tolerance=1+3*cal[0]*abs(stereo.baseline)*max(scene.get('noise_mm',0),.1/np.sqrt(12))/np.maximum(row_z,1)**2
                    supported=np.abs(x-expected)<=tolerance
                    # A mat ending before the overlap cannot establish an
                    # optical boundary. Leave those rows unmarked.
                    yy,x=yy[supported],x[supported]
                    if len(yy)<np.sqrt(REGISTRATION_MIN_POINTS):return None
        # A row median follows the measured edge without joining top/bottom or
        # treating the object silhouette as the sensor rectangle.
        from scipy.ndimage import median_filter
        x=median_filter(x,size=2*ACCEPT_HITS+1,mode='nearest')
        self.side='left' if use_left else 'right';self.distance_mm=key*RANGE_STEP_MM
        return np.column_stack([x,yy]).astype(float)


def interior(depth,edge):
    mask=np.ones(depth.shape,bool)
    if edge is None:return mask
    yy=np.arange(depth.shape[0]);x=np.interp(yy,edge[:,1],edge[:,0],left=np.nan,right=np.nan)
    left=np.nanmedian(edge[:,0])<depth.shape[1]/2
    col=np.arange(depth.shape[1])[None,:]
    return np.isfinite(x[:,None]) & ((col>=x[:,None]) if left else (col<=x[:,None]))


def assess(scene,depth,footprint,ir=None):
    if footprint is not None and np.array_equal(footprint[0],footprint[-1]):footprint=None
    scene['depth_outline']=footprint;scene['missing_depth_pixels']=0;scene['missing_dark_pixels']=0
    scene['missing_extension_possible']=False;scene['missing_extension_pixels']=0
    mask=scene.get('mask')
    if scene.get('source')!='part' or mask is None or not mask.any():return scene
    # Filled connected contours establish interior holes. A convex envelope
    # bridges arms and background and cannot establish missing part depth.
    roi=np.zeros(depth.shape,np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi,contours,-1,1,cv2.FILLED)
    holes=roi.astype(bool)&interior(depth,footprint)&(depth==0)
    scene['missing_depth_pixels']=int(holes.sum())
    if ir is not None and ir.shape==depth.shape:
        scene['missing_dark_pixels']=int((holes&(ir<=np.percentile(ir[mask],25))).sum())
    scene['missing_depth']=scene['missing_depth_pixels']>=REGISTRATION_MIN_POINTS
    if footprint is not None:
        edge=np.zeros(depth.shape,np.uint8);cv2.polylines(edge,[np.rint(footprint).astype(np.int32)],False,1,1)
        edge=cv2.dilate(edge,np.ones((3,3),np.uint8)).astype(bool)
        scene['clipped']=bool(scene.get('clipped') or np.count_nonzero(mask&edge)>=np.sqrt(REGISTRATION_MIN_POINTS))
    return scene


def nan_percentile(a,q):
    """np.nanpercentile(a,q,axis=0) by one sort: NaNs sort last, so each column's finite values lead and the linear
    interpolation reads them by index. numpy's own falls back to a Python loop per column (profiled live: 8% of a frame)."""
    a=np.sort(np.asarray(a,float),axis=0);k=np.isfinite(a).sum(0);out=np.full(a.shape[1],np.nan)
    ok=k>0;pos=(k[ok]-1)*q/100.0;lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,k[ok]-1);cols=np.flatnonzero(ok)
    out[ok]=a[lo,cols]+(a[hi,cols]-a[lo,cols])*(pos-lo);return out
