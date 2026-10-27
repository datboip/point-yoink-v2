# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured circular rim and optional reflective rim angle markers."""
from collections import deque
import numpy as np
import cv2


def basis(normal,x_axis=None):
    n=np.asarray(normal,float);n=n/np.linalg.norm(n)
    x=np.cross(n,[0,1,0]) if x_axis is None else np.asarray(x_axis,float)
    x=x-n*(x@n)
    if np.linalg.norm(x)<np.finfo(float).eps:x=np.cross(n,[1,0,0])
    x=x/np.linalg.norm(x);return x,np.cross(n,x)


class RimFit:
    def __init__(self,window):self.window=window;self.rows=deque();self.result=None
    def update(self,depth,cal,geo,noise,now):
        if geo is None or geo[2] is None or not np.isscalar(geo[3]):return self.result
        n,off,center,nominal=geo;n=np.asarray(n);center=np.asarray(center);v,u=np.nonzero(depth);z=depth[v,u].astype(float)*.1;p=np.column_stack([(u-cal[2])*z/cal[0],(v-cal[3])*z/cal[1],z]);height=p@n+off;delta=p-center;rad=np.linalg.norm(delta-np.outer(delta@n,n),axis=1)
        on=(np.abs(height)<=3*max(noise,.1/np.sqrt(12)))&(rad>=nominal/2)&(rad<=2*nominal)
        # Use boundary pixels of the measured plane, rather than all interior
        # plane pixels. An area percentile alone shrinks a perfectly observed
        # disk even when its edge is clear.
        plane=np.zeros(depth.shape,np.uint8);plane[v[on],u[on]]=1
        boundary=plane&~cv2.erode(plane,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        on &= boundary[v,u].astype(bool)
        x,y=basis(n);angle=np.arctan2(delta@y,delta@x)
        from scanner.bed import HINT_APART_DEG,ACCEPT_HITS
        bins=np.floor((angle+np.pi)/np.radians(HINT_APART_DEG)).astype(int);outer=[]
        for key in np.unique(bins[on]):
            q=rad[on&(bins==key)]
            if len(q)>=ACCEPT_HITS:outer.append(np.percentile(q,95))
        if len(outer)<ACCEPT_HITS:return self.result
        outer=np.asarray(outer)
        # Occluded sectors end at interior holes, not at the platter edge.
        # Require independent angular sectors agreeing near the learned rim.
        # A noisy spread must never enlarge the allowed departure from it.
        from app2.scan.bedcut import RIM_MM
        outer=outer[np.abs(outer-nominal)<=RIM_MM]
        pitch=float(np.median(z[on]))/min(cal[:2]);tolerance=3*max(noise,pitch)
        groups=[outer[np.abs(outer-value)<=tolerance] for value in outer]
        groups=[g for g in groups if len(g)>=ACCEPT_HITS]
        if not groups:return self.result
        group=min(groups,key=lambda g:(abs(float(np.median(g))-nominal),-len(g)))
        r=float(np.median(group));spread=float(1.482602218505602*np.median(np.abs(group-r)))
        # If only interior cutoffs are visible, retain the last supported fit.
        # A learned rim must not shrink by several millimeters on that evidence.
        if abs(r-nominal)>tolerance:return self.result
        self.rows.append((now,r,n,off,center))
        while self.rows and now-self.rows[0][0]>self.window:self.rows.popleft()
        if len(self.rows)<ACCEPT_HITS:return self.result
        radii=np.array([row[1] for row in self.rows]);median=float(np.median(radii));sigma=max(noise,1.482602218505602*float(np.median(np.abs(radii-median))));rows=[row for row in self.rows if abs(row[1]-median)<=3*sigma]
        normal=np.median([row[2] for row in rows],axis=0);normal/=np.linalg.norm(normal)
        self.result=dict(radius_mm=float(np.median([row[1] for row in rows])),spread_mm=max(sigma,spread),frames=len(rows),normal=normal.tolist(),offset_mm=float(np.median([row[3] for row in rows])),center=np.median([row[4] for row in rows],axis=0).tolist())
        return self.result
    def geometry(self,original):
        if self.result is None:return original
        r=self.result;return np.asarray(r['normal']),r['offset_mm'],np.asarray(r['center']),r['radius_mm']


def detect_markers(depth,ir,cal,geo,noise,diameter,band=None):
    """Use measured bright blobs in a known cylinder and below-plane band.

    Unknown marker height is learned only from actual rim depth. A missing
    depth return never substitutes a guessed point or a top-dot pose.
    """
    if geo is None or geo[2] is None or not np.isscalar(geo[3]) or not diameter:return []
    n,off,c,r=geo;n=np.asarray(n);c=np.asarray(c);x,y=basis(n);bright=(ir==np.iinfo(ir.dtype).max).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(bright,8);found=[]
    for k in range(1,count):
        vv,uu=np.nonzero((labels==k)&(depth>0))
        if not len(uu):continue
        z=depth[vv,uu].astype(float)*.1;p=np.column_stack([(uu-cal[2])*z/cal[0],(vv-cal[3])*z/cal[1],z]);p=np.median(p,axis=0);h=float(p@n+off);d=p-c;rad=float(np.linalg.norm(d-(d@n)*n))
        if abs(rad-r)>diameter/2+3*noise or h>=-3*noise:continue
        if band is not None and not band[0]<=h<=band[1]:continue
        found.append(dict(angle_deg=float(np.degrees(np.arctan2(d@y,d@x)))%360,height_mm=h,pixel=centroids[k].tolist(),size_px=float(np.sqrt(stats[k,cv2.CC_STAT_AREA]))))
    # One blob marks start. Only a spatially adjacent pair marks 180 degrees.
    if len(found)==1:return [dict(found[0],reference_deg=0)]
    if len(found)==2:
        a,b=found;delta=abs((a['angle_deg']-b['angle_deg']+180)%360-180);limit=np.degrees(2*diameter/max(r,np.finfo(float).eps))
        if delta<=limit:
            z=np.mean(np.exp(1j*np.radians([a['angle_deg'],b['angle_deg']])));return [dict(angle_deg=float(np.degrees(np.angle(z)))%360,height_mm=float(np.mean([a['height_mm'],b['height_mm']])),pixel=np.mean([a['pixel'],b['pixel']],axis=0).tolist(),reference_deg=180)]
    return []


class RimMarkers:
    def __init__(self,learned=None):self.learned=learned;self.samples=[];self.last=None
    def observe(self,markers,top_angle=None,learn=False):
        if len(markers)!=1:self.last=None;return None
        marker=markers[0];phase=(marker['angle_deg']-marker['reference_deg'])%360
        if learn and top_angle is not None:self.samples.append(dict(offset_deg=(phase-top_angle)%360,height_mm=marker['height_mm']))
        if self.learned is None:self.last=None;return None
        angle=(phase-self.learned['offset_deg'])%360;self.last=dict(angle_deg=angle,source='rim markers',reference_deg=marker['reference_deg']);return self.last
    def finish(self,noise):
        from scanner.bed import ACCEPT_HITS
        if len(self.samples)<ACCEPT_HITS:return None
        a=np.radians([s['offset_deg'] for s in self.samples]);center=np.angle(np.mean(np.exp(1j*a)));residual=np.angle(np.exp(1j*(a-center)));spread=1.482602218505602*np.median(np.abs(residual-np.median(residual)));keep=np.abs(residual)<=max(3*spread,np.finfo(float).eps)
        if keep.sum()<ACCEPT_HITS:return None
        heights=np.array([s['height_mm'] for s in self.samples])[keep];median=float(np.median(heights));width=max(3*noise,3*1.482602218505602*float(np.median(np.abs(heights-median))))
        self.learned=dict(offset_deg=float(np.degrees(np.angle(np.mean(np.exp(1j*a[keep])))))%360,band_mm=[median-width,median+width],spread_deg=float(np.degrees(spread)),samples=int(keep.sum()))
        return self.learned
