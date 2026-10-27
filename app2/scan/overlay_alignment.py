# SPDX-License-Identifier: AGPL-3.0-or-later
"""Color overlay transforms only. Depth, poses and tracking never use these matrices."""
import numpy as np
from app2.scan.fmt import W,H


def manual_matrix(values,size=(W,H)):
    w,h=size;dx=float(values.get('x',0));dy=float(values.get('y',0));angle=np.deg2rad(float(values.get('angle',0)));scale=float(values.get('scale',1))
    if not np.isfinite([dx,dy,angle,scale]).all() or scale<=0:raise ValueError('Overlay values must be finite with positive scale.')
    # Manual movement is bounded by the image, rotation by a quarter turn.
    dx=np.clip(dx,-w/2,w/2);dy=np.clip(dy,-h/2,h/2);angle=np.clip(angle,-np.pi/2,np.pi/2);scale=np.clip(scale,.5,2)
    A=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=np.array([w/2,h/2]);M=np.eye(3);M[:2,:2]=A;M[:2,2]=center+np.array([dx,dy])-A@center;return M


def apply(points,matrix):
    p=np.asarray(points,float);return p@np.asarray(matrix)[:2,:2].T+np.asarray(matrix)[:2,2]


def refine(color):
    """Bound a similarity fit by the detected dot radius, never a wrong camera mode.

    Requires at least the bed match minimum, resolved spread in two dimensions,
    and every fitted movement within the median observed dot radius. Improvement
    must exceed the residual standard error; outliers beyond 3 MAD are refit once.
    """
    from scanner.bed import HIT_MIN
    if not color or not color.get('accepted'):return None
    p=np.asarray(color.get('projected_points',[]),float);q=np.asarray(color.get('circle_points',[]),float)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1:]!=(2,) or len(p)<HIT_MIN:return None
    radius=float(np.median(color.get('radii',[0])))
    if radius<=0 or not np.isfinite(p).all() or not np.isfinite(q).all():return None
    def fit(a,b):
        x=a-a.mean(0);y=b-b.mean(0);U,S,Vt=np.linalg.svd(x.T@y);R=U@Vt
        if np.linalg.det(R)<=0:return None
        scale=S.sum()/np.sum(x*x);M=np.eye(3);M[:2,:2]=scale*R.T;M[:2,2]=b.mean(0)-scale*a.mean(0)@R;return M
    spread=np.linalg.svd(p-p.mean(0),compute_uv=False)
    if spread[-1]<=radius:return None
    M=fit(p,q)
    if M is None:return None
    residual=np.linalg.norm(apply(p,M)-q,axis=1);mad=1.482602218505602*np.median(np.abs(residual-np.median(residual)))
    roundoff=np.finfo(float).eps*max(1,float(np.abs(q).max()))*len(q)
    keep=residual<=np.median(residual)+max(3*mad,roundoff)
    if keep.sum()<HIT_MIN:return None
    M=fit(p[keep],q[keep]);moved=np.linalg.norm(apply(p,M)-p,axis=1)
    before=np.linalg.norm(p[keep]-q[keep],axis=1);after=np.linalg.norm(apply(p[keep],M)-q[keep],axis=1)
    improvement=float(np.median(before)-np.median(after));error=float(np.std(after)/np.sqrt(len(after)))
    if moved.max()>radius or improvement<=max(error,np.finfo(float).eps):return None
    return dict(matrix=M.tolist(),before_px=float(np.median(before)),after_px=float(np.median(after)),matches=int(keep.sum()),bound_px=radius)
