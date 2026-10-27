# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded color thumbnail without a display or scanner connection."""
import numpy as np
from PIL import Image,ImageDraw


def render(V,F,C,size=(560,320)):
    W,H=size;az,el=np.radians([-35.,30.])
    rz=np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(el),-np.sin(el)],[0,np.sin(el),np.cos(el)]])
    q=V@rz.T@rx.T;distance=3.2+q[:,1]
    xy=np.column_stack((W/2+q[:,0]/distance*2.6*.95*W*.42,H*.48-q[:,2]/distance*2.6*.95*H*.42))
    normals=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-9)
    light=np.array([-.4,-.6,.7]);light/=np.linalg.norm(light)
    shade=np.clip(.3+.62*np.maximum(normals@light,0)+.12*np.maximum(-normals[:,1],0),0,1)
    colors=np.uint8(np.clip(C[F].astype(float).mean(1)*shade[:,None],0,255))
    image=Image.new('RGB',size,(22,25,30));draw=ImageDraw.Draw(image)
    for k in np.argsort(-q[F,1].mean(1)):
        draw.polygon([tuple(x) for x in xy[F[k]]],fill=tuple(colors[k]))
    return image
