# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subtract a positioned closed primitive from a closed mesh."""
import json
import os
import numpy as np
from ..methods import Tool, register
from .fin import nearest_surface

LIMITS = dict(diameter=(.2,100,5.), width=(.2,100,5.), height=(.2,100,5.), depth=(.1,500,10.),
              tilt=(-89,89,0.), rotation=(0,360,0.), offset_u=(-100,100,0.), offset_v=(-100,100,0.), inset=(-100,500,0.))


def parameters(p):
    try:
        anchors=np.asarray(json.loads(p['anchors']),dtype=float)
        values={key:float(p.get(key,default)) for key,(_,_,default) in LIMITS.items()}
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Pick a surface spot and valid cut sizes.') from exc
    if anchors.shape!=(1,2,3) or not np.isfinite(anchors).all():
        raise ValueError('Pick one spot on the current mesh.')
    for key,value in values.items():
        low,high,_=LIMITS[key]
        if not np.isfinite(value) or not low<=value<=high:
            raise ValueError('%s must be between %g and %g.' % (key.replace('_',' ').capitalize(),low,high))
    shape=p.get('shape','cylinder')
    if shape not in ('cylinder','box','sphere'): raise ValueError('Choose Cylinder, Box, or Sphere.')
    length=np.linalg.norm(anchors[0,1])
    if length<1e-8: raise ValueError('Pick a spot with a valid surface direction.')
    anchors[0,1]/=length
    return anchors,values,shape


def cut_geometry(p):
    import trimesh
    anchors,q,shape=parameters(p)
    point,normal=anchors[0]
    seed=np.eye(3)[np.argmin(np.abs(normal))]
    u=seed-normal*np.dot(seed,normal);u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    angle,tilt=np.radians([q['rotation'],q['tilt']])
    tangent=u*np.cos(angle)+v*np.sin(angle)
    axis=-normal*np.cos(tilt)+tangent*np.sin(tilt)
    across=u*(-np.sin(angle))+v*np.cos(angle)
    local_x=np.cross(across,axis)
    start=point+u*q['offset_u']+v*q['offset_v']+axis*q['inset']
    margin=max(.001,min(q['diameter']*.01,.05))
    transform=np.eye(4);transform[:3,:3]=np.column_stack((local_x,across,axis))
    if shape=='sphere':
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=q['diameter']/2)
        transform[:3,3]=start+axis*(q['diameter']/2-margin)
    else:
        transform[:3,3]=start+axis*(q['depth']-margin)/2
        if shape=='box': mesh=trimesh.creation.box(extents=(q['width'],q['height'],q['depth']+margin))
        else:
            mesh=trimesh.creation.cylinder(radius=q['diameter']/2,height=q['depth']+margin,sections=96)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/96,[0,0,1]))
    mesh.apply_transform(transform)
    return np.asarray(mesh.vertices),np.asarray(mesh.faces)


def cut_shape(V,F,p):
    import trimesh
    import pymeshlab as ml
    body=trimesh.Trimesh(V,F,process=True);body.fix_normals()
    if not body.is_watertight or not body.is_volume:
        raise ValueError('Cut shape needs a closed mesh. Fill holes or use Finish for print first.')
    anchors,q,_=parameters(p)
    _,distance,indices=nearest_surface(body.vertices,body.faces,anchors[:,0])
    if distance[0]>.01: raise ValueError('The spot is not on the current mesh. Pick it again.')
    if np.dot(anchors[0,1],body.face_normals[indices[0]])<-.05:
        raise ValueError('The spot faces into the part. Pick it again.')
    cv,cf=cut_geometry(p)
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(np.asarray(body.vertices),np.asarray(body.faces)),'Part')
    ms.add_mesh(ml.Mesh(cv,cf),'Cut shape')
    try: ms.generate_boolean_difference(first_mesh=0,second_mesh=1)
    except Exception as exc: raise ValueError('Could not cut this shape. Adjust its size or position.') from exc
    result=ms.current_mesh()
    if result.face_number()==0: raise ValueError('This cut removes the whole part. Use a smaller shape.')
    out=trimesh.Trimesh(result.vertex_matrix(),result.face_matrix(),process=True);out.fix_normals()
    if not out.is_watertight or not out.is_volume:
        raise ValueError('The cut could not close its new faces. Adjust its size or position.')
    threshold=max(1e-8,abs(body.volume)*1e-9)
    if out.volume>=body.volume-threshold: raise ValueError('The shape missed the part. Adjust its position or depth.')
    if out.volume<=threshold: raise ValueError('This cut removes the whole part. Use a smaller shape.')
    return np.asarray(out.vertices),np.asarray(out.faces)


@register
class Drill(Tool):
    name='drill';label='Cut shape';kinds=('mesh',);version=1;mem_gb=6
    scripts=();modules=('tools/drill.py','tools/fin.py');libs=('pymeshlab','trimesh')

    def execute(self,ctx):
        from ..plyio import read_mesh,write_mesh
        V,F,_=read_mesh(ctx.inputs['parent'].files['mesh'])
        V,F=cut_shape(V,F,ctx.params)
        write_mesh(os.path.join(ctx.out_dir,'mesh.ply'),V,F)
        return {'outputs':{'mesh':'mesh.ply'},'stats':{'faces':len(F)}}

    def summary(self,p): return '%s cut' % p.get('shape','cylinder').capitalize()
