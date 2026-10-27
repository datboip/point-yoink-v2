# SPDX-License-Identifier: AGPL-3.0-or-later
"""A local slice that caps the cut faces without filling the intended gap."""
import numpy as np
from app2.qt.fin_tool import FinTool
from stages.tools.slice_gap import slot_geometry


class SliceTool(FinTool):
    minimum_picks = 2
    maximum_picks = 2
    def __init__(self,page): super().__init__(page,tab='Cut gap',preview_key='slice-preview')

    def begin(self):
        super().begin()
        if self.source is not None: self.body().set_msg('Pick the first end of the cut on the mesh.')

    def preview(self):
        if len(self.anchors)!=2: return
        try:
            import trimesh
            V,F=slot_geometry(self.params());mesh=trimesh.Trimesh(V,F,process=False)
            item=dict(key=self.preview_key,kind='mesh',V=V.astype('float32'),F=F.astype('uint32'),N=np.asarray(mesh.vertex_normals,dtype='float32'),colour=np.array([1.,.2,.15],dtype='float32'),model=self.source.get('model'))
            v=self.page.vp.view
            v.show_items([it for key,it in v.items.items() if key!=self.preview_key]+[item],edit=v.edit_key,frame=False)
            self.body().set_msg('Red marks the cut. Apply keeps the gap open and closes its cut faces.');self.body().second.setEnabled(True)
            if getattr(self.body(), 'main', None) is not None: self.body().main.setEnabled(True)
        except ValueError as e:self.invalid_preview(str(e))
