# SPDX-License-Identifier: AGPL-3.0-or-later
"""One surface pick and a positioned negative shape preview.

2026-10-22 ("this editing is horrible"): another click on the surface moves the cut instead of being ignored, the
view stays in pick mode (right drag turns it), and only the sliders the chosen shape uses are shown."""
import json
import numpy as np
from app2.qt.fin_tool import FinTool
from stages.tools.drill import cut_geometry

SHAPE_PARAMS = {'cylinder': ('diameter',), 'sphere': ('diameter',), 'box': ('width', 'height')}
SIZE_PARAMS = ('diameter', 'width', 'height')
MSG_PICK = 'Click the surface where the cut goes. Right drag turns the view.'
MSG_SET = 'Red is what Apply removes. Drag an arrow to slide it, or click the surface again to move it there. The sliders set size and depth.'


class DrillTool(FinTool):
    minimum_picks = 1
    maximum_picks = None                      # another pick replaces the spot (see pick)
    def __init__(self,page):
        super().__init__(page,tab='Cut shape',preview_key='drill-preview')
        self._shape_wired=False

    def arrow_size(self):
        d=self.page.panel[self.page.stage].tab_values(self.tab)['params']
        return max(float(d.get('diameter',5.0)),float(d.get('width',5.0)),float(d.get('height',5.0)))*0.9

    def begin(self):
        super().begin()
        self._wire_shape()
        if self.source is not None: self.body().set_msg(MSG_PICK)
        self.show_shape_controls()

    def _wire_shape(self):
        if self._shape_wired: return
        b=self.body()
        if b is None: return
        c=self.page._ctl(b,param='shape')
        if c is not None: c.moved.connect(lambda _v=None:self.show_shape_controls()); self._shape_wired=True

    def show_shape_controls(self):
        """Diameter for round shapes, width and height for Box; the rest of the sliders hide."""
        b=self.body()
        if b is None: return
        c=self.page._ctl(b,param='shape'); shape=c.value() if c is not None else 'cylinder'
        want=SHAPE_PARAMS.get(shape,('diameter',))
        for name in SIZE_PARAMS:
            ctl=self.page._ctl(b,param=name)
            if ctl is not None: ctl.setVisible(name in want)

    def pick(self,point):
        if self.anchors and not self.pending: self.anchors=[]                 # a new click moves the cut
        super().pick(point)

    def _picked(self,payload):
        super()._picked(payload)
        if self.source is not None and len(self.anchors)==1 and not self.pending:
            self.page.vp.view.set_tool('fin');self.preview()

    def params(self):
        if len(self.anchors)!=1: raise ValueError('Click the surface where the cut goes first.')
        p=self.page.panel[self.page.stage].tab_values(self.tab)['params']
        return dict(p,anchors=json.dumps(self.anchors))

    def preview(self):
        if len(self.anchors)!=1:return
        try:
            import trimesh
            V,F=cut_geometry(self.params());mesh=trimesh.Trimesh(V,F,process=False)
            item=dict(key=self.preview_key,kind='mesh',V=V.astype('float32'),F=F.astype('uint32'),N=np.asarray(mesh.vertex_normals,dtype='float32'),colour=np.array([1.,.2,.15],dtype='float32'),model=self.source.get('model'))
            view=self.page.vp.view
            view.show_items([it for key,it in view.items.items() if key!=self.preview_key]+[item],edit=view.edit_key,frame=False)
            self.show_arrows()
            if hasattr(view,'set_markers'): view.set_markers([(self.world_anchor(),'#ff6655',1)])
            self.body().set_msg(MSG_SET);self.body().second.setEnabled(True)
            if getattr(self.body(), 'main', None) is not None: self.body().main.setEnabled(True)
        except ValueError as exc:self.invalid_preview(str(exc))
