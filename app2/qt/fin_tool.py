# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surface picks and a live red fin preview in the existing Build scene."""
import json
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool, Qt
from stages.tools.fin import rib_geometry, nearest_surface


class SurfacePick(QRunnable):
    def __init__(self, owner, generation, source, point):
        super().__init__()
        self.owner, self.generation, self.source, self.point = owner, generation, source, point

    def run(self):
        try:
            V, F = self.source['V'], self.source['F']
            closest, _, indices = nearest_surface(V, F, [self.point])
            tri = V[F[indices[0]]]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            size = np.linalg.norm(normal)
            if size < 1e-12: raise ValueError('Pick a different surface spot.')
            payload = (self.generation, [closest[0].tolist(), (normal / size).tolist()], None)
        except Exception as exc:
            payload = (self.generation, None, str(exc))
        try: self.owner.picked.emit(payload)
        except RuntimeError: pass  # The window may have closed while the pick was running.


class FinTool(QObject):
    minimum_picks = 3
    maximum_picks = None
    picked = Signal(object)

    def __init__(self, page, tab='Add fin', preview_key='fin-preview'):
        super().__init__(page)
        self.tab, self.preview_key = tab, preview_key
        self.page=page; self.anchors=[]; self.source=None
        self.generation=0; self.pending=False
        self.picked.connect(self._picked, Qt.QueuedConnection)
        self._arrow_base=None
        if page.gl: page.vp.view.surface_picked.connect(self.pick); page.vp.view.arrow_move.connect(self.arrow_move)

    def body(self): return getattr(self.page.panel.get(self.page.stage), 'bodies', {}).get(self.tab)

    def clear(self):
        self.generation += 1; self.pending=False
        if self.page.gl:
            v=self.page.vp.view
            if v.gizmo is not None and v.gizmo.get('owner')==self.preview_key: v.set_gizmo(None)
            if self.preview_key in v.items:
                v.show_items([it for k,it in v.items.items() if k!=self.preview_key],edit=v.edit_key,frame=False)
            if v.tool=='fin': v.set_tool('orbit')
        self.anchors=[];self.source=None
        if self.page.gl and hasattr(self.page.vp.view, 'set_markers'): self.page.vp.view.set_markers([])
        body = self.body()
        if body is not None and getattr(body, 'main', None) is not None: body.main.setEnabled(False)

    def begin(self):
        self.clear(); p=self.page; b=self.body()
        if not p.gl or p.comparing or not p.vp.view.tools_on:
            b.set_msg('Load an editable mesh and leave Compare first.');return
        v=p.vp.view; item=v.items.get(v.edit_key)
        if not item or item['kind']!='mesh': b.set_msg('Build a mesh first.');return
        self.source=item; v.set_tool('fin'); b.second.setEnabled(True)
        b.set_msg('Pick the first attachment spot. Right drag turns the view.')

    def pick(self, point):
        if self.source is None or self.pending or (self.maximum_picks is not None and len(self.anchors) >= self.maximum_picks) or self.page.vp.view.tool != 'fin': return
        self.pending=True
        self.page.vp.view.set_tool('orbit')
        self.body().set_msg('Finding the attachment surface...')
        M = np.asarray(self.source.get('model', np.eye(4)), float)
        local = np.linalg.solve(M[:3,:3], np.asarray(point)-M[:3,3])
        QThreadPool.globalInstance().start(SurfacePick(self, self.generation, self.source, local))

    @Slot(object)
    def _picked(self, payload):
        generation, anchor, error = payload
        if generation != self.generation or self.source is None: return
        self.pending=False
        view = self.page.vp.view
        if (view.items.get(view.edit_key) is not self.source or
                view.tool != 'orbit' or getattr(self.page, 'comparing', False) or
                not getattr(view, 'tools_on', True)):
            self.clear(); return
        body = self.body()
        if body is None:
            self.clear(); return
        if error:
            body.set_msg(error); view.set_tool('fin'); return
        self.anchors.append(anchor)
        if hasattr(view, 'set_markers'):
            M = np.asarray(self.source.get('model', np.eye(4)), float)
            view.set_markers([(M[:3,:3] @ np.asarray(a[0]) + M[:3,3], '#ff6655', i+1) for i,a in enumerate(self.anchors)])
        if len(self.anchors)==1:
            body.set_msg('Pick the second attachment spot.')
            view.set_tool('fin'); return
        if len(self.anchors) < self.minimum_picks:
            body.set_msg('Pick a third spot to set which way the fin faces. Apply needs three spots.')
            view.set_tool('fin'); return
        self.preview()
        view.set_tool('fin' if self.maximum_picks is None else 'orbit')

    # ---- move arrows on the last spot picked (Bambu's move gizmo): drag one to slide it along that world axis
    def _model(self): return np.asarray(self.source.get('model',np.eye(4)) if self.source and self.source.get('model') is not None else np.eye(4),float)

    def world_anchor(self, i=-1):
        M=self._model(); a=np.asarray(self.anchors[i][0],float); return M[:3,:3]@a+M[:3,3]

    def arrow_size(self): return 8.0

    def show_arrows(self):
        if not self.page.gl or self.source is None or not self.anchors: return
        v=self.page.vp.view; v.set_gizmo(self.world_anchor(),max(6.0,self.arrow_size()),rings=False,arrows=True); v.gizmo['owner']=self.preview_key

    def arrow_move(self, k, mm, done):
        v=self.page.vp.view
        if self.source is None or not self.anchors or v.gizmo is None or v.gizmo.get('owner')!=self.preview_key: return
        from app2.qt.rotate_gizmo import AXES
        if self._arrow_base is None: self._arrow_base=np.asarray(self.anchors[-1][0],float)
        step=np.linalg.solve(self._model()[:3,:3],AXES[k]*mm)            # a world-axis slide in the model's own coordinates
        self.anchors[-1]=[[float(x) for x in self._arrow_base+step],self.anchors[-1][1]]
        if done: self._arrow_base=None
        if hasattr(v,'set_markers'):
            M=self._model(); v.set_markers([(M[:3,:3]@np.asarray(a[0])+M[:3,3],'#ff6655',i+1) for i,a in enumerate(self.anchors)])
        if len(self.anchors)>=self.minimum_picks: self.preview()
        else: self.show_arrows()

    def invalid_preview(self, message):
        if self.page.gl:
            view = self.page.vp.view
            if self.preview_key in view.items:
                view.show_items([item for key, item in view.items.items() if key != self.preview_key],
                                edit=view.edit_key, frame=False)
        self.body().set_msg(message)
        if getattr(self.body(), 'main', None) is not None: self.body().main.setEnabled(False)

    def params(self):
        if len(self.anchors) < self.minimum_picks: raise ValueError('Pick three spots to set the fin plane first.' if self.minimum_picks == 3 else 'Pick two attachment spots first.')
        p=self.page.panel[self.page.stage].tab_values(self.tab)['params']
        return dict(p,anchors=json.dumps(self.anchors))

    def preview(self):
        if len(self.anchors) < self.minimum_picks:return
        try:
            import trimesh
            V,F=rib_geometry(self.params()); mesh=trimesh.Trimesh(V,F,process=False)
            item=dict(key=self.preview_key,kind='mesh',V=V.astype('float32'),F=F.astype('uint32'),N=np.asarray(mesh.vertex_normals,dtype='float32'),colour=np.array([1.,.2,.15],dtype="float32"), model=self.source.get('model'))
            v=self.page.vp.view
            v.show_items([it for key,it in v.items.items() if key!=self.preview_key]+[item],edit=v.edit_key,frame=False)
            self.show_arrows()
            self.body().set_msg('Red fin preview. Drag an arrow to slide the last spot. Apply joins it to the mesh.');self.body().second.setEnabled(True)
            if getattr(self.body(), 'main', None) is not None: self.body().main.setEnabled(True)
        except ValueError as e:self.invalid_preview(str(e))
