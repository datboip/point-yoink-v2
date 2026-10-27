# SPDX-License-Identifier: AGPL-3.0-or-later
"""Worker-owned scene guides and explicit framing actions for ScanSession."""
import copy
import math
import numpy as np
from app2.scan.preview import Composer
from app2.scan.framing import Framing,analyze
from app2.scan.depth_footprint import DepthFootprint,assess
from app2.scan.range_guide import from_scene,SteadyGuide,setup_check


class FramingSession:
    def _ensure_framing(self):
        if not hasattr(self,'framer'):
            self.framer=Framing(Composer.COVER_S);self.depth_footprint=DepthFootprint(Composer.COVER_S);self.guide_hold=SteadyGuide(Composer.COVER_S)
            self.framing_view=None;self.scene_setup=None;self.frame_range_undo=None
    def _reset_framing(self,preserve_undo=False):
        self._ensure_framing();undo=self.frame_range_undo if preserve_undo else None
        self.framer.reset();self.depth_footprint.reset();self.guide_hold=SteadyGuide(Composer.COVER_S)
        self.framing_view=None;self.scene_setup=None;self.frame_range_undo=undo;self._overlay_last=None;self._last_framing_geo=None
    def _frame_geometry(self):
        live=getattr(self,'_bed_live',None) or {}
        if live.get('bed_seen') and live.get('geo') is not None:return live['geo']
        geo=getattr(self,'_overlay_last',None)
        if geo is not None:self._last_framing_geo=geo;return geo
        evidence=getattr(self,'mode_evidence',{})
        mode=evidence.get('mode')
        if mode!=getattr(self,'scan_mode','near'):return getattr(self,'_last_framing_geo',None)
        rows=evidence.get('frames') or []
        for row in reversed(rows):
            plane=row.get(mode,{}).get('plane')
            if plane:
                rim=plane.get('radius_mm')
                if plane.get('outline') and plane.get('x_axis') is not None:
                    from scanner.bed import Outline
                    outline=Outline.from_json(plane['outline'],rim)
                    rim=outline.radius if outline.centred_circle else outline.placed(np.asarray(plane['x_axis']))
                self._last_framing_geo=(np.asarray(plane['normal']),plane['offset_mm'],np.asarray(plane['center']) if plane.get('center') is not None else None,rim)
                return self._last_framing_geo
        return getattr(self,'_last_framing_geo',None)
    def _measure_scene(self,fr):
        self._ensure_framing()
        ir=fr.irl
        if getattr(self.source,'kind',None)=='replay' and not np.any(ir):ir=None
        scene=analyze(fr.depth,self.cal,self._frame_geometry(),(self.near,self.far),ir=ir)
        return assess(scene,fr.depth,self.depth_footprint.update(fr.depth,scene,fr.t),ir)
    def _update_scene(self,fr,scene):
        self._ensure_framing()
        # Auto range may have changed since the geometry was measured.
        lo,hi=sorted((self.near,self.far));inside=(fr.depth>0)&(fr.depth*.1>=lo)&(fr.depth*.1<=hi)
        scene['kept_part_pixels']=int((inside&scene['mask']).sum());scene['background_pixels']=int((inside&~scene['mask']).sum())
        scene['cut_off']=bool(scene['range_mm'] is not None and (scene['range_mm'][0]<lo or scene['range_mm'][1]>hi))
        guide=from_scene(fr.depth,getattr(self,'scan_mode','near'),scene,getattr(self,'range_profile',None));self.range_guide=self.guide_hold.update(guide,fr.t)
        self.framing_view=self.framer.update(scene,self.range_guide,fr.t,automatic=not self.recording and not self.bed.learning)
        if scene['source']=='part':self._bed_framing_ready=True
        self.framing_view['finding_bed']=not getattr(self,'_bed_framing_ready',False)
        self.framing_view['rim']=(getattr(self,'_bed_live',None) or {}).get('rim')
        composer=getattr(self,'composer',None)
        observation=getattr(self,'_bed_live',None) or {}
        if composer is not None and getattr(composer,'color_angle_enabled',False):
            from app2.scan.preview import RGB_CAL,Composer
            age=fr.t-observation.get('t',float('-inf'))
            geo=observation.get('geo')
            if geo is not None and observation.get('source')=='top dots' and 0<=age<Composer.COVER_S:
                composer.color_observer.anchor(getattr(self.bed,'bed',None),RGB_CAL,geo[:3],Composer.COVER_S-age)
            elif geo is None or getattr(self.bed,'bed',None) is None:composer.color_observer.clear()

        self.framing_view['range_test']=getattr(self,'_range_sweep',None) is not None
        if self.framing_view['finding_bed']:self.framing_view['hint']='Finding the bed';self.framing_view['rect']=None
        rows=getattr(self,'mode_evidence',{}).get('frames') or []
        matched=rows[-1].get(self.scan_mode,{}) if rows else {}
        dots=matched.get('inliers') if matched.get('bed_fit') else None
        self.scene_setup=setup_check(scene,guide,dots,steady=self.framer.steady)
        return dict(framing=copy.deepcopy(self.framing_view),guide_depth=np.where(scene['mask'],fr.depth,0) if scene['source']=='part' else None)
    def reframe(self):self._post(self._reframe)
    def _reframe(self):
        if self.recording or self.busy or self.building:return
        self._reset_framing(preserve_undo=True);self._emit_state(force=True)
    def use_frame_range(self):self._post(self._use_frame_range)
    def _use_frame_range(self):
        self._ensure_framing()
        if self.framer.union is None or self.framer.union['source']!='part' or self.recording or self.busy or self.building:return
        from app2.scan.assist import RANGE_STEP_MM
        a,b=self.framer.union['range_mm'];previous=(self.near,self.far,self.auto_range)
        self._set(near=math.floor(a/RANGE_STEP_MM)*RANGE_STEP_MM,far=math.ceil(b/RANGE_STEP_MM)*RANGE_STEP_MM,auto_range=False)
        self.frame_range_undo=previous;self._emit('range',(self.near,self.far));self._emit_state(force=True)
    def undo_frame_range(self):self._post(self._undo_frame_range)
    def _undo_frame_range(self):
        self._ensure_framing()
        if self.frame_range_undo is None or self.recording or self.busy or self.building:return
        near,far,automatic=self.frame_range_undo;self.frame_range_undo=None
        self._set(near=near,far=far,auto_range=automatic);self._emit('range',(near,far));self._emit_state(force=True)
    def switch_camera_match(self):self._post(self._switch_camera_match)
    def _switch_camera_match(self):
        mode=getattr(self,'mode_evidence',{}).get('mode')
        if mode not in ('near','far') or mode==self.scan_mode or self.recording or self.busy or self.building or self.bed.learning:return
        side=self.picked
        if side is not None and side.frames:
            self._save(side);self.host.release();self.picked=None
            self._set_scan_mode(mode);self._new_side()
        else:self._set_scan_mode(mode)
        self._emit_state(force=True)
