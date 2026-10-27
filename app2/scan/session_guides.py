# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session commands for display alignment and an explicit empty-mat sweep."""
import numpy as np


class ScanGuides:
    def _setup_guide_calibration(self,selected):
        from app2.scan.settings import load
        from app2.scan.range_guide import measured_profile
        saved=load();self._guide_calibration_key=selected['source']['calibration_sha256']
        self._overlay_manual=dict(saved.get('overlay_nudges',{}).get(self._guide_calibration_key,{}));self._overlay_auto=None;self._overlay_fit=None;self._overlay_points=None
        key=self.scan_mode+':'+self._guide_calibration_key
        profile=saved.get('range_tests',{}).get(key)
        from app2.scan.range_test import covers_span,supported_bins
        usable=bool(profile and profile.get('complete') and supported_bins(profile) and covers_span(self.scan_mode,profile.get('observed_distance_mm',[0,0])))
        self.range_profile=profile if usable else measured_profile(self.scan_mode)   # an old short test stays a partial result
        self.range_test_result=profile;self._apply_overlay()
    def _apply_overlay(self):
        from app2.scan import preview
        from app2.scan.overlay_alignment import manual_matrix
        preview.OVERLAY_MATRIX=manual_matrix(getattr(self,'_overlay_manual',{})) @ np.asarray(getattr(self,'_overlay_auto',None) or np.eye(3))
        composer=getattr(self,'composer',None)
        if composer is not None:composer.area=None;composer.scan_geometry=None;composer.live=None
    def set_overlay(self,values):self._post(self._set_overlay,dict(values))
    def _set_overlay(self,values):
        from app2.scan.overlay_alignment import manual_matrix
        manual_matrix(values)
        if not getattr(self,'_guide_calibration_key',None):return
        from app2.scan.settings import load,save
        self._overlay_manual=values;all_values=load().get('overlay_nudges',{});all_values[self._guide_calibration_key]=values;save(overlay_nudges=all_values)
        self._apply_overlay();self._update_overlay_error();self._log('Color overlay nudge: '+__import__('json').dumps(values,sort_keys=True));self._emit_state(force=True)
    def reset_overlay(self):self._post(self._reset_overlay)
    def _reset_overlay(self):
        self._overlay_auto=None;self._overlay_fit=None;self._overlay_points=None;self._set_overlay({})
    def auto_overlay(self):self._post(self._auto_overlay_command)
    def _auto_overlay_command(self):
        evidence=getattr(self,'mode_evidence',{});mode=evidence.get('mode')
        if mode!=self.scan_mode:self._log('Check the camera mode before aligning the overlay.');return
        rows=evidence.get('frames') or [];self._overlay_auto=None
        self._refine_overlay(rows[-1].get(mode,{}) if rows else {},evidence);self._emit_state(force=True)
    def _refine_overlay(self,row,evidence):
        if evidence.get('mode')!=self.scan_mode or any(n for mode,n in evidence.get('votes',{}).items() if mode!=self.scan_mode):return
        if getattr(self,'_overlay_auto',None) is not None:return
        from app2.scan.overlay_alignment import refine
        fit=refine(row.get('color'))
        if fit is None:return
        self._overlay_auto=fit['matrix'];self._overlay_fit={k:v for k,v in fit.items() if k!='matrix'}
        color=row['color'];self._overlay_points=(np.asarray(color['projected_points']),np.asarray(color['circle_points']))
        self._apply_overlay();self._update_overlay_error()
        self._log('Color overlay fit: '+__import__('json').dumps(self._overlay_fit,sort_keys=True))
    def _update_overlay_error(self):
        points=getattr(self,'_overlay_points',None)
        if points is None:return
        from app2.scan.overlay_alignment import apply
        from app2.scan.preview import OVERLAY_MATRIX
        p,q=points
        self._overlay_fit.update(before_px=float(np.median(np.linalg.norm(p-q,axis=1))),after_px=float(np.median(np.linalg.norm(apply(p,OVERLAY_MATRIX)-q,axis=1))))
    def range_test(self):self._post(self._range_test)
    def _range_test(self):
        if getattr(self,'_range_sweep',None) is not None:self._finish_range_test();return
        if self.recording or self.busy or self.building or self.bed.learning:return
        detector=getattr(self,'mode_detector',None)
        if detector is None or len(detector.bundles)<2 or detector.model is None:
            self._log('Range test needs a learned empty bed and both cameras set up (Settings, Camera setup).');return
        from app2.scan.range_test import Sweep
        self._range_sweep=Sweep(self.scan_mode,detector.bundles,detector.model);self._emit_state(force=True)
    def _range_test_frame(self,fr):
        sweep=getattr(self,'_range_sweep',None)
        if sweep is not None:sweep.offer(fr.depth,fr.irl,self.source.colour() if getattr(self.source,'colour',None) else None)
    def _finish_range_test(self):
        sweep=getattr(self,'_range_sweep',None)
        if sweep is None:return
        self._range_sweep=None;profile=sweep.close();self.range_test_result=profile;self._range_test_attempted=True
        self._log('Range test done. Press Apply to use it.' if profile and profile['complete'] and profile['camera_pair_confirmed'] else 'Range test not finished. Cover more distances and hold still at each one, or keep the defaults.')
        self._emit_state(force=True)
    def apply_range_test(self):self._post(self._apply_range_test)
    def _apply_range_test(self):
        from app2.scan.range_test import covers_span,supported_bins
        profile=getattr(self,'range_test_result',None)
        if self.recording or getattr(self,'_range_sweep',None) is not None:return
        if not (profile and profile.get('complete') and profile.get('camera_pair_confirmed') and
                profile.get('mode')==self.scan_mode and profile.get('calibration_sha256')==getattr(self,'_guide_calibration_key',None) and
                covers_span(self.scan_mode,profile.get('observed_distance_mm',[0,0])) and supported_bins(profile)):return
        from app2.scan.settings import load,save
        profiles=load().get('range_tests',{});profiles[self.scan_mode+':'+profile['calibration_sha256']]=profile;save(range_tests=profiles)
        self.range_profile=profile;self._sync_scan_geometry();self._log('Range test applied: '+str(len(profile['bins']))+' distance bins.');self._emit_state(force=True)
    def reset_range_test(self):self._post(self._reset_range_test)
    def _reset_range_test(self):
        if getattr(self,'_range_sweep',None) is not None:return
        from app2.scan.settings import load,save
        from app2.scan.range_guide import measured_profile
        profiles=load().get('range_tests',{});profiles.pop(self.scan_mode+':'+getattr(self,'_guide_calibration_key',''),None);save(range_tests=profiles)
        self.range_profile=measured_profile(self.scan_mode);self.range_test_result=None;self._range_test_attempted=False;self._sync_scan_geometry();self._emit_state(force=True)
    def _guide_state(self):
        sweep=getattr(self,'_range_sweep',None)
        from app2.scan.range_test import progress
        return dict(overlay=dict(manual=getattr(self,'_overlay_manual',{}),fit=getattr(self,'_overlay_fit',None)),range_test=dict(active=sweep is not None,attempted=getattr(self,'_range_test_attempted',False),hint=sweep.hint if sweep else '',progress=progress(sweep.samples,self.scan_mode) if sweep else None,result=getattr(self,'range_test_result',None),setup=dict(empty_mat=bool((getattr(self,'_bed_live',None) or {}).get('bed_seen') and not ((getattr(self,'framer',None) and self.framer.last) or {}).get('points',0)),camera_match=bool(getattr(self,'mode_evidence',{}).get('mode')==getattr(self,'scan_mode','near')),light_state=getattr(getattr(self.source,'device',None),'led_state','unknown'))))
