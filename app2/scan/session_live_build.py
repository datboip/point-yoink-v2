# SPDX-License-Identifier: AGPL-3.0-or-later
"""Worker-owned Live build settings, recent Undo, surface starts, and light ownership."""
from pathlib import Path
import json,time,copy
import numpy as np
from app2.scan import settings


class LiveBuildSession:
    def _init_live_build(self):
        saved=settings.load();self.live_detail=saved.get('live_detail','Balanced');self.live_support=saved.get('live_support');self.live_share=saved.get('live_share',.03)
        self.live_pass=saved.get('live_pass','Free');self.light_during_scans=bool(saved.get('light_during_scans',False));self._scan_led_owned=False;self._manual_led=False
        self.surface_name=saved.get('surface_name','Normal');self.surface_hint=None;self.surface_start=None
        self._live_fields={};self._live_recent=None
        try:                                                  # the saved Surface's Auto gain bounds and rule apply from the start, not only after a click
            from app2.scan.surface import preset
            self._apply_auto_gain_rule(preset(getattr(self,'scan_mode','near'),self.surface_name))
        except (KeyError,OSError,ValueError,AttributeError):pass
    def _apply_auto_gain_rule(self,value):
        ag=getattr(self,'autogain',None)
        if ag is None:return
        ag.bounds=tuple(value['auto_gain_bounds']);rule=value.get('auto_gain_rule') or {}
        ag.clip_max=rule.get('clip_max',ag.CLIP_MAX);ag.dark_max=rule.get('dark_max',ag.DARK_MAX)
    def set_color_angle(self,on):self._post(self._set_color_angle,bool(on))
    def _set_color_angle(self,on):
        composer=getattr(self,'composer',None)
        if composer is not None:
            composer.color_angle_enabled=on
            composer.color_angle=None
            if not on:composer.color_observer.clear()
        settings.save(color_angle=on)
    def set_stereo_refine(self,on):self._post(self._set_stereo_refine,bool(on))
    def _set_stereo_refine(self,on):
        composer=getattr(self,'composer',None)
        stereo=getattr(composer,'stereo',None)
        if stereo is not None:
            stereo.refine_enabled=on
            stereo.shifts.clear()
        settings.save(ir_refine=on)
    def set_light_during_scans(self,on):self._post(self._set_light_during_scans,bool(on))
    def _set_light_during_scans(self,on):
        self.light_during_scans=on;settings.save(light_during_scans=on);self._emit_state(force=True)
    def _sync_scan_light(self):
        if not self.recording and getattr(self,'_scan_led_owned',False):
            self._scan_led_owned=False
            if not getattr(self,'_manual_led',False):self._led_auto=False;self._set_led(False)
    def _start_scan_light(self):
        if not getattr(self,'light_during_scans',False) or not self.source.has_led or getattr(self,'_manual_led',False):return
        self._scan_led_owned=True;self._led_auto=True;self._set_led(True)
    def set_surface(self,name):self._post(self._set_surface,name)
    def _set_surface(self,name):
        if name not in ('Normal','Dark','Shiny','Sprayed') or self.recording or self.busy or self.building:return
        from app2.scan.surface import preset
        value=preset(self.scan_mode,name);self.surface_name=name;self.surface_start=value
        self._set(who='the Surface preset %s'%name,gain=value['gain']);self.autogain.reset();
        if self.picked is not None:
            self.picked.scan_profile['surface_start']=value
            if self.host.has(self.picked):self.host.raw().depth_jump_mm=value['depth_jump_mm']
        self._apply_auto_gain_rule(value);self._initial_auto_gain=False
        # Presets are explicit starting settings. Later gain/light/Range changes
        # remain the user's settings, with no frame-driven preset changes.
        if not getattr(self,'_manual_led',False) and self.source.has_led:
            self._led_auto=bool(value['fill_light']);self._set_led(value['fill_light'])
        settings.save(surface_name=name);self._log('Surface starting settings: '+json.dumps(dict(name=name,**value),sort_keys=True));self._emit_state(force=True)
    def set_live_build(self,detail=None,support=None,share=None,angle_pass=None):self._post(self._set_live_build,detail,support,share,angle_pass)
    def _set_live_build(self,detail,support,share,angle_pass):
        if self.recording or self.busy or self.building:return
        from app2.scan.live_build import detail_levels
        if detail in detail_levels(self.scan_mode):self.live_detail=detail
        if support is not None:
            if not 1<=int(support)<=np.iinfo(np.uint16).max:return
            self.live_support=int(support)
        if share is not None:
            if not 0<=float(share)<=1:return
            self.live_share=float(share)
        if angle_pass in ('Free','Low','Level','Flipped'):self.live_pass=angle_pass
        settings.save(live_detail=self.live_detail,live_support=self.live_support,live_share=self.live_share,live_pass=self.live_pass)
        # Detail and saving rules are fixed per side. A populated side keeps its
        # recorded rule; the selected setting starts the next side.
        sd=self.picked
        if sd is not None and not sd.frames and self.capture_mode=='live':
            self._prepare_live_side(sd);self.host.release();self.host.new(sd)
        self._emit_state(force=True)
    def next_live_pass(self):self._post(self._next_live_pass)
    def _next_live_pass(self):
        if self.recording or self.busy or self.building:return
        passes=('Low','Level','Flipped');current=getattr(self,'live_pass','Free');self.live_pass=passes[(passes.index(current)+1)%len(passes)] if current in passes else passes[0]
        settings.save(live_pass=self.live_pass);self._new_side();self._emit_state(force=True)
    def _prepare_live_side(self,sd):
        if getattr(self,'capture_mode',None)!='live':return
        from app2.scan.live_build import detail_levels,QualityField
        old=getattr(sd,'scan_profile',{}).get('live_build');rule=old or dict(detail=self.live_detail,voxel_mm=detail_levels(self.scan_mode).get(self.live_detail,detail_levels(self.scan_mode)['Balanced']),support=self.live_support,share=self.live_share,angle_pass=self.live_pass)
        sd.scan_profile['surface_start']=getattr(self,'surface_start',None);sd.scan_profile['live_build']=rule;sd.scan_profile['voxel_mm']=rule['voxel_mm']
        from app2.scan.scan_mode import write
        write(sd.folder,sd.scan_profile)
        if sd not in self._live_fields:
            from app2.scan.live_cache import read,restore_field
            cached=read(sd)
            if cached is not None:
                self._live_fields[sd]=restore_field(cached);return
            from app2.scan.range_guide import measured_profile
            field=QualityField(rule['voxel_mm'],getattr(self,'range_profile',None) or measured_profile(self.scan_mode),rule.get('support'),rule['share'])
            # Continue rebuilds saved evidence from kept frames, never inventing
            # support from the frames intentionally not stored.
            for ps,idx,M in sd.entries:
                from track import F
                from app2.scan.live_build import frame_points
                d=self._read_depth(sd,ps,idx)
                if d is None:continue
                T=M@F;T=T.copy();T[:3,3]/=1000;p,z=frame_points(d,self.cal,T);field.observe(p,z,force=True)
            saved=Path(sd.folder)/'live-build.json'
            try:
                previous=json.loads(saved.read_text());field.seen=max(field.seen,int(previous.get('seen',0)));field.refused=int(previous.get('refused',0))
            except (OSError,ValueError,TypeError):pass
            self._live_fields[sd]=field
    def _save_live_surface(self,sd,later=None):
        """The voxel cache and the live model for this side. The GPU reads happen here; the disk writes (compressing the
        grid, the PLY) run through later(fn) when given, so Pause comes back at once."""
        field=getattr(self,'_live_fields',{}).get(sd)
        if field is None or not self.host.has(sd) or not sd.entries or not sd.dirty:return
        trk=self.host.raw()
        if not hasattr(trk,'vbg'):return
        from app2.scan.live_cache import snapshot,write_snapshot
        snap=snapshot(sd,trk,field)
        mesh=None
        try:
            import open3d as o3d
            from app2.scan.bedcut import LOW_MM
            mesh=trk.vbg.cpu().extract_triangle_mesh(weight_threshold=min(field.target,trk.COVER_THIN)).to_legacy()
            mesh.vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)*1000)
            plane=self._base_plane(sd)
            if self.remove_base and plane is not None:mesh.remove_vertices_by_mask(np.asarray(mesh.vertices)@np.asarray(plane[0])+plane[1]<=LOW_MM)
            mesh.remove_unreferenced_vertices();points=np.asarray(mesh.vertices);bands,quality=field.levels(points)
            from app2.scan.capture_distance import rgb
            mesh.vertex_colors=o3d.utility.Vector3dVector(rgb(bands).astype(float)/255);mesh.compute_vertex_normals()
        except (RuntimeError,ValueError) as error:mesh=None;self._log('Live model needs more supported views: '+str(error))
        folder=Path(sd.folder);log=self._log
        def to_disk():
            if snap is not None:write_snapshot(snap)
            if mesh is None:return
            import open3d as o3d
            path=folder/'live-model.ply';tmp=path.with_suffix('.tmp.ply')
            if not o3d.io.write_triangle_mesh(str(tmp),mesh,write_ascii=False):raise OSError('Could not write live mesh')
            tmp.replace(path);log('Live model saved: %d vertices, %d triangles. Full Build uses kept frames.' % (len(mesh.vertices),len(mesh.triangles)))
        if later is not None:later(to_disk)
        else:to_disk()
    def _restore_live_surface(self,sd):
        if not getattr(sd,'scan_profile',{}).get('live_build'):return False
        from app2.scan.live_cache import read,restore_tracker,restore_field
        self._flush_disk()                                        # a save from the last Pause may still be writing
        data=read(sd)
        if data is None:return False
        self.host.new(sd)
        if not hasattr(self.host.raw(),'vbg'):return False
        try:restore_tracker(self.host.raw(),data)
        except (RuntimeError,ValueError):self.host.release();return False
        self._live_fields[sd]=restore_field(data);self._log('Live model restored from its exact local voxel cache.');return True
    def export_live_model(self,path):self._post(self._export_live_model,path)
    def _export_live_model(self,path):
        if self.recording or self.busy or self.building or self.picked is None:return
        self._save_live_surface(self.picked);self._flush_disk()
        source=Path(self.picked.folder)/'live-model.ply'
        if source.exists():
            import shutil
            shutil.copy2(source,path);self._log('Live model exported.')
    def _live_before(self,sd,t):
        if getattr(self,'capture_mode',None)!='live':return
        from app2.scan.live_build import RecentFusion
        from app2.scan.preview import Composer
        from app2.scan.tracking import TrackerHost
        if self._live_recent is None or self._live_recent[0] is not sd:
            self._live_recent=(sd,RecentFusion(Composer.COVER_S,TrackerHost.CKPT_KEEP),[])
        recent=self._live_recent;recent[1].before(self.host.raw(),t,sd.entries,sd.stamps)
        # The small evidence checkpoint matches the oldest exact TSDF snapshot.
        recent[2].append((t,copy.deepcopy(self._live_fields[sd])))
        first=recent[1].frames[0][0] if recent[1].frames else t
        recent[2][:]=[row for row in recent[2] if row[0]>=first]
    def _live_keep(self,sd,d,r,t):
        from app2.scan.live_build import frame_points
        field=self._live_fields[sd]
        if not r['ok']:return field.observe(np.empty((0,3)),np.empty(0),ok=False)['keep']
        raw=getattr(self.host.raw(),'last_integration_depth',None)
        if raw is None:raw=d
        mask=getattr(self,'_scene_part_mask',None);p,z=frame_points(raw,self.cal,r['T'],mask)
        decision=field.observe(p,z,ok=bool(r['ok'] and not r.get('unsure') and not r.get('refused')))
        sd.scan_profile['live_build']['latest']=field.state();return decision['keep']
    def undo_live_build(self):self._post(self._undo_live_build)
    def _undo_live_build(self):
        sd=self.picked;recent=getattr(self,'_live_recent',None)
        if sd is None or recent is None or recent[0] is not sd or self.busy or self.building:return
        self.recording=False;self._sync_scan_light()
        if getattr(self,'_color_writer',None) is not None:self._color_writer.flush()
        row=recent[1].restore(self.host.raw())
        if row is None:return
        t,_,_,en,st=row
        # Kept files leave the active side just like Back. Unsaved contributions
        # are removed by the exact voxel snapshot above.
        from app2.scan import fmt
        discard=Path(sd.folder)/'discarded';discard.mkdir(exist_ok=True)
        removed=sd.stamps[st:]
        for stamp in removed:
            for ext in fmt.FRAME_EXTS:
                p=Path(sd.cache)/(stamp['frame']+ext)
                if p.exists():p.replace(discard/p.name)
        del sd.entries[en:];del sd.stamps[st:]
        self._live_fields[sd]=recent[2][0][1];self._live_recent=None;self.host.ckpts=[];self.last=None;sd.dirty=True
        # Rebuild platter samples after Undo; later fits must not retain the
        # removed camera poses. Continue still uses the existing gap guards.
        from table_plane import SidePlane
        sd.table=SidePlane();self._cut_poses.clear();self.table_pose=None;sd.pass_ok=0
        for ps,idx,M in sd.entries:
            d=self._read_depth(sd,ps,idx)
            if d is not None:
                from track import F
                sd.table.add(d,self.cal,M@F)
        self._save(sd);self._show_home(sd);self._emit_sides();self._emit_state(force=True);self._log('Live build Undo removed recent fusion and %d kept frames.' % len(removed))
    def _update_bed_observation(self,fr):
        model=getattr(getattr(self,'bed',None),'bed',None)
        if model is None:return
        observer=getattr(self,'_bed_observer',None)
        if observer is None or observer.model is not model or tuple(observer.cal)!=tuple(self.cal):
            if observer is not None:observer.close()
            from app2.scan.bed_observer import BedObserver
            from app2.scan.preview import Composer
            observer=self._bed_observer=BedObserver(self.cal,model,Composer.COVER_S)
        value=observer.offer(fr.depth,fr.irl,fr.t);self._bed_live=value
        if value and value.get('rim'):
            old=getattr(model,'rim',None);model.rim=value['rim']
            if old is None:self._log('Measured bed rim: %.2f mm radius, %.2f mm spread.' % (model.rim['radius_mm'],model.rim['spread_mm']))
            model.save(str(Path(self.project.path)/'bed.json'))
        if value and value.get('angle_deg') is not None and self.picked is not None:
            profile=getattr(self.picked,'scan_profile',{})
            if self.recording and 'start_angle_deg' not in profile:profile['start_angle_deg']=value['angle_deg'];profile['start_angle_source']=value['source']
        if self.picked is not None:self._emit_turn(self.picked)
        else:
            self._emit('turn',dict(deg=None,coverage=dict(levels=[],degrees=None,live_angle_deg=None if value is None else value.get('angle_deg'),bed_seen=bool(value and value.get('bed_seen')),start_angle_deg=0 if getattr(model,'rim_markers',None) else None)))
    def _live_state(self):
        field=getattr(self,'_live_fields',{}).get(self.picked);state=field.state() if field else None
        if state is not None:
            state=dict(state,angle_pass=getattr(self.picked,'scan_profile',{}).get('live_build',{}).get('angle_pass','Free'))
            from app2.scan.turn_coverage import snapshot
            ring=snapshot(self.picked);full=bool(ring.get('levels') and all(v>=0 for v in ring['levels']))
            state['pass_done']=state['pass_done'] and full
            state['message']='This pass has enough support. Pause, then choose Next pass.' if state['pass_done'] else 'Keep turning to fill the thin regions'
        recent=getattr(self,'_live_recent',None)
        return dict(live_build=state,live_detail=getattr(self,'live_detail','Balanced'),live_pass=getattr(self,'live_pass','Free'),live_undo=bool(recent and recent[0] is self.picked and recent[1].frames),light_during_scans=getattr(self,'light_during_scans',False),surface_name=getattr(self,'surface_name','Normal'),surface_hint=getattr(self,'surface_hint',None))
