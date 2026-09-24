"""Shared causal RGB state for online navigation and recorded-stream replay.

This module accepts only observations and versioned runtime model results. It
has no simulator dependency or privileged-label interface.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from contracts import Calibration,VisualTaskConfig
from startup import StartupState
from goal_matching import GoalMatcherPipeline,visual_task_tensor
from learning_models import FastVisualOdometry,rotation_increment
from metric_alignment import CausalMetricAlignment
from spatial_memory import SpatialMemory,SpatialRecord,ConservativeGeometry


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class CheckpointSet:
    def __init__(self,manifest,*,integration_only=False):
        self.path=Path(manifest).resolve();self.spec=json.loads(self.path.read_text());self.paths={}
        if self.spec.get('schema')!='visual-navigation-checkpoints/v1':raise ValueError('Versioned checkpoint set required')
        if not integration_only and self.spec.get('accepted') is not True:
            raise ValueError('Unqualified checkpoints are allowed only in explicitly labeled integration work')
        for role,item in self.spec['artifacts'].items():
            path=(self.path.parent/item['path']).resolve()
            if not path.is_relative_to(self.path.parent) or checksum(path)!=item['sha256']:
                raise ValueError('Changed or escaping checkpoint artifact: '+role)
            self.paths[role]=path
        if not {'goal','odometry'}<=self.paths.keys():raise ValueError('Goal and odometry snapshots required')
        self.identity=checksum(self.path)


class FrontierTeacher:
    """Deterministic observed-frontier/keyframe configurator; no hidden goal."""
    def __init__(self,episode_id,match_threshold):
        self.episode_id=episode_id;self.threshold=float(match_threshold);self.visits={}

    def configure(self,memory,position,now_ns,match_probability):
        goals=[(memory.goal_similarity.get(identifier,0.),identifier) for identifier,entry in memory.observed_ids.items()
               if entry['kind']=='goal_match' and entry['observed_ns']<=now_ns]
        candidates=[]
        for identifier,frontier in memory.frontiers.items():
            if frontier['observed_ns']>now_ns:continue
            token=memory.target_context(identifier,now_ns,device='cpu')
            distance=float(torch.linalg.vector_norm(token[256:259]-position.cpu()))
            utility=frontier['information_gain']/(1+distance)/(1+self.visits.get(identifier,0))
            candidates.append((utility,identifier))
        if goals and max(goals)[0]>=self.threshold:
            confidence,identifier=max(goals);kind='goal_match'
            intention='inspect' if match_probability<self.threshold else 'approach'
        elif candidates:
            _,identifier=max(candidates);kind='observed_frontier';intention='search';confidence=.5
        else:
            return None  # No invented frontier/target ID for bootstrap.
        return VisualTaskConfig(self.episode_id,identifier,kind,intention,1.,1.,1.,1.,confidence,
                                now_ns/1e9+2.,True)


class CausalNavigationState:
    def __init__(self,checkpoints,episode_id,goal_rgb,*,device='cuda',backbone='/models/mobilenet-v3-large-imagenet1k-v2.pt'):
        self.checkpoints=checkpoints;self.episode_id=episode_id;self.device=torch.device(device)
        self.goal=GoalMatcherPipeline(backbone).to(device).eval().requires_grad_(False)
        saved=torch.load(checkpoints.paths['goal'],map_location='cpu',weights_only=True)
        if saved['module']!='goal':raise ValueError('Wrong goal checkpoint role')
        self.goal.load_state_dict(saved['model'])
        self.odometry=None;self.required_warmup=1
        if 'vision' not in checkpoints.paths:
            saved=torch.load(checkpoints.paths['odometry'],map_location='cpu',weights_only=True)
            if saved.get('objective_version') not in ('metric-sequence-nll-se3/v1','metric-long-sequence-fixed-bn/v2','metric-long-sequence-motion-rate/v3','metric-motion-rate-balanced-regimes/v4'):
                raise ValueError('Metric sequence odometry required')
            mode='rate' if saved['objective_version'] in ('metric-long-sequence-motion-rate/v3','metric-motion-rate-balanced-regimes/v4') else 'increment'
            if saved.get('motion_parameterization',mode)!=mode:raise ValueError('Inconsistent odometry objective/parameterization')
            self.odometry=FastVisualOdometry(backbone,motion_parameterization=mode).to(device).eval().requires_grad_(False)
            self.odometry.load_state_dict(saved['model'])
            self.required_warmup=saved.get('warmup_steps',4)
        with torch.inference_mode():self.goal_tokens=self.goal.encoder(goal_rgb.to(device))[None]
        self.projection=None
        if 'projection' in checkpoints.paths:
            projection=torch.load(checkpoints.paths['projection'],map_location=device,weights_only=True)
            if projection.get('fit_split')!='train':raise ValueError('Current-video projection must be training-only')
            self.projection=projection
        self.memory=SpatialMemory(episode_id);self.alignment=CausalMetricAlignment(episode_id)
        self.geometry=ConservativeGeometry((-40.,-40.,-20.),.5,shape=(160,160,80))
        self.teacher=FrontierTeacher(episode_id,checkpoints.spec['goal_match_threshold'])
        self.hidden=torch.zeros(1,256,device=device)
        self.position=torch.zeros(3,device=device);self.rotation=torch.eye(3,device=device)
        self.velocity=torch.zeros(3,device=device);self.variance=torch.zeros(6,device=device)
        self.covariance=torch.zeros(6,6,device=device);self.last_body_motion=self.last_motion_std=None
        self.previous_feature=None;self.previous_ns=None;self.latest_frame=-1;self.warmup=0
        self.previous_command=torch.zeros(1,4,device=device);self.configuration=None
        self.history=OrderedDict();self.visual=None;self.tracking=None;self.scale=None
        self.latest_map_version=-1;self.gauge_version=0;self.last_rgb=None;self.pending=[]
        self.last_goal_probability=0.;self.closed=False;self.diagnostics=[]
        self.cached_map=None
        self.supported_maps_received=0;self.optimized_surface_samples_received=0
        self.first_supported_map_available_ns=None
        self.source_features={}
        self.source_rgb=OrderedDict();self.keyframe_rgb=OrderedDict()
        self.startup=StartupState()
        self.hazard_version=0
        self.map_correction_version=0;self.geometry_scale=None
        self.metric_vision=None
        self.depth_worker=None
        if 'vision' in checkpoints.paths:
            from metric_navigation import MetricNavigation
            self.metric_vision=MetricNavigation(self.device)
            self.required_warmup=1
            from metric_depth import AsyncLocalDepth
            self.depth_worker=AsyncLocalDepth(checkpoints.paths['vision'])

    def enqueue(self,component,payload):
        if component not in ('video','tracking','map','configuration','depth'):raise ValueError('Unexpected runtime result')
        if payload.get('episode_id')!=self.episode_id:raise ValueError('Cross-episode asynchronous result')
        if not isinstance(payload.get('version'),int):raise ValueError('Runtime result needs a version')
        if not math.isfinite(payload['available_monotonic']):raise ValueError('Invalid availability timestamp')
        self.pending.append((component,payload))

    def _deliver(self,ns,wall):
        ready=[];pending=[]
        for component,row in self.pending:
            if row['observation_ns']<=ns and row['available_monotonic']<=wall:ready.append((component,row))
            else:pending.append((component,row))
        self.pending=pending
        for component,row in sorted(ready,key=lambda pair:pair[1]['available_monotonic']):
            if component=='depth':
                if self.metric_vision:self.metric_vision.ingest_depth(self,row)
            elif component=='video':
                if self.projection is None:continue
                if row.get('encoder_checkpoint_sha256')!=self.projection.get('encoder_checkpoint_sha256'):
                    raise ValueError('Video encoder differs from training-fitted projection')
                if self.visual and row['observation_ns']<=self.visual['observation_ns']:continue
                raw=row['tokens'].to(self.device).float()
                if raw.shape!=(64,1024):raise ValueError('Released video grid shape changed')
                row=dict(row,tokens=(raw-self.projection['mean'])@self.projection['components'])
                self.visual=row
                historical=self.history.get(row['observation_ns'])
                if historical is None:continue
                identifier='keyframe-'+str(row['source_frame_id'])
                record=SpatialRecord(identifier,self.episode_id,row['observation_ns'],row['source_frame_id'],
                    historical['position'],torch.diag(historical['variance'][:3]),row['tokens'].mean(0),1,historical['confidence'])
                self.memory.append(record,available_ns=ns)
                self.source_features[row['source_frame_id']]=record
                self.memory.observe_keyframe(identifier,[identifier],row['observation_ns'],row['rgb_sha256'],historical['goal_probability'])
                if row['source_frame_id'] in self.source_rgb:
                    self.keyframe_rgb[row['source_frame_id']]=self.source_rgb[row['source_frame_id']]
                    while len(self.keyframe_rgb)>256:self.keyframe_rgb.popitem(last=False)
                # Appearance memory remains useful before metric tracking.
                # Its masked placeholder position cannot become a waypoint.
                if historical['confidence']>0 and historical['goal_probability']>=self.teacher.threshold:
                    self.memory.associate(identifier,'goal_match',[identifier],row['observation_ns'])
            elif component=='tracking':
                if self.tracking and row['observation_ns']<=self.tracking['observation_ns']:continue
                self.tracking=row
                if self.metric_vision is not None:
                    self.metric_vision.ingest(self,row)
                if not row['initialized']:
                    self.scale=None;self.alignment.pairs.clear();self._clear_frontiers()
                    self.diagnostics.append(dict(component='alignment',observation_ns=row['observation_ns'],
                        available_ns=ns,gauge_version=row['gauge_version'],usable=False,reason='tracking_lost'))
                    continue
                if row['gauge_version']!=self.gauge_version:
                    self.gauge_version=row['gauge_version'];self.latest_map_version=-1
                    self.geometry=ConservativeGeometry((-40.,-40.,-20.),.5,shape=(160,160,80))
                    self.cached_map=None;self._clear_frontiers()
                    self.geometry_scale=None
                    if self.metric_vision is None:self.scale=None
                    self.alignment.pairs.clear()
            elif component=='map':
                if row['version']<=self.latest_map_version or row['gauge_version']!=self.gauge_version:continue
                if self.cached_map is None or row['version']>self.cached_map['version']:
                    self.cached_map=row
                    pairs=[]
                    for camera in row['alignment_cameras']:
                        historical=self.history.get(camera['observation_ns'])
                        if historical is None:continue
                        pairs.append((camera['observation_ns'],camera['map_position'],
                            historical['camera_position'].cpu().numpy(),
                            float(historical['camera_covariance'].trace().clamp_min(0).sqrt())))
                    if self.metric_vision is None:
                        self.scale=self.alignment.update_prefix(self.episode_id,row['observation_ns'],ns,
                            self.gauge_version,pairs,ns)
                    self.diagnostics.append(dict(self.scale or {'usable':False,'reason':'metric_depth_pending'},component='alignment',map_version=row['version'],
                        observation_ns=row['observation_ns'],available_ns=ns,
                        published_camera_count=len(row['alignment_cameras'])))
                    samples=sum(len(c['points']) for c in row.get('optimized_surfaces',[]))
                    if samples and row.get('optimizer_updates',0)>0:
                        self.supported_maps_received+=1
                        self.optimized_surface_samples_received+=samples
                        if self.first_supported_map_available_ns is None:self.first_supported_map_available_ns=ns
            elif component=='configuration':
                config=VisualTaskConfig(**row['configuration'])
                config.validate_grounding(self.episode_id,self.memory.observed_ids,ns/1e9)
                self.configuration=config
        # A map can arrive before scale has sufficient support. Retain it and
        # integrate only after a later causal fit becomes usable.
        row=self.cached_map
        if (row is not None and row['gauge_version']==self.gauge_version and
                row['version']>self.latest_map_version and self.scale and self.scale.get('usable')
                and row.get('optimizer_updates',0)>0 and any(len(c['points']) for c in row.get('optimized_surfaces',[]))):
            # Only geometrically supported, observed RGB-estimated points.
            rotation=torch.tensor(self.scale['rotation']);translation=torch.tensor(self.scale['translation'])
            if self.geometry_scale is not None:
                previous=self.geometry_scale
                relative=(self.scale['meters_per_map_unit']/previous['meters_per_map_unit'])*rotation@torch.tensor(previous['rotation']).T
                shift=translation-relative@torch.tensor(previous['translation'])
                # Bound a similarity correction over the whole navigation
                # grid, not just at the current pose. Half a voxel is the
                # material geometry-change threshold for an existing plan.
                corners=torch.cartesian_prod(*[torch.tensor([0.,float(n)]) for n in self.geometry.shape])
                corners=self.geometry.origin+corners*self.geometry.resolution
                correction=float((corners@relative.T+shift-corners).norm(dim=1).max())
                correction+=abs(self.scale['fit_rmse_m']-previous['fit_rmse_m'])
                if correction>self.geometry.resolution/2:self.map_correction_version+=1
            self.geometry_scale=dict(self.scale)
            # Rays belong to their source view; historical surfaces are not
            # assumed visible from the latest camera.
            previous_occupied=self.geometry.occupied.clone()
            self.geometry=ConservativeGeometry((-40.,-40.,-20.),.5,shape=(160,160,80))
            for ray in row['observed_rays']:
                points=ray['points'].float().cpu()
                points=self.scale['meters_per_map_unit']*(points@rotation.T)+translation
                camera=self.scale['meters_per_map_unit']*(torch.as_tensor(ray['camera_position']).float()@rotation.T)+translation
                self.geometry.integrate(camera,points,self.scale['fit_rmse_m'],occupy_endpoints=False)
            for identifier in [key for key in self.memory.records if key.startswith('surface-')]:
                del self.memory.records[identifier]
            for chunk in row['optimized_surfaces']:
                if not len(chunk['points']):continue
                scale=self.scale['meters_per_map_unit']
                centers=scale*(chunk['points'].float()@rotation.T)+translation
                self.geometry.integrate_surfaces(centers,scale*chunk['extent'],self.scale['fit_rmse_m'])
                source=self.source_features.get(chunk['source']['frame_id'])
                if source is None:continue # No appearance invented for unsupported source images.
                for index in range(0,len(centers),max(1,len(centers)//64)):
                    record=SpatialRecord('surface-'+str(chunk['source']['frame_id'])+'-'+str(index),
                        self.episode_id,int(chunk['observed_ns'][index]),source.source_frame,centers[index],
                        torch.eye(3)*self.scale['fit_rmse_m']**2,source.feature,
                        int(chunk['observation_count'][index]),source.confidence)
                    self.memory.append(record,available_ns=ns)
            self.hazard_version+=int(bool(((self.geometry.occupied>0)&(previous_occupied==0)).any()))
            self.latest_map_version=row['version']
            self._clear_frontiers()
            self._frontiers(row,ns)

    def _clear_frontiers(self):
        # Rebuilt/corrected geometry cannot leave stale free-space targets.
        removed=set(self.memory.frontiers)
        for identifier in removed:
            self.memory.records.pop(identifier,None);self.memory.observed_ids.pop(identifier,None)
        self.memory.frontiers.clear()
        if removed:self.memory.version+=1
        if self.configuration and self.configuration.target_id in removed:self.configuration=None

    def grounding_context(self,now_ns,config=None):
        """Bind retrieved target IDs to real episode RGB for online/SFT parity."""
        def reference(record):
            saved=self.keyframe_rgb.get(record.source_frame)
            if saved is None or saved['observed_ns']>now_ns:return None
            return dict(frame_id=record.source_frame,observed_ns=saved['observed_ns'],rgb_sha256=saved['rgb_sha256'])
        target=config.target_id if config else None
        keys=[]
        for identifier,entry in self.memory.keyframes.items():
            record=self.memory.records[entry['record_ids'][0]];ref=reference(record)
            if ref is not None:keys.append((identifier,entry,ref))
        keys.sort(key=lambda item:(item[0]!=target,-self.memory.goal_similarity.get(item[0],0.),-item[1]['observed_ns']))
        keys=keys[:3];keyframes=[row[2] for row in keys];frontiers=[];observed=[]
        for index,(identifier,entry,ref) in enumerate(keys):
            association=self.memory.observed_ids.get(identifier)
            if association and association['kind']=='goal_match':
                observed.append(dict(id=identifier,kind='goal_match',observed_ns=association['observed_ns'],image_position=6+index))
        ordered=sorted(self.memory.frontiers.items(),key=lambda item:(item[0]!=target,-item[1]['information_gain'],item[0]))
        for identifier,entry in ordered:
            record=self.memory.records[entry['evidence'][0]];ref=reference(record)
            if ref is None or entry['observed_ns']>now_ns:continue
            observed.append(dict(id=identifier,kind='observed_frontier',observed_ns=entry['observed_ns'],
                image_position=6+len(keyframes)+len(frontiers)))
            frontiers.append(ref)
            if len(frontiers)==8:break
        visible={row['id'] for row in observed}
        return dict(schema='observed-rgb-grounding/v1',keyframes=keyframes,frontiers=frontiers,observed=observed,
            teacher_configuration=asdict(config) if config and config.target_id in visible else None)

    def _frontiers(self,row,now_ns):
        # Frontier cells are observed free cells adjacent to unknown cells.
        free=self.geometry.free>=2;unknown=(self.geometry.free==0)&(self.geometry.occupied==0)
        adjacent=torch.zeros_like(free)
        for axis in range(3):
            lower=[slice(None)]*3;upper=lower.copy();lower[axis]=slice(None,-1);upper[axis]=slice(1,None)
            adjacent[tuple(lower)]|=unknown[tuple(upper)];adjacent[tuple(upper)]|=unknown[tuple(lower)]
        candidates=(free&adjacent&(self.geometry.occupied==0)).nonzero()
        if not len(candidates) or not self.memory.records:return
        positions=self.geometry.origin+(candidates.float()+.5)*self.geometry.resolution
        distances=(positions-self.position.cpu()).norm(dim=1)
        candidates=candidates[(distances>=2)&(distances<=15)]
        # Deterministic spatial thinning gives stable observed target IDs.
        cells={}
        for cell in candidates:
            key=tuple((cell//4).tolist())
            if key not in cells:cells[key]=cell
        for key in sorted(cells)[:32]:
            identifier='frontier-'+'-'.join(map(str,key))
            position=self.geometry.origin+(cells[key].float()+.5)*self.geometry.resolution
            # A free-space target gets observed keyframe context, not invented
            # surface appearance at an empty voxel. Its timestamp includes
            # both the geometric evidence and that feature's source image.
            keyframe_records=[self.memory.records[k] for item in self.memory.keyframes.values() for k in item['record_ids']
                              if self.memory.records[k].confidence>0]
            if not keyframe_records:continue
            record=min(keyframe_records,key=lambda r:float((r.position-position).norm()))
            r=SpatialRecord(identifier,self.episode_id,max(row['observation_ns'],record.observed_ns),
                record.source_frame,position,record.covariance,record.feature,1,record.confidence)
            self.memory.append(r,available_ns=now_ns)
            self.memory.set_frontier(identifier,[identifier],1.,row['rgb_sha256'],r.observed_ns)

    @torch.inference_mode()
    def observe(self,metadata,rgb):
        if self.closed or metadata['episode_id']!=self.episode_id or metadata['frame_id']<=self.latest_frame:
            raise ValueError('Closed/cross-episode/noncausal runtime observation')
        ns=metadata['sim_ns'];wall=metadata['received_monotonic']
        if self.depth_worker:
            depth=self.depth_worker.observe(metadata,rgb)
            if depth is not None:self.enqueue('depth',depth)
        calibration=Calibration(**metadata['calibration'])
        if calibration.camera_origin_body_m is None:raise ValueError('Metric camera alignment requires calibrated extrinsics')
        if self.previous_ns is not None and ns<=self.previous_ns:raise ValueError('Nonmonotonic exposure time')
        image=torch.from_numpy(np.frombuffer(rgb,np.uint8).copy().reshape(480,640,3)).permute(2,0,1)[None].to(self.device)
        self.source_rgb[metadata['frame_id']]=dict(observed_ns=ns,rgb_sha256=hashlib.sha256(rgb).hexdigest(),rgb=bytes(rgb))
        while len(self.source_rgb)>96:self.source_rgb.popitem(last=False)
        feature=self.odometry.encode(image) if self.metric_vision is None else image.new_zeros((1,1),dtype=torch.float32)
        history=metadata.get('command_history',[])
        previous_command=torch.tensor(history[-1]['values'] if history else [0.,0.,0.,0.],
            device=self.device,dtype=feature.dtype)[None]
        interval=0. if self.previous_ns is None else (ns-self.previous_ns)/1e9
        if self.metric_vision is not None:
            self._deliver(ns,wall)
        elif self.previous_feature is not None and interval<=.25:
            prediction=self.odometry.forward_features(self.previous_feature,feature,self.previous_command,
                feature.new_tensor([interval]),self.hidden)
            motion=prediction['body_motion'][0].float();self.hidden=prediction['hidden'];self.warmup+=1
            self.last_body_motion=motion;self.last_motion_std=prediction['motion_std'][0].float()
            delta=self.rotation@motion[:3];self.position+=delta;self.velocity=delta/interval
            # First-order SE(3) covariance in the episode frame. Orientation
            # uncertainty contributes to future translation uncertainty.
            transition=torch.eye(6,device=self.device);x,y,z=delta;zero=x.new_zeros(())
            skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
            transition[:3,3:]=-skew
            noise=torch.block_diag(self.rotation,self.rotation)
            self.covariance=transition@self.covariance@transition.T+noise@torch.diag(self.last_motion_std.square())@noise.T
            self.variance=self.covariance.diag().clamp_min(0)
            self.rotation=self.rotation@rotation_increment(motion[3:])
        else:
            self.hidden.zero_();self.warmup=0
            self.last_body_motion=self.last_motion_std=None
            if self.previous_feature is not None:
                self.covariance+=100*torch.eye(6,device=self.device);self.variance=self.covariance.diag()
        self.previous_feature=feature;self.previous_ns=ns;self.latest_frame=metadata['frame_id'];self.last_rgb=image
        self.previous_command=previous_command
        current_tokens=self.goal.encoder(image);match=self.goal.matcher(current_tokens,self.goal_tokens)
        self.source_features[metadata['frame_id']]=SpatialRecord('source-'+str(metadata['frame_id']),
            self.episode_id,ns,metadata['frame_id'],self.position.clone(), self.covariance[:3,:3].clone(),
            current_tokens[0].mean(0),1,1.)
        while len(self.source_features)>2048:del self.source_features[next(iter(self.source_features))]
        probability=float(match['match_logit'].sigmoid());self.last_goal_probability=probability
        confidence=float(torch.exp(-self.variance[:3].sum().sqrt())) if self.warmup>=self.required_warmup else 0.
        if self.metric_vision is not None and not (self.scale and self.scale.get('usable') and
                self.tracking and self.tracking['initialized'] and 0<=ns-self.tracking['observation_ns']<=1_000_000_000):
            confidence=0.
        lever=self.rotation@feature.new_tensor(calibration.camera_origin_body_m)
        camera_position=self.position+lever
        x,y,z=lever;zero=x.new_zeros(())
        lever_skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
        camera_jacobian=torch.cat((torch.eye(3,device=self.device),-lever_skew),1)
        camera_covariance=camera_jacobian@self.covariance@camera_jacobian.T
        self.history[ns]=dict(position=self.position.clone(),camera_position=camera_position,
                              camera_covariance=camera_covariance,
                              variance=self.variance.clone(),confidence=confidence,
                              goal_probability=probability)
        while len(self.history)>1024:self.history.popitem(last=False)
        self._deliver(ns,wall)
        usable=bool(self.scale and self.scale.get('usable'))
        scale_log=math.log(self.scale['meters_per_map_unit']) if usable else 0.
        scale_sigma=self.scale['log_scale_sigma'] if usable else 0.
        feature_age=(ns-self.visual['observation_ns'])/1e9 if self.visual else 0.
        tracking_age=(ns-self.tracking['observation_ns'])/1e9 if self.tracking else 0.
        tracking_valid=bool(self.tracking and self.tracking['initialized'] and tracking_age<=1.)
        # Camera-relative metric depth supports local flight even while the
        # global translation/scale remains explicitly invalid in state masks.
        local_ready=bool(self.metric_vision and self.metric_vision.usable(ns) and
                         float(self.metric_vision.depth_tokens[:,1].mean())>=.5)
        map_status=self.startup.update(ns,usable and self.latest_map_version>=0,tracking_valid,local_ready)
        state=torch.cat((self.position,self.velocity,self.rotation[:,0],self.rotation[:,1],
            feature.new_tensor([scale_log,scale_sigma,confidence,feature_age,float(self.variance[:3].sum().sqrt()),tracking_age])))
        # Explicit masks and current timing context are model inputs, never
        # large numeric sentinels. Dispatch clock is not an application clock.
        last=history[-1] if history else {}
        timing_valid=last.get('submitted_monotonic') is not None
        timing=[last.get('observation_age_seconds',last.get('source_received_to_command_wall_seconds',0.)) or 0.,
            last.get('dispatch_interval_seconds',0.) or 0.,last.get('timing_uncertainty_seconds',0.) or 0.,
            float(last.get('watchdog_override',False)),float(timing_valid)]
        # Local navigation shares the moving-state bit; map validity is an
        # independent mask. Keep the 32-D world interface; version semantics.
        flags=[float(map_status==name or name=='mapped' and map_status=='local_navigation') for name in ('initializing','mapped','recovering','terminated')]
        state=torch.cat((state,feature.new_tensor(flags+[float(usable),float(self.visual is not None),
            float(tracking_valid),min(30.,self.startup.elapsed(ns)),float(self.warmup>=self.required_warmup)]+timing)))
        memory,valid=self.memory.retrieve(self.position,ns,device=self.device)
        config=self.teacher.configure(self.memory,self.position,ns,probability)
        if self.configuration and self.configuration.valid_until_sim_seconds>=ns/1e9:
            try:
                self.configuration.validate_grounding(self.episode_id,self.memory.observed_ids,ns/1e9)
                config=self.configuration
            except ValueError:
                # A new target or map correction can invalidate an earlier
                # abstention/target. Fall back to current observed evidence.
                self.configuration=None
                self.diagnostics.append(dict(component='configuration',observation_ns=ns,
                    reason='configuration_invalidated_by_current_observed_memory'))
        target=self.memory.target_context(config.target_id,ns,device=self.device) if config and config.target_id is not None else state.new_zeros(264)
        task=visual_task_tensor(dict(match_probability=probability,time_to_goal_seconds=float(match['time_to_goal_seconds']),
            terminal_value=float(match['terminal_value'])),config,0.,scale_sigma,1. if config else 0.).to(self.device)
        return dict(depth_tokens=self.metric_vision.depth_tokens if self.metric_vision and self.metric_vision.usable(ns) else state.new_zeros(300,2),
            local_geometry_available=local_ready,
            geometry_age_seconds=(ns-self.metric_vision.local_ns)/1e9 if self.metric_vision and self.metric_vision.local_ns is not None else None,
            episode_id=self.episode_id,frame_id=metadata['frame_id'],sim_ns=ns,
            latest_observation_ns=self.visual['observation_ns'] if self.visual else ns,
            state=state,memory=memory,memory_valid=valid,goal_tokens=self.goal_tokens[0],target_context=target,
            task=task,previous_command=previous_command[0],image=image[0],
            z=self.visual['tokens'] if self.visual else state.new_zeros(64,256),visual_available=self.visual is not None,
            target_available=config is not None and config.target_id is not None,config=config,goal_probability=probability,current_tokens=current_tokens,
            goal_context=match['goal_context'],metric_geometry_available=local_ready or usable and self.latest_map_version>=0,
            memory_version=self.memory.version,gauge_version=self.gauge_version,
            tracking_confidence=confidence,feature_age_seconds=feature_age,tracking_age_seconds=tracking_age,
            map_status=map_status,initialization_elapsed_seconds=self.startup.elapsed(ns),
            termination_reason=self.startup.reason,hazard_version=self.hazard_version,
            map_correction_version=self.map_correction_version,
            goal_match_threshold=self.teacher.threshold,
            input_validity=dict(scale=usable,visual=self.visual is not None,tracking=tracking_valid))

    def close(self):
        if self.depth_worker:self.depth_worker.close()
        self.memory.close();self.history.clear();self.pending.clear();self.source_rgb.clear();self.keyframe_rgb.clear();self.closed=True
