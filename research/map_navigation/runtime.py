"""Mode 1 owns vehicle commands; Mode 2 only publishes expiring subgoals."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace,asdict
import json
from pathlib import Path
import threading
import time
import uuid
import hashlib
import numpy as np
import torch
from PIL import Image
from wire import BrokerClient
from .common import config,read,write,digest,contained
from .models import load_model
from .maps import MapPrior
from .contracts import assemble_context,relative_times,preceding_command,context_record
from .temporal import pool,embed_subgoal
from .compute import ComputeLane
from .perception import BackgroundPerception
from .subgoals import Mode2
from .safety import Safety


def image_tensor(rgb,device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).to(device)


class Navigator:
    def __init__(self,package,map_folder,goal_rgb,variant,sample_policy=False,initial_subgoal=None,perception_only=False,teacher_spec=None):
        if len(goal_rgb)!=1:raise ValueError('Exactly one goal photograph required')
        if variant not in config()['variants']:raise ValueError('Unknown temporal variant')
        self.device='cuda';self.variant=variant;self.package=Path(package);self.sample_policy=sample_policy
        self.behavior_identity=digest(self.package/'model.pt')
        self.initial_subgoal=initial_subgoal
        self.perception_only=perception_only;self.session_id=uuid.uuid4().hex
        self.teacher=None
        if teacher_spec:
            if not perception_only:raise ValueError('Collection teacher requires a perception-only package')
            from .observation_teacher import ObservationTeacher
            specification=read(teacher_spec)
            if specification.get('schema')!='photo-map-teacher/v1':raise ValueError('Invalid teacher descriptor')
            if specification.get('implementation_sha256')!=digest(Path(__file__).with_name('observation_teacher.py')):
                raise ValueError('Teacher implementation changed')
            if specification.get('perception_sha256')!=digest(self.package/'package.json'):raise ValueError('Teacher perception changed')
            self.teacher=ObservationTeacher(goal_rgb[0],specification)
        self.branch=initial_subgoal is not None;self.started_s=None;self.branch_context_sha256=None
        self.rgb_hashes=deque(maxlen=4)
        spec=read(self.package/'package.json')
        if spec['schema']!='photo-map-package/v6':raise ValueError('Temporal-window package required')
        if not perception_only and spec.get('capability')!='navigation':raise ValueError('Perception package cannot control a vehicle')
        required={'model.pt','vision.json','photo-slam.json'}
        if not required<=set(spec['files']) or set(spec['files'])-required-{'qwen.pt'}:raise ValueError('Unexpected assets')
        for name,sha in spec['files'].items():
            if digest(contained(self.package,name))!=sha:raise ValueError('Changed inference package')
        self.model,self.saved=load_model('/models/mobilenet-v3-large-imagenet1k-v2.pt',self.package/'model.pt')
        if digest('/models/mobilenet-v3-large-imagenet1k-v2.pt')!=self.saved['backbone_sha256']:raise ValueError('Backbone mismatch')
        stages={'localization','goal'}|({'policy'} if not perception_only else set())|({'world'} if variant=='mode1_vlm_world' and not perception_only else set())
        if not stages<=set(self.saved['trained_stages']):raise ValueError('Untrained runtime stages')
        self.calibration=self.saved['calibration']
        from .provenance import verify_release,identities
        verify_release(self.saved,world=variant=='mode1_vlm_world' and not perception_only,perception_only=perception_only)
        self.actor_identity=identities(self.saved['model'])['actor']
        self.model.eval().requires_grad_(False);torch.set_num_threads(2)
        self.lane=ComputeLane();self.safety=Safety();self.prior=MapPrior(map_folder)
        self.reference_features={};self.feature_cache={};self.history=deque(maxlen=4)
        self.lock=threading.Lock();self.context=None;self.current_references={}
        self.goal_image=Image.fromarray(goal_rgb[0]);self.map_image=Image.fromarray(self.prior.rgb).resize((640,480))
        with torch.inference_mode():
            self.goals=self.model.encode(image_tensor(goal_rgb[0],self.device)[None])[None]
            for i in range(len(self.prior.tiles)):
                tokens=self.model.encode(image_tensor(self.prior.tile(i),self.device)[None])
                self.reference_features[('map',f'map-{i}')]=tokens.mean(1)
            descriptors=torch.cat([self.model.descriptors(value[:,None],True) for value in self.reference_features.values()])
            indices=(self.model.descriptors(self.goals[:,0])@descriptors.T)[0].topk(min(8,len(descriptors))).indices.tolist()
            self.goal_map_ids=indices[:4]
            self.map_tokens=torch.cat(list(self.reference_features.values()))[:,None]
            self.map_descriptors=descriptors
        self.map_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='map-retrieval')
        self.map_pending=None;self.map_last=-float('inf');self.map_observed_s=-float('inf');self.current_map_ids=[];self.map_belief=[]
        self.background=BackgroundPerception(read(self.package/'photo-slam.json'),self.package/'vision.json',self.lane)
        self.mode2=None if perception_only else Mode2(self.model,self.lane,variant,self.latest,self.calibration,
                         self.package/'qwen.pt' if 'qwen.pt' in spec['files'] else None)
        self.arrival_since=None;self.arrival_frames=0;self.last_match=0.;self.no_progress=0

    def latest(self):
        with self.lock:return self.context,dict(self.current_references)

    @torch.inference_mode()
    def step(self,metadata,rgb_bytes,perception_only=None):
        from .aerial import camera_pitch,phase_for
        if perception_only is not None and perception_only!=self.perception_only:raise ValueError('Perception capability is fixed at initialization')
        started=time.monotonic();now=metadata['sim_ns']/1e9;frame=metadata['frame_id']
        if self.started_s is None:self.started_s=now
        warming=self.branch and now-self.started_s<2.
        self.rgb_hashes.append(hashlib.sha256(rgb_bytes).hexdigest())
        if abs(camera_pitch(metadata['calibration']))>1:raise ValueError('Fixed forward camera required')
        if self.history and (frame<=self.history[-1][0] or now<=self.history[-1][1]):raise ValueError('Observation order changed')
        rgb=np.frombuffer(rgb_bytes,np.uint8).reshape(480,640,3).copy()
        spatial,local,keyframe_images=self.background.observe(metadata,rgb)
        if self.map_pending is not None and self.map_pending.done():
            try:
                self.map_observed_s,self.map_belief=self.map_pending.result();self.current_map_ids=[r.tile_id for r in self.map_belief[:4]]
            except RuntimeError:
                self.map_belief=[];self.current_map_ids=[]
            self.map_pending=None
        if now-self.map_observed_s>1.25:self.current_map_ids=[];self.map_belief=[]
        lookup=self.prior.reference_lookup(self.goal_map_ids,self.current_map_ids)
        spatial=replace(spatial,map_references=tuple(r[0] for r in lookup),map_lookup=lookup,
            map_hypotheses=tuple((r[0],float(self.prior.height(self.prior.tiles[int(r[0].split('-')[-1])])),
                float(self.prior.meta['quantization_m'])) for r in lookup))
        previous=preceding_command(metadata)
        with self.lane.fast():
            current=self.model.encode(image_tensor(rgb,self.device)[None])
            evidence=self.model.goal_evidence(current,self.goals)
            features=pool(current);self.history.append((frame,now,features,tuple(previous)))
            self.feature_cache[frame]=current.mean(1)
            while len(self.feature_cache)>128:del self.feature_cache[next(iter(self.feature_cache))]
            spatial=replace(spatial,keyframes=tuple(row for row in spatial.keyframes
                if row[1] in self.feature_cache and row[0] in keyframe_images))
            context=assemble_context(list(self.history),self.goals,spatial,metadata['received_monotonic'])
            context=replace(context,goal_context=evidence['goal_context'])
            window,actions,valid=context.features,context.commands,context.valid;times=relative_times(context)
            references=dict(self.reference_features)
            for ident,source,_ in spatial.keyframes:
                if source in self.feature_cache:references[('keyframe',ident)]=self.feature_cache[source]
            for row in spatial.geometry:references[('geometry',row[0])]=current.mean(1)
            from .subgoal_encoding import reference_descriptor
            for ident,source,stamp,*roi in spatial.image_regions:
                if source==frame:references[('image_region',ident)]=reference_descriptor(current,roi)
            with self.lock:self.context=context;self.current_references=references
            if self.map_pending is None and now-self.map_last>=1 and self.lane.slow_admitted:
                self.map_last=now;self.map_pending=self.map_pool.submit(self._locate,current.detach(),now)
            if self.perception_only:
                result=dict(frame_id=frame,observed_s=now,context=context_record(context,None),perception_only=True,
                    belief=dict(hypotheses=[asdict(r) for r in self.map_belief],aligned=False),
                    perception_error=self.background.error,work_s=time.monotonic()-started)
                if self.teacher:
                    teacher=self.teacher.step(rgb,now,frame,spatial,local,float(evidence['match_logit'].sigmoid()),float(evidence['arrival_logit'].sigmoid()))
                    result.update(teacher=teacher,source_wall=metadata['received_monotonic'],command=teacher['teacher_command'],
                        proposed_command=teacher['teacher_command'],stop=teacher['stop'],decision_id=f'{self.session_id}:{frame}',
                        work_s=time.monotonic()-started,flight_phase=teacher['phase'])
                return result
            if self.initial_subgoal is not None and not warming:
                from .contracts import Subgoal
                # Strict matching: no claims of restored state or equivalent
                # evidence when independently initialized perception differs.
                record=context_record(context,None)
                record['timestamps']=[round(t-now,3) for t in context.timestamps]
                record['frame_ids']=list(range(len(context.frame_ids)))
                record['spatial']['observed_s']=round(spatial.observed_s-now,3)
                record['spatial']['publication_wall']=0.
                record['spatial']['keyframes']=[(r[0],r[1]-frame,round(r[2]-now,3)) for r in spatial.keyframes]
                record['rgb_sha256']=list(self.rgb_hashes)
                record['goal_sha256']=hashlib.sha256(np.asarray(self.goal_image).tobytes()).hexdigest()
                self.branch_context_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
                self.mode2.active=Subgoal.proposal(self.initial_subgoal,frame,now,spatial)
                self.initial_subgoal=None
            selected=self.mode2.active
            embedding=embed_subgoal(self.model,selected,now,spatial,references,self.device)
            proposal,stop_logit=self.model.policy(window,times,actions,valid,evidence['goal_context'],embedding)
            sampled_latent=None;behavior_logprob=None
            if self.sample_policy:
                mean=torch.atanh((proposal/proposal.new_tensor([3,3,1,45])).clamp(-.999,.999))
                distribution=torch.distributions.Normal(mean,.15)
                latent=distribution.sample();behavior_logprob=float(distribution.log_prob(latent).sum())
                sampled_latent=latent[0].tolist();proposal=latent.tanh()*proposal.new_tensor([3,3,1,45])
                proposal[:,:2]/=(proposal[:,:2].norm(dim=-1,keepdim=True)/3).clamp_min(1)
            proposed=proposal[0].cpu().numpy();stop_probability=float(stop_logit.sigmoid()[0])
            match=float(evidence['match_logit'].sigmoid()[0]);arrival=float(evidence['arrival_logit'].sigmoid()[0])
        queue_s=self.lane.last_fast_queue_s
        fresh=(local is not None and 0<=now-local['observed_s']<=.25 and
               0<=time.monotonic()-local['source_wall']<=.25)
        speed=local['speed'] if local else None
        allowed=False;distance=0.
        if fresh:
            allowed,distance=self.safety.clearance(local['depth'],local['calibration'],proposed,speed if speed is not None else 6.,max(0.,now-local['observed_s']))
        command=proposed.copy() if allowed else np.zeros(4)
        command[:2]*=min(1.,2.999/max(float(np.linalg.norm(command[:2])),1e-9))
        agreement=(fresh and speed is not None and speed<.5 and
                   min(match,arrival)>=self.calibration['threshold'] and stop_probability>=.5)
        if agreement:
            if self.arrival_since is None:self.arrival_since=now
            self.arrival_frames+=1;command[:]=0
        else:self.arrival_since=None;self.arrival_frames=0
        stop=bool(agreement and self.arrival_frames>=5 and now-self.arrival_since>=1.)
        if warming:command[:]=0.;stop=False;self.arrival_since=None;self.arrival_frames=0
        self.no_progress=self.no_progress+1 if match<=self.last_match+.005 else 0;self.last_match=match
        images=[Image.fromarray(rgb),self.goal_image,self.prior.annotated(lookup)]
        images.extend(Image.fromarray(keyframe_images[row[0]]) for row in spatial.keyframes if row[0] in keyframe_images)
        completed=self.mode2.completed(context,references,match,arrival,speed)
        if not warming:self.mode2.update(context,images,triggered=self.no_progress>=60 or completed)
        used=selected if selected and selected.valid(now,spatial) else None
        return dict(decision_id=f'{self.session_id}:{frame}',frame_id=frame,observed_s=now,source_wall=metadata['received_monotonic'],
            actor_identity=self.actor_identity,
            branch_context_sha256=self.branch_context_sha256,warmup=warming,
            mode='settle' if agreement else 'actor',command=command.tolist(),proposed_command=proposed.tolist(),
            stop=stop,stop_probability=stop_probability,braked=not allowed,match=match,arrival=arrival,
            safety_reason=None if allowed else 'stale_perception' if not fresh else 'unsupported_or_blocked_corridor',
            context=context_record(context,used),selected_subgoal=asdict(used) if used else None,
            speed_mps=speed,flight_phase=phase_for(command),stopping_distance_m=distance,
            work_s=time.monotonic()-started,queue_s=queue_s,slow=self.mode2.receipt,
            perception_error=self.background.error,slow_admitted=self.lane.slow_admitted,
            background_timing={k:local.get(k) for k in ('queue_s','work_s','tracking_s','depth_and_queue_s','mapping_and_export_s')} if local else None,
            maximum_slow_slice_s=self.lane.slowest_slice_s,perception_only=self.perception_only,
            sampled_latent=sampled_latent,behavior_logprob=behavior_logprob,behavior_sha256=self.behavior_identity)

    @torch.inference_mode()
    def _locate(self,tokens,observed_s):
        from .localization import locate
        with self.lane.slow():return observed_s,locate(self.model,tokens,self.prior,self.map_tokens,self.map_descriptors)

    def close(self):
        if self.mode2:self.mode2.close()
        self.map_pool.shutdown(wait=True,cancel_futures=True);self.background.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--socket',default='/ipc/rgb.sock');parser.add_argument('--episode-id',required=True)
    parser.add_argument('--package',default='/navigation');parser.add_argument('--map',default='/prior')
    parser.add_argument('--output',type=Path,default=Path('/output'))
    parser.add_argument('--variant',choices=config()['variants'],default='mode1_vlm_world')
    parser.add_argument('--sample-policy',action='store_true')
    parser.add_argument('--perception-only',action='store_true')
    parser.add_argument('--teacher')
    parser.add_argument('--initial-subgoal')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=BrokerClient(args.socket,args.episode_id);pool=ThreadPoolExecutor(max_workers=1)
    pending=None;active=None;nav=None;frames=0;last=-1;error=None
    try:
        first,raw=client.goal_view(0)
        if first.get('view_count',1)!=1:raise ValueError('One goal image required')
        nav=Navigator(args.package,args.map,[np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()],args.variant,args.sample_policy,
                      json.loads(args.initial_subgoal) if args.initial_subgoal else None,args.perception_only,args.teacher)
        (args.output/'CONTROLLER_READY').touch()
        with (args.output/'decisions.jsonl').open('x') as log:
            while not (args.output/'CONTROLLER_STOP').exists():
                metadata,rgb=client.observe(last);last=metadata['frame_id'];frames+=1
                if pending is not None and pending.done():
                    active=pending.result();pending=None
                    log.write(json.dumps(active,allow_nan=False)+'\n');log.flush()
                if pending is None:pending=pool.submit(nav.step,dict(metadata),bytes(rgb))
                if args.perception_only and not args.teacher:continue
                command=[0.,0.,0.,0.];stop=False;source=last
                if active and 0<=time.monotonic()-active['source_wall']<=.25:
                    command=active['command'];stop=active['stop'];source=active['frame_id']
                try:client.command(source,command,stop,decision_id=active['decision_id'] if active and source==active['frame_id'] else None)
                except RuntimeError as exc:
                    if not any(s in str(exc) for s in ('Stale RGB','Stale or future command frame','no longer retained')):raise
    except Exception as exc:
        if (args.output/'CONTROLLER_STOP').exists():return
        error=type(exc).__name__+': '+str(exc)
        write(args.output/'termination.json',dict(reason='controller_error',detail=error));raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        if nav is not None:nav.close()
        client.close();write(args.output/'result.json',dict(status='failed' if error else 'runtime_finished',frames=frames,error=error,tested_claim=False,variant=args.variant))
        if args.perception_only and (args.output/'decisions.jsonl').exists():
            write(args.output/'receipt.json',dict(decisions_sha256=digest(args.output/'decisions.jsonl'),
                perception_only=True,source='live RGB broker',status='failed' if error else 'finished',
                package_sha256=digest(Path(args.package)/'package.json')))


if __name__=='__main__':main()
