"""Mode 1 owns vehicle commands; Mode 2 only publishes expiring subgoals."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace,asdict
import json
from pathlib import Path
import threading
import time
import numpy as np
import torch
from PIL import Image
from wire import BrokerClient
from .common import config,read,write,digest,contained
from .models import load_model
from .maps import MapPrior
from .contracts import ObservationContext,context_record
from .temporal import pool,embed_subgoal
from .compute import ComputeLane
from .perception import BackgroundPerception
from .subgoals import Mode2
from .safety import Safety


def image_tensor(rgb,device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).to(device)


class Navigator:
    def __init__(self,package,map_folder,goal_rgb,variant,sample_policy=False,initial_subgoal=None):
        if len(goal_rgb)!=1:raise ValueError('Exactly one goal photograph required')
        if variant not in config()['variants']:raise ValueError('Unknown temporal variant')
        self.device='cuda';self.variant=variant;self.package=Path(package);self.sample_policy=sample_policy
        self.behavior_identity=digest(self.package/'model.pt')
        self.initial_subgoal=initial_subgoal
        spec=read(self.package/'package.json')
        if spec['schema']!='photo-map-package/v4':raise ValueError('Temporal-window package required')
        required={'model.pt','vision.json','photo-slam.json'}
        if not required<=set(spec['files']) or set(spec['files'])-required-{'qwen.pt'}:raise ValueError('Unexpected assets')
        for name,sha in spec['files'].items():
            if digest(contained(self.package,name))!=sha:raise ValueError('Changed inference package')
        self.model,self.saved=load_model('/models/mobilenet-v3-large-imagenet1k-v2.pt',self.package/'model.pt')
        if digest('/models/mobilenet-v3-large-imagenet1k-v2.pt')!=self.saved['backbone_sha256']:raise ValueError('Backbone mismatch')
        stages={'localization','goal','policy'}|({'world'} if variant=='mode1_vlm_world' else set())
        if not stages<=set(self.saved['trained_stages']):raise ValueError('Untrained runtime stages')
        self.calibration=self.saved['calibration']
        if variant=='mode1_vlm_world' and 'score_scales' not in self.calibration:raise ValueError('Validate world score scales first')
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
            self.map_references=tuple(f'map-{i}' for i in indices)
            self.map_hypotheses=tuple((f'map-{i}',float(self.prior.height(self.prior.tiles[i])),2.) for i in indices)
        self.background=BackgroundPerception(read(self.package/'photo-slam.json'),self.package/'vision.json',self.lane)
        self.mode2=Mode2(self.model,self.lane,variant,self.latest,self.calibration,
                         self.package/'qwen.pt' if 'qwen.pt' in spec['files'] else None)
        self.arrival_since=None;self.arrival_frames=0;self.last_match=0.;self.no_progress=0

    def latest(self):
        with self.lock:return self.context,dict(self.current_references)

    @torch.inference_mode()
    def step(self,metadata,rgb_bytes,perception_only=False):
        from .aerial import camera_pitch,phase_for
        started=time.monotonic();now=metadata['sim_ns']/1e9;frame=metadata['frame_id']
        if abs(camera_pitch(metadata['calibration']))>1:raise ValueError('Fixed forward camera required')
        if self.history and (frame<=self.history[-1][0] or now<=self.history[-1][1]):raise ValueError('Observation order changed')
        rgb=np.frombuffer(rgb_bytes,np.uint8).reshape(480,640,3).copy()
        spatial,local=self.background.observe(metadata,rgb)
        spatial=replace(spatial,map_references=self.map_references,map_hypotheses=self.map_hypotheses)
        commands=[r for r in metadata.get('command_history',[]) if r['sim_ns']<=metadata['sim_ns']]
        previous=commands[-1]['values'] if commands else [0.,0.,0.,0.]
        with self.lane.fast():
            current=self.model.encode(image_tensor(rgb,self.device)[None])
            evidence=self.model.goal_evidence(current,self.goals)
            features=pool(current);self.history.append((frame,now,features,tuple(previous)))
            self.feature_cache[frame]=current.mean(1)
            while len(self.feature_cache)>128:del self.feature_cache[next(iter(self.feature_cache))]
            spatial=replace(spatial,keyframes=tuple(row for row in spatial.keyframes
                if row[1] in self.feature_cache and row[0] in self.background.keyframes))
            window=features.new_zeros(1,4,64,256);actions=features.new_zeros(1,4,4)
            times=features.new_zeros(1,4);valid=torch.zeros(1,4,dtype=torch.bool,device=self.device)
            for j,(_,stamp,feature,command) in enumerate(self.history,4-len(self.history)):
                window[:,j]=feature;actions[:,j]=actions.new_tensor(command);times[:,j]=stamp-now;valid[:,j]=True
            context=ObservationContext('observation-context/v1',tuple(r[0] for r in self.history),
                tuple(r[1] for r in self.history),window,actions,valid,self.goals,evidence['goal_context'],
                spatial,metadata['received_monotonic']).validate()
            references=dict(self.reference_features)
            for ident,source,_ in spatial.keyframes:
                if source in self.feature_cache:references[('keyframe',ident)]=self.feature_cache[source]
            for row in spatial.geometry:references[('geometry',row[0])]=current.mean(1)
            with self.lock:self.context=context;self.current_references=references
            if self.initial_subgoal is not None:
                from .contracts import Subgoal
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
        self.no_progress=self.no_progress+1 if match<=self.last_match+.005 else 0;self.last_match=match
        images=[Image.fromarray(rgb),self.goal_image,self.map_image]
        images.extend(Image.fromarray(self.background.keyframes[row[0]]) for row in spatial.keyframes if row[0] in self.background.keyframes)
        self.mode2.update(context,images,triggered=self.no_progress>=60 or (selected is not None and match>.95))
        used=selected if selected and selected.valid(now,spatial) else None
        return dict(frame_id=frame,observed_s=now,source_wall=metadata['received_monotonic'],
            mode='settle' if agreement else 'actor',command=command.tolist(),proposed_command=proposed.tolist(),
            stop=stop,stop_probability=stop_probability,braked=not allowed,match=match,arrival=arrival,
            context=context_record(context,used),selected_subgoal=asdict(used) if used else None,
            speed_mps=speed,flight_phase=phase_for(command),stopping_distance_m=distance,
            work_s=time.monotonic()-started,queue_s=queue_s,slow=self.mode2.receipt,
            perception_error=self.background.error,slow_admitted=self.lane.slow_admitted,
            background_timing={k:local[k] for k in ('queue_s','work_s','tracking_s','depth_and_queue_s','mapping_and_export_s')} if local else None,
            maximum_slow_slice_s=self.lane.slowest_slice_s,perception_only=perception_only,
            sampled_latent=sampled_latent,behavior_logprob=behavior_logprob,behavior_sha256=self.behavior_identity)

    def close(self):self.mode2.close();self.background.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--socket',default='/ipc/rgb.sock');parser.add_argument('--episode-id',required=True)
    parser.add_argument('--package',default='/navigation');parser.add_argument('--map',default='/prior')
    parser.add_argument('--output',type=Path,default=Path('/output'))
    parser.add_argument('--variant',choices=config()['variants'],default='mode1_vlm_world')
    parser.add_argument('--sample-policy',action='store_true')
    parser.add_argument('--initial-subgoal')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=BrokerClient(args.socket,args.episode_id);pool=ThreadPoolExecutor(max_workers=1)
    pending=None;active=None;nav=None;frames=0;last=-1;error=None
    try:
        first,raw=client.goal_view(0)
        if first.get('view_count',1)!=1:raise ValueError('One goal image required')
        nav=Navigator(args.package,args.map,[np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()],args.variant,args.sample_policy,
                      json.loads(args.initial_subgoal) if args.initial_subgoal else None)
        (args.output/'CONTROLLER_READY').touch()
        with (args.output/'decisions.jsonl').open('x') as log:
            while not (args.output/'CONTROLLER_STOP').exists():
                metadata,rgb=client.observe(last);last=metadata['frame_id'];frames+=1
                if pending is not None and pending.done():
                    active=pending.result();pending=None
                    log.write(json.dumps(active,allow_nan=False)+'\n');log.flush()
                if pending is None:pending=pool.submit(nav.step,dict(metadata),bytes(rgb))
                command=[0.,0.,0.,0.];stop=False;source=last
                if active and 0<=time.monotonic()-active['source_wall']<=.25:
                    command=active['command'];stop=active['stop'];source=active['frame_id']
                try:client.command(source,command,stop)
                except RuntimeError as exc:
                    if not any(s in str(exc) for s in ('Stale RGB','Stale or future command frame','no longer retained')):raise
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        write(args.output/'termination.json',dict(reason='controller_error',detail=error));raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        if nav is not None:nav.close()
        client.close();write(args.output/'result.json',dict(status='failed' if error else 'runtime_finished',frames=frames,error=error,tested_claim=False,variant=args.variant))


if __name__=='__main__':main()
