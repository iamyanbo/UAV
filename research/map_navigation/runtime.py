"""Isolated inference entry point. No simulator imports or evaluator files.

One perception/planning job is in flight. The main loop owns dispatch and
brakes on stale results; a slow model cannot pause the simulator clock.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import math
from pathlib import Path
import time
import json
import numpy as np
import torch
from PIL import Image
from wire import BrokerClient
from metric_depth import MetricDepth
from .common import config, read, write, digest, contained
from .maps import MapPrior
from .models import load_model
from .records import Hypothesis, GoalBelief
from .navigation import BeliefFilter, Mission


class VisualMotion:
    """RGB feature tracking with RGB-predicted depth; no commanded-motion scale."""
    def __init__(self):
        import cv2
        self.cv=cv2;self.orb=cv2.ORB_create(nfeatures=1800);self.previous=None

    def update(self,rgb,depth,calibration,now):
        cv=self.cv
        gray=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY)
        keys,features=self.orb.detectAndCompute(gray,None)
        result=None
        if self.previous is not None and features is not None:
            old_keys,old_features,old_depth,old_time=self.previous
            if old_features is not None and 0<now-old_time<1:
                matches=cv.BFMatcher(cv.NORM_HAMMING).knnMatch(old_features,features,k=2)
                pairs=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
                points=[];pixels=[]
                for m in pairs:
                    u,v=old_keys[m.queryIdx].pt;d=float(old_depth[int(v),int(u)])
                    if .2<d<80:
                        points.append([(u-calibration['cx'])*d/calibration['fx'],(v-calibration['cy'])*d/calibration['fy'],d])
                        pixels.append(keys[m.trainIdx].pt)
                if len(points)>=16:
                    k=np.array([[calibration['fx'],0,calibration['cx']],[0,calibration['fy'],calibration['cy']],[0,0,1]],float)
                    ok,rvec,tvec,inliers=cv.solvePnPRansac(np.asarray(points,float),np.asarray(pixels,float),k,None,reprojectionError=3.,iterationsCount=100)
                    if ok and inliers is not None and len(inliers)>=12:
                        rotation=cv.Rodrigues(rvec)[0].T
                        translation=-rotation@tvec[:,0]
                        extrinsic=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
                        body_rotation=extrinsic@rotation@extrinsic.T
                        lever=np.asarray(calibration['camera_origin_body_m'])
                        delta=extrinsic@translation+lever-body_rotation@lever
                        result=dict(delta=np.r_[delta,math.atan2(body_rotation[1,0],body_rotation[0,0])],
                                    speed=float(np.linalg.norm(delta)/(now-old_time)),inliers=len(inliers))
        self.previous=(keys,features,depth,now)
        return result


def image_tensor(rgb,device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).to(device)


class Navigator:
    def __init__(self,package,map_folder,goal_rgb,variant):
        self.cfg=config();self.nav=self.cfg['navigation'];self.variant=variant
        self.package=Path(package);spec=read(self.package/'package.json')
        self.learned_local_policy=spec.get('learned_local_policy',False);self.policy_hidden=None
        if spec['schema']!='photo-map-package/v1':raise ValueError('New runtime package required')
        if set(spec['files'])!={'model.pt','vision.json'}:raise ValueError('Unexpected inference assets')
        for name,sha in spec['files'].items():
            if digest(contained(self.package,name))!=sha:raise ValueError('Changed model package')
        self.device='cuda';torch.set_num_threads(2)
        self.model,self.saved=load_model('/models/mobilenet-v3-large-imagenet1k-v2.pt',self.package/'model.pt')
        if digest('/models/mobilenet-v3-large-imagenet1k-v2.pt')!=self.saved['backbone_sha256']:
            raise ValueError('Runtime backbone differs from training')
        required={'localization','goal'}|({'world'} if variant!='geometry' else set())
        if not required<=set(self.saved['trained_stages']):raise ValueError('Missing trained stages: '+str(required-set(self.saved['trained_stages'])))
        self.model.eval().requires_grad_(False)
        self.prior=MapPrior(map_folder);self.filter=BeliefFilter(self.nav);self.mission=Mission(self.prior,self.nav)
        self.motion=VisualMotion();self.depth=MetricDepth(self.package/'vision.json')
        self.goal_images=[Image.fromarray(x) for x in goal_rgb]
        with torch.inference_mode():
            self.goals=self.model.encoder(torch.stack([image_tensor(x,self.device) for x in goal_rgb]))[None]
            self.map_tokens=torch.cat([self.model.encoder(torch.stack([image_tensor(self.prior.tile(j),self.device)
                         for j in range(i,min(i+4,len(self.prior.tiles)))])) for i in range(0,len(self.prior.tiles),4)])
            self.map_descriptors=self.model.descriptors(self.map_tokens,True)
            self.goal_candidates=self.locate(self.goals.flatten(0,1),average=True)
        self.history=deque(maxlen=32);self.last_qwen=-math.inf;self.qwen=None
        self.planning_pool=ThreadPoolExecutor(max_workers=1);self.planning_pending=None;self.active_plan=None
        self.qwen_pool=ThreadPoolExecutor(max_workers=1);self.qwen_pending=None;self.qwen_active=None
        if variant=='map_world_qwen':
            from configurator import Configurator
            self.qwen=Configurator('/models/qwen2.5-vl-3b',output_format='compact/v2')
            self.qwen.model.eval().requires_grad_(False)
            self.qwen.enable_goal_cache()

    def close(self):
        self.planning_pool.shutdown(wait=True,cancel_futures=True)
        self.qwen_pool.shutdown(wait=True,cancel_futures=True)

    @torch.inference_mode()
    def predict(self,tokens,commands,maps,subgoal,memory,belief,now):
        actions=tokens.new_tensor(commands)[:,None,None].expand(-1,20,4,-1)
        prediction=self.model.future(tokens.expand(len(commands),-1,-1),maps.expand(len(commands),-1,-1),
            self.goals.expand(len(commands),-1,-1,-1),actions,use_map=self.variant!='recent_world',
            memory=None if memory is None else memory.expand(len(commands),-1,-1))
        scores=prediction['goal'][:,-1].sigmoid()+.1*prediction['information'][:,-1]-2*prediction['collision'].sigmoid().amax(1)
        if subgoal is not None:scores-=.05*(prediction['state'][:,-1,:3]-tokens.new_tensor(subgoal)).norm(dim=-1)
        return dict(command=commands[int(scores.argmax())],scores=scores.cpu().tolist(),observed_s=now,
                    origin=belief.best.position,alignment=belief.alignment_version,created_wall=time.monotonic())

    def locate(self,tokens,average=False):
        descriptors=self.model.descriptors(tokens)
        score=(descriptors@self.map_descriptors.T).mean(0) if average else (descriptors@self.map_descriptors.T)[0]
        indices=score.topk(min(8,len(score))).indices
        camera=tokens.mean(0,keepdim=True).expand(len(indices),-1,-1)
        prediction=self.model.register(camera,self.map_tokens[indices])
        probabilities=(score[indices]/.07).softmax(0)
        candidates=[]
        for j,index in enumerate(indices.tolist()):
            xy=self.prior.tiles[index]+prediction['offset'][j].cpu().numpy()
            surface=float(self.prior.height(xy))
            if not math.isfinite(surface):continue
            position=(*xy,surface-float(prediction['above_surface'][j]))
            confidence=float(probabilities[j]*prediction['match_logit'][j].sigmoid())
            yaw=math.atan2(float(prediction['yaw'][j,0]),float(prediction['yaw'][j,1]))
            candidates.append(Hypothesis(tuple(float(x) for x in position),yaw,confidence,float(prediction['sigma'][j]),index))
        return tuple(sorted(candidates,key=lambda r:r.probability,reverse=True))

    def clearance(self,depth,calibration,command,speed,age):
        velocity=np.asarray(command[:3]);proposed=float(np.linalg.norm(velocity))
        if proposed<1e-5:return True,0.
        braking=max(speed,proposed);distance=braking*age+braking**2/(2*self.nav['braking_mps2'])+.75
        rotation=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
        origin=np.asarray(calibration['camera_origin_body_m'])
        points=np.linspace(0,distance,max(3,int(distance/.25)))[:,None]*velocity[None]/proposed
        points=(points-origin)@rotation
        for p in points:
            if np.linalg.norm(p)<.8:continue
            if p[2]<=0:return False,distance
            u=int(calibration['fx']*p[0]/p[2]+calibration['cx']);v=int(calibration['fy']*p[1]/p[2]+calibration['cy'])
            radius=int(max(calibration['fx'],calibration['fy'])*.9/p[2])+1
            if not 0<=u<640 or not 0<=v<480:return False,distance
            # Visible corridor only; this monocular veto is not a certified
            # swept-volume guarantee for portions outside the camera frustum.
            patch=depth[max(0,v-radius):min(480,v+radius+1),max(0,u-radius):min(640,u+radius+1)]
            if np.any(patch<=0) or float(np.min(patch))*.7<=p[2]+.25:return False,distance
        return True,distance

    @torch.inference_mode()
    def step(self,metadata,rgb_bytes):
        started=time.monotonic();now=metadata['sim_ns']/1e9
        rgb=np.frombuffer(rgb_bytes,np.uint8).reshape(480,640,3).copy();cal=metadata['calibration']
        tokens=self.model.encoder(image_tensor(rgb,self.device)[None])
        depth=self.depth(rgb,cal).numpy()
        motion=self.motion.update(rgb,depth,cal,now)
        belief=self.filter.update(self.locate(tokens),now,None if motion is None else motion['delta'])
        evidence=self.model.goal_evidence(tokens,self.goals)
        match=float(evidence['match_logit'].sigmoid());arrival=float(evidence['arrival_logit'].sigmoid())
        goal=GoalBelief(self.goal_candidates,not self.goal_candidates or self.goal_candidates[0].probability<.8,now)
        speed=motion['speed'] if motion else math.inf
        decision=self.mission.step(belief,goal,match,arrival,speed,now)
        if 'subgoal_body' in decision:
            decision['teacher_command']=list(decision['command'])
            decision['expert_observation_conditioned']=True
        if self.learned_local_policy and 'subgoal_body' in decision:
            command,self.policy_hidden=self.model.local_action(tokens,evidence['goal_context'],
                tokens.new_tensor([decision['subgoal_body']]),tokens.new_zeros(1,4),None)
            decision['command']=command[0].cpu().tolist()
        if not belief.aligned:self.policy_hidden=None
        # Current depth endpoints supplement (never clear) the immutable map.
        if belief.aligned and motion:
            ys,xs=np.mgrid[0:480:24,0:640:24];d=depth[::24,::24]
            good=(d>.2)&(d<30)
            points=np.stack(((xs-cal['cx'])*d/cal['fx'],(ys-cal['cy'])*d/cal['fy'],d),-1)[good]
            extrinsic=np.asarray(cal['camera_to_body_rotation']).reshape(3,3)
            points=points@extrinsic.T+np.asarray(cal['camera_origin_body_m'])
            c,s=math.cos(belief.best.yaw),math.sin(belief.best.yaw)
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            self.mission.add_obstacles(points@rotation.T+belief.best.position)
        if self.variant!='geometry' and belief.aligned and motion and decision['mode'] in ('search','approach'):
            if self.planning_pending is not None and self.planning_pending.done():
                self.active_plan=self.planning_pending.result();self.planning_pending=None
            commands=[decision['command'],[0.,0.,0.,15.],[0.,0.,0.,-15.],[0.,0.,0.,0.]]
            commands=[c for c in commands if self.clearance(depth,cal,c,speed,0.)[0]]
            if commands and self.planning_pending is None:
                idx=belief.best.tile_id
                memory=torch.cat([row['tokens'] for row in self.history],1) if self.history else None
                self.planning_pending=self.planning_pool.submit(self.predict,tokens,commands,self.map_tokens[idx:idx+1],
                                      decision.get('subgoal_body'),memory,belief,now)
            plan=self.active_plan
            if (plan and 0<=now-plan['observed_s']<=1 and plan['alignment']==belief.alignment_version
                    and np.linalg.norm(np.asarray(plan['origin'])-belief.best.position)<=2):
                decision['command']=plan['command'];decision['prediction_scores']=plan['scores']
                decision['prediction_age_s']=now-plan['observed_s']
        if self.qwen_pending is not None and self.qwen_pending.done():
            self.qwen_active=self.qwen_pending.result();self.qwen_pending=None
        if self.qwen and self.qwen_pending is None and belief.aligned and now-self.last_qwen>=3 and self.history:
            observed=[dict(id=f'view-{i}',kind='observed_frontier',observed_ns=int(r['now']*1e9),image_position=len(self.goal_images)+2+i)
                      for i,r in enumerate(list(self.history)[-3:])]
            self.qwen_pending=self.qwen_pool.submit(self.qwen.configure,self.goal_images,Image.fromarray(rgb),[r['image'] for r in list(self.history)[-3:]],[],observed,
                    {'match':match,'mode':decision['mode']},metadata['episode_id'],now)
            self.last_qwen=now
        cfg=self.qwen_active
        # Qwen may ask for inspection; late output cannot command or stop.
        if cfg and now<=cfg.valid_until_sim_seconds and cfg.intention=='inspect':decision['command']=[0.,0.,0.,15.]
        decision['qwen']=None if cfg is None else asdict(cfg)
        if not self.history or now-self.history[-1]['now']>=1:
            self.history.append(dict(now=now,tokens=tokens.detach(),image=Image.fromarray(rgb)))
        command=np.asarray(decision['command'],dtype=float)
        if command.shape!=(4,) or not np.isfinite(command).all():raise ValueError('Invalid planned command')
        command[:2]*=min(1.,3./max(float(np.linalg.norm(command[:2])),1e-9))
        command[2]=np.clip(command[2],-1.,1.);command[3]=np.clip(command[3],-45.,45.)
        decision['command']=command.tolist()
        allowed,stopping=self.clearance(depth,cal,decision['command'],speed if math.isfinite(speed) else 6.,time.monotonic()-metadata['received_monotonic'])
        if not allowed:decision['command']=[0.,0.,0.,0.];decision['stop']=False
        decision.update(observed_s=now,frame_id=metadata['frame_id'],source_wall=metadata['received_monotonic'],
            match=match,arrival=arrival,belief=asdict(belief),braked=not allowed,stopping_distance_m=stopping,
            work_s=time.monotonic()-started,speed_mps=speed if math.isfinite(speed) else None)
        return decision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--socket',default='/ipc/rgb.sock');parser.add_argument('--episode-id',required=True)
    parser.add_argument('--package',default='/navigation');parser.add_argument('--map',default='/prior')
    parser.add_argument('--output',type=Path,default=Path('/output'))
    parser.add_argument('--variant',choices=config()['variants'],default='geometry')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=BrokerClient(args.socket,args.episode_id);pool=ThreadPoolExecutor(max_workers=1)
    error=None;pending=None;active=None;frames=0;last=-1;nav=None
    try:
        first,_=client.goal_view(0)
        views=[np.frombuffer(client.goal_view(i)[1],np.uint8).reshape(480,640,3).copy() for i in range(first.get('view_count',4))]
        nav=Navigator(args.package,args.map,views,args.variant)
        (args.output/'CONTROLLER_READY').touch()
        with (args.output/'decisions.jsonl').open('x') as log:
            while not (args.output/'CONTROLLER_STOP').exists():
                metadata,rgb=client.observe(last);last=metadata['frame_id'];frames+=1
                if pending and pending.done():
                    active=pending.result();pending=None
                    log.write(json.dumps(active,allow_nan=False)+'\n');log.flush()
                if pending is None:pending=pool.submit(nav.step,dict(metadata),bytes(rgb))
                command=[0.,0.,0.,0.];stop=False
                if active and time.monotonic()-active['source_wall']<=config()['navigation']['maximum_result_age_s']:
                    command=active['command'];stop=active['stop']
                if active and active['mode']=='localization_failure':
                    write(args.output/'termination.json',dict(reason='localization_failure'))
                    active=None
                try:client.command(last,command,stop)
                except RuntimeError as exc:
                    if 'Stale RGB' not in str(exc):raise
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        write(args.output/'termination.json',dict(reason='controller_error',detail=error))
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        if nav is not None:nav.close()
        client.close()
        write(args.output/'result.json',dict(status='failed' if error else 'runtime_finished',frames=frames,error=error,
              tested_claim=False,variant=args.variant))


if __name__=='__main__':main()
