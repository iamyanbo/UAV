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
            old_keys,old_features,old_depth,old_time,old_calibration=self.previous
            if old_features is not None and 0<now-old_time<1:
                matches=cv.BFMatcher(cv.NORM_HAMMING).knnMatch(old_features,features,k=2)
                pairs=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
                points=[];pixels=[]
                for m in pairs:
                    u,v=old_keys[m.queryIdx].pt;d=float(old_depth[int(v),int(u)])
                    if .2<d<80:
                        points.append([(u-old_calibration['cx'])*d/old_calibration['fx'],(v-old_calibration['cy'])*d/old_calibration['fy'],d])
                        pixels.append(keys[m.trainIdx].pt)
                if len(points)>=16:
                    k=np.array([[calibration['fx'],0,calibration['cx']],[0,calibration['fy'],calibration['cy']],[0,0,1]],float)
                    ok,rvec,tvec,inliers=cv.solvePnPRansac(np.asarray(points,float),np.asarray(pixels,float),k,None,reprojectionError=3.,iterationsCount=100)
                    if ok and inliers is not None and len(inliers)>=12:
                        rotation=cv.Rodrigues(rvec)[0].T
                        translation=-rotation@tvec[:,0]
                        extrinsic=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
                        old_extrinsic=np.asarray(old_calibration['camera_to_body_rotation']).reshape(3,3)
                        body_rotation=old_extrinsic@rotation@extrinsic.T
                        lever=np.asarray(calibration['camera_origin_body_m'])
                        old_lever=np.asarray(old_calibration['camera_origin_body_m'])
                        delta=old_extrinsic@translation+old_lever-body_rotation@lever
                        result=dict(delta=np.r_[delta,math.atan2(body_rotation[1,0],body_rotation[0,0])],
                                    speed=float(np.linalg.norm(delta)/(now-old_time)),inliers=len(inliers),delta_seconds=now-old_time)
        self.previous=(keys,features,depth,now,dict(calibration))
        return result


def image_tensor(rgb,device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).to(device)


class Navigator:
    def __init__(self,package,map_folder,goal_rgb,variant):
        from .compute import ComputeLane
        self.cfg=config();self.nav=self.cfg['navigation'];self.variant=variant
        if variant not in self.cfg['variants']:raise ValueError('Unknown aerial comparison variant')
        self.package=Path(package);spec=read(self.package/'package.json')
        if spec['schema']!='photo-map-package/v2':raise ValueError('New camera-aware package required')
        if spec.get('learned_local_policy'):raise ValueError('Learned actor is deferred')
        if set(spec['files'])!={'model.pt','vision.json'}:raise ValueError('Unexpected inference assets')
        for name,sha in spec['files'].items():
            if digest(contained(self.package,name))!=sha:raise ValueError('Changed package')
        self.device='cuda';torch.set_num_threads(2)
        self.model,self.saved=load_model('/models/mobilenet-v3-large-imagenet1k-v2.pt',self.package/'model.pt')
        if digest('/models/mobilenet-v3-large-imagenet1k-v2.pt')!=self.saved['backbone_sha256']:raise ValueError('Backbone mismatch')
        required={'localization','goal'}|({'world'} if variant=='predictive_candidates' else set())
        if not required<=set(self.saved['trained_stages']):raise ValueError('Untrained runtime stages')
        self.nav['goal_threshold']=float(self.saved['calibration']['threshold'])
        self.model.eval().requires_grad_(False);self.lane=ComputeLane()
        self.prior=MapPrior(map_folder);self.filter=BeliefFilter(self.nav);self.mission=Mission(self.prior,self.nav)
        self.motion=VisualMotion();self.depth=MetricDepth(self.package/'vision.json')
        with torch.inference_mode():
            self.goals=self.model.encode(torch.stack([image_tensor(x,self.device) for x in goal_rgb]),[0.]*len(goal_rgb))[None]
            self.map_tokens=torch.cat([self.model.encode(torch.stack([image_tensor(self.prior.tile(j),self.device)
                for j in range(i,min(i+4,len(self.prior.tiles)))])) for i in range(0,len(self.prior.tiles),4)])
            self.map_descriptors=self.model.descriptors(self.map_tokens,True)
            self.goal_candidates=self.locate(self.goals.flatten(0,1),average=True)
        self.history=deque(maxlen=32);self.last_plan_s=-math.inf;self.last_obstacles_s=-math.inf
        self.planning_pool=ThreadPoolExecutor(max_workers=1);self.planning_pending=None;self.active_plan=None
        self.prediction_receipt=None

    def close(self):
        self.planning_pool.shutdown(wait=True,cancel_futures=True)
        self.mission.close()

    @torch.inference_mode()
    def predict(self,tokens,candidates,maps,memory,state,belief,now,target_id,obstacle_version):
        started=time.monotonic();scores=[];components=[]
        try:
            for candidate in candidates:
                if time.monotonic()-started>1.:
                    return dict(discarded='prediction_deadline',observed_s=now)
                with self.lane.slow():
                    actions=tokens.new_tensor(candidate.actions)[None]
                predicted=self.model.future(tokens,maps,self.goals,actions,memory=memory,runtime_state=state,
                                             compute_guard=self.lane.slow)
                with self.lane.slow():
                    collision=float(predicted['collision'].sigmoid().amax())
                    goal=float(predicted['goal'][0,-1].sigmoid())
                    error=float((predicted['state'][0,-1,:3]-tokens.new_tensor(candidate.terminal_body)).norm())
                remaining=candidate.remaining_seconds+error/self.nav['maximum_speed_mps']
                score=goal-2*collision-.05*remaining
                scores.append(score);components.append(dict(candidate_id=candidate.identity,goal=goal,
                    collision_evidence=collision,remaining_seconds=remaining,score=score))
        except RuntimeError as exc:
            if str(exc)!='prediction_slice_budget_exceeded':raise
            return dict(discarded=str(exc),observed_s=now)
        chosen=candidates[int(np.argmax(scores))]
        return dict(actions=chosen.actions.tolist(),candidate_id=chosen.identity,route_family=chosen.route_family,
            components=components,observed_s=now,origin=belief.best.position,alignment=belief.alignment_version,
            target_id=target_id,obstacle_version=obstacle_version,work_s=time.monotonic()-started)

    def locate(self,tokens,average=False):
        from .localization import locate
        return locate(self.model,tokens,self.prior,self.map_tokens,self.map_descriptors,average)

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
            if not np.isfinite(patch).all() or np.any(patch<=0) or float(np.min(patch))*.7<=p[2]+.25:return False,distance
        return True,distance

    @torch.inference_mode()
    def step(self,metadata,rgb_bytes,perception_only=False):
        from .aerial import camera_pitch,phase_for
        from .candidates import generate,select_geometric
        started=time.monotonic();now=metadata['sim_ns']/1e9
        rgb=np.frombuffer(rgb_bytes,np.uint8).reshape(480,640,3).copy();cal=metadata['calibration'];pitch=camera_pitch(cal)
        with self.lane.fast():
            tokens=self.model.encode(image_tensor(rgb,self.device)[None],[pitch])
            depth=self.depth(rgb,cal).numpy()
            located=self.locate(tokens) if float(rgb.mean(-1).std())>=5 else ()
            evidence=self.model.goal_evidence(tokens,self.goals)
            match=float(evidence['match_logit'].sigmoid());arrival=float(evidence['arrival_logit'].sigmoid())
        motion=self.motion.update(rgb,depth,cal,now)
        belief=self.filter.update(located,now,None if motion is None else motion['delta'])
        state=np.zeros(32,np.float32);state[6]=1;state[10]=1
        state[18]=math.sin(math.radians(pitch));state[19]=math.cos(math.radians(pitch))
        state[20]=float(cal.get('camera_settled',True));state[21]=float(belief.aligned)
        state[22]=float(motion is not None)
        if motion:
            state[3:6]=motion['delta'][:3]/motion['delta_seconds'];state[23]=motion['delta'][3]/motion['delta_seconds']
        if belief.best:state[16]=belief.best.sigma_m;state[14]=belief.best.probability
        if perception_only:
            return dict(frame_id=metadata['frame_id'],observed_s=now,predictor_state=state.tolist(),belief=asdict(belief),
                        match=match,arrival=arrival,work_s=time.monotonic()-started,flight_phase='unlabelled')
        if belief.aligned:
            self.mission.observe(belief.best,cal,depth)
            if motion and now-self.last_obstacles_s>=1:
                ys,xs=np.mgrid[0:480:24,0:640:24];d=depth[::24,::24];good=np.isfinite(d)&(d>.2)&(d<30)
                points=np.stack(((xs-cal['cx'])*d/cal['fx'],(ys-cal['cy'])*d/cal['fy'],d),-1)[good]
                points=points@np.asarray(cal['camera_to_body_rotation']).reshape(3,3).T+np.asarray(cal['camera_origin_body_m'])
                c,s=math.cos(belief.best.yaw),math.sin(belief.best.yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                self.mission.add_obstacles(points@rotation.T+belief.best.position);self.last_obstacles_s=now
        goal=GoalBelief(self.goal_candidates,not self.goal_candidates or self.goal_candidates[0].probability<.8,now)
        speed=motion['speed'] if motion else math.inf
        decision=self.mission.step(belief,goal,match,arrival,speed,now)
        if self.planning_pending is not None and self.planning_pending.done():
            result=self.planning_pending.result();self.planning_pending=None;self.prediction_receipt=result
            if 'discarded' not in result:self.active_plan=result
        if self.variant!='geometry' and belief.aligned and motion and decision['mode'] in ('search','approach'):
            if now-self.last_plan_s>=1/self.nav['prediction_hz']:
                candidates=generate(self.mission,belief,decision,pitch);self.last_plan_s=now
                decision['candidate_ids']=[c.identity for c in candidates]
                if candidates:
                    chosen=select_geometric(candidates)
                    geometric=dict(actions=chosen.actions.tolist(),candidate_id=chosen.identity,route_family=chosen.route_family,
                        observed_s=now,origin=belief.best.position,alignment=belief.alignment_version,
                        target_id=decision.get('target_id'),obstacle_version=self.mission.obstacle_version)
                    self.active_plan=geometric
                    if self.variant=='predictive_candidates' and self.planning_pending is None and self.lane.slow_admitted:
                        memory=torch.cat([r['tokens'] for r in self.history],1) if self.history else None
                        self.planning_pending=self.planning_pool.submit(self.predict,tokens,candidates,
                            self.map_tokens[belief.best.tile_id:belief.best.tile_id+1],memory,
                            tokens.new_tensor(state)[None],belief,now,decision.get('target_id'),self.mission.obstacle_version)
            plan=self.active_plan
            if (plan and 0<=now-plan['observed_s']<=1 and plan['alignment']==belief.alignment_version
                and plan['target_id']==decision.get('target_id') and plan['obstacle_version']==self.mission.obstacle_version
                and np.linalg.norm(np.asarray(plan['origin'])-belief.best.position)<=2):
                slot=min(79,int((now-plan['observed_s'])/.05));action=np.asarray(plan['actions']).reshape(80,5)[slot]
                decision['command']=action[:4].tolist();decision['camera_pitch_deg']=float(action[4])
                decision['selected_candidate']=plan['candidate_id'];decision['route_family']=plan['route_family']
                decision['prediction_used']='components' in plan
                decision['prediction_age_s']=now-plan['observed_s']
                if 'components' in plan:decision['prediction_components']=plan['components']
        if not self.history or now-self.history[-1]['now']>=1:
            self.history.append(dict(now=now,tokens=tokens.detach()))
        command=np.asarray(decision['command'],float)
        if command.shape!=(4,) or not np.isfinite(command).all():raise ValueError('Invalid planned command')
        command[:2]*=min(1.,3./max(np.linalg.norm(command[:2]),1e-9));command[2]=np.clip(command[2],-1,1);command[3]=np.clip(command[3],-45,45)
        requested_pitch=float(decision.get('camera_pitch_deg',pitch))
        if not cal.get('camera_settled',True) or abs(requested_pitch-pitch)>2:command[:3]=0
        allowed,stopping=self.clearance(depth,cal,command,speed if math.isfinite(speed) else 6.,time.monotonic()-metadata['received_monotonic'])
        if not allowed:command[:]=0;decision['stop']=False
        if motion is None and np.linalg.norm(command[:3])>.01:command[:3]=0;decision['stop']=False
        decision['command']=command.tolist()
        decision.update(observed_s=now,frame_id=metadata['frame_id'],source_wall=metadata['received_monotonic'],
            match=match,arrival=arrival,belief=asdict(belief),braked=not allowed,stopping_distance_m=stopping,
            work_s=time.monotonic()-started,speed_mps=speed if math.isfinite(speed) else None,
            predictor_state=state.tolist(),flight_phase=phase_for(command,decision['mode']),
            slow_admitted=self.lane.slow_admitted,maximum_slow_slice_s=self.lane.slowest_slice_s,
            last_prediction_discard=(self.prediction_receipt or {}).get('discarded'))
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
                command=[0.,0.,0.,0.];stop=False;source_frame=last;candidate=None
                if active and time.monotonic()-active['source_wall']<=config()['navigation']['maximum_result_age_s']:
                    command=active['command'];stop=active['stop'];source_frame=active['frame_id'];candidate=active.get('selected_candidate','geometry')
                if active and active['mode']=='localization_failure':
                    write(args.output/'termination.json',dict(reason='localization_failure'))
                    active=None
                try:client.command(source_frame,command,stop,camera_pitch_deg=active.get('camera_pitch_deg',0.) if active else 0.,candidate_id=candidate)
                except RuntimeError as exc:
                    if 'Stale RGB' not in str(exc) and 'Stale or future command frame' not in str(exc) and 'no longer retained' not in str(exc):raise
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
