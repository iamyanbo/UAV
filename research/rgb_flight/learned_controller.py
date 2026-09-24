"""Live Mode 1 actor with episode-local RGB state and mandatory geometry safety.

This is the planner-disabled controller used for the initial policy/PPO stage.
It requires trained compatible policy weights and a measured safety profile.
Startup uses bounded simulator exploration; mapped flight uses metric safety.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from async_mapping import AsyncMapping
from async_video import AsyncVideoFeatures
from contracts import Command
from controller import HardSafetyFilter
from startup import demonstration, VERSION
from action_intervals import ACTION_SEMANTICS
from runtime_capacity import reserve_fast
from learning_models import RecurrentPolicy
from navigation_state import CheckpointSet,CausalNavigationState,checksum
from wire import BrokerClient


class Mode1Actor:
    def __init__(self,core,maximum_speed,stochastic=False,demonstrate=False):
        pack=core.checkpoints
        if not {'safety','projection'}<=pack.paths.keys():
            raise ValueError('Live navigation requires projection and measured safety artifacts')
        self.policy=None;self.policy_state_dim=32
        if stochastic and demonstrate:raise ValueError('Demonstrations are not stochastic PPO rollouts')
        if not demonstrate:
            if 'policy' not in pack.paths:raise ValueError('Learned navigation requires a trained Mode 1 policy')
            saved=torch.load(pack.paths['policy'],map_location='cpu',weights_only=True)
            if saved.get('module')!='policy' or saved.get('update',0)<1:
                raise ValueError('A trained Mode 1 checkpoint is required')
            required={role:pack.spec['artifacts'][role]['sha256'] for role in ('goal','odometry','projection','vision') if role in pack.spec['artifacts']}
            if saved.get('perception_artifacts_sha256')!=required:
                raise ValueError('Policy perception provenance differs from the live checkpoint set')
            self.policy=RecurrentPolicy(pack.paths['goal'],state_dim=32,depth_input='vision' in pack.paths).to(core.device).eval().requires_grad_(False)
            self.policy.load_state_dict(saved['model'])
            for key,value in core.goal.state_dict().items():
                if not torch.equal(value,self.policy.goal_pipeline.state_dict()[key]):
                    raise ValueError('Policy contains a different frozen goal encoder/matcher')
        profile=json.loads(pack.paths['safety'].read_text())
        if (profile.get('schema')!='measured-navigation-safety/v1' or
                profile.get('maximum_speed_mps')!=maximum_speed or
                not isinstance(profile.get('measurement_sha256'),str) or len(profile['measurement_sha256'])!=64):
            raise ValueError('Measured braking/safety profile for this speed is required')
        self.radius=float(profile['vehicle_radius_m'])
        if not math.isfinite(self.radius) or self.radius<=0:raise ValueError('Invalid vehicle radius')
        self.radius=max(1.,self.radius)  # Campaign 0.75 m vehicle + 0.25 m margin.
        self.safety=HardSafetyFilter(maximum_speed,float(profile['braking_acceleration_mps2']))
        self.hidden=torch.zeros(1,256,device=core.device)
        self.core=core;self.maximum_speed=maximum_speed;self.stochastic=stochastic;self.last_ns=None
        self.demonstrate=demonstrate

    @torch.inference_mode()
    def propose(self,metadata,value,planned_command=None):
        if self.last_ns is not None and metadata['sim_ns']-self.last_ns>250000000:self.hidden.zero_()
        self.last_ns=metadata['sim_ns'];initial_hidden=self.hidden.clone()
        teacher_command,teacher_stop,teacher_reason=demonstration(value)
        if self.demonstrate:
            proposal=value['state'].new_tensor(teacher_command)
            sampled_stop=proposal.new_tensor([teacher_stop])
            latent=logprob=reward_value=collision_value=None
        else:
            mask=value['memory_valid'][:,None]
            context=(value['memory'][:,:256]*mask).sum(0)/mask.sum().clamp_min(1)
            distribution,reward_value,self.hidden=self.policy.forward_features(value['current_tokens'],value['goal_context'],
                value['state'][None,:self.policy_state_dim],context[None],value['task'][None],value['previous_command'][None],
                self.hidden,value['target_context'][None],value['depth_tokens'][None])
            stop_distribution=self.policy.stop_distribution(self.hidden)
            latent=distribution.sample() if self.stochastic else distribution.mean
            sampled_stop=stop_distribution.sample() if self.stochastic else (stop_distribution.probs>=.5).float()
            logprob=distribution.log_prob(latent).sum(-1)+stop_distribution.log_prob(sampled_stop)
            proposal=self.policy.command(latent,self.maximum_speed)[0]
            collision_value=float(self.policy.collision_value(self.hidden)[0,0])
        command=Command(*(planned_command if planned_command is not None else proposal.tolist()))
        speed=float(value['state'][3:6].norm())
        if self.core.metric_vision is not None and not value['input_validity']['scale']:
            speed=max(speed,.5) # bounded exploration envelope, not a measured velocity
        geometry_ready=value['metric_geometry_available']
        clearance=0.;unknown=1.
        if geometry_ready and self.core.metric_vision is None:
            distance,missing=self.core.geometry(self.core.position[None])
            unknown=float(missing[0]);distance=float(distance[0])
            # Inflate for metric pose/scale uncertainty and aircraft extent.
            clearance=max(0.,distance-self.radius-2*float(value['state'][16])-
                          abs(distance)*math.expm1(min(5.,2*abs(float(value['state'][13])))))
        now=time.monotonic();age=max(0.,now-metadata['received_monotonic'])
        # Mode 1 has no learned planner-risk result. Unknown space has risk one;
        # observed clearance is handled conservatively by braking distance.
        if age>.25:
            filtered=Command(0,0,0,0);safety=dict(overridden=True,reason='stale_rgb',stopping_distance_m=None)
        elif self.core.metric_vision is not None and self.core.metric_vision.usable(metadata['sim_ns']) and value['map_status'] not in ('recovering','terminated'):
            # The first integration stays at exploration speed, including
            # locally navigable and mapped states, until navigation qualifies.
            command,_=self.core.startup.filter(command,age)
            geometry_age=value['geometry_age_seconds'] or 0.
            clearance,unknown=self.core.metric_vision.sweep(self.core,command,speed,self.safety.braking_acceleration,geometry_age+age)
            if unknown or clearance<=self.radius+.5*geometry_age:
                filtered=Command(0,0,0,command.yaw_dps)
                safety=dict(overridden=True,reason='local_path_unknown' if unknown else 'local_path_obstacle',stopping_distance_m=None)
            else:
                filtered=command;safety=dict(overridden=False,reason=None,stopping_distance_m=None)
        elif self.core.metric_vision is not None and value['map_status'] in ('local_navigation','mapped'):
            filtered=Command(0,0,0,0);safety=dict(overridden=True,reason='stale_local_geometry',stopping_distance_m=None)
        elif value['map_status'] in ('initializing','recovering','terminated'):
            filtered,reason=self.core.startup.filter(command,age)
            safety=dict(overridden=filtered!=command,reason=reason,stopping_distance_m=None)
        else:
            filtered,safety=self.safety.apply(command,age,
                not value['input_validity']['tracking'] or not geometry_ready,float(value['state'][13]),
                1. if unknown>0 else 0.,clearance,speed,age)
        stop=value['map_status'] in ('mapped','local_navigation') and bool(sampled_stop.item()) and value['goal_probability']>=self.core.teacher.threshold and not safety['overridden']
        if bool(sampled_stop.item()) and value['map_status'] in ('mapped','local_navigation'):
            filtered=Command(0,0,0,0)
            if not stop:safety=dict(safety,overridden=True,reason=safety['reason'] or 'stop_not_visually_supported')
        trace=dict(action_semantics=ACTION_SEMANTICS,startup_contract=VERSION,map_status=value['map_status'],
            initialization_elapsed_seconds=value['initialization_elapsed_seconds'],
            termination_reason=value['termination_reason'],
            teacher_reason=teacher_reason if self.demonstrate else None,
            teacher_provenance='observed-depth-motion-demonstrations/v6' if self.demonstrate else None,
            episode_id=self.core.episode_id,frame_id=metadata['frame_id'],sim_ns=metadata['sim_ns'],
            behavior_policy_sha256=None if self.demonstrate else self.core.checkpoints.spec['artifacts']['policy']['sha256'],
            stochastic=self.stochastic,latent_action=None if latent is None else latent[0].tolist(),sampled_stop=bool(sampled_stop.item()),
            proposal=proposal.tolist(),proposal_log_probability=None if logprob is None else float(logprob[0]),
            reward_value=None if reward_value is None else float(reward_value[0]),collision_cost_value=collision_value,
            submitted_after_safety=list(asdict(filtered).values()),submitted_stop=stop,safety=safety,
            local_geometry_age_seconds=value.get('geometry_age_seconds'),
            depth_valid_fraction=float(value['depth_tokens'][:,1].mean()),
            global_metric_pose_valid=value['input_validity']['scale'],
            source_available_monotonic=metadata['received_monotonic'],decision_monotonic=time.monotonic(),
            safety_vehicle_radius_m=self.radius,
            mode='deterministic_observation_teacher' if self.demonstrate else 'predictive_then_independent_safety' if planned_command is not None else 'mode_1_geometry_safety_planner_disabled')
        return filtered,stop,trace,initial_hidden[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode-id',required=True)
    parser.add_argument('--socket',type=Path,default=Path('/ipc/rgb.sock'))
    parser.add_argument('--checkpoints',type=Path,default=Path('/navigation/checkpoints.json'))
    parser.add_argument('--maximum-speed-mps',type=float,choices=(3.,4.5,6.),default=3.)
    parser.add_argument('--sample-policy',action='store_true');parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--with-deliberation',action='store_true')
    parser.add_argument('--demonstrate',action='store_true')
    args=parser.parse_args();output=Path('/output');client=BrokerClient(args.socket,args.episode_id)
    if args.with_deliberation and (args.sample_policy or args.demonstrate):parser.error('PPO and deterministic demonstrations disable slow planning')
    core=video=mapping=deliberation=None;count=0;error=None;chunks=[];shards=[];latencies=[];interventions=0;last_grounding_ns=-1
    def flush():
        if not chunks:return
        path=output/f'controller-{len(shards):06d}.pt';torch.save(dict(samples=list(chunks)),path)
        shards.append(dict(path=path.name,sha256=checksum(path),samples=len(chunks)));chunks.clear()
    try:
        pack=CheckpointSet(args.checkpoints,integration_only=args.integration_only);views=[];identities=[]
        for index in range(4):
            metadata,rgb=client.goal_view(index);identities.append(metadata['panorama_sha256'])
            views.append(np.frombuffer(rgb,np.uint8).reshape(480,640,3).copy())
        if len(set(identities))!=1:raise ValueError('Inconsistent goal panorama')
        if hashlib.sha256(b''.join(view.tobytes() for view in views)).hexdigest()!=identities[0]:
            raise ValueError('Goal pixels differ from the broker panorama identity')
        capacity=reserve_fast();torch.set_num_threads(2)
        (output/"capacity.json").write_text(json.dumps(capacity))
        torch.manual_seed(0);np.random.seed(0)
        core=CausalNavigationState(pack,args.episode_id,torch.from_numpy(np.stack(views)).permute(0,3,1,2))
        actor=Mode1Actor(core,args.maximum_speed_mps,args.sample_policy,args.demonstrate)
        torch.save(dict(tokens=core.goal_tokens[0].cpu(),goal_sha256=identities[0]),output/'goal-tokens.pt')
        video=AsyncVideoFeatures(args.episode_id,output/'video')
        mapping=AsyncMapping(args.episode_id,output/'reconstruction',args.socket,pack.paths.get('vision'))
        if args.with_deliberation:
            from live_deliberation import LiveDeliberation
            deliberation=LiveDeliberation(core,views,output/'deliberation')
        # Separate the fast GPU queue from Qwen/planner work. CPU affinity
        # alone cannot prevent default-stream ordering from stalling control.
        torch.cuda.current_stream().synchronize()
        fast_stream=torch.cuda.Stream(priority=-1)
        torch.cuda.set_stream(fast_stream)
        capacity['fast_cuda_stream_priority']=fast_stream.priority
        (output/'capacity.json').write_text(json.dumps(capacity))
        # Loading the large models can outlive the broker's idle socket
        # timeout. Goal retrieval has completed; open fresh channels for
        # control rather than retrying an ambiguously dispatched command.
        client.close()
        (output/'CONTROLLER_READY').touch();last=-1
        while not (output/'CONTROLLER_STOP').exists() and not (output/'CHECKPOINT_REQUEST').exists():
            metadata,rgb=client.observe(last);last=metadata['frame_id'];started=time.monotonic()
            metadata=dict(metadata,rgb_sha256=hashlib.sha256(rgb).hexdigest())
            profile={}
            mapping.poll(core,metadata);features=video.poll()
            if features is not None:core.enqueue('video',features)
            video.observe(metadata,rgb);profile['slow_publication']=time.monotonic()-started
            phase=time.monotonic();value=core.observe(metadata,rgb);profile['perception_belief']=time.monotonic()-phase
            phase=time.monotonic()
            planned=deliberation.update(metadata,rgb,value) if deliberation else None
            profile['deliberation_poll_submit']=time.monotonic()-phase;phase=time.monotonic()
            command,stop,trace,hidden=actor.propose(metadata,value,planned)
            profile['policy_safety']=time.monotonic()-phase;phase=time.monotonic()
            try:trace['broker_acceptance']=client.command(last,list(asdict(command).values()),stop=stop)
            except (ValueError,RuntimeError) as failure:
                if str(failure) not in ('Stale RGB; braking','Stale or future command frame'):raise
                trace['broker_acceptance']=dict(accepted=False,reason=str(failure))
            profile['broker_command']=time.monotonic()-phase;trace['profile_seconds']=profile
            trace['processing_seconds']=time.monotonic()-started;trace['deadline_missed']=trace['processing_seconds']>.05
            latencies.append(trace['processing_seconds']);interventions+=int(trace['safety']['overridden'])
            with (output/'proposals.jsonl').open('a') as stream:stream.write(json.dumps(trace)+'\n')
            runtime={k:value[k].detach().cpu() for k in ('state','memory','memory_valid','task','previous_command','target_context','z')}
            runtime.update(current_tokens=value['current_tokens'][0].cpu(),goal_context=value['goal_context'][0].cpu(),depth_tokens=value['depth_tokens'].cpu())
            teacher_command,teacher_stop,teacher_reason=demonstration(value)
            correction=dict(episode_id=args.episode_id,frame_id=last,sim_ns=metadata['sim_ns'],
                expert_observation_conditioned=True,expert_command=teacher_command,explicit_stop=teacher_stop,
                teacher='observed-depth-motion-demonstrations/v6',reason=teacher_reason,
                behavior_policy_sha256=trace['behavior_policy_sha256'])
            if args.demonstrate:
                correction.update(expert_command=list(asdict(command).values()),explicit_stop=stop,
                    reason=safety_reason if (safety_reason:=trace['safety'].get('reason')) and trace['safety']['overridden'] else teacher_reason)
            with (output/'dagger-corrections.jsonl').open('a') as stream:stream.write(json.dumps(correction)+'\n')
            if value['termination_reason']:
                (output/'termination.json').write_text(json.dumps(dict(reason=value['termination_reason'])))
            grounding=None
            if last_grounding_ns<0 or metadata['sim_ns']-last_grounding_ns>=3000000000:
                grounding=core.grounding_context(metadata['sim_ns'],value['config']);last_grounding_ns=metadata['sim_ns']
            chunks.append(dict(frame_id=last,sim_ns=metadata['sim_ns'],runtime=runtime,initial_hidden=hidden.cpu(),proposal=trace,
                config=asdict(value['config']) if value['config'] else None,configuration_evidence=grounding,
                map_status=value['map_status'],initialization_elapsed_seconds=value['initialization_elapsed_seconds'],
                input_validity=value['input_validity'],goal_probability=value['goal_probability'],
                goal_match_threshold=value['goal_match_threshold'],target_available=value['target_available'],
                metric_geometry_available=value['metric_geometry_available'],visual_available=value['visual_available'],
                latest_observation_ns=value['latest_observation_ns']))
            count+=1
            if len(chunks)>=64:flush()
            if core.diagnostics:
                with (output/'alignment.jsonl').open('a') as stream:
                    for row in core.diagnostics:stream.write(json.dumps(row)+'\n')
                core.diagnostics.clear()
    except Exception as failure:
        import traceback
        expected_shutdown=(output/'CONTROLLER_STOP').exists() and (
            isinstance(failure,(EOFError,ConnectionError)) or
            isinstance(failure,RuntimeError) and str(failure) in ('RGB unavailable','Broker unavailable'))
        if not expected_shutdown:
            error=dict(type=type(failure).__name__,message=str(failure));(output/'failure.log').write_text(traceback.format_exc())
    finally:
        flush();client.close()
        for worker in (deliberation,video,mapping,core):
            if worker is not None:
                try:worker.close()
                except Exception as failure:
                    error=error or dict(type=type(failure).__name__,message=str(failure))
        result=dict(status='failed' if error else 'completed',accepted=False,frames=count,error=error,shards=shards,
                    scope='Mode 1, Qwen and frozen predictive planning' if deliberation else 'Live Mode 1 only; no planner or Qwen configuration',
                    sampled_policy=args.sample_policy,slow_planner_enabled=args.with_deliberation,
                    teacher_provenance='observed-depth-motion-demonstrations/v6' if args.demonstrate else None,
                    final_map_status=core.startup.state if core else None,
                    map_version=core.latest_map_version if core else None,
                    map_fusion=core.map_fusion.receipt() if core else None,
                    supported_maps_received=core.supported_maps_received if core else 0,
                    optimized_surface_samples_received=core.optimized_surface_samples_received if core else 0,
                    first_supported_map_available_ns=core.first_supported_map_available_ns if core else None,
                    handover_sim_ns=core.startup.handover_ns if core else None,
                    metric_handover_sim_ns=core.startup.metric_handover_ns if core else None,
                    control_safety_p95_seconds=float(np.quantile(latencies,.95)) if latencies else None,
                    control_safety_missed_deadlines=sum(x>.05 for x in latencies),safety_interventions=interventions)
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return 2 if error else 0


if __name__=='__main__':raise SystemExit(main())
