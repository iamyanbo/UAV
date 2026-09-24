"""Asynchronous Qwen configuration and frozen-model action optimization."""
from collections import deque,Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import json
import queue
from pathlib import Path
import time

from PIL import Image
import torch
from configurator import Configurator
from runtime_capacity import slow_worker
from contracts import VisualTaskConfig
from learning_models import WorldModel,PrimitiveCritic
from action_intervals import executed_slots
from predictive_planner import CostScales,PredictivePlanner


class LiveDeliberation:
    def __init__(self,core,goal_views,output):
        self.core=core;self.output=Path(output);self.output.mkdir()
        pack=core.checkpoints
        if not {'world','policy','qwen'}<=pack.paths.keys():raise ValueError('Deliberation requires trained world, critic and Qwen adapter')
        world=WorldModel().cuda();saved=torch.load(pack.paths['world'],map_location='cpu',weights_only=True)
        world.load_state_dict(saved['model'])
        policy=torch.load(pack.paths['policy'],map_location='cpu',weights_only=True)
        if policy.get('world_checkpoint_sha256')!=pack.spec['artifacts']['world']['sha256']:
            raise ValueError('Terminal critic world provenance differs')
        critic=PrimitiveCritic().cuda();critic.load_state_dict(policy['critic'])
        safety=json.loads(pack.paths['safety'].read_text())
        scales=CostScales(tuple(critic.primitive_scale.tolist()),(1.,1.,1.,1.,1.,1.,1.,1.),
            max(1.,safety['vehicle_radius_m']),safety['braking_acceleration_mps2'],1.)
        self.planner=PredictivePlanner(world,critic,scales,3.)
        self.qwen=Configurator(adapter=pack.paths['qwen'])
        self.qwen.model.requires_grad_(False)
        self.qwen.enable_goal_cache()
        self.goals=[Image.fromarray(view) for view in goal_views]
        self.override=json.loads(pack.paths['configuration'].read_text()) if 'configuration' in pack.paths else None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='planner',initializer=slow_worker)
        self.qwen_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='qwen',initializer=slow_worker)
        self.qwen_pending=None;self.last_qwen_ns=-1
        self.pending=None;self.active=None;self.version=0;self.last_ns=-1;self.history=deque(maxlen=10)
        self.last_history_ns=-1;self.completed=0;self.errors=[]
        self.configuration_results=queue.SimpleQueue()
        self.rejections=Counter();self.selected_actions=0;self.published_configurations=0
        self.configuration_rejections=Counter();self.qwen_calls=0
        self.log=(self.output/'events.jsonl').open('x')

    def _event(self,event):
        self.log.write(json.dumps(event)+'\n');self.log.flush()

    def _work(self,request):
        torch.cuda.current_stream().wait_event(request['inputs_ready'])
        started=time.monotonic();value=request['value'];ns=value['sim_ns']
        # Clone outside inference_mode so candidate gradients can pass through
        # frozen world/critic computations without inference-tensor errors.
        tensor=lambda x:x.detach().clone()
        config=request['config'];qwen_finished=started
        hidden=torch.zeros(1,256,device='cuda')
        with torch.no_grad():
            for historical in request['history'][:-1]:
                slots=historical.get('dispatched_slots')
                if slots is None:
                    hidden.zero_()
                    continue
                action=torch.tensor(slots,device='cuda',dtype=torch.float32)[None]
                prediction=self.planner.model(tensor(historical['z'])[None],tensor(historical['state'])[None],hidden,
                    tensor(historical['memory'])[None],tensor(historical['memory_valid'])[None],action,
                    tensor(historical['task'])[None],tensor(self.core.goal_tokens),tensor(historical['target_context'])[None])
                hidden=prediction['belief'].mean(0)
        initial={name:tensor(value[name])[None] for name in ('z','state','memory','memory_valid','previous_command','target_context')}
        initial.update(belief=hidden.detach(),goal_tokens=tensor(self.core.goal_tokens))
        multipliers=torch.tensor([config.goal_weight,config.time_weight,config.information_weight,config.additional_caution]
            if config else [1.,1.,1.,1.],device='cuda')
        planned=self.planner.optimize(initial,tensor(value['task'])[None],request['geometry'],multipliers,
            tensor(value['previous_command'])[None].expand(20,4),delay_seconds=max(0.,time.monotonic()-request['available']))
        result=dict(episode_id=self.core.episode_id,version=request['version'],observation_ns=ns,
            memory_version=value['memory_version'],gauge_version=value['gauge_version'],hazard_version=value['hazard_version'],
            map_correction_version=value['map_correction_version'],
            available_monotonic=time.monotonic(),processing_seconds=time.monotonic()-started,
            configuration=None,configuration_error=None,qwen_available_monotonic=None,
            qwen_processing_seconds=qwen_finished-started,
            planning_seconds=time.monotonic()-qwen_finished,
            observed=request['observed'],source_frame_id=value['frame_id'],
            applied_configuration=asdict(config) if config else None,
            initial_state=value['state'].cpu().tolist(),
            belief_action_semantics='post-safety-dispatch/50ms-v3',
            **{k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in planned.items()})
        path=self.output/f'plan-{request["version"]:06d}.pt';torch.save(result,path)
        return result

    def _configure(self,request):
        value=request['value'];ns=value['sim_ns']
        try:
            keyframes=[Image.frombytes('RGB',(640,480),rgb) for rgb in request['keyframe_rgb']]
            frontiers=[Image.frombytes('RGB',(640,480),rgb) for rgb in request['frontier_rgb']]
            configured=self.qwen.configure(self.goals,request['image'],keyframes,frontiers,request['observed'],
                dict(goal_probability=value['goal_probability']),self.core.episode_id,ns/1e9)
        finally:
            (self.output/f"qwen-{ns}.json").write_text(json.dumps(dict(observation_ns=ns,
                response=self.qwen.last_response,goal_cache_hits=self.qwen.goal_cache_hits,
                goal_cache_misses=self.qwen.goal_cache_misses,grounding_evidence=request['grounding_evidence'])))
        if configured is not None:
            self.configuration_results.put(dict(episode_id=self.core.episode_id,version=request['version'],
                observation_ns=ns,available_monotonic=time.monotonic(),gauge_version=value['gauge_version'],
                configuration=asdict(configured)))

    def update(self,metadata,rgb,value):
        ns=value['sim_ns'];now=time.monotonic()
        observed=[dict(id=identifier,kind=entry['kind'],observed_ns=entry['observed_ns'])
            for identifier,entry in self.core.memory.observed_ids.items() if entry['observed_ns']<=ns]
        while not self.configuration_results.empty():
            delivered=self.configuration_results.get_nowait()
            if self.override:continue
            try:
                if delivered['gauge_version']!=value['gauge_version']:raise ValueError('Map gauge changed')
                VisualTaskConfig(**delivered['configuration']).validate_grounding(
                    self.core.episode_id,[x['id'] for x in observed],ns/1e9)
                self.core.enqueue('configuration',delivered)
                self.published_configurations+=1
                self._event(dict(component='configuration',status='published',
                    version=delivered['version'],observation_ns=delivered['observation_ns']))
            except ValueError:
                self.configuration_rejections['stale_or_ungrounded']+=1
                self._event(dict(component='configuration',status='rejected_stale_or_ungrounded'))
        if self.qwen_pending is not None and self.qwen_pending.done():
            try:self.qwen_pending.result()
            except Exception as error:
                self.configuration_rejections['invalid_output_or_execution_error']+=1
                self._event(dict(component='configuration',status='failed',reason=str(error)))
            self.qwen_pending=None
        if not self.override and self.qwen_pending is None and ns-self.last_qwen_ns>=3000000000:
            self.last_qwen_ns=ns
            self.qwen_calls+=1
            evidence=self.core.grounding_context(ns,value['config'])
            self.qwen_pending=self.qwen_executor.submit(self._configure,dict(value=value,
                image=Image.frombytes('RGB',(640,480),rgb),observed=evidence['observed'],version=ns,
                grounding_evidence=evidence,
                keyframe_rgb=[self.core.keyframe_rgb[ref['frame_id']]['rgb'] for ref in evidence['keyframes']],
                frontier_rgb=[self.core.keyframe_rgb[ref['frame_id']]['rgb'] for ref in evidence['frontiers']]))
        config=value['config']
        if self.override:
            parsed=dict(self.override);horizon=parsed.pop('validity_horizon_seconds')
            if not 3<=horizon<=5:raise ValueError('Matched configuration validity out of bounds')
            if parsed['target_id'] is None and observed:
                selected=self.core.teacher.configure(self.core.memory,self.core.position,ns,value['goal_probability'])
                if selected is None:raise ValueError('Observed configuration has no causally selectable target')
                parsed.update(target_id=selected.target_id,grounded_kind=selected.grounded_kind,
                    intention=selected.intention,confidence=selected.confidence)
            config=VisualTaskConfig(episode_id=self.core.episode_id,valid_until_sim_seconds=ns/1e9+horizon,**parsed)
            config.validate_grounding(self.core.episode_id,[x['id'] for x in observed],ns/1e9)
            self.core.configuration=config
        for historical in self.history:
            if historical['dispatched_slots'] is None:
                slots=executed_slots(metadata['command_history'],historical['sim_ns'])
                if all(slots['valid']):historical['dispatched_slots']=slots['values']
        if ns-self.last_history_ns>=200000000:
            self.history.append(dict(value,dispatched_slots=None));self.last_history_ns=ns
        if self.pending is not None and self.pending.done():
            try:
                self.active=self.pending.result();self.completed+=1
                event={k:v for k,v in self.active.items() if not torch.is_tensor(v)}
                event['candidate_cost_improvement']=float((self.active['candidate_initial_costs']-self.active['candidate_final_costs']).max())
                self._event(event)
            except Exception as error:
                self.errors.append(type(error).__name__+': '+str(error));self._event(dict(status='failed',error=self.errors[-1]))
            self.pending=None
        if not self.errors and self.pending is None and value['map_status'] in ('initializing','mapped') and value['visual_available'] and len(self.history)>=10 and ns-self.last_ns>=3000000000:
            self.version+=1;self.last_ns=ns
            image=Image.frombytes('RGB',(640,480),rgb)
            request=dict(value=value,observed=observed,image=image,config=config,version=self.version,
                geometry=copy.deepcopy(self.core.geometry),history=[dict(row) for row in self.history],available=metadata['received_monotonic'])
            request['inputs_ready']=torch.cuda.Event()
            request['inputs_ready'].record(torch.cuda.current_stream())
            self.pending=self.executor.submit(self._work,request)
        active=self.active
        if active is None:return None
        elapsed=(ns-active['observation_ns'])/1e9;index=int(elapsed/.2)
        reasons=[]
        if value['map_status']!='mapped':reasons.append('startup_or_recovery_mode1_active')
        if active['episode_id']!=self.core.episode_id:reasons.append('wrong_episode')
        if active['hazard_version']!=value['hazard_version']:reasons.append('new_observed_hazard')
        if active['gauge_version']!=value['gauge_version']:reasons.append('map_gauge_changed')
        if active['map_correction_version']!=value['map_correction_version']:reasons.append('metric_map_correction')
        applied=active.get('applied_configuration') or {};current=asdict(config) if config else {}
        if any(applied.get(key)!=current.get(key) for key in
               ('target_id','grounded_kind','intention','goal_weight','time_weight','information_weight','additional_caution')):
            reasons.append('configuration_changed')
        if applied.get('target_id') is not None and applied['target_id'] not in self.core.memory.observed_ids:reasons.append('target_removed')
        if now-active['available_monotonic']>1:reasons.append('publication_stale')
        if not 0<=index<20:reasons.append('prediction_horizon_expired')
        if float(active['collision_probabilities'].max())>.05:reasons.append('predicted_collision_risk')
        if reasons:
            self.rejections.update(reasons);return None
        expected=active['predicted_states'][max(0,index-1)]
        if float((value['state'][:3].cpu()-expected[:3]).norm())>1:self.rejections['position_mismatch']+=1;return None
        if float((value['state'][3:6].cpu()-expected[3:6]).norm())>1:self.rejections['velocity_mismatch']+=1;return None
        if float((value['state'][6:12].cpu()-expected[6:12]).norm())>.3:self.rejections['rotation_mismatch']+=1;return None
        self.selected_actions+=1
        return active['commands'][index].tolist()

    def close(self):
        self.qwen_executor.shutdown(wait=True,cancel_futures=False)
        self.executor.shutdown(wait=True,cancel_futures=False)
        if self.pending:
            try:
                result=self.pending.result();self.completed+=1
                self.rejections['episode_ended_before_publication']+=1
                self._event(dict(component='planner',status='discarded_episode_ended',
                    processing_seconds=result['processing_seconds'],version=result['version']))
            except Exception as error:self.errors.append(type(error).__name__+': '+str(error))
        if self.qwen_pending:
            try:self.qwen_pending.result()
            except Exception as error:
                self.configuration_rejections['invalid_output_or_execution_error']+=1
                self._event(dict(component='configuration',status='failed',reason=str(error)))
        while not self.configuration_results.empty():
            self.configuration_results.get_nowait()
            self.configuration_rejections['episode_ended_before_publication']+=1
        self.log.close()
        result=dict(status='failed' if self.errors else 'completed',accepted=False,
            actual_planner_calls=self.completed,errors=self.errors,
            selected_planner_actions=self.selected_actions,published_qwen_configurations=self.published_configurations,
            actual_qwen_calls=self.qwen_calls,configuration_rejections=dict(self.configuration_rejections),
            goal_vision_cache_hits=self.qwen.goal_cache_hits,goal_vision_cache_misses=self.qwen.goal_cache_misses,
            plan_rejections=dict(self.rejections),
            scope='Real Qwen and 12-step candidate optimization; unqualified cost scales and command-time estimates')
        (self.output/'result.json').write_text(json.dumps(result,indent=2));return result
