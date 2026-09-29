"""Fixed-batch, proposal-conditioned PPO. Simulation-only; all admission is explicit.

Run `--preflight` to inspect readiness without loading a model or simulator.
There is no fallback to corridor tasks, geometric control, or an untrained world
selector. The historical ppo_pilot entry point remains an engineering baseline.
"""
import argparse
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
import math
import platform
from pathlib import Path
import random
import shutil
import threading
import time
import uuid
import numpy as np
import torch
from PIL import Image
from .common import read,write,digest,FlightLock,Window,seed_all
from .compute import ComputeLane
from .ppo_admission import quality_gate,workload_gate
from .ppo_budget import Budget
from .ppo_core import ActorCritic,command_from_latent,transition_reward,update,potential
from .ppo_guidance import RegionProposer,Guidance
from .ppo_pilot import Worker
from .ppo_scheduler import FeatureBank,Inference,batch_for
from .ppo_tasks import ObservedVolume,validate_tasks
from .ppo_geometry import CostField


def identity_for(args):
    root=Path(__file__).parent
    files=list(root.glob('*.py'))+[root.parent/'rgb_flight/goal_matching.py']
    runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=platform.python_version(),machine=platform.machine())
    if torch.cuda.is_available():
        gpu=torch.cuda.get_device_properties(0);runtime.update(gpu=gpu.name,capability=[gpu.major,gpu.minor],device_memory=gpu.total_memory)
    return dict(config=digest(args.config),manifest=digest(args.manifest),backbone=digest(args.backbone),runtime=runtime,
        source={str(p.relative_to(root.parent)):digest(p) for p in sorted(files)},
        migration=digest(args.migrate) if args.migrate else None)


def preflight(args):
    cfg=read(args.config);manifest=read(args.manifest)
    if cfg['schema']!='photo-map-ppo/v2' or cfg['rollout_steps']!=8192 or cfg['minibatch']!=512:
        raise ValueError('Expected the fixed 8192/512 overnight configuration')
    if cfg.get('endpoint_pilot'):
        from .ppo_endpoint_prepare import validate_endpoint_manifest
        if not args.endpoint_pilot:raise ValueError('Endpoint pilot requires explicit --endpoint-pilot')
        if args.benchmark:raise ValueError('Endpoint pilot is not two-mode workload admission')
        if len(args.worker)!=1 or len(args.authfile)!=1:raise ValueError('Endpoint pilot starts with one measured worker')
        scenes=validate_endpoint_manifest(manifest,cfg,args.config)
        return cfg,manifest,scenes,identity_for(args),dict(passed=False,mode2_enabled=False,scope='endpoint-mode1-pilot')
    if args.endpoint_pilot:raise ValueError('Endpoint flag requires its dedicated configuration')
    if not all((args.audit,args.grades,args.evaluation_worker,args.evaluation_authfile)):
        raise ValueError('Two-mode experiment requires audit, grades and isolated evaluation endpoint')
    if len(args.worker) not in (1,2) or len(args.worker)!=len(args.authfile):raise ValueError('One or two isolated workers required')
    if len(set(args.worker+[args.evaluation_worker]))!=len(args.worker)+1:raise ValueError('Evaluation needs a separate worker endpoint')
    scenes=validate_tasks(manifest,cfg);identity=identity_for(args)
    if manifest['config_sha256']!=identity['config']:raise ValueError('Task configuration changed')
    quality=quality_gate(args.audit,args.grades)
    if not quality['passed']:raise ValueError('mode2_not_admitted: Qwen quality audit failed')
    if not args.benchmark:
        if not args.admission:raise ValueError('Concurrent workload admission required')
        admitted=workload_gate(args.admission,identity,len(args.worker))
        if admitted.get('model_identity')!=quality['model_identity']:raise ValueError('Audit and workload use different Qwen weights')
    sanity=[t for t in manifest['tasks'] if t.get('sanity_validation')]
    if len(sanity)!=5 or {t['difficulty'] for t in sanity}!={'direct','detour','multiple_decisions','altitude_alternatives'}:
        raise ValueError('Five fixed sanity missions must cover all four difficulty classes')
    if any(t['split']!='validation' or t['distance_bin']!='40-100' for t in sanity):raise ValueError('Invalid sanity validation pool')
    return cfg,manifest,scenes,identity,quality


def migrate(model,path,backbone,prior):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('schema')!='photo-map-ppo/v1' or saved['identity']['backbone']!=digest(backbone):raise ValueError('Migration requires matching historical PPO backbone')
    old=saved['counts']
    if prior['ppo_transitions']<old['transitions'] or prior['learner_attempts']<old['attempts']:raise ValueError('Migration would erase campaign usage')
    target=model.state_dict();copied=[]
    for name,value in saved['model'].items():
        if name.startswith(('encoder.','matcher.','actor.')) or name=='log_std':
            if name not in target or value.shape!=target[name].shape:raise ValueError('Incompatible migrated tensor: '+name)
            target[name]=value;copied.append(name)
    model.load_state_dict(target)
    with torch.no_grad():
        torch.nn.init.orthogonal_(model.actor.action[-1].weight[4:5],gain=.01)
        model.actor.action[-1].bias[4]=math.log(.0005/.9995)
    return dict(source=str(Path(path).resolve()),sha256=digest(path),copied=copied,
        old_counts=old,old_transitions_reused=0,reset=['optimizer','stop_output_row','value','execution_value','subgoal_encoder'])


class Curriculum:
    def __init__(self,tasks,cfg,counts=None):
        self.tasks=tasks;self.cfg=cfg;self.counts=Counter(counts or {});self.indices=Counter();self.lock=threading.Lock()

    def choose(self,scene):
        with self.lock:
            total=sum(self.counts.values());mix=({'execution':.5,'arrival':.5} if total<self.cfg['auxiliary_bootstrap_transitions'] else self.cfg['mixture'])
            available={kind:[t for t in self.tasks if t['scene_id']==scene and t.get('kind','mission')==kind] for kind in mix}
            if any(not rows for rows in available.values()):raise ValueError('Scene lacks required curriculum stream')
            kind=max(mix,key=lambda k:mix[k]*(total+1)-self.counts[k])
            # Round robin through measured difficulty and distance within a stream.
            rows=sorted(available[kind],key=lambda t:(t.get('difficulty',''),t.get('distance_bin',''),t['id']))
            buckets={}
            for t in rows:buckets.setdefault((t.get('difficulty'),t.get('distance_bin')),[]).append(t)
            cells=sorted(buckets,key=str);i=self.indices[(scene,kind)];cell=cells[i%len(cells)]
            row=buckets[cell][(i//len(cells))%len(buckets[cell])];self.indices[(scene,kind)]+=1
            return row

    def add(self,kind):
        with self.lock:self.counts[kind]+=1


class Collector:
    def __init__(self,index,worker,scene,scheduler,bank,budget,curriculum,cfg,root,window,guidance):
        self.index=index;self.worker=worker;self.scene=scene;self.scheduler=scheduler;self.bank=bank;self.budget=budget
        self.curriculum=curriculum;self.cfg=cfg;self.root=root;self.window=window;self.guidance=guidance
        self.task=None;self.obs=None;self.decision=None;self.paused=False;self.fields={};self.costs={};self.episodes=[];self.coverage=deque(maxlen=20)

    def select(self,phase='training'):
        key=(self.scene['scene_id'],phase)
        if self.worker.scene==key:return
        reply=self.worker.call('select',scene=read(self.scene['descriptor']),session=f'{phase}-{self.index}-{uuid.uuid4().hex}',phase=phase)
        if reply.get('config')!=self.cfg or reply.get('source_sha256')!=digest(Path(__file__).with_name('ppo_env.py')):
            raise ValueError('Host worker source/config differs from trainer')
        for owner,port in self.window.sim_ports.items():
            if owner!=self.index and port==reply['sim_port']:raise ValueError('Simulator workers share a native RPC port')
        self.window.sim_ports[self.index]=reply['sim_port']
        self.worker.scene=key

    def begin(self,task=None,evaluate=False):
        self.task=task or self.curriculum.choose(self.scene['scene_id']);self.attempt=('eval-' if evaluate else 'train-')+uuid.uuid4().hex
        if not evaluate:self.budget.attempt(self.attempt,self.index)
        self.guidance.forget(self.index)
        path=self.task['field']
        if not self.cfg.get('endpoint_pilot') and path not in self.fields:self.fields[path]=ObservedVolume.load(path)
        # Load evaluator assets before releasing reset control, never while a
        # freshly captured command source is aging.
        if not self.cfg.get('endpoint_pilot'):self.costs=CostField.load(self.task['cost_field'])
        with Image.open(self.task['goal_image']) as im:image=im.convert('RGB').copy()
        self.scheduler.call('initialize',worker=self.index,image=image,path=self.task['goal_image'],
            exercise=self.task.get('exercise'),task_kind=self.task.get('kind','mission'))
        self.obs=self.worker.call('reset',task=self.task,attempt_id=self.attempt)
        self.decision=self.infer(self.obs);self.episode_reward=0.;self.episode_steps=0;self.stop_attempts=0;self.goal_visits=0

    def infer(self,obs,replace=False):
        visible={k:obs[k] for k in ('rgb','rgb_path','sim_s','preceding_command')}
        return self.scheduler.call('observation',worker=self.index,obs=visible,execution=self.task.get('kind')=='execution',replace=replace)

    def phi(self,obs):
        if self.cfg.get('endpoint_pilot'):
            return potential(obs['state']['position'],self.task['goal'],math.dist(self.task['start'],self.task['goal']))
        return self.fields[self.task['field']].potential(obs['state']['position'],self.costs,self.task['reference_s'])

    def step(self,freeze=False,evaluate=False,batch=None,iteration=0):
        obs=self.obs;decision=self.decision;context=decision['context']
        command=command_from_latent(decision['latent'],obs['preceding_command'],self.cfg['step_s'],self.cfg['limits'],self.cfg['acceleration'])
        phi=self.phi(obs)
        if not evaluate:self.budget.transition(batch,self.index)
        age=time.monotonic()-obs['source_wall']
        if age>=self.cfg['freshness_s']:raise RuntimeError('Active control freshness fault')
        op='resume_boundary' if self.paused else 'step'
        payload=dict(command=command,stop=bool(decision['stop']),frame=obs['frame'])
        if op=='step':payload['freeze_after']=freeze
        elif freeze:raise ValueError('Cannot freeze immediately on a resumed first step')
        response=self.worker.call(op,**payload);self.paused=freeze
        if not evaluate:self.budget.confirm(batch,self.index)
        nxt=response['observation'];next_decision=self.infer(nxt)
        unsupported=False
        try:next_phi=self.phi(nxt)
        except LookupError:next_phi=phi;unsupported=True
        event=response['event'];kind=self.task.get('kind','mission')
        reward,parts,_=transition_reward(phi,next_phi,response['dt'],event,self.cfg)
        if kind=='execution':
            terminal=event is not None
            parts=dict(terminal={'subgoal_success':2.,'false_stop':-.5,'collision':-10.,'envelope':-10.}.get(event,0.),
                time=-.01*response['dt'],shaping=self.cfg['gamma']**(response['dt']/self.cfg['step_s'])*(0 if terminal else next_phi)-phi)
            reward=sum(parts.values())*self.cfg['reward_scale']
        row=dict(context,latent=decision['latent'],stop=decision['stop'],stop_probability=decision['stop_probability'],
            logprob=decision['logprob'],value=decision['value'],next_value=next_decision['value'],reward=reward,reward_parts=parts,
            dt=response['dt'],terminated=response['terminated'],truncated=response['truncated'],rollout_boundary=freeze,
            proposed_command=command,dispatched_command=nxt['preceding_command'],event=event,task_id=self.task['id'],
            task_kind=kind,difficulty=self.task.get('difficulty'),distance_bin=self.task.get('distance_bin'),scene_id=self.scene['scene_id'],
            rgb_path=obs['rgb_path'],next_rgb_path=nxt['rgb_path'],sim_s=obs['sim_s'],next_sim_s=nxt['sim_s'],
            state=obs['state'],next_state=nxt['state'],decision_s=decision['decision_s'],source_age_s=age,
            step_wall_s=nxt['source_wall']-obs['source_wall'],
            action_saturation_fraction=float(np.mean(np.abs(np.tanh(decision['latent']))>.95)),
            attempt_id=self.attempt,policy_iteration=iteration,valid_for_ppo=not unsupported)
        self.obs=nxt;self.decision=next_decision;self.episode_reward+=reward;self.episode_steps+=1;self.stop_attempts+=decision['stop']
        delta=np.asarray(nxt['state']['position'])-self.task['goal']
        self.goal_visits+=bool(np.linalg.norm(delta[:2])<=3 and abs(delta[2])<=2)
        if event or response['truncated'] or unsupported:
            summary=dict(attempt_id=self.attempt,task_id=self.task['id'],kind=kind,scene_id=self.scene['scene_id'],
                difficulty=self.task.get('difficulty'),distance_bin=self.task.get('distance_bin'),
                event='coverage_exit' if unsupported else event or 'timeout',reward=self.episode_reward,steps=self.episode_steps,
                duration_s=nxt['elapsed_s'],reset_wall_s=nxt['reset_wall_s'],pause_wall_s=nxt['pause_wall_s'],
                stops=self.stop_attempts,goal_region_observations=self.goal_visits)
            self.episodes.append(summary);self.coverage.append(unsupported)
            write(self.root/(self.attempt+'-outcome.json'),summary)
            if unsupported:self.worker.call('end_episode');self.paused=False
            elif self.paused:self.worker.call('release_terminal');self.paused=False
            self.obs=None;self.decision=None
        if unsupported:
            # Preserve the actual crossing action. Do not assign fabricated
            # geometric reward outside the survey, and never optimize this batch.
            write(self.root/(self.attempt+'-unsupported.json'),row)
            if sum(self.coverage)>=2:raise RuntimeError('Two coverage exits in twenty launches: resurvey required')
        return row

    def collect(self,quota,batch,iteration):
        try:return self._collect(quota,batch,iteration)
        except BaseException:
            self.window.abort.set();raise

    def _collect(self,quota,batch,iteration):
        rows=[];path=self.root/f'{batch}-worker{self.index}.jsonl'
        with path.open('a',encoding='utf-8') as stream:
            while len(rows)<quota:
                if self.window.abort.is_set() or not self.window.remaining():raise RuntimeError('Collection interrupted; batch archived')
                if self.obs is None:self.begin()
                elif self.paused:
                    self.obs=self.worker.call('refresh_boundary');self.decision=self.infer(self.obs,replace=True)
                boundary=len(rows)==quota-1
                try:
                    row=self.step(freeze=boundary and not self.cfg.get('endpoint_pilot'),batch=batch,iteration=iteration)
                except RuntimeError as error:
                    freshness=any(s in str(error) for s in ('Active control watchdog exceeded source freshness',
                        'Active control freshness fault','Stale policy decision; watchdog brakes'))
                    if not self.cfg.get('endpoint_pilot') or not freshness:raise
                    self.worker.call('end_episode')
                    self.budget.discard_unobserved(batch,self.index)
                    self.window.freshness_faults+=1;self.window.freshness_by_batch[batch]+=1
                    if rows and rows[-1]['attempt_id']==self.attempt:rows[-1]['truncated']=True
                    summary=dict(attempt_id=self.attempt,task_id=self.task['id'],kind=self.task.get('kind','mission'),
                        scene_id=self.scene['scene_id'],event='infrastructure_freshness_failure',reason=str(error),
                        reward=self.episode_reward,steps=self.episode_steps,duration_s=self.obs['elapsed_s'],
                        reset_wall_s=self.obs['reset_wall_s'],pause_wall_s=0,stops=self.stop_attempts,
                        goal_region_observations=self.goal_visits,uncertain_transition_excluded=True,
                        last_confirmed_rgb=self.obs['rgb_path'])
                    self.episodes.append(summary);write(self.root/(self.attempt+'-outcome.json'),summary)
                    self.obs=None;self.decision=None
                    if (self.window.freshness_by_batch[batch]>self.cfg['max_recoverable_freshness_faults_per_batch'] or
                        self.window.freshness_faults>self.cfg['max_recoverable_freshness_faults_total']):
                        raise RuntimeError('Repeated endpoint pilot freshness faults; operator review required') from error
                    break
                if boundary and self.cfg.get('endpoint_pilot') and self.obs is not None:
                    # Explicit training truncation with bootstrap; no pause or
                    # unrecorded continuation across an optimizer update.
                    self.worker.call('end_episode')
                    row['truncated']=True;row['rollout_boundary']=True
                    summary=dict(attempt_id=self.attempt,task_id=self.task['id'],kind=self.task.get('kind','mission'),
                        scene_id=self.scene['scene_id'],difficulty=self.task.get('difficulty'),distance_bin=self.task.get('distance_bin'),
                        event='rollout_boundary',reward=self.episode_reward,steps=self.episode_steps,
                        duration_s=self.obs['elapsed_s'],reset_wall_s=self.obs['reset_wall_s'],pause_wall_s=0,
                        stops=self.stop_attempts,goal_region_observations=self.goal_visits)
                    self.episodes.append(summary);write(self.root/(self.attempt+'-outcome.json'),summary)
                    self.obs=None;self.decision=None
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if not row['valid_for_ppo']:
                    if rows and rows[-1]['attempt_id']==row['attempt_id']:rows[-1]['truncated']=True
                    self.budget.replace_invalid(batch);continue
                rows.append(row)
                self.curriculum.add(row['task_kind'])
                if self.obs is None:break
        return rows

    def benchmark(self,quota,batch,starts):
        # Controlled engineering truncations measure reset overhead. They never
        # enter PPO, and are explicitly separated from complete flight outcomes.
        missions=[t for t in self.curriculum.tasks if t['scene_id']==self.scene['scene_id'] and t.get('kind')=='mission']
        rows=[];reset_rows=[];started=time.monotonic();episode=0;pause_checks=[]
        with (self.root/f'{batch}-benchmark-worker{self.index}.jsonl').open('x',encoding='utf-8') as stream:
            while len(rows)<quota or episode<starts:
                if not self.window.remaining():raise RuntimeError('Benchmark window exhausted')
                if len(rows)>=quota:raise RuntimeError('Reset qualification needs more reserved transitions')
                self.begin(missions[episode%len(missions)]);reset_rows.append(self.obs['reset_wall_s']);episode+=1
                limit=max(1,min(400,(quota-len(rows))//max(1,starts-episode+1)))
                for _ in range(limit):
                    if self.paused:
                        time.sleep(1.)
                        before=dict(self.obs['state']);stamp=self.obs['sim_s']
                        self.obs=self.worker.call('refresh_boundary');self.decision=self.infer(self.obs,replace=True)
                        pause_checks.append(dict(state_unchanged=self.obs['state']==before,stamp_unchanged=self.obs['sim_s']==stamp))
                    row=self.step(freeze=not pause_checks and not self.paused,batch=batch);row['qualification_only']=True
                    if not row['valid_for_ppo']:raise RuntimeError('Unsupported benchmark transition; resurvey before qualification')
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    if self.obs is None:break
                self.worker.call('brake');self.obs=None;self.decision=None
        if not pause_checks or not all(r['state_unchanged'] and r['stamp_unchanged'] for r in pause_checks):raise RuntimeError('Training pause/resume qualification failed')
        return dict(rows=rows,resets=reset_rows,pause_checks=pause_checks,wall_s=time.monotonic()-started)


def run(args):
    cfg,manifest,scenes,identity,quality=preflight(args)
    if args.preflight:
        print(json.dumps(dict(ready=True,identity=identity,quality=quality)));return
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    campaign=Path(args.workspace)/'ppo-campaign'
    with FlightLock(campaign,'ppo-overnight'),ExitStack() as resources:
        budget=Budget(campaign,cfg);window=Window(args.hours);window.stop_file=root/'STOP';seed_all(cfg['seed'])
        resources.callback(budget.close);window.abort=threading.Event();window.sim_ports={}
        model=ActorCritic(args.backbone,cfg['stop_prior']).to(args.device).eval()
        counts=dict(iteration=0,transitions=0);evaluated=[];mixture={};reports=[];lineage=None
        checkpoint=root/'latest.pt'
        if checkpoint.exists():
            if not args.resume:raise ValueError('Existing run requires --resume')
            saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
            if saved['identity']!=identity or saved['schema']!=cfg['schema']:raise ValueError('Resume identity changed')
            model.load_state_dict(saved['model']);counts=saved['counts'];evaluated=saved['evaluated'];mixture=saved['mixture'];lineage=saved['lineage']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        elif args.resume:raise ValueError('No checkpoint to resume')
        elif args.migrate:lineage=migrate(model,args.migrate,args.backbone,budget.campaign);write(root/'migration.json',lineage)
        window.freshness_faults=saved.get('freshness_faults',0) if args.resume else 0
        window.freshness_by_batch=Counter()
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
        if checkpoint.exists():optimizer.load_state_dict(saved['optimizer'])
        lane=ComputeLane();log_lock=threading.Lock()
        def proposal_log(row):
            with log_lock,(root/'proposals.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if cfg.get('endpoint_pilot'):
            class NoGuidance:
                receipts=[]
                def __bool__(self):return False
                def forget(self,*args):pass
                def invalidate(self):pass
                def close(self):pass
            guidance=NoGuidance()
            model.execution_value.requires_grad_(False)
        else:
            proposer=RegionProposer(lane,path=args.qwen_model)
            if proposer.base_identity!=quality['model_identity']:raise ValueError('Live Qwen differs from independently audited weights')
            guidance=Guidance(proposer,proposal_log)
        resources.callback(guidance.close)
        banks={i:FeatureBank() for i in range(len(args.worker)+1)}
        scheduler=Inference(model,lane,banks,guidance)
        resources.callback(scheduler.close)
        curriculum=Curriculum([t for t in manifest['tasks'] if t['split']=='train'],cfg,mixture)
        if args.resume:
            curriculum.indices=Counter(dict(saved.get('curriculum_indices',[])))
            if (root/'updates.jsonl').exists():reports=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines() if json.loads(line)['counts']['iteration']<=counts['iteration']]
        workers=[]
        for address,auth in zip(args.worker,args.authfile):
            worker=Worker(address,auth);workers.append(worker);resources.callback(worker.close)
        evaluation=Worker(args.evaluation_worker,args.evaluation_authfile) if not cfg.get('endpoint_pilot') else None
        if evaluation:resources.callback(evaluation.close)
        train_scenes=[s for s in scenes.values() if s['split']=='train'];collectors=[];sanity=[t for t in manifest['tasks'] if t.get('sanity_validation')]
        for i,worker in enumerate(workers):
            collectors.append(Collector(i,worker,train_scenes[i%len(train_scenes)],scheduler,banks[i],budget,curriculum,cfg,root,window,guidance))
        # All cold initialization and kernel warm-up precede active flight control.
        with torch.no_grad(),lane.fast():
            for _ in range(8):
                raw=model.encode_backbone(torch.zeros(len(workers),3,480,640,device=args.device,dtype=torch.uint8))
                model(raw[:,None].expand(-1,4,-1,-1,-1),raw,raw.new_zeros(len(workers),4),raw.new_zeros(len(workers),4,4),
                    torch.ones(len(workers),4,device=args.device,dtype=torch.bool))
        for collector in collectors:collector.select()
        if args.resume:
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        if args.benchmark:
            import psutil,subprocess
            sampled=[];gpu_samples=[];stop_sampling=threading.Event()
            def sample():
                last_gpu=-math.inf
                while not stop_sampling.is_set():
                    memory=psutil.virtual_memory();sampled.append(memory.total-memory.available)
                    if time.monotonic()-last_gpu>=2:
                        last_gpu=time.monotonic()
                        try:
                            query=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
                            values=[float(v.strip()) for v in query.stdout.splitlines()[0].split(',')]
                            gpu_samples.append(dict(wall=last_gpu,utilization_percent=values[0],temperature_c=values[1],power_w=values[2]))
                        except (OSError,ValueError,IndexError,subprocess.SubprocessError):pass
                    stop_sampling.wait(.25)
            thread=threading.Thread(target=sample,daemon=True)
            batch='benchmark-'+uuid.uuid4().hex;results=[];started=time.monotonic();failed=None
            if not budget.reserve_batch(batch,cfg['smoke_transitions']):raise RuntimeError('No whole benchmark reservation remains')
            thread.start();resources.callback(lambda:(stop_sampling.set(),thread.join(timeout=3)))
            try:
                with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                    fs=[pool.submit(c.benchmark,8192//len(workers),batch,math.ceil(20/len(workers))) for c in collectors]
                    for f in fs:results.append(f.result())
            except Exception as error:failed=str(error)
            finally:
                elapsed=time.monotonic()-started;stop_sampling.set();thread.join()
                for w in workers:
                    w.call('flush');w.call('finish_session');w.close()
                evaluation.close()
                scheduler.close();guidance.close();budget.finish(batch,False);budget.close()
            rows=[r for result in results for r in result['rows']];resets=[v for result in results for v in result['resets']]
            proposal_rows=guidance.receipts
            # Per-worker physical dt / active decision-step wall is measured by
            # the environment response, not by optimizer-free wall estimates.
            ratios=[r['dt']/r['step_wall_s'] for r in rows if r['step_wall_s']>0]
            telemetry=sorted({Path(r['rgb_path']).parent/'telemetry.jsonl' for r in rows})
            if any(not p.resolve().is_relative_to(Path(args.recordings_root).resolve()) for p in telemetry):
                raise ValueError('Workload recordings escape the declared private root')
            watchdogs=0;recorded_bytes=0
            for path in telemetry:
                for line in path.open(encoding='utf-8-sig'):
                    r=json.loads(line)
                    if r.get('kind')=='dispatch' and r.get('stale') and r.get('active_policy',True):watchdogs+=1
                recorded_bytes+=sum(p.stat().st_size for p in path.parent.glob('*.png'))
            free=shutil.disk_usage(args.recordings_root).free;memory=psutil.virtual_memory()
            summary=dict(reset_starts=len(resets),reset_exhausted=int(failed is not None),reset_mean_s=float(np.mean(resets)) if resets else 1e9,
                pause_resume_passed=bool(results) and all(r.get('pause_checks') for r in results),
                source_age_p99_s=float(np.percentile([r['source_age_s'] for r in rows],99)) if rows else 1e9,
                active_watchdog_faults=watchdogs,fresh_proposal_fraction=sum(r.get('fresh_valid',False) for r in proposal_rows)/max(1,len(proposal_rows)),
                recording_failures=int(failed is not None),sim_wall_ratio=float(np.median(ratios)) if ratios else 0.,
                peak_memory_bytes=max(sampled or [memory.total]),measured_growth_bytes=max(sampled or [0])-min(sampled or [0]),
                total_memory_bytes=memory.total,disk_free_bytes=free,
                projected_recording_bytes=math.ceil(recorded_bytes/max(1,len(rows))*(cfg['smoke_transitions']-budget.campaign['ppo_transitions'])),
                actor_qwen_recording_concurrent=bool(proposal_rows and telemetry and not failed),
                valid_transitions_per_wall_s=len(rows)/elapsed)
            summary.update(gpu_utilization_mean_percent=float(np.mean([r['utilization_percent'] for r in gpu_samples])) if gpu_samples else None,
                gpu_temperature_max_c=max(r['temperature_c'] for r in gpu_samples) if gpu_samples else None)
            write(root/'benchmark.json',dict(schema='ppo-workload-trial/v1',identity=identity,workers=len(workers),
                model_identity=quality['model_identity'],summary=summary,failed=failed,wall_s=elapsed,
                resets=resets,gpu_samples=gpu_samples,proposal_receipts=proposal_rows,telemetry=[dict(path=str(p.resolve()),sha256=digest(p)) for p in telemetry]))
            if failed:raise RuntimeError(failed)
            return
        def save():
            saved=dict(schema=cfg['schema'],capability='simulation-only',identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                experiment=cfg.get('experiment','proposal-conditioned'),mode2_enabled=bool(guidance),
                freshness_faults=window.freshness_faults,
                counts=counts,evaluated=evaluated,mixture=dict(curriculum.counts),lineage=lineage,
                curriculum_indices=list(curriculum.indices.items()),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            pending=root/'latest.pending';torch.save(saved,pending);pending.replace(checkpoint)
        def evaluate(mark):
            guidance.invalidate();results=[]
            for t in sanity:
                if not window.remaining():break
                scene=scenes[t['scene_id']];task=dict(t,timeout_s=min(180,t['timeout_s']))
                c=Collector(len(workers),evaluation,scene,scheduler,banks[len(workers)],budget,curriculum,cfg,root,window,guidance)
                c.select('evaluation');c.begin(task,True)
                path=root/f'eval-{mark}-{task["id"]}.jsonl'
                with path.open('a',encoding='utf-8') as f:
                    while c.obs is not None and window.remaining():
                        row=c.step(evaluate=True,iteration=counts['iteration']);f.write(json.dumps(row)+'\n')
                evaluation.call('brake');results.extend(c.episodes);c.bank.trim()
            write(root/f'evaluation-{mark}.json',dict(episodes=results,complete=len(results)==5,milestone_eligible=False))
            if len(results)==5:evaluated.append(mark)
            guidance.invalidate();save()
        status='not_started';batch=None;pre_model=None;pre_optimizer=None
        try:
            if not cfg.get('endpoint_pilot') and 0 not in evaluated:evaluate(0)
            if args.hours*3600-(window.deadline-time.monotonic())>3600:raise RuntimeError('Admission/warm-up exceeded one hour')
            with ThreadPoolExecutor(max_workers=len(workers),thread_name_prefix='collector') as pool:
                while window.remaining():
                    if window.deadline-time.monotonic()<=min(1800,args.hours*3600/16):status='reporting_window';break
                    batch=f'rollout-{counts["iteration"]+1:06d}-{uuid.uuid4().hex[:8]}'
                    if not budget.reserve_batch(batch,cfg['smoke_transitions']):status='full_batch_budget_limit';break
                    save()
                    policy_path=root/f'policy-{counts["iteration"]:06d}.pt'
                    if not policy_path.exists():shutil.copyfile(checkpoint,policy_path)
                    start=time.monotonic();by_worker={c.index:[] for c in collectors};quota=8192//len(workers)
                    write(root/'status.json',dict(status='collecting',counts=counts,batch=batch,mode2_enabled=bool(guidance),
                        scope='endpoint-mode1-pilot' if cfg.get('endpoint_pilot') else 'proposal-conditioned',training_running=True))
                    while any(len(r)<quota for r in by_worker.values()):
                        active=[c for c in collectors if len(by_worker[c.index])<quota]
                        futures=[(c,pool.submit(c.collect,quota-len(by_worker[c.index]),batch,counts['iteration'])) for c in active]
                        for c,f in futures:by_worker[c.index].extend(f.result())
                        # Every collector is now either terminal or frozen at its
                        # quota. Cold scene changes never overlap a live actor.
                        for c in active:
                            if c.obs is None:
                                i=(train_scenes.index(c.scene)+len(workers))%len(train_scenes)
                                if train_scenes[i]!=c.scene:c.scene=train_scenes[i];c.select()
                    rows=[r for c in collectors for r in by_worker[c.index]]
                    if len(rows)!=8192:raise ValueError('Incomplete rollout')
                    for worker in workers:worker.call('flush')
                    guidance.invalidate()
                    path=root/f'rollout-{counts["iteration"]+1:06d}.jsonl'
                    if path.exists():
                        if (root/f'update-{counts["iteration"]+1:06d}.json').exists():raise RuntimeError('Accepted update is ahead of checkpoint; reconcile before resuming')
                        path.rename(root/f'archived-{path.stem}-{uuid.uuid4().hex}.jsonl')
                    with path.open('x',encoding='utf-8') as f:
                        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
                    write(root/'status.json',dict(status='optimizing',counts=counts,batch=batch,fresh_transitions=len(rows),training_running=True))
                    pre_model=copy.deepcopy(model.state_dict());pre_optimizer=copy.deepcopy(optimizer.state_dict());pre_counts=dict(counts);optimized=time.monotonic()
                    with lane.fast():report=update(model,optimizer,rows,lambda ids:batch_for([rows[i] for i in ids],banks,args.device),cfg)
                    report.update(update_wall_s=time.monotonic()-optimized,collection_wall_s=optimized-start,
                        collection_policy_sha256=digest(policy_path),
                        rollout_sha256=digest(path),
                        counts=dict(iteration=counts['iteration']+1,transitions=counts['transitions']+8192),
                        decision_p95_s=float(np.percentile([r['decision_s'] for r in rows],95)),
                        source_age_p99_s=float(np.percentile([r['source_age_s'] for r in rows],99)),
                        action_saturation_fraction=float(np.mean([r['action_saturation_fraction'] for r in rows])),
                        guidance_fraction=float(np.mean([r['guidance']['subgoal'] is not None for r in rows])),
                        mean_reward=float(np.mean([r['reward'] for r in rows])),mixture=dict(curriculum.counts))
                    complete=[e for c in collectors for e in c.episodes]
                    missions=[e for e in complete if e['kind']=='mission' and not e['event'].startswith('infrastructure_')]
                    report.update(mean_episode_length_steps=float(np.mean([e['steps'] for e in complete])) if complete else None,
                        success_rate=sum(e['event']=='success' for e in missions)/len(missions) if missions else None,
                        outcome_counts_by_kind={kind:dict(Counter(e['event'] for e in complete if e['kind']==kind)) for kind in ('mission','execution','arrival')},
                        collisions=sum(e['event']=='collision' for e in complete),
                        false_stops=sum(e['event']=='false_stop' for e in complete),
                        correct_stops=sum(e['event']=='success' for e in complete),
                        goal_visits_without_stop=sum(e['goal_region_observations']>0 and e['stops']==0 for e in complete),
                        reset_mean_s=float(np.mean([e['reset_wall_s'] for e in complete])) if complete else None,
                        value_warning=report['explained_variance'] is not None and report['explained_variance']<0,
                        gaussian_std=model.log_std.detach().clamp(-3,.5).exp().cpu().tolist())
                    report['freshness_faults']=window.freshness_faults
                    with (root/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
                    if report['final_rollout_kl']>.1:
                        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),report=report),root/f'rejected-{batch}.pt')
                        model.load_state_dict(pre_model);optimizer.load_state_dict(pre_optimizer)
                        raise RuntimeError('Final rollout KL exceeded 0.1; candidate saved and pre-update state restored')
                    reports.append(report);counts=report['counts'];budget.finish(batch,True);batch=None;save()
                    pre_model=pre_optimizer=None
                    if len(guidance.receipts)>=10 and sum(r.get('fresh_valid',False) for r in guidance.receipts[-20:])/len(guidance.receipts[-20:])<.8:
                        raise RuntimeError('mode2_not_admitted: live fresh-valid proposal rate fell below 80 percent')
                    torch.save(torch.load(checkpoint,map_location='cpu',weights_only=False),root/f'update-{counts["iteration"]:06d}.pt')
                    write(root/f'update-{counts["iteration"]:06d}.json',dict(iteration=counts['iteration'],
                        checkpoint_sha256=digest(root/f'update-{counts["iteration"]:06d}.pt'),rollout_sha256=digest(path),
                        policy_sha256=digest(policy_path),accepted=True))
                    write(root/'video-request.json',dict(schema='photo-map-ppo-video/v1',iteration=counts['iteration'],
                        checkpoint_sha256=digest(checkpoint),rollout_sha256=digest(path),selection='first complete mission and first auxiliary episode by attempt ID'))
                    if len(reports)>=3 and all(r['action_saturation_fraction']>.95 for r in reports[-3:]):raise RuntimeError('Persistent action saturation')
                    if len(reports)>=6:
                        baseline=np.median([r['normalized_value_mse'] for r in reports[:3]])
                        if all(r['explained_variance'] is not None and r['explained_variance']<-1 and r['normalized_value_mse']>10*baseline for r in reports[-3:]):
                            raise RuntimeError('Persistent combined critic divergence')
                    for mark in (() if cfg.get('endpoint_pilot') else (50000,100000)):
                        if budget.campaign['ppo_transitions']>=mark and mark not in evaluated:evaluate(mark)
                    for b in banks.values():b.trim()
                else:status='window_complete'
        except BaseException as error:
            if pre_model is not None:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),reason=str(error)),root/'interrupted-optimizer.pt')
                model.load_state_dict(pre_model);optimizer.load_state_dict(pre_optimizer);counts=pre_counts
            status='stopped';write(root/'failure.json',dict(reason=str(error),type=type(error).__name__,counts=counts))
            if batch:budget.finish(batch,False)
            raise
        finally:
            for w in workers+([evaluation] if evaluation else []):w.close()
            scheduler.close();guidance.close();save();budget_snapshot=budget.snapshot();budget.close()
            write(root/'status.json',dict(status=status,counts=counts,mixture=dict(curriculum.counts),
                budget=budget_snapshot,
                trained_navigation_accepted=False,world_selector=False,photo_slam_operational_claim=False))
            from .ppo_report import summarize
            summarize(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).with_name('ppo_overnight.json')))
    for name in ('manifest','backbone','output','workspace'):p.add_argument('--'+name,required=True)
    for name in ('audit','grades','evaluation-worker','evaluation-authfile'):p.add_argument('--'+name)
    p.add_argument('--endpoint-pilot',action='store_true',help='Explicit Mode 1 pilot using checked endpoints and Euclidean shaping')
    p.add_argument('--admission');p.add_argument('--benchmark',action='store_true')
    p.add_argument('--recordings-root',help='Private recording root for this workload qualification only')
    p.add_argument('--worker',action='append',required=True);p.add_argument('--authfile',action='append',required=True)
    p.add_argument('--qwen-model',default='/models/qwen2.5-vl-3b');p.add_argument('--device',default='cuda')
    p.add_argument('--migrate');p.add_argument('--resume',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--hours',type=float,default=8)
    args=p.parse_args()
    if args.benchmark and not args.recordings_root:p.error('--benchmark requires --recordings-root')
    run(args)
