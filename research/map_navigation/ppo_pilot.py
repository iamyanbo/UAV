"""Fresh-trajectory PPO pilot. Requires independently qualified task manifests.

Host worker: python -m research.map_navigation.ppo_env serve ...
GPU process: python -m research.map_navigation.ppo_pilot ...
The authenticated loopback worker carries reward labels separately from RGB.
"""
import argparse
from collections import deque
import json
import math
from multiprocessing.connection import Client
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from PIL import Image
from .common import read,write,digest,Window,FlightLock,seed_all
from .ppo_core import ActorCritic,command_from_latent,potential,transition_reward,update


def validate_manifest(path,cfg):
    manifest=read(path)
    if manifest.get('schema')!='photo-map-ppo-tasks/v1':raise ValueError('PPO task manifest required')
    if manifest.get('config_sha256')!=digest(Path(__file__).with_name('ppo_pilot.json')):
        raise ValueError('Pilot configuration identity differs')
    groups={};scenes={}
    for scene in manifest['scenes']:
        ident=scene['scene_id'];split=scene['split']
        if split not in ('train','validation') or ident in scenes:raise ValueError('Invalid/duplicate scene split')
        group=scene['geography_id']
        if not group or group in groups and groups[group]!=split:raise ValueError('Geography leakage')
        groups[group]=split
        q=read(scene['qualification'])
        if digest(scene['qualification'])!=scene['qualification_sha256']:raise ValueError('Qualification changed')
        required=('camera','geometry','motion','collision','stop','stale_frame','timing')
        if (not q.get('qualified') or not q.get('camera_reviewed') or not q.get('geometry_reviewed') or
                q.get('config')!=cfg or not all(q.get('checks',{}).get(k) for k in required)):
            raise ValueError('Scene qualification incomplete: '+str(ident))
        if len(q.get('resets',[]))<cfg['reset']['qualification_repetitions'] or not all(r[-1]['passed'] for r in q['resets']):
            raise ValueError('Twenty valid reset repetitions required')
        if len(q.get('camera_attitudes',[]))!=9 or not all(r['pose_agreement'] for r in q['camera_attitudes']):
            raise ValueError('Camera images must cover nine verified body attitudes')
        if q.get('scene_sha256')!=digest(scene['descriptor']):raise ValueError('Scene descriptor changed')
        descriptor=read(scene['descriptor'])
        if str(descriptor['scene_id'])!=str(ident):raise ValueError('Scene identity mismatch')
        for asset in descriptor['assets']:
            if digest(asset['path'])!=asset['sha256']:raise ValueError('Scene asset changed')
        scenes[ident]=scene
    for split,key in [('train','minimum_train_scenes'),('validation','minimum_validation_scenes')]:
        if len({s['geography_id'] for s in scenes.values() if s['split']==split})<cfg[key]:
            raise ValueError('Insufficient independent qualified '+split+' geography')
    ids=set()
    for t in manifest['tasks']:
        if t['id'] in ids:raise ValueError('Duplicate task')
        ids.add(t['id']);s=scenes[t['scene_id']]
        if t['split']!=s['split']:raise ValueError('Task split mismatch')
        if digest(t['goal_image'])!=t['goal_sha256']:raise ValueError('Goal photograph changed')
        if t['camera']!=cfg['camera'] or not t.get('unobstructed') or not t.get('geometry_evidence'):
            raise ValueError('Task camera/clearance evidence missing')
        if digest(t['geometry_evidence'])!=t['geometry_evidence_sha256']:raise ValueError('Task geometry changed')
        geometry=read(t['geometry_evidence'])
        if geometry.get('task_id')!=t['id'] or not geometry.get('swept_volume_free') or geometry.get('camera')!=cfg['camera']:
            raise ValueError('Task requires observed swept-volume clearance including camera mount')
        if geometry.get('start')!=t['start'] or geometry.get('goal')!=t['goal']:
            raise ValueError('Clearance evidence does not bind the task endpoints')
        if 'captures' in geometry:
            for capture in geometry['captures']:
                if digest(capture['path'])!=capture['sha256']:raise ValueError('Geometry capture changed')
            if digest(geometry['physical_evidence'])!=geometry['physical_sha256'] or not read(geometry['physical_evidence'])['passed']:
                raise ValueError('Physical task qualification changed or failed')
        if t['behavior'] not in ('level','climb','descent'):raise ValueError('Unknown curriculum category')
        distance=math.dist(t['start'],t['goal'])
        if not 10<=distance<=30:raise ValueError('Initial curriculum requires 10–30 m tasks')
        numbers=t['start']+t['goal']+t['bounds'][0]+t['bounds'][1]+[t['start_yaw_deg'],t['goal_yaw_deg']]
        if not np.isfinite(numbers).all():raise ValueError('Nonfinite task geometry')
        for position in [t['start'],t['goal']]:
            if any(x<a or x>b for x,a,b in zip(position,*t['bounds'])):raise ValueError('Task leaves its envelope')
    for split in ('train','validation'):
        if {t['behavior'] for t in manifest['tasks'] if t['split']==split}!={'level','climb','descent'}:
            raise ValueError('Each split needs level, climb and descent tasks')
    budget=manifest['prior_budget_usage']
    if digest(budget['receipt'])!=budget['sha256']:raise ValueError('Budget reconciliation receipt changed')
    reconciled=read(budget['receipt'])
    if not all(k in reconciled for k in ('training_attempts','ppo_transitions','learner_attempts')):
        raise ValueError('Missing reconciled campaign counts')
    return manifest,scenes,reconciled


class Worker:
    def __init__(self,address,authfile):
        host,port=address.rsplit(':',1)
        if host!='127.0.0.1':raise ValueError('Worker must be on loopback')
        self.connection=Client((host,int(port)),authkey=Path(authfile).read_bytes());self.scene=None

    def call(self,op,**kwargs):
        self.connection.send(dict(op=op,**kwargs))
        if not self.connection.poll(180):raise RuntimeError('Worker timeout; watchdog owns braking')
        answer=self.connection.recv()
        if not answer['ok']:raise RuntimeError(answer['error'])
        return answer['result']

    def select(self,scene,phase):
        # A phase switch always creates fresh simulator state, including validation.
        key=(scene['scene_id'],phase)
        if key!=self.scene:
            self.call('select',scene=read(scene['descriptor']),session=phase+'-'+uuid.uuid4().hex)
            self.scene=key

    def close(self):
        try:self.connection.send(dict(op='close'))
        except (EOFError,OSError):pass
        self.connection.close()


class Observations:
    def __init__(self,model):self.model=model;self.frames={};self.paths={};self.next_id=0;self.history=deque(maxlen=4)

    def encode(self,pixels,path):
        device=next(self.model.parameters()).device
        tensor=torch.from_numpy(np.asarray(pixels).copy()).permute(2,0,1)[None].to(device)
        raw=self.model.encode_backbone(tensor)[0].cpu().half()
        ident=self.next_id;self.next_id+=1;self.frames[ident]=raw;self.paths[ident]=path;return ident

    def reset(self,goal_image):
        self.history.clear()
        with Image.open(goal_image) as image:
            if image.size!=(640,480):raise ValueError('Goal photograph calibration differs')
            self.goal=self.encode(image.convert('RGB'),goal_image)

    def observe(self,obs):
        pixels=np.frombuffer(obs['rgb'],np.uint8).reshape(480,640,3)
        ident=self.encode(pixels,obs['rgb_path']);self.history.append((ident,obs['sim_s'],obs['preceding_command']))
        return dict(history_ids=[r[0] for r in self.history],goal_id=self.goal,
                    history_rgb=[self.paths[r[0]] for r in self.history],goal_image=self.paths[self.goal],
                    stamps=[r[1] for r in self.history],preceding=[r[2] for r in self.history])

    def batch(self,records):
        device=next(self.model.parameters()).device;histories=[];goals=[];times=[];commands=[];valid=[]
        for row in records:
            ids=row['history_ids'];n=len(ids);template=self.frames[ids[-1]]
            histories.append(torch.stack([torch.zeros_like(template)]*(4-n)+[self.frames[i] for i in ids]))
            goals.append(self.frames[row['goal_id']]);times.append([0.]*(4-n)+[s-row['stamps'][-1] for s in row['stamps']])
            commands.append([[0.]*4]*(4-n)+row['preceding']);valid.append([False]*(4-n)+[True]*n)
        result=dict(history=torch.stack(histories).to(device),goal=torch.stack(goals).to(device),
            times=torch.tensor(times,device=device),commands=torch.tensor(commands,device=device,dtype=torch.float32),
            valid=torch.tensor(valid,device=device))
        if 'latent' in records[0]:
            result.update(latent=torch.tensor([r['latent'] for r in records],device=device),
                          stop=torch.tensor([r['stop'] for r in records],device=device,dtype=torch.float32))
        return result

    def trim(self):
        keep={r[0] for r in self.history}|{self.goal};self.frames={k:v for k,v in self.frames.items() if k in keep}
        self.paths={k:v for k,v in self.paths.items() if k in keep}


def choose_tasks(manifest,split):
    # Stable round robin through altitude categories and scenes; no cherry-picking.
    rows=[t for t in manifest['tasks'] if t['split']==split]
    buckets={b:sorted([t for t in rows if t['behavior']==b],key=lambda t:(str(t['scene_id']),t['id'])) for b in ('level','climb','descent')}
    result=[]
    for i in range(max(map(len,buckets.values()))):
        for bucket in buckets.values():result.append(bucket[i%len(bucket)])
    return result


def train(args):
    workspace=Path(args.workspace or Path(args.output).resolve().parent)/'ppo-campaign'
    workspace.mkdir(parents=True,exist_ok=True)
    with FlightLock(workspace,'ppo-pilot'):return _train(args,workspace)


def _train(args,workspace):
    cfg=read(Path(__file__).with_name('ppo_pilot.json'));manifest,scenes,prior=validate_manifest(args.manifest,cfg)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    seed_all(cfg['seed']);model=ActorCritic(args.backbone).to(args.device).eval()
    # Compile/cache first-use GPU kernels before a live observation ages.
    with torch.no_grad():
        for _ in range(8):
            raw=model.encode_backbone(torch.zeros(1,3,480,640,device=args.device,dtype=torch.uint8))
            model(raw[:,None].expand(-1,4,-1,-1,-1),raw,raw.new_zeros(1,4),
                  raw.new_zeros(1,4,4),torch.ones(1,4,device=args.device,dtype=torch.bool))
    if str(args.device).startswith('cuda'):torch.cuda.synchronize()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    files=list(Path(__file__).parent.glob('ppo_*.py'))+[Path(__file__).with_name('temporal.py'),Path(__file__).parents[1]/'rgb_flight/goal_matching.py']
    identity=dict(config=digest(Path(__file__).with_name('ppo_pilot.json')),manifest=digest(args.manifest),backbone=digest(args.backbone),
        source={p.name:digest(p) for p in files})
    count=dict(transitions=0,attempts=0,iteration=0,evaluation_attempts=0)
    evaluations=[]
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        if not args.resume:raise ValueError('Checkpoint exists; explicit --resume required')
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if saved['schema']!=cfg['schema'] or saved['identity']!=identity:raise ValueError('Incompatible pilot resume')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);count=saved['counts'];evaluations=saved['evaluations']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    elif args.resume:raise ValueError('No pilot checkpoint to resume')
    # Journal counts reservations/transitions even when a crash preceded checkpoint.
    if (out/'budget.json').exists():
        budget=read(out/'budget.json')
        if budget['identity']!=identity:raise ValueError('Budget belongs to another pilot')
        count={k:max(count[k],budget['counts'][k]) for k in count}
    campaign_path=workspace/'budget.json'
    if campaign_path.exists():
        campaign=read(campaign_path)
        if campaign['prior_receipt_sha256']!=manifest['prior_budget_usage']['sha256']:
            raise ValueError('Campaign budget baseline changed; reconcile explicitly')
    else:
        campaign=dict(prior_receipt_sha256=manifest['prior_budget_usage']['sha256'],
            training_attempts=prior['training_attempts'],learner_attempts=prior['learner_attempts'],ppo_transitions=prior['ppo_transitions'])
        write(campaign_path,campaign)
    def reserve(kind):
        if kind=='attempt':
            if campaign['training_attempts']>=cfg['campaign_attempts'] or campaign['learner_attempts']>=cfg['learner_attempts']:
                raise RuntimeError('Campaign attempt ceiling reached')
            campaign['training_attempts']+=1;campaign['learner_attempts']+=1
        else:
            if campaign['ppo_transitions']>=cfg['campaign_transitions']:raise RuntimeError('Campaign transition ceiling reached')
            campaign['ppo_transitions']+=1
        write(campaign_path,campaign)
    bank=Observations(model);window=Window(args.hours);window.stop_file=out/'STOP'
    tasks=choose_tasks(manifest,'train');validation=choose_tasks(manifest,'validation');phase=args.phase
    ceiling={'smoke':cfg['smoke_transitions'],'million':cfg['first_run_transitions'],'extension':cfg['extension_transitions']}[phase]
    cap=cfg['smoke_attempts'] if phase=='smoke' else cfg['learner_attempts']-prior['learner_attempts']
    if phase!='smoke':
        smoke=read(out/'smoke-review.json')
        if smoke.get('identity')!=identity or not smoke.get('infrastructure_passed') or not smoke.get('numerics_passed'):
            raise ValueError('Extension requires measured smoke review, not a success-rate threshold')
    if phase=='extension':
        if len(evaluations)<3:raise ValueError('Three evaluations required for extension')
        a,b,c=evaluations[-3:]
        if not (a['success_rate']<b['success_rate']<c['success_rate'] or c['median_closest_goal_m']<=.9*a['median_closest_goal_m']):
            raise ValueError('No measured improvement for extension')
    worker=Worker(args.worker,args.authfile)

    def journal():write(out/'budget.json',dict(identity=identity,counts=count,prior=prior))
    def save():
        state=dict(schema=cfg['schema'],capability='simulation-only',identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            counts=count,evaluations=evaluations,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        pending=out/'latest.pending';torch.save(state,pending);pending.replace(checkpoint)

    def episode(task,evaluate=False):
        worker.select(scenes[task['scene_id']],'evaluation' if evaluate else 'training')
        if evaluate:count['evaluation_attempts']+=1
        else:reserve('attempt');count['attempts']+=1
        journal();attempt=('eval-' if evaluate else 'train-')+uuid.uuid4().hex
        bank.reset(task['goal_image'])
        obs=worker.call('reset',task=task,attempt_id=attempt);context=bank.observe(obs)
        return obs,context,attempt

    def rollout_step(task,obs,context,evaluate=False):
        decision_started=time.monotonic()
        with torch.no_grad():
            normal,stop_dist,value=model(**bank.batch([context]))
            latent=normal.mean if evaluate else normal.sample()
            stop=(stop_dist.probs>=.5).float() if evaluate else stop_dist.sample()
            logp=normal.log_prob(latent).sum(-1)+stop_dist.log_prob(stop)
        command=command_from_latent(latent[0].cpu().tolist(),obs['preceding_command'],cfg['step_s'],cfg['limits'],cfg['acceleration'])
        # Reserve before dispatch so crashes cannot undercount executed transitions.
        if not evaluate:reserve('transition');count['transitions']+=1;journal()
        decision_s=time.monotonic()-decision_started;source_age_s=time.monotonic()-obs['source_wall']
        response=worker.call('step',command=command,stop=bool(stop.item()),frame=obs['frame'])
        nxt=response['observation'];next_context=bank.observe(nxt)
        with torch.no_grad():next_value=float(model(**bank.batch([next_context]))[2].item())
        initial=math.dist(task['start'],task['goal'])
        phi=potential(obs['state']['position'],task['goal'],initial);nphi=potential(nxt['state']['position'],task['goal'],initial)
        reward,parts,_=transition_reward(phi,nphi,response['dt'],response['event'],cfg)
        row=dict(context,latent=latent[0].cpu().tolist(),stop=float(stop.item()),logprob=float(logp.item()),value=float(value.item()),
            next_value=next_value,reward=reward,reward_parts=parts,dt=response['dt'],terminated=response['terminated'],truncated=response['truncated'],
            proposed_command=command,dispatched_command=nxt['preceding_command'],event=response['event'],task_id=task['id'],
            scene_id=task['scene_id'],rgb_path=obs['rgb_path'],next_rgb_path=nxt['rgb_path'],
            sim_s=obs['sim_s'],next_sim_s=nxt['sim_s'],state=obs['state'],next_state=nxt['state'],
            decision_s=decision_s,source_age_s=source_age_s,
            action_saturated=bool((latent.tanh().abs()>.95).any().item()))
        return row,nxt,next_context

    status='budget_complete'
    try:
        with (out/'updates.jsonl').open('a') as metrics:
            obs=context=task=attempt=None
            starting_iteration=count['iteration']
            while (window.remaining() and count['transitions']<ceiling and count['attempts']<cap and
                   (args.max_updates is None or count['iteration']-starting_iteration<args.max_updates)):
                if campaign['training_attempts']>=cfg['campaign_attempts'] or campaign['ppo_transitions']>=cfg['campaign_transitions']:break
                rows=[]
                while len(rows)<cfg['rollout_steps'] and window.remaining() and count['transitions']<ceiling:
                    if campaign['ppo_transitions']>=cfg['campaign_transitions']:break
                    if obs is None:
                        if count['attempts']>=cap or campaign['training_attempts']>=cfg['campaign_attempts'] or campaign['learner_attempts']>=cfg['learner_attempts']:break
                        task=tasks[count['attempts']%len(tasks)];obs,context,attempt=episode(task)
                    row,obs,context=rollout_step(task,obs,context);row['attempt_id']=attempt;rows.append(row)
                    if row['terminated'] or row['truncated']:obs=None
                if not rows:break
                # Pause collection at a valid rollout boundary. No simulator pause:
                # watchdog brakes while gradients run; resume begins a fresh reset.
                if not rows[-1]['terminated']:rows[-1]['truncated']=True
                worker.call('brake')
                obs=None
                count['iteration']+=1
                with (out/f'rollout-{count["iteration"]:06d}.jsonl').open('w') as stream:
                    for row in rows:stream.write(json.dumps(dict(row,policy_iteration=count['iteration']-1))+'\n')
                update_started=time.monotonic()
                report=update(model,optimizer,rows,lambda ids:bank.batch([rows[i] for i in ids]),cfg)
                report.update(counts=dict(count),update_wall_s=time.monotonic()-update_started,
                    decision_p95_s=float(np.percentile([r['decision_s'] for r in rows],95)),
                    source_age_p99_s=float(np.percentile([r['source_age_s'] for r in rows],99)),
                    reward_components={k:sum(r['reward_parts'][k] for r in rows) for k in ('terminal','time','shaping')},
                    action_saturation_fraction=sum(r['action_saturated'] for r in rows)/len(rows))
                metrics.write(json.dumps(report,allow_nan=False)+'\n');metrics.flush()
                bank.trim();save()
                if count['iteration']==1:
                    write(out/'video-request.json',dict(schema='photo-map-ppo-video/v1',iteration=1,
                        checkpoint_sha256=digest(checkpoint),rollout_sha256=digest(out/'rollout-000001.jsonl'),
                        description='First actual learner rollout and PPO optimizer update; no scripted flight substitutions',
                        render_command='python -m research.map_navigation.ppo_video --run RUN --iteration 1 --output first-ppo-update.mp4'))
                if count['iteration']%cfg['eval_every']==0:
                    results=[]
                    for i in range(cfg['eval_episodes']):
                        if not window.remaining():break
                        task=validation[i%len(validation)];eobs,ectx,_=episode(task,True);closest=math.inf
                        while True:
                            row,eobs,ectx=rollout_step(task,eobs,ectx,True)
                            closest=min(closest,math.dist(eobs['state']['position'],task['goal']))
                            if row['terminated'] or row['truncated']:break
                        results.append(dict(task_id=task['id'],event=row['event'],closest_goal_m=closest,duration_s=eobs['elapsed_s']))
                        bank.trim()
                    write(out/f'evaluation-{count["iteration"]:06d}.json',dict(complete=len(results)==cfg['eval_episodes'],episodes=results))
                    if len(results)==cfg['eval_episodes']:
                        evaluations.append(dict(iteration=count['iteration'],success_rate=sum(r['event']=='success' for r in results)/len(results),
                            collision_rate=sum(r['event']=='collision' for r in results)/len(results),
                            median_closest_goal_m=float(np.median([r['closest_goal_m'] for r in results]))))
                    save()
            if not window.remaining():status='window_complete'
    except Exception as error:
        status='infrastructure_or_numerical_failure';write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),counts=count))
        if 'rows' in locals() and rows:
            with (out/('interrupted-'+uuid.uuid4().hex+'.jsonl')).open('w') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
        raise
    finally:
        worker.close();journal();save();write(out/'status.json',dict(status=status,counts=count,trained_navigation_accepted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--backbone',required=True)
    p.add_argument('--output',required=True);p.add_argument('--worker',default='127.0.0.1:43651');p.add_argument('--authfile',required=True)
    p.add_argument('--phase',choices=['smoke','million','extension'],default='smoke');p.add_argument('--resume',action='store_true')
    p.add_argument('--hours',type=float,default=8);p.add_argument('--device',default='cuda')
    p.add_argument('--max-updates',type=int,help='Bound an initial integration run; resume preserves the campaign budget')
    p.add_argument('--workspace',help='Shared campaign workspace for all seeds/runs; defaults to output parent')
    train(p.parse_args())
