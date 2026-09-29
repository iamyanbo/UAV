"""Local rollout ownership; Spark receives immutable, complete PPO batches only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tarfile
import time
import uuid
import numpy as np
import torch
from PIL import Image
from .common import read, write, digest, FlightLock
from .native import configure, brake
from .ppo_env import PilotEnvironment
from .ppo_core import ActorCritic, command_from_latent, potential, transition_reward, update
from .ppo_scheduler import FeatureBank, Inference, batch_for
from .compute import ComputeLane
from .ppo_budget import Budget


def frozen_digest(model):
    h=hashlib.sha256()
    for k,v in model.encoder.backbone.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimize(bundle,backbone,output):
    """Runs inside the existing native ARM PyTorch image on Spark."""
    torch.set_num_threads(4)
    data=torch.load(bundle,map_location='cpu',weights_only=False)
    checkpoint=data['checkpoint'];cfg=data['config'];rows=data['rows']
    np.random.set_state(checkpoint['numpy_rng'])
    if len(rows)!=8192:raise ValueError('Exactly 8192 fresh rows required')
    if {r['policy_sha256'] for r in rows}!={data['policy_sha256']}:raise ValueError('Mixed policies')
    model=ActorCritic(backbone).cuda();model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    before=frozen_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    bank=FeatureBank();bank.frames=data['features'];banks={0:bank}
    def batch(ids):return batch_for([rows[int(i)] for i in ids],banks,'cuda')
    # Cross-device parity must hold before accepting on-policy ratios.
    errors=[]
    with torch.no_grad():
        for begin in range(0,len(rows),cfg['minibatch']):
            ids=np.arange(begin,min(begin+cfg['minibatch'],len(rows)))
            logp,_,_,_=model.evaluate(batch(ids))
            errors.extend((logp.cpu()-torch.tensor([rows[i]['logprob'] for i in ids])).abs().tolist())
    if max(errors)>.01:raise RuntimeError('Windows/Spark log-probability parity failed')
    started=time.perf_counter();report=update(model,optimizer,rows,batch,cfg)
    if not report['optimizer_steps'] or report['final_rollout_kl']>.1:
        raise RuntimeError('PPO candidate rejected; no accepted update')
    after=frozen_digest(model)
    if before!=after:raise RuntimeError('Frozen backbone changed')
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),checkpoint['model'][k].cpu())]
    if not changed:raise RuntimeError('No model parameters changed')
    report.update(optimization_wall_s=time.perf_counter()-started,changed_tensors=changed,
        frozen_backbone_sha256=after,pre_update_logprob_max_error=max(errors),
        policy_sha256=data['policy_sha256'],rollout_sha256=digest(bundle),
        training_proof=True,trained_navigation_accepted=False)
    report['action_saturation_fraction']=float(np.mean(np.abs(np.tanh([r['latent'] for r in rows]))>.95))
    checkpoint.update(model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
        counts=dict(iteration=checkpoint['counts']['iteration']+1,transitions=checkpoint['counts']['transitions']+len(rows)),
        native_report=report,experiment='native-city-mode1-qualification',mode2_enabled=False)
    checkpoint['numpy_rng']=np.random.get_state()
    torch.save(checkpoint,output)
    write(Path(output).with_suffix('.json'),report)
    # Verify the artifact, not only the live model.
    restored=torch.load(output,map_location='cpu',weights_only=False)
    model.load_state_dict(restored['model'],strict=True)
    print(json.dumps(report),flush=True)


def transfer_update(args,root,bundle,iteration):
    remote='/home/iamyanbo/photo-goal-native'
    ssh=['ssh','-i',args.ssh_key,'-o','BatchMode=yes',args.spark]
    scp=['scp','-q','-i',args.ssh_key]
    subprocess.run(ssh+[f'mkdir -p {remote}'],check=True)
    source=root/'source.tar'
    with tarfile.open(source,'w') as archive:
        for p in Path(__file__).parent.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                archive.add(p,arcname='photo_goal/'+str(p.relative_to(Path(__file__).parent)))
    for p,name in [(source,'source.tar'),(Path(args.backbone),'backbone.pt'),(bundle,'batch.pt')]:
        subprocess.run(scp+[str(p),args.spark+':'+remote+'/'+name],check=True)
    subprocess.run(ssh+[f'cd {remote} && tar xf source.tar && docker run --rm --gpus all --ipc=host '
        f'--entrypoint python -v {remote}:/work -w /work rgb-flight-models:25.11-native '
        '-m photo_goal.native_training --bundle /work/batch.pt --backbone /work/backbone.pt --output /work/updated.pt'],check=True)
    out=root/f'update-{iteration:06d}.pt'
    subprocess.run(scp+[args.spark+':'+remote+'/updated.pt',str(out)],check=True)
    subprocess.run(scp+[args.spark+':'+remote+'/updated.json',str(out.with_suffix('.json'))],check=True)
    return out


def run(args):
    root=Path(args.root).resolve();scene,cfg=configure(root)
    qualification=read(root/'qualification.json')
    if not qualification.get('passed') or qualification['scene_sha256']!=digest(root/'scene.json') or qualification['config_sha256']!=digest(root/'config.json'):
        raise RuntimeError('Matching native simulator qualification is required')
    if qualification['task_manifest_sha256']!=digest(root/'tasks.json'):raise RuntimeError('Tasks changed since qualification')
    evaluation=args.command=='evaluate'
    if not evaluation and args.updates!=2:raise ValueError('This qualification milestone admits exactly two updates')
    run_root=root/('evaluation-' if evaluation else 'training-')/time.strftime('%Y%m%dT%H%M%S')
    run_root.mkdir(parents=True,exist_ok=False)
    tasks=read(root/'tasks.json')['tasks'];checkpoint_path=Path(args.checkpoint).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=ActorCritic(args.backbone).cuda();model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    torch.set_num_threads(2)
    random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    status=dict(status='running',source_checkpoint=str(checkpoint_path),source_sha256=digest(checkpoint_path),
        prior_counts=checkpoint['counts'],new_transitions=0,new_updates=0,mode2_enabled=False,
        trained_navigation_accepted=False,episodes=[])
    deadline=time.perf_counter()+8*3600
    write(run_root/'status.json',status)
    budget=None;scheduler=None;batch_id=None;rows=[]
    with FlightLock(root,'native-photo-goal'):
        try:
            if not evaluation:
                # The imported campaign ledger is mandatory and never reset by this entry point.
                budget=Budget(root/'campaign',cfg)
            bank=FeatureBank();banks={0:bank};scheduler=Inference(model,ComputeLane(),banks,None,wait_s=0.)
            with PilotEnvironment(scene,run_root/'worker',cfg) as env:
                env.phase='evaluation' if evaluation else 'training';env.calibrate()
                for iteration in range(1 if evaluation else args.updates):
                    rows=[];policy_sha=digest(checkpoint_path)
                    if budget:
                        batch_id=uuid.uuid4().hex
                        if not budget.reserve_batch(batch_id,cfg['smoke_transitions']):raise RuntimeError('Campaign transition ceiling reached')
                    while len(rows)<8192:
                        if time.perf_counter()>deadline:raise RuntimeError('Eight-hour execution window expired')
                        task=tasks[len(status['episodes'])%len(tasks)]
                        attempt_id='episode-'+uuid.uuid4().hex[:12]
                        if budget:budget.attempt(attempt_id,0)
                        goal=Image.open(task['goal_image']).convert('RGB')
                        scheduler.call('initialize',worker=0,image=goal,path=task['goal_image'])
                        # Warm the real model/goal before handing over live controls.
                        dummy=dict(rgb=goal.tobytes(),rgb_path=task['goal_image'],sim_s=0.,preceding_command=[0.]*4)
                        scheduler.call('decision',worker=0,obs=dummy,execution=False)
                        bank.history.clear();bank.memory.frames.clear()
                        obs=env.reset(task,attempt_id);decision=scheduler.call('decision',worker=0,obs=obs,execution=False)
                        steps=0;episode_reward=0.;event=None
                        while len(rows)<8192:
                            if time.perf_counter()>deadline:raise RuntimeError('Eight-hour execution window expired')
                            command=command_from_latent(decision['latent'],obs['preceding_command'],cfg['step_s'],cfg['limits'],cfg['acceleration'])
                            if budget:budget.transition(batch_id,0)
                            result=env.step(command,bool(decision['stop']),obs['frame'])
                            if budget:budget.confirm(batch_id,0)
                            next_obs=result['observation'];next_decision=scheduler.call('decision',worker=0,obs=next_obs,execution=False)
                            reward,parts,gamma=transition_reward(potential(obs['state']['position'],task['goal'],task['distance_m']),
                                potential(next_obs['state']['position'],task['goal'],task['distance_m']),result['dt'],result['event'],cfg)
                            row=dict(decision['context'],latent=decision['latent'],stop=decision['stop'],logprob=decision['logprob'],
                                value=decision['value'],next_value=next_decision['value'],reward=reward,dt=result['dt'],
                                terminated=result['terminated'],truncated=result['truncated'],attempt_id=attempt_id,
                                policy_iteration=checkpoint['counts']['iteration'],policy_sha256=policy_sha,
                                proposed_command=command,decision_s=decision['decision_s'],event=result['event'])
                            rows.append(row);steps+=1;episode_reward+=reward;status['new_transitions']+=1
                            obs=next_obs;decision=next_decision;event=result['event']
                            if result['terminated'] or result['truncated']:break
                        if not rows[-1]['terminated']:
                            rows[-1]['truncated']=True;rows[-1]['rollout_boundary']=True
                        env.done=True;brake(env)
                        if env.recorder:env.recorder.flush()
                        status['episodes'].append(dict(id=attempt_id,task_id=task['id'],steps=steps,reward=episode_reward,event=event or 'truncated'))
                        if len(status['episodes'])==1:
                            from .video import render
                            status['video']=render(env.recorder.root,task,run_root/'first-policy-flight.mp4',
                                'Learned policy: '+str(event or 'truncated'),policy_sha)
                        write(run_root/'status.json',status)
                        if evaluation:break
                    if evaluation:break
                    bundle=run_root/f'rollout-{iteration:03d}.pt'
                    checkpoint.update(python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    torch.save(dict(checkpoint=checkpoint,config=cfg,rows=rows,features=bank.frames,policy_sha256=policy_sha),bundle)
                    checkpoint_path=transfer_update(args,run_root,bundle,checkpoint['counts']['iteration']+1)
                    updated=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                    before=frozen_digest(model);model.load_state_dict(updated['model'],strict=True)
                    if frozen_digest(model)!=before:raise RuntimeError('Returned checkpoint changes frozen backbone')
                    model.eval();checkpoint=updated;budget.finish(batch_id,True);batch_id=None
                    np.random.set_state(checkpoint['numpy_rng'])
                    status['new_updates']+=1;status['checkpoint']=str(checkpoint_path)
                    write(run_root/'status.json',status);bank.trim()
                status['status']='complete'
        except BaseException as error:
            status.update(status='failed',failure=type(error).__name__+': '+str(error))
            if rows:torch.save(rows,run_root/'interrupted-rows.pt')
            raise
        finally:
            if scheduler:scheduler.close()
            if budget:
                if batch_id:budget.finish(batch_id,False)
                status['budget']=budget.snapshot();budget.close()
            write(run_root/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--backbone',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();optimize(a.bundle,a.backbone,a.output)
