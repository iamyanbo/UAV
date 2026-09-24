"""Actual-interval recurrent odometry with warm-up and accumulated SE(3) loss."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from learning_models import FastVisualOdometry, rotation_increment
from train_visual_components import recorded_rgb
from trajectory_bundle import digest

OBJECTIVE='metric-motion-rate-balanced-regimes/v4'
WARMUP=32
STEPS=160


@lru_cache(maxsize=4)
def episode(root, runtime_path, runtime_hash, label_path, label_hash):
    paths=[Path(root)/runtime_path,Path(root)/label_path]
    if any(not p.resolve().is_relative_to(Path(root).resolve()) for p in paths):
        raise ValueError('Escaping bundle path')
    if [digest(p) for p in paths]!=[runtime_hash,label_hash]:raise ValueError('Changed trajectory view')
    runtime,labels=[json.loads(p.read_text()) for p in paths]
    return runtime,{r['frame_id']:r for r in labels['frames']}


def windows(root,manifest):
    result={split:[] for split in ('train','validation')}
    for item in manifest['attempts']:
        runtime,labels=episode(str(root),item['runtime']['path'],item['runtime']['sha256'],
                              item['training_labels']['path'],item['training_labels']['sha256'])
        rows=runtime['frames']
        for start in range(0,len(rows)-STEPS,STEPS//2):
            selected=rows[start:start+STEPS+1]
            if (all(labels[r['frame_id']]['motion_valid'] for r in selected) and
                all(0<(b['sim_ns']-a['sim_ns'])/1e9<=.25 for a,b in zip(selected,selected[1:]))):
                result[item['split']].append((item,start))
    if not all(result.values()):raise ValueError('Need disjoint train/development contiguous sequences')
    return result


def sequence(root,item,start,device):
    runtime,labels=episode(str(root),item['runtime']['path'],item['runtime']['sha256'],
                          item['training_labels']['path'],item['training_labels']['sha256'])
    rows=runtime['frames'][start:start+STEPS+1]
    folder=Path(runtime['rgb_stream']).parent.parent
    index={r['frame_id']:r for r in rows}
    images=[recorded_rgb(folder,index,r['frame_id']).unsqueeze(0).to(device) for r in rows]
    motion=[]; dt=[]; commands=[]
    for a,b in zip(rows,rows[1:]):
        before,after=labels[a['frame_id']],labels[b['frame_id']]
        rotation=Rotation.from_quat(before['true_quaternion_xyzw'])
        translation=rotation.inv().apply(np.asarray(after['true_position_ned_m'])-before['true_position_ned_m'])
        angle=(rotation.inv()*Rotation.from_quat(after['true_quaternion_xyzw'])).as_rotvec()
        motion.append(np.r_[translation,angle]);dt.append((b['sim_ns']-a['sim_ns'])/1e9)
        # Only command history published with the earlier image is an input.
        history=a.get('command_history',[])
        commands.append(history[-1]['values'] if history else [0.,0.,0.,0.])
    return images,torch.tensor(np.asarray(motion),dtype=torch.float32,device=device),\
        torch.tensor(dt,dtype=torch.float32,device=device),torch.tensor(commands,dtype=torch.float32,device=device)


def objective(model, data):
    images,truth,dt,commands=data
    hidden=model.temporal.weight_hh.new_zeros(1,256)
    position=truth.new_zeros(1,3);target_position=position.clone()
    orientation=torch.eye(3,device=truth.device)[None];target_orientation=orientation.clone()
    losses=[]; errors=[]; coverage=[];rate_errors=[]
    previous=None
    for step in range(STEPS):
        with torch.set_grad_enabled(model.training and step>=WARMUP):
            if previous is None:previous=model.encode(images[step])
            current=model.encode(images[step+1])
            pred=model.forward_features(previous,current,commands[step:step+1],dt[step:step+1],hidden)
            hidden=pred['hidden'];previous=current
            if step<WARMUP:
                hidden=hidden.detach();previous=previous.detach();continue
            motion=pred['body_motion'].float();std=pred['motion_std'].float()
            error=motion-truth[step:step+1].detach()
            position=position+(orientation@motion[:,:3,None]).squeeze(-1)
            orientation=orientation@rotation_increment(motion[:,3:])
            target_position=target_position+(target_orientation@truth[step:step+1,:3,None]).squeeze(-1)
            target_orientation=target_orientation@rotation_increment(truth[step:step+1,3:])
            # Rates use the actual exposure interval. A direct error term is
            # independent of uncertainty, so widening sigma cannot hide bias.
            physical_scales=truth.new_tensor([3.,3.,1.,.785398,.785398,.785398])
            rate_error=error/dt[step]/physical_scales
            rate_std=std/dt[step]/physical_scales
            nll=(.5*(rate_error/rate_std).square()+rate_std.log()).mean()
            elapsed=dt[WARMUP:step+1].sum().clamp_min(.2)
            drift=F.smooth_l1_loss(position/(3*elapsed),target_position/(3*elapsed))
            losses.append(rate_error.square().mean()+.05*nll+drift+F.smooth_l1_loss(orientation,target_orientation))
            errors.append(error.detach());coverage.append((error.detach().abs()<=1.96*std.detach()).float())
            rate_errors.append(rate_error.detach())
    error=torch.cat(errors)
    metrics=dict(translation_rmse_m=float(error[:,:3].square().mean().sqrt()),
        rotation_rmse_rad=float(error[:,3:].square().mean().sqrt()),
        accumulated_translation_error_m=float((position-target_position).detach().norm()),
        normalized_motion_rate_rmse=float(torch.cat(rate_errors).square().mean().sqrt()),
        marginal_95pct_coverage=float(torch.cat(coverage).mean()),
        evaluated_seconds=float(dt[WARMUP:].sum()))
    return torch.stack(losses).mean(),metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--output',type=Path,default=Path('/output/visual-training'))
    parser.add_argument('--backbone',default='/models/mobilenet-v3-large-imagenet1k-v2.pt')
    parser.add_argument('--resume');parser.add_argument('--initialize-from');parser.add_argument('--updates',type=int,default=100000)
    parser.add_argument('--validation-every',type=int,default=2000);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.resume and args.initialize_from:parser.error('Choose exact resume or a new initialized round')
    root=args.dataset.resolve();manifest=json.loads((root/'manifest.json').read_text());sha=digest(root/'manifest.json')
    if manifest['schema']!='visual-trajectory-bundle/v1':raise ValueError('Versioned trajectories required')
    records=windows(root,manifest)
    # Balance actual stationary, slow exploration and cruising motion. The
    # regime is training-only supervision; no true speed enters inference.
    regimes={name:[] for name in ('stationary','slow','cruise')}
    for item,start in records['train']:
        runtime,labels=episode(str(root),item['runtime']['path'],item['runtime']['sha256'],
                              item['training_labels']['path'],item['training_labels']['sha256'])
        rows=runtime['frames'][start:start+STEPS+1]
        positions=np.asarray([labels[row['frame_id']]['true_position_ned_m'] for row in rows])
        speed=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()/((rows[-1]['sim_ns']-rows[0]['sim_ns'])/1e9))
        regimes['stationary' if speed<.05 else 'slow' if speed<=1 else 'cruise'].append((item,start))
    choices=[name for name,rows in regimes.items() if rows]
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=FastVisualOdometry(args.backbone,motion_parameterization='rate').cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    update=0;best=float('inf');best_state=None;initialization=None;resume_evidence=None
    if args.initialize_from:
        saved=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        if saved.get('module')!='odometry':raise ValueError('Odometry initialization checkpoint required')
        if saved.get('objective_version') in (OBJECTIVE,'metric-long-sequence-motion-rate/v3','metric-sequence-nll-se3/v1','metric-long-sequence-fixed-bn/v2'):
            model.load_state_dict(saved['model']);discarded=[]
        else:
            source={k:v for k,v in saved['model'].items() if not k.startswith('log_scale.')}
            missing,unexpected=model.load_state_dict(source,strict=False)
            if set(missing)!={'motion_uncertainty.weight','motion_uncertainty.bias'} or unexpected:
                raise ValueError('Unexpected legacy odometry initialization mismatch')
            discarded=['log_scale']
        if saved.get('motion_parameterization')!='rate' and saved.get('objective_version') not in (OBJECTIVE,'metric-long-sequence-motion-rate/v3'):
            # Preserve nominal 50 ms mean predictions as initialization only.
            # The nonlinear uncertainty head gets an explicit new broad prior.
            with torch.no_grad():
                model.motion.weight.div_(.05);model.motion.bias.div_(.05)
                model.motion_uncertainty.weight.zero_()
                scales=model.motion_uncertainty.bias.new_tensor([3.,3.,1.,.785398,.785398,.785398])
                model.motion_uncertainty.bias.copy_(torch.expm1(scales-.001).log())
            discarded.append('increment_uncertainty')
        initialization=dict(path=args.initialize_from,sha256=digest(args.initialize_from),source_update=saved['update'],
                            optimizer_reset=True,discarded_heads=discarded,
                            nominal_increment_to_rate_seconds=.05 if 'increment_uncertainty' in discarded else None)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['manifest_sha256']!=sha or saved['objective_version']!=OBJECTIVE:raise ValueError('New data/objective requires new round')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        update=saved['update'];best=saved['best'];rng.bit_generator.state=saved['sample_rng']
        best_state=saved.get('best_state')
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        initialization=saved['initialization']
        steps=[float(v['step']) for v in optimizer.state.values() if 'step' in v]
        resume_evidence=dict(checkpoint_sha256=digest(args.resume),restored_update=update,
            optimizer_states=len(optimizer.state),optimizer_step_min=min(steps) if steps else None,
            optimizer_step_max=max(steps) if steps else None)
        if not steps or any(step!=update for step in steps):
            raise ValueError('Optimizer moments/step counters did not restore consistently')
    args.output.mkdir(parents=True,exist_ok=False)
    if best_state is not None:torch.save(best_state,args.output/'best.pt')
    resume_counter=update
    def checkpoint(name):
        nonlocal best_state
        state=dict(module='odometry',objective_version=OBJECTIVE,manifest_sha256=sha,model=model.state_dict(),
            warmup_steps=WARMUP,sequence_steps=STEPS,motion_parameterization='rate',
            optimizer=optimizer.state_dict(),update=update,best=best,sample_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),initialization=initialization)
        if name=='best':best_state=copy.deepcopy(state)
        else:state['best_state']=best_state
        pending=args.output/(name+'.pending');torch.save(state,pending);pending.replace(args.output/(name+'.pt'))
    started=time.monotonic();gradient_evidence=None
    while update<args.updates and not Path('/output/CHECKPOINT_REQUEST').exists():
        model.train();optimizer.zero_grad(set_to_none=True)
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
        regime=choices[int(rng.integers(len(choices)))];pool=regimes[regime]
        item,start=pool[int(rng.integers(len(pool)))]
        before={n:p.detach().clone() for n,p in model.named_parameters()} if gradient_evidence is None else None
        loss,metrics=objective(model,sequence(root,item,start,'cuda'))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite recurrent odometry objective')
        loss.backward()
        gradients={name:bool(p.grad is not None and torch.isfinite(p.grad).all()) for name,p in model.named_parameters()}
        if not all(gradients.values()):raise RuntimeError('Missing/nonfinite gradient in odometry parameters')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();update+=1
        if before is not None:
            groups=('encoder','temporal','motion','motion_uncertainty')
            gradient_evidence={group:dict(finite_gradients=all(v for n,v in gradients.items() if n.startswith(group+'.')),
                changed=any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n.startswith(group+'.')))
                for group in groups}
            if not all(v['changed'] for v in gradient_evidence.values()):raise RuntimeError('Intended parameter group did not change')
        validation=None
        if update%args.validation_every==0 or update==args.updates:
            model.eval();scores=[];details=[]
            # Fixed, evenly spaced development subset for engineering updates;
            # full-episode drift acceptance is separately required.
            selected=np.unique(np.linspace(0,len(records['validation'])-1,min(22,len(records['validation']))).astype(int))
            with torch.no_grad():
                for index in selected:
                    item,start=records['validation'][int(index)]
                    value,detail=objective(model,sequence(root,item,start,'cuda'));scores.append(float(value));details.append(detail)
            validation=dict(loss=float(np.mean(scores)),windows=len(scores),
                **{k:float(np.mean([r[k] for r in details])) for k in details[0]})
            # Checkpoint choice also cannot improve by widening uncertainty.
            selection=validation['normalized_motion_rate_rmse']+validation['accumulated_translation_error_m']/(3*validation['evaluated_seconds'])
            validation['selection_score']=selection
            if selection<best:best=selection;checkpoint('best')
        if update%50==0:checkpoint('latest')
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=update,loss=float(loss.detach()),validation=validation,sampling_regime=regime,**metrics))+'\n')
    checkpoint('latest')
    result=dict(status='completed' if update==args.updates else 'checkpointed',accepted=False,updates=update,
        resumed_from_update=resume_counter,objective_version=OBJECTIVE,manifest_sha256=sha,
        resume_evidence=resume_evidence,
        gradient_evidence=gradient_evidence,train_windows=len(records['train']),development_windows=len(records['validation']),
        elapsed_seconds=time.monotonic()-started,reason='Full held-out trajectory drift and uncertainty qualification still required')
    result.update(sampling_regimes={k:len(v) for k,v in regimes.items()},sequence_steps=STEPS,warmup_steps=WARMUP,normalization='fixed inference BatchNorm statistics',
                  activation_checkpointing=True,motion_parameterization='rate',
                  checkpoint_selection='development normalized motion-rate error plus normalized accumulated drift')
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
