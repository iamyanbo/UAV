"""Resumable training for fast visual odometry and the cross-view goal matcher."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from goal_matching import CrossViewGoalMatcher, GoalMatcherPipeline
from goal_io import load_goal
from episode_store import RGB_BYTES, verified_rgb_storage
from learning_models import FastVisualOdometry

CHECKPOINT_INTERVAL=50
OBJECTIVE_VERSION='goal-recognition-fixed-bn-prevalence/v4'


def load_manifest(root,module,filename='visual-training.json'):
    path=(root/filename).resolve()
    if not path.is_relative_to(root):raise ValueError('Escaping visual manifest')
    raw=path.read_bytes(); manifest=json.loads(raw)
    if manifest['valid_expert_episodes']<250:
        raise ValueError('Visual training requires 250 valid expert episodes')
    episodes={row['episode_id']:row['split'] for row in manifest['episodes']}
    records={split:[row for row in manifest['windows'] if row['module']==module and episodes[row['episode_id']]==split]
             for split in ('train','validation')}
    if not all(records.values()):
        raise ValueError('Training and held-out visual windows are required')
    return manifest,records,hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=8)
def indexed_episode(root_string, relative, expected_goal_hash):
    root=Path(root_string)
    episode=(root/relative).resolve()
    if not episode.is_relative_to(root):
        raise ValueError('Escaping episode reference')
    receipt=verified_rgb_storage(episode/'observations')
    index={row['frame_id']:row for row in map(json.loads,(episode/'observations/frames.jsonl').read_text().splitlines())}
    goal=load_goal(episode/'goal')
    if goal.content_sha256!=expected_goal_hash or len(index)!=receipt['frames']:
        raise ValueError('Changed or incomplete visual episode')
    goal_rgb=torch.from_numpy(np.stack([np.frombuffer(view,np.uint8).reshape(480,640,3).copy()
                                        for view in goal.rgb_views])).permute(0,3,1,2)
    return episode,index,goal_rgb


def recorded_rgb(episode,index,frame_id):
    row=index[frame_id]
    with (episode/'observations/rgb.zlib').open('rb') as stream:
        stream.seek(row['offset'])
        length=struct.unpack('!I',stream.read(4))[0]
        if length!=row['compressed_bytes'] or length>RGB_BYTES+1024:
            raise ValueError('Corrupt recorded RGB packet')
        raw=zlib.decompress(stream.read(length))
    if len(raw)!=RGB_BYTES or hashlib.sha256(raw).hexdigest()!=row['rgb_sha256']:
        raise ValueError('Recorded RGB checksum mismatch')
    return torch.from_numpy(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()).permute(2,0,1)


def load_record(root,row,goal_hashes):
    if 'episode_path' in row:
        episode,index,goal_rgb=indexed_episode(str(root),row['episode_path'],goal_hashes[row['episode_id']])
        current=recorded_rgb(episode,index,row['frame_id'])
        if row['module']=='goal':
            runtime=dict(current_rgb=current,goal_rgb=goal_rgb)
        else:
            runtime=dict(previous_rgb=recorded_rgb(episode,index,row['previous_frame_id']),
                         current_rgb=current,previous_command=torch.tensor(row['previous_command'],dtype=torch.float32),
                         delta_seconds=torch.tensor(row['delta_seconds'],dtype=torch.float32))
        labels={key:torch.tensor(value,dtype=torch.float32 if key not in ('near_goal','match_valid','scale_valid') else torch.bool)
                for key,value in row['training_labels'].items()}
        return dict(episode_id=row['episode_id'],runtime=runtime,training_labels=labels)
    path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Modified visual training window')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['episode_id']!=row['episode_id'] or set(value)-{'episode_id','runtime','training_labels'}:
        raise ValueError('Invalid separated visual window')
    return value


def batch(root,rows,indices,goal_hashes):
    values=[load_record(root,rows[int(index)],goal_hashes) for index in indices]
    runtime={key:torch.stack([row['runtime'][key] for row in values]).cuda() for key in values[0]['runtime']}
    labels={key:torch.stack([row['training_labels'][key] for row in values]).cuda() for key in values[0]['training_labels']}
    return runtime,labels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',choices=('odometry','goal'),required=True)
    parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--output',type=Path,default=Path('/output/visual-training'))
    parser.add_argument('--resume')
    parser.add_argument('--initialize-from')
    parser.add_argument('--validation-every',type=int,default=2000)
    parser.add_argument('--manifest',default='visual-training.json')
    parser.add_argument('--updates',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.module=='odometry':
        raise ValueError('Pairwise odometry objective retired; use train_odometry_sequences.py with a trajectory bundle')
    root=args.dataset.resolve(); manifest,records,manifest_hash=load_manifest(root,args.module,args.manifest)
    if manifest.get('schema')!='visual-goal-supervision/v3':
        raise ValueError('New training requires timestamp-aligned goal supervision v3; legacy artifacts remain evaluation-only')
    data_root=Path(manifest['collection_root']).resolve()
    if args.resume and args.initialize_from:raise ValueError('Choose resume or a new initialized round')
    goal_hashes={row['episode_id']:row['goal_sha256'] for row in manifest['episodes']}
    if args.module=='goal':
        classes={value:[index for index,row in enumerate(records['train'])
                        if bool(row['training_labels']['near_goal'])==value]
                 for value in (False,True)}
        if not all(classes.values()):
            raise ValueError('Goal training needs both near-goal and separated negatives')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    model=(FastVisualOdometry('/models/mobilenet-v3-large-imagenet1k-v2.pt') if args.module=='odometry'
           else GoalMatcherPipeline('/models/mobilenet-v3-large-imagenet1k-v2.pt')).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); update=0;best=float('inf');best_state=None
    initialization=None;resume_evidence=None
    if args.initialize_from:
        saved=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        if saved['module']!='goal':raise ValueError('Goal initialization checkpoint required')
        model.load_state_dict(saved['model'])
        initialization=dict(sha256=hashlib.sha256(Path(args.initialize_from).read_bytes()).hexdigest(),
                            update=saved['update'],optimizer_reset=True)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['manifest_sha256']!=manifest_hash or saved['module']!=args.module or saved.get('objective_version')!=OBJECTIVE_VERSION:
            raise ValueError('Resume data/module changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);update=saved['update'];best=saved['best']
        rng.bit_generator.state=saved['sample_rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        initialization=saved['initialization']
        best_state=saved.get('best_state')
        if best_state is not None:torch.save(best_state,args.output/'best.pt')
        steps=[float(value['step']) for value in optimizer.state.values() if 'step' in value]
        if not steps or any(step!=update for step in steps):raise ValueError('Goal optimizer step counters did not restore')
        resume_evidence=dict(checkpoint_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest(),
            restored_update=update,optimizer_states=len(optimizer.state),optimizer_step_min=min(steps),optimizer_step_max=max(steps))
        (args.output/'resume.json').write_text(json.dumps(resume_evidence,indent=2))
    training_prevalence=len(classes[True])/sum(len(v) for v in classes.values())
    def objective(runtime,labels):
        pred=model(runtime['current_rgb'],runtime['goal_rgb'])
        valid=labels['match_valid'].bool()
        if not bool(valid.any()):
            raise ValueError('Goal batch has no near or spatially separated examples')
        target=labels['near_goal'][valid].float()
        terms=F.binary_cross_entropy_with_logits(pred['match_logit'][valid],target,reduction='none')
        # One positive and one negative are sampled for gradient coverage.
        # Correct that sampling prior instead of training 50% match prevalence.
        weights=2*torch.where(target.bool(),training_prevalence,1-training_prevalence) if model.training else torch.ones_like(target)
        recognition=(terms*weights).mean()
        time_valid=labels['time_valid'].bool()&valid;value_valid=labels['value_valid'].bool()&valid
        time_loss=F.smooth_l1_loss(pred['time_to_goal_seconds'][time_valid]/60,
                    labels['time_to_goal_seconds'][time_valid].detach()/60) if time_valid.any() else recognition*0
        value_loss=F.smooth_l1_loss(pred['terminal_value'][value_valid],labels['terminal_return'][value_valid].detach()) if value_valid.any() else recognition*0
        return recognition+time_loss+.1*value_loss,pred
    def checkpoint(name):
        nonlocal best_state
        value=dict(module=args.module,model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,best=best,
                   objective_version=OBJECTIVE_VERSION,initialization=initialization,
                   manifest_sha256=manifest_hash,sample_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all())
        if name=='best':best_state=copy.deepcopy(value)
        else:value['best_state']=best_state
        temporary=args.output/(name+'.pending');torch.save(value,temporary);temporary.replace(args.output/(name+'.pt'))
    def evaluate():
            model.eval();values=[];probabilities=[];truth=[];time_errors=[]
            with torch.no_grad():
                for offset in range(0,len(records['validation']),batch_size):
                    runtime,labels=batch(data_root,records['validation'],range(offset,min(offset+batch_size,len(records['validation']))),goal_hashes)
                    value,pred=objective(runtime,labels);values.append((float(value),len(runtime['current_rgb'])))
                    probabilities.extend(pred['match_logit'].sigmoid().cpu().tolist());truth.extend(labels['near_goal'].cpu().tolist())
                    mask=labels['time_valid'].bool();time_errors.extend((pred['time_to_goal_seconds'][mask]-labels['time_to_goal_seconds'][mask]).abs().cpu().tolist())
            p=np.asarray(probabilities);y=np.asarray(truth,dtype=bool);guess=p>=.5
            tp=int((guess&y).sum());fp=int((guess&~y).sum());fn=int((~guess&y).sum());tn=int((~guess&~y).sum())
            # Selection uses recognition quality independently of seconds/value
            # losses. All metrics remain visible and no quality acceptance is inferred.
            score=float(np.mean((p-y)**2))
            validation=dict(loss=sum(v*n for v,n in values)/sum(n for _,n in values),brier_score=score,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,time_mae_seconds=float(np.mean(time_errors)) if time_errors else None)
            return score,validation
    started=time.monotonic();batch_size=2;gradient_evidence=None
    if not args.resume:
        best,initial_validation=evaluate();checkpoint('best')
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=0,train_loss=None,validation_loss=initial_validation,
                                         elapsed_seconds=time.monotonic()-started))+'\n')
    while update<args.updates and not Path('/output/CHECKPOINT_REQUEST').exists():
        model.train()
        # Deployment encodes one current frame and caches four goal views.
        # Tiny, separately encoded training batches must use those same fixed
        # normalization statistics. Affine and convolution weights still learn.
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
        optimizer.zero_grad(set_to_none=True)
        indices=[int(rng.choice(classes[False])),int(rng.choice(classes[True]))]
        runtime,labels=batch(data_root,records['train'],indices,goal_hashes)
        before={n:p.detach().clone() for n,p in model.named_parameters()} if gradient_evidence is None else None
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=objective(runtime,labels)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite visual-component objective')
        loss.backward()
        finite={n:p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in model.named_parameters()}
        if not all(finite.values()):raise RuntimeError('Missing or nonfinite goal gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();update+=1
        if before is not None:
            gradient_evidence={group:dict(finite_gradients=all(v for n,v in finite.items() if n.startswith(group+'.')),
                changed=any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n.startswith(group+'.')))
                for group in ('encoder','matcher')}
            if not all(v['changed'] for v in gradient_evidence.values()):raise RuntimeError('Intended goal parameter group did not change')
        validation=None
        if update%args.validation_every==0 or update==args.updates:
            score,validation=evaluate()
            if score<best:best=score;checkpoint('best')
        if update%CHECKPOINT_INTERVAL==0:
            checkpoint('latest')
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=update,train_loss=float(loss.detach()),validation_loss=validation,
                                         elapsed_seconds=time.monotonic()-started))+'\n')
    checkpoint('final' if update==args.updates else 'latest')
    result=dict(status='completed' if update==args.updates else 'checkpointed',accepted=False,module=args.module,updates=update,
                manifest_sha256=manifest_hash,objective_version=OBJECTIVE_VERSION,
                gradient_evidence=gradient_evidence,training_prevalence=training_prevalence,
                checkpoint_selection='development recognition Brier score',initialization=initialization,resume_evidence=resume_evidence)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    main()
