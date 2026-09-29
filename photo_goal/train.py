"""Resumable stage-specific training. Never claims acceptance from update count."""
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from .common import SCHEMA,config,digest,write,seed_all,reserve_memory
from .data import Dataset
from .models import PhotoNavigationModel
from .contracts import HORIZON
from .provenance import identities


def objective(model,items,stage,device,horizon_limit=HORIZON):
    if stage not in ('localization','goal'):
        from .learning import objective as temporal_objective
        return temporal_objective(model,items,stage,device,horizon_limit)
    current=torch.stack([r['current'] for r in items]).to(device)
    z=model.encode(current,[r['camera_pitch_deg'] for r in items])
    labels=[r['labels'] for r in items]
    target=lambda name:torch.tensor([row[name] for row in labels],device=device,dtype=torch.float32)
    if stage=='localization':
        tiles=torch.stack([r['retrieval_tiles'] for r in items]).to(device)
        bank=model.encode(tiles.flatten(0,1)).reshape(len(items),5,300,256)
        descriptors=model.descriptors(bank,True);query=model.descriptors(z)
        logits=(query[:,None]*descriptors).sum(-1)/.07
        positive=torch.stack([r['retrieval_positive'] for r in items]).to(device)
        usable=target('localization_usable').bool()
        retrieval=-(torch.logsumexp(logits.masked_fill(~positive,-torch.inf),-1)-torch.logsumexp(logits,-1))
        registration=model.register(z,bank[:,0]);bad=model.register(z,bank[:,2:].mean(1))
        loss=(retrieval*usable).sum()/usable.sum().clamp_min(1)
        delta=registration['offset']-target('offset');sigma=registration['sigma']
        regression=delta.square().sum(-1)/(2*sigma.square())+2*sigma.log()
        regression+=.1*F.smooth_l1_loss(registration['above_surface'],target('above_surface'),reduction='none')
        regression+=F.mse_loss(registration['yaw'],target('yaw'),reduction='none').mean(-1)
        loss+=(regression*usable).sum()/usable.sum().clamp_min(1)
        loss+=F.binary_cross_entropy_with_logits(registration['match_logit'],usable.float())
        loss+=F.binary_cross_entropy_with_logits(bad['match_logit'],torch.zeros(len(items),device=device))
        return loss
    goals=torch.stack([row['goals'][0] for row in items]).to(device)
    goal_tokens=model.encode(goals)[:,None]
    evidence=model.goal_evidence(z,goal_tokens)
    return F.binary_cross_entropy_with_logits(evidence['match_logit'],target('near_goal'))+F.binary_cross_entropy_with_logits(evidence['arrival_logit'],target('arrival'))

def train(dataset,stage,backbone,output,window,updates=None,resume=None,initialize=None,teacher_root=None,seed=0,fine_tune=False):
    if resume and initialize:raise ValueError('Choose resume OR initialize from a preceding stage')
    if (Path(output)/'latest.pt').exists() and not resume:
        raise ValueError('Checkpoint exists; explicitly resume or choose a new output directory')
    if stage in ('configurator','preferences','ppo'):
        from .staged import train_external
        return train_external(dataset,stage,backbone,output,window,updates,resume,initialize,seed)
    cfg=config()['training']
    ceiling=cfg['budgets']['world_updates'] if stage=='world' else cfg['budgets']['imitation_updates'] if stage in ('policy','dagger') else cfg['updates_per_stage']
    # Leave room inside the full-pipeline ceilings for recovery and branch
    # fine tuning; a default initial stage must not exhaust the entire budget.
    allocation={'policy':150000,'dagger':50000,'world':250000}.get(stage,ceiling)
    updates=updates or (cfg['fine_tune_updates'] if fine_tune else min(allocation,ceiling))
    if fine_tune and (stage!='world' or not (initialize or resume)):raise ValueError('Fine tuning requires an initialized world stage')
    if updates>ceiling:raise ValueError('Explicit campaign revision required to extend the budget')
    seed_all(seed);reserve_memory()
    training=Dataset(dataset,stage,'train',teacher_root);validation=Dataset(dataset,stage,'validation',teacher_root)
    if fine_tune and {training.spec['episodes'][w['episode']]['collection_source']=='learner' for w in training.windows}!={False,True}:raise ValueError('Fine tuning requires both original and learner-round windows')
    if training.teacher_identity!=validation.teacher_identity:raise ValueError('Train/validation teacher identity mismatch')
    model=PhotoNavigationModel(backbone).cuda();trained=[];start=0
    data_identity=digest(dataset);source_identity=digest(backbone)
    saved=None
    if resume or initialize:
        saved=torch.load(resume or initialize,map_location='cpu',weights_only=True)
        if saved['schema']!=SCHEMA or saved['backbone_sha256']!=source_identity:raise ValueError('Incompatible checkpoint')
        model.load_state_dict(saved['model'],strict=True);trained=list(saved['trained_stages'])
    budget_key='world_updates' if stage=='world' else 'imitation_updates' if stage in ('policy','dagger') else stage+'_updates'
    usage=dict(saved.get('budget_usage',{})) if saved else {}
    base_used=usage.get(budget_key,0)-(saved['updates'] if resume else 0)
    if base_used+updates>ceiling:raise ValueError('Cumulative stage budget would be exceeded; choose a remaining-budget update count')
    dependencies={'localization':set(),'goal':{'localization'},'odometry':{'localization','goal'},'policy':{'localization','goal','odometry'},'dagger':{'policy','world'},'world':{'localization','goal','policy'}}
    required=dependencies[stage]-({'world'} if resume and stage=='dagger' else set())
    if not required<=set(trained):raise ValueError('Missing prior trained stages: '+str(required-set(trained)))
    if stage in ('policy','dagger','world','localization','goal'):
        trained=[s for s in trained if s!='world']
    invalidate={'localization':{'goal','odometry','policy','dagger','ppo'},'goal':{'policy','dagger','ppo'},
                'policy':{'dagger','ppo'}}.get(stage,set())
    trained=[s for s in trained if s not in invalidate]
    if stage=='world':
        model.assessment_actor_identity=identities(model.state_dict())['actor']
        for data in (training,validation):
            if not any(w.get('behavior_actor_identity')==model.assessment_actor_identity for w in data.windows):
                raise ValueError('Collect train and validation outcomes from this actor before world/value training')
    # Later stages cannot silently invalidate localization by modifying its encoder.
    trainable={'localization':('encoder.','camera_projection.','map_projection.','registration.'),
               'goal':('goal.','arrival.'),
               'world':('world.',),'policy':('policy.','subgoal_encoder.','reference_geometry.'),'dagger':('policy.',),'odometry':('motion.',)}[stage]
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(trainable))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    if resume:
        if saved['stage']!=stage or saved['dataset_sha256']!=data_identity or saved['seed']!=seed or saved.get('target_updates')!=updates or saved.get('fine_tune',False)!=fine_tune or saved.get('teacher_sha256')!=training.teacher_identity:
            raise ValueError('Resume requires identical stage, seed and dataset')
        optimizer.load_state_dict(saved['optimizer']);start=saved['updates']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    last_validation=None;best_validation=float("inf")
    validation_rng=random.Random(731)
    validation_indices=[validation.sample_index(validation_rng) for _ in range(min(cfg["validation_windows"],len(validation)))]
    if resume:best_validation=saved.get("best_validation",float("inf"))
    def checkpoint(update,is_best=False):
        stages=sorted(set(trained)|({stage} if update>=updates else set()))
        value=dict(teacher_sha256=training.teacher_identity,calibration=None,schema=SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),stage=stage,updates=update,
            trained_stages=stages,seed=seed,dataset_sha256=data_identity,backbone_sha256=source_identity,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            validation_loss=last_validation,accepted=False,target_updates=updates,fine_tune=fine_tune,best_validation=best_validation)
        value['budget_usage']=dict(usage,**{budget_key:base_used+update})
        value['world_actor_identity']=identities(value['model'])['actor'] if stage=='world' else (saved or {}).get('world_actor_identity')
        value['data_quality']=training.spec.get('data_quality',{})
        temporary=output/'latest.pending';torch.save(value,temporary);temporary.replace(output/'latest.pt')
        if is_best:
            temporary=output/'best.pending';torch.save(value,temporary);temporary.replace(output/'best.pt')
        if update in (1000,10000,50000,100000,200000,300000):
            torch.save(value,output/f'checkpoint-{stage}-{update:06d}.pt')
        if update>=updates and (output/'best.pt').exists():
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
            best['trained_stages']=stages;best['stage_budget_completed']=True
            temporary=output/'best.pending';torch.save(best,temporary);temporary.replace(output/'best.pt')
        write(output/'receipt.json',dict(stage=stage,updates=update,target=updates,complete=update>=updates,
              checkpoint_sha256=digest(output/'latest.pt'),accepted=False,validation_loss=last_validation))
    with (output/'loss.jsonl').open('a') as log:
        for update in range(start,updates):
            if not window.remaining():checkpoint(update);return
            reserve_memory();model.eval()
            for name,module in model.named_children():
                if any((name+'.').startswith(prefix) for prefix in trainable):module.train()
            # objective augmentation is explicit even with frozen modules in eval.
            model.training=True
            indices=[training.sample_index(random,learner_mix=fine_tune or stage=='dagger') for _ in range(cfg['batch_size'])]
            horizon=80 if fine_tune or (update+1)/updates>.5 else 20 if (update+1)/updates<=.2 else 40
            optimizer.zero_grad(set_to_none=True)
            items=[training.get(i) for i in indices]
            loss=objective(model,items,stage,'cuda',horizon)
            if stage=='world' and fine_tune:
                from .learning import objective as temporal_objective
                loss=loss+temporal_objective(model,items,stage,'cuda',horizon,actor_rollouts=True)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; preserve previous checkpoint')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
            log.write(__import__('json').dumps(dict(update=update+1,loss=float(loss.detach())))+'\n')
            is_best=False
            if (update+1)%cfg['validate_every']==0 or update+1==updates:
                model.eval()
                # Validation always uses the full horizon, including during curriculum.
                with torch.inference_mode():
                    values=[float(objective(model,[validation.get(i)],stage,'cuda',HORIZON)) for i in validation_indices]
                last_validation=sum(values)/len(values)
                if last_validation<best_validation:best_validation=last_validation;is_best=True
            if (update+1)%cfg['save_every']==0 or update+1==updates or is_best:
                checkpoint(update+1,is_best);log.flush()
