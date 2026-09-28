"""Resumable stage-specific training. Never claims acceptance from update count."""
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from .common import SCHEMA,config,digest,write,seed_all,reserve_memory
from .data import Dataset
from .models import PhotoNavigationModel


def objective(model,items,stage,device,horizon_limit=20):
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
    goals=current.new_zeros(len(items),4,3,480,640)
    valid=torch.zeros(len(items),4,dtype=torch.bool,device=device)
    for i,row in enumerate(items):
        count=len(row['goals'])
        if model.training:count=1 if random.random()<.5 else random.randint(1,count)
        indices=random.sample(range(len(row['goals'])),count) if model.training else list(range(count))
        goals[i,:count]=row['goals'][indices].to(device);valid[i,:count]=True
    goal_tokens=model.encode(goals.flatten(0,1),[0.]*(len(items)*4)).reshape(len(items),4,300,256)
    evidence=model.goal_evidence(z,goal_tokens,valid)
    if stage=='goal':
        return F.binary_cross_entropy_with_logits(evidence['match_logit'],target('near_goal'))+F.binary_cross_entropy_with_logits(evidence['arrival_logit'],target('arrival'))
    if stage!='world':raise ValueError('Planner-first training has no learned actor stage')
    maps=model.encode(torch.stack([r['tile'] for r in items]).to(device))
    horizon=min(horizon_limit,max(len(r['actions']) for r in items));batch=len(items)
    actions=z.new_zeros(batch,horizon,4,5);teacher=z.new_zeros(batch,horizon,64,1024)
    positions=z.new_zeros(batch,horizon,3);goals_label=z.new_zeros(batch,horizon);collision=z.new_zeros(batch,horizon)
    mask=torch.zeros(batch,horizon,dtype=torch.bool,device=device);visual_mask=mask.clone()
    memory=[]
    for i,row in enumerate(items):
        n=min(horizon,len(row['actions']));actions[i,:n]=row['actions'][:n].to(device);mask[i,:n]=True
        teacher[i,:n]=row['teacher'][:n].to(device);visual_mask[i,:n]=row['teacher_valid'][:n].to(device)
        positions[i,:n]=z.new_tensor(row['labels']['future_position'][:n])
        goals_label[i,:n]=z.new_tensor(row['labels']['future_goal'][:n])
        collision[i,:n]=z.new_tensor(row['labels']['future_collision'][:n])
        with torch.no_grad():
            if row['history_rgb']:
                features=model.encode(torch.stack(row['history_rgb']).to(device),row['history_pitch']).flatten(0,1)
                memory.append(F.adaptive_avg_pool1d(features.T[None],64).transpose(1,2)[0])
            else:memory.append(z.new_zeros(64,256))
    predicted=model.future(z,maps,goal_tokens,actions,goal_valid=valid,memory=torch.stack(memory),
                           runtime_state=z.new_tensor([r['runtime_state'] for r in items]))
    def masked(value,which):return (value*which).sum()/which.sum().clamp_min(1)
    visual=F.mse_loss(F.layer_norm(predicted['visual'],(1024,)),F.layer_norm(teacher,(1024,)),reduction='none').mean((-1,-2))
    return (masked(visual,visual_mask&mask)+.1*masked(F.smooth_l1_loss(predicted['state'][...,:3],positions,reduction='none').mean(-1),mask)+
            masked(F.binary_cross_entropy_with_logits(predicted['goal'],goals_label,reduction='none'),mask)+
            masked(F.binary_cross_entropy_with_logits(predicted['collision'],collision,reduction='none'),mask))


def train(dataset,stage,backbone,output,window,updates=None,resume=None,initialize=None,teacher_root=None,seed=0,fine_tune=False):
    if resume and initialize:raise ValueError('Choose resume OR initialize from a preceding stage')
    if (Path(output)/'latest.pt').exists() and not resume:
        raise ValueError('Checkpoint exists; explicitly resume or choose a new output directory')
    cfg=config()['training'];updates=updates or (cfg['fine_tune_updates'] if fine_tune else cfg['updates_per_stage'])
    if fine_tune and (stage!='world' or not (initialize or resume)):raise ValueError('Fine tuning requires an initialized world stage')
    if updates>(cfg['fine_tune_updates'] if fine_tune else cfg['updates_per_stage']):raise ValueError('Explicit campaign revision required to extend the budget')
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
    dependencies={'localization':set(),'goal':{'localization'},'world':{'localization','goal'}}
    if not dependencies[stage]<=set(trained):raise ValueError('Missing prior trained stages: '+str(dependencies[stage]-set(trained)))
    # Later stages cannot silently invalidate localization by modifying its encoder.
    trainable={'localization':('encoder.','camera_projection.','map_projection.','registration.','camera_embedding.'),
               'goal':('goal.','arrival.'),
               'world':('world.','map_attention.','teacher_projection.')}[stage]
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
        value=dict(teacher_sha256=training.teacher_identity,calibration=saved.get('calibration') if saved and stage=='world' else None,schema=SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),stage=stage,updates=update,
            trained_stages=stages,seed=seed,dataset_sha256=data_identity,backbone_sha256=source_identity,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            validation_loss=last_validation,accepted=False,target_updates=updates,fine_tune=fine_tune,best_validation=best_validation)
        temporary=output/'latest.pending';torch.save(value,temporary);temporary.replace(output/'latest.pt')
        if is_best:
            temporary=output/'best.pending';torch.save(value,temporary);temporary.replace(output/'best.pt')
        if update>=updates and (output/'best.pt').exists():
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
            best['trained_stages']=stages;best['stage_budget_completed']=True
            temporary=output/'best.pending';torch.save(best,temporary);temporary.replace(output/'best.pt')
        write(output/'receipt.json',dict(stage=stage,updates=update,target=updates,complete=update>=updates,
              checkpoint_sha256=digest(output/'latest.pt'),accepted=False,validation_loss=last_validation))
    with (output/'loss.jsonl').open('a') as log:
        for update in range(start,updates):
            if not window.remaining():checkpoint(update);return
            reserve_memory();model.train()
            if stage!='localization':model.encoder.eval()
            indices=[training.sample_index(random,learner_mix=fine_tune) for _ in range(cfg['batch_size'])]
            horizon=20 if fine_tune or (update+1)/updates>.5 else 5 if (update+1)/updates<=.2 else 10
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model,[training.get(i) for i in indices],stage,'cuda',horizon)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; preserve previous checkpoint')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
            log.write(__import__('json').dumps(dict(update=update+1,loss=float(loss.detach())))+'\n')
            is_best=False
            if (update+1)%cfg['validate_every']==0 or update+1==updates:
                model.eval()
                # Validation always uses the full horizon, including during curriculum.
                with torch.inference_mode():
                    values=[float(objective(model,[validation.get(i)],stage,'cuda',20)) for i in validation_indices]
                last_validation=sum(values)/len(values)
                if last_validation<best_validation:best_validation=last_validation;is_best=True
            if (update+1)%cfg['save_every']==0 or update+1==updates or is_best:
                checkpoint(update+1,is_best);log.flush()
