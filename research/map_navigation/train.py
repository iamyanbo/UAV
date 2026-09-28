"""Resumable stage-specific training. Never claims acceptance from update count."""
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from .common import SCHEMA,config,digest,write,seed_all,reserve_memory
from .data import Dataset
from .models import PhotoNavigationModel


def objective(model,items,stage,device):
    current=torch.stack([r['current'] for r in items]).to(device)
    maps=torch.stack([r['tile'] for r in items]).to(device)
    z=model.encoder(current);map_tokens=model.encoder(maps)
    labels=[r['labels'] for r in items]
    target=lambda name:torch.tensor([row[name] for row in labels],device=device,dtype=torch.float32)
    if stage=='localization':
        negative=model.encoder(torch.stack([r['negative_tile'] for r in items]).to(device))
        q=model.descriptors(z);positive=model.descriptors(map_tokens,True);neg=model.descriptors(negative,True)
        logits=torch.stack(((q*positive).sum(-1),(q*neg).sum(-1)),1)/.07
        regression=model.register(z,map_tokens);bad=model.register(z,negative)
        delta=regression['offset']-target('offset');sigma=regression['sigma']
        loss=F.cross_entropy(logits,torch.zeros(len(items),dtype=torch.long,device=device))
        loss+=((delta.square().sum(-1)/(2*sigma.square()))+2*sigma.log()).mean()
        loss+=F.smooth_l1_loss(regression['above_surface'],target('above_surface'))*.1
        loss+=F.mse_loss(regression['yaw'],target('yaw'))
        loss+=F.binary_cross_entropy_with_logits(regression['match_logit'],torch.ones(len(items),device=device))
        loss+=F.binary_cross_entropy_with_logits(bad['match_logit'],torch.zeros(len(items),device=device))
        return loss
    # Pad images with a mask; padding is never repeated visual evidence.
    goals=current.new_zeros(len(items),4,3,480,640)
    valid=torch.zeros(len(items),4,dtype=torch.bool,device=device)
    for i,row in enumerate(items):
        count=len(row['goals'])
        if model.training:count=1 if random.random()<.5 else random.randint(1,count)
        indices=random.sample(range(len(row['goals'])),count) if model.training else list(range(count))
        goals[i,:count]=row['goals'][indices].to(device);valid[i,:count]=True
    goal_tokens=model.encoder(goals.flatten(0,1)).reshape(len(items),4,300,256)
    evidence=model.goal_evidence(z,goal_tokens,valid)
    if stage=='goal':
        return F.binary_cross_entropy_with_logits(evidence['match_logit'],target('near_goal'))+F.binary_cross_entropy_with_logits(evidence['arrival_logit'],target('arrival'))
    if stage=='policy':
        command,_=model.local_action(z,evidence['goal_context'],target('subgoal_body'),z.new_zeros(len(items),4))
        scale=command.new_tensor([3.,3.,1.,45.])
        return F.smooth_l1_loss(command/scale,target('command')/scale)
    horizon=min(min(len(r['teacher']),len(r['actions'])) for r in items)
    actions=torch.stack([r['actions'][:horizon] for r in items]).to(device)
    teacher=torch.stack([r['teacher'][:horizon] for r in items]).to(device)
    memory=None
    if all(r['history_rgb'] for r in items):
        # Causal observation history is trained, not introduced only at inference.
        with torch.no_grad():
            histories=[]
            for row in items:
                features=model.encoder(torch.stack(row['history_rgb']).to(device)).flatten(0,1)
                histories.append(F.adaptive_avg_pool1d(features.T[None],64).transpose(1,2)[0])
            memory=torch.stack(histories)
    # Equal training exposure for the same-weights map and recent-only ablations.
    predicted=model.future(z,map_tokens,goal_tokens,actions,use_map=(random.random()>=.5 if model.training else True),goal_valid=valid,memory=memory)
    visual=F.mse_loss(F.layer_norm(predicted['visual'],(1024,)),F.layer_norm(teacher,(1024,)))
    positions=torch.tensor([r['future_position'][:horizon] for r in labels],device=device)
    goal=torch.tensor([r['future_goal'][:horizon] for r in labels],device=device)
    collision=torch.tensor([r['future_collision'][:horizon] for r in labels],device=device)
    with torch.no_grad():
        future_rgb=torch.stack([r['future_rgb'][:horizon] for r in items]).to(device)
        future_features=model.encoder(future_rgb.flatten(0,1)).reshape(len(items),horizon,300,256)
        future_match=model.register(future_features.flatten(0,1),map_tokens[:,None].expand(-1,horizon,-1,-1).flatten(0,1))['match_logit'].sigmoid().reshape(len(items),horizon)
        current_match=model.register(z,map_tokens)['match_logit'].sigmoid()[:,None]
        information=(future_match-current_match).clamp_min(0)+(goal-goal[:,:1]).clamp_min(0)
    return (visual+.1*F.smooth_l1_loss(predicted['state'][...,:3],positions)+
            F.binary_cross_entropy_with_logits(predicted['goal'],goal)+
            F.binary_cross_entropy_with_logits(predicted['collision'],collision)+
            .1*F.smooth_l1_loss(predicted['information'],information))


def train(dataset,stage,backbone,output,window,updates=None,resume=None,initialize=None,teacher_root=None,seed=0):
    if resume and initialize:raise ValueError('Choose resume OR initialize from a preceding stage')
    if (Path(output)/'latest.pt').exists() and not resume:
        raise ValueError('Checkpoint exists; explicitly resume or choose a new output directory')
    cfg=config()['training'];updates=updates or cfg['updates_per_stage']
    seed_all(seed);reserve_memory()
    training=Dataset(dataset,stage,'train',teacher_root);validation=Dataset(dataset,stage,'validation',teacher_root)
    model=PhotoNavigationModel(backbone).cuda();trained=[];start=0
    data_identity=digest(dataset);source_identity=digest(backbone)
    saved=None
    if resume or initialize:
        saved=torch.load(resume or initialize,map_location='cpu',weights_only=True)
        if saved['schema']!=SCHEMA or saved['backbone_sha256']!=source_identity:raise ValueError('Incompatible checkpoint')
        model.load_state_dict(saved['model'],strict=True);trained=list(saved['trained_stages'])
    dependencies={'localization':set(),'goal':{'localization'},'policy':{'localization','goal'},'world':{'localization','goal'}}
    if not dependencies[stage]<=set(trained):raise ValueError('Missing prior trained stages: '+str(dependencies[stage]-set(trained)))
    # Later stages cannot silently invalidate localization by modifying its encoder.
    trainable={'localization':('encoder.','camera_projection.','map_projection.','registration.'),
               'goal':('goal.','arrival.'),'policy':('policy.','command.'),
               'world':('world.','map_attention.','teacher_projection.')}[stage]
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(trainable))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    if resume:
        if saved['stage']!=stage or saved['dataset_sha256']!=data_identity or saved['seed']!=seed:
            raise ValueError('Resume requires identical stage, seed and dataset')
        optimizer.load_state_dict(saved['optimizer']);start=saved['updates']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    last_validation=None
    def checkpoint(update):
        stages=sorted(set(trained)|({stage} if update>=updates else set()))
        value=dict(schema=SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),stage=stage,updates=update,
            trained_stages=stages,seed=seed,dataset_sha256=data_identity,backbone_sha256=source_identity,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            validation_loss=last_validation,accepted=False)
        temporary=output/'latest.pending';torch.save(value,temporary);temporary.replace(output/'latest.pt')
        write(output/'receipt.json',dict(stage=stage,updates=update,target=updates,complete=update>=updates,
              checkpoint_sha256=digest(output/'latest.pt'),accepted=False,validation_loss=last_validation))
    with (output/'loss.jsonl').open('a') as log:
        for update in range(start,updates):
            if not window.remaining():checkpoint(update);return
            reserve_memory();model.train()
            if stage!='localization':model.encoder.eval()
            indices=[random.randrange(len(training)) for _ in range(cfg['batch_size'])]
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model,[training.get(i) for i in indices],stage,'cuda')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; preserve previous checkpoint')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
            log.write(__import__('json').dumps(dict(update=update+1,loss=float(loss.detach())))+'\n')
            if (update+1)%cfg['save_every']==0 or update+1==updates:
                model.eval()
                with torch.inference_mode():
                    values=[float(objective(model,[validation.get(i)],stage,'cuda')) for i in range(min(32,len(validation)))]
                last_validation=sum(values)/len(values);checkpoint(update+1);log.flush()
