"""Shared temporal tensorization and staged objectives; evaluator labels stay here."""
import random
import torch
from torch.nn import functional as F
from .contracts import Subgoal
from .temporal import pool,shift,geometry_embedding


def batch_context(model, items, device, augment=False):
    features=[];times=[];commands=[];masks=[];goal_grids=[];vectors=[];references=[]
    for row in items:
        if augment and row.get('delayed_spatial') and random.random()<.25:
            row=dict(row,spatial=random.choice(row['delayed_spatial']))
        images=row['history_rgb'];n=len(images)
        if not 1<=n<=4:raise ValueError('Four real observations at most; padding is not a frame')
        z=model.encode(torch.stack(images).to(device));p=pool(z)
        features.append(F.pad(p,(0,0,0,0,4-n,0)))
        t=p.new_zeros(4);t[-n:]=p.new_tensor([stamp-row['history_times'][-1] for stamp in row['history_times']]);times.append(t)
        c=p.new_zeros(4,4);c[-n:]=p.new_tensor(row['history_commands']);commands.append(c)
        masks.append(torch.arange(4,device=device)>=4-n)
        goal_grids.append(model.encode(row['goals'][:1].to(device)))
        subgoal=row['subgoal'] or Subgoal();now=row['history_times'][-1]
        # Match delayed guidance: occasionally deliver an expired proposal or
        # retain its older snapshot. Never fabricate additional observations.
        if augment and random.random()<.25:now+=random.uniform(0,6)
        vector=subgoal.vector(now,row['spatial'])
        vectors.append(vector)
        reference=row.get('reference_rgb')
        ref=model.encode(reference[None].to(device)).mean(1) if reference is not None and subgoal.valid(now,row['spatial']) else p.new_zeros(1,256)
        if subgoal.valid(now,row['spatial']):ref=ref+geometry_embedding(model,subgoal,row['spatial'],ref)
        references.append(ref[0])
    features=torch.stack(features);times=torch.stack(times);commands=torch.stack(commands);valid=torch.stack(masks)
    goals=torch.stack(goal_grids)
    evidence=model.goal_evidence(features[:,-1],goals)
    embedding=model.subgoal_encoder(torch.cat((features.new_tensor(vectors),torch.stack(references)),-1))
    return features,times,commands,valid,goals,evidence['goal_context'],embedding


def objective(model,items,stage,device,horizon_limit=20,actor_rollouts=False):
    features,times,commands,valid,goals,goal_context,embedding=batch_context(model,items,device,model.training)
    labels=[r['labels'] for r in items]
    target=lambda name:features.new_tensor([r[name] for r in labels])
    if stage in ('policy','dagger'):
        action,stop=model.policy(features,times,commands,valid,goal_context,embedding)
        loss=F.smooth_l1_loss(action/action.new_tensor([3,3,1,45]),target('command')[:,:4]/action.new_tensor([3,3,1,45]))
        return loss+F.binary_cross_entropy_with_logits(stop,target('arrival'))
    if stage=='odometry':
        future=model.encode(torch.stack([r['future_rgb'][0] for r in items]).to(device))
        motion,sigma=model.relative_motion(features[:,-1],future)
        expected=features.new_tensor([[*r['future_position'][0],r['future_yaw'][0]] for r in labels])
        return ((motion-expected).square()/(2*sigma.square())+sigma.log()).mean()
    if stage!='world':raise ValueError('Unknown temporal objective')
    horizon=min(horizon_limit,max(len(r['actions']) for r in items));loss=features.sum()*0.;count=0
    origin=features.new_zeros(len(items),3)
    supported=torch.ones(len(items),dtype=torch.bool,device=device)
    for step in range(horizon):
        active=torch.tensor([step<len(r['actions']) for r in items],device=device)
        slots=torch.stack([r['actions'][step] if step<len(r['actions']) else torch.zeros(4,4) for r in items]).to(device)
        if actor_rollouts:
            with torch.no_grad():
                actor_action,_=model.policy(features,times,commands,valid,goal_context,embedding)
            # Off-policy labels cannot supervise an unexecuted alternate action.
            # Recorded actor branches support imagination only while commands
            # match the dispatched branch within the measured control tolerance.
            match=((slots-actor_action[:,None])/slots.new_tensor([3,3,1,45])).abs().amax((1,2))<.1
            supported=supported&match
            active=active&supported
            slots=actor_action[:,None].expand(-1,4,-1)
        spatial=features.new_tensor([r['spatial'].vector(r['history_times'][-1]+step*.2) for r in items])
        pred=model.world(features,times,commands,valid,goal_context,embedding,slots,spatial)
        with torch.no_grad():
            next_z=pool(model.encode(torch.stack([r['future_rgb'][step] if step<len(r['future_rgb']) else r['current'] for r in items]).to(device)))
        def masked(value,mask=active):return (value*mask).sum()/mask.sum().clamp_min(1)
        loss+=masked(F.mse_loss(pred['feature'],next_z,reduction='none').mean((1,2)))
        teacher=torch.stack([r['teacher'][step] if step<len(r['teacher']) else torch.zeros(64,1024) for r in items]).to(device)
        teacher_mask=active&torch.tensor([step<len(r['teacher_valid']) and bool(r['teacher_valid'][step]) for r in items],device=device)
        loss+=masked(F.mse_loss(F.layer_norm(pred['teacher'],(1024,)),F.layer_norm(teacher,(1024,)),reduction='none').mean((1,2)),teacher_mask)
        get=lambda name:features.new_tensor([r[name][step] if step<len(r[name]) else 0. for r in labels])
        position=features.new_tensor([r['future_position'][step] if step<len(r['future_position']) else [0.]*3 for r in labels])
        loss+=.1*masked(F.smooth_l1_loss(pred['motion'][:,:3],position-origin,reduction='none').mean(-1))
        previous_yaw=features.new_tensor([r['future_yaw'][step-1] if step and step-1<len(r['future_yaw']) else 0. for r in labels])
        dyaw=(get('future_yaw')-previous_yaw+torch.pi)%(2*torch.pi)-torch.pi
        loss+=.1*masked(F.smooth_l1_loss(pred['motion'][:,3],dyaw,reduction='none'))
        origin=position
        for name,key in [('collision','future_collision'),('goal','future_goal')]:
            loss+=masked(F.binary_cross_entropy_with_logits(pred[name],get(key),reduction='none'))
        visibility=features.new_tensor([float(r['future_rgb'][step].float().std()>5) if step<len(r['future_rgb']) else 0. for r in items])
        loss+=masked(F.binary_cross_entropy_with_logits(pred['visibility'],visibility,reduction='none'))
        previous_goal=features.new_tensor([r['future_goal'][step-1] if step and step-1<len(r['future_goal']) else float(r['near_goal']) for r in labels])
        loss+=masked(F.smooth_l1_loss(pred['progress'],get('future_goal')-previous_goal,reduction='none'))
        # Complete-flight outcomes extend well beyond the four-second horizon.
        loss+=masked(F.smooth_l1_loss(pred['value'],target('terminal_value'),reduction='none'))
        features,times,commands,valid=shift(features,times,commands,valid,pred['feature'],slots[:,-1])
        goal_context=model.goal_evidence(features[:,-1],goals)['goal_context']
        count+=1
    return loss/max(count,1)
