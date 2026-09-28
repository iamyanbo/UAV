"""Shared temporal tensorization and staged objectives; evaluator labels stay here."""
import random
import torch
from torch.nn import functional as F
from dataclasses import replace
from .contracts import Subgoal,assemble_context,relative_times,HORIZON,STEP_S
from .temporal import pool,shift,geometry_embedding,embed_subgoal,advance_spatial


def batch_context(model, items, device, augment=False):
    features=[];times=[];commands=[];masks=[];goal_grids=[];vectors=[];references=[]
    for row in items:
        images=row['history_rgb'];n=len(images)
        if not 1<=n<=4:raise ValueError('Four real observations at most; padding is not a frame')
        z=model.encode(torch.stack(images).to(device));p=pool(z)
        grid=model.encode(row['goals'][:1].to(device))
        context=assemble_context([(ident,stamp,p[j:j+1],command) for j,(ident,stamp,command) in enumerate(
            zip(row['history_ids'],row['history_times'],row['history_commands']))],grid[None],row['spatial'],0.)
        features.append(context.features[0]);times.append(relative_times(context)[0])
        commands.append(context.commands[0]);masks.append(context.valid[0]);goal_grids.append(grid)
        subgoal=row['subgoal'] or Subgoal();now=row['history_times'][-1]
        # Match delayed guidance: occasionally deliver an expired proposal or
        # retain its older snapshot. Never fabricate additional observations.
        vector=subgoal.vector(now,row['spatial'])
        vectors.append(vector)
        reference=row.get('reference_rgb')
        ref=model.encode(reference[None].to(device)).mean(1) if reference is not None and subgoal.valid(now,row['spatial']) else p.new_zeros(1,256)
        if subgoal.valid(now,row['spatial']):ref=ref+geometry_embedding(model,subgoal,row['spatial'],ref,now)
        references.append(ref[0])
    features=torch.stack(features);times=torch.stack(times);commands=torch.stack(commands);valid=torch.stack(masks)
    goals=torch.stack(goal_grids)
    evidence=model.goal_evidence(features[:,-1],goals)
    embedding=model.subgoal_encoder(torch.cat((features.new_tensor(vectors),torch.stack(references)),-1))
    return features,times,commands,valid,goals,evidence['goal_context'],embedding


def objective(model,items,stage,device,horizon_limit=HORIZON,actor_rollouts=False):
    items=[dict(row) for row in items]
    if model.training and not actor_rollouts:
        for row in items:
            if row.get('delayed_spatial') and random.random()<.25:row['spatial']=random.choice(row['delayed_spatial'])
            if row['subgoal'] and random.random()<.25:
                delay=random.uniform(0,6);g=row['subgoal']
                row['subgoal']=replace(g,source_s=g.source_s-delay,expires_s=g.expires_s-delay)
    features,times,commands,valid,goals,goal_context,embedding=batch_context(model,items,device)
    labels=[r['labels'] for r in items]
    target=lambda name:features.new_tensor([r[name] for r in labels])
    if stage in ('policy','dagger'):
        action,stop=model.policy(features,times,commands,valid,goal_context,embedding)
        loss=F.smooth_l1_loss(action/action.new_tensor([3,3,1,45]),target('command')[:,:4]/action.new_tensor([3,3,1,45]))
        return loss+F.binary_cross_entropy_with_logits(stop,target('arrival'))
    if stage=='odometry':
        future=model.encode(torch.stack([r['future_rgb'][0] for r in items]).to(device))
        motion,sigma=model.relative_motion(features[:,-1],future)
        expected=features.new_tensor([r['future_motion'][0] for r in labels])
        return ((motion-expected).square()/(2*sigma.square())+sigma.log()).mean()
    if stage!='world':raise ValueError('Unknown temporal objective')
    horizon=min(horizon_limit,max(len(r['actions']) for r in items));loss=features.sum()*0.;count=0
    spatial_states=[r['spatial'] for r in items];reference_features=[]
    for row in items:
        reference={};g=row['subgoal']
        if g and row.get('reference_rgb') is not None:
            reference[(g.target_source,g.target_reference)]=model.encode(row['reference_rgb'][None].to(device)).mean(1)
        reference_features.append(reference)
    elapsed=[0.]*len(items);supported_steps=0
    supported=torch.ones(len(items),dtype=torch.bool,device=device)
    for step in range(horizon):
        active=torch.tensor([step<len(r['actions']) for r in items],device=device)
        slots=torch.stack([r['actions'][step] if step<len(r['actions']) else torch.zeros(1,4) for r in items]).to(device)
        dt=features.new_tensor([r['intervals'][step]['dt_s'] if step<len(r['intervals']) else STEP_S for r in items])
        for j,row in enumerate(items):
            if row['subgoal'] and row['subgoal'].target_source=='geometry':
                reference_features[j][('geometry',row['subgoal'].target_reference)]=features[j:j+1,-1].mean(1)
        embedding=torch.cat([embed_subgoal(model,r['subgoal'],r['history_times'][-1]+elapsed[j],
            spatial_states[j],reference_features[j],device) for j,r in enumerate(items)])
        if actor_rollouts:
            with torch.no_grad():
                actor_action,_=model.policy(features,times,commands,valid,goal_context,embedding)
            # Off-policy labels cannot supervise an unexecuted alternate action.
            # Recorded actor branches support imagination only while commands
            # match the dispatched branch within the measured control tolerance.
            match=torch.tensor([step<len(r['intervals']) and all(
                max(abs(a-b)/scale for a,b,scale in zip(segment['values'],actor_action[j].tolist(),[3,3,1,45]))<.1
                for segment in r['intervals'][step]['segments']) for j,r in enumerate(items)],device=device)
            supported=supported&match
            active=active&supported
            slots=actor_action[:,None]
        if not bool(active.any()):break
        supported_steps+=int(active.sum())
        spatial=features.new_tensor([spatial_states[j].vector(r['history_times'][-1]+elapsed[j]) for j,r in enumerate(items)])
        pred=model.world(features,times,commands,valid,goal_context,embedding,slots,spatial,dt)
        with torch.no_grad():
            next_z=pool(model.encode(torch.stack([r['future_rgb'][step] if step<len(r['future_rgb']) else r['current'] for r in items]).to(device)))
        def masked(value,mask=active):return (value*mask).sum()/mask.sum().clamp_min(1)
        observed=active&torch.tensor([step<len(r['future_observed']) and r['future_observed'][step] for r in items],device=device)
        loss+=masked(F.mse_loss(pred['feature'],next_z,reduction='none').mean((1,2)),observed)
        teacher=torch.stack([r['teacher'][step] if step<len(r['teacher']) else torch.zeros(64,1024) for r in items]).to(device)
        teacher_mask=active&torch.tensor([step<len(r['teacher_valid']) and bool(r['teacher_valid'][step]) for r in items],device=device)
        loss+=masked(F.mse_loss(F.layer_norm(pred['teacher'],(1024,)),F.layer_norm(teacher,(1024,)),reduction='none').mean((1,2)),teacher_mask)
        get=lambda name:features.new_tensor([r[name][step] if step<len(r[name]) else 0. for r in labels])
        motion=features.new_tensor([r['future_motion'][step] if step<len(r['future_motion']) else [0.]*4 for r in labels])
        loss+=.1*masked(F.smooth_l1_loss(pred['motion'],motion,reduction='none').mean(-1))
        for name,key in [('collision','future_collision'),('goal','future_goal')]:
            loss+=masked(F.binary_cross_entropy_with_logits(pred[name],get(key),reduction='none'),active if name=='collision' else observed)
        loss+=masked(F.binary_cross_entropy_with_logits(pred['visibility'],get('future_visibility'),reduction='none'),observed)
        loss+=masked(F.smooth_l1_loss(pred['progress'],get('future_progress'),reduction='none'),observed)
        # Complete-flight outcomes extend well beyond the four-second horizon.
        identity=getattr(model,'assessment_actor_identity',None)
        value_mask=active&torch.tensor([bool(identity) and r.get('behavior_actor_identity')==identity for r in items],device=device)
        loss+=masked(F.smooth_l1_loss(pred['value'],get('future_value'),reduction='none'),value_mask)
        for j in range(len(items)):
            spatial_states[j]=advance_spatial(spatial_states[j],pred['motion'][j].detach().cpu().tolist(),float(dt[j]))
            elapsed[j]+=float(dt[j])
        features,times,commands,valid=shift(features,times,commands,valid,pred['feature'],slots[:,-1],dt)
        goal_context=model.goal_evidence(features[:,-1],goals)['goal_context']
        count+=1
    if not supported_steps:raise ValueError('No supported actor rollout supervision; collect matched branches first')
    return loss/max(count,1)
