"""Validation-only component metrics and arrival-threshold selection.

This offline command must be run later. Frame-level calibration is not a
substitute for complete-flight false-stop evidence.
"""
from pathlib import Path
import math
import random
import numpy as np
import torch
from .common import digest,write
from .data import Dataset
from .models import load_model


def calibrate(dataset,checkpoint,backbone,output,window):
    model,saved=load_model(backbone,checkpoint);model.eval().requires_grad_(False)
    if not {'localization','goal'}<=set(saved['trained_stages']):raise ValueError('Complete perception stages first')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=Dataset(dataset,'goal','validation');rng=random.Random(731);scores=[];labels=[]
    with torch.inference_mode():
        for _ in range(min(1024,len(data))):
            if not window.remaining():
                write(out/'receipt.json',dict(complete=False,accepted=False,examples=len(scores)));return
            row=data.get(data.sample_index(rng))
            current=model.encode(row['current'][None].cuda(),[row['camera_pitch_deg']])
            goals=model.encode(row['goals'].cuda(),[0.]*len(row['goals']))[None]
            result=model.goal_evidence(current,goals)
            scores.append(min(float(result['match_logit'].sigmoid()),float(result['arrival_logit'].sigmoid())))
            labels.append(bool(row['labels']['arrival']))
    scores=np.asarray(scores);labels=np.asarray(labels)
    if not labels.any() or labels.all():raise ValueError('Validation needs both arrival and nonarrival examples')
    sweep=[]
    for threshold in np.arange(.5,1.,.01):
        positive=scores>=threshold
        sweep.append(dict(threshold=float(threshold),false_positive_rate=float(positive[~labels].mean()),
                          recall=float(positive[labels].mean())))
    qualified=[r for r in sweep if r['false_positive_rate']<=.01 and r['recall']>0]
    receipt=dict(complete=True,accepted=False,scope='validation-frame arrival evidence',sweep=sweep,
                 dataset_sha256=digest(dataset),input_checkpoint_sha256=digest(checkpoint))
    if not qualified:
        write(out/'receipt.json',receipt);raise ValueError('No useful threshold meets validation false-positive target')
    from .maps import MapPrior
    from .localization import locate
    localization=Dataset(dataset,'localization','validation');cached={};pose_errors=[];yaw_errors=[];false_locks=0;localized=0;retrieved=0
    with torch.inference_mode():
        for index in range(min(256,len(localization))):
            if not window.remaining():
                receipt['complete']=False;write(out/'receipt.json',receipt);return
            row=localization.get(localization.sample_index(rng));path=row['map_path']
            if path not in cached:
                prior=MapPrior(path)
                tokens=torch.cat([model.encode(torch.stack([torch.from_numpy(prior.tile(j)).permute(2,0,1)
                    for j in range(i,min(i+4,len(prior.tiles)))]).cuda()) for i in range(0,len(prior.tiles),4)])
                cached[path]=(prior,tokens,model.descriptors(tokens,True))
            prior,maps,descriptors=cached[path]
            current=model.encode(row['current'][None].cuda(),[row['camera_pitch_deg']])
            candidates=locate(model,current,prior,maps,descriptors) if row['labels']['localization_usable'] else ()
            if candidates:
                best=candidates[0];error=float(np.linalg.norm(np.asarray(best.position)-row['position_ned_m']))
                pose_errors.append(error);expected=math.atan2(*row['labels']['yaw'])
                yaw_error=abs((best.yaw-expected+math.pi)%(2*math.pi)-math.pi);yaw_errors.append(math.degrees(yaw_error))
                aligned=best.probability>=.8 and best.sigma_m<=5.;localized+=int(aligned)
                false_locks+=int(aligned and (error>5 or yaw_error>math.radians(30)))
                retrieved+=int(any(c.tile_id in row['positive_tile_ids'] for c in candidates))
    receipt['localization']=dict(examples=min(256,len(localization)),retrieved=retrieved,confident=localized,
        false_confident=false_locks,position_error_m=pose_errors,yaw_error_deg=yaw_errors,
        scope='full-map single-frame estimates; temporal locking requires flight evaluation')
    selected=max(qualified,key=lambda r:(r['recall'],r['threshold']))
    saved['calibration']=dict(selected,scope=receipt['scope'],dataset_sha256=digest(dataset))
    from .provenance import identities
    saved['calibration']['model_identities']=identities(saved['model'])
    if {'policy','world'}<=set(saved['trained_stages']):
        if saved.get('world_actor_identity')!=saved['calibration']['model_identities']['actor']:
            raise ValueError('World training must follow the latest actor update')
        from contextlib import nullcontext
        from .learning import batch_context
        from .contracts import ObservationContext
        from .temporal import rollout
        from .subgoals import score_components
        world=Dataset(dataset,'odometry','validation');components=[];outcomes=[]
        world.windows=[w for w in world.windows if w.get('behavior_actor_identity')==saved['world_actor_identity']]
        if not world.windows:raise ValueError('Collect validation flights with this actor before world-score calibration')
        with torch.no_grad():
            for index in range(min(128,len(world))):
                if not window.remaining():
                    receipt['complete']=False;write(out/'receipt.json',receipt);return
                row=world.get(world.sample_index(rng))
                x,t,c,v,g,goal,embedding=batch_context(model,[row],'cuda')
                context=ObservationContext('observation-context/v3',tuple(row['history_ids']),tuple(row['history_times']),
                    x,c,v,g,goal,row['spatial'],0.).validate()
                refs={}
                if row['subgoal'] and row['reference_rgb'] is not None:
                    refs[(row['subgoal'].target_source,row['subgoal'].target_reference)]=model.encode(row['reference_rgb'][None].cuda()).mean(1)
                predicted=rollout(model,context,row['subgoal'],refs,nullcontext)
                components.append(score_components(predicted))
                outcomes.append(row['labels']['terminal_value'])
        values=np.asarray(components);scales=np.maximum(values.std(0),.1)
        if len(values)<16 or np.std(outcomes)<.05:raise ValueError('World calibration needs diverse validation flight outcomes')
        design=values/scales
        weights=np.linalg.solve(design.T@design+np.eye(5),design.T@np.asarray(outcomes))
        weights[0]=min(weights[0],-.1);weights[3]=-.05
        saved['calibration'].update(score_scales=scales.tolist(),score_weights=weights.tolist(),
            score_examples=len(values),score_scope='validation complete-flight outcomes; unfiltered actor risk rollouts')
    torch.save(saved,out/'calibrated.pt');receipt['selected']=selected
    write(out/'receipt.json',receipt)
