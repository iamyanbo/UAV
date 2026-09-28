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
    torch.save(saved,out/'calibrated.pt');receipt['selected']=selected
    write(out/'receipt.json',receipt)
