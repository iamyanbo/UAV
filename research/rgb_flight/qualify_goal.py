"""Separate recognition, calibration and demonstration-conditioned time metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from goal_matching import GoalMatcherPipeline
from train_visual_components import load_manifest,load_record
from trajectory_bundle import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',type=Path,default=Path('/output/goal-qualification'))
    args=parser.parse_args();root=args.dataset.resolve()
    manifest,records,sha=load_manifest(root,'goal')
    data_root=Path(manifest['collection_root']).resolve() if manifest.get('schema')=='visual-goal-supervision/v3' else root
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved['module']!='goal' or saved['manifest_sha256']!=sha:raise ValueError('Wrong goal checkpoint dataset')
    model=GoalMatcherPipeline('/models/mobilenet-v3-large-imagenet1k-v2.pt').cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);torch.set_num_threads(4)
    hashes={r['episode_id']:r['goal_sha256'] for r in manifest['episodes']}
    cache={};rows=[]
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        for row in records['validation']:
            value=load_record(data_root,row,hashes);rgb=value['runtime'];labels=value['training_labels']
            if row['episode_id'] not in cache:
                cache[row['episode_id']]=model.encoder(rgb['goal_rgb'].cuda())[None]
            prediction=model.matcher(model.encoder(rgb['current_rgb'][None].cuda()),cache[row['episode_id']])
            rows.append(dict(episode_id=row['episode_id'],frame_id=row['frame_id'],
                probability=float(prediction['match_logit'].sigmoid()),near_goal=bool(labels['near_goal']),
                time_prediction=float(prediction['time_to_goal_seconds']),time_label=float(labels['time_to_goal_seconds']),
                time_valid=bool(labels.get('time_valid',True))))
    p=np.asarray([r['probability'] for r in rows]);y=np.asarray([r['near_goal'] for r in rows]);guess=p>=.5
    tp=int(np.sum(guess&y));fp=int(np.sum(guess&~y));fn=int(np.sum(~guess&y));tn=int(np.sum(~guess&~y))
    bins=[]
    for index in range(10):
        mask=(p>=index/10)&(p<(index+1)/10 if index<9 else p<=1)
        bins.append(dict(lower=index/10,upper=(index+1)/10,count=int(mask.sum()),
            confidence=float(p[mask].mean()) if mask.any() else None,
            accuracy=float(y[mask].mean()) if mask.any() else None))
    result=dict(status='completed',accepted=False,checkpoint_sha256=digest(args.checkpoint),
        checkpoint_update=saved['update'],manifest_sha256=sha,development_examples=len(rows),development_episodes=len(cache),
        threshold=.5,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None,confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
        brier_score=float(np.mean((p-y)**2)),calibration_bins=bins,
        expected_calibration_error=sum(b['count']*abs(b['confidence']-b['accuracy']) for b in bins if b['count'])/len(rows),
        time_mae_seconds=float(np.mean([abs(r['time_prediction']-r['time_label']) for r in rows if r['time_valid']])),
        time_supervision='remaining time on successful demonstration; not optimal time-to-goal',
        limitations=[('Development includes aligned boundary negatives; visually similar negative mining is separate'
                     if manifest.get('schema')=='visual-goal-supervision/v3' else 'Existing development set omits 3–12 m hard boundary negatives'),
                    'Operational stopping precision and complete-flight acceptance not established'])
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
