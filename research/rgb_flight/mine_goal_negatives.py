"""Select visually confusable training negatives using one immutable matcher."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from build_visual_dataset import spaced
from goal_matching import GoalMatcherPipeline
from train_visual_components import load_manifest,load_record
from trajectory_bundle import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--output',type=Path,default=Path('/output/goal-data'))
    args=parser.parse_args();root=args.dataset.resolve()
    manifest,records,sha=load_manifest(root,'goal');data_root=Path(manifest['collection_root'])
    checkpoint=root/'initialization.pt'
    if digest(checkpoint)!=manifest['initialization_sha256']:raise ValueError('Negative-mining checkpoint changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=GoalMatcherPipeline('/models/mobilenet-v3-large-imagenet1k-v2.pt').cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);torch.set_num_threads(4)
    hashes={row['episode_id']:row['goal_sha256'] for row in manifest['episodes']}
    cache={};scored={};scores=[]
    with torch.inference_mode():
        for row in records['train']:
            if row['category']!='far_negative':continue
            data=load_record(data_root,row,hashes)['runtime'];episode=(row['episode_id'],row.get('episode_path',''))
            if episode not in cache:cache[episode]=model.encoder(data['goal_rgb'].cuda())[None]
            prediction=model.matcher(model.encoder(data['current_rgb'][None].cuda()),cache[episode])
            score=float(prediction['match_logit'].sigmoid())
            scored.setdefault(episode,[]).append((score,row))
            scores.append(dict(episode_id=row['episode_id'],episode_path=row.get('episode_path'),
                               frame_id=row['frame_id'],probability=score))
    selected=[];hard=set()
    for episode,group in scored.items():
        ranked=sorted(group,key=lambda item:(-item[0],item[1]['frame_id']))
        mined=[r for _,r in ranked[:16]]
        hard.update((r['episode_id'],r.get('episode_path',''),r['frame_id']) for r in mined)
        remaining=[r for _,r in sorted(group,key=lambda item:item[1]['frame_id']) if (r['episode_id'],r.get('episode_path',''),r['frame_id']) not in hard]
        selected+=mined+spaced(remaining,16)
    far_keys={(r['episode_id'],r.get('episode_path',''),r['frame_id']) for r in selected}
    train_ids={r['episode_id'] for r in manifest['episodes'] if r['split']=='train'}
    windows=[]
    for row in manifest['windows']:
        key=(row['episode_id'],row.get('episode_path',''),row['frame_id'])
        if row['episode_id'] in train_ids and row['category']=='far_negative':
            if key not in far_keys:continue
            row=dict(row,category='visually_confusable_negative' if key in hard else 'far_negative')
        windows.append(row)
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(checkpoint,args.output/'initialization.pt')
    manifest.update(windows=windows,negative_mining=dict(source_manifest_sha256=sha,
        encoder_checkpoint_sha256=digest(checkpoint),split='train',scored_examples=len(scores),
        selection='top 16 false match scores per training attempt plus 16 temporally spaced remaining far negatives'))
    (args.output/'visual-training.json').write_text(json.dumps(manifest,allow_nan=False))
    (args.output/'mining-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
    result=dict(status='completed',accepted=False,scored_training_negatives=len(scores),
        mined_examples=len(hard),windows=len(windows),manifest_sha256=digest(args.output/'visual-training.json'))
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
