"""Audit causal teacher examples with their exact five source images."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from episode_store import frames
from goal_io import load_goal
from navigation_state import checksum

parser=argparse.ArgumentParser()
for name in ('episode','replay','output'):parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();manifest=json.loads((args.replay/'manifest.json').read_text())
if json.loads((args.episode/'result.json').read_text()).get('split')!='train':raise ValueError('Configuration learning excludes sealed splits')
goal=load_goal(args.episode/'goal')
if manifest['episode_id']!=goal.episode_id:raise ValueError('Cross-episode configuration example')
selected=None
for ref in manifest['shards']:
    path=args.replay/ref['path']
    if checksum(path)!=ref['sha256']:raise ValueError('Changed causal replay')
    for row in torch.load(path,weights_only=True)['samples']:
        if not row['config'] and not bool(row['memory_valid'].any()):
            selected=row;break
    if selected:break
if selected is None:raise ValueError('No audited empty-memory bootstrap state available')
args.output.mkdir(parents=True,exist_ok=False);images=[]
for index,raw in enumerate(goal.rgb_views):
    path=args.output/f'goal-{index}.png'
    Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
    images.append(dict(path=path.name,sha256=checksum(path),role='goal_view'))
for row,raw in frames(args.episode/'observations'):
    if row['frame_id']!=selected['frame_id']:continue
    if row['rgb_sha256']!=selected['rgb_reference']['sha256']:raise ValueError('RGB reference changed')
    path=args.output/'current.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
    images.append(dict(path=path.name,sha256=checksum(path),role='current',observed_ns=row['sim_ns']));break
if len(images)!=5:raise ValueError('Missing actual current RGB')
example=dict(split='train',episode_id=goal.episode_id,sim_ns=selected['sim_ns'],images=images,observed=[],
    progress=dict(metric_geometry_available=selected['metric_geometry_available'],goal_probability=selected['goal_probability']),
    response=json.dumps(dict(t=None,i='search',w=[1.,1.,1.,1.],c=0.,h=5.,d=True),separators=(',',':')),keyframe_image_count=0,
    teacher='deterministic causal frontier configurator abstains when its observed memory is empty')
path=args.output/'grounding.jsonl';path.write_text(json.dumps(example)+'\n')
spec=dict(audited_observation_conditioned_configurations=True,output_format='compact/v2',grounding=path.name,grounding_sha256=checksum(path),
    source_replay_sha256=checksum(args.replay/'manifest.json'),scope='Bootstrap abstention only; no directional search quality claimed')
(args.output/'configurator.json').write_text(json.dumps(spec,indent=2))
(args.output/'result.json').write_text(json.dumps(dict(status='completed',accepted=False,examples=1),indent=2))
