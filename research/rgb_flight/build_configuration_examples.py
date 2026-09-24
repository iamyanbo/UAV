"""Audited observed-target configurations with exact episode RGB evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from episode_store import frames
from goal_io import load_goal
from navigation_state import checksum


def build_episode(episode,replay,output):
    manifest=json.loads((replay/'manifest.json').read_text())
    outcome=json.loads((episode/'result.json').read_text())
    if outcome.get('split')!='train':raise ValueError('Configuration learning excludes sealed splits')
    goal=load_goal(episode/'goal')
    if manifest['episode_id']!=goal.episode_id:raise ValueError('Cross-episode configuration example')
    goal_ref=manifest['goal_tokens'];goal_path=replay/goal_ref['path']
    if checksum(goal_path)!=goal_ref['sha256']:raise ValueError('Changed cached goal')
    if torch.load(goal_path,map_location='cpu',weights_only=True)['goal_sha256']!=goal.content_sha256:
        raise ValueError('Replay and original goal pixels differ')
    selected=[];legacy_selected=False
    for ref in manifest['shards']:
        path=replay/ref['path']
        if checksum(path)!=ref['sha256']:raise ValueError('Changed causal replay')
        for row in torch.load(path,map_location='cpu',weights_only=True)['samples']:
            evidence=row.get('configuration_evidence')
            if evidence is not None:legacy_selected=True
            if evidence is None:
                # Old recordings support exactly their audited empty-memory
                # abstention; do not invent missing target IDs or thumbnails.
                if legacy_selected or row.get('config') or bool(row['memory_valid'].any()):continue
                evidence=dict(schema='observed-rgb-grounding/v1',observed=[],keyframes=[],frontiers=[],teacher_configuration=None)
                legacy_selected=True
            if evidence['schema']!='observed-rgb-grounding/v1':raise ValueError('Unknown grounding evidence contract')
            observed=evidence['observed'];config=evidence['teacher_configuration']
            if any(item['observed_ns']>row['sim_ns'] for item in observed):raise ValueError('Future observed target')
            if config is None:
                if observed:continue
                response=dict(t=None,i='search',w=[1.,1.,1.,1.],c=0.,h=5.,d=True)
            else:
                matches=[item for item in observed if item['id']==config['target_id']]
                if (len(matches)!=1 or config['episode_id']!=goal.episode_id or
                        matches[0]['kind']!=config['grounded_kind']):raise ValueError('Unobserved teacher configuration')
                response=dict(t=config['target_id'],i=config['intention'],
                    w=[config[k] for k in ('goal_weight','time_weight','information_weight','additional_caution')],
                    c=config['confidence'],h=3.,d=config['deliberate_immediately'])
            selected.append((row,evidence,response))
    if not selected:return [],dict(episode_id=goal.episode_id,reason='No audited configuration context')
    output.mkdir(parents=True,exist_ok=False);goals=[]
    for index,raw in enumerate(goal.rgb_views):
        path=output/f'goal-{index}.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
        goals.append(dict(path=path.name,sha256=checksum(path),role='goal_view'))
    needed={}
    for row,evidence,response in selected:
        refs=[dict(frame_id=row['frame_id'],rgb_sha256=row['rgb_reference']['sha256'],observed_ns=row['sim_ns']),
              *evidence['keyframes'],*evidence['frontiers']]
        for ref in refs:
            if ref['observed_ns']>row['sim_ns']:raise ValueError('Future context RGB')
            if ref['frame_id'] in needed and needed[ref['frame_id']]!=ref:raise ValueError('Inconsistent RGB identity')
            needed[ref['frame_id']]=ref
    saved={}
    for metadata,raw in frames(episode/'observations'):
        identifier=metadata['frame_id']
        if identifier not in needed:continue
        ref=needed[identifier]
        if (metadata['rgb_sha256']!=ref['rgb_sha256'] or hashlib.sha256(raw).hexdigest()!=ref['rgb_sha256'] or
                metadata['sim_ns']!=ref['observed_ns']):
            raise ValueError('Changed source RGB or exposure timestamp')
        path=output/f'frame-{identifier}.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
        saved[identifier]=dict(path=path.name,sha256=checksum(path),observed_ns=metadata['sim_ns'],
            source_frame_id=identifier,source_rgb_sha256=metadata['rgb_sha256'])
    if set(saved)!=set(needed):raise ValueError('Missing actual historical RGB')
    examples=[]
    for row,evidence,response in selected:
        images=[*[dict(item) for item in goals],dict(saved[row['frame_id']],role='current')]
        for kind,key in (('keyframe','keyframes'),('frontier','frontiers')):
            images.extend(dict(saved[ref['frame_id']],role=kind) for ref in evidence[key])
        if any(not 6<=item['image_position']<=len(images) for item in evidence['observed']):
            raise ValueError('Observed target lacks its declared supporting image')
        examples.append(dict(split='train',episode_id=goal.episode_id,sim_ns=row['sim_ns'],images=images,
            observed=evidence['observed'],keyframe_image_count=len(evidence['keyframes']),
            progress=dict(metric_geometry_available=row['metric_geometry_available'],goal_probability=row['goal_probability']),
            response=json.dumps(response,separators=(',',':')),teacher='causal observed-frontier/visual-goal configurator',
            source_replay_sha256=checksum(replay/'manifest.json')))
    return examples,None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,action='append',required=True)
    parser.add_argument('--replay',type=Path,action='append',required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(args.episode)!=len(args.replay):raise ValueError('Each physical episode needs its exact causal replay')
    bundle=json.loads(args.bundle.read_text());splits={}
    for item in bundle['attempts']:
        prior=splits.setdefault(item['episode_id'],item['split'])
        if prior!=item['split']:raise ValueError('An episode crossed the internal split')
    args.output.mkdir(parents=True,exist_ok=False);examples=[];excluded=[]
    for index,(episode,replay) in enumerate(zip(args.episode,args.replay)):
        identifier=json.loads((episode/'result.json').read_text())['episode_id']
        if splits.get(identifier)!='train':
            excluded.append(dict(episode_id=identifier,reason='Outside internal training partition'));continue
        folder=args.output/str(index);rows,exclusion=build_episode(episode,replay,folder)
        if exclusion:excluded.append(exclusion)
        for row in rows:
            for image in row['images']:image['path']=str(Path(str(index))/image['path'])
        examples.extend(rows)
    if not examples:raise ValueError('No training-only audited configurations')
    # Put the richest actual context first so a one-update integration cycle
    # exercises historical-image grounding when the recording supports it.
    examples.sort(key=lambda row:(not bool(row['observed']),len(row['images'])<=5,row['episode_id'],row['sim_ns']))
    path=args.output/'grounding.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in examples))
    spec=dict(audited_observation_conditioned_configurations=True,output_format='compact/v2',
        grounding=path.name,grounding_sha256=checksum(path),source_bundle_sha256=checksum(args.bundle),
        training_episode_ids=sorted({row['episode_id'] for row in examples}),excluded=excluded,
        evidence_contract='observed-rgb-grounding/v1',example_order='observed targets, historical RGB, episode and exposure time',
        scope='Actual observed targets or valid abstentions; unavailable supervision is absent')
    (args.output/'configurator.json').write_text(json.dumps(spec,indent=2))
    result=dict(status='completed',accepted=False,examples=len(examples),
        observed_target_examples=sum(bool(row['observed']) for row in examples),
        historical_rgb_examples=sum(len(row['images'])>5 for row in examples),excluded=excluded)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
