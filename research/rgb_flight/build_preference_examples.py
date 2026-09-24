"""Turn matched physical outcomes into audited Qwen preference supervision."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import Image
from episode_store import frames
from goal_io import load_goal
from navigation_state import checksum

parser=argparse.ArgumentParser()
for name in ('outcomes','supervised-adapter','output'):parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();source=json.loads(args.outcomes.read_text())
adapter=torch.load(args.supervised_adapter,map_location='cpu',weights_only=True)
output_format=adapter.get('output_format','verbose/v1')
if output_format not in ('verbose/v1','compact/v2'):raise ValueError('Unknown supervised configuration encoding')
def response(value):
    if output_format=='compact/v2':
        value=dict(t=value['target_id'],i=value['intention'],
            w=[value[k] for k in ('goal_weight','time_weight','information_weight','additional_caution')],
            c=value['confidence'],h=value['validity_horizon_seconds'],d=value['deliberate_immediately'])
    return json.dumps(value,separators=(',',':'))
args.output.mkdir(parents=True,exist_ok=False)
(args.output/'all-outcomes.json').write_text(json.dumps(source.get('pairs',[source]),indent=2))
episode=Path(source['outcomes'][0]['episode_path'])
if any(json.loads((Path(item['episode_path'])/'result.json').read_text()).get('split')!='train' for item in source['outcomes']):
    raise ValueError('Preference learning excludes sealed splits')
goal=load_goal(episode/'goal')
images=[]
for i,raw in enumerate(goal.rgb_views):
    path=args.output/f'goal-{i}.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
    images.append(dict(path=path.name,sha256=checksum(path),role='goal_view'))
proposal=json.loads((episode/'learned-controller/runtime/proposals.jsonl').read_text().splitlines()[0])
for row,raw in frames(episode/'observations'):
    if row['frame_id']!=proposal['frame_id']:continue
    path=args.output/'current.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(path)
    images.append(dict(path=path.name,sha256=checksum(path),role='current',observed_ns=row['sim_ns']));break
if len(images)!=5:raise ValueError('Matched preference context lacks exact current RGB')
pair=dict(split='train',episode_id=goal.episode_id,sim_ns=proposal['sim_ns'],images=images,observed=[],progress={},
    controller_sha256=source['controller_sha256'],matched_conditions_sha256=source['matched_conditions_sha256'],
    ambiguous=source['ambiguous'],outcome_relation='tie' if source['ambiguous'] else 'strict_preference',keyframe_image_count=0)
winner=source['winner']
if winner is not None:
    chosen,rejected=source['outcomes'][winner],source['outcomes'][1-winner]
    pair.update(chosen=response(chosen['configuration']),rejected=response(rejected['configuration']),
        chosen_outcome=chosen,rejected_outcome=rejected)
else:
    pair.update(configurations=[r['configuration'] for r in source['outcomes']],outcomes=source['outcomes'])
all_pairs=args.output/'matched-pairs.jsonl';all_pairs.write_text(json.dumps(pair)+'\n')
training=args.output/'preferences.jsonl';training.write_text('' if winner is None else json.dumps(pair)+'\n')
shutil.copyfile(args.supervised_adapter,args.output/'supervised.pt')
manifest=dict(audited_observation_conditioned_configurations=True,output_format=output_format,
    controller=dict(meaningful_complete_flights=True,sha256=source['controller_sha256']),
    preferences=training.name,preferences_sha256=checksum(training),all_matched_pairs=all_pairs.name,
    all_matched_pairs_sha256=checksum(all_pairs),source_outcomes_sha256=checksum(args.outcomes),
    preference_supervision_available=winner is not None,
    blocking_reason='Matched timeouts tie; no preference winner exists; retain SFT adapter' if winner is None else None)
(args.output/'configurator.json').write_text(json.dumps(manifest,indent=2))
result=dict(status='completed',accepted=False,matched_pairs=source.get('matched_pairs',1),strict_preferences=int(winner is not None),
    ties=source.get('ties',int(winner is None)),preference_update_ready=winner is not None,reason=manifest['blocking_reason'])
(args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
