"""Prepare actual live demonstrations through the existing sequence builders."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from navigation_state import checksum
from episode_store import verified_rgb_storage
from build_world_sequences import build as world_data
from build_policy_sequences import build as policy_data


def export_live(episode,output,checkpoint_identity):
    runtime=episode/'learned-controller/runtime'
    receipt=json.loads((runtime/'result.json').read_text())
    if receipt['status']!='completed':raise ValueError('Incomplete live inference')
    traces=[json.loads(line) for line in (runtime/'proposals.jsonl').read_text().splitlines()]
    trace_by_frame={r['frame_id']:r for r in traces}
    frames={r['frame_id']:r for r in (json.loads(line) for line in (episode/'observations/frames.jsonl').read_text().splitlines())}
    output.mkdir(parents=True);shards=[]
    for ref in receipt['shards']:
        path=runtime/ref['path']
        if checksum(path)!=ref['sha256']:raise ValueError('Modified demonstration')
        rows=[]
        for row in torch.load(path,map_location='cpu',weights_only=True)['samples']:
            trace=trace_by_frame[row['frame_id']]
            rows.append(dict(row['runtime'],**{k:v for k,v in row.items() if k not in ('runtime','proposal','initial_hidden')},
                tracking_confidence=float(row['runtime']['state'][14]),config=None,
                rgb_reference=dict(frame_id=row['frame_id'],sha256=frames[row['frame_id']]['rgb_sha256'])))
        target=output/ref['path'];torch.save(dict(samples=rows),target)
        shards.append(dict(path=target.name,sha256=checksum(target)))
    shutil.copyfile(runtime/'goal-tokens.pt',output/'goal-tokens.pt')
    with (output/'timing.jsonl').open('w') as stream:
        for row in traces:stream.write(json.dumps(dict(frame_id=row['frame_id'],sim_ns=row['sim_ns'],
            available_monotonic=row['source_available_monotonic'],
            processing_seconds=row['decision_monotonic']-row['source_available_monotonic']))+'\n')
    manifest=dict(episode_id=traces[0]['episode_id'],checkpoint_set_sha256=checkpoint_identity,
        shards=shards,source_rgb_sha256=verified_rgb_storage(episode/'observations')['stream_sha256'],
        goal_tokens=dict(path='goal-tokens.pt',sha256=checksum(output/'goal-tokens.pt')),
        slow_feature_mode='actual online publication at physical learner-visited states')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser()
    for key in ('collection','bundle','checkpoints','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--world',type=Path);parser.add_argument('--phase',choices=('world','policy'),required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    rows=json.loads(args.collection.read_text());identity=checksum(args.checkpoints/'checkpoints.json')
    collection_receipt=json.loads((args.collection.parent/'result.json').read_text())
    if collection_receipt.get('status')!='completed' or len(rows)!=10 or len({r['result']['episode_id'] for r in rows})!=10:
        raise ValueError('Preparation requires the completed ten-episode demonstration batch')
    if any(r['result'].get('controller_checkpoint_sha256')!=identity or r['result'].get('teacher_provenance')!='observed-exploration/v3' for r in rows):
        raise ValueError('Demonstration checkpoint or teacher identity differs')
    replay_root=args.output.parent/(args.output.name+'-replays');replay_root.mkdir()
    replays=[]
    for index,row in enumerate(rows):
        episode=Path(row['episode_path']);replay=replay_root/str(index)
        export_live(episode,replay,identity);replays.append(replay)
    if args.phase=='world':result=world_data(args.bundle,args.checkpoints/'checkpoints.json',replays,args.output)
    else:
        args.output.mkdir();(args.output/'windows').mkdir();merged=None;scales=[];train_ids=[]
        for index,replay in enumerate(replays):
            part=replay_root/('policy-'+str(index))
            policy_data(args.bundle,replay,args.checkpoints,args.world,part)
            spec=json.loads((part/'manifest.json').read_text())
            if merged is None:
                merged=dict(spec,windows=[],episodes=[],attempts=[])
                shutil.copyfile(part/'goal.pt',args.output/'goal.pt')
            for ref in spec['windows']:
                target=args.output/'windows'/f'{index}-{Path(ref["path"]).name}'
                shutil.copyfile(part/ref['path'],target)
                merged['windows'].append(dict(ref,path=str(target.relative_to(args.output))))
            merged['episodes']+=spec['episodes'];merged['attempts']+=spec['attempts']
            norm=torch.load(part/'normalization.pt',weights_only=True)
            if norm['episode_ids']:scales.append(norm['primitive_scale']);train_ids+=norm['episode_ids']
        if not scales or not any(e['split']=='validation' for e in merged['episodes']):
            raise ValueError('Training and disjoint development demonstrations required')
        norm=args.output/'normalization.pt'
        torch.save(dict(fit_split='train',episode_ids=train_ids,primitive_scale=torch.stack(scales).mean(0)),norm)
        merged['normalization']=dict(path=norm.name,sha256=checksum(norm))
        result=dict(status='completed',training_ready=True,deployment_accepted=False,accepted=False,
            windows=len(merged['windows']),episodes=len(merged['episodes']))
        merged['readiness']=result
        (args.output/'manifest.json').write_text(json.dumps(merged,indent=2))
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
