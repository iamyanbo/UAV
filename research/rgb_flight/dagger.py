"""DAgger aggregation for policy-generated physical states and expert relabels."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rollout_roots,output):
    rows=[];episodes=set()
    for root in map(Path,rollout_roots):
        result=json.loads((root/'episode/result.json').read_text())
        states=root/'episode/observations/storage.json'
        labels=root/'episode/training_labels/dagger.jsonl'
        if result.get('controller_kind')!='learned_mode_1' or not states.exists() or not labels.exists():
            raise ValueError('DAgger requires actual policy-generated physical states and expert relabels')
        attempt=str(root.resolve())
        if attempt in episodes: raise ValueError('Repeated DAgger physical attempt')
        episodes.add(attempt)
        for line in labels.read_text().splitlines():
            row=json.loads(line)
            if row['episode_id']!=result['episode_id'] or not row['expert_observation_conditioned']:
                raise ValueError('Privileged path action is not a valid observation-conditioned DAgger label')
        rows.append(dict(episode_id=result['episode_id'],rollout=str(root),rollout_result_sha256=digest(root/'episode/result.json'),
                         labels=str(labels),labels_sha256=digest(labels),termination=result['termination']))
    value=dict(schema='physical-dagger-aggregation/v1',episodes=rows,policy_generated_states=True,
               all_failures_retained=True,training_labels_separate=True)
    Path(output).write_text(json.dumps(value,indent=2));return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rollout',type=Path,action='append')
    parser.add_argument('--dataset',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rollouts=args.rollout
    if args.dataset:
        root=args.dataset.resolve();manifest=json.loads((root/'dagger-rollouts.json').read_text())
        rollouts=[]
        for relative in manifest['rollouts']:
            path=(root/relative).resolve()
            if not path.is_relative_to(root): raise ValueError('DAgger rollout escapes dataset')
            rollouts.append(path)
    if not rollouts: raise ValueError('DAgger needs physical rollouts')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=aggregate(rollouts,args.output);print(json.dumps(dict(episodes=len(result['episodes']))))


if __name__=='__main__': main()
