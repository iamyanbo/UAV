"""Resumable deterministic random-point expert collection on Spark."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

MAX_INFRASTRUCTURE_ATTEMPTS=2
MAX_CONSECUTIVE_INFRASTRUCTURE_FAILURES=3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prior_artifacts(old, campaign, study_root, episode_ids):
    if not campaign.resolve().is_relative_to((study_root/'runs').resolve()):
        raise ValueError('Prior campaign must be inside the Spark study runs')
    if old.get('manifest_verified') is not True:
        raise ValueError('Prior campaign did not verify its manifest')
    source_hashes=campaign.parent/'source_hashes.json'
    if digest(source_hashes)!=old['signature']['source_hashes_sha256']:
        raise ValueError('Prior campaign source receipt changed')
    completed_ids=set()
    for row in old['completed']:
        identifier=row['episode_id']
        if identifier not in episode_ids or identifier in completed_ids:
            raise ValueError('Invalid or repeated completed episode in prior campaign')
        completed_ids.add(identifier)
        artifact=Path(row['result']).resolve(strict=True)
        if not artifact.is_relative_to(study_root) or digest(artifact)!=row['result_sha256']:
            raise ValueError('Prior completed episode receipt changed')
    for row in old['failures']:
        if row['episode_id'] not in episode_ids:
            raise ValueError('Prior failure has an unknown episode')
        for path_key,hash_key in (('log','log_sha256'),('launcher_receipt','launcher_receipt_sha256'),
                                  ('result','result_sha256')):
            if row.get(hash_key):
                artifact=Path(row[path_key]).resolve(strict=True)
                if not artifact.is_relative_to(study_root) or digest(artifact)!=row[hash_key]:
                    raise ValueError('Prior failure evidence changed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluator-labels',type=Path,required=True)
    parser.add_argument('--obstacle-field',type=Path,required=True)
    parser.add_argument('--maximum-speed-mps',type=float,choices=(3.,4.5,6.),required=True)
    parser.add_argument('--resume-campaign',type=Path)
    parser.add_argument('--import-campaign',type=Path,
                        help='Explicitly carry verified receipts across a source revision')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--episode-id',action='append',help='Select explicit IDs after verifying the complete original manifest')
    args=parser.parse_args()
    if args.resume_campaign and args.import_campaign:
        parser.error('Choose exact resume or explicit source migration')
    job=Path(os.environ['RGB_JOB_DIR']);source=Path(__file__).resolve().parent
    output=job/'visual-goal-campaign';output.mkdir(exist_ok=False)
    labels=json.loads(args.evaluator_labels.read_text())
    manifest_path=args.evaluator_labels.parent.parent/'MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    split=args.evaluator_labels.stem
    if (manifest.get('status')!='deterministic_manifests_complete'
            or not manifest.get('complete_pair_disjointness') or not manifest.get('goal_regions_disjoint')
            or manifest.get('counts',{}).get(split)!=len(labels['episodes'])
            or manifest.get('evaluator_labels',{}).get(split)!=digest(args.evaluator_labels)
            or manifest.get('obstacle_field_sha256')!=digest(args.obstacle_field)):
        raise ValueError('Evaluator labels or geometry do not match the completed deterministic manifest')
    episodes=labels['episodes']
    if args.episode_id:
        selected=set(args.episode_id)
        if len(selected)!=len(args.episode_id) or not selected<={r['episode_id'] for r in episodes}:
            raise ValueError('Repeated or unknown selected episode ID')
        episodes=[r for r in episodes if r['episode_id'] in selected]
    episodes=episodes[:args.limit] if args.limit else episodes
    signature=dict(labels_sha256=digest(args.evaluator_labels),obstacle_field_sha256=digest(args.obstacle_field),
                   maximum_speed_mps=args.maximum_speed_mps,source_hashes_sha256=digest(job/'source_hashes.json'))
    if args.episode_id:signature['selected_episode_ids']=[r['episode_id'] for r in episodes]
    state=dict(schema='visual-goal-collection-state/v1',signature=signature,completed=[],failures=[],
               status='running',all_failures_retained=True,manifest_verified=True,
               manifest=str(manifest_path),manifest_sha256=digest(manifest_path),imported_campaigns=[])
    previous=args.resume_campaign or args.import_campaign
    if previous:
        old_path=previous/'state.json'
        old=json.loads(old_path.read_text())
        same_task=all(old['signature'].get(key)==signature[key] for key in
                      ('labels_sha256','obstacle_field_sha256','maximum_speed_mps'))
        if not same_task or old.get('manifest_sha256')!=state['manifest_sha256']:
            raise ValueError('Prior campaign uses different labels, geometry, speed or manifest')
        if args.resume_campaign and old['signature']!=signature:
            raise ValueError('Exact resume requires the original source revision')
        if args.import_campaign and old['status']=='running':
            raise ValueError('Cannot import an active campaign')
        verify_prior_artifacts(old,previous,Path.home()/'uav-rgb-flight',
                               {row['episode_id'] for row in episodes})
        state['completed']=old['completed'];state['failures']=old['failures']
        state['imported_campaigns']=old.get('imported_campaigns',[])
        if args.import_campaign:
            state['imported_campaigns'].append(dict(campaign=str(previous),
                state_sha256=digest(old_path),source_hashes_sha256=old['signature']['source_hashes_sha256']))
    def terminal_ids():
        completed={row['episode_id'] for row in state['completed']}
        failed=Counter(row['episode_id'] for row in state['failures'])
        return completed|{identifier for identifier,count in failed.items()
                          if count>=MAX_INFRASTRUCTURE_ATTEMPTS}
    def save(status):
        state['status']=status
        state['processed_episodes']=len({row['episode_id'] for row in state['completed']+state['failures']})
        state['attempted_launches']=len(state['completed'])+len(state['failures'])
        state['valid_expert_episodes']=sum(row['success'] for row in state['completed'])
        done=terminal_ids()
        state['next_episode_id']=next((row['episode_id'] for row in episodes if row['episode_id'] not in done),None)
        temporary=output/'state.pending';temporary.write_text(json.dumps(state,indent=2));temporary.replace(output/'state.json')
        print(json.dumps(dict(status=status,processed=state['processed_episodes'],valid=state['valid_expert_episodes'],next=state['next_episode_id'])),flush=True)
    save('running')
    consecutive_failures=0
    for episode in episodes:
        identifier=episode['episode_id']
        if identifier in terminal_ids(): continue
        if not re.fullmatch(r'[A-Za-z0-9_-]+',identifier): raise ValueError('Invalid episode identity')
        prior_attempts=sum(row['episode_id']==identifier for row in state['failures'])
        for attempt in range(prior_attempts+1,MAX_INFRASTRUCTURE_ATTEMPTS+1):
            if Path(os.environ['RGB_CHECKPOINT_REQUEST']).exists():
                save('checkpointed_between_episodes');return 0
            log=output/(identifier+f'-attempt-{attempt:02d}.log')
            command=['python3',str(source/'spark_launch.py'),'--probe','visual-goal-flight','--clock-speed','1',
                     '--evaluator-labels',str(args.evaluator_labels),'--episode-id',identifier,
                     '--obstacle-field',str(args.obstacle_field),'--maximum-speed-mps',str(args.maximum_speed_mps)]
            started=time.monotonic()
            with log.open('x') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            launches=[json.loads(line)['launch'] for line in log.read_text().splitlines() if line.startswith('{"launch":')]
            failure=None
            if len(launches)!=1:
                failure=dict(episode_id=identifier,attempt=attempt,kind='launch_failure',log=str(log),
                             log_sha256=digest(log),return_code=process.returncode)
            else:
                launch=Path(launches[0]);result_path=launch/'episode/result.json'
                if not result_path.is_file():
                    launcher_receipt=launch/'result.json'
                    failure=dict(episode_id=identifier,attempt=attempt,kind='missing_episode_result',
                        result=str(result_path),launcher_receipt=str(launcher_receipt),
                        launcher_receipt_sha256=digest(launcher_receipt) if launcher_receipt.is_file() else None,
                        log=str(log),log_sha256=digest(log),return_code=process.returncode,
                        wall_seconds=time.monotonic()-started)
                else:
                    try:
                        result=json.loads(result_path.read_text())
                    except (OSError,ValueError) as error:
                        failure=dict(episode_id=identifier,attempt=attempt,kind='invalid_episode_result',
                            result=str(result_path),result_sha256=digest(result_path),error=str(error),
                            log=str(log),log_sha256=digest(log),return_code=process.returncode,
                            wall_seconds=time.monotonic()-started)
                    else:
                        row=dict(episode_id=identifier,result=str(result_path),result_sha256=digest(result_path),
                                 success=bool(result.get('success')),termination=result.get('termination',result.get('status')),
                                 wall_seconds=time.monotonic()-started)
                        if result.get('training_label_frames',0)>0:
                            state['completed'].append(row)
                        else:
                            failure=dict(row,attempt=attempt,kind='unusable_episode_result',log=str(log),
                                         log_sha256=digest(log),return_code=process.returncode)
            if failure is None:
                consecutive_failures=0
                save('running')
                break
            state['failures'].append(failure)
            consecutive_failures+=1
            if consecutive_failures>=MAX_CONSECUTIVE_INFRASTRUCTURE_FAILURES:
                save('infrastructure_failed');return 2
            save('running')
    if not state['valid_expert_episodes']:
        save('no_valid_expert_episodes');return 2
    save('collection_complete');return 0


if __name__=='__main__':
    raise SystemExit(main())
