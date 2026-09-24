"""Resumable Stage-A physical route survey, never a learned-policy dataset.

Run inside one eight-hour Spark flight lease. Each episode launches fresh
observation-only perception and a separate map. Stop on an actual prerequisite
failure; completing the survey does not certify language/scale/route coverage.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--route-file',type=Path,required=True)
    parser.add_argument('--clock-speed',type=float,required=True)
    parser.add_argument('--resume-campaign',type=Path)
    args=parser.parse_args()
    if not 0 < args.clock_speed <= 1:
        parser.error('Invalid continuous simulation clock speed')
    job=Path(os.environ['RGB_JOB_DIR'])
    source=Path(__file__).resolve().parent
    output=job/'reference-campaign'
    output.mkdir(exist_ok=False)
    routes=json.loads(args.route_file.read_text())['routes']
    if len(routes)<20:
        raise ValueError('Stage-A survey requires at least 20 route proposals')
    routes=routes[:20]
    ids=[r['id'] for r in routes]
    if (len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in ids)
            or any(not 150<=r['reference_length_m']<=180 for r in routes)):
        raise ValueError('Expected unique long-route engineering proposals')
    source_digest=hashlib.sha256(json.dumps(json.loads((job/'source_hashes.json').read_text()),sort_keys=True).encode()).hexdigest()
    signature=dict(route_file_sha256=digest(args.route_file),clock_speed=args.clock_speed,
                   source_hashes_sha256=source_digest)
    state=dict(signature=signature,completed=[],failed=[],full_flight_foundation_complete=False,
               scope='privileged varied-route perception survey; not learned navigation or imitation labels')
    if args.resume_campaign:
        old=json.loads((args.resume_campaign/'state.json').read_text())
        if old['signature']!=signature:
            raise ValueError('Survey inputs/source changed; start an explicit new calibration round')
        if old['failed']:
            raise ValueError('A failed survey needs a documented fix and new calibration round')
        for previous in old['completed']:
            if digest(previous['episode_result'])!=previous['episode_result_sha256']:
                raise ValueError('Previous accepted episode receipt changed')
        state['completed']=old['completed']
    def save(status):
        state['status']=status
        state['completed_unique_routes']=len({x['route_id'] for x in state['completed']})
        state['next_route_id']=next((x for x in ids if x not in {r['route_id'] for r in state['completed']}),None)
        temporary=output/'state.pending'
        temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(output/'state.json')
        print(json.dumps(dict(status=status,completed_unique_routes=state['completed_unique_routes'],
                              next_route_id=state['next_route_id'],state=str(output/'state.json'))),flush=True)
    save('running')
    try:
        for route in routes:
            if route['id'] in {r['route_id'] for r in state['completed']}:
                continue
            if Path(os.environ['RGB_CHECKPOINT_REQUEST']).exists():
                save('checkpointed_between_episodes')
                return 0
            started=time.monotonic()
            logfile=output/(route['id']+'.log')
            with logfile.open('x') as stream:
                process=subprocess.run(['python3',str(source/'spark_launch.py'),'--probe','reference',
                    '--route-file',str(args.route_file),'--route-id',route['id'],
                    '--clock-speed',str(args.clock_speed),'--live-perception'],stdout=stream,stderr=subprocess.STDOUT)
            launches=[]
            for line in logfile.read_text().splitlines():
                if line.startswith('{"launch":'):
                    launches.append(json.loads(line)['launch'])
            if len(launches)!=1:
                raise RuntimeError('Route launch did not produce exactly one immutable launch record')
            episode=Path(launches[0])/'reference_episode'
            result_path=episode/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else {}
            if process.returncode or not result.get('success') or not result.get('capture_20hz_passed'):
                if result.get('status')=='interrupted_checkpoint':
                    state['interrupted_attempt']=str(episode)
                    save('checkpointed_with_incomplete_episode')
                    return 0
                state['failed'].append(dict(route_id=route['id'],episode=str(episode),result=result))
                save('prerequisite_failed')
                return 2
            archive_dir=output/('archive-'+episode.parent.name)
            archive_dir.mkdir()
            environment=dict(os.environ,RGB_JOB_DIR=str(archive_dir),RGB_OWNING_JOB_DIR=str(job))
            subprocess.run(['python3',str(source/'container_job.py'),'--script','archive_video.py',
                            '--data',str(episode/'observations')],env=environment,check=True)
            subprocess.run(['python3',str(source/'verify_reference_evidence.py'),str(episode)],check=True)
            root=Path.home()/'uav-rgb-flight'
            subprocess.run([str(root/'envs/airsim/bin/python'),str(source/'perception_metrics.py'),str(episode)],check=True)
            state['completed'].append(dict(route_id=route['id'],episode_result=str(result_path),
                 episode_result_sha256=digest(result_path),wall_seconds=time.monotonic()-started,
                 color_calibration_sha256=digest(episode/'color_calibration.json'),
                 consistency_receipt_sha256=digest(episode/'evidence_verification.json'),
                 perception_metrics=str(episode/'engineering_only/perception_metrics.json')))
            save('running')
        save('physical_survey_completed; annotation_and_scale_acceptance_pending')
        return 0
    except (RuntimeError,ValueError,OSError,subprocess.SubprocessError) as error:
        state['error']=str(error)
        save('prerequisite_failed')
        return 2
    finally:
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    raise SystemExit(main())
