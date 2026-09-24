"""Privileged offline calibration-data launcher, never an inference process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',type=Path,required=True)
    args = parser.parse_args()
    root = (Path.home()/'uav-rgb-flight').resolve()
    survey = args.survey.resolve()
    if not survey.is_relative_to(root):
        raise ValueError('Expected an existing survey inside the study directory')
    state = json.loads((survey/'state.json').read_text())
    source = Path(__file__).resolve().parent
    registry = json.loads((source/'scale-calibration.json').read_text())
    if (state['failed'] or not state['status'].startswith('physical_survey_completed')
            or state['signature']['route_file_sha256'] != registry['route_file_sha256']):
        raise ValueError('The declared physical survey has not completed cleanly')
    episodes = {row['route_id']:row for row in state['completed']}
    if set(episodes) != {row['route_id'] for row in registry['episodes']}:
        raise ValueError('Calibration survey does not cover the declared route splits')
    job = Path(os.environ['RGB_JOB_DIR'])
    mounts = []
    for route_id, row in episodes.items():
        result = Path(row['episode_result']).resolve()
        episode = result.parent
        if not episode.is_relative_to(root) or sha(result) != row['episode_result_sha256']:
            raise ValueError('Modified or out-of-scope survey episode')
        if sha(episode/'evidence_verification.json') != row['consistency_receipt_sha256']:
            raise ValueError('Modified full-flight evidence receipt')
        mounts += ['-v',str(episode)+':/recordings/'+route_id+':ro']
    registry['source_survey'] = str(survey)
    (job/'scale-input.json').write_text(json.dumps(dict(registry=registry,survey_state_sha256=sha(survey/'state.json')),indent=2))
    image = json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    name = 'rgb-scale-preparation-'+str(os.getpid())
    try:
        return subprocess.run(['docker','run','--rm','--name',name,'--label','rgb-flight.job='+str(job),
            '--network','none','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4',
            '--user',f'{os.getuid()}:{os.getgid()}',
            '-v',str(source)+':/source:ro','-v',str(job)+':/output',*mounts,image,'python','/source/scale_data.py',
            '--survey-input','/output/scale-input.json','--output','/output/scale-preparation']).returncode
    finally:
        subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    raise SystemExit(main())
