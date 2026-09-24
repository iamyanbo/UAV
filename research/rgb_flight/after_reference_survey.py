"""One-shot bounded dependency chain: completed survey -> scale data -> prior.

Does not launch main learning or learned flight. Runs from an immutable local
source snapshot; stops on failed prerequisites, incomplete workers, or STOP.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey-job',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    args = parser.parse_args()
    if not re.fullmatch(r'/home/iamyanbo/uav-rgb-flight/runs/[A-Za-z0-9_-]+',args.survey_job) or not 0<args.hours<=8:
        raise ValueError('Expected a bounded existing survey job')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    ssh = ['ssh','-i',str(Path.home()/'.ssh/gx10_codex_ed25519'),'-o','IdentitiesOnly=yes',
           '-o','BatchMode=yes','-o','ConnectTimeout=5','iamyanbo@10.31.12.8']
    deadline = time.monotonic()+args.hours*3600
    state = dict(status='waiting_for_physical_survey',survey=args.survey_job,main_navigation_training=False,
                 metric_navigation_accepted=False,children=[])
    def save():
        temp = output/'state.pending'
        temp.write_text(json.dumps(state,indent=2))
        temp.replace(output/'state.json')
        (output/'REPORT.md').write_text('# Bounded scale calibration follow-up\n\nStatus: **'+state['status']+'**.\n\n'+
            state.get('reason','Waiting for the declared varied-route survey; no competing model work is launched.')+'\n\n'+
            'This chain cannot unlock main navigation training or autonomous flight. State and artifact paths: `state.json`.\n')
    def check():
        if (output/'STOP').exists() or time.monotonic()>=deadline:
            raise RuntimeError('Follow-up STOP or eight-hour window reached')
    def remote_json(path, optional=False):
        result = subprocess.run([*ssh,shlex.join(['cat',path])],capture_output=True,text=True,timeout=15)
        if result.returncode:
            if optional:
                return None
            raise RuntimeError('Required remote receipt is unavailable: '+path)
        return json.loads(result.stdout)
    def execute(step, extra):
        check()
        run_id = output.name+'-'+step
        run = args.root/'runs'/run_id
        command = [sys.executable,str(Path(__file__).parent/'run.py'),'--stage','preflight',
            '--preflight-step',step,'--root',str(args.root),'--run-id',run_id,
            '--hours',str(min(8.,(deadline-time.monotonic())/3600)),*extra]
        with (output/(step+'.log')).open('x') as stream:
            child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            state['children'].append(dict(step=step,local_run=str(run),pid=child.pid))
            state['status']=step
            save()
            try:
                while child.poll() is None:
                    check()
                    time.sleep(1)
            finally:
                if child.poll() is None:
                    run.mkdir(parents=True,exist_ok=True)
                    (run/'STOP').write_text('Bounded follow-up stopped')
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait(timeout=10)
        receipt = json.loads((run/'state.json').read_text())
        # Preflight intentionally returns foundation_incomplete after a valid
        # calibration worker. Require the actual worker receipt below.
        if receipt['status']!='foundation_incomplete' or len(receipt['jobs'])!=1 or receipt['jobs'][0]['return_code']!=0:
            raise RuntimeError('Prerequisite worker did not complete: '+step)
        return receipt['jobs'][0]['remote_run']
    save()
    try:
        while True:
            check()
            survey = remote_json(args.survey_job+'/reference-campaign/state.json')
            state['completed_survey_routes']=survey['completed_unique_routes']
            guard = remote_json(args.survey_job+'/result.json',optional=True)
            if survey['failed'] or survey['status']=='prerequisite_failed':
                raise RuntimeError('Survey prerequisite failed; calibration has not been started')
            if guard is not None:
                if guard['status']!='finished' or guard.get('return_code')!=0 or not survey['status'].startswith('physical_survey_completed'):
                    raise RuntimeError('Survey was stopped or incomplete; resume the physical prerequisite first')
                break
            save()
            time.sleep(30)
        prepared = execute('scale-prepare',['--remote-campaign',args.survey_job+'/reference-campaign'])
        preparation = remote_json(prepared+'/scale-preparation/result.json')
        if preparation['status']!='scale_bundle_prepared':
            raise RuntimeError('Calibration preparation did not finish; no training started')
        dataset = prepared+'/scale-preparation/dataset'
        state['dataset']=dataset
        trained = execute('scale',['--remote-dataset',dataset])
        result = remote_json(trained+'/scale-training/result.json')
        if result['status']!='calibration_budget_completed':
            raise RuntimeError('Scale fitting checkpointed before its declared budget')
        state.update(status='scale_calibration_completed',training_result=result,
                     artifact=trained+'/scale-training/scale-prior.pt',
                     reason='Engineering scale prior fitted and calibrated; metric full-flight validation remains required.')
    except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as error:
        state.update(status='stopped',reason=str(error))
        return 2
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
