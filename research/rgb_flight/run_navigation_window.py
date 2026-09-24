"""Continue perception, then rebuild compatible learners in one bounded window.

The prior complete learning flow is verified first. Perception optimizer state
is restored only on its unchanged dataset/objective; downstream learners use a
new demonstration dataset and new optimizer round. No navigation acceptance is
inferred from either schedule completing.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from program_scheduler import atomic, digest, run


def verified(row):
    if row['status'] != 'completed':
        raise ValueError('Incomplete dependency')
    for ref in [row['worker_receipt'], *row.get('artifacts', [])]:
        if digest(ref['path']) != ref['sha256']:
            raise ValueError('Changed dependency: ' + ref['path'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perception-cycle', type=Path, required=True)
    parser.add_argument('--after-flow', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    if not 0 < args.hours <= 8 or args.updates < 1:
        parser.error('Positive update target and at most eight hours required')
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    receipt = args.output / 'continuation.json'
    required = ('world-update', 'policy-update', 'qwen-supervised-update',
                'dagger-update', 'ppo-update', 'updated-policy-flight',
                'final-reload-flight', 'development-flights')
    while time.monotonic() < deadline - 150:
        if (args.output / 'STOP').exists() or (Path.home() / 'uav-rgb-flight/STOP').exists():
            atomic(receipt, dict(status='operator_stopped', accepted=False))
            return 2
        path = args.after_flow / 'execution/state.json'
        if path.exists():
            rows = json.loads(path.read_text())['stages']
            failed = {name: row.get('reason') for name, row in rows.items()
                      if row['status'] == 'failed' or row['status'] == 'blocked'
                      and name in required}
            if failed:
                atomic(receipt, dict(status='blocked', failures=failed, accepted=False))
                return 2
            if all(rows[name]['status'] == 'completed' for name in required):
                for name in required:
                    verified(rows[name])
                break
        atomic(receipt, dict(status='waiting_for_real_learning_flow', accepted=False))
        time.sleep(5)
    else:
        atomic(receipt, dict(status='window_expired_waiting', accepted=False))
        return 2
    # The preceding flow still finishes its preference attempt. Postpone only
    # its optional cumulative world/policy work while fixing perception; the
    # newly compatible flow below resumes those independent budgets afterward.
    stop = args.after_flow / 'cumulative-window/execution/STOP'
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text('Superseded by navigation-first perception continuation: ' + str(args.output))
    perception = args.output / 'perception-cycle'
    perception.mkdir(exist_ok=True)
    path = perception / 'programme-spec.json'
    if not path.exists():
        prior = json.loads((args.perception_cycle / 'execution/state.json').read_text())['stages']
        row = prior['balanced-motion-training']
        verified(row)
        if args.updates <= row['progress']['updates']:
            raise ValueError('Perception continuation must add actual updates')
        checkpoint = str(Path(row['job']) / 'visual-training/latest.pt')
        spec = copy.deepcopy(json.loads((args.perception_cycle / 'programme-spec.json').read_text()))
        training = next(s for s in spec['stages'] if s['id'] == 'balanced-motion-training')
        previous = training['command'][training['command'].index('--resume') + 1]
        training['command'][training['command'].index('--resume') + 1] = checkpoint
        training['inputs'] = [checkpoint if p == previous else p for p in training['inputs']]
        for command in (training['command'], training['resume_command']):
            command[command.index('--updates') + 1] = str(args.updates)
            command[command.index('--validation-every') + 1] = '200'
        spec['continuation_of'] = str(args.perception_cycle.resolve())
        spec['optimizer_continuation'] = dict(path=checkpoint, sha256=digest(checkpoint),
            restored_update=row['progress']['updates'], target_updates=args.updates)
        spec['mapping_revision'] = 'incremental-observed-multiview/v1'
        atomic(path, spec)
    atomic(receipt, dict(status='perception_continuation_running', accepted=False))
    code = run(path, perception / 'execution', (deadline - time.monotonic()) / 3600)
    if code or deadline - time.monotonic() <= 150:
        atomic(receipt, dict(status='perception_incomplete', scheduler_return_code=code, accepted=False))
        return code or 2
    atomic(receipt, dict(status='compatible_full_flow_running', accepted=False))
    with (args.output / 'learning-flow.log').open('a') as log:
        code = subprocess.call([sys.executable, str(Path(__file__).with_name('run_after_perception.py')),
            '--perception-cycle', str(perception), '--output', str(args.output / 'learning-flow'),
            '--hours', str((deadline - time.monotonic()) / 3600)], stdout=log, stderr=subprocess.STDOUT)
    atomic(receipt, dict(status='window_ended', scheduler_return_code=code, accepted=False,
        reason='Actual navigation, map handover and cumulative budgets require their own receipts'))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
