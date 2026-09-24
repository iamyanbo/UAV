"""Continue a retained training window through a fresh physical learning cycle.

This is an operational scheduler, not a test harness. Its deadline includes
waiting, data preparation, optimizer updates and physical simulator flights.
All work uses the existing stage guards and immutable source snapshots.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from program_scheduler import atomic, digest, run
from run_connected_cycle import specification


def verified_stage(state, name):
    row = state['stages'][name]
    if row['status'] != 'completed':
        raise ValueError('Required training stage incomplete: ' + name)
    for ref in [row['worker_receipt'], *row.get('artifacts', [])]:
        if digest(ref['path']) != ref['sha256']:
            raise ValueError('Retained artifact changed: ' + ref['path'])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-cycle', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, required=True)
    parser.add_argument('--demonstration-checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--perception-source', type=Path)
    parser.add_argument('--perception-spec', type=Path)
    parser.add_argument('--perception-round', type=Path)
    args = parser.parse_args()
    if not 0 < args.hours <= 8:
        parser.error('At most eight hours, including waiting')
    perception = (args.perception_source, args.perception_spec, args.perception_round)
    if any(perception) and not all(perception):
        parser.error('Supply all three existing perception continuation paths')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / 'continuation.json'
    deadline = time.monotonic() + args.hours * 3600
    required = ('world-continuation', 'policy-continuation', 'next-training-collection')
    while time.monotonic() < deadline - 150:
        if (args.output / 'STOP').exists() or (Path.home() / 'uav-rgb-flight/STOP').exists():
            atomic(receipt, dict(status='operator_stopped', accepted=False))
            return 2
        state_path=args.training_cycle / 'execution/state.json'
        if not state_path.exists():
            atomic(receipt, dict(status='waiting_for_training_schedule', accepted=False))
            time.sleep(5)
            continue
        state = json.loads(state_path.read_text())
        rows = {name: state['stages'][name]['status'] for name in required}
        if any(value in ('failed', 'blocked') for value in rows.values()):
            atomic(receipt, dict(status='blocked', stages=rows, accepted=False,
                                 reason='Required prior training or collection failed; inspect retained receipts'))
            return 2
        if all(value == 'completed' for value in rows.values()):
            break
        atomic(receipt, dict(status='waiting_for_training', stages=rows, accepted=False))
        time.sleep(5)
    else:
        atomic(receipt, dict(status='window_expired_waiting', accepted=False,
                             reason='Resume the prior exact training window before this continuation'))
        return 2
    world = Path(verified_stage(state, 'world-continuation')['job']) / 'training/final.pt'
    policy = Path(verified_stage(state, 'policy-continuation')['job']) / 'training/final.pt'
    collected = verified_stage(state, 'next-training-collection')
    packed = verified_stage(state, 'continued-policy-pack')
    pack = Path(packed['artifacts'][0]['path']).parent
    manifest = json.loads((pack / 'checkpoints.json').read_text())
    qwen = pack / manifest['artifacts']['qwen']['path']
    safety = pack / manifest['artifacts']['safety']['path']
    # New trajectories create a new manifest and fresh optimizers. Retain the
    # learned parameters, then rebuild critic beliefs against the updated world.
    spec = specification(args.demonstration_checkpoints, safety, 'train-00001', args.demonstrations,
                         world, policy, qwen, Path(collected['job']) / 'collection/flights.json', pack)
    spec['prior_training_cycle'] = str(args.training_cycle.resolve())
    path = args.output / 'programme-spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Changed continuation inputs require a new round')
    if not path.exists():
        atomic(path, spec)
    atomic(receipt, dict(status='full_flow_running', accepted=False,
                         initialized_world=str(world), initialized_policy=str(policy), initialized_qwen=str(qwen)))
    code = run(path, args.output / 'execution', (deadline - time.monotonic()) / 3600)
    final = json.loads((args.output / 'execution/state.json').read_text())
    rows = final['stages']
    independent = all(rows[name]['status'] == 'completed' for name in
                      ('world-update', 'policy-update', 'qwen-supervised-update', 'dagger-update',
                       'ppo-update', 'updated-policy-flight', 'final-reload-flight', 'development-flights'))
    preference = rows['qwen-preference-update']['status']
    # A tied preference cannot hold up independently eligible odometry work.
    if all(perception) and independent and deadline - time.monotonic() > 150:
        (args.perception_round / 'STOP').unlink(missing_ok=True)
        command = [sys.executable, str(args.perception_source / 'program_scheduler.py'),
                   '--spec', str(args.perception_spec), '--round', str(args.perception_round),
                   '--hours', str((deadline - time.monotonic()) / 3600)]
        with (args.output / 'perception.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    atomic(receipt, dict(status='window_ended', independent_flow_completed=independent,
                         preference_status=preference, accepted=False, scheduler_return_code=code,
                         reason='Use flight timing/navigation receipts for acceptance; budgets are separate'))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
