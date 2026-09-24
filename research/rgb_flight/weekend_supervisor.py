"""Run fresh RGB flight learning cycles across bounded eight-hour windows.

The supervisor chains only existing guarded training and flight schedulers.
It records each handoff and stops at the requested wall-clock deadline or an
operator STOP file. It never treats a completed update as navigation success.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CORE_FLOW = (
    'world-update', 'policy-update', 'qwen-supervised-update', 'dagger-update',
    'ppo-update', 'updated-policy-flight', 'final-reload-flight', 'development-flights',
)


class WeekendRun:
    def __init__(self, root, source, output, deadline):
        self.root = root
        self.source = source
        self.output = output
        self.deadline = deadline
        self.journal = output / 'journal.json'
        self.log = output / 'supervisor.log'
        self.entries = []

    def record(self, event, **fields):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        self.entries.append(row)
        self.journal.write_text(json.dumps(dict(
            schema='weekend-learning/v1', deadline_unix=self.deadline,
            deployment_accepted=False, entries=self.entries,
        ), indent=2) + '\n')
        with self.log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')

    def remaining_hours(self):
        return max(0., min(8., (self.deadline - time.time() - 120) / 3600))

    def stop_requested(self):
        return (self.output / 'STOP').exists() or (self.root / 'STOP').exists()

    def run(self, command, state_path, required, completed_policy='all'):
        while self.remaining_hours() > 0 and not self.stop_requested():
            state = json.loads(state_path.read_text()) if state_path.exists() else None
            rows = (state or {}).get('stages', {})
            statuses = {name: rows.get(name, {}).get('status', 'missing') for name in required}
            if all(value == 'completed' for value in statuses.values()):
                return True
            failures = {name: rows[name].get('reason') for name in required
                        if rows.get(name, {}).get('status') in ('failed', 'blocked')}
            if failures:
                self.record('stage_failure', stages=failures, state=str(state_path))
                return False
            hours = self.remaining_hours()
            if hours <= 0:
                break
            argv = list(command) + ['--hours', f'{hours:.5f}']
            self.record('window_start', command=argv, stages=statuses)
            with self.log.open('a', encoding='utf-8') as stream:
                result = subprocess.run(argv, cwd=self.source, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            state = json.loads(state_path.read_text()) if state_path.exists() else None
            rows = (state or {}).get('stages', {})
            statuses = {name: rows.get(name, {}).get('status', 'missing') for name in required}
            self.record('window_end', return_code=result.returncode, stages=statuses)
            failures = {name: rows[name].get('reason') for name in required
                        if rows.get(name, {}).get('status') in ('failed', 'blocked')}
            if failures:
                self.record('stage_failure', stages=failures, state=str(state_path))
                return False
            if all(value == 'completed' for value in statuses.values()):
                return True
            if result.returncode == 0:
                self.record('incomplete_schedule', stages=statuses)
                return False
            # A bounded window can end with checkpointed work. Reinvoke the
            # identical specification; its guard validates hashes and resumes.
        return False

    def scheduler(self, spec, execution, required):
        command = [sys.executable, str(self.source / 'program_scheduler.py'),
                   '--spec', str(spec), '--round', str(execution)]
        return self.run(command, execution / 'state.json', required)

    def complete_flow(self, training_cycle, demo, pack, output):
        command = [sys.executable, str(self.source / 'continue_complete_flow.py'),
                   '--training-cycle', str(training_cycle), '--demonstrations', str(demo),
                   '--demonstration-checkpoints', str(pack), '--output', str(output)]
        # A tied/no-exposure preference stage is allowed to remain blocked;
        # every independent learner and physical flow must still complete.
        ok = self.run(command, output / 'execution/state.json', CORE_FLOW)
        if ok:
            rows = json.loads((output / 'execution/state.json').read_text())['stages']
            for name in ('development-flights', 'updated-policy-flight', 'final-reload-flight'):
                if rows[name]['status'] != 'completed':
                    return False
            self.record('learning_cycle_complete', cycle=str(output), preference_status=
                        rows.get('qwen-preference-update', {}).get('status'), accepted=False)
        return ok

    def collection_inputs(self, execution):
        state = json.loads((execution / 'state.json').read_text())['stages']
        collection = Path(state['next-training-collection']['job']) / 'collection/flights.json'
        pack = Path(state['continued-policy-pack']['artifacts'][0]['path']).parent
        if not collection.is_file() or not (pack / 'checkpoints.json').is_file():
            raise RuntimeError('Completed continuation lacks its flight collection or checkpoint pack')
        return collection, pack

    def wait_for_existing_scheduler(self, round_path):
        receipt = round_path / 'continuation.json'
        while self.remaining_hours() > 0 and not self.stop_requested():
            pid = None
            if receipt.exists():
                try:
                    pid = json.loads(receipt.read_text()).get('pid')
                except (OSError, json.JSONDecodeError):
                    pass
            if not pid:
                return True
            try:
                status = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                return True
            if status == 'Z':
                return True
            time.sleep(15)
        return False

    def run_all(self, initial_cycle):
        initial = self.root / 'rounds' / initial_cycle
        current_execution = initial
        current_cumulative = initial
        initial_required = ('source-reload-flight', 'world-continuation',
                            'policy-continuation', 'continued-policy-pack', 'next-training-collection')
        # The initial window is already owned by a live scheduler. Wait for its
        # lock owner to exit before resuming its exact immutable specification.
        if not self.wait_for_existing_scheduler(initial):
            self.record('initial_window_wait_stopped')
            return
        while self.remaining_hours() > 0 and not self.stop_requested():
            if current_execution == initial:
                if not self.scheduler(initial / 'programme-spec.json', initial / 'execution', initial_required):
                    self.record('initial_window_incomplete', round=str(initial))
                    return
            collection, pack = self.collection_inputs(current_execution / 'execution')
            cycle_index = 1
            while (self.output / f'full-flow-{cycle_index:03d}').exists():
                cycle_index += 1
            flow = self.output / f'full-flow-{cycle_index:03d}'
            if not self.complete_flow(current_execution, collection, pack, flow):
                self.record('cycle_stopped', cycle=str(flow), reason='Independent learner/flight stage incomplete')
                return
            cumulative = flow / 'cumulative-window'
            command = [sys.executable, str(self.source / 'run_continuation_window.py'),
                       '--cycle', str(flow), '--output', str(cumulative),
                       '--previous-window', str(current_cumulative)]
            required = ('source-reload-flight', 'world-continuation', 'policy-continuation',
                        'continued-policy-pack', 'next-training-collection')
            if not self.run(command, cumulative / 'execution/state.json', required):
                self.record('cumulative_cycle_stopped', cycle=str(cumulative),
                            reason='Continuation or physical collection incomplete')
                return
            current_execution = cumulative
            current_cumulative = cumulative
        status = 'operator_stopped' if self.stop_requested() else 'three_day_deadline_reached'
        self.record(status, accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/iamyanbo/uav-rgb-flight'))
    parser.add_argument('--source', type=Path, default=Path('/home/iamyanbo/uav-rgb-flight/submissions/metric-vision-r92'))
    parser.add_argument('--initial-cycle', default='metric-vision-r92')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--days', type=float, default=3)
    args = parser.parse_args()
    if not 0 < args.days <= 3.5:
        parser.error('The requested unattended period must be between zero and 3.5 days')
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.days * 86400
    worker = WeekendRun(args.root.resolve(), args.source.resolve(), args.output.resolve(), deadline)
    worker.record('started', initial_cycle=args.initial_cycle, days=args.days,
                  windows='At most eight hours each; existing Spark guards retained')
    worker.run_all(args.initial_cycle)


if __name__ == '__main__':
    main()
