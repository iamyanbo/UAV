"""Publish bounded overnight execution receipts, without launching research.

The remote scheduler owns training and simulator admission. This observer only
copies its existing receipts and, when explicitly requested, commits those
receipts and this generated status to the already configured GitHub remote.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--round', required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 8 or args.interval < 60:
        parser.error('At most eight hours and at least sixty seconds between observations')
    if Path(args.round).name != args.round or not args.round.startswith('navigation-repair-'):
        parser.error('A single navigation repair round name is required')
    source = Path(__file__).resolve().parent
    repository = source.parent.parent
    evidence = source / 'evidence/learning-repair-20260924'
    status_path = source / 'OVERNIGHT_STATUS.json'
    paths = [evidence.relative_to(repository).as_posix(), status_path.relative_to(repository).as_posix()]

    def git(*arguments, **kwargs):
        return subprocess.run(['git', *arguments], cwd=repository, text=True,
            capture_output=True, timeout=120, **kwargs)

    if args.push:
        if git('remote', 'get-url', 'origin', check=True).stdout.strip() != 'https://github.com/iamyanbo/UAV.git':
            raise ValueError('Receipt publishing remote differs from the authorized repository')
        if git('branch', '--show-current', check=True).stdout.strip() != 'master':
            raise ValueError('Receipt publishing expects the existing master branch')
    deadline = time.monotonic() + args.hours * 3600
    terminal = False
    while time.monotonic() < deadline:
        try:
            subprocess.run([sys.executable, str(source / 'sync_repair_evidence.py'),
                '--key', str(args.key), '--output', str(evidence)], check=True, timeout=150)
            directory = evidence / 'rounds' / args.round
            def read(path):
                return json.loads(path.read_text()) if path.exists() else None
            continuation = read(directory / 'continuation.json')
            state = read(directory / 'execution/state.json')
            cumulative = read(directory / 'cumulative-window/execution/state.json')
            stages = (state or {}).get('stages', {})
            if not continuation and stages:
                terminal_states = {'completed', 'accepted', 'failed', 'blocked'}
                terminal = all(row.get('status') in terminal_states for row in stages.values())
            else:
                terminal = bool(continuation and continuation['status'] in
                    ('window_ended', 'blocked', 'operator_stopped', 'window_expired_waiting'))
            status = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
                round=args.round, continuation=continuation,
                stages={name: {key: row.get(key) for key in ('status', 'reason', 'job', 'progress')}
                        for name, row in stages.items()},
                cumulative_stages={name: {key: row.get(key) for key in ('status', 'reason', 'job', 'progress')}
                        for name, row in (cumulative or {}).get('stages', {}).items()},
                deployment_accepted=False,
                scope='Execution snapshot only; optimizer counts do not establish navigation or complete cumulative budgets')
            temporary = status_path.with_suffix('.pending')
            temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            temporary.replace(status_path)
            if args.push:
                git('add', '--', *paths, check=True)
                changed = git('diff', '--cached', '--quiet', '--', *paths)
                if changed.returncode == 1:
                    result = git('commit', '--only', '-m', 'Record overnight learning execution receipts', '--', *paths, check=True)
                    print(result.stdout.splitlines()[0], flush=True)
                elif changed.returncode != 0:
                    raise RuntimeError(changed.stderr)
                git('push', 'origin', 'master', check=True)
            print(json.dumps(dict(status='observed', round=args.round, terminal=terminal)), flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print(json.dumps(dict(status='observation_failed', error=str(error))), flush=True)
            if args.once:
                return 2
        if args.once or terminal:
            break
        time.sleep(min(args.interval, max(0., deadline - time.monotonic())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
