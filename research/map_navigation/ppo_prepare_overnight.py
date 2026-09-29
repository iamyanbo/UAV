"""Bounded live preparation for PPO; never substitutes surveys for training.

Released trajectories suggest capture poses only. Actual rendered depth certifies
free space. A separate CPU lane fuses completed surveys while the simulator moves
to the next scene. Independent review and trainer admission remain mandatory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np

from .common import digest, read, write


def utc():
    return datetime.now(timezone.utc).isoformat()


def survey_plan(archive, scene_id, split, positions=96):
    filename = 'train.json' if split == 'train' else 'val_unseen.json'
    rows = json.loads(archive.read(filename))['episodes']
    routes = {str(r['trajectory_id']): r for r in rows if int(r['scene_id']) == scene_id}
    candidates = []
    all_points = []
    for ident, row in sorted(routes.items()):
        points = np.asarray(row['reference_path'], dtype=float)[:, :3]
        all_points.extend(points.tolist())
        travelled = 8.
        for i in range(len(points) - 1):
            delta = points[i + 1] - points[i]
            travelled += float(np.linalg.norm(delta))
            if travelled < 8 or np.linalg.norm(delta[:2]) < .1:
                continue
            travelled = 0.
            candidates.append((points[i], math.degrees(math.atan2(delta[1], delta[0])), ident, i))
    if not candidates:
        raise ValueError('No released capture-pose candidates')
    # Farthest-point coverage avoids repeatedly sampling nearby copies of one route.
    points = np.asarray([r[0] for r in candidates])
    nearest = np.full(len(points), np.inf)
    selected = []
    index = int(np.argmin(points[:, 0]))
    for _ in range(min(positions, len(points))):
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[index], axis=1))
        nearest[selected] = -1
        index = int(np.argmax(nearest))
        if nearest[index] < 3:
            break
    captures = []
    # Interleave heights/headings across the whole scene: a deadline must not
    # leave only one corner or only low-altitude observations.
    for height, offset in [(2, 0), (2, 180), (8, 0), (8, 180),
                           (2, 90), (2, 270), (8, 90), (8, 270)]:
        for index in selected:
            position, yaw, ident, step = candidates[index]
            position = position.copy()
            position[2] -= height
            captures.append(dict(position=position.tolist(), yaw_deg=(yaw + offset + 180) % 360 - 180,
                                 source_trajectory=ident, source_step=step, height_offset_m=height))
    points = np.asarray(all_points)
    return dict(schema='ppo-survey-plan/v1', scene_id=f'env_{scene_id}', split=split,
                captures=captures, independent_pose_regions=len(selected),
                released_routes=len(routes), released_bounds=[points.min(0).tolist(), points.max(0).tolist()],
                released_bbox_diagonal_m=float(np.linalg.norm(points.max(0)-points.min(0))),
                geometry_qualified=False, purpose='offline capture proposals; no clearance or expert labels')


def run(args):
    if not 0 < args.hours <= 8:
        raise ValueError('Preparation window must be at most eight hours')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace = Path(args.workspace).resolve()
    start = time.monotonic()
    deadline = start + args.hours * 3600 - 60
    stop = threading.Event()
    guard = threading.Lock()
    stages = {}
    status = dict(schema='ppo-preparation-run/v1', started_utc=utc(), pid=os.getpid(),
                  hours=args.hours, training_running=False, optimizer_updates=0,
                  status='preparing', source_root=str(Path(__file__).resolve().parents[2]),
                  required_before_training=['nontrivial task coverage and physical qualification',
                                            'independent camera/geography review',
                                            '30-context Qwen review', 'combined workload admission'], stages=stages)

    def publish():
        status['updated_utc'] = utc()
        status['elapsed_s'] = time.monotonic() - start
        write(root/'status.json', status)

    def signal_stop(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)

    def stage(name, argv, seconds):
        if stop.is_set() or time.monotonic() >= deadline:
            return False
        until = min(deadline, time.monotonic() + seconds)
        with (root/(name+'.log')).open('w') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            with guard:
                stages[name] = dict(status='running', pid=process.pid, started_utc=utc(), argv=argv)
                publish()
            while process.poll() is None and time.monotonic() < until and not stop.wait(2):
                with guard:
                    publish()
            timed_out = process.poll() is None
            if timed_out:
                # Capture's context manager cleans up its owned native process.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    # The enclosing systemd cgroup additionally owns all native
                    # children, including the simulator's separate process group.
                    import psutil
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in children:
                        try: child.terminate()
                        except psutil.NoSuchProcess: pass
                    process.kill()
                    process.wait(timeout=10)
                    _, alive = psutil.wait_procs(children, timeout=10)
                    for child in alive:
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
            with guard:
                stages[name].update(status='deadline' if timed_out else 'complete' if process.returncode == 0 else 'failed',
                                    exit_code=process.returncode, ended_utc=utc())
                publish()
            return process.returncode == 0 and not timed_out

    def process_survey(scene_id, split, survey):
        receipts = survey/'receipts.json'
        if not receipts.exists() or not read(receipts).get('receipts'):
            return
        field = root/f'env_{scene_id}-field.npz'
        if stage(f'fusion-{scene_id}', [sys.executable, '-u', '-m', 'research.map_navigation.ppo_tasks',
                'fuse', '--receipts', str(receipts), '--output', str(field)], 7200):
            stage(f'candidates-{scene_id}', [sys.executable, '-u', '-m', 'research.map_navigation.ppo_tasks',
                  'generate', '--field', str(field), '--scene-id', f'env_{scene_id}', '--split', split,
                  '--per-cell', '20' if split == 'train' else '5', '--max-candidates', '4000',
                  '--output', str(root/f'env_{scene_id}-candidates')], 2700)

    try:
        with zipfile.ZipFile(args.annotations) as archive:
            for sid, split in [(5, 'train'), (2, 'train'), (9, 'validation')]:
                plan = survey_plan(archive, sid, split)
                plan['annotations_sha256'] = digest(args.annotations)
                write(root/f'env_{sid}-plan.json', plan)
            # Keep the original scene-size failure as evidence; scene selection
            # is based on available geographic extent, never learner outcomes.
            old_validation = survey_plan(archive, 13, 'validation', positions=1)
            write(root/'original-validation-extent.json', old_validation)
        with guard:
            publish()
        futures = []
        with ThreadPoolExecutor(max_workers=1) as cpu:
            for sid, split in [(5, 'train'), (2, 'train'), (9, 'validation')]:
                survey = root/f'env_{sid}-survey'
                stage(f'survey-{sid}', [sys.executable, '-u', '-m', 'research.map_navigation.ppo_task_capture',
                      'survey', '--scene', str(workspace/f'qualification/aerialvln-preflight-executable/env_{sid}.json'),
                      '--plan', str(root/f'env_{sid}-plan.json'), '--workspace', str(workspace),
                      '--output', str(survey), '--hours', '1.65'], 6000)
                futures.append(cpu.submit(process_survey, sid, split, survey))
                if stop.is_set() or time.monotonic() >= deadline:
                    break
            for future in futures:
                future.result()
        status['status'] = 'preparation_finished_review_required'
    except BaseException as error:
        status.update(status='preparation_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop.set()
        status['ended_utc'] = utc()
        status['surveys'] = {}
        for path in root.glob('env_*-survey/receipts.json'):
            saved = read(path)
            status['surveys'][path.parent.name] = dict(captures=len(saved['receipts']), complete=saved['complete'])
        with guard:
            publish()
        write(root/'morning-report.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace', 'annotations', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--hours', type=float, default=8)
    run(parser.parse_args())
