"""Qualify near-goal learner resets and seal explicit native task splits.

No expert route or control demonstration is collected. Original capture manifests
remain untouched. Starts/poses are evaluator and reset labels, never actor inputs.
"""
from pathlib import Path
import math
import uuid
from PIL import Image
from .common import read, write, digest, FlightLock
from .mission_contracts import city_config, task_band, identity
from .mission_environment import CityEnvironment
from .mission_resources import Resources, RunWindow


def run(args):
    root, cfg = Path(args.root).resolve(), city_config(args.config)
    original = read(root/'tasks.json')
    if original['schema'] != 'photo-goal-native-tasks/v1':
        raise ValueError('Actual native endpoint captures required')
    scene = read(root/'scene.json')
    output = root/'city-tasks.json'
    if output.exists():
        raise ValueError('Task split is immutable; retain or explicitly archive the prior manifest')
    resources, window = Resources(root, cfg, args.device), RunWindow(args.hours)
    resources.check()
    groups = [[] for _ in cfg['distance_bands_m']]
    for task in original['tasks']:
        groups[task_band(task['distance_m'], cfg)].append(task)
    if any(len(group) < 4 for group in groups):
        raise ValueError('Need four genuine endpoint pairs in every distance band')
    tasks = []
    with FlightLock(root, 'city-task-capture'):
        with CityEnvironment(scene, root/('city-reset-capture-'+uuid.uuid4().hex[:12]), cfg) as env:
            env.calibrate()
            for group in groups:
                for index, old in enumerate(sorted(group, key=lambda t: t['id'])):
                    if not window.admits(120):
                        raise RuntimeError('Capture window exhausted; raw evidence retained, split not published')
                    resources.check()
                    split = 'train' if index < len(group)-2 else 'development' if index == len(group)-2 else 'sealed'
                    task = dict(old, split=split, endpoint_qualified=True, support_starts=[])
                    for field in ('start_image', 'goal_image'):
                        task[field+'_sha256'] = digest(task[field])
                    # Requalify endpoints with this actual camera/reset implementation.
                    task['city_reset_evidence'] = {}
                    for field in ('start', 'goal'):
                        task['city_reset_evidence'][field] = env.reset_pose(task[field], task[field+'_yaw_deg'])
                    if split == 'train':
                        for angle, radius in ((0, 3.), (120, 6.), (240, 12.)):
                            radians = math.radians(angle)
                            position = [task['goal'][0]+radius*math.cos(radians),
                                        task['goal'][1]+radius*math.sin(radians), task['goal'][2]]
                            lo, hi = task['bounds']
                            if any(not a <= x <= b for x, a, b in zip(position, lo, hi)):
                                continue
                            try:
                                reports = env.reset_pose(position, task['goal_yaw_deg'])
                                rgb, stamp, _ = env.image()
                                image = root/'tasks'/task['id']/f'support-{angle}.png'
                                Image.fromarray(rgb).save(image)
                                task['support_starts'].append(dict(position=position, yaw_deg=task['goal_yaw_deg'],
                                    qualified=True, reset_evidence=reports, image=str(image.resolve()),
                                    image_sha256=digest(image), capture_sim_ns=stamp))
                            except RuntimeError as error:
                                task.setdefault('rejected_support_starts', []).append(dict(position=position, error=str(error)))
                        if not task['support_starts']:
                            raise RuntimeError('No physically qualified support start for '+task['id'])
                    tasks.append(task)
    write(output, dict(schema='photo-goal-city-tasks/v1', tasks=tasks,
                       original_capture_sha256=digest(root/'tasks.json'), config_sha256=identity(cfg),
                       evidence_scope='endpoint and learner reset qualification; route connectivity unverified'))
