"""Deterministic random visual-goal episode manifests.

Coordinates and reference paths are written only to evaluator_labels.  The
runtime manifest has identities, visual-goal record locations and curriculum
metadata, but cannot reveal where the destination is.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from obstacle_field import PrivilegedObstacleField


COUNTS = {'train': 10_000, 'validation': 150, 'test': 200}
PHASES = {
    1: dict(path_m=(20., 80.), agl_m=(5., 10.)),
    2: dict(path_m=(50., 180.), agl_m=(5., 15.)),
    3: dict(path_m=(100., 300.), agl_m=(5., 20.)),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_length(path):
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def goal_region(point, size_m=10.):
    return ':'.join(str(int(math.floor(value/size_m))) for value in point)


def region_owner(region, counts):
    """Reserve regions before sampling so the large train split cannot claim all."""
    total = sum(counts.values())
    value = int.from_bytes(hashlib.sha256(region.encode()).digest()[:8], 'big') % total
    cursor = 0
    for split, count in counts.items():
        cursor += count
        if value < cursor:
            return split
    raise AssertionError('Unreachable region assignment')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def generate(field_path, output, seed=20260921, counts=None):
    field_path = Path(field_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'evaluator_labels').mkdir()
    field = PrivilegedObstacleField.load(field_path)
    rng = np.random.default_rng(seed)
    counts = counts or COUNTS
    owners, pairs, public, privileged = {}, set(), [], []
    total = sum(counts.values())
    split_sequence = [split for split, count in counts.items() for _ in range(count)]
    phase_counters = {split: 0 for split in counts}
    attempts = 0
    rejected = dict(bounds=0, ground=0, owner=0, occupied=0, path=0, length=0)
    while len(public) < total:
        attempts += 1
        if attempts <= 5 or attempts % 50 == 0:
            print(json.dumps(dict(attempts=attempts, accepted=len(public), rejected=rejected)), flush=True)
        if attempts > total * 200:
            raise RuntimeError('Unable to fill deterministic episode manifest from surveyed free space')
        split = split_sequence[len(public)]
        phase = 1 + phase_counters[split] % 3
        limits = PHASES[phase]
        start_xy = rng.uniform(field.bounds[0, :2]+3, field.bounds[1, :2]-3)
        angle = rng.uniform(-math.pi, math.pi)
        nominal = rng.uniform(*limits['path_m'])
        goal_xy = start_xy + nominal*np.array([math.cos(angle), math.sin(angle)])
        if np.any(goal_xy < field.bounds[0, :2]+3) or np.any(goal_xy > field.bounds[1, :2]-3):
            rejected['bounds'] += 1
            continue
        try:
            start = np.r_[start_xy, field.ground_z(start_xy)-rng.uniform(*limits['agl_m'])]
            goal = np.r_[goal_xy, field.ground_z(goal_xy)-rng.uniform(*limits['agl_m'])]
        except ValueError:
            rejected['ground'] += 1
            continue
        region = goal_region(goal)
        if region_owner(region, counts) != split or (region in owners and owners[region] != split):
            rejected['owner'] += 1
            continue
        pair = (tuple(np.round(start, 1)), tuple(np.round(goal, 1)))
        if pair in pairs or field.contains_vehicle(start) or field.contains_vehicle(goal):
            rejected['occupied'] += 1
            continue
        try:
            path = np.stack((start, goal)) if field.segment_free(start, goal) else field.reference_path(start, goal)
        except ValueError:
            rejected['path'] += 1
            continue
        distance = path_length(path)
        if not limits['path_m'][0] <= distance <= limits['path_m'][1]:
            rejected['length'] += 1
            continue
        index = phase_counters[split]
        episode_id = f'{split}-{index:05d}'
        owners[region] = split; pairs.add(pair); phase_counters[split] += 1
        public.append(dict(episode_id=episode_id, split=split, phase=phase,
                           goal_record=f'goals/{episode_id}', goal_views=4,
                           allowed_runtime_inputs=['RGB','goal_RGB','camera_calibration','timestamps','previous_commands'],
                           eligible_speed_stages_mps=[3., 4.5, 6.]))
        if len(public) % 10 == 0:
            print(json.dumps(dict(accepted=len(public), attempts=attempts, rejected=rejected)), flush=True)
        privileged.append(dict(episode_id=episode_id, split=split, phase=phase,
                               start_ned_m=start.tolist(), start_yaw_degrees=float(rng.uniform(-180, 180)),
                               goal_ned_m=goal.tolist(), reference_path_ned_m=path.tolist(),
                               reference_length_m=distance, goal_region_id=region,
                               obstacle_field_sha256=digest(field_path)))
    by_split = {split: [row for row in public if row['split']==split] for split in counts}
    labels = {split: [row for row in privileged if row['split']==split] for split in counts}
    for split in counts:
        save_json(output/f'{split}.json', dict(schema='visual-goal-runtime-manifest/v1', seed=seed,
                                              scene='env_airsim_16', episodes=by_split[split]))
        save_json(output/'evaluator_labels'/f'{split}.json', dict(schema='privileged-evaluator-labels/v1',
                                              obstacle_field_sha256=digest(field_path), episodes=labels[split]))
    region_sets = {split: {row['goal_region_id'] for row in labels[split]} for split in counts}
    if any(region_sets[a] & region_sets[b] for index, a in enumerate(counts) for b in list(counts)[index+1:]):
        raise RuntimeError('Goal region leakage across splits')
    receipt = dict(status='deterministic_manifests_complete', seed=seed, counts={k:len(v) for k,v in by_split.items()},
                   attempts=attempts, rejected=rejected, complete_pair_disjointness=len(pairs)==total,
                   goal_regions_disjoint=True, obstacle_field_sha256=digest(field_path),
                   manifests={split:digest(output/f'{split}.json') for split in counts},
                   evaluator_labels={split:digest(output/'evaluator_labels'/f'{split}.json') for split in counts},
                   sealed_test_labels=str(output/'evaluator_labels/test.json'))
    save_json(output/'MANIFEST.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obstacle-field', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--diagnostic-counts', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='Small sampler feasibility run; never use its manifests for the campaign')
    args = parser.parse_args()
    output=args.output or Path(__import__('os').environ['RGB_JOB_DIR'])/'manifests'
    counts = dict(zip(COUNTS, args.diagnostic_counts)) if args.diagnostic_counts else None
    if counts and any(value < 1 for value in counts.values()):
        parser.error('Diagnostic counts must all be positive')
    receipt = generate(args.obstacle_field, output, args.seed, counts)
    if counts:
        receipt['diagnostic_only'] = True
        save_json(output/'MANIFEST.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
