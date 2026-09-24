"""Index real random-goal flights for lazy, checksum-verified visual training.

The manifest contains image references and separate privileged labels, never
copied or repeated panorama pixels. Internal validation holds out complete
training goal regions; the sealed campaign validation/test splits remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from episode_store import verified_rgb_storage
from goal_io import load_goal


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def spaced(rows, limit):
    if len(rows) <= limit:
        return rows
    return [rows[index] for index in np.unique(np.linspace(0, len(rows)-1, limit).round().astype(int))]


def command_before(frame):
    history = frame.get('command_history', ())
    return list(history[-1]['values']) if history else [0., 0., 0., 0.]


def make_manifest(collection_root, minimum_expert_episodes=250):
    root = Path(collection_root).resolve()
    episodes, windows = [], []
    seen_episode_ids = set()
    duplicate_successful_attempts = 0
    for result_path in sorted(root.rglob('episode/result.json')):
        episode = result_path.parent
        result = json.loads(result_path.read_text())
        if (result.get('status') != 'expert_flight_finished' or not result.get('success')
                or result.get('split') != 'train'):
            continue
        observation = episode/'observations'
        storage = verified_rgb_storage(observation)
        if not storage['complete'] or storage['frames'] < 40:
            continue
        goal = load_goal(episode/'goal', result['episode_id'])
        label_path = episode/'training_labels/frames.jsonl'
        evaluator = json.loads((episode/'evaluator_labels/episode.json').read_text())
        if goal.content_sha256 != result['runtime_goal_sha256']:
            raise ValueError('Goal observation changed after expert flight')
        if result['episode_id'] in seen_episode_ids:
            # Smoke/pilot reruns remain in the raw archive but must not count
            # twice toward the number of distinct random start-goal tasks.
            duplicate_successful_attempts += 1
            continue
        frames = {row['frame_id']: row for row in map(json.loads, (observation/'frames.jsonl').read_text().splitlines())}
        labels = {row['frame_id']: row for row in map(json.loads, label_path.read_text().splitlines())}
        common = sorted(frames.keys() & labels.keys())
        if not common or len(common) < 40:
            raise ValueError('Incomplete RGB/privileged-label pairing')
        seen_episode_ids.add(result['episode_id'])
        region = evaluator['goal_region_id']
        internal_split = 'validation' if int(hashlib.sha256(region.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
        relative = str(episode.relative_to(root))
        episodes.append(dict(episode_id=result['episode_id'], split=internal_split,
                             campaign_split='train', goal_region_id=region,
                             episode_path=relative, goal_sha256=goal.content_sha256,
                             observation_stream_sha256=storage['stream_sha256'],
                             training_labels_sha256=digest(label_path)))
        end_seconds = labels[common[-1]]['sim_seconds']
        goal_position = np.asarray(evaluator['goal_ned_m'])
        positive, negative = [], []
        for frame_id in common:
            delta = np.asarray(labels[frame_id]['true_position_ned_m'])-goal_position
            if np.linalg.norm(delta[:2]) <= 3 and abs(delta[2]) <= 2:
                positive.append(frame_id)
            elif np.linalg.norm(delta) >= 12:
                negative.append(frame_id)
        for frame_id in spaced(positive, 12)+spaced(negative, 24):
            near = frame_id in positive
            remaining = max(0., end_seconds-labels[frame_id]['sim_seconds'])
            windows.append(dict(module='goal', episode_id=result['episode_id'], episode_path=relative,
                                frame_id=frame_id, training_labels=dict(near_goal=near, match_valid=True,
                                time_to_goal_seconds=remaining, terminal_return=1.-min(1., remaining/180.))))
        adjacent = [(previous, current) for previous, current in zip(common, common[1:])
                    if current == previous+1 and 0 < (frames[current]['sim_ns']-frames[previous]['sim_ns'])/1e9 < .25]
        for previous, current in spaced(adjacent, 80):
            before, after = labels[previous], labels[current]
            dt = (frames[current]['sim_ns']-frames[previous]['sim_ns'])/1e9
            rotation = Rotation.from_quat(before['true_quaternion_xyzw'])
            body_translation = rotation.inv().apply(np.asarray(after['true_position_ned_m'])-
                                                    np.asarray(before['true_position_ned_m']))
            body_rotation = (rotation.inv()*Rotation.from_quat(after['true_quaternion_xyzw'])).as_rotvec()
            command = np.asarray(command_before(frames[current]), dtype=float)
            windows.append(dict(module='odometry', episode_id=result['episode_id'], episode_path=relative,
                                frame_id=current, previous_frame_id=previous, previous_command=command.tolist(),
                                delta_seconds=dt,
                                training_labels=dict(body_motion=np.r_[body_translation, body_rotation].tolist())))
    if len(episodes) < minimum_expert_episodes:
        raise ValueError(f'Only {len(episodes)} valid training expert episodes; need {minimum_expert_episodes}')
    if not {'train', 'validation'} <= {row['split'] for row in episodes}:
        raise ValueError('Internal held-out goal-region split is empty')
    if not all(any(row['module'] == module and next(e for e in episodes if e['episode_id'] == row['episode_id'])['split'] == split
                   for row in windows) for module in ('goal', 'odometry') for split in ('train', 'validation')):
        raise ValueError('Missing visual training or validation windows')
    return dict(schema='visual-training-lazy/v2', valid_expert_episodes=len(episodes),
                duplicate_successful_attempts_skipped=duplicate_successful_attempts,
                collection_root=str(root), episodes=episodes, windows=windows,
                internal_validation='SHA256(goal_region_id) modulo 10 == 0; campaign split remains train')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--minimum-expert-episodes', type=int, default=250)
    args = parser.parse_args()
    manifest = make_manifest(args.collection_root, args.minimum_expert_episodes)
    output = args.collection_root/'visual-training.json'
    if output.exists():
        raise ValueError('Visual manifest already exists; preserve it and build an explicit new trajectory round')
    temporary = output.with_suffix('.pending')
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(output)
    print(json.dumps(dict(episodes=len(manifest['episodes']), windows=len(manifest['windows']), output=str(output))))


if __name__ == '__main__':
    main()
