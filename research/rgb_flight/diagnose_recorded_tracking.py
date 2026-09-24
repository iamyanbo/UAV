"""Offline privileged diagnosis of a completed RGB tracking recording.

Post-hoc similarities below are evaluation results, never runtime calibration
or training inputs. Excludes unbracketed labels and reports each gauge apart.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.episode / 'result.json').exists():
        raise ValueError('Diagnose only a completed episode')
    tracking = args.episode / 'learned-controller/runtime/reconstruction/tracking.jsonl'
    labels = args.episode / 'training_labels/frames.jsonl'
    observations = args.episode / 'observations/frames.jsonl'
    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    truth = sorted({r['state_sim_ns']: r for r in lines(labels)}.values(), key=lambda r: r['state_sim_ns'])
    ns = np.array([r['state_sim_ns'] for r in truth], dtype=np.int64)
    calibration = lines(observations)[0]['calibration']
    extrinsic = np.asarray(calibration['camera_to_body_rotation']).reshape(3, 3)
    lever = np.asarray(calibration['camera_origin_body_m'])
    groups = defaultdict(list)
    rejected = defaultdict(int)
    for row in lines(tracking):
        if not row['initialized']:
            rejected['uninitialized'] += 1
            continue
        now = row['sim_ns']
        right = int(np.searchsorted(ns, now, side='left'))
        if right < len(ns) and ns[right] == now:
            a = b = truth[right]
            fraction = 0.
        elif right == 0 or right == len(ns) or ns[right] - ns[right - 1] > 250_000_000:
            rejected['unbracketed_or_label_gap'] += 1
            continue
        else:
            a, b = truth[right - 1], truth[right]
            fraction = float((now - ns[right - 1]) / (ns[right] - ns[right - 1]))
        body = Slerp([0., 1.], Rotation.from_quat([a['true_quaternion_xyzw'], b['true_quaternion_xyzw']]))([fraction]).as_matrix()[0]
        position = (1 - fraction) * np.asarray(a['true_position_ned_m']) + fraction * np.asarray(b['true_position_ned_m'])
        pose = np.asarray(row['estimated_c2w_arbitrary_scale'])
        groups[row['gauge_changes']].append((pose[:3, 3], pose[:3, :3], position + body @ lever, body @ extrinsic))
    results = []
    for gauge, values in groups.items():
        estimated, rotation, actual, actual_rotation = map(np.asarray, zip(*values))
        relative_error = Rotation.from_matrix(
            np.swapaxes(actual_rotation[0].T @ actual_rotation, -1, -2) @ (rotation[0].T @ rotation)).magnitude()
        item = dict(gauge=gauge, paired_frames=len(values),
                    relative_rotation_rmse_degrees=float(np.rad2deg(np.sqrt(np.mean(relative_error ** 2)))))
        x, y = estimated - estimated.mean(0), actual - actual.mean(0)
        energy = float(np.mean(np.sum(x * x, axis=1)))
        if len(values) >= 3 and energy > 1e-12:
            u, singular, vt = np.linalg.svd(y.T @ x / len(x))
            sign = np.ones(3)
            sign[-1] = np.linalg.det(u @ vt)
            transform = u @ np.diag(sign) @ vt
            scale = float((singular * sign).sum() / energy)
            error = y - scale * x @ transform.T
            radius = float(np.sqrt(np.mean(np.sum(y * y, axis=1))))
            rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
            item.update(posthoc_scale_for_diagnosis_only=scale,
                        posthoc_translation_rmse_m=rmse, true_motion_radius_m=radius,
                        residual_to_motion_radius=rmse / radius if radius > 0 else None)
        results.append(item)
    result = dict(status='completed', accepted=False,
        purpose='Offline privileged diagnosis only; no fitted transform may enter runtime',
        episode=str(args.episode), gauges=results, rejected=dict(rejected),
        source_hashes={str(p.relative_to(args.episode)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (tracking, labels, observations)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
