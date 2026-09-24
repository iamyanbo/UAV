"""Training-only scale supervision from the same *past* observed motion window.

Engineering calibration episodes are permitted before the navigation gate.
Their privileged commands are never exported as policy imitation targets.
Runtime tensors and simulator-derived labels are stored in separate files.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from episode_store import verified_rgb_storage
from perception_metrics import align_similarity, quaternion_rotation
from scale_features import ScaleInputs, INTERVAL_NS, STATISTICS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def build_episode(episode, output):
    episode, output = Path(episode), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    storage = verified_rgb_storage(episode/'observations')
    if not storage['complete']:
        raise ValueError('Scale preparation requires a finalized physical recording')
    result = json.loads((episode/'result.json').read_text())
    reconstruction = episode/'runtime_perception/reconstruction'
    if not json.loads((reconstruction/'result.json').read_text())['live_async_execution']:
        raise ValueError('Offline completion times cannot be used as live scale inputs')
    frames = read_lines(episode/'observations/frames.jsonl')
    by_id = {row['frame_id']: row for row in frames}
    tracking = read_lines(reconstruction/'tracking.jsonl')
    visual_root = episode/'runtime_perception/vjepa'
    features = read_lines(visual_root/'outputs.jsonl')
    events = [(x['processed_monotonic_seconds'], 'pose', x) for x in tracking]
    events += [(x['completed_monotonic'], 'visual', x) for x in features]
    events.sort(key=lambda x: x[0])
    states = json.loads((episode/'engineering_only/states.json').read_text())
    origin = states[0]['sim_ns']
    true_times = np.array([(s['sim_ns']-origin)/1e9 for s in states])
    if not bool((np.diff(true_times) > 0).all()):
        raise ValueError('Nonmonotonic simulator label time')
    offset = np.asarray(frames[0]['calibration']['camera_origin_body_m'])
    true_positions = np.asarray([np.asarray(s['position'])+quaternion_rotation(s['quaternion'])@offset for s in states])
    causal = ScaleInputs(frames[0]['episode_id'])
    event_index, next_ns = 0, frames[0]['sim_ns']
    runtime, labels, feature_sources = [], [], []
    for frame in frames:
        while event_index < len(events) and events[event_index][0] <= frame['received_monotonic']:
            _, kind, row = events[event_index]
            if kind == 'pose':
                causal.observe_pose(row, by_id[row['frame_id']])
            else:
                path = visual_root/row['feature_path']
                if path.resolve().parent != visual_root.resolve():
                    raise ValueError('Feature path escapes observation-derived directory')
                record = torch.load(path, map_location='cpu', weights_only=True)
                source = [by_id[index] for index in record['source_frame_ids']]
                if ([s['sim_ns'] for s in source] != record['source_sim_ns']
                        or max(record['source_sim_ns']) > row['latest_observation_ns']):
                    raise ValueError('Invalid causal visual sources')
                tokens = record['tokens'].float().reshape(64, 1024)
                causal.observe_visual(row, tokens.mean(0).numpy())
                feature_sources.append(dict(path=str(path.resolve()), sha256=sha(path)))
            event_index += 1
        if frame['sim_ns'] < next_ns:
            continue
        next_ns = frame['sim_ns'] + INTERVAL_NS
        sample = causal.sample(frame)
        if sample is None:
            continue
        times = (sample.pop('source_pose_ns')-origin)/1e9
        estimate = sample.pop('source_pose_positions')
        label = dict(log_scale=0., valid=False, available=False, fit_relative_rmse=0., motion_m=0.)
        # Labels interpolate simulator motion only inside this observed past
        # interval. Never use an episode-wide/future alignment as a target.
        label_end = int(np.searchsorted(true_times, (sample['sim_ns']-origin)/1e9, side='right'))
        past_times, past_truth = true_times[:label_end], true_positions[:label_end]
        if len(times) >= 8 and len(past_times) >= 2 and times[0] >= past_times[0] and times[-1] <= past_times[-1] and times[-1]-times[0] >= 3.:
            truth = np.stack([np.interp(times, past_times, past_truth[:, i]) for i in range(3)], -1)
            motion = float(np.linalg.norm(truth-truth.mean(0), axis=1).max())
            label.update(available=True, motion_m=motion)
            try:
                aligned, scale = align_similarity(estimate, truth)
                relative_error = float(np.sqrt(np.square(aligned-truth).sum(-1).mean()) / max(motion, 1e-6))
                if scale > 0 and np.isfinite(scale) and np.isfinite(relative_error):
                    label.update(log_scale=float(np.log(scale)), fit_relative_rmse=relative_error,
                                 valid=bool(motion >= 1. and relative_error <= .25))
            except ValueError:
                pass  # No translational excitation: a recorded unusable label.
        runtime.append(sample)
        labels.append(label)
    if not runtime:
        raise ValueError('No causally available scale inputs in this episode')
    tensors = {key: torch.from_numpy(np.asarray([row[key] for row in runtime])) for key in runtime[0]}
    targets = {key: torch.from_numpy(np.asarray([row[key] for row in labels])) for key in labels[0]}
    torch.save(tensors, output/'runtime.pt')
    torch.save(targets, output/'training_labels.pt')
    sources = [episode/'observations/storage.json', episode/'observations/frames.jsonl', episode/'result.json',
               episode/'engineering_only/states.json', reconstruction/'tracking.jsonl', visual_root/'outputs.jsonl']
    manifest = dict(schema=1, episode_id=frames[0]['episode_id'], route_id=result['route_id'],
        scope='engineering scale calibration only; not a navigation or imitation bundle',
        color_calibration_sha256=storage['color_calibration_sha256'], statistics=list(STATISTICS),
        runtime_sha256=sha(output/'runtime.pt'), labels_sha256=sha(output/'training_labels.pt'),
        sources=[dict(path=str(p.resolve()), sha256=sha(p)) for p in sources], feature_sources=feature_sources,
        samples=len(runtime), available_labels=int(targets['available'].sum()), usable_labels=int(targets['valid'].sum()),
        simulated_seconds=(runtime[-1]['sim_ns']-runtime[0]['sim_ns'])/1e9,
        privileged_fields_in_runtime=False, label_alignment='each past 5s observed window only',
        acceptance=dict(minimum_camera_excitation_m=1., maximum_relative_fit_rmse=.25))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def assemble(registry, output):
    """Registry must assign routes before extracting any train/held-out windows."""
    registry, output = Path(registry), Path(output)
    declaration = json.loads(registry.read_text())
    if declaration['scope'] != 'engineering_scale_calibration' or not declaration['split_declared_before_window_extraction']:
        raise ValueError('Explicit calibration split declaration required')
    groups, rows, split_counts = {}, [], dict(train=0, validation=0, calibration=0)
    for item in declaration['episodes']:
        split = item['split']
        if split not in split_counts or item['route_id'] in groups:
            raise ValueError('Unknown split or repeated route in scale dataset')
        groups[item['route_id']] = split
        folder = (registry.parent/item['prepared']).resolve()
        receipt = json.loads((folder/'manifest.json').read_text())
        if receipt['route_id'] != item['route_id']:
            raise ValueError('Prepared episode route differs from declared split')
        for field, name in [('runtime_sha256','runtime.pt'), ('labels_sha256','training_labels.pt')]:
            if sha(folder/name) != receipt[field]:
                raise ValueError('Modified scale episode artifact')
        rows.append(dict(item, folder=folder, receipt=receipt))
        split_counts[split] += 1
    if split_counts['train'] < 5 or min(split_counts['validation'], split_counts['calibration']) < 2:
        raise ValueError('Scale fitting needs at least 5 training, 2 validation and 2 separate calibration routes')
    output.mkdir(parents=True, exist_ok=False)
    total, mean, scatter = 0, torch.zeros(1024, dtype=torch.float64), torch.zeros(1024, 1024, dtype=torch.float64)
    train_ids = []
    for row in rows:
        if row['split'] != 'train':
            continue
        train_ids.append(row['receipt']['episode_id'])
        for source in row['receipt']['feature_sources']:
            if sha(source['path']) != source['sha256']:
                raise ValueError('Modified source of training-only scale projection')
            values = torch.load(source['path'], weights_only=True, map_location='cpu')['tokens'].double().reshape(64, 1024)
            n, center = len(values), values.mean(0)
            delta = center-mean
            scatter += (values-center).T@(values-center)+torch.outer(delta,delta)*(total*n/(total+n))
            mean += delta*(n/(total+n))
            total += n
    if total < 1024:
        raise ValueError('Insufficient training tokens for projection')
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter/(total-1))
    projection = dict(fit_split='train', episode_ids=train_ids, source_manifest_sha256=sha(registry),
                      mean=mean.float(), components=eigenvectors[:, -256:].flip(1).float(), samples=total,
                      explained_variance_fraction=float(eigenvalues[-256:].sum()/eigenvalues.sum()))
    torch.save(projection, output/'projection.pt')
    episodes, stats, visuals = [], [], []
    for index, row in enumerate(rows):
        runtime = torch.load(row['folder']/'runtime.pt', weights_only=True, map_location='cpu')
        runtime['visual'] = (runtime.pop('visual_raw').float()-projection['mean'])@projection['components']
        if row['split'] == 'train':
            stats.append(runtime['statistics'].float())
            visuals.append(runtime['visual'])
        name = f'episode-{index:03d}'
        (output/name).mkdir()
        torch.save(runtime, output/name/'runtime.pt')
        (output/name/'training_labels.pt').write_bytes((row['folder']/'training_labels.pt').read_bytes())
        episodes.append(dict(episode_id=row['receipt']['episode_id'], route_id=row['route_id'], split=row['split'],
                             path=name, runtime_sha256=sha(output/name/'runtime.pt'),
                             labels_sha256=sha(output/name/'training_labels.pt'),
                             preparation_sha256=sha(row['folder']/'manifest.json')))
    statistics, visual = torch.cat(stats), torch.cat(visuals)
    torch.save(dict(fit_split='train', episode_ids=train_ids, statistics_mean=statistics.mean(0),
                    statistics_std=statistics.std(0).clamp_min(.01), visual_mean=visual.mean(0),
                    visual_std=visual.std(0).clamp_min(.01)), output/'normalization.pt')
    manifest = dict(schema=1, scope='engineering_scale_calibration', registry_sha256=sha(registry),
                    projection_sha256=sha(output/'projection.pt'), normalization_sha256=sha(output/'normalization.pt'),
                    episodes=episodes, split_counts=split_counts, window_steps=25, maximum_gap_seconds=.4,
                    acceptance=declaration['acceptance'], main_navigation_training_authorized=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def prepare_survey(source, output):
    declaration = json.loads(Path(source).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registry = declaration['registry']
    for row in registry['episodes']:
        if Path('/output/CHECKPOINT_REQUEST').exists():
            (output/'result.json').write_text(json.dumps(dict(status='checkpointed_preparation',
                next_action='Resume preparation from finalized per-episode artifacts; no model was trained')))
            return dict(status='checkpointed_preparation')
        build_episode('/recordings/'+row['route_id'], output/row['prepared'])
    path = output/'registry.json'
    path.write_text(json.dumps(registry,indent=2))
    result = assemble(path, output/'dataset')
    (output/'result.json').write_text(json.dumps(dict(status='scale_bundle_prepared',dataset=str(output/'dataset'),
        manifest_sha256=sha(output/'dataset/manifest.json'),split_counts=result['split_counts']),indent=2))
    return dict(status='scale_bundle_prepared',dataset=str(output/'dataset'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode')
    source.add_argument('--registry')
    source.add_argument('--survey-input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = (build_episode(args.episode,args.output) if args.episode else
              prepare_survey(args.survey_input,args.output) if args.survey_input else assemble(args.registry,args.output))
    print(json.dumps({key:value for key,value in result.items() if key not in ('sources','feature_sources','episodes')},indent=2))
