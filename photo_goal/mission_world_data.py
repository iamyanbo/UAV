"""Tensorize recorded physical windows for independent supervised dynamics."""
import torch
from .ppo_scheduler import batch_for
from .temporal import pool
from .mission_data import rgb_tensor


def _assemble(actor, current, following, records, labels, device):
    with torch.no_grad():
        features = pool(actor.project(current['history'].flatten(0, 1))).reshape(len(records), 4, 64, 256)
        target = pool(actor.project(following['history'][:, -1]))
        goals = actor.project(current['goal'])
        goal_context = goals.mean(1)
    width = max(len(r['command_intervals']) for r in records)
    segments = features.new_zeros(len(records), width, 5)
    valid = torch.zeros(len(records), width, dtype=torch.bool, device=device)
    for i, record in enumerate(records):
        intervals = record['command_intervals']
        segments[i, :len(intervals)] = segments.new_tensor(intervals)
        valid[i, :len(intervals)] = True
        if abs(sum(s[4] for s in intervals)-record['dt']) > 1e-5:
            raise ValueError('Command intervals do not cover the physical transition')
    result = dict(features=features.detach(), times=current['times'], commands=current['commands'],
                  valid=current['valid'], next_features=target.detach(), segments=segments,
                  segment_valid=valid, goal_context=goal_context.detach(),
                  mission_context=current['mission_context'].detach())
    for key in ('motion', 'collision', 'reward', 'terminated', 'goal', 'visibility', 'stop_success'):
        default = [0.]*4 if key == 'motion' else 0.
        result[key] = torch.tensor([row.get(key, default) for row in labels], device=device, dtype=torch.float32)
        result[key+'_valid'] = torch.tensor([row.get(key+'_valid', False) for row in labels], device=device, dtype=torch.bool)
    return result


def from_rows(actor, rows, banks, device):
    current = batch_for(rows, banks, device)
    following = batch_for([row['next_context'] for row in rows], banks, device)
    return _assemble(actor, current, following, rows, [row['world_labels'] for row in rows], device)


def from_shard(actor, records, labels, root, device):
    providers={}
    def encoded(next_state=False):
        contexts = [row['next_context'] if next_state else row['context'] for row in records]
        history, goals = [], []
        for row, context in zip(records, contexts):
            images = [rgb_tensor(root, row['images'][i],providers) for i in context['history_images']]
            images = [torch.zeros_like(images[-1])]*(4-len(images))+images
            history.append(torch.stack(images))
            goals.append(rgb_tensor(root, row['images'][context['goal_image_index']],providers))
        history = torch.stack(history).to(device)
        goals = torch.stack(goals).to(device)
        with torch.no_grad():
            raw = actor.encode_backbone(history.flatten(0, 1)).reshape(len(records), 4, 960, 15, 20)
            goal = actor.encode_backbone(goals)
        n = [len(c['history_images']) for c in contexts]
        return dict(history=raw, goal=goal,
                    times=torch.tensor([[0.]*(4-k)+[t-c['stamps'][-1] for t in c['stamps']]
                                        for c, k in zip(contexts, n)], device=device),
                    commands=torch.tensor([[[0.]*4]*(4-k)+c['preceding'] for c, k in zip(contexts, n)], device=device),
                    valid=torch.tensor([[False]*(4-k)+[True]*k for k in n], device=device),
                    mission_context=torch.tensor([c['mission_context'] for c in contexts], device=device))
    try:result = _assemble(actor, encoded(), encoded(True), records, labels, device)
    finally:
        for provider in providers.values():provider.close()
    # Optional independently computed official V-JEPA targets. Missing causal
    # clips are masked; corrupt declared targets are errors, never silent masks.
    targets, mask = [], []
    from .common import contained, digest
    for record in records:
        spec = record.get('teacher_target')
        if spec is None or spec.get('valid') is False:
            targets.append(torch.zeros(64, 1024))
            mask.append(False)
            continue
        import os
        from pathlib import Path
        target_root=Path(os.environ['UAV_PROJECT_ROOT']) if Path(spec['path']).is_absolute() else root
        path = contained(target_root, spec['path'])
        if digest(path) != spec['sha256']:
            raise ValueError('Declared video target changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['calibration_id'] != record['calibration_id'] or
                abs(saved['end_ns']-round(record['next_context']['stamps'][-1]*1e9)) > 1000 or
                saved['cache_key'] != spec['cache_key'] or saved['base_sha256'] != spec['base_sha256']):
            raise ValueError('Video target provenance does not match this physical transition')
        value = saved['tokens'].reshape(-1, 1024)
        if saved['schema'] != 'photo-goal-teacher-target/v1' or value.shape != (64, 1024) or not torch.isfinite(value).all():
            raise ValueError('Invalid official video target')
        targets.append(value)
        mask.append(True)
    result['teacher'] = torch.stack(targets).to(device)
    result['teacher_valid'] = torch.tensor(mask, dtype=torch.bool, device=device)
    return result
