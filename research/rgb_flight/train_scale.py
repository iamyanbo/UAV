"""Resumable scale-prior training, route validation, and held-out calibration.

This is a foundation calibration stage, not world/policy/configurator training.
No metric-navigation acceptance follows automatically from fitting this model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from learning_models import ScaleEstimator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScaleBundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.digest = sha(self.root/'manifest.json')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['scope'] != 'engineering_scale_calibration':
            raise ValueError('Expected the dedicated engineering scale bundle')
        self.steps = self.manifest['window_steps']
        self.episodes, self.windows = {}, {s: [] for s in ('train', 'validation', 'calibration')}
        groups = set()
        for row in self.manifest['episodes']:
            if row['route_id'] in groups or row['episode_id'] in self.episodes:
                raise ValueError('Duplicate/cross-split calibration route or episode')
            groups.add(row['route_id'])
            folder = (self.root/row['path']).resolve()
            if not folder.is_relative_to(self.root):
                raise ValueError('Scale bundle path escapes its data directory')
            for name, field in [('runtime.pt','runtime_sha256'), ('training_labels.pt','labels_sha256')]:
                if sha(folder/name) != row[field]:
                    raise ValueError('Scale bundle checksum mismatch')
            runtime = torch.load(folder/'runtime.pt', weights_only=True, map_location='cpu')
            labels = torch.load(folder/'training_labels.pt', weights_only=True, map_location='cpu')
            allowed = {'sim_ns','received_monotonic','visual','statistics','command','latest_observation_ns'}
            if set(runtime) != allowed or not bool((runtime['latest_observation_ns'] <= runtime['sim_ns']).all()):
                raise ValueError('Unexpected/privileged or future runtime scale input')
            self.episodes[row['episode_id']] = runtime, labels
            for end in range(self.steps-1, len(runtime['sim_ns'])):
                start = end+1-self.steps
                delta = runtime['sim_ns'][start+1:end+1]-runtime['sim_ns'][start:end]
                if bool(labels['available'][end]) and bool(((delta>0)&(delta<=int(self.manifest['maximum_gap_seconds']*1e9))).all()):
                    self.windows[row['split']].append((row['episode_id'], start, end))
        if not all(self.windows.values()):
            raise ValueError('Scale train, validation, and calibration windows are all required')
        for split in self.windows:
            valid = sum(bool(self.episodes[e][1]['valid'][end]) for e, _, end in self.windows[split])
            if valid < 25:
                raise ValueError('Insufficient usable observed scale windows in '+split)
        for name in ('projection', 'normalization'):
            if sha(self.root/(name+'.pt')) != self.manifest[name+'_sha256']:
                raise ValueError('Modified training-fit scale '+name)
            value = torch.load(self.root/(name+'.pt'), weights_only=True, map_location='cpu')
            training_ids = {r['episode_id'] for r in self.manifest['episodes'] if r['split']=='train'}
            if value['fit_split']!='train' or set(value['episode_ids']) != training_ids:
                raise ValueError('Held-out data in scale projection or normalization')

    def batch(self, split, indices, device):
        runtime, labels = [], []
        for index in indices:
            episode, start, end = self.windows[split][int(index)]
            x, y = self.episodes[episode]
            runtime.append({key: x[key][start:end+1] for key in ('visual','statistics','command')})
            labels.append({key: y[key][end] for key in ('log_scale','valid','available')})
        return ({key: torch.stack([x[key] for x in runtime]).float().to(device) for key in runtime[0]},
                {key: torch.stack([x[key] for x in labels]).to(device) for key in labels[0]})


def predict(model, batch):
    result = model(batch['visual'], batch['statistics'], batch['command'])
    return {key: value[:, -1] for key, value in result.items() if key != 'hidden'}


@torch.no_grad()
def evaluate(model, bundle, split, device):
    model.eval()
    all_predictions, all_labels, losses = [], [], []
    for start in range(0, len(bundle.windows[split]), 128):
        indices = list(range(start, min(start+128, len(bundle.windows[split]))))
        x, y = bundle.batch(split, indices, device)
        p = predict(model, x)
        losses.append((float(model.loss(p, y['log_scale'], y['valid'], y['available'])), len(indices)))
        all_predictions.append({key: value.cpu() for key, value in p.items()})
        all_labels.append({key: value.cpu() for key, value in y.items()})
    p = {key: torch.cat([x[key] for x in all_predictions]) for key in all_predictions[0]}
    y = {key: torch.cat([x[key] for x in all_labels]) for key in all_labels[0]}
    return sum(loss*n for loss,n in losses)/sum(n for _,n in losses), p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/dataset')
    parser.add_argument('--output', default='/output/scale-training')
    parser.add_argument('--resume')
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.updates < 200:
        raise ValueError('Use a declared substantive calibration budget')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bundle = ScaleBundle(args.dataset)
    model = ScaleEstimator().to(args.device)
    normalization = torch.load(bundle.root/'normalization.pt', weights_only=True, map_location='cpu')
    with torch.no_grad():
        for key in ('visual_mean','visual_std','statistics_mean','statistics_std'):
            getattr(model,key).copy_(normalization[key])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.updates,eta_min=1e-6)
    update, best, best_model, exposed = 0, float('inf'), None, set()
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != bundle.digest or saved['target_updates'] != args.updates:
            raise ValueError('Resume requires the same declared dataset and budget')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        update, best, best_model = saved['update'], saved['best'], saved['best_model']
        rng.bit_generator.state = saved['sample_rng']
        torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        exposed.update(saved['exposed_windows'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    def checkpoint():
        value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            update=update, best=best, best_model=best_model, target_updates=args.updates,
            manifest_sha256=bundle.digest, sample_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
            exposed_windows=sorted(exposed))
        torch.save(value, output/'latest.pending')
        (output/'latest.pending').replace(output/'latest.pt')
    started = time.monotonic()
    while update < args.updates and not Path('/output/CHECKPOINT_REQUEST').exists():
        model.train()
        indices = rng.integers(len(bundle.windows['train']),size=64)
        exposed.update(int(i) for i in indices)
        x, y = bundle.batch('train',indices,args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(predict(model,x),y['log_scale'],y['valid'],y['available'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite scale loss; checkpoint remains available')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        scheduler.step()
        update += 1
        validation = None
        if update % 200 == 0 or update == args.updates:
            validation, _, _ = evaluate(model,bundle,'validation',args.device)
            if validation < best:
                best = validation
                best_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint()
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=update,loss=float(loss),validation_loss=validation,
                window_exposures=update*64,unique_exposed_windows=len(exposed),elapsed_seconds=time.monotonic()-started))+'\n')
    checkpoint()
    result = dict(status='checkpointed',updates=update,metric_navigation_accepted=False)
    if update == args.updates:
        model.load_state_dict(best_model)
        _, p, y = evaluate(model,bundle,'calibration',args.device)
        valid = y['valid']
        residual = (p['mean']-y['log_scale']).abs()
        # A held-out empirical 95% interval multiplier; no formal coverage claim
        # for correlated flight windows or for new route distributions.
        multiplier = max(1.,float(torch.quantile((residual[valid]/p['sigma'][valid]).float(),.95)))
        bounds = bundle.manifest['acceptance']
        eligible = p['usable_logit'].sigmoid() >= bounds['minimum_usable_probability']
        eligible &= multiplier*p['sigma'] <= bounds['maximum_log_scale_radius']
        wrong = int((eligible&~valid).sum())
        selected = int(eligible.sum())
        accepted_valid = eligible&valid
        relative_error = (p['mean'].exp()/y['log_scale'].exp()-1).abs()
        report = dict(calibration_windows=len(valid),usable_labels=int(valid.sum()),accepted_windows=selected,
            invalid_windows_accepted=wrong,acceptance_fraction=selected/len(valid),
            interval_coverage=float((residual[valid]<=multiplier*p['sigma'][valid]).float().mean()),
            median_relative_scale_error=float(relative_error[valid].median()),
            p95_relative_scale_error=float(torch.quantile(relative_error[valid].float(),.95)),
            accepted_p95_relative_error=float(torch.quantile(relative_error[accepted_valid].float(),.95)) if bool(accepted_valid.any()) else None)
        # Validation still needs complete flights with causal metric geometry;
        # a calibration receipt can never authorize navigation by itself.
        artifact = dict(schema=1,model=best_model,projection=torch.load(bundle.root/'projection.pt',weights_only=True),
            manifest_sha256=bundle.digest,interval_multiplier=multiplier,acceptance=bounds,
            calibration=report,metric_navigation_accepted=False,window_steps=bundle.steps,
            maximum_gap_seconds=bundle.manifest['maximum_gap_seconds'])
        torch.save(artifact,output/'scale-prior.pt')
        result.update(status='calibration_budget_completed',calibration=report)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
